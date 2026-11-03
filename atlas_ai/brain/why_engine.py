from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atlas_ai.decision_support import build_decision_trace
from atlas_ai.market_context import build_context_snapshot


WHY_TRACE_STAGES = (
    "signal_generation",
    "signal_ranking",
    "conservative_filter",
    "risk_management",
    "execution",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text or default


def _status(action: str, *, blocked: bool = False) -> str:
    if blocked:
        return "blocked"
    if action in {"paper", "execute", "failed"}:
        return "passed"
    return "blocked" if action in {"hold", "flat", "blocked", "no_trade"} else "passed"


def _entry(stage: str, status: str, reason: str, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "reason": reason,
        "data": dict(data or {}),
    }


def _first_block(entries: list[dict[str, Any]]) -> str:
    for row in entries:
        if row.get("status") == "blocked":
            return str(row.get("stage", "") or "")
    return ""


def build_why_trace(
    status: dict[str, Any] | None,
    runtime_flags: dict[str, Any] | None = None,
    execution_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only ordered explanation of the latest trade pipeline state."""
    status = dict(status or {})
    runtime_flags = dict(runtime_flags or {})
    execution_snapshot = dict(execution_snapshot or {})
    context_snapshot = build_context_snapshot(status)
    decision_trace = build_decision_trace(status, context_snapshot, runtime_flags)

    action = _text(status.get("action"), "unknown").lower()
    message = _text(status.get("message"), "No decision message recorded.")
    chosen = dict(status.get("chosen", {}) or {})
    ranking = dict(decision_trace.get("ranking", {}) or {})
    filters = dict(decision_trace.get("filters", {}) or {})
    risk = dict(decision_trace.get("risk", {}) or {})
    conservative = dict(status.get("conservative_pretrade_filter", {}) or {})
    timing_discipline = dict(status.get("timing_discipline", {}) or {})
    timing_result = dict(timing_discipline.get("structured_result", {}) or timing_discipline)
    gate_decision = dict(
        execution_snapshot.get("gate_decisions_last_cycle", {})
        or execution_snapshot.get("last_gate_decision", {})
        or {}
    )
    execution_truth = dict(
        execution_snapshot.get("execution_truth", {})
        or execution_snapshot.get("last_execution_truth", {})
        or {}
    )

    entries: list[dict[str, Any]] = []
    raw_signals = list(dict(decision_trace.get("signal", {}) or {}).get("raw", []) or [])
    active_signals = list(dict(decision_trace.get("signal", {}) or {}).get("active", []) or [])
    has_signal = bool(raw_signals or active_signals or _text(chosen.get("strategy"), "").lower() not in {"", "none"})
    signal_blocked = not has_signal and action in {"hold", "flat", "blocked", "no_trade", "unknown"}
    entries.append(
        _entry(
            "signal_generation",
            "blocked" if signal_blocked else "passed",
            message if signal_blocked else f"Signal observed: {_text(chosen.get('strategy'), 'candidate available')}.",
            data={"chosen": chosen, "raw_count": len(raw_signals), "active_count": len(active_signals)},
        )
    )

    if signal_blocked:
        entries.append(_entry("signal_ranking", "not_reached", "No generated signal was available to rank."))
    else:
        top = list(ranking.get("top", []) or [])
        score = ranking.get("chosen_score")
        ranking_blocked = (
            action in {"hold", "flat", "blocked", "no_trade"}
            and not dict(ranking.get("best_actionable", {}) or {})
            and not top
            and score is None
            and not bool(timing_result and timing_result.get("passed") is False)
        )
        reason = (
            message
            if ranking_blocked
            else f"Selected candidate ranked with score {score}." if score is not None else "Ranking state was recorded for the selected candidate."
        )
        entries.append(_entry("signal_ranking", "blocked" if ranking_blocked else "passed", reason, data={"chosen_score": score, "top": top[:5]}))

    prior_block = _first_block(entries)
    if prior_block:
        entries.append(_entry("conservative_filter", "not_reached", f"Pipeline stopped at {prior_block}."))
    else:
        guards = list(filters.get("guards", []) or [])
        hard_blockers = list(filters.get("hard_blockers", []) or [])
        timing_blocked = bool(timing_result and timing_result.get("passed") is False)
        conservative_blocked = bool(
            conservative.get("blocked", False)
            or conservative.get("hard_blocked", False)
            or hard_blockers
            or timing_blocked
        )
        enabled = bool(conservative.get("enabled", False))
        reason = _text(conservative.get("reason"), "")
        if timing_blocked:
            reason = _text(timing_result.get("detail"), reason or "Timing discipline blocked the candidate.")
        if not reason:
            reason = "Conservative filter blocked the candidate." if conservative_blocked else ("Conservative filter passed." if enabled else "Conservative filter not enabled; no block recorded.")
        entries.append(
            _entry(
                "conservative_filter",
                "blocked" if conservative_blocked else "passed",
                reason,
                data={
                    "enabled": enabled,
                    "guards": guards,
                    "hard_blockers": hard_blockers,
                    "reason_codes": list(filters.get("reason_codes", []) or []),
                    "timing_discipline": timing_result,
                },
            )
        )

    prior_block = _first_block(entries)
    if prior_block:
        entries.append(_entry("risk_management", "not_reached", f"Pipeline stopped at {prior_block}."))
    else:
        risk_block_reasons = [
            _text(item.get("reason") or item.get("code") or item.get("name"))
            for item in list(filters.get("hard_blockers", []) or [])
            if any(token in _text(item).lower() for token in ("risk", "loss", "spread", "liquidity", "drawdown"))
        ]
        risk_blocked = bool(risk_block_reasons)
        entries.append(
            _entry(
                "risk_management",
                "blocked" if risk_blocked else "passed",
                "; ".join(risk_block_reasons) if risk_blocked else "Risk state did not record a blocking condition.",
                data=risk,
            )
        )

    prior_block = _first_block(entries)
    has_execution_record = bool(gate_decision or execution_truth)
    if prior_block and not has_execution_record:
        entries.append(_entry("execution", "not_reached", f"Pipeline stopped at {prior_block}."))
    else:
        gate_allowed = gate_decision.get("allowed")
        if gate_allowed is False:
            entries.append(
                _entry(
                    "execution",
                    "blocked",
                    _text(gate_decision.get("reason"), _text(gate_decision.get("detail"), "ExecutionGate blocked the request.")),
                    data={"gate_decision": gate_decision, "execution_truth": execution_truth},
                )
            )
        elif gate_allowed is True:
            entries.append(
                _entry(
                    "execution",
                    "passed",
                    "ExecutionGate approved the request.",
                    data={"gate_decision": gate_decision, "execution_truth": execution_truth},
                )
            )
        else:
            entries.append(
                _entry(
                    "execution",
                    "not_reached",
                    "No ExecutionGate decision is recorded for this cycle.",
                    data={"gate_decision": gate_decision, "execution_truth": execution_truth},
                )
            )

    return {
        "schema": "atlas_why_trace_v1",
        "generated_at": _now_iso(),
        "read_only": True,
        "cycle": {
            "symbol": _text(status.get("symbol")),
            "timeframe": _text(status.get("timeframe")),
            "updated_at": _text(status.get("updated_at")),
            "action": action,
        },
        "stages": entries,
        "first_blocked_stage": _first_block(entries),
        "decision_trace": decision_trace,
        "sources": {
            "status": True,
            "runtime_flags": bool(runtime_flags),
            "execution_snapshot": bool(execution_snapshot),
        },
    }
