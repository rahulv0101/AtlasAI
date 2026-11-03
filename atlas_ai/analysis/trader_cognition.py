from __future__ import annotations

from typing import Any


def _text(value: Any, default: str = "unknown") -> str:
    raw = str(value if value is not None else "").strip()
    return raw or default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return float(default)


def _primary_blocker(status: dict[str, Any], analysis: dict[str, Any]) -> str:
    explicit = _text(status.get("primary_blocker", ""), "")
    if explicit:
        return explicit
    blockers = list(analysis.get("hard_blockers", []) or analysis.get("blockers", []) or [])
    if blockers:
        return _text(blockers[0])
    return _text(status.get("message", ""), "")


def build_trader_cognition(
    *,
    status: dict[str, Any],
    runtime: dict[str, Any] | None = None,
    mandate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build observable trader-like reasoning without creating execution authority."""

    st = dict(status or {})
    rt = dict(runtime or {})
    md = dict(mandate or {})
    chosen = dict(st.get("chosen", {}) or {})
    context = dict(st.get("context_snapshot", {}) or {})
    analysis = dict(st.get("decision_analysis", {}) or {})
    action = _text(st.get("action", "hold"), "hold").lower()
    side = _text(chosen.get("side", "flat"), "flat").lower()
    confidence = _safe_float(chosen.get("confidence", 0.0))
    threshold = _safe_float(
        st.get(
            "min_confidence_threshold",
            dict(st.get("thresholds", {}) or {}).get("confidence", 0.0),
        )
    )
    regime = _text(dict(st.get("regime", {}) or {}).get("name", context.get("regime", context.get("trend_state", "unknown"))))
    structure = _text(context.get("trend_structure", context.get("trend_state", "unknown")))
    momentum = _text(context.get("momentum", context.get("momentum_state", "unknown")))
    location = _text(context.get("location", context.get("location_state", "unknown")))
    session = _text(context.get("session", st.get("session", "unknown")))
    spread = _text(context.get("spread_state", context.get("spread_status", "unknown")))
    htf = _text(dict(st.get("higher_timeframe", {}) or {}).get("bias", st.get("higher_tf_bias", "unknown")))
    blocker = _primary_blocker(st, analysis)
    recommended = _text(analysis.get("recommended_action", action), action)
    service_goal = _text(
        dict(md.get("operator_intent", {}) or {}).get("primary_service_goal", ""),
        "Serve the operator by pursuing profitable XAUUSD trading decisions inside approved risk boundaries.",
    )

    if side in {"buy", "sell"}:
        thesis = f"{side.upper()} thesis from {chosen.get('strategy', 'selected_strategy')} if evidence stays aligned."
    else:
        thesis = "No directional thesis is strong enough yet."

    counter_bits: list[str] = []
    if blocker:
        counter_bits.append(blocker.rstrip("."))
    if htf in {"buy", "sell"} and side in {"buy", "sell"} and htf != side:
        counter_bits.append(f"higher timeframe bias is {htf} against {side}")
    if confidence and threshold and confidence < threshold:
        counter_bits.append(f"confidence {confidence:.3f} is below threshold {threshold:.3f}")
    if spread in {"wide", "poor", "blocked", "elevated"}:
        counter_bits.append(f"spread/liquidity condition is {spread}")
    counter_thesis = "; ".join(dict.fromkeys(counter_bits)) or "No dominant invalidation is visible in the current status."

    discipline_flags: list[str] = []
    if location in {"mid_range", "middle", "unknown"}:
        discipline_flags.append("avoid chasing from poor location")
    if momentum in {"weak", "mixed", "unknown"}:
        discipline_flags.append("wait for cleaner momentum")
    if "revenge" not in discipline_flags and int(dict(st.get("risk", {}) or {}).get("consecutive_losses", 0) or 0) > 0:
        discipline_flags.append("do not revenge-trade after recent losses")
    if not discipline_flags:
        discipline_flags.append("stay patient and only act if the edge remains clean")

    if action in {"execute", "paper"}:
        decision = "enter_or_request_execution"
    elif action == "blocked":
        decision = "block"
    elif side in {"buy", "sell"} and confidence >= max(threshold, 0.0):
        decision = "wait_for_execution_gate"
    else:
        decision = "wait"

    required: list[str] = []
    if confidence < threshold:
        required.append("stronger confidence")
    if side not in {"buy", "sell"}:
        required.append("directional candidate")
    if htf in {"buy", "sell"} and side in {"buy", "sell"} and htf != side:
        required.append("higher-timeframe alignment or approved context-aware exception")
    support = dict(st.get("same_side_support", {}) or {})
    if int(support.get("count", 0) or 0) < int(rt.get("min_same_side_signals", st.get("min_same_side_signals", 2)) or 2):
        required.append("stronger same-side support")
    if not required and decision in {"wait", "block"}:
        required.append("cleaner setup evidence")

    return {
        "schema": "atlas_trader_cognition_v1",
        "mode": "bounded_trader_like_reasoning",
        "observation": (
            f"{session} session, regime={regime}, structure={structure}, "
            f"momentum={momentum}, location={location}, spread={spread}."
        ),
        "thesis": thesis,
        "counter_thesis": counter_thesis,
        "discipline_check": {
            "status": "disciplined",
            "checks": list(dict.fromkeys(discipline_flags)),
            "policy_note": "Simulated trader discipline only; Atlas has no real emotions or private desires.",
        },
        "service_intent": {
            "statement": service_goal,
            "boundary": "Profit pursuit remains subordinate to risk limits, evidence quality, and ExecutionGate.",
            "real_feelings": False,
        },
        "decision": decision,
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 4),
        "required_next_evidence": list(dict.fromkeys(required))[:6],
        "autonomy_boundary": {
            "target_role": _text(md.get("autonomy_model", {}).get("target_role", "bounded_autonomous_trader")),
            "execution_authority": "ExecutionGate",
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_bypass_execution_policy": True,
        },
    }
