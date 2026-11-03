from __future__ import annotations

from collections import Counter
from typing import Any

from atlas_ai.strategies.strategy_inventor import StrategyInventor


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _compact_list(values: Any, limit: int = 6) -> list[str]:
    out: list[str] = []
    for item in list(values or [])[: max(1, int(limit))]:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _recent_outcomes(learning_state: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    state = dict(learning_state.get("state", learning_state) or {})
    intelligence = dict(state.get("trade_outcome_intelligence", {}) or {})
    outcome_memory = dict(intelligence.get("outcome_memory", {}) or {})
    trades = list(outcome_memory.get("trades", []) or [])
    if not trades:
        loss_memory = dict(state.get("loss_memory", {}) or {})
        trades = list(loss_memory.get("recent_outcomes", []) or [])
    rows = [dict(row or {}) for row in trades if str(dict(row or {}).get("outcome", "")).lower() in {"win", "loss", "draw"}]
    return rows[-max(1, int(limit)) :]


def _learning_lanes(learning_state: dict[str, Any]) -> dict[str, Any]:
    state = dict(learning_state.get("state", learning_state) or {})
    outcomes = _recent_outcomes(state, limit=20)
    counts = Counter(str(row.get("outcome", "")).lower() for row in outcomes)
    closed = counts.get("win", 0) + counts.get("loss", 0)
    win_rate = (counts.get("win", 0) / closed) if closed else 0.0
    loss_memory = dict(state.get("loss_memory", {}) or {})
    allocator = dict(state.get("allocator", {}) or {})
    learned = dict(state.get("learned_patterns", {}) or state.get("pattern_memory", {}) or {})
    return {
        "recent_sample_size": len(outcomes),
        "closed_sample_size": closed,
        "wins": counts.get("win", 0),
        "losses": counts.get("loss", 0),
        "draws": counts.get("draw", 0),
        "win_rate": round(win_rate, 4),
        "loss_memory_available": bool(loss_memory),
        "allocator_strategy_count": len(allocator),
        "pattern_memory_available": bool(learned),
        "latest_outcome": dict(outcomes[-1]) if outcomes else {},
    }


def _pattern_counter(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        parts = [
            str(row.get("setup_family") or row.get("setup") or row.get("strategy") or "unknown"),
            str(row.get("session") or "unknown_session"),
            str(row.get("side") or "unknown_side"),
            str(row.get("outcome") or "unknown_outcome"),
        ]
        counter["|".join(parts)] += 1
    return [{"fingerprint": key, "count": count} for key, count in counter.most_common(6)]


def build_cognitive_core(
    *,
    status: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
    learning_state: dict[str, Any] | None = None,
    knowledge_payload: dict[str, Any] | None = None,
    adaptation_summary: dict[str, Any] | None = None,
    evidence_separation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build Atlas' read-only system-wide understanding layer.

    The cognitive core joins state, lessons, learning memory, and decision evidence
    into one inspectable model. It intentionally does not mutate strategy, risk, or
    execution policy.
    """
    learning_state = dict(learning_state or {})
    knowledge_payload = dict(knowledge_payload or {})
    adaptation_summary = dict(adaptation_summary or {})
    evidence_separation = dict(evidence_separation or {})
    context = dict(status.get("context_snapshot", {}) or {})
    chosen = dict(status.get("chosen", {}) or {})
    execution = dict(runtime.get("execution", {}) or {})
    session = dict(execution.get("session_control", {}) or {})
    system_health = dict(status.get("system_health", {}) or {})
    learning = _learning_lanes(learning_state)
    outcomes = _recent_outcomes(learning_state, limit=20)
    lessons = _compact_list(dict(knowledge_payload.get("lessons_learned", {}) or {}).get("lessons", []), limit=8)
    teaching_reviews = list(dict(knowledge_payload.get("operator_teaching_reviews", {}) or {}).get("reviews", []) or [])
    recent_events = [dict(e or {}) for e in list(events or [])[-12:]]
    blocker = str(status.get("primary_blocker") or status.get("message") or "").strip()
    side = str(chosen.get("side", "flat") or "flat").lower()
    confidence = _safe_float(chosen.get("confidence", 0.0))
    threshold = _safe_float(status.get("min_confidence_threshold", 0.0))
    readiness = "ready_for_review" if confidence >= threshold and side in {"buy", "sell"} else "developing_or_blocked"
    if evidence_separation:
        readiness = str(evidence_separation.get("setup_readiness", readiness) or readiness)

    current_question = {
        "what_atlas_sees": {
            "symbol": str(status.get("symbol", "XAUUSD") or "XAUUSD"),
            "timeframe": str(status.get("timeframe", "M1") or "M1"),
            "session": str(session.get("current_session") or context.get("session") or "unknown"),
            "regime": str(context.get("regime") or dict(status.get("market_regime", {}) or {}).get("regime") or "unknown"),
            "structure": str(context.get("trend_structure") or "unknown"),
            "momentum": str(context.get("momentum") or "unknown"),
            "location": str(context.get("location") or "unknown"),
            "blocker": blocker,
        },
        "decision_lanes": {
            "market_bias": str(evidence_separation.get("market_bias", "unknown")),
            "signal_direction": str(evidence_separation.get("signal_direction", side if side in {"buy", "sell"} else "neutral")),
            "higher_timeframe_bias": str(evidence_separation.get("higher_timeframe_bias", "unknown")),
            "final_trade_bias": str(evidence_separation.get("final_trade_bias", "neutral")),
            "confidence": round(confidence, 4),
            "threshold": round(threshold, 4),
            "setup_readiness": readiness,
            "execution_permission": str(evidence_separation.get("execution_permission", "not_allowed")),
        },
    }
    critic_flags: list[str] = []
    if blocker:
        critic_flags.append(f"current_blocker:{blocker}")
    if str(evidence_separation.get("execution_permission", "not_allowed")) != "allowed":
        critic_flags.append("execution_not_allowed")
    if side not in {"buy", "sell"}:
        critic_flags.append("no_directional_signal")
    if _safe_float(dict(status.get("risk", {}) or {}).get("spread_points", 0.0)) >= 35.0:
        critic_flags.append("spread_or_execution_friction_watch")
    if str(context.get("location", "")).lower() == "mid_range":
        critic_flags.append("mid_range_location_watch")
    invented_strategy_plans = StrategyInventor().invent(
        status=status,
        runtime=runtime,
        learning_state=learning_state,
        cognitive_core={
            "world_model": current_question["what_atlas_sees"],
            "decision_understanding": current_question["decision_lanes"],
            "learning_state": {
                "stability_score": adaptation_summary.get("stability_score", 0.0),
                "overreaction_score": adaptation_summary.get("overreaction_score", 0.0),
            },
        },
        limit=3,
    )

    return {
        "schema": "atlas_cognitive_core_v1",
        "mode": "read_only_understanding",
        "world_model": current_question["what_atlas_sees"],
        "decision_understanding": current_question["decision_lanes"],
        "trade_memory": learning,
        "pattern_memory": {
            "top_recent_fingerprints": _pattern_counter(outcomes),
            "lesson_count": len(lessons),
            "latest_lessons": lessons[:5],
            "operator_teaching_review_count": len(teaching_reviews),
        },
        "learning_state": {
            "active_learning_mode": str(adaptation_summary.get("active_learning_mode", "unknown")),
            "evidence_strength_score": round(_safe_float(adaptation_summary.get("evidence_strength_score", 0.0)), 4),
            "stability_score": round(_safe_float(adaptation_summary.get("stability_score", 0.0)), 4),
            "overreaction_score": round(_safe_float(adaptation_summary.get("overreaction_score", 0.0)), 4),
            "guardrails_triggered": list(adaptation_summary.get("guardrails_triggered", []) or []),
        },
        "decision_critic": {
            "flags": critic_flags,
            "main_question_before_execution": "Is this setup still valid after separating market bias, signal direction, setup readiness, confidence, and execution permission?",
            "loss_review_question": "If this loses, was the failure direction, timing, location, regime, volatility, spread, or execution friction?",
        },
        "strategy_invention": {
            "mode": "ai_designed_proposals_only",
            "candidate_count": len(invented_strategy_plans),
            "candidates": [plan.to_dict() for plan in invented_strategy_plans],
            "promotion_path": [
                "gold_validation",
                "offline_backtest",
                "shadow",
                "demo",
                "operator_approved_live_proposal",
                "execution_gate",
            ],
        },
        "teaching_plan": [
            "Classify each win/loss by setup fingerprint, session, side, regime, location, momentum, volatility, spread, and higher-timeframe alignment.",
            "Separate idea quality from entry timing so Atlas learns whether the trade concept was wrong or just early/late.",
            "Use repeated evidence to update explanatory memory and bounded confidence haircuts only; do not rewrite strategy/risk/execution silently.",
        ],
        "improvement_backlog": [
            "Show cognitive-core lanes in Ask Atlas answers and debug traces.",
            "Keep post-trade lessons structured and fingerprint-specific.",
            "Use visual chart reads as advisory evidence only, matched against candle-derived structure.",
        ],
        "safety_contract": {
            "can_explain": True,
            "can_learn_from_evidence": True,
            "can_propose_changes": True,
            "can_design_strategy_proposals": True,
            "can_mutate_strategy_policy": False,
            "can_mutate_risk_policy": False,
            "can_bypass_execution_gate": False,
            "policy_note": "Cognitive core is Atlas' AI coordination layer. It may design strategy proposals, but does not change strategy, risk, or execution policy without the guarded promotion path.",
        },
        "recent_event_count": len(recent_events),
    }
