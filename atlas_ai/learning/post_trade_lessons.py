from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _failure_axis(*, classification: str, context: dict[str, Any], outcome: str) -> str:
    if outcome != "loss":
        return "none"
    code = _label(classification)
    location = _label(context.get("location") or context.get("entry_location_state"))
    momentum = _label(context.get("momentum") or context.get("entry_momentum_state"))
    spread = _safe_float(context.get("spread_points", 0.0))
    htf_aligned = bool(context.get("htf_aligned", False))
    if code in {"spread_liquidity_distortion", "news_spike"} or spread >= 35.0:
        return "execution_friction"
    if code in {"late_entry", "poor_timing", "stop_hunt"}:
        return "timing"
    if location == "mid_range" or code in {"weak_pullback", "weak_structure"}:
        return "location_or_structure"
    if not htf_aligned and _label(context.get("higher_tf_bias")) in {"buy", "sell"}:
        return "higher_timeframe_conflict"
    if momentum in {"weak", "mixed"} or code == "false_continuation":
        return "confirmation_quality"
    return "unknown_loss_axis"


def _idea_quality(*, outcome: str, context: dict[str, Any], failure_axis: str) -> str:
    if outcome == "win":
        return "validated"
    if failure_axis in {"timing", "execution_friction"}:
        return "possibly_valid_but_poorly_executed"
    if failure_axis in {"location_or_structure", "confirmation_quality", "higher_timeframe_conflict"}:
        return "unproven_or_weak"
    return "unknown"


def _timing_quality(*, outcome: str, context: dict[str, Any], duration_minutes: float, failure_axis: str) -> str:
    if outcome == "win":
        return "acceptable"
    if failure_axis == "timing":
        return "early_or_late"
    if duration_minutes <= 6.0:
        return "too_early_or_stop_hunt_vulnerable"
    if _label(context.get("location") or context.get("entry_location_state")) == "mid_range":
        return "poor_location_timing"
    return "needs_review"


def build_post_trade_lesson(outcome_row: dict[str, Any], decision_context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create a bounded lesson from a closed trade.

    This is teaching evidence only. It does not mutate strategy, risk, or execution policy.
    """
    row = dict(outcome_row or {})
    context = {**dict(decision_context or {}), **dict(row.get("context", {}) or {})}
    outcome = _label(row.get("outcome"), "draw")
    classification = _label(row.get("loss_classification") or dict(row.get("post_trade_review", {}) or {}).get("classification"), "none")
    duration = _safe_float(row.get("duration_minutes", 0.0))
    failure_axis = _failure_axis(classification=classification, context=context, outcome=outcome)
    idea_quality = _idea_quality(outcome=outcome, context=context, failure_axis=failure_axis)
    timing_quality = _timing_quality(outcome=outcome, context=context, duration_minutes=duration, failure_axis=failure_axis)
    setup_family = _label(row.get("setup_family"))
    side = _label(row.get("side"), "flat")
    session = _label(row.get("entry_session") or row.get("session"))
    regime = _label(row.get("regime") or context.get("regime"))
    location = _label(context.get("location") or context.get("entry_location_state"))
    momentum = _label(context.get("momentum") or context.get("entry_momentum_state"))
    htf_bias = _label(context.get("higher_tf_bias") or context.get("higher_timeframe_bias"))
    spread = _safe_float(context.get("spread_points", row.get("spread_points", 0.0)))
    recognition = [
        f"setup={setup_family}",
        f"side={side}",
        f"session={session}",
        f"regime={regime}",
        f"location={location}",
        f"momentum={momentum}",
    ]
    if htf_bias != "unknown":
        recognition.append(f"higher_tf_bias={htf_bias}")
    if spread > 0:
        recognition.append(f"spread_points={spread:.1f}")
    if outcome == "loss":
        action = "treat_similar_setup_as_developing_until_confirmation_improves"
        if failure_axis == "execution_friction":
            action = "wait_for_execution_friction_to_normalize_before_repeating"
        elif failure_axis == "timing":
            action = "wait_for_fresher_entry_or_retest_before_repeating"
    elif outcome == "win":
        action = "remember_as_positive_fingerprint_when_conditions_match"
    else:
        action = "record_as_neutral_evidence_only"
    return {
        "schema": "atlas_post_trade_lesson_v1",
        "source": "closed_trade_learning",
        "outcome": outcome,
        "classification": classification,
        "failure_axis": failure_axis,
        "idea_quality": idea_quality,
        "timing_quality": timing_quality,
        "setup_family": setup_family,
        "side": side,
        "session": session,
        "regime": regime,
        "pattern_fingerprint": str(row.get("pattern_fingerprint", "") or ""),
        "future_recognition": recognition,
        "next_time_action": action,
        "bounded_learning_only": True,
        "validation_required": True,
        "policy_note": "Post-trade lesson may inform explanation and bounded confidence evidence only; it cannot rewrite strategy, risk, guards, or execution policy.",
    }
