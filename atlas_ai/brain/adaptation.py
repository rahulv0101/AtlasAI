from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atlas_ai.brain.chart_learning import summarize_chart_learning
from atlas_ai.brain.regime_monitor import detect_regime_shift


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def _evidence_strength(
    *,
    regime: dict[str, Any],
    learning: dict[str, Any],
    events: list[dict[str, Any]],
) -> tuple[float, dict[str, float], list[str]]:
    observed_changes = list(regime.get("observed_changes", []) or [])
    shift_conf = float(regime.get("shift_confidence", 0.0) or 0.0)
    noise = float(regime.get("noise_score", 0.0) or 0.0)
    setup_change = float(learning.get("setup_quality_change", 0.0) or 0.0)
    rej_change = float(learning.get("filter_rejection_rate_change", 0.0) or 0.0)
    breakout_failure = float(learning.get("breakout_failure_rate", 0.0) or 0.0)
    teaching_count = float(learning.get("operator_teaching_signal_count", 0.0) or 0.0)
    clean_teaching = float(learning.get("clean_operator_teaching_count", 0.0) or 0.0)
    caution_teaching = float(learning.get("caution_operator_teaching_count", 0.0) or 0.0)

    risk_msgs = [str(e.get("message", "") or "").lower() for e in events if str(e.get("type", "")).lower() == "risk"]
    repeating_breakout_fail = sum(1 for m in risk_msgs if "breakout" in m and ("fail" in m or "blocked" in m))
    repeating_breakout_confirm = sum(1 for m in risk_msgs if "breakout" in m and any(k in m for k in ("confirm", "follow-through", "aligned")))
    repeating_rejections = sum(1 for m in risk_msgs if "filter" in m or "blocked" in m)
    teaching_persistence = _clamp(teaching_count / 8.0)
    teaching_confirmation = _clamp((clean_teaching + caution_teaching * 0.7) / 5.0)

    agreement = _clamp((len(observed_changes) / 3.0) * 0.45 + shift_conf * 0.55)
    persistence_signal = min(1.0, (repeating_breakout_fail + repeating_breakout_confirm + repeating_rejections) / 5.0)
    persistence = _clamp(persistence_signal * 0.55 + shift_conf * 0.45 + teaching_persistence * 0.10)
    recency = _clamp(0.45 + min(0.4, len(events) * 0.05))
    cleanliness = _clamp(1.0 - noise)
    regime_consistency = _clamp(shift_conf * 0.7 + (1.0 - noise) * 0.3)
    breakout_follow_through_quality = _clamp(1.0 - breakout_failure)
    rejection_cluster = _clamp(rej_change)
    contradiction = _clamp(abs(setup_change) * 0.5 + (1.0 if (setup_change > 0 and breakout_failure > 0.4) else 0.0) * 0.5 - teaching_confirmation * 0.12)
    confirmation_strength = _clamp(max(repeating_breakout_confirm / 3.0, teaching_confirmation))

    components = {
        "agreement_signals": agreement,
        "persistence": persistence,
        "confirmation_strength": confirmation_strength,
        "recency": recency,
        "cleanliness": cleanliness,
        "regime_consistency": regime_consistency,
        "breakout_follow_through_quality": breakout_follow_through_quality,
        "rejection_failure_clustering": rejection_cluster,
        "contradiction_level": contradiction,
        "noise_level": noise,
        "operator_teaching_persistence": teaching_persistence,
        "operator_teaching_confirmation": teaching_confirmation,
    }

    raw_score = (
        agreement * 0.22
        + persistence * 0.20
        + confirmation_strength * 0.10
        + recency * 0.08
        + cleanliness * 0.12
        + regime_consistency * 0.14
        + breakout_follow_through_quality * 0.08
        + rejection_cluster * 0.10
        + (1.0 - contradiction) * 0.06
    )
    noise_penalty = 1.0 - noise * 0.38
    contradiction_penalty = 1.0 - contradiction * 0.20
    breakout_penalty = 0.08 if breakout_failure > 0.50 else 0.0
    score = _clamp(raw_score * noise_penalty * contradiction_penalty - breakout_penalty)
    score = _clamp(
        score
        + confirmation_strength * 0.18
        + max(0.0, cleanliness - 0.5) * 0.10
        + max(0.0, persistence - 0.45) * 0.08
        - (0.04 if noise > 0.60 else 0.0)
    )

    reasons: list[str] = []
    if agreement >= 0.7:
        reasons.append("strong signal agreement")
    elif agreement <= 0.45:
        reasons.append("mixed signal agreement")
    if persistence >= 0.6:
        reasons.append("evidence persisted across recent windows")
    else:
        reasons.append("evidence persistence is limited")
    if teaching_confirmation >= 0.45:
        reasons.append("operator teaching evidence is reinforcing the read")
    if noise >= 0.55:
        reasons.append("elevated chart noise")
    if contradiction >= 0.45:
        reasons.append("contradictory evidence is present")
    return score, components, reasons


def build_adaptation_summary(status: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    regime = detect_regime_shift(status)
    learning = summarize_chart_learning(status, events)

    shift_conf = float(regime.get("shift_confidence", 0.0) or 0.0)
    conf_speed = float(learning.get("confidence_adjustment_speed", 0.0) or 0.0)
    rej_change = float(learning.get("filter_rejection_rate_change", 0.0) or 0.0)
    sig_stability = float(learning.get("signal_stability_change", 0.0) or 0.0)
    setup_change = float(learning.get("setup_quality_change", 0.0) or 0.0)
    noise = float(regime.get("noise_score", 0.0) or 0.0)

    adaptation_latency = max(1.0, 6.0 - (shift_conf * 4.0 + conf_speed))
    regime_shift_detection_delay = max(1.0, 8.0 - shift_conf * 5.0)
    environment_mismatch_score = max(0.0, min(1.0, rej_change * 0.6 + noise * 0.4))
    overreaction_score = max(0.0, min(1.0, abs(conf_speed) * 0.5 + max(0.0, -sig_stability) * 0.5))
    stability_score = _clamp(1.0 - (overreaction_score * 0.65 + noise * 0.35))
    evidence_strength_score, evidence_components, evidence_reasons = _evidence_strength(regime=regime, learning=learning, events=events)

    guardrails_triggered: list[str] = []
    persistence_ok = float(evidence_components.get("persistence", 0.0)) >= 0.55
    contradiction_level = float(evidence_components.get("contradiction_level", 0.0))
    confirmation_strength = float(evidence_components.get("confirmation_strength", 0.0))
    if not persistence_ok:
        guardrails_triggered.append("fast_path_blocked_low_persistence")
    if contradiction_level > 0.40:
        guardrails_triggered.append("fast_path_blocked_high_contradiction")
    if noise > 0.60:
        guardrails_triggered.append("fast_path_blocked_high_noise")
    if overreaction_score > 0.65:
        guardrails_triggered.append("fast_path_blocked_overreaction_risk")

    learning_mode = "slow_path"
    fast_path_ready = (
        evidence_strength_score >= 0.65
        and confirmation_strength >= 0.60
        and float(evidence_components.get("persistence", 0.0)) >= 0.50
    )
    if fast_path_ready and not guardrails_triggered:
        learning_mode = "fast_path"
    elif evidence_strength_score >= 0.45:
        learning_mode = "medium_path"
    if noise > 0.62 and float(evidence_components.get("persistence", 0.0)) < 0.70:
        learning_mode = "slow_path"
        if "fast_path_blocked_high_noise" not in guardrails_triggered:
            guardrails_triggered.append("fast_path_blocked_high_noise")
    if "fast_path_blocked_overreaction_risk" in guardrails_triggered and learning_mode == "fast_path":
        learning_mode = "medium_path"

    mode_scale = {"slow_path": 0.72, "medium_path": 1.0, "fast_path": 1.28}[learning_mode]
    max_confidence_delta_per_update = {"slow_path": 0.04, "medium_path": 0.08, "fast_path": 0.12}[learning_mode]
    confidence_delta_unbounded = conf_speed * 0.12 * mode_scale
    bounded_confidence_delta = max(-max_confidence_delta_per_update, min(max_confidence_delta_per_update, confidence_delta_unbounded))
    confidence_adjustment_speed_effective = abs(bounded_confidence_delta) / max(1e-6, max_confidence_delta_per_update)
    hold_bias_adjustment_speed = _clamp((1.0 - evidence_strength_score) * 0.65 + (noise * 0.35)) * mode_scale

    # Learning-speed metrics: bounded observable estimates from current evidence window.
    evidence_to_detection_latency = max(0.5, (regime_shift_detection_delay * (1.0 + noise * 0.35)) / mode_scale)
    detection_to_confidence_update_latency = max(0.5, (5.0 - conf_speed * 3.0 + noise * 1.5) / mode_scale)
    detection_to_behavior_change_latency = max(0.5, (adaptation_latency * 1.1 + max(0.0, -setup_change) * 2.0) / mode_scale)
    regime_learning_speed = max(0.0, min(1.0, shift_conf * 0.6 + conf_speed * 0.3 + sig_stability * 0.1))
    breakout_failure_rate = float(learning.get("breakout_failure_rate", 0.0) or 0.0)
    setup_decay_detection_speed = max(
        0.0,
        min(1.0, abs(setup_change) * 0.20 + rej_change * 0.25 + breakout_failure_rate * 0.55),
    )
    recovery_learning_speed = max(
        0.0,
        min(1.0, max(0.0, setup_change) * 0.55 + max(0.0, sig_stability) * 0.25 + (1.0 - noise) * 0.2),
    )
    risk_events = sum(1 for e in events if str(e.get("type", "")).lower() == "risk")
    trades_to_stable_adjustment = max(
        1.0,
        2.0 + (1.0 - regime_learning_speed) * 6.0 + max(0.0, overreaction_score - 0.4) * 3.0 - min(2.0, risk_events * 0.25),
    )

    speed_label = "moderate"
    if adaptation_latency <= 2.5:
        speed_label = "fast"
    elif adaptation_latency >= 4.5:
        speed_label = "slow"

    stability = "stable"
    if overreaction_score > 0.65:
        stability = "overreactive"
    elif noise > 0.55:
        stability = "noisy"

    status_label = "adjusting cautiously"
    if speed_label == "slow":
        status_label = "adjusting slowly"
    if stability == "overreactive":
        status_label = "adjusting too aggressively"
    if learning_mode == "fast_path":
        status_label = "adjusting quickly on strong evidence"

    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "adaptation_status": status_label,
        "active_learning_mode": learning_mode,
        "evidence_strength_score": evidence_strength_score,
        "evidence_components": evidence_components,
        "learning_mode_reasons": evidence_reasons,
        "guardrails_triggered": guardrails_triggered,
        "max_confidence_delta_per_update": max_confidence_delta_per_update,
        "bounded_confidence_delta": bounded_confidence_delta,
        "hold_bias_adjustment_speed": hold_bias_adjustment_speed,
        "adaptation_latency": adaptation_latency,
        "regime_shift_detection_delay": regime_shift_detection_delay,
        "evidence_to_detection_latency": evidence_to_detection_latency,
        "detection_to_confidence_update_latency": detection_to_confidence_update_latency,
        "detection_to_behavior_change_latency": detection_to_behavior_change_latency,
        "trades_to_stable_adjustment": trades_to_stable_adjustment,
        "regime_learning_speed": regime_learning_speed,
        "setup_decay_detection_speed": setup_decay_detection_speed,
        "recovery_learning_speed": recovery_learning_speed,
        "confidence_adjustment_speed": conf_speed,
        "confidence_adjustment_speed_effective": confidence_adjustment_speed_effective,
        "setup_quality_change": setup_change,
        "filter_rejection_rate_change": rej_change,
        "signal_stability_change": sig_stability,
        "environment_mismatch_score": environment_mismatch_score,
        "overreaction_score": overreaction_score,
        "stability_score": stability_score,
        "observed_changes": list(regime.get("observed_changes", []) or []),
        "regime_state": regime.get("regime_state", "stable"),
        "regime_shift_confidence": shift_conf,
        "stability": stability,
        "speed_label": speed_label,
        # Explicit safety guardrail statement.
        "safety_note": "Observation and reporting only. No automatic core strategy/risk rule rewrite is performed.",
        "learning_speed_note": "Learning speed is measured from evidence and response timing proxies; no uncontrolled self-modification is enabled.",
    }
