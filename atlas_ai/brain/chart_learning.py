from __future__ import annotations

from typing import Any


def summarize_chart_learning(status: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    confidence = float(chosen.get("confidence", 0.0) or 0.0)
    threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
    gap = confidence - threshold

    risk_msgs = [str(e.get("message", "") or "").lower() for e in events if str(e.get("type", "")).lower() == "risk"]
    teaching_msgs = [
        str(e.get("message", "") or "").lower()
        for e in events
        if str(e.get("type", "")).lower() == "learning" and "operator teaching review recorded" in str(e.get("message", "")).lower()
    ]
    filter_rejections = sum(1 for m in risk_msgs if "filter" in m or "blocked" in m)
    breakout_fail = sum(1 for m in risk_msgs if "breakout" in m and ("fail" in m or "blocked" in m))
    clean_teaching = sum(1 for m in teaching_msgs if any(label in m for label in ("correct_hold", "good_trade", "execution_issue")))
    caution_teaching = sum(1 for m in teaching_msgs if any(label in m for label in ("too_cautious", "missed_trade")))
    setup_quality_change = -0.15 if filter_rejections > 0 else 0.05
    if clean_teaching >= 3 and filter_rejections == 0:
        setup_quality_change = max(setup_quality_change, 0.12)
    elif caution_teaching >= 3:
        setup_quality_change = max(setup_quality_change, 0.08)

    return {
        "confidence_adjustment_speed": min(1.0, abs(gap) * 0.8 + min(0.18, clean_teaching * 0.03 + caution_teaching * 0.02)),
        "setup_quality_change": setup_quality_change,
        "filter_rejection_rate_change": min(1.0, filter_rejections / max(1, len(risk_msgs) + 1)),
        "breakout_failure_rate": min(1.0, breakout_fail / max(1, len(risk_msgs) + 1)),
        "signal_stability_change": (-0.2 if confidence < threshold else 0.1) + min(0.18, clean_teaching * 0.03),
        "confidence_gap": gap,
        "operator_teaching_signal_count": len(teaching_msgs),
        "clean_operator_teaching_count": clean_teaching,
        "caution_operator_teaching_count": caution_teaching,
    }
