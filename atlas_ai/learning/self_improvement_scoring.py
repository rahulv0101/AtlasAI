from __future__ import annotations

from typing import Any

from atlas_ai.learning.promotion_thresholds import evidence_strength_grade_thresholds


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def score_evidence_strength(
    *,
    sample_size_score: Any,
    persistence_score: Any,
    cross_window_score: Any,
    regime_alignment_score: Any,
    risk_impact_score: Any,
    overreaction_risk_score: Any,
) -> dict[str, float | str]:
    sample = _clamp(sample_size_score)
    persistence = _clamp(persistence_score)
    cross_window = _clamp(cross_window_score)
    regime = _clamp(regime_alignment_score)
    risk = _clamp(risk_impact_score)
    overreaction = _clamp(overreaction_risk_score)

    strength = (
        (sample * 0.22)
        + (persistence * 0.22)
        + (cross_window * 0.18)
        + (regime * 0.16)
        + (risk * 0.12)
        + ((1.0 - overreaction) * 0.10)
    )
    strength = round(max(0.0, min(1.0, strength)), 4)
    thresholds = evidence_strength_grade_thresholds()
    if strength >= float(thresholds["strong_min_strength"]):
        grade = "strong"
    elif strength >= float(thresholds["moderate_min_strength"]):
        grade = "moderate"
    else:
        grade = "weak"

    return {
        "strength": strength,
        "grade": grade,
        "sample_count_score": round(sample, 4),
        "persistence_score": round(persistence, 4),
        "cross_window_score": round(cross_window, 4),
        "regime_alignment_score": round(regime, 4),
        "risk_impact_score": round(risk, 4),
        "overreaction_risk_score": round(overreaction, 4),
    }
