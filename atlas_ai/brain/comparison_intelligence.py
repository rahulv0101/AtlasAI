from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from atlas_ai.brain.types import ComparisonIntelligenceResult


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class ComparisonIntelligence:
    def __init__(self, history_limit: int = 60) -> None:
        self._history: deque[dict[str, float]] = deque(maxlen=history_limit)

    def evaluate(
        self,
        *,
        adaptation_summary: dict[str, Any],
        performance_feedback: dict[str, Any],
        recommendation_result: dict[str, Any],
        paper_tuning: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> ComparisonIntelligenceResult:
        pf_scores = dict(performance_feedback.get("scores", {}) or {})
        risk_msgs = [str(e.get("message", "")).lower() for e in events if str(e.get("type", "")).lower() == "risk"]
        blocker_pressure = float(sum(1 for m in risk_msgs if "blocked" in m or "filter" in m))
        blocker_pressure = min(4.0, blocker_pressure) / 4.0

        comparisons = list(dict(paper_tuning or {}).get("comparisons", []) or [])
        tuning_score = 0.0
        if comparisons:
            outcome = str(comparisons[0].get("outcome", "inconclusive"))
            if outcome == "helpful":
                tuning_score = 0.2
            elif outcome == "hurts_safety":
                tuning_score = -0.25

        current = {
            "recommendation_quality": float(pf_scores.get("recommendation_quality_score", 0.5) or 0.5),
            "adaptation_effectiveness": float(pf_scores.get("adaptation_effectiveness_score", 0.5) or 0.5),
            "stability": float(pf_scores.get("stability_quality_score", 0.5) or 0.5),
            "regime_confidence": float(adaptation_summary.get("regime_shift_confidence", 0.0) or 0.0),
            "blocker_pressure": blocker_pressure,
            "tuning_effectiveness": tuning_score,
        }

        baseline = self._rolling_average()
        sparse = baseline is None
        if sparse:
            baseline = dict(current)
        deltas = {
            "recommendation_quality_delta": current["recommendation_quality"] - baseline["recommendation_quality"],
            "adaptation_speed_delta": current["adaptation_effectiveness"] - baseline["adaptation_effectiveness"],
            "stability_delta": current["stability"] - baseline["stability"],
            "blocker_pressure_delta": baseline["blocker_pressure"] - current["blocker_pressure"],  # lower blocker pressure is better
            "regime_confidence_delta": current["regime_confidence"] - baseline["regime_confidence"],
            "tuning_effectiveness_delta": current["tuning_effectiveness"] - baseline["tuning_effectiveness"],
        }
        deltas = {k: round(_clamp(v), 3) for (k, v) in deltas.items()}

        direction_score = (
            deltas["recommendation_quality_delta"]
            + deltas["adaptation_speed_delta"]
            + deltas["stability_delta"]
            + deltas["blocker_pressure_delta"]
            + 0.5 * deltas["regime_confidence_delta"]
            + 0.5 * deltas["tuning_effectiveness_delta"]
        ) / 5.0
        if sparse:
            direction = "no clear comparison yet"
        elif direction_score > 0.03:
            direction = "improving"
        elif direction_score < -0.03:
            direction = "degrading"
        else:
            direction = "mixed"

        improved: list[str] = []
        worsened: list[str] = []
        for key, value in deltas.items():
            if value > 0.03:
                improved.append(key)
            elif value < -0.03:
                worsened.append(key)

        if sparse:
            safe = "No clear comparison yet. Keep collecting bounded paper evidence."
        elif worsened and improved:
            safe = "Comparison is mixed. Keep caution and monitor weak deltas before any further change."
        elif worsened:
            safe = "Comparison is negative on key dimensions. Hold conservative posture and review weak areas first."
        else:
            safe = "Comparison is positive but bounded. Continue paper validation before relaxing caution."

        self._history.append(dict(current))
        return ComparisonIntelligenceResult(
            direction=direction,
            compared_to=f"rolling_baseline_last_{len(self._history)}",
            deltas=deltas,
            improved=improved[:5],
            worsened=worsened[:5],
            safe_interpretation=safe,
            sparse_data=sparse,
        )

    def _rolling_average(self) -> dict[str, float] | None:
        if not self._history:
            return None
        keys = list(self._history[0].keys())
        return {k: sum(float(x.get(k, 0.0)) for x in self._history) / max(1, len(self._history)) for k in keys}

    @staticmethod
    def to_dict(result: ComparisonIntelligenceResult) -> dict[str, Any]:
        return asdict(result)

