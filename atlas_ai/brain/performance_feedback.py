from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from atlas_ai.brain.types import PerformanceFeedbackItem, PerformanceFeedbackResult


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class PerformanceFeedbackLoop:
    def __init__(self, history_limit: int = 40) -> None:
        self._history: deque[dict[str, float]] = deque(maxlen=history_limit)

    def evaluate(
        self,
        *,
        status: dict[str, Any],
        runtime: dict[str, Any],
        adaptation_summary: dict[str, Any],
        recommendation_result: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> PerformanceFeedbackResult:
        evidence_strength = float(adaptation_summary.get("evidence_strength_score", 0.0) or 0.0)
        overreaction = float(adaptation_summary.get("overreaction_score", 0.0) or 0.0)
        stability_score = float(adaptation_summary.get("stability_score", 0.0) or 0.0)
        regime_conf = float(adaptation_summary.get("regime_shift_confidence", 0.0) or 0.0)
        detection_delay = float(adaptation_summary.get("regime_shift_detection_delay", 0.0) or 0.0)
        conf_update_latency = float(adaptation_summary.get("detection_to_confidence_update_latency", 0.0) or 0.0)
        regime_state = str(adaptation_summary.get("regime_state", "stable"))
        learning_mode = str(adaptation_summary.get("active_learning_mode", "slow_path"))
        readiness_low = not bool(runtime.get("execute_trades", False))
        action = str(status.get("action", "unknown")).lower()

        rec_items = list(dict(recommendation_result or {}).get("items", []) or [])
        top_rec = rec_items[0] if rec_items else {}
        top_category = str(top_rec.get("category", "action_posture"))
        top_rec_text = str(top_rec.get("recommendation", "No recommendation available."))

        risk_msgs = [str(e.get("message", "")).lower() for e in events if str(e.get("type", "")).lower() == "risk"]
        regime_flip_events = sum(1 for m in risk_msgs if "regime flip" in m or "regime changed" in m)
        contradiction_events = sum(1 for m in risk_msgs if "mixed" in m or "contradict" in m)
        blocker_events = sum(1 for m in risk_msgs if "blocked" in m or "filter" in m)

        # Bounded scoring with explicit rules.
        adaptation_effectiveness_score = _clamp(
            0.38 * evidence_strength
            + 0.22 * (1.0 - min(1.0, detection_delay / 8.0))
            + 0.20 * (1.0 - min(1.0, conf_update_latency / 8.0))
            + 0.20 * (1.0 - overreaction)
        )
        recommendation_quality_score = _clamp(
            0.45 * (1.0 if top_category in {"action_posture", "blocker_resolution"} else 0.65)
            + 0.35 * (1.0 if (evidence_strength < 0.5 and "hold" in top_rec_text.lower()) or evidence_strength >= 0.5 else 0.6)
            + 0.20 * (1.0 if blocker_events == 0 or top_category in {"blocker_resolution", "action_posture"} else 0.5)
        )
        regime_interpretation_score = _clamp(
            0.40 * regime_conf
            + 0.30 * (1.0 - min(1.0, regime_flip_events / 3.0))
            + 0.15 * (1.0 - min(1.0, contradiction_events / 3.0))
            + 0.15 * (1.0 - overreaction)
        )
        confidence_update_quality_score = _clamp(
            0.5 * (1.0 - min(1.0, conf_update_latency / 7.0))
            + 0.25 * (1.0 - overreaction)
            + 0.25 * (0.7 if learning_mode == "medium_path" else (0.6 if learning_mode == "slow_path" else 0.8))
        )
        stability_quality_score = _clamp(
            0.50 * stability_score + 0.20 * (1.0 - min(1.0, regime_flip_events / 4.0)) + 0.30 * (1.0 - overreaction)
        )

        current = {
            "adaptation_effectiveness_score": adaptation_effectiveness_score,
            "recommendation_quality_score": recommendation_quality_score,
            "regime_interpretation_score": regime_interpretation_score,
            "confidence_update_quality_score": confidence_update_quality_score,
            "stability_quality_score": stability_quality_score,
        }
        prev_avg = self._rolling_average()
        self._history.append(dict(current))
        if prev_avg is None:
            recent_improvement_score = 0.5
            direction = "no clear change"
        else:
            prev_comp = sum(prev_avg.values()) / max(1, len(prev_avg))
            now_comp = sum(current.values()) / max(1, len(current))
            recent_improvement_score = _clamp(0.5 + (now_comp - prev_comp) * 1.5)
            if now_comp - prev_comp > 0.05:
                direction = "improving"
            elif now_comp - prev_comp < -0.05:
                direction = "degrading"
            else:
                direction = "no clear change"
        current["recent_improvement_score"] = recent_improvement_score

        items: list[PerformanceFeedbackItem] = []
        notes: list[str] = ["performance_feedback_is_observe_evaluate_only"]
        scoring_breakdown: dict[str, str] = {}
        items.append(
            PerformanceFeedbackItem(
                category="adaptation_review",
                label="effective" if adaptation_effectiveness_score >= 0.62 else ("partially effective" if adaptation_effectiveness_score >= 0.45 else "ineffective"),
                summary=(
                    "Adaptation response appears aligned with recent evidence."
                    if adaptation_effectiveness_score >= 0.62
                    else "Adaptation response shows some lag or mismatch to evidence."
                ),
                confidence=_clamp(0.55 + 0.35 * evidence_strength),
                priority=3,
            )
        )
        scoring_breakdown["adaptation_review"] = (
            f"evidence_strength={evidence_strength:.2f}, detection_delay={detection_delay:.2f}, "
            f"confidence_update_latency={conf_update_latency:.2f}, overreaction={overreaction:.2f}"
        )
        items.append(
            PerformanceFeedbackItem(
                category="regime_review",
                label="stable" if regime_interpretation_score >= 0.62 else "partially stable",
                summary=(
                    "Regime interpretation remained mostly stable."
                    if regime_interpretation_score >= 0.62
                    else "Regime interpretation had uncertainty or flip-risk."
                ),
                confidence=_clamp(0.5 + 0.35 * regime_conf),
                priority=4,
            )
        )
        scoring_breakdown["regime_review"] = (
            f"regime_confidence={regime_conf:.2f}, regime_flip_events={regime_flip_events}, "
            f"contradictions={contradiction_events}, overreaction={overreaction:.2f}"
        )
        items.append(
            PerformanceFeedbackItem(
                category="recommendation_review",
                label="appropriate" if recommendation_quality_score >= 0.6 else "needs refinement",
                summary=(
                    "Recent recommendation posture was appropriately cautious."
                    if recommendation_quality_score >= 0.6
                    else "Recommendation prioritization may need refinement."
                ),
                confidence=_clamp(0.55 + 0.3 * recommendation_quality_score),
                priority=3,
            )
        )
        scoring_breakdown["recommendation_review"] = (
            f"top_category={top_category}, blocker_events={blocker_events}, "
            f"evidence_strength={evidence_strength:.2f}, top_rec={top_rec_text[:80]}"
        )
        items.append(
            PerformanceFeedbackItem(
                category="stability_review",
                label="stable" if stability_quality_score >= 0.62 else "reactivity concern",
                summary=(
                    "Behavior remained stable with bounded reactivity."
                    if stability_quality_score >= 0.62
                    else "Behavior may be too reactive or inconsistent."
                ),
                confidence=_clamp(0.5 + 0.35 * stability_quality_score),
                priority=1 if stability_quality_score < 0.62 else 4,
            )
        )
        scoring_breakdown["stability_review"] = (
            f"stability_score={stability_score:.2f}, regime_flip_events={regime_flip_events}, overreaction={overreaction:.2f}"
        )
        items.append(
            PerformanceFeedbackItem(
                category="learning_review",
                label="timely" if confidence_update_quality_score >= 0.6 else "slightly slow",
                summary=(
                    "Confidence adjustments were mostly proportional to evidence."
                    if confidence_update_quality_score >= 0.6
                    else "Confidence adjustments may lag strong evidence persistence."
                ),
                confidence=_clamp(0.5 + 0.35 * confidence_update_quality_score),
                priority=2 if confidence_update_quality_score < 0.6 else 4,
            )
        )
        scoring_breakdown["learning_review"] = (
            f"confidence_update_latency={conf_update_latency:.2f}, learning_mode={learning_mode}, overreaction={overreaction:.2f}"
        )
        items.append(
            PerformanceFeedbackItem(
                category="recent_performance_direction",
                label=direction,
                summary=f"Recent direction: {direction}.",
                confidence=_clamp(0.45 + 0.4 * abs(recent_improvement_score - 0.5) * 2.0),
                priority=5,
            )
        )
        scoring_breakdown["recent_performance_direction"] = (
            f"previous_window_direction_baseline={'available' if prev_avg is not None else 'none'}, "
            f"recent_improvement_score={recent_improvement_score:.3f}"
        )

        # Weakness prioritization: safety/stability > blocker > adaptation > regime > recommendation.
        weakness_rank: list[tuple[str, float, str]] = []
        if stability_quality_score < 0.62 or overreaction > 0.6:
            stability_impact = max(1.0 - stability_quality_score, 0.85 if overreaction > 0.75 else 0.0)
            weakness_rank.append(("stability/reactivity", stability_impact, "Tighten caution and monitor overreaction before pursuing opportunities."))
        if blocker_events >= 1 and action == "hold":
            weakness_rank.append(("blocker pressure", _clamp(blocker_events / 4.0), "Review blocker rejection causes before adjusting setup posture."))
        if adaptation_effectiveness_score < 0.6:
            weakness_rank.append(("adaptation speed", 1.0 - adaptation_effectiveness_score, "Paper-review strong-persistence cases for delayed adaptation."))
        if regime_interpretation_score < 0.6:
            weakness_rank.append(("regime interpretation", 1.0 - regime_interpretation_score, "Track regime transition consistency against baseline windows."))
        if recommendation_quality_score < 0.6:
            weakness_rank.append(("recommendation quality", 1.0 - recommendation_quality_score, "Re-check recommendation prioritization against blocker severity."))
        weakness_rank.sort(key=lambda x: x[1], reverse=True)
        biggest_weakness = weakness_rank[0][0] if weakness_rank else "no dominant weakness detected"
        next_safe_focus = weakness_rank[0][2] if weakness_rank else "Continue cautious monitoring and paper validation."
        weakness_ranking = [f"{name}:{score:.3f}" for (name, score, _) in weakness_rank]
        if biggest_weakness != "no dominant weakness detected":
            items.append(
                PerformanceFeedbackItem(
                    category="weakness_analysis",
                    label=biggest_weakness,
                    summary=f"Main weakness identified: {biggest_weakness}.",
                    confidence=_clamp(0.55 + 0.35 * (weakness_rank[0][1] if weakness_rank else 0.0)),
                    priority=1,
                )
            )
            scoring_breakdown["weakness_analysis"] = (
                f"top_weakness={biggest_weakness}, ranking={', '.join(weakness_ranking[:3]) or 'none'}"
            )

        # explicit constraints
        notes.append("no_strategy_or_risk_auto_mutation")
        notes.append("no_trade_execution_from_feedback")
        if readiness_low:
            notes.append("low_readiness_keeps_feedback_conservative")
        if regime_state != "stable":
            notes.append("regime_uncertainty_reduces_feedback_certainty")
            notes.append("uncertainty_adjusted_feedback_confidence")

        return PerformanceFeedbackResult(
            scores=current,
            items=sorted(items, key=lambda x: x.priority),
            biggest_weakness=biggest_weakness,
            next_safe_focus=next_safe_focus,
            direction=direction,
            notes=notes,
            inputs_snapshot={
                "evidence_strength": evidence_strength,
                "overreaction": overreaction,
                "stability_score": stability_score,
                "regime_confidence": regime_conf,
                "detection_delay": detection_delay,
                "confidence_update_latency": conf_update_latency,
            },
            scoring_breakdown=scoring_breakdown,
            weakness_ranking=weakness_ranking,
        )

    def _rolling_average(self) -> dict[str, float] | None:
        if not self._history:
            return None
        # Snapshot deque first to avoid RuntimeError when concurrent requests append.
        history = list(self._history)
        if not history:
            return None
        keys = list(history[0].keys())
        out: dict[str, float] = {}
        for k in keys:
            out[k] = sum(float(x.get(k, 0.0)) for x in history) / max(1, len(history))
        return out

    @staticmethod
    def to_dict(result: PerformanceFeedbackResult) -> dict[str, Any]:
        return {
            "scores": dict(result.scores),
            "items": [asdict(x) for x in result.items],
            "biggest_weakness": result.biggest_weakness,
            "next_safe_focus": result.next_safe_focus,
            "direction": result.direction,
            "notes": list(result.notes),
            "inputs_snapshot": dict(result.inputs_snapshot),
            "scoring_breakdown": dict(result.scoring_breakdown),
            "weakness_ranking": list(result.weakness_ranking),
        }
