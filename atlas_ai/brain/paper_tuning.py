from __future__ import annotations

from collections import deque
from dataclasses import asdict
from typing import Any

from atlas_ai.brain.types import TuningCandidate, TuningComparison, TuningResult


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class PaperTuningEngine:
    def __init__(self, history_limit: int = 20) -> None:
        self._history: deque[dict[str, Any]] = deque(maxlen=history_limit)

    def evaluate(
        self,
        *,
        query: str,
        status: dict[str, Any],
        runtime: dict[str, Any],
        adaptation_summary: dict[str, Any],
        recommendation_result: dict[str, Any],
        performance_feedback: dict[str, Any],
        risk_policy: dict[str, Any],
        strategy_knowledge: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> TuningResult:
        q = str(query or "").lower()
        paper_only = not bool(runtime.get("execute_trades", False))
        evidence_strength = float(adaptation_summary.get("evidence_strength_score", 0.0) or 0.0)
        stability_score = float(adaptation_summary.get("stability_score", 0.0) or 0.0)
        overreaction = float(adaptation_summary.get("overreaction_score", 0.0) or 0.0)
        learning_mode = str(adaptation_summary.get("active_learning_mode", "slow_path"))
        blocker_name = str((list(dict(status.get("comparison", {}) or {}).keys()) and "No high-confidence setup") or "No high-confidence setup")
        filter_rejection_change = float(adaptation_summary.get("filter_rejection_rate_change", 0.0) or 0.0)
        setup_decay_speed = float(adaptation_summary.get("setup_decay_detection_speed", 0.0) or 0.0)
        feedback_scores = dict(performance_feedback.get("scores", {}) or {})
        adaptation_quality = float(feedback_scores.get("adaptation_effectiveness_score", 0.5) or 0.5)
        recommendation_quality = float(feedback_scores.get("recommendation_quality_score", 0.5) or 0.5)
        biggest_weakness = str(performance_feedback.get("biggest_weakness", ""))

        candidates: list[TuningCandidate] = []
        rejected: list[str] = []
        notes: list[str] = [
            "paper_mode_tuning_only",
            "no_live_write_path",
            "no_strategy_rewrite_or_risk_limit_mutation",
        ]
        safety_gates_triggered: list[str] = []
        opportunities_considered: list[str] = [
            "threshold_tuning_candidate",
            "persistence_tuning_candidate",
            "caution_tuning_candidate",
            "comparison_only_experiment",
        ]

        if not paper_only:
            rejected.append("tuning_rejected_non_paper_mode")
            safety_gates_triggered.append("paper_mode_required")
            return TuningResult(
                candidates=[],
                rejected=rejected,
                comparisons=[],
                top_recommendation="No tuning: live mode active.",
                notes=notes,
                safety_gates_triggered=safety_gates_triggered,
                opportunities_considered=opportunities_considered,
            )

        risk_msgs = [str(e.get("message", "")).lower() for e in events if str(e.get("type", "")).lower() == "risk"]
        borderline_rejections = sum(1 for m in risk_msgs if "blocked" in m or "filter" in m)
        noisy = float(dict(adaptation_summary.get("evidence_components", {}) or {}).get("noise_level", 0.0) or 0.0) > 0.58

        # Candidate 1: bounded confidence threshold offset.
        if borderline_rejections >= 1 and evidence_strength >= 0.55 and stability_score >= 0.58 and overreaction <= 0.62:
            candidates.append(
                TuningCandidate(
                    target="confidence_threshold_offset",
                    category="threshold_tuning_candidate",
                    current_value=0.0,
                    proposed_delta=-0.02,
                    bounded_range=(-0.03, 0.03),
                    rationale="Repeated borderline rejections under acceptable stability suggest slight threshold strictness.",
                    expected_effect="Reduce missed acceptable setups while monitoring false positives.",
                    safety_constraints=[
                        "paper-mode only",
                        "bounded delta",
                        "no live promotion",
                        "revert immediately on stability degradation",
                    ],
                    confidence=_clamp(0.55 + 0.25 * evidence_strength),
                    safe_to_paper_test=True,
                    priority=2,
                )
            )
        else:
            rejected.append("threshold_tuning_not_justified_by_stability_or_evidence")

        # Candidate 2: persistence offset for lagging adaptation.
        if "adaptation speed" in biggest_weakness.lower() or (adaptation_quality < 0.55 and evidence_strength >= 0.6 and learning_mode == "slow_path"):
            candidates.append(
                TuningCandidate(
                    target="persistence_requirement_offset",
                    category="persistence_tuning_candidate",
                    current_value=0.0,
                    proposed_delta=-0.01,
                    bounded_range=(-0.02, 0.02),
                    rationale="Adaptation lag under stronger persistence suggests a small paper-only sensitivity increase.",
                    expected_effect="Slightly faster confirmation under strong persistent evidence.",
                    safety_constraints=[
                        "paper-mode only",
                        "one-factor-at-a-time",
                        "rollback if overreaction worsens",
                        "no live parameter write",
                    ],
                    confidence=_clamp(0.5 + 0.25 * (1.0 - adaptation_quality)),
                    safe_to_paper_test=True,
                    priority=3,
                )
            )
        else:
            rejected.append("persistence_tuning_not_justified")

        # Candidate 3: caution threshold for noisy regimes.
        if noisy or overreaction > 0.62 or setup_decay_speed > 0.58:
            candidates.append(
                TuningCandidate(
                    target="noisy_regime_caution_offset",
                    category="caution_tuning_candidate",
                    current_value=0.0,
                    proposed_delta=0.02,
                    bounded_range=(-0.02, 0.03),
                    rationale="Noisy/choppy behavior requires stronger caution gating to protect stability.",
                    expected_effect="Lower reactive behavior and reduce low-quality acceptance in noise.",
                    safety_constraints=[
                        "paper-mode only",
                        "bounded caution increase",
                        "must not weaken hard risk constraints",
                        "no automatic rollout",
                    ],
                    confidence=_clamp(0.6 + 0.2 * max(overreaction, float(noisy))),
                    safe_to_paper_test=True,
                    priority=1,
                )
            )
        else:
            rejected.append("caution_tuning_not_needed")

        # Safety gate rejects aggressive tuning in noisy environments.
        if noisy and evidence_strength < 0.5:
            candidates = [c for c in candidates if c.category != "threshold_tuning_candidate"]
            rejected.append("aggressive_threshold_relaxation_rejected_under_noise")
            safety_gates_triggered.append("noise_guard_blocked_threshold_relaxation")

        # Risk-policy compatibility safety gate.
        risk_rules = [str(x).lower() for x in list(dict(risk_policy or {}).get("rules", []) or [])]
        if any("do not trade without high-confidence setup" in r for r in risk_rules):
            candidates = [c for c in candidates if c.target != "confidence_threshold_offset"]
            rejected.append("risk_policy_guard_blocked_confidence_relaxation")
            safety_gates_triggered.append("risk_policy_guard_blocked_confidence_relaxation")

        # Hard bounds enforcement gate.
        bounded_candidates: list[TuningCandidate] = []
        for c in candidates:
            lo, hi = c.bounded_range
            if c.proposed_delta < lo or c.proposed_delta > hi:
                rejected.append(f"candidate_delta_out_of_bounds:{c.target}")
                safety_gates_triggered.append(f"delta_bound_violation:{c.target}")
                continue
            bounded_candidates.append(c)
        candidates = bounded_candidates

        # Simulated bounded comparison framework.
        comparisons: list[TuningComparison] = []
        for c in candidates:
            baseline_acceptance = _clamp(0.30 + evidence_strength * 0.4 - overreaction * 0.15)
            baseline_stability = stability_score
            baseline_overreaction = overreaction

            tuned_acceptance = baseline_acceptance
            tuned_stability = baseline_stability
            tuned_overreaction = baseline_overreaction
            if c.target == "confidence_threshold_offset":
                tuned_acceptance = _clamp(baseline_acceptance + 0.06)
                tuned_stability = _clamp(baseline_stability - (0.08 if noisy else 0.02))
                tuned_overreaction = _clamp(baseline_overreaction + (0.06 if noisy else 0.02))
            elif c.target == "persistence_requirement_offset":
                tuned_acceptance = _clamp(baseline_acceptance + 0.03)
                tuned_stability = _clamp(baseline_stability - 0.01)
                tuned_overreaction = _clamp(baseline_overreaction + 0.01)
            elif c.target == "noisy_regime_caution_offset":
                tuned_acceptance = _clamp(baseline_acceptance - 0.03)
                tuned_stability = _clamp(baseline_stability + 0.06)
                tuned_overreaction = _clamp(baseline_overreaction - 0.08)

            delta_stability = tuned_stability - baseline_stability
            delta_overreaction = tuned_overreaction - baseline_overreaction
            delta_acceptance = tuned_acceptance - baseline_acceptance
            if delta_stability < -0.04 or delta_overreaction > 0.05:
                outcome = "hurts_safety"
                recommendation = "Do not promote; keep as cautionary paper evidence only."
            elif delta_acceptance > 0.0 and delta_stability >= -0.02 and delta_overreaction <= 0.03:
                outcome = "helpful"
                recommendation = "Keep in paper comparison queue. No live promotion."
            else:
                outcome = "inconclusive"
                recommendation = "Do not promote yet; collect more paper evidence before any decision."

            comparisons.append(
                TuningComparison(
                    candidate_target=c.target,
                    baseline_summary=f"acceptance={baseline_acceptance:.2f}, stability={baseline_stability:.2f}, overreaction={baseline_overreaction:.2f}",
                    tuned_summary=f"acceptance={tuned_acceptance:.2f}, stability={tuned_stability:.2f}, overreaction={tuned_overreaction:.2f}",
                    delta_acceptance_rate=delta_acceptance,
                    delta_stability=delta_stability,
                    delta_overreaction=delta_overreaction,
                    outcome=outcome,
                    recommendation=recommendation,
                )
            )

        # Persist compact history (inspectable, non-controlling).
        self._history.append(
            {
                "evidence_strength": evidence_strength,
                "learning_mode": learning_mode,
                "top_candidate": candidates[0].target if candidates else "none",
                "comparison_outcome": comparisons[0].outcome if comparisons else "none",
                "stability_score": stability_score,
            }
        )

        top_recommendation = "No safe tuning candidate currently."
        if candidates:
            top = sorted(candidates, key=lambda x: x.priority)[0]
            top_recommendation = f"Paper-only candidate: {top.category} on {top.target} (delta {top.proposed_delta:+.3f})."
        if "last paper tuning help" in q and comparisons:
            top_cmp = comparisons[0]
            top_recommendation = f"Last paper tuning appears {top_cmp.outcome}: {top_cmp.recommendation}"

        notes.append("promotion_allowed_is_false_by_default")
        return TuningResult(
            candidates=sorted(candidates, key=lambda x: x.priority),
            rejected=rejected,
            comparisons=comparisons,
            top_recommendation=top_recommendation,
            notes=notes,
            safety_gates_triggered=safety_gates_triggered,
            opportunities_considered=opportunities_considered,
        )

    @staticmethod
    def to_dict(result: TuningResult) -> dict[str, Any]:
        return {
            "candidates": [asdict(x) for x in result.candidates],
            "rejected": list(result.rejected),
            "comparisons": [asdict(x) for x in result.comparisons],
            "top_recommendation": result.top_recommendation,
            "notes": list(result.notes),
            "safety_gates_triggered": list(result.safety_gates_triggered),
            "opportunities_considered": list(result.opportunities_considered),
        }
