from __future__ import annotations

from dataclasses import asdict
from typing import Any

from atlas_ai.brain.types import RecommendationItem, RecommendationResult


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class RecommendationEngine:
    def build(
        self,
        *,
        query: str,
        status: dict[str, Any],
        runtime: dict[str, Any],
        comparison: dict[str, Any],
        adaptation_summary: dict[str, Any],
        blockers_catalog: dict[str, Any],
        risk_policy: dict[str, Any],
        readiness_rules: dict[str, Any],
        strategy_knowledge: dict[str, Any],
    ) -> RecommendationResult:
        q = str(query or "").lower()
        items: list[RecommendationItem] = []
        rejected: list[str] = []
        notes: list[str] = ["recommendation_engine_is_non_executing"]

        learning_mode = str(adaptation_summary.get("active_learning_mode", "slow_path"))
        evidence_strength = float(adaptation_summary.get("evidence_strength_score", 0.0) or 0.0)
        regime_conf = float(adaptation_summary.get("regime_shift_confidence", 0.0) or 0.0)
        overreaction = float(adaptation_summary.get("overreaction_score", 0.0) or 0.0)
        stability_score = float(adaptation_summary.get("stability_score", 0.0) or 0.0)
        noise = float(dict(adaptation_summary.get("evidence_components", {}) or {}).get("noise_level", 0.0) or 0.0)
        setup_decay_speed = float(adaptation_summary.get("setup_decay_detection_speed", 0.0) or 0.0)
        readiness_low = not bool(runtime.get("execute_trades", False))

        blockers = list(dict(blockers_catalog or {}).get("entries", []) or [])
        blocker_name = str((blockers[0] if blockers else {}).get("name", "No high-confidence setup"))
        blocker_severity = 0.85 if "no high-confidence" in blocker_name.lower() else 0.6
        if str(status.get("action", "")).lower() == "hold":
            blocker_severity = max(blocker_severity, 0.8)
        if readiness_low:
            blocker_severity = max(blocker_severity, 0.85)
        risk_rules = [str(x) for x in list(dict(risk_policy or {}).get("rules", []) or [])]
        readiness_criteria = [str(x) for x in list(dict(readiness_rules or {}).get("criteria", []) or [])]
        missed_summary = dict(status.get("missed_entry_learning_summary", {}) or {})
        repair_candidate = dict(missed_summary.get("top_repair_candidate", {}) or {})
        repair_blocker = str(repair_candidate.get("primary_blocker", "") or "").strip()
        repair_priority = str(repair_candidate.get("repair_priority", "") or "").strip().lower()
        repair_samples = int(repair_candidate.get("samples", 0) or 0)
        repair_missed = int(repair_candidate.get("missed_opportunity", 0) or 0)
        repair_correct = int(repair_candidate.get("correct_hold_or_block", 0) or 0)
        self_improvement_state = dict(dict(status.get("system_health", {}) or {}).get("self_improvement", {}) or {})
        approval_pending_count = int(self_improvement_state.get("approval_pending_count", 0) or 0)
        latest_proposal_id = str(
            self_improvement_state.get("latest_proposal_id", "")
            or self_improvement_state.get("pending_proposal_id", "")
            or ""
        ).strip()
        latest_proposal_target = str(
            self_improvement_state.get("latest_proposal_target_blocker", "")
            or self_improvement_state.get("pending_proposal_target_blocker", "")
            or ""
        ).strip()
        pending_posture = str(self_improvement_state.get("pending_operator_posture", "") or "").strip()
        pending_posture_reason = str(self_improvement_state.get("pending_operator_posture_reason", "") or "").strip()
        pending_suggested_action = str(self_improvement_state.get("pending_operator_suggested_action", "") or "").strip()

        if approval_pending_count > 0:
            proposal_label = latest_proposal_id or "the pending self-improvement proposal"
            items.append(
                RecommendationItem(
                    category="operator_review_gate",
                    recommendation=f"Review operator approval gate for {proposal_label}.",
                    reason=(
                        "Atlas has a self-improvement proposal waiting for explicit operator review; "
                        "this is a governance decision, not an automatic trading action."
                    ),
                    supporting_evidence=[
                        f"approval_pending_count={approval_pending_count}",
                        f"latest_proposal_id={latest_proposal_id or 'unknown'}",
                        f"target_blocker={latest_proposal_target or 'unknown'}",
                        f"rollout_active={bool(self_improvement_state.get('rollout_active', False))}",
                        f"operator_posture={pending_posture or 'unknown'}",
                        f"operator_posture_reason={pending_posture_reason or 'unknown'}",
                        f"suggested_operator_action={pending_suggested_action or 'review_packet'}",
                        f"review_packet_action=load operator review packet for {proposal_label}",
                        f"approval_action=approve {proposal_label}",
                        f"close_actions=defer or reject {proposal_label}",
                    ],
                    confidence=0.92,
                    constraint="Review, approve, defer, or reject explicitly. Do not auto-approve or change execution policy.",
                    priority=0,
                    safety_level="high",
                )
            )
            notes.append("operator_review_gate_prioritized")

        if bool(missed_summary.get("available")) and repair_blocker and repair_priority in {"high", "medium"}:
            items.append(
                RecommendationItem(
                    category="missed_entry_repair",
                    recommendation=f"Run an offline blocker-selectivity review for: {repair_blocker}",
                    reason=(
                        f"Forward validation marked this as a {repair_priority} repair candidate "
                        f"with {repair_missed} missed opportunities versus {repair_correct} correct blocks."
                    ),
                    supporting_evidence=[
                        f"samples={repair_samples}",
                        f"missed_opportunity={repair_missed}",
                        f"correct_hold_or_block={repair_correct}",
                        f"suggested_action={repair_candidate.get('suggested_action', 'review_blocker_selectivity')}",
                    ],
                    confidence=_clamp(0.62 + (0.18 if repair_priority == "high" else 0.08)),
                    constraint="Offline review only. Do not loosen guards, thresholds, risk policy, or execution policy without explicit approval.",
                    priority=1,
                    safety_level="high",
                )
            )
            notes.append("missed_entry_repair_candidate_prioritized")

        # Priority 1: risk/safety posture first.
        safety_trigger = readiness_low or evidence_strength < 0.5 or noise > 0.55 or overreaction > 0.6
        if safety_trigger:
            items.append(
                RecommendationItem(
                    category="action_posture",
                    recommendation="Hold and wait for confirmation.",
                    reason="Evidence quality is not strong enough for opportunity-seeking posture.",
                    supporting_evidence=[
                        f"evidence_strength={evidence_strength:.2f}",
                        f"noise={noise:.2f}",
                        f"overreaction_score={overreaction:.2f}",
                        f"risk_rule_count={len(risk_rules)}",
                    ],
                    confidence=_clamp(0.7 + (0.2 if safety_trigger else 0.0)),
                    constraint="No execution change. Recommendation only.",
                    priority=1,
                    safety_level="high",
                )
            )
            notes.append("safety_recommendation_prioritized")
        else:
            items.append(
                RecommendationItem(
                    category="action_posture",
                    recommendation="Use confirmation-first posture for favored setup family.",
                    reason="Regime confidence and evidence strength are supportive but still bounded by risk policy.",
                    supporting_evidence=[
                        f"regime_confidence={regime_conf:.2f}",
                        f"evidence_strength={evidence_strength:.2f}",
                        f"stability_score={stability_score:.2f}",
                    ],
                    confidence=_clamp(0.62 + 0.25 * evidence_strength),
                    constraint="Wait for explicit confirmation trigger. No auto-entry.",
                    priority=3,
                    safety_level="medium",
                )
            )

        if noise > 0.55 or overreaction > 0.6 or learning_mode == "slow_path":
            items.append(
                RecommendationItem(
                    category="caution_adjustment",
                    recommendation="Lower trust in continuation setups until signal cleanliness improves.",
                    reason="Current evidence profile is noisy or overreactive for aggressive continuation bias.",
                    supporting_evidence=[
                        f"learning_mode={learning_mode}",
                        f"noise={noise:.2f}",
                        f"overreaction_score={overreaction:.2f}",
                    ],
                    confidence=_clamp(0.62 + 0.22 * max(noise, overreaction)),
                    constraint="Risk posture only. No strategy rewrite and no execution change.",
                    priority=3,
                    safety_level="high",
                )
            )

        # Priority 2: blocker resolution outranks opportunity.
        if blocker_severity >= 0.75:
            items.append(
                RecommendationItem(
                    category="blocker_resolution",
                    recommendation=f"Review blocker first: {blocker_name}.",
                    reason="High blocker severity reduces reliability of opportunity-focused actions.",
                    supporting_evidence=[
                        f"blocker_severity={blocker_severity:.2f}",
                        f"status_action={status.get('action', 'unknown')}",
                    ],
                    confidence=_clamp(0.72 + 0.2 * blocker_severity),
                    constraint="Diagnose rejection causes first. No execution changes.",
                    priority=2,
                    safety_level="high",
                )
            )
            notes.append("blocker_resolution_overrides_aggressive_posture")

        # Priority 3/4: monitoring + comparison follow-up.
        items.append(
            RecommendationItem(
                category="monitoring_priority",
                recommendation="Monitor volatility persistence and directional cleanliness.",
                reason="These signals determine whether slow/medium/fast adaptation posture should change.",
                supporting_evidence=[
                    f"learning_mode={learning_mode}",
                    f"noise={noise:.2f}",
                    f"regime_confidence={regime_conf:.2f}",
                ],
                confidence=_clamp(0.58 + 0.2 * regime_conf),
                    constraint="Observation only; no parameter rewrites and no execution changes.",
                priority=4,
                safety_level="high",
            )
        )

        if comparison or dict(status.get("comparison", {}) or {}).get("active"):
            items.append(
                RecommendationItem(
                    category="comparison_followup",
                    recommendation="Compare current regime behavior against recent breakout-failure baseline.",
                    reason="Regime shifts are safer to interpret when tied to control-vs-recent baseline.",
                    supporting_evidence=[
                        f"comparison_active={bool(comparison or status.get('comparison'))}",
                        f"setup_decay_detection_speed={setup_decay_speed:.2f}",
                    ],
                    confidence=_clamp(0.55 + 0.2 * min(1.0, setup_decay_speed)),
                    constraint="Paper comparison only. No execution changes.",
                    priority=4,
                    safety_level="high",
                )
            )

        if "why are you recommending caution" in q:
            items.append(
                RecommendationItem(
                    category="explanation_assist",
                    recommendation="Caution is recommended because evidence quality is below safe aggression thresholds.",
                    reason="Safety and blocker policy override opportunity-seeking guidance when uncertainty is high.",
                    supporting_evidence=[
                        f"evidence_strength={evidence_strength:.2f}",
                        f"blocker_severity={blocker_severity:.2f}",
                        f"readiness_low={readiness_low}",
                    ],
                    confidence=_clamp(0.7 + 0.15 * (1.0 - evidence_strength)),
                    constraint="Explanation only. No automatic action.",
                    priority=3,
                    safety_level="high",
                )
            )

        # Priority 5: bounded experiment suggestion.
        strategy_name = str((list(dict(strategy_knowledge or {}).get("strategies", []) or [{}])[0]).get("name", "continuation"))
        items.append(
            RecommendationItem(
                category="experiment_suggestion",
                recommendation=f"Paper-test tighter {strategy_name} filters under noisy conditions.",
                reason="Bounded testing can validate whether false signals are reduced without live risk.",
                supporting_evidence=[
                    f"learning_mode={learning_mode}",
                    f"evidence_strength={evidence_strength:.2f}",
                    f"readiness_low={readiness_low}",
                    f"readiness_criteria_count={len(readiness_criteria)}",
                ],
                confidence=_clamp(0.5 + (0.2 if readiness_low else 0.1)),
                constraint="Paper mode only. No live logic or risk-policy change.",
                priority=5,
                safety_level="high",
            )
        )

        # Conservative override notes.
        if overreaction > 0.65:
            notes.append("overreaction_risk_forced_caution")
        if evidence_strength < 0.5:
            notes.append("weak_evidence_forced_slow_posture")
        if readiness_low:
            notes.append("readiness_low_forced_non_live_recommendations")

        # Query-focused trimming.
        if any(k in q for k in ("reviewed first", "review first", "what should be reviewed first")):
            items = sorted(items, key=lambda x: x.priority)
            top = [it for it in items if it.category in {"blocker_resolution", "action_posture"}]
            if top:
                items = top + [it for it in items if it not in top]
        if any(k in q for k in ("safe experiment", "experiment")):
            items = [it for it in items if it.category == "experiment_suggestion"] + [it for it in items if it.category != "experiment_suggestion"]
        if any(k in q for k in ("watch right now", "what should i watch")):
            items = [it for it in items if it.category == "monitoring_priority"] + [it for it in items if it.category != "monitoring_priority"]
        if "why are you recommending caution" in q:
            items = [it for it in items if it.category in {"explanation_assist", "caution_adjustment", "action_posture"}] + [
                it for it in items if it.category not in {"explanation_assist", "caution_adjustment", "action_posture"}
            ]

        items = sorted(items, key=lambda x: x.priority)
        # Explicitly reject execution recommendations.
        rejected.append("trade_execution_or_parameter_rewrite_recommendations_blocked_by_policy")
        return RecommendationResult(items=items, rejected=rejected, policy_notes=notes)

    @staticmethod
    def to_dict(result: RecommendationResult) -> dict[str, Any]:
        return {
            "items": [asdict(x) for x in result.items],
            "rejected": list(result.rejected),
            "policy_notes": list(result.policy_notes),
        }
