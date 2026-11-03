from __future__ import annotations

import re
import time
import json
import os
import tempfile
import uuid
from pathlib import Path
from dataclasses import asdict
from typing import Any

from atlas_ai.brain.types import ControlItem


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


class ControlLayer:
    def __init__(self, *, max_items: int = 300, state_path: str = "knowledge/control_queue_state.json") -> None:
        self._items: list[ControlItem] = []
        self._next_id = 1
        self._max_items = max(50, int(max_items))
        default_path = str(state_path or "knowledge/control_queue_state.json")
        if os.getenv("PYTEST_CURRENT_TEST") and default_path == "knowledge/control_queue_state.json":
            # purpose: keep tests deterministic by isolating control queue persistence
            # from any real runtime artifacts in knowledge/control_queue_state.json.
            default_path = str(Path(tempfile.gettempdir()) / f"atlas_control_queue_state_{uuid.uuid4().hex}.json")
        self._state_path = default_path
        self._load_state()

    def ingest_from_debug(self, debug: dict[str, Any]) -> list[ControlItem]:
        created: list[ControlItem] = []

        rec = dict(debug.get("recommendation_debug", {}) or {})
        for item in list(rec.get("items", []) or []):
            created_item = self._create_if_new(
                item_type="recommendation",
                summary=str(item.get("recommendation", "Recommendation item")),
                detailed_reason=str(item.get("reason", "Recommendation produced by safety-first policy.")),
                supporting=list(item.get("supporting_evidence", []) or []),
                constraints=[str(item.get("constraint", "No automatic execution."))],
                confidence=float(item.get("confidence", 0.6) or 0.6),
                priority=int(item.get("priority", 5) or 5),
                related_domain="runtime_status",
                source_module="recommendation_engine",
                priority_reason="safety-first recommendation policy",
                trigger_conditions=[
                    f"category={item.get('category', 'unknown')}",
                    f"safety_level={item.get('safety_level', 'unknown')}",
                ],
                approval_level="review_required",
                approval_reason="Operator review is required before acting on recommendation guidance.",
            )
            if created_item is not None:
                created.append(created_item)
            if float(item.get("confidence", 0.0) or 0.0) >= 0.85:
                escalation_item = self._create_if_new(
                    item_type="recommendation_confidence_escalation",
                    summary=f"confidence escalation review: {str(item.get('recommendation', 'recommendation'))[:72]}",
                    detailed_reason="Recommendation confidence is high enough to justify explicit operator escalation review.",
                    supporting=list(item.get("supporting_evidence", []) or []),
                    constraints=["analysis only", "no trade execution", "no live policy mutation"],
                    confidence=float(item.get("confidence", 0.85) or 0.85),
                    priority=max(1, int(item.get("priority", 3) or 3) - 1),
                    related_domain="runtime_status",
                    source_module="recommendation_engine",
                    priority_reason="high-confidence recommendation escalation requires explicit approval",
                    trigger_conditions=[
                        f"category={item.get('category', 'unknown')}",
                        f"confidence={float(item.get('confidence', 0.0) or 0.0):.3f}",
                    ],
                    approval_level="approval_required",
                    approval_reason="Escalating confidence interpretation can impact operator posture and must be explicitly approved.",
                )
                if escalation_item is not None:
                    created.append(escalation_item)

        tuning = dict(debug.get("paper_tuning_debug", {}) or {})
        for cand in list(tuning.get("candidates", []) or []):
            created_item = self._create_if_new(
                item_type="tuning_candidate",
                summary=f"{cand.get('category', 'tuning_candidate')}: {cand.get('target', 'unknown_target')} ({float(cand.get('proposed_delta', 0.0)):+.3f})",
                detailed_reason=str(cand.get("rationale", "Paper-mode tuning candidate.")),
                supporting=[str(cand.get("expected_effect", "No expected effect provided."))],
                constraints=[str(x) for x in list(cand.get("safety_constraints", []) or [])],
                confidence=float(cand.get("confidence", 0.55) or 0.55),
                priority=int(cand.get("priority", 5) or 5),
                related_domain="experiment_recommendation",
                source_module="paper_tuning",
                priority_reason="bounded paper-mode tuning candidate",
                trigger_conditions=[
                    f"target={cand.get('target', 'unknown')}",
                    f"safe_to_paper_test={cand.get('safe_to_paper_test', False)}",
                ],
                approval_level="approval_required",
                approval_reason="Paper tuning candidates can influence future behavior and require explicit operator approval.",
            )
            if created_item is not None:
                created.append(created_item)

        for cmp_item in list(tuning.get("comparisons", []) or []):
            created_item = self._create_if_new(
                item_type="paper_experiment",
                summary=f"paper comparison: {cmp_item.get('candidate_target', 'unknown')} -> {cmp_item.get('outcome', 'inconclusive')}",
                detailed_reason=str(cmp_item.get("recommendation", "Paper comparison result.")),
                supporting=[
                    str(cmp_item.get("baseline_summary", "baseline: n/a")),
                    str(cmp_item.get("tuned_summary", "tuned: n/a")),
                ],
                constraints=["paper-mode only", "no automatic promotion to live"],
                confidence=0.6,
                priority=4,
                related_domain="experiment_recommendation",
                source_module="paper_tuning",
                priority_reason="paper comparison outcome for operator review",
                trigger_conditions=[
                    f"outcome={cmp_item.get('outcome', 'inconclusive')}",
                ],
                approval_level="review_required",
                approval_reason="Paper comparison outcomes require operator interpretation before any follow-up workflow.",
            )
            if created_item is not None:
                created.append(created_item)
            outcome = str(cmp_item.get("outcome", "inconclusive"))
            if outcome == "helpful":
                promotion_item = self._create_if_new(
                    item_type="paper_tuning_promotion_candidate",
                    summary=f"promotion candidate review: {cmp_item.get('candidate_target', 'unknown')}",
                    detailed_reason="Paper comparison looks helpful but remains paper-only until explicit governance review.",
                    supporting=[
                        str(cmp_item.get("baseline_summary", "baseline: n/a")),
                        str(cmp_item.get("tuned_summary", "tuned: n/a")),
                    ],
                    constraints=[
                        "paper validation only",
                        "no automatic promotion to live",
                        "explicit operator approval required for any further review",
                    ],
                    confidence=0.68,
                    priority=2,
                    related_domain="experiment_recommendation",
                    source_module="paper_tuning",
                    priority_reason="helpful paper result requires high-risk explicit confirmation for promotion workflow",
                    trigger_conditions=[
                        f"candidate_target={cmp_item.get('candidate_target', 'unknown')}",
                        "outcome=helpful",
                    ],
                    approval_level="high_risk_requires_explicit_confirmation",
                    approval_reason="Promotion candidates can affect future parameter governance and require explicit approval confirmation.",
                )
                if promotion_item is not None:
                    created.append(promotion_item)
            elif outcome == "inconclusive":
                validation_item = self._create_if_new(
                    item_type="extended_paper_validation",
                    summary=f"extended validation: {cmp_item.get('candidate_target', 'unknown')}",
                    detailed_reason="Comparison is inconclusive and needs extended paper-only observation before any promotion discussion.",
                    supporting=[
                        str(cmp_item.get("baseline_summary", "baseline: n/a")),
                        str(cmp_item.get("tuned_summary", "tuned: n/a")),
                    ],
                    constraints=["paper-only observation", "no live changes", "operator approval required to continue workflow"],
                    confidence=0.58,
                    priority=4,
                    related_domain="experiment_recommendation",
                    source_module="paper_tuning",
                    priority_reason="inconclusive result requires additional governed observation",
                    trigger_conditions=[
                        f"candidate_target={cmp_item.get('candidate_target', 'unknown')}",
                        "outcome=inconclusive",
                    ],
                    approval_level="approval_required",
                    approval_reason="Extended validation consumes review bandwidth and must be explicitly approved.",
                )
                if validation_item is not None:
                    created.append(validation_item)

        perf = dict(debug.get("performance_feedback_debug", {}) or {})
        for item in list(perf.get("items", []) or []):
            category = str(item.get("category", ""))
            label = str(item.get("label", ""))
            if category in {"stability_review", "adaptation_review", "recommendation_review", "learning_review"} and any(
                bad in label for bad in ("ineffective", "reactivity concern", "needs refinement", "slightly slow", "partially")
            ):
                created_item = self._create_if_new(
                    item_type="performance_warning",
                    summary=f"{category}: {label}",
                    detailed_reason=str(item.get("summary", "Performance warning flagged.")),
                    supporting=[f"priority={item.get('priority', 5)}"],
                    constraints=["analysis only", "no strategy/risk auto-change"],
                    confidence=float(item.get("confidence", 0.55) or 0.55),
                    priority=int(item.get("priority", 3) or 3),
                    related_domain="stability_assessment",
                    source_module="performance_feedback",
                    priority_reason="potential safety or stability degradation",
                    trigger_conditions=[
                        f"category={category}",
                        f"label={label}",
                    ],
                    approval_level="approval_required",
                    approval_reason="Performance degradation warnings require explicit review before any advanced workflow step.",
                )
                if created_item is not None:
                    created.append(created_item)
                if category in {"stability_review", "adaptation_review"}:
                    strategy_flag = self._create_if_new(
                        item_type="strategy_review_flag",
                        summary=f"strategy review flag: {category}",
                        detailed_reason=f"{category} indicates possible persistent weakness requiring governed operator review.",
                        supporting=[str(item.get("summary", "No summary available"))],
                        constraints=["review only", "no strategy rewrite", "no auto policy changes"],
                        confidence=float(item.get("confidence", 0.55) or 0.55),
                        priority=max(1, int(item.get("priority", 3) or 3) - 1),
                        related_domain="stability_assessment",
                        source_module="performance_feedback",
                        priority_reason="persistent safety/stability concern flagged for strategy review workflow",
                        trigger_conditions=[f"category={category}", "persistent_weakness=true"],
                        approval_level="high_risk_requires_explicit_confirmation",
                        approval_reason="Potential strategy-impacting review flags require explicit high-risk confirmation.",
                    )
                    if strategy_flag is not None:
                        created.append(strategy_flag)
            elif category == "recent_performance_direction":
                created_item = self._create_if_new(
                    item_type="improvement_suggestion",
                    summary=f"performance direction: {label}",
                    detailed_reason=str(item.get("summary", "Improvement assessment.")),
                    supporting=[str(perf.get("next_safe_focus", "continue monitoring"))],
                    constraints=["operator review required before action"],
                    confidence=float(item.get("confidence", 0.5) or 0.5),
                    priority=5,
                    related_domain="improvement_assessment",
                    source_module="performance_feedback",
                    priority_reason="recent performance direction insight",
                    trigger_conditions=[f"direction={label}"],
                    approval_level="review_required",
                    approval_reason="Improvement suggestions remain advisory and require operator review.",
                )
                if created_item is not None:
                    created.append(created_item)
                if "degrading" in label or "no clear change" in label:
                    weakness_item = self._create_if_new(
                        item_type="persistent_weakness_investigation",
                        summary=f"weakness investigation: {label}",
                        detailed_reason="Recent direction indicates possible persistent weakness that should be investigated under governance.",
                        supporting=[str(perf.get("biggest_weakness", "No explicit weakness provided"))],
                        constraints=["investigation only", "no execution", "no live mutation"],
                        confidence=float(item.get("confidence", 0.5) or 0.5),
                        priority=3,
                        related_domain="improvement_assessment",
                        source_module="performance_feedback",
                        priority_reason="repeated weakness patterns require controlled investigation",
                        trigger_conditions=[f"direction={label}"],
                        approval_level="approval_required",
                        approval_reason="Persistent weakness investigations can influence future governance decisions and require approval.",
                    )
                    if weakness_item is not None:
                        created.append(weakness_item)

        return created

    def _create_if_new(
        self,
        *,
        item_type: str,
        summary: str,
        detailed_reason: str,
        supporting: list[str],
        constraints: list[str],
        confidence: float,
        priority: int,
        related_domain: str,
        source_module: str,
        priority_reason: str,
        trigger_conditions: list[str],
        approval_level: str,
        approval_reason: str,
    ) -> ControlItem | None:
        normalized = summary.strip().lower()
        for item in reversed(self._items):
            if item.summary.strip().lower() == normalized and item.type == item_type and item.workflow_state not in {"expired"}:
                # Do not repeat blindly if previously rejected/deferred/pending.
                return None
        risk_boost = self._approval_priority_weight(str(approval_level))
        repeated_boost = self._repeat_signal_boost(trigger_conditions)
        adjusted_priority = max(1, min(9, int(priority) - risk_boost - repeated_boost))
        control_item = ControlItem(
            id=self._next_id,
            type=item_type,  # type: ignore[arg-type]
            summary=summary.strip() or f"{item_type} item",
            detailed_reason=detailed_reason.strip() or "No reason provided.",
            supporting_evidence=[str(x) for x in supporting[:6]],
            safety_constraints=[str(x) for x in constraints[:6]] or ["no automatic execution"],
            confidence=_clamp(confidence),
            priority=adjusted_priority,
            timestamp=time.time(),
            status="pending",
            workflow_state="proposed",
            approval_level=approval_level,  # type: ignore[arg-type]
            approval_reason=str(approval_reason or "").strip() or "Operator governance review is required.",
            related_domain=str(related_domain or "").strip(),
            source_module=str(source_module or "").strip(),
            priority_reason=str(priority_reason or "").strip()
            + (
                f" | risk_boost={risk_boost} repeated_signal_boost={repeated_boost}"
                if (risk_boost or repeated_boost)
                else ""
            ),
            trigger_conditions=[str(x) for x in list(trigger_conditions or [])[:6]],
        )
        self._items.append(control_item)
        self._next_id += 1
        self._prune_if_needed()
        self._save_state()
        return control_item

    def pending(self) -> list[ControlItem]:
        return sorted(
            (i for i in self._items if i.status == "pending"),
            key=lambda x: (x.priority, -self._approval_priority_weight(x.approval_level), x.id),
        )

    def create_manual_item(
        self,
        *,
        item_type: str,
        summary: str,
        detailed_reason: str,
        supporting: list[str],
        constraints: list[str],
        confidence: float,
        priority: int,
        related_domain: str,
        source_module: str,
        priority_reason: str,
        trigger_conditions: list[str],
        approval_level: str,
        approval_reason: str,
    ) -> ControlItem | None:
        return self._create_if_new(
            item_type=item_type,
            summary=summary,
            detailed_reason=detailed_reason,
            supporting=supporting,
            constraints=constraints,
            confidence=confidence,
            priority=priority,
            related_domain=related_domain,
            source_module=source_module,
            priority_reason=priority_reason,
            trigger_conditions=trigger_conditions,
            approval_level=approval_level,
            approval_reason=approval_reason,
        )

    def upsert_forced_flat_review(
        self,
        *,
        summary: str,
        detailed_reason: str,
        supporting: list[str],
        constraints: list[str],
        confidence: float,
        priority: int,
        related_domain: str,
        source_module: str,
        priority_reason: str,
        trigger_conditions: list[str],
        approval_level: str,
        approval_reason: str,
        dedupe_key: str,
    ) -> tuple[ControlItem | None, str]:
        key = str(dedupe_key or "").strip().lower()
        if not key:
            key = f"forced_flat:{str(summary or '').strip().lower()}"
        marker = f"forced_flat_key={key}"
        for idx in range(len(self._items) - 1, -1, -1):
            item = self._items[idx]
            if item.type != "forced_flat_review":
                continue
            if item.status != "pending":
                continue
            if marker not in {str(x).strip().lower() for x in list(item.trigger_conditions or [])}:
                continue
            # Update existing pending item instead of duplicating.
            updated = ControlItem(
                id=item.id,
                type=item.type,
                summary=summary.strip() or item.summary,
                detailed_reason=detailed_reason.strip() or item.detailed_reason,
                supporting_evidence=[str(x) for x in list(supporting or [])[:8]] or list(item.supporting_evidence),
                safety_constraints=[str(x) for x in list(constraints or [])[:8]] or list(item.safety_constraints),
                confidence=_clamp(confidence if confidence is not None else item.confidence),
                priority=max(1, min(9, int(priority or item.priority))),
                timestamp=time.time(),
                status=item.status,
                workflow_state=item.workflow_state,
                approval_level=approval_level,  # type: ignore[arg-type]
                approval_reason=str(approval_reason or item.approval_reason),
                approval_timestamp=item.approval_timestamp,
                approval_context=item.approval_context,
                related_domain=str(related_domain or item.related_domain),
                source_module=str(source_module or item.source_module),
                priority_reason=str(priority_reason or item.priority_reason),
                trigger_conditions=list({*(list(item.trigger_conditions or [])), *([str(x) for x in list(trigger_conditions or [])[:8]]), marker}),
            )
            self._items[idx] = updated
            self._save_state()
            return updated, "updated"

        created = self._create_if_new(
            item_type="forced_flat_review",
            summary=summary,
            detailed_reason=detailed_reason,
            supporting=supporting,
            constraints=constraints,
            confidence=confidence,
            priority=priority,
            related_domain=related_domain,
            source_module=source_module,
            priority_reason=priority_reason,
            trigger_conditions=[*list(trigger_conditions or [])[:8], marker],
            approval_level=approval_level,
            approval_reason=approval_reason,
        )
        self._save_state()
        return created, ("created" if created is not None else "dedup_skipped")

    def upsert_order_reconciliation_review(
        self,
        *,
        summary: str,
        detailed_reason: str,
        supporting: list[str],
        constraints: list[str],
        confidence: float,
        priority: int,
        related_domain: str,
        source_module: str,
        priority_reason: str,
        trigger_conditions: list[str],
        approval_level: str,
        approval_reason: str,
        dedupe_key: str,
    ) -> tuple[ControlItem | None, str]:
        key = str(dedupe_key or "").strip().lower()
        if not key:
            key = f"reconciliation:{str(summary or '').strip().lower()}"
        marker = f"reconciliation_key={key}"
        for idx in range(len(self._items) - 1, -1, -1):
            item = self._items[idx]
            if item.type != "order_reconciliation_review":
                continue
            if item.status != "pending":
                continue
            if marker not in {str(x).strip().lower() for x in list(item.trigger_conditions or [])}:
                continue
            updated = ControlItem(
                id=item.id,
                type=item.type,
                summary=summary.strip() or item.summary,
                detailed_reason=detailed_reason.strip() or item.detailed_reason,
                supporting_evidence=[str(x) for x in list(supporting or [])[:10]] or list(item.supporting_evidence),
                safety_constraints=[str(x) for x in list(constraints or [])[:10]] or list(item.safety_constraints),
                confidence=_clamp(confidence if confidence is not None else item.confidence),
                priority=max(1, min(9, int(priority or item.priority))),
                timestamp=time.time(),
                status=item.status,
                workflow_state=item.workflow_state,
                approval_level=approval_level,  # type: ignore[arg-type]
                approval_reason=str(approval_reason or item.approval_reason),
                approval_timestamp=item.approval_timestamp,
                approval_context=item.approval_context,
                related_domain=str(related_domain or item.related_domain),
                source_module=str(source_module or item.source_module),
                priority_reason=str(priority_reason or item.priority_reason),
                trigger_conditions=list({*(list(item.trigger_conditions or [])), *([str(x) for x in list(trigger_conditions or [])[:10]]), marker}),
            )
            self._items[idx] = updated
            self._save_state()
            return updated, "updated"

        created = self._create_if_new(
            item_type="order_reconciliation_review",
            summary=summary,
            detailed_reason=detailed_reason,
            supporting=supporting,
            constraints=constraints,
            confidence=confidence,
            priority=priority,
            related_domain=related_domain,
            source_module=source_module,
            priority_reason=priority_reason,
            trigger_conditions=[*list(trigger_conditions or [])[:10], marker],
            approval_level=approval_level,
            approval_reason=approval_reason,
        )
        self._save_state()
        return created, ("created" if created is not None else "dedup_skipped")

    def upsert_order_orphan_review(
        self,
        *,
        summary: str,
        detailed_reason: str,
        supporting: list[str],
        constraints: list[str],
        confidence: float,
        priority: int,
        related_domain: str,
        source_module: str,
        priority_reason: str,
        trigger_conditions: list[str],
        approval_level: str,
        approval_reason: str,
        dedupe_key: str,
    ) -> tuple[ControlItem | None, str]:
        key = str(dedupe_key or "").strip().lower()
        marker = f"orphan_key={key}"
        for idx in range(len(self._items) - 1, -1, -1):
            item = self._items[idx]
            if item.type != "order_orphan_review" or item.status != "pending":
                continue
            if marker not in {str(x).strip().lower() for x in list(item.trigger_conditions or [])}:
                continue
            updated = ControlItem(
                id=item.id,
                type=item.type,
                summary=summary.strip() or item.summary,
                detailed_reason=detailed_reason.strip() or item.detailed_reason,
                supporting_evidence=[str(x) for x in list(supporting or [])[:10]] or list(item.supporting_evidence),
                safety_constraints=[str(x) for x in list(constraints or [])[:10]] or list(item.safety_constraints),
                confidence=_clamp(confidence if confidence is not None else item.confidence),
                priority=max(1, min(9, int(priority or item.priority))),
                timestamp=time.time(),
                status=item.status,
                workflow_state=item.workflow_state,
                approval_level=approval_level,  # type: ignore[arg-type]
                approval_reason=str(approval_reason or item.approval_reason),
                approval_timestamp=item.approval_timestamp,
                approval_context=item.approval_context,
                related_domain=str(related_domain or item.related_domain),
                source_module=str(source_module or item.source_module),
                priority_reason=str(priority_reason or item.priority_reason),
                trigger_conditions=list({*(list(item.trigger_conditions or [])), *([str(x) for x in list(trigger_conditions or [])[:10]]), marker}),
            )
            self._items[idx] = updated
            self._save_state()
            return updated, "updated"

        created = self._create_if_new(
            item_type="order_orphan_review",
            summary=summary,
            detailed_reason=detailed_reason,
            supporting=supporting,
            constraints=constraints,
            confidence=confidence,
            priority=priority,
            related_domain=related_domain,
            source_module=source_module,
            priority_reason=priority_reason,
            trigger_conditions=[*list(trigger_conditions or [])[:10], marker],
            approval_level=approval_level,
            approval_reason=approval_reason,
        )
        self._save_state()
        return created, ("created" if created is not None else "dedup_skipped")

    def by_status(self, status: str, limit: int = 10) -> list[ControlItem]:
        st = str(status or "pending").lower().strip()
        return [i for i in reversed(self._items) if i.status == st][: max(1, int(limit))]

    def get(self, item_id: int) -> ControlItem | None:
        for item in self._items:
            if item.id == int(item_id):
                return item
        return None

    def set_status(self, item_id: int, status: str) -> ControlItem | None:
        target = self.get(item_id)
        if target is None:
            return None
        new_status = str(status).lower().strip()
        if new_status not in {"approved", "rejected", "deferred"}:
            return None
        if target.status != "pending":
            return None
        workflow_state = self._workflow_state_for_status(new_status)
        approval_ts = time.time() if new_status == "approved" else target.approval_timestamp
        updated = ControlItem(
            id=target.id,
            type=target.type,
            summary=target.summary,
            detailed_reason=target.detailed_reason,
            supporting_evidence=list(target.supporting_evidence),
            safety_constraints=list(target.safety_constraints),
            confidence=target.confidence,
            priority=target.priority,
            timestamp=target.timestamp,
            status=new_status,  # type: ignore[arg-type]
            workflow_state=workflow_state,  # type: ignore[arg-type]
            approval_level=target.approval_level,
            approval_reason=target.approval_reason,
            approval_timestamp=approval_ts,
            approval_context="explicit_operator_command",
            related_domain=target.related_domain,
            source_module=target.source_module,
            priority_reason=target.priority_reason,
            trigger_conditions=list(target.trigger_conditions),
        )
        self._items = [updated if i.id == target.id else i for i in self._items]
        self._save_state()
        return updated

    def set_all_pending_status(self, status: str) -> list[ControlItem]:
        new_status = str(status).lower().strip()
        if new_status not in {"approved", "rejected", "deferred"}:
            return []
        updated_items: list[ControlItem] = []
        pending_ids = [int(item.id) for item in self._items if item.status == "pending"]
        for item_id in pending_ids:
            updated = self.set_status(item_id, new_status)
            if updated is not None:
                updated_items.append(updated)
        return updated_items

    def transition_workflow(self, item_id: int, to_state: str, *, note: str = "") -> ControlItem | None:
        target = self.get(item_id)
        if target is None:
            return None
        next_state = str(to_state or "").strip().lower()
        allowed = self._allowed_workflow_transitions(target.workflow_state)
        if next_state not in allowed:
            return None
        mapped_status = self._status_for_workflow_state(next_state)
        approval_ts = target.approval_timestamp
        if next_state == "approved" and approval_ts is None:
            approval_ts = time.time()
        updated = ControlItem(
            id=target.id,
            type=target.type,
            summary=target.summary,
            detailed_reason=target.detailed_reason,
            supporting_evidence=list(target.supporting_evidence),
            safety_constraints=list(target.safety_constraints),
            confidence=target.confidence,
            priority=target.priority,
            timestamp=target.timestamp,
            status=mapped_status,  # type: ignore[arg-type]
            workflow_state=next_state,  # type: ignore[arg-type]
            approval_level=target.approval_level,
            approval_reason=target.approval_reason,
            approval_timestamp=approval_ts,
            approval_context=note.strip() or target.approval_context,
            related_domain=target.related_domain,
            source_module=target.source_module,
            priority_reason=target.priority_reason,
            trigger_conditions=list(target.trigger_conditions),
        )
        self._items = [updated if i.id == target.id else i for i in self._items]
        self._save_state()
        return updated

    @staticmethod
    def parse_command(query: str) -> dict[str, Any] | None:
        q = str(query or "").strip().lower()
        if not q:
            return None
        patterns = [
            (r"^approve item (\d+)$", "approve"),
            (r"^approve all(?: pending items)?$", "approve_all"),
            (r"^reject item (\d+)$", "reject"),
            (r"^defer item (\d+)$", "defer"),
            (r"^explain item (\d+)$", "explain"),
            (r"^why did you suggest item (\d+)$", "why"),
        ]
        for pat, action in patterns:
            m = re.match(pat, q)
            if m:
                parsed = {"action": action}
                if m.lastindex:
                    parsed["item_id"] = int(m.group(1))
                return parsed
        if q in {"show pending items", "what are the pending recommendations?", "what are pending recommendations?"}:
            return {"action": "show_pending"}
        if q in {"what did i approve recently?", "what did i approve recently"}:
            return {"action": "show_approved"}
        if q in {"what has been rejected?", "what has been rejected"}:
            return {"action": "show_rejected"}
        if q in {"which suggestions are high priority?", "which suggestions are high priority"}:
            return {"action": "show_high_priority"}
        if q in {"what needs approval right now?", "what needs approval right now"}:
            return {"action": "show_needs_approval"}
        if q in {"what is high risk?", "what is high risk"}:
            return {"action": "show_high_risk"}
        if q in {"what advanced workflows are active?", "what advanced workflows are active"}:
            return {"action": "show_active_workflows"}
        if q in {"what is under observation?", "what is under observation"}:
            return {"action": "show_under_observation"}
        if q in {"what has been validated in paper mode?", "what has been validated in paper mode"}:
            return {"action": "show_validated_paper"}
        if q in {"what should be reviewed before any promotion?", "what should be reviewed before any promotion"}:
            return {"action": "show_promotion_review"}
        return None

    def to_debug(self) -> dict[str, Any]:
        pending = self.pending()
        approved = [i for i in self._items if i.status == "approved"]
        rejected = [i for i in self._items if i.status == "rejected"]
        deferred = [i for i in self._items if i.status == "deferred"]
        under_observation = [i for i in self._items if i.workflow_state == "under_observation"]
        validated_in_paper = [i for i in self._items if i.workflow_state == "validated_in_paper"]
        expired = [i for i in self._items if i.workflow_state == "expired"]
        approval_required = [i for i in self._items if i.approval_level in {"approval_required", "high_risk_requires_explicit_confirmation"} and i.status == "pending"]
        high_risk = [i for i in self._items if i.approval_level == "high_risk_requires_explicit_confirmation" and i.status == "pending"]
        return {
            "pending_count": len(pending),
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "deferred_count": len(deferred),
            "under_observation_count": len(under_observation),
            "validated_in_paper_count": len(validated_in_paper),
            "expired_count": len(expired),
            "approval_required_count": len(approval_required),
            "high_risk_count": len(high_risk),
            "pending": [asdict(i) for i in pending[:25]],
            "approved": [asdict(i) for i in approved[-25:]],
            "rejected": [asdict(i) for i in rejected[-25:]],
            "deferred": [asdict(i) for i in deferred[-25:]],
            "under_observation": [asdict(i) for i in under_observation[-25:]],
            "validated_in_paper": [asdict(i) for i in validated_in_paper[-25:]],
            "expired": [asdict(i) for i in expired[-25:]],
            "approval_required": [asdict(i) for i in approval_required[:25]],
            "high_risk": [asdict(i) for i in high_risk[:25]],
        }

    def _prune_if_needed(self) -> None:
        if len(self._items) <= self._max_items:
            return
        # Keep pending items first, then most recent finalized items.
        pending = [x for x in self._items if x.status == "pending"]
        finalized = [x for x in self._items if x.status != "pending"]
        finalized = sorted(finalized, key=lambda x: x.timestamp, reverse=True)
        keep_finalized = max(0, self._max_items - len(pending))
        kept = pending + list(reversed(finalized[:keep_finalized]))
        kept = sorted(kept, key=lambda x: x.id)
        self._items = kept

    @staticmethod
    def item_to_dict(item: ControlItem) -> dict[str, Any]:
        return asdict(item)

    @staticmethod
    def _approval_priority_weight(level: str) -> int:
        lvl = str(level or "").strip().lower()
        if lvl == "high_risk_requires_explicit_confirmation":
            return 2
        if lvl == "approval_required":
            return 1
        return 0

    def _repeat_signal_boost(self, trigger_conditions: list[str]) -> int:
        if not trigger_conditions:
            return 0
        prior = 0
        existing_signals = {
            sig.strip().lower()
            for item in self._items
            for sig in list(item.trigger_conditions or [])
            if sig
        }
        for sig in trigger_conditions:
            norm = str(sig or "").strip().lower()
            if norm and norm in existing_signals:
                prior += 1
        return 1 if prior >= 2 else 0

    @staticmethod
    def _workflow_state_for_status(status: str) -> str:
        if status == "approved":
            return "approved"
        if status == "rejected":
            return "rejected"
        if status == "deferred":
            return "deferred"
        return "proposed"

    @staticmethod
    def _status_for_workflow_state(state: str) -> str:
        if state in {"approved", "under_observation", "validated_in_paper"}:
            return "approved"
        if state == "rejected":
            return "rejected"
        if state == "deferred":
            return "deferred"
        return "pending"

    @staticmethod
    def _allowed_workflow_transitions(state: str) -> set[str]:
        current = str(state or "").strip().lower()
        transitions: dict[str, set[str]] = {
            "proposed": {"approved", "rejected", "deferred", "expired"},
            "approved": {"under_observation", "expired"},
            "deferred": {"proposed", "rejected", "expired"},
            "under_observation": {"validated_in_paper", "deferred", "rejected", "expired"},
            "validated_in_paper": {"expired"},
            "rejected": {"expired"},
            "expired": set(),
        }
        return transitions.get(current, set())

    def _save_state(self) -> None:
        try:
            path = Path(self._state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "next_id": int(self._next_id),
                "max_items": int(self._max_items),
                "items": [asdict(i) for i in self._items],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            # Never break core control flow on persistence failure.
            return

    def _load_state(self) -> None:
        try:
            path = Path(self._state_path)
            if not path.exists():
                return
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = list(raw.get("items", []) or [])
            loaded: list[ControlItem] = []
            for row in rows:
                loaded.append(ControlItem(**row))
            self._items = loaded
            self._next_id = max(
                int(raw.get("next_id", 1) or 1),
                (max((i.id for i in self._items), default=0) + 1),
            )
            self._prune_if_needed()
        except Exception:
            # If state is corrupted, start clean rather than crash.
            self._items = []
            self._next_id = 1
