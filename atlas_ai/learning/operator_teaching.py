from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from atlas_ai.memory.knowledge_store import KnowledgeStore


TEACHING_LABEL_EFFECTS: dict[str, str] = {
    "correct_hold": "reinforce_patience",
    "too_cautious": "review_missed_opportunity",
    "missed_trade": "review_entry_selectivity",
    "bad_setup": "tighten_setup_recognition",
    "good_trade": "reinforce_setup_pattern",
    "bad_trade": "review_false_positive",
    "execution_issue": "separate_idea_from_execution_quality",
    "confusing_answer": "improve_explanation_relevance",
}


REVIEW_READY_SAMPLE_COUNT = 12
RICH_CONTEXT_MINIMUM = 6


def _reviews(store: KnowledgeStore) -> list[dict[str, Any]]:
    payload = store.load_structured_knowledge_file(
        "operator_teaching_reviews.json",
        {"schema_version": 1, "reviews": []},
    )
    return [dict(row or {}) for row in list(dict(payload or {}).get("reviews", []) or [])]


def _build_teaching_review_plan(
    *,
    rows: list[dict[str, Any]],
    playbook_summary: list[dict[str, Any]],
    context_quality: Counter[str],
) -> dict[str, Any]:
    sample_count = len(rows)
    ready = sample_count >= REVIEW_READY_SAMPLE_COUNT
    rich_context_count = int(context_quality.get("decision_context_linked", 0) or 0)
    thin_context = rich_context_count < RICH_CONTEXT_MINIMUM
    dominant = dict(playbook_summary[0]) if playbook_summary else {}
    dominant_label = str(dominant.get("dominant_label", "") or "")
    dominant_playbook = str(dominant.get("playbook", "") or "")

    allowed_actions = [
        "summarize_teaching_clusters",
        "compare_labels_against_current_decision_context",
        "propose_bounded_review_items",
    ]
    blocked_actions = [
        "do_not_change_strategy_logic_automatically",
        "do_not_change_risk_policy_automatically",
        "do_not_change_execution_policy_automatically",
        "do_not_use_conversational_memory_as_trade_control",
    ]
    focus = "collect_more_operator_labels"
    next_step = f"Collect {max(0, REVIEW_READY_SAMPLE_COUNT - sample_count)} more labels before review."
    if ready:
        focus = "review_teaching_clusters"
        next_step = "Review dominant teaching clusters and convert only stable findings into proposal-only improvement items."
        if thin_context:
            focus = "review_with_context_caution"
            next_step = "Use the teaching set for review, but require more rich decision-context samples before any rollout proposal."

    if dominant_label == "correct_hold":
        hypothesis = "Atlas is often being taught that patience is correct when setups are forming or incomplete."
    elif dominant_label in {"too_cautious", "missed_trade"}:
        hypothesis = "Atlas may be missing clean opportunities and should review entry selectivity."
    elif dominant_label in {"bad_trade", "bad_setup"}:
        hypothesis = "Atlas may need tighter setup recognition before considering execution."
    elif dominant_label == "execution_issue":
        hypothesis = "Atlas should separate good trade ideas from poor execution conditions."
    else:
        hypothesis = "Teaching evidence is mixed and should be summarized before drawing a narrow lesson."

    review_items = []
    if dominant_playbook:
        review_items.append(
            {
                "scope": "dominant_playbook",
                "target": dominant_playbook,
                "label": dominant_label or "unknown",
                "question": "Does the repeated label still match the live decision context when visual, timing, and friction evidence are included?",
            }
        )
    if thin_context:
        review_items.append(
            {
                "scope": "context_quality",
                "target": "decision_context_snapshot",
                "label": "needs_more_rich_samples",
                "question": "Are there enough location, momentum, visual, volatility, and selected-signal fields to trust this lesson?",
            }
        )
    review_items.append(
        {
            "scope": "safety_boundary",
            "target": "self_improvement_queue",
            "label": "proposal_only",
            "question": "Should this lesson become a review proposal, while leaving live strategy, risk, and execution unchanged?",
        }
    )

    return {
        "status": "ready" if ready else "collecting",
        "focus": focus,
        "next_step": next_step,
        "dominant_hypothesis": hypothesis,
        "rich_context_samples": rich_context_count,
        "rich_context_minimum": RICH_CONTEXT_MINIMUM,
        "allowed_actions": allowed_actions,
        "blocked_actions": blocked_actions,
        "review_items": review_items[:5],
        "policy_note": "Review plan is advisory and proposal-only; it cannot mutate strategy, risk, or execution policy.",
    }


def build_operator_teaching_digest(store: KnowledgeStore, *, limit: int = 200) -> dict[str, Any]:
    rows = _reviews(store)[-max(1, int(limit)) :]
    label_counts = Counter(str(row.get("label", "") or "unknown") for row in rows)
    playbook_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rich_quality = Counter()
    for row in rows:
        playbook = str(row.get("playbook_name", "") or "unknown")
        label = str(row.get("label", "") or "unknown")
        playbook_counts[playbook][label] += 1
        quality = dict(row.get("context_quality", {}) or {})
        rich_quality["decision_context_linked"] += int(bool(quality.get("decision_context_linked", False)))
        rich_quality["location_known"] += int(bool(quality.get("location_known", False)))
        rich_quality["momentum_known"] += int(bool(quality.get("momentum_known", False)))
        rich_quality["volatility_known"] += int(bool(quality.get("volatility_known", False)))
        rich_quality["visual_context_known"] += int(bool(quality.get("visual_context_known", False)))
        rich_quality["selected_signal_known"] += int(bool(quality.get("selected_signal_known", False)))
        rich_quality["setup_context_known"] += int(bool(quality.get("setup_context_known", False)))

    playbook_summary = []
    for playbook, counts in sorted(playbook_counts.items(), key=lambda item: sum(item[1].values()), reverse=True):
        total = sum(counts.values())
        dominant = counts.most_common(1)[0][0] if counts else "unknown"
        playbook_summary.append(
            {
                "playbook": playbook,
                "samples": total,
                "dominant_label": dominant,
                "label_counts": dict(counts),
                "teaching_effect": TEACHING_LABEL_EFFECTS.get(dominant, "observe_more"),
            }
        )

    enough = len(rows) >= REVIEW_READY_SAMPLE_COUNT
    review_plan = _build_teaching_review_plan(
        rows=rows,
        playbook_summary=playbook_summary,
        context_quality=rich_quality,
    )
    return {
        "schema_version": 1,
        "sample_count": len(rows),
        "label_counts": dict(label_counts),
        "playbook_summary": playbook_summary[:8],
        "context_quality": dict(rich_quality),
        "training_readiness": "usable_for_review" if enough else "collect_more_operator_labels",
        "samples_needed": max(0, REVIEW_READY_SAMPLE_COUNT - len(rows)),
        "review_plan": review_plan,
        "policy_note": "Operator teaching digest is advisory. It does not mutate strategy, risk, or execution policy.",
    }


def build_current_decision_teaching_prompt(status: dict[str, Any]) -> dict[str, Any]:
    playbook = dict(status.get("setup_playbook", {}) or {})
    top = dict(playbook.get("top_candidate", {}) or {})
    readiness = str(top.get("readiness", "") or "")
    missing = list(top.get("missing", []) or [])
    risks = list(top.get("risk_notes", []) or [])
    action = str(status.get("action", "") or "")

    suggested = "correct_hold"
    reason = "Atlas is holding while the setup is not ready."
    if action in {"buy", "sell"}:
        suggested = "good_trade"
        reason = "Atlas has a directional decision; review whether the setup quality was actually clean."
    elif readiness == "ready_for_existing_gates" and not missing and not risks:
        suggested = "too_cautious"
        reason = "The playbook appears clean but Atlas is still holding."
    elif "execution_friction_elevated" in risks:
        suggested = "execution_issue"
        reason = "The idea may be recognizable, but execution conditions are the main concern."
    elif missing:
        suggested = "correct_hold"
        reason = "The closest playbook is missing required ingredients."

    return {
        "suggested_label": suggested,
        "reason": reason,
        "playbook": str(top.get("name", "") or "unknown"),
        "readiness": readiness or "unknown",
        "missing": missing[:6],
        "risks": risks[:6],
        "structured_lesson": build_structured_lesson_from_status(status),
        "policy_note": "Prompt only. Operator label remains explicit and does not alter policy.",
    }


def build_structured_lesson_from_status(status: dict[str, Any]) -> dict[str, Any]:
    status = dict(status or {})
    analysis = dict(status.get("decision_analysis", {}) or {})
    radar = dict(analysis.get("setup_radar", {}) or {})
    top = dict(radar.get("top_watch", {}) or radar.get("top_candidate", {}) or {})
    blocker = dict(status.get("blocker_classification", {}) or {})
    primary = str(blocker.get("primary") or status.get("primary_blocker") or status.get("message") or "").strip()
    structured_blocker = dict(status.get("structured_blocker", {}) or {})
    loss_learning = dict(status.get("loss_learning", {}) or {})
    visual = dict(status.get("visual_chart_read", {}) or {})
    chosen = dict(status.get("chosen", {}) or {})
    bias = dict(analysis.get("bias_breakdown", {}) or {})
    condition_parts = [
        f"action={status.get('action', 'unknown')}",
        f"candidate_side={chosen.get('side', 'unknown')}",
        f"top_playbook={top.get('name', 'unknown')}",
        f"trigger_state={top.get('trigger_state', top.get('phase', 'unknown'))}",
    ]
    if bias:
        condition_parts.append(f"htf_bias={bias.get('higher_timeframe_bias', 'unknown')}")
        condition_parts.append(f"final_trade_bias={bias.get('final_trade_bias', 'unknown')}")
    missing = [str(x) for x in list(top.get("missing_to_entry", top.get("missing", [])) or [])]
    risks = [str(x) for x in list(top.get("risks", top.get("risk_notes", [])) or [])]
    correction = "Wait for the missing setup confirmations before treating the idea as executable."
    confidence_effect = "no_confidence_boost"
    evidence_required = missing[:6] or ["clearer confluence evidence"]
    if "higher-timeframe" in primary.lower() or "htf_bias" in primary.lower():
        correction = "Do not treat M1 direction as executable while higher-timeframe bias conflicts."
        confidence_effect = "reduce_or_hold_confidence_until_alignment"
        evidence_required = ["M1 direction aligns with HTF bias", "setup trigger remains valid"]
    loss_text = " ".join(
        [
            primary.lower(),
            str(structured_blocker.get("code", "") or "").lower(),
            str(loss_learning.get("reason", "") or "").lower(),
        ]
    )
    if "loss-pattern" in loss_text or "loss-learning guard" in loss_text or "loss memory penalty" in loss_text or "loss_pattern_memory" in loss_text:
        correction = "Similar side/setup patterns need cleaner confirmation until loss memory improves."
        confidence_effect = "bounded_confidence_penalty_only"
        evidence_required = list(loss_learning.get("evidence_required", []) or [])[:6] or [
            "fresh live samples",
            "cleaner entry location",
            "improved side-specific outcome evidence",
        ]
    elif str(visual.get("visual_confirmation", "") or "") == "does_not_confirm":
        correction = "Use the visual chart read as caution: wait for a cleaner visible pattern before treating the setup as executable."
        confidence_effect = "visual_caution_only"
        evidence_required = list(visual.get("evidence_required", []) or [])[:6] or ["cleaner visual structure", "non-mid-range location"]
    elif "no high-confidence" in primary.lower():
        correction = "Wait for signal confidence and setup quality to cross existing thresholds together."
        confidence_effect = "hold_confidence_until_confluence_improves"
    return {
        "condition": " | ".join(condition_parts),
        "blocker": primary or "none",
        "mistake_to_avoid": "Do not confuse a forming idea with an executable trade.",
        "correction": correction,
        "confidence_effect": confidence_effect,
        "evidence_required": evidence_required[:6],
        "persistence_required": "repeat across more than one noisy tick/candle before learning fast",
        "policy_note": "Lesson is advisory memory; it cannot rewrite strategy, risk, or execution policy.",
    }
