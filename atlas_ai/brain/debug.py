from __future__ import annotations

from atlas_ai.brain.types import DomainCandidate, EvidenceItem, IntentCandidate


def build_debug_summary(
    *,
    intent_candidates: list[IntentCandidate],
    domain_candidates: list[DomainCandidate],
    selected_plan: str,
    evidence_used: list[EvidenceItem],
    relevance_result: str,
    recommendation_debug: dict | None = None,
    performance_feedback_debug: dict | None = None,
    paper_tuning_debug: dict | None = None,
    comparison_debug: dict | None = None,
) -> dict:
    knowledge_hits = []
    for e in evidence_used[:8]:
        if e.source.startswith("knowledge."):
            knowledge_hits.append({"source": e.source, "domain": e.domain, "relevance": round(e.relevance_score, 3)})
    selected_intent = intent_candidates[0].name if intent_candidates else "unknown_operator_query"
    selected_domain = domain_candidates[0].name if domain_candidates else "general_operator_help"
    evidence_trace = [
        {
            "source": e.source,
            "domain": e.domain,
            "kind": e.fact_vs_inference,
            "relevance": round(e.relevance_score, 3),
        }
        for e in evidence_used[:8]
    ]
    return {
        "intent_candidates": [c.__dict__ for c in intent_candidates],
        "domain_candidates": [c.__dict__ for c in domain_candidates],
        "selected_plan": selected_plan,
        "evidence_chosen": [f"{e.source}:{e.domain}:{e.fact_vs_inference}:{e.relevance_score:.2f}" for e in evidence_used[:8]],
        "knowledge_selected": knowledge_hits,
        "answer_trace": {
            "normalized_question": "",
            "selected_intent": selected_intent,
            "selected_domain": selected_domain,
            "selected_plan": selected_plan,
            "evidence": evidence_trace,
            "relevance_result": relevance_result,
        },
        "recommendation_debug": dict(recommendation_debug or {}),
        "performance_feedback_debug": dict(performance_feedback_debug or {}),
        "paper_tuning_debug": dict(paper_tuning_debug or {}),
        "comparison_debug": dict(comparison_debug or {}),
        "relevance_guard": relevance_result,
    }
