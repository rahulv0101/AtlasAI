from __future__ import annotations

from atlas_ai.brain.types import AnswerPlan


def choose_plan(intent: str, domain: str) -> AnswerPlan:
    if intent == "status":
        return AnswerPlan("status_plan", ["status", "reason", "mode", "run", "next_safe_step"])
    if intent == "blocker":
        return AnswerPlan("blocker_plan", ["main_blocker", "supporting_reasons", "impact", "next_safe_step"])
    if intent == "readiness":
        return AnswerPlan("readiness_plan", ["readiness", "reasons", "confidence_summary", "next_requirement"])
    if intent == "comparison":
        return AnswerPlan("comparison_plan", ["what_changed", "biggest_difference", "safest_interpretation", "follow_up"])
    if intent in {"comparison_review", "trend_comparison"}:
        return AnswerPlan("recent_vs_prior_plan", ["direction", "improved", "weak", "safe_interpretation"])
    if intent == "baseline_comparison":
        return AnswerPlan("baseline_comparison_plan", ["baseline", "improvement", "concern", "safe_interpretation"])
    if intent == "blocker_trend":
        return AnswerPlan("blocker_trend_plan", ["blocker_trend", "reason", "safe_interpretation"])
    if intent in {"adaptation_trend", "recommendation_trend"}:
        return AnswerPlan("adaptation_trend_plan", ["trend", "delta", "safe_interpretation"])
    if intent == "self_knowledge":
        return AnswerPlan("self_knowledge_plan", ["knowledge_level", "understands", "limitations", "current_role"])
    if intent == "self_assessment":
        return AnswerPlan("self_assessment_plan", ["current_blocker", "system_limitations", "learning_constraints", "improvement_path"])
    if intent == "market_knowledge":
        return AnswerPlan("market_knowledge_plan", ["general_knowledge", "operational_knowledge", "limits"])
    if intent == "strategy_knowledge":
        return AnswerPlan("strategy_knowledge_plan", ["strategy", "conditions", "risks"])
    if intent == "risk_policy":
        return AnswerPlan("risk_policy_plan", ["rules", "safety_boundary"])
    if intent == "glossary":
        return AnswerPlan("glossary_plan", ["term", "definition"])
    if intent in {"recommendation", "safe_next_step"}:
        return AnswerPlan("safe_next_step_plan", ["recommended_next_step", "reason", "what_to_watch"])
    if intent == "monitoring_priority":
        return AnswerPlan("market_recommendation_plan", ["recommendation", "why", "next_focus"])
    if intent == "blocker_resolution":
        return AnswerPlan("blocker_resolution_plan", ["main_recommendation", "why_this_matters", "safe_follow_up"])
    if intent == "experiment_suggestion":
        return AnswerPlan("experiment_suggestion_plan", ["safe_experiment", "constraint"])
    if intent in {"performance_review"}:
        return AnswerPlan("performance_review_plan", ["summary", "scores", "biggest_weakness", "next_safe_focus"])
    if intent == "adaptation_review":
        return AnswerPlan("adaptation_review_plan", ["adaptation_review", "what_went_well", "concern", "next_safe_focus"])
    if intent == "recommendation_review":
        return AnswerPlan("recommendation_review_plan", ["recommendation_review", "why"])
    if intent == "stability_review":
        return AnswerPlan("stability_review_plan", ["stability_review", "evidence"])
    if intent == "improvement_assessment":
        return AnswerPlan("improvement_assessment_plan", ["recent_direction", "why"])
    if intent == "weakness_analysis":
        return AnswerPlan("weakness_analysis_plan", ["main_weakness", "why_this_matters", "safe_next_step"])
    if intent in {"tuning_recommendation", "threshold_review", "filter_review"}:
        return AnswerPlan("tuning_recommendation_plan", ["recommended_tuning", "why", "constraint", "expected_effect"])
    if intent == "paper_experiment":
        return AnswerPlan("paper_experiment_plan", ["safe_paper_experiment", "success_criteria"])
    if intent == "tuning_comparison":
        return AnswerPlan("tuning_comparison_plan", ["paper_tuning_result", "observed_effect", "recommendation"])
    if intent in {"capabilities", "limitations"}:
        return AnswerPlan("capabilities_plan", ["can_do", "cannot_do", "read_only_boundary"])
    if intent == "cognitive_core" or domain == "cognitive_core":
        return AnswerPlan("cognitive_core_plan", ["world_model", "decision_understanding", "learning_memory", "decision_critic", "safety_contract"])
    if intent == "ai_decision_review" or domain == "ai_decision_trace":
        return AnswerPlan("ai_decision_review_plan", ["latest_decision", "reason", "blockers", "strategy_invention", "safety"])
    if intent == "worker_neurons" or domain == "worker_neurons":
        return AnswerPlan("worker_neurons_plan", ["coordination_model", "worker_roles", "brain_contract", "safety_boundary"])
    if intent == "help":
        return AnswerPlan("help_plan", ["supported_types", "examples"])
    if intent in {"adaptation_status", "learning_speed", "evidence_strength", "stability_assessment", "regime_analysis", "chart_learning"}:
        return AnswerPlan("adaptation_plan", ["adaptation_status", "observed_changes", "effect", "assessment", "safety_note"])
    if intent == "next_safe_experiment" or domain == "experiment_recommendation":
        return AnswerPlan("next_safe_experiment_plan", ["next_safe_step", "scope", "safety"])
    return AnswerPlan("unknown_but_relevant_plan", ["closest_interpretation", "safe_answer", "uncertainty_note"])
