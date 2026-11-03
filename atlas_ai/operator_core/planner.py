from __future__ import annotations


def choose_answer_plan(intent: str, domain: str) -> str:
    if intent in {"capabilities", "limitations"}:
        return "capabilities_plan"
    if intent == "next_safe_experiment" or domain == "experiment_recommendation":
        return "next_safe_experiment_plan"
    if intent == "status" or domain == "current_system_state":
        return "status_plan"
    if intent == "blocker" or domain == "blockers":
        return "blocker_plan"
    if intent == "readiness" or domain == "readiness_assessment":
        return "readiness_plan"
    if intent == "comparison" or domain == "comparison_results":
        return "comparison_plan"
    if intent == "self_knowledge" or domain == "self_model":
        return "self_knowledge_plan"
    if intent == "market_knowledge" or domain == "market_knowledge":
        return "market_knowledge_plan"
    if intent == "help" or domain == "general_operator_help":
        return "help_plan"
    return "unknown_but_relevant_plan"
