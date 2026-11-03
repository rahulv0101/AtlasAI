from __future__ import annotations

from atlas_ai.brain.types import DomainCandidate, IntentCandidate


def detect_domain_candidates(query: str, intents: list[IntentCandidate]) -> list[DomainCandidate]:
    q = str(query or "").lower()
    top_intent = intents[0].name if intents else "unknown_operator_query"
    out: list[DomainCandidate] = []
    add = out.append

    mapping = {
        "status": "current_system_state",
        "blocker": "blockers",
        "summary": "runtime_status",
        "comparison": "comparison_results",
        "comparison_review": "comparison_results",
        "trend_comparison": "comparison_results",
        "baseline_comparison": "comparison_results",
        "blocker_trend": "comparison_results",
        "adaptation_trend": "comparison_results",
        "recommendation_trend": "comparison_results",
        "explanation": "current_system_state",
        "readiness": "readiness_assessment",
        "confidence": "runtime_status",
        "next_safe_experiment": "experiment_recommendation",
        "self_knowledge": "self_model",
        "self_assessment": "self_model",
        "capabilities": "self_model",
        "cognitive_core": "cognitive_core",
        "worker_neurons": "worker_neurons",
        "ai_decision_review": "ai_decision_trace",
        "limitations": "self_model",
        "market_knowledge": "market_knowledge",
        "strategy_knowledge": "strategy_knowledge",
        "risk_policy": "risk_policy",
        "glossary": "glossary",
        "help": "general_operator_help",
        "adaptation_status": "runtime_status",
        "learning_speed": "runtime_status",
        "evidence_strength": "runtime_status",
        "stability_assessment": "runtime_status",
        "recommendation": "recommendation",
        "safe_next_step": "recommendation",
        "monitoring_priority": "recommendation",
        "blocker_resolution": "blockers",
        "experiment_suggestion": "experiment_recommendation",
        "performance_review": "performance_feedback",
        "adaptation_review": "performance_feedback",
        "recommendation_review": "performance_feedback",
        "stability_review": "performance_feedback",
        "improvement_assessment": "performance_feedback",
        "weakness_analysis": "performance_feedback",
        "tuning_recommendation": "paper_tuning",
        "paper_experiment": "paper_tuning",
        "threshold_review": "paper_tuning",
        "filter_review": "paper_tuning",
        "tuning_comparison": "paper_tuning",
        "regime_analysis": "comparison_results",
        "chart_learning": "runtime_status",
        "system_command": "command_execution",
        "unknown_operator_query": "general_operator_help",
    }
    add(DomainCandidate(mapping.get(top_intent, "general_operator_help"), 0.9, f"from_intent:{top_intent}"))

    if any(p in q for p in ("volatility", "liquidity", "trend", "momentum", "market", "spread", "session quality", "execution friction")):
        add(DomainCandidate("market_knowledge", 0.84, "market_terms"))
    if any(p in q for p in ("strategy", "strategies", "breakout retest", "liquidity sweep", "mean reversion", "continuation")):
        add(DomainCandidate("strategy_knowledge", 0.86, "strategy_terms"))
    if any(p in q for p in ("own brain", "atlas brain", "one big brain", "giant brain", "one giant brain", "big brain", "processes everything", "understanding and learning", "brain to do all those things")):
        add(DomainCandidate("cognitive_core", 0.98, "cognitive_core_terms"))
    if any(p in q for p in ("last decision", "latest decision", "decision journal", "why did you wait", "why are you waiting", "what was your decision")):
        add(DomainCandidate("ai_decision_trace", 0.97, "ai_decision_review_terms"))
    if any(
        p in q
        for p in (
            "why isn't it trading",
            "why is it not trading",
            "why isnt it trading",
            "why didn't it trade",
            "why didnt it trade",
            "why didn't you trade",
            "why didnt you trade",
            "why no trade",
            "why not trading",
            "why isn't atlas trading",
        )
    ):
        add(DomainCandidate("blockers", 0.99, "why_trace_blocker_terms"))
    if any(p in q for p in ("what did you learn", "what have you learned", "what lessons", "lessons to teach", "teach atlas", "what did atlas learn")):
        add(DomainCandidate("cognitive_core", 0.97, "lesson_learning_terms"))
    if any(p in q for p in ("worker", "workers", "neuron", "neurons", "brain")):
        add(DomainCandidate("worker_neurons", 0.95, "worker_neuron_terms"))
    if any(p in q for p in ("risk rules", "risk policy")):
        add(DomainCandidate("risk_policy", 0.9, "risk_policy_terms"))
    if any(p in q for p in ("what does hold mean", "define hold", "meaning of hold")):
        add(DomainCandidate("glossary", 0.92, "glossary_terms"))
    if any(
        p in q
        for p in (
            "what should you do next",
            "what should i do next",
            "what do you recommend in this market",
            "safest move",
            "what should i watch right now",
            "why are you recommending caution",
            "what should be reviewed first",
            "safe experiment",
        )
    ):
        add(DomainCandidate("recommendation", 0.94, "recommendation_terms"))
    if any(
        p in q
        for p in (
            "how well are you adapting",
            "adaptation effective",
            "recommendations helping",
            "too reactive",
            "too cautious",
            "regime detection stable",
            "are you improving",
            "biggest current weakness",
            "improve next",
        )
    ):
        add(DomainCandidate("performance_feedback", 0.95, "performance_feedback_terms"))
    if any(
        p in q
        for p in (
            "small tuning",
            "paper-mode adjustment",
            "tuned first",
            "thresholds too strict",
            "paper experiment",
            "last paper tuning",
            "tuning comparison",
        )
    ):
        add(DomainCandidate("paper_tuning", 0.96, "paper_tuning_terms"))
    if "comparison" in q or "changed" in q:
        add(DomainCandidate("comparison_results", 0.82, "comparison_terms"))
    if any(p in q for p in ("versus earlier", "recent vs prior", "compared to what", "baseline")):
        add(DomainCandidate("comparison_results", 0.94, "comparison_review_terms"))
    if any(
        p in q
        for p in (
            "adapting",
            "overreacting",
            "learning from charts",
            "setup quality",
            "learning mode",
            "how fast do you learn",
            "how quickly do you learn",
            "fast-path",
            "slow-path",
            "evidence strength",
            "stability",
        )
    ):
        add(DomainCandidate("runtime_status", 0.86, "adaptation_terms"))
    if "regime" in q:
        add(DomainCandidate("comparison_results", 0.86, "regime_terms"))
    if "help" in q:
        add(DomainCandidate("general_operator_help", 0.86, "help_terms"))

    # Prevent system-state as universal default.
    if not out:
        out.append(DomainCandidate("general_operator_help", 0.4, "safe_default"))
    out.sort(key=lambda x: x.confidence, reverse=True)
    return out
