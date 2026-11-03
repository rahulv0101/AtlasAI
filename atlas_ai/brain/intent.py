from __future__ import annotations

import re

from atlas_ai.brain.types import IntentCandidate


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]


def classify_intent_candidates(query: str) -> list[IntentCandidate]:
    q = str(query or "").strip().lower()
    t = set(_tokens(q))
    if not q:
        return [IntentCandidate("unknown_operator_query", 0.2, notes="empty_query")]

    cands: list[IntentCandidate] = []
    add = cands.append

    if any(p in q for p in ("start system", "close system", "atlas wake", "atlas sleep")):
        add(IntentCandidate("system_command", 0.98, ["system_command"]))
    if any(p in q for p in ("how much do you know", "what do you know", "how intelligent are you", "are you intelligent")):
        add(IntentCandidate("self_knowledge", 0.96, ["self_knowledge_phrase"]))
    if any(
        p in q
        for p in (
            "do you have feelings",
            "do you feel",
            "do you have desire",
            "do you desire",
            "human like thoughts",
            "human-like thoughts",
            "are you autonomous",
            "autonomous trader",
            "bounded autonomous trader",
        )
    ):
        add(IntentCandidate("self_knowledge", 0.97, ["autonomy_feeling_self_model_phrase"]))
    if any(
        p in q
        for p in (
            "what is holding you back from becoming smarter",
            "how can you improve",
            "what limits your intelligence",
            "why are you not getting smarter",
            "what is limiting your intelligence",
            "why are you not smarter",
            "how can you become smarter",
            "how do you improve safely",
            "what is limiting your learning",
        )
    ):
        add(IntentCandidate("self_assessment", 0.97, ["abstract_self_assessment_phrase"]))
    abstract_capability_terms = {"intelligence", "smarter", "learning", "improve", "capability", "limitations", "limits"}
    abstract_probe_terms = {"why", "how", "what", "holding", "back"}
    if len(t.intersection(abstract_capability_terms)) >= 1 and len(t.intersection(abstract_probe_terms)) >= 1:
        add(IntentCandidate("self_assessment", 0.9, ["abstract_semantic_fallback"]))
    if any(p in q for p in ("what can you do", "what are your capabilities", "what are you capable of")):
        add(IntentCandidate("capabilities", 0.96, ["capabilities_phrase"]))
    if any(k in t for k in ("worker", "workers", "neuron", "neurons")) and any(k in t for k in ("brain", "listen", "coordinate", "works", "working")):
        add(IntentCandidate("worker_neurons", 0.97, ["worker_neuron_coordination_terms"]))
    if "brain" in t and any(k in t for k in ("understanding", "learning", "process", "processes", "everything", "whole", "system", "big", "giant")):
        add(IntentCandidate("cognitive_core", 0.98, ["system_wide_brain_terms"]))
    if any(p in q for p in ("own brain", "atlas brain", "one big brain", "giant brain", "one giant brain", "big brain", "brain to do all those things", "processes everything", "understanding and learning")):
        add(IntentCandidate("cognitive_core", 0.98, ["cognitive_core_phrase"]))
    if any(
        p in q
        for p in (
            "last decision",
            "latest decision",
            "previous decision",
            "decision journal",
            "why did you wait",
            "why are you waiting",
            "what was your decision",
            "what are you thinking",
        )
    ):
        add(IntentCandidate("ai_decision_review", 0.97, ["ai_decision_review_terms"]))
    if any(p in q for p in ("what are your limitations", "what are your limits")):
        add(IntentCandidate("limitations", 0.96, ["limitations_phrase"]))
    if any(k in t for k in ("state", "status", "mode")):
        add(IntentCandidate("status", 0.9, ["status_terms"]))
    if (
        "holding you back" in q
        and any(k in q for k in ("smarter", "intelligence", "improve", "learning"))
    ):
        add(IntentCandidate("self_assessment", 0.95, ["holding_back_abstract"]))
    if "holding you back" in q or "why are you not taking trades" in q or ({"why", "trade"} <= t) or ({"why", "trading"} <= t):
        add(IntentCandidate("blocker", 0.9, ["blocker_terms"]))
    if any(k in t for k in ("summary", "summarize", "recap")):
        add(IntentCandidate("summary", 0.86, ["summary_terms"]))
    if any(
        p in q
        for p in (
            "adapting",
            "adaptation speed",
            "overreacting",
            "learning from the charts",
            "how fast are you learning",
            "learning too slowly",
            "adjust confidence",
            "detect this regime change",
            "regime change",
        )
    ):
        add(IntentCandidate("adaptation_status", 0.92, ["adaptation_terms"]))
    if any(p in q for p in ("how fast are you learning", "learning mode", "fast-path", "slow-path", "adapting slowly", "adapting quickly")):
        add(IntentCandidate("learning_speed", 0.9, ["learning_speed_terms"]))
    if any(p in q for p in ("what did you learn", "what have you learned", "what lessons", "lessons to teach", "teach atlas", "what did atlas learn")):
        add(IntentCandidate("cognitive_core", 0.97, ["lesson_learning_terms"]))
    if any(p in q for p in ("how fast do you learn", "how quickly do you learn", "how fast can you learn", "how quickly can you learn")):
        add(IntentCandidate("learning_speed", 0.94, ["learning_speed_direct_phrase"]))
    if any(p in q for p in ("how strong is the current evidence", "evidence strength", "strength of evidence")):
        add(IntentCandidate("evidence_strength", 0.88, ["evidence_strength_terms"]))
    if any(p in q for p in ("stability", "stable right now", "overreacting", "overreaction risk")):
        add(IntentCandidate("stability_assessment", 0.86, ["stability_terms"]))
    if any(p in q for p in ("regime", "regime shift", "current regime")):
        add(IntentCandidate("regime_analysis", 0.9, ["regime_terms"]))
    if any(p in q for p in ("chart learning", "learning from charts", "setup quality getting better", "setup quality getting worse")):
        add(IntentCandidate("chart_learning", 0.9, ["chart_learning_terms"]))
    if any(k in t for k in ("comparison", "compare", "changed", "difference")):
        add(IntentCandidate("comparison", 0.9, ["comparison_terms"]))
    if any(
        p in q
        for p in (
            "compared to what",
            "versus earlier",
            "recent vs prior",
            "compare to baseline",
            "how does current behavior compare to baseline",
        )
    ):
        add(IntentCandidate("comparison_review", 0.93, ["comparison_review_terms"]))
    if any(p in q for p in ("trend comparison", "what changed versus earlier", "versus earlier")):
        add(IntentCandidate("trend_comparison", 0.92, ["trend_comparison_terms"]))
    if any(p in q for p in ("compared to what", "compare to baseline", "baseline")):
        add(IntentCandidate("baseline_comparison", 0.95, ["baseline_comparison_terms"]))
    if any(p in q for p in ("adaptation getting better", "adaptation trend", "adaptation compare")):
        add(IntentCandidate("adaptation_trend", 0.93, ["adaptation_trend_terms"]))
    if any(p in q for p in ("recommendation quality improving", "recommendation trend")):
        add(IntentCandidate("recommendation_trend", 0.93, ["recommendation_trend_terms"]))
    if any(p in q for p in ("blockers getting worse or better", "blocker trend")):
        add(IntentCandidate("blocker_trend", 0.93, ["blocker_trend_terms"]))
    if any(k in t for k in ("ready", "readiness", "live")):
        add(IntentCandidate("readiness", 0.9, ["readiness_terms"]))
    if "confidence" in t:
        add(IntentCandidate("confidence", 0.9, ["confidence_terms"]))
    if "next safe experiment" in q or ({"next", "experiment"} <= t):
        add(IntentCandidate("next_safe_experiment", 0.9, ["experiment_terms"]))
    if any(
        p in q
        for p in (
            "what's next",
            "whats next",
            "what next",
            "next step",
            "what should you do next",
            "what should i do next",
            "what is the next safe step",
            "what is the safest move",
            "what do you recommend in this market",
            "what should i watch right now",
            "why are you recommending caution",
            "what should be reviewed first",
            "give me a safe experiment",
        )
    ):
        add(IntentCandidate("recommendation", 0.94, ["recommendation_terms"]))
    if any(
        p in q
        for p in (
            "should i buy or sell",
            "should we buy or sell",
            "what do you think i should do",
            "what do you think we should do",
            "is this a buy",
            "should i take a trade right now",
            "should we take a trade right now",
            "is there a trade opportunity",
            "what direction do you prefer",
            "should i buy",
            "should i sell",
            "buy or sell",
            "trade opportunity",
        )
    ):
        add(IntentCandidate("recommendation", 0.95, ["conversational_trade_question"]))
    if any(p in q for p in ("next safe step", "safest move", "should you do next", "should i do next")):
        add(IntentCandidate("safe_next_step", 0.95, ["safe_next_step_terms"]))
    if q in {"what's next", "whats next", "what next"} or any(p in q for p in ("what is next", "next step")):
        add(IntentCandidate("safe_next_step", 0.96, ["short_next_step_terms"]))
    if any(p in q for p in ("what should i watch", "watch right now", "monitoring priority")):
        add(IntentCandidate("monitoring_priority", 0.96, ["monitoring_priority_terms"]))
    if any(p in q for p in ("what should be reviewed first", "reviewed first")):
        add(IntentCandidate("blocker_resolution", 0.97, ["blocker_resolution_terms"]))
    if "next safe experiment" not in q and any(p in q for p in ("give me a safe experiment", "safe experiment")):
        add(IntentCandidate("experiment_suggestion", 0.98, ["experiment_suggestion_terms"]))
    if any(
        p in q
        for p in (
            "how well are you adapting",
            "was your recent adaptation effective",
            "are your recommendations helping",
            "are you becoming too reactive",
            "are you being too cautious",
            "is your regime detection stable",
            "are you improving",
            "what is your biggest current weakness",
            "what should you improve next",
        )
    ):
        add(IntentCandidate("performance_review", 0.88, ["performance_review_terms"]))
    if any(p in q for p in ("how well are you adapting", "adaptation effective", "adaptation review")):
        add(IntentCandidate("adaptation_review", 0.96, ["adaptation_review_terms"]))
    if any(p in q for p in ("are your recommendations helping", "recommendation review")):
        add(IntentCandidate("recommendation_review", 0.96, ["recommendation_review_terms"]))
    if any(p in q for p in ("becoming too reactive", "being too cautious", "stability review")):
        add(IntentCandidate("stability_review", 0.96, ["stability_review_terms"]))
    if any(p in q for p in ("are you improving", "recent direction", "improvement")):
        add(IntentCandidate("improvement_assessment", 0.96, ["improvement_assessment_terms"]))
    if any(p in q for p in ("biggest current weakness", "what should you improve next", "weakness")):
        add(IntentCandidate("weakness_analysis", 0.97, ["weakness_analysis_terms"]))
    if any(
        p in q
        for p in (
            "what small tuning do you recommend",
            "safe paper-mode adjustment",
            "what should be tuned first",
            "are your thresholds too strict",
            "what paper experiment would you run next",
            "did the last paper tuning help",
            "did paper tuning improve things",
        )
    ):
        add(IntentCandidate("tuning_recommendation", 0.9, ["tuning_recommendation_terms"]))
    if any(p in q for p in ("paper experiment", "safe paper-mode adjustment", "run next")):
        add(IntentCandidate("paper_experiment", 0.98, ["paper_experiment_terms"]))
    if any(p in q for p in ("thresholds too strict", "threshold review")):
        add(IntentCandidate("threshold_review", 0.97, ["threshold_review_terms"]))
    if any(p in q for p in ("filter review", "filter strict", "too cautious")):
        add(IntentCandidate("filter_review", 0.95, ["filter_review_terms"]))
    if any(p in q for p in ("last paper tuning help", "tuning comparison", "did paper tuning improve things")):
        add(IntentCandidate("tuning_comparison", 0.99, ["tuning_comparison_terms"]))
    if any(k in t for k in ("market", "volatility", "liquidity", "trend", "momentum")) and any(k in t for k in ("know", "understand")):
        add(IntentCandidate("market_knowledge", 0.88, ["market_knowledge_terms"]))
    if any(k in t for k in ("strategy", "strategies")) or "what strategies do you use" in q:
        add(IntentCandidate("strategy_knowledge", 0.9, ["strategy_terms"]))
    if any(p in q for p in ("risk rules", "risk policy", "what are your risk rules")):
        add(IntentCandidate("risk_policy", 0.92, ["risk_policy_terms"]))
    market_definition_terms = (
        "what is volatility",
        "what is a trend",
        "what is trend",
        "what is liquidity",
        "what is momentum",
        "what is execution friction",
        "explain liquidity",
        "explain momentum",
        "explain execution friction",
        "explain momentum quality",
        "do you understand volatility",
    )
    if any(p in q for p in market_definition_terms):
        add(IntentCandidate("market_knowledge", 0.93, ["market_definition_terms"]))
    strategy_definition_terms = (
        "how does breakout retest work",
        "what is breakout retest",
        "explain breakout retest",
        "what is liquidity sweep reversal",
        "explain liquidity sweep",
        "what is range mean reversion",
        "explain range mean reversion",
    )
    if any(p in q for p in strategy_definition_terms):
        add(IntentCandidate("strategy_knowledge", 0.93, ["strategy_definition_terms"]))
    if any(p in q for p in ("what does hold mean", "define hold", "meaning of hold")):
        add(IntentCandidate("glossary", 0.92, ["glossary_terms"]))
    if any(p in q for p in ("what blocks you from trading", "what blocks trading")):
        add(IntentCandidate("blocker", 0.94, ["blocker_direct_terms"]))
    if any(k in t for k in ("strategy", "risk", "rules")) and any(k in t for k in ("know", "understand")):
        add(IntentCandidate("strategy_knowledge", 0.84, ["strategy_knowledge_terms"]))
    if "help" in t or "help me understand" in q:
        add(IntentCandidate("help", 0.9, ["help_terms"]))
    if any(k in t for k in ("explain", "reason", "because")):
        add(IntentCandidate("explanation", 0.72, ["explanation_terms"]))
    if any(k in q for k in ("follow-up context", "prior_intent=blocker")) and not any(c.name == "blocker" for c in cands):
        add(IntentCandidate("blocker", 0.78, ["followup_blocker_context"]))
    if any(k in q for k in ("follow-up context", "prior_intent=readiness")) and not any(c.name == "readiness" for c in cands):
        add(IntentCandidate("readiness", 0.78, ["followup_readiness_context"]))
    if any(k in q for k in ("follow-up context", "prior_intent=comparison")) and not any(c.name == "comparison" for c in cands):
        add(IntentCandidate("comparison", 0.78, ["followup_comparison_context"]))
    if any(k in q for k in ("follow-up context", "prior_intent=self_knowledge", "prior_intent=market_knowledge")) and not any(c.name in {"self_knowledge", "market_knowledge"} for c in cands):
        add(IntentCandidate("self_knowledge", 0.76, ["followup_self_context"]))
    if any(k in q for k in ("prior_intent=capabilities", "prior_intent=limitations", "prior_intent=self_assessment")) and not any(c.name in {"capabilities", "limitations", "self_assessment"} for c in cands):
        add(IntentCandidate("self_knowledge", 0.76, ["followup_self_context"]))
    if any(k in q for k in ("prior_intent=recommendation", "prior_intent=safe_next_step", "prior_intent=blocker_resolution", "prior_intent=trading_analysis")) and not any(c.name in {"recommendation", "safe_next_step", "blocker_resolution"} for c in cands):
        add(IntentCandidate("recommendation", 0.94, ["followup_recommendation_context"]))
    if any(k in q for k in ("prior_intent=strategy_knowledge", "prior_intent=risk_policy", "prior_intent=glossary")) and not any(c.name in {"strategy_knowledge", "risk_policy", "glossary"} for c in cands):
        add(IntentCandidate("strategy_knowledge", 0.74, ["followup_knowledge_context"]))

    guidance_phrases = (
        "what should i do about",
        "what should we do about",
        "how should i handle",
        "how should we handle",
        "what should we change",
        "what do you recommend",
        "what is the best approach",
        "how do we improve",
    )
    guidance_components = {"consensus", "guard", "guards", "filter", "filters", "blocker", "blockers", "signal", "signals", "strategy", "strategies"}
    if any(p in q for p in guidance_phrases) and (t.intersection(guidance_components) or "right now" in q):
        add(IntentCandidate("recommendation", 0.95, ["operator_action_guidance_terms"]))

    trade_tokens = {"trade", "buy", "sell", "entry", "direction", "opportunity", "long", "short"}
    suggestion_tokens = {"should", "think", "prefer", "take", "now", "right"}
    has_trade_shape = len(t.intersection(trade_tokens)) >= 1 and ("?" in q or len(t.intersection(suggestion_tokens)) >= 1)
    if has_trade_shape and not any(c.name in {"recommendation", "safe_next_step", "blocker_resolution", "experiment_suggestion"} for c in cands):
        add(IntentCandidate("recommendation", 0.89, ["trade_related_safe_fallback"]))

    if not cands:
        cands.append(IntentCandidate("unknown_operator_query", 0.35, notes="no_high_match"))
    cands.sort(key=lambda x: x.confidence, reverse=True)
    return cands
