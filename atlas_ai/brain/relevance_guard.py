from __future__ import annotations

from atlas_ai.brain.types import RelevanceCheckResult


def check_relevance(*, query: str, domain: str, answer: str) -> RelevanceCheckResult:
    q = str(query or "").lower()
    a = str(answer or "").lower()
    if not a.strip():
        return RelevanceCheckResult(False, "empty_answer")
    self_query = any(
        p in q
        for p in (
            "how much do you know",
            "what do you know",
            "what can you do",
            "capabilities",
            "capable of",
            "limitations",
            "limits",
            "intelligent",
            "how smart",
            "understand markets",
            "understand the market",
        )
    )
    self_shape = any(k in a for k in ("knowledge level", "capabilities", "limitations", "current role", "i understand:"))
    state_dump_markers = (
        "status:",
        "status: hold",
        "status: blocked",
        "hold |",
        "blocked |",
        "mode: paper",
        "mode: live",
        "run: atlas_",
        "current action:",
        "trade state:",
    )
    state_dump = any(p in a for p in state_dump_markers)
    if self_query and state_dump and not self_shape:
        return RelevanceCheckResult(False, "irrelevant_state_leakage")
    if self_query and any(p in a for p in ("no high-confidence trade", "blocked trade", "conservative pre-trade filter")) and not self_shape:
        return RelevanceCheckResult(False, "irrelevant_trade_state_answer")
    if domain == "self_model" and not any(k in a for k in ("knowledge level", "capabilities", "limitations", "current role")):
        return RelevanceCheckResult(False, "missing_self_model_sections")
    knowledge_query = any(
        p in q
        for p in (
            "what is volatility",
            "what is a trend",
            "what is trend",
            "what is liquidity",
            "what is momentum",
            "explain liquidity",
            "explain momentum",
            "explain momentum quality",
            "breakout retest",
            "liquidity sweep",
            "mean reversion",
            "what strategies do you use",
            "risk rules",
            "risk policy",
            "what does hold mean",
        )
    )
    if knowledge_query and any(p in a for p in ("status:", "mode:", "run:", "blocked with reason", "no high-confidence trade")):
        return RelevanceCheckResult(False, "knowledge_query_state_leakage")
    if domain in {"market_knowledge", "strategy_knowledge", "risk_policy", "glossary"} and any(p in a for p in ("status:", "mode:", "run:")):
        return RelevanceCheckResult(False, "knowledge_domain_state_leakage")
    recommendation_query = any(
        p in q
        for p in (
            "what should you do next",
            "what should i do next",
            "what do you recommend",
            "safest move",
            "what should i watch",
            "reviewed first",
            "safe experiment",
        )
    )
    if recommendation_query and not any(
        p in a
        for p in (
            "recommended next step",
            "recommendation:",
            "main recommendation:",
            "safe experiment:",
            "next safe step:",
        )
    ):
        return RelevanceCheckResult(False, "missing_recommendation_shape")
    performance_query = any(
        p in q
        for p in (
            "how well are you adapting",
            "recommendations helping",
            "too reactive",
            "are you improving",
            "biggest current weakness",
            "improve next",
        )
    )
    if performance_query and not any(
        p in a
        for p in (
            "adaptation review:",
            "recommendation review:",
            "stability review:",
            "recent direction:",
            "main weakness:",
            "performance review:",
        )
    ):
        return RelevanceCheckResult(False, "missing_performance_review_shape")
    tuning_query = any(
        p in q
        for p in (
            "small tuning",
            "paper-mode adjustment",
            "thresholds too strict",
            "paper experiment",
            "last paper tuning",
            "tuning comparison",
        )
    )
    if tuning_query and not any(
        p in a
        for p in (
            "recommended paper-mode tuning:",
            "safe paper experiment:",
            "paper tuning result:",
        )
    ):
        return RelevanceCheckResult(False, "missing_tuning_shape")
    return RelevanceCheckResult(True, "ok")
