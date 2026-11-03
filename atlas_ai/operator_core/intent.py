from __future__ import annotations

import re


INTENTS = {
    "status",
    "blocker",
    "summary",
    "comparison",
    "explanation",
    "readiness",
    "confidence",
    "self_knowledge",
    "capabilities",
    "limitations",
    "next_safe_experiment",
    "system_command",
    "market_knowledge",
    "strategy_knowledge",
    "help",
    "unknown_operator_query",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def classify_intent(query: str) -> str:
    q = str(query or "").strip().lower()
    t = _tokens(q)
    if not q:
        return "unknown_operator_query"

    if any(
        phrase in q
        for phrase in (
            "how much do you know",
            "what do you know",
            "how intelligent are you",
            "are you intelligent",
            "what are you capable of",
            "what can you do",
            "what are your capabilities",
            "what are your limits",
            "what are your limitations",
        )
    ):
        if any(k in q for k in ("limit", "limitation")):
            return "limitations"
        if any(k in q for k in ("capable", "capabilities", "can you do")):
            return "capabilities"
        return "self_knowledge"

    if any(phrase in q for phrase in ("start system", "close system", "atlas wake", "atlas sleep", "wake atlas", "go to sleep")):
        return "system_command"
    if any(k in t for k in ("help", "usage")) or ("answer" in t and "can" in t) or "help me understand" in q:
        return "help"
    if any(k in t for k in ("volatility", "liquidity", "trend", "momentum")) and any(k in t for k in ("understand", "know")):
        return "market_knowledge"
    if any(k in t for k in ("market", "volatility", "liquidity", "trend", "momentum")) and "know" in t:
        return "market_knowledge"
    if any(k in t for k in ("strategy", "strategies", "risk", "rules")) and "know" in t:
        return "strategy_knowledge"
    if any(k in t for k in ("blocker", "holding", "back", "stuck", "why")) and any(k in t for k in ("trade", "trading", "back", "holding", "blocked")):
        return "blocker"
    if "holding you back" in q or "why are you not taking trades" in q:
        return "blocker"
    if any(k in t for k in ("state", "status", "mode")):
        return "status"
    if any(k in t for k in ("summary", "summarize", "overview", "recap")):
        return "summary"
    if any(k in t for k in ("comparison", "compare", "changed", "difference", "tradeoff")):
        return "comparison"
    if any(k in t for k in ("ready", "readiness", "live")):
        return "readiness"
    if any(k in t for k in ("confidence", "sure")):
        return "confidence"
    if "next safe experiment" in q or ("next" in t and "experiment" in t):
        return "next_safe_experiment"
    if any(k in t for k in ("explain", "because", "reason")):
        return "explanation"
    return "unknown_operator_query"
