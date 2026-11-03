from __future__ import annotations


def relevance_guard(*, query: str, domain: str, answer: str) -> tuple[bool, str]:
    q = str(query or "").lower()
    a = str(answer or "").lower()
    if not a.strip():
        return False, "empty_answer"

    forbidden_state_dump = any(token in a for token in ("blocked", "mode: paper", "run: atlas_"))
    self_query = any(
        phrase in q
        for phrase in (
            "how much do you know",
            "what do you know",
            "what can you do",
            "capabilities",
            "limitations",
            "intelligent",
        )
    )
    if self_query and forbidden_state_dump and not any(k in a for k in ("knowledge level", "capabilities", "limitations", "current role")):
        return False, "state_leakage_for_self_query"

    if domain == "self_model" and not any(k in a for k in ("knowledge level", "capabilities", "limitations", "current role", "i understand")):
        return False, "self_model_missing_sections"
    if domain == "market_knowledge" and not any(k in a for k in ("market", "volatility", "liquidity", "trend", "risk")):
        return False, "market_answer_not_market_focused"
    return True, "ok"

