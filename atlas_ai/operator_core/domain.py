from __future__ import annotations

import re
from dataclasses import dataclass


DOMAINS = {
    "current_system_state",
    "blockers",
    "runtime_status",
    "comparison_results",
    "readiness_assessment",
    "self_model",
    "market_knowledge",
    "strategy_knowledge",
    "experiment_recommendation",
    "command_execution",
    "general_operator_help",
}


@dataclass(frozen=True)
class DomainCandidate:
    domain: str
    confidence: float
    reason: str


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if t]


def detect_domain_candidates(query: str) -> list[DomainCandidate]:
    q = str(query or "").strip().lower()
    t = set(_tokens(q))
    if not q:
        return [DomainCandidate("general_operator_help", 0.2, "empty_query")]

    out: list[DomainCandidate] = []
    add = out.append

    if any(k in q for k in ("start system", "close system", "atlas wake", "atlas sleep")):
        add(DomainCandidate("command_execution", 0.98, "explicit_system_command"))
    if any(k in t for k in ("status", "state", "mode")):
        add(DomainCandidate("current_system_state", 0.92, "status_terms"))
    if any(k in t for k in ("blocker", "holding", "stuck")) or "why are you not taking trades" in q:
        add(DomainCandidate("blockers", 0.93, "blocker_terms"))
    if any(k in t for k in ("comparison", "changed", "difference", "tradeoff")):
        add(DomainCandidate("comparison_results", 0.92, "comparison_terms"))
    if any(k in t for k in ("ready", "readiness", "live")):
        add(DomainCandidate("readiness_assessment", 0.9, "readiness_terms"))
    if any(
        phrase in q
        for phrase in (
            "how much do you know",
            "what do you know",
            "what can you do",
            "what are your capabilities",
            "what are your limitations",
            "how intelligent are you",
        )
    ):
        add(DomainCandidate("self_model", 0.96, "self_model_query"))
    if any(k in t for k in ("market", "volatility", "liquidity", "trend", "support", "resistance", "momentum")):
        add(DomainCandidate("market_knowledge", 0.9, "market_knowledge_terms"))
    if any(k in t for k in ("strategy", "strategies", "risk", "rules")):
        add(DomainCandidate("strategy_knowledge", 0.86, "strategy_knowledge_terms"))
    if "next safe experiment" in q or ("next" in t and "experiment" in t):
        add(DomainCandidate("experiment_recommendation", 0.91, "safe_experiment_terms"))
    if any(k in t for k in ("help", "what can i ask", "can answer")):
        add(DomainCandidate("general_operator_help", 0.95, "help_terms"))

    if not out:
        out.append(DomainCandidate("general_operator_help", 0.35, "unknown_defaults_to_help"))
    out.sort(key=lambda x: x.confidence, reverse=True)
    return out

