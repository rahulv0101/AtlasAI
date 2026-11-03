from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atlas_ai.operator_core.domain import detect_domain_candidates
from atlas_ai.operator_core.evidence import EvidenceItem, collect_candidate_evidence
from atlas_ai.operator_core.intent import classify_intent
from atlas_ai.operator_core.knowledge import KnowledgeBase
from atlas_ai.operator_core.planner import choose_answer_plan
from atlas_ai.operator_core.relevance_guard import relevance_guard
from atlas_ai.operator_core.self_model import build_self_model
from atlas_ai.operator_core.synthesis import synthesize_answer


@dataclass(frozen=True)
class AnswerEngineResult:
    response_text: str
    selected_intent: str
    selected_domain: str
    confidence: float
    evidence_summary: list[str]
    debug: dict[str, Any]


class AnswerEngine:
    def __init__(self, knowledge: KnowledgeBase | None = None) -> None:
        self.knowledge = knowledge or KnowledgeBase()

    def answer(
        self,
        question: str,
        atlas_context: dict[str, Any],
        mode: str = "standard",
    ) -> AnswerEngineResult:
        q = " ".join(str(question or "").strip().split())
        status = dict(atlas_context.get("status", {}) or {})
        runtime = dict(atlas_context.get("runtime", {}) or {})
        events = list(atlas_context.get("events", []) or [])
        comparison = dict(atlas_context.get("comparison", {}) or {})

        intent = classify_intent(q)
        domain_candidates = detect_domain_candidates(q)
        domain = domain_candidates[0].domain
        domain_conf = domain_candidates[0].confidence

        knowledge_payload = {
            "market_knowledge": self.knowledge.load("market_knowledge"),
            "strategy_knowledge": self.knowledge.load("strategy_knowledge"),
            "system_overview": self.knowledge.load("system_overview"),
            "risk_policy": self.knowledge.load("risk_policy"),
            "readiness_rules": self.knowledge.load("readiness_rules"),
            "glossary": self.knowledge.load("glossary"),
        }
        self_model = build_self_model(status, runtime, comparison)
        evidence = collect_candidate_evidence(
            query=q,
            domain=domain,
            status=status,
            runtime=runtime,
            events=events,
            comparison=comparison,
            self_model=self_model,
            knowledge=knowledge_payload,
        )
        plan = choose_answer_plan(intent, domain)

        response = self._generate(plan, intent, domain, q, status, runtime, events, comparison, self_model, evidence, mode)
        passed, reason = relevance_guard(query=q, domain=domain, answer=response)
        if not passed:
            narrowed = [e for e in evidence if e.domain in {domain, "self_model", "market_knowledge", "strategy_knowledge"}][:5]
            response = self._generate("unknown_but_relevant_plan", intent, domain, q, status, runtime, events, comparison, self_model, narrowed, mode)
            passed2, reason2 = relevance_guard(query=q, domain=domain, answer=response)
            reason = f"{reason}->regen:{reason2}"

        evidence_summary = [f"{e.source}:{e.domain}:{e.kind}:{e.relevance:.2f}" for e in evidence[:6]]
        debug = {
            "intent_candidates": [{"intent": intent, "confidence": 0.78}],
            "domain_candidates": [c.__dict__ for c in domain_candidates],
            "selected_plan": plan,
            "evidence_chosen": evidence_summary,
            "relevance_guard": reason,
        }
        return AnswerEngineResult(
            response_text=response,
            selected_intent=intent,
            selected_domain=domain,
            confidence=max(0.25, min(0.99, (domain_conf + 0.78) / 2.0)),
            evidence_summary=evidence_summary,
            debug=debug,
        )

    def _generate(
        self,
        plan: str,
        intent: str,
        domain: str,
        query: str,
        status: dict[str, Any],
        runtime: dict[str, Any],
        events: list[dict[str, Any]],
        comparison: dict[str, Any],
        self_model: dict[str, Any],
        evidence: list[EvidenceItem],
        mode: str,
    ) -> str:
        if plan == "self_knowledge_plan":
            return (
                f"Knowledge level: {self_model['purpose']}\n\n"
                "I understand:\n"
                + "\n".join(f"- {x}" for x in self_model["knowledge_domains"])
                + "\n\nLimitations:\n"
                + "\n".join(f"- {x}" for x in self_model["limitations"])
                + f"\n\nCurrent role:\n- {self_model['current_role']}"
            )
        if plan == "capabilities_plan":
            if intent == "limitations":
                return (
                    "Limitations:\n"
                    + "\n".join(f"- {x}" for x in self_model["limitations"])
                    + f"\n\nCurrent role:\n- {self_model['current_role']}"
                )
            return (
                "Capabilities:\n"
                + "\n".join(f"- {x}" for x in self_model["capabilities"])
                + "\n\nRead-only boundary:\n- Ask Atlas does not execute trading actions."
                + f"\n\nCurrent role:\n- {self_model['current_role']}"
            )
        if plan == "market_knowledge_plan":
            mk = self.knowledge.load("market_knowledge")
            core = mk.get("core_concepts", {}) if isinstance(mk, dict) else {}
            items = []
            for k, v in list(core.items())[:5]:
                items.append(f"- {k}: {v}")
            if not items:
                items = [
                    "- Trend: directional persistence over time.",
                    "- Volatility: magnitude and speed of price movement.",
                    "- Liquidity: ease of execution with low slippage.",
                    "- Momentum: continuation pressure in price direction.",
                    "- Risk management: exposure sizing and loss control first.",
                ]
            return "Market knowledge summary:\n" + "\n".join(items)
        if plan == "help_plan":
            return (
                "I can answer: status, blockers, readiness, confidence, comparison, next safe experiment, "
                "self-knowledge/capabilities/limitations, market and strategy knowledge.\n"
                "Try: 'What is holding you back?' or 'What do you know about volatility?'"
            )
        if plan == "next_safe_experiment_plan":
            return (
                "Next safe step: keep paper mode and run one bounded comparison review.\n"
                "Scope: inspect blocked candidates and confidence-gap reasons in the current run.\n"
                "Safety: no live execution and no threshold mutation without explicit operator approval."
            )
        if plan == "unknown_but_relevant_plan":
            likely = domain.replace("_", " ")
            top = evidence[0].content if evidence else "No strong evidence available."
            return f"Closest relevant interpretation: {likely}.\nBest grounded answer: {top}"

        return synthesize_answer(
            intent=intent,
            mode=mode,
            query=query,
            status=status,
            runtime=runtime,
            events=events,
            comparison=comparison,
        )
