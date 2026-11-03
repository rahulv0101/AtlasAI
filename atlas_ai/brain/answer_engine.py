from __future__ import annotations

from typing import Any

from atlas_ai.brain.debug import build_debug_summary
from atlas_ai.brain.domain import detect_domain_candidates
from atlas_ai.brain.evidence import collect_evidence
from atlas_ai.brain.external_context import load_external_context
from atlas_ai.brain.intent import classify_intent_candidates
from atlas_ai.brain.knowledge import BrainKnowledge
from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.brain.ai_decision_core import AIDecisionCore
from atlas_ai.brain.recommendation_engine import RecommendationEngine
from atlas_ai.brain.performance_feedback import PerformanceFeedbackLoop
from atlas_ai.brain.paper_tuning import PaperTuningEngine
from atlas_ai.brain.comparison_intelligence import ComparisonIntelligence
from atlas_ai.brain.cognitive_core import build_cognitive_core
from atlas_ai.brain.planner import choose_plan
from atlas_ai.brain.relevance_guard import check_relevance
from atlas_ai.brain.self_model import build_self_model
from atlas_ai.brain.synthesis import synthesize
from atlas_ai.brain.types import AnswerResult, EvidenceItem
from atlas_ai.decision_support import build_evidence_separation


class AnswerEngine:
    def __init__(self, knowledge: BrainKnowledge | None = None) -> None:
        self.knowledge = knowledge or BrainKnowledge()
        self.recommendation_engine = RecommendationEngine()
        self.performance_feedback = PerformanceFeedbackLoop()
        self.paper_tuning = PaperTuningEngine()
        self.comparison_intelligence = ComparisonIntelligence()
        self.ai_decision_core = AIDecisionCore()

    def answer(self, question: str, atlas_context: dict[str, Any], mode: str = "standard") -> AnswerResult:
        q = " ".join(str(question or "").strip().split())
        status = dict(atlas_context.get("status", {}) or {})
        runtime = dict(atlas_context.get("runtime", {}) or {})
        events = list(atlas_context.get("events", []) or [])
        comparison = dict(atlas_context.get("comparison", {}) or {})
        learning_state = dict(atlas_context.get("learning_state", {}) or {})
        ai_decision_journal_recent = list(atlas_context.get("ai_decision_journal_recent", []) or [])
        ai_decision_reviews_recent = list(atlas_context.get("ai_decision_reviews_recent", []) or [])
        ai_strategy_preferences = dict(atlas_context.get("ai_strategy_preferences", {}) or {})
        ai_promotion_intelligence = dict(atlas_context.get("ai_promotion_intelligence", {}) or {})
        adaptation_summary = build_adaptation_summary(status, events)

        intent_candidates = classify_intent_candidates(q)
        selected_intent = intent_candidates[0].name
        domain_candidates = detect_domain_candidates(q, intent_candidates)
        selected_domain = domain_candidates[0].name
        ql = q.lower()
        trade_tokens = {"trade", "buy", "sell", "entry", "opportunity", "direction", "long", "short"}
        if selected_intent == "unknown_operator_query" and any(x in ql for x in trade_tokens):
            selected_intent = "recommendation"
            selected_domain = "recommendation"
        if selected_intent in {"self_assessment", "self_knowledge", "capabilities", "limitations"}:
            selected_domain = "self_model"

        knowledge_payload = {
            "system_overview": self.knowledge.load("system_overview"),
            "market_knowledge": self.knowledge.load("market_knowledge"),
            "strategy_knowledge": self.knowledge.load("strategy_knowledge"),
            "risk_policy": self.knowledge.load("risk_policy"),
            "readiness_rules": self.knowledge.load("readiness_rules"),
            "blockers_catalog": self.knowledge.load("blockers_catalog"),
            "experiment_history": self.knowledge.load("experiment_history"),
            "lessons_learned": self.knowledge.load("lessons_learned"),
            "operator_teaching_examples": self.knowledge.load("operator_teaching_examples"),
            "operator_teaching_reviews": self.knowledge.load("operator_teaching_reviews"),
            "worker_neurons": self.knowledge.load("worker_neurons"),
            "glossary": self.knowledge.load("glossary"),
            "adaptation_summary": adaptation_summary,
            "external_context": load_external_context(),
            "ai_decision_journal_recent": ai_decision_journal_recent,
            "ai_decision_reviews_recent": ai_decision_reviews_recent,
            "ai_strategy_preferences": ai_strategy_preferences,
            "ai_promotion_intelligence": ai_promotion_intelligence,
        }
        evidence_separation = build_evidence_separation(status, runtime)
        cognitive_core = build_cognitive_core(
            status=status,
            runtime=runtime,
            events=events,
            learning_state=learning_state,
            knowledge_payload=knowledge_payload,
            adaptation_summary=adaptation_summary,
            evidence_separation=evidence_separation,
        )
        knowledge_payload["cognitive_core"] = cognitive_core
        ai_decision_trace = self.ai_decision_core.decide(
            status=status,
            runtime=runtime,
            events=events,
            learning_state=learning_state,
            cognitive_core=cognitive_core,
            strategy_preferences=ai_strategy_preferences,
            mode=mode,
        ).to_dict()
        knowledge_payload["ai_decision_trace"] = ai_decision_trace
        recommendation_result = self.recommendation_engine.build(
            query=q,
            status=status,
            runtime=runtime,
            comparison=comparison,
            adaptation_summary=adaptation_summary,
            blockers_catalog=knowledge_payload["blockers_catalog"],
            risk_policy=knowledge_payload["risk_policy"],
            readiness_rules=knowledge_payload["readiness_rules"],
            strategy_knowledge=knowledge_payload["strategy_knowledge"],
        )
        knowledge_payload["recommendation_result"] = RecommendationEngine.to_dict(recommendation_result)
        performance_feedback_result = self.performance_feedback.evaluate(
            status=status,
            runtime=runtime,
            adaptation_summary=adaptation_summary,
            recommendation_result=knowledge_payload["recommendation_result"],
            events=events,
        )
        knowledge_payload["performance_feedback"] = PerformanceFeedbackLoop.to_dict(performance_feedback_result)
        tuning_result = self.paper_tuning.evaluate(
            query=q,
            status=status,
            runtime=runtime,
            adaptation_summary=adaptation_summary,
            recommendation_result=knowledge_payload["recommendation_result"],
            performance_feedback=knowledge_payload["performance_feedback"],
            risk_policy=knowledge_payload["risk_policy"],
            strategy_knowledge=knowledge_payload["strategy_knowledge"],
            events=events,
        )
        knowledge_payload["paper_tuning"] = PaperTuningEngine.to_dict(tuning_result)
        comparison_intel_result = self.comparison_intelligence.evaluate(
            adaptation_summary=adaptation_summary,
            performance_feedback=knowledge_payload["performance_feedback"],
            recommendation_result=knowledge_payload["recommendation_result"],
            paper_tuning=knowledge_payload["paper_tuning"],
            events=events,
        )
        knowledge_payload["comparison_intelligence"] = ComparisonIntelligence.to_dict(comparison_intel_result)
        self_model = build_self_model(status, runtime, comparison)
        evidence = collect_evidence(
            query=q,
            domain=selected_domain,
            status=status,
            runtime=runtime,
            events=events,
            comparison=comparison,
            self_model=self_model,
            knowledge_payload=knowledge_payload,
        )
        plan = choose_plan(selected_intent, selected_domain)
        text = synthesize(
            plan=plan,
            intent=selected_intent,
            query=q,
            mode=mode,
            status=status,
            runtime=runtime,
            comparison=comparison,
            self_model=self_model,
            evidence=evidence,
            knowledge_payload=knowledge_payload,
        )
        relevance = check_relevance(query=q, domain=selected_domain, answer=text)
        if not relevance.passed:
            narrowed: list[EvidenceItem] = [e for e in evidence if e.domain in {selected_domain, "self_model", "market_knowledge", "strategy_knowledge"}][:5]
            fallback_plan = choose_plan("unknown_operator_query", selected_domain)
            text = synthesize(
                plan=fallback_plan,
                intent="unknown_operator_query",
                query=q,
                mode=mode,
                status=status,
                runtime=runtime,
                comparison=comparison,
                self_model=self_model,
                evidence=narrowed,
                knowledge_payload=knowledge_payload,
            )
            relevance = check_relevance(query=q, domain=selected_domain, answer=text)

        debug = build_debug_summary(
            intent_candidates=intent_candidates,
            domain_candidates=domain_candidates,
            selected_plan=plan.name,
            evidence_used=evidence,
            relevance_result=relevance.reason,
            recommendation_debug=knowledge_payload.get("recommendation_result", {}),
            performance_feedback_debug=knowledge_payload.get("performance_feedback", {}),
            paper_tuning_debug=knowledge_payload.get("paper_tuning", {}),
            comparison_debug=knowledge_payload.get("comparison_intelligence", {}),
        )
        trace = dict(debug.get("answer_trace", {}) or {})
        trace["normalized_question"] = q
        trace["selected_intent"] = selected_intent
        trace["selected_domain"] = selected_domain
        trace["selected_plan"] = plan.name
        trace["relevance_result"] = relevance.reason
        trace["evidence_separation"] = evidence_separation
        trace["cognitive_core"] = {
            "mode": cognitive_core.get("mode"),
            "world_model": cognitive_core.get("world_model", {}),
            "decision_understanding": cognitive_core.get("decision_understanding", {}),
            "learning_state": cognitive_core.get("learning_state", {}),
            "safety_contract": cognitive_core.get("safety_contract", {}),
        }
        trace["ai_decision_trace"] = {
            "schema": ai_decision_trace.get("schema"),
            "market_read": ai_decision_trace.get("market_read", {}),
            "recommendation": ai_decision_trace.get("recommendation", {}),
            "blocker_analysis": ai_decision_trace.get("blocker_analysis", {}),
            "strategy_invention": ai_decision_trace.get("strategy_invention", {}),
            "safety": ai_decision_trace.get("safety", {}),
        }
        debug["answer_trace"] = trace
        debug["evidence_separation"] = dict(trace["evidence_separation"])
        debug["cognitive_core"] = dict(trace["cognitive_core"])
        debug["ai_decision_trace"] = dict(trace["ai_decision_trace"])
        conf = max(0.2, min(0.99, (intent_candidates[0].confidence + domain_candidates[0].confidence) / 2))
        return AnswerResult(
            text=text,
            selected_intent=selected_intent,
            candidate_intents=intent_candidates,
            selected_domain=selected_domain,
            selected_plan=plan.name,
            confidence=conf,
            evidence_used=evidence[:8],
            relevance_passed=relevance.passed,
            debug_summary=debug,
        )
