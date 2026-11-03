from __future__ import annotations

import re
from typing import Any

from atlas_ai.brain.types import EvidenceItem
from atlas_ai.brain.why_engine import build_why_trace


def _contains_term(text: str, term: str) -> bool:
    return str(term or "").lower() in str(text or "").lower()


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(t) >= 3}


def _item_tokens(item: dict[str, Any], fields: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for field in fields:
        value = item.get(field)
        if isinstance(value, str):
            out.update(_tokens(value))
        elif isinstance(value, list):
            out.update(_tokens(" ".join(str(x) for x in value)))
    return out


def _best_item_by_tokens(
    query: str,
    items: list[dict[str, Any]],
    *,
    name_field: str,
    fields: tuple[str, ...],
) -> tuple[dict[str, Any] | None, str]:
    q_tokens = _tokens(query)
    if not q_tokens:
        return None, "no_query_tokens"
    best: dict[str, Any] | None = None
    best_score = 0
    for item in items:
        name = str(item.get(name_field, "")).lower()
        if name and _contains_term(query, name):
            return item, "direct_name_match"
        name_tokens = _tokens(name)
        item_tokens = name_tokens | _item_tokens(item, fields)
        score = len(q_tokens.intersection(name_tokens)) * 4 + len(q_tokens.intersection(item_tokens))
        if score > best_score:
            best = item
            best_score = score
    if best is not None and best_score > 0:
        return best, "token_match"
    return None, "no_relevant_match"


def _pick_market_concept(query: str, market_knowledge: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    concepts = list(market_knowledge.get("concepts", []) or [])
    return _best_item_by_tokens(query, concepts, name_field="name", fields=("definition", "summary", "signals", "implications"))


def _pick_strategy(query: str, strategy_knowledge: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    strategies = list(strategy_knowledge.get("strategies", []) or [])
    picked, reason = _best_item_by_tokens(
        query,
        strategies,
        name_field="name",
        fields=("description", "notes", "conditions", "when_it_works", "when_to_avoid", "risks"),
    )
    if picked is not None:
        return picked, reason
    if strategies:
        return strategies[0], "generic_strategy_summary"
    return None, "no_strategy"


def _pick_glossary_term(query: str, glossary: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    terms = list(glossary.get("terms", []) or [])
    return _best_item_by_tokens(query, terms, name_field="term", fields=("definition",))


def collect_evidence(
    *,
    query: str,
    domain: str,
    status: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]],
    comparison: dict[str, Any],
    self_model: dict[str, Any],
    knowledge_payload: dict[str, dict[str, Any]],
) -> list[EvidenceItem]:
    q = str(query or "").lower()
    out: list[EvidenceItem] = []
    add = out.append

    if domain in {"current_system_state", "blockers", "runtime_status", "readiness_assessment", "comparison_results", "recommendation", "performance_feedback", "paper_tuning"}:
        if domain in {"current_system_state", "blockers", "runtime_status", "readiness_assessment"}:
            execution_snapshot = dict(runtime.get("execution", {}) or status.get("execution", {}) or {})
            why_trace = build_why_trace(status, runtime, execution_snapshot)
            add(EvidenceItem("why.execution_trace", "blockers", str(why_trace), 0.99, 0.98, "fact", 0.99))
        add(EvidenceItem("status", "current_system_state", str({"action": status.get("action"), "message": status.get("message")}), 0.95, 0.95, "fact", 0.9))
        add(EvidenceItem("runtime", "runtime_status", str({"execute_trades": runtime.get("execute_trades", False)}), 0.95, 0.95, "fact", 0.8))
        add(EvidenceItem("comparison", "comparison_results", str(comparison or status.get("comparison", {})), 0.85, 0.9, "fact", 0.82))
    if domain == "self_model":
        add(EvidenceItem("self_model", "self_model", str(self_model), 0.8, 0.9, "fact", 0.96))
    if domain == "worker_neurons":
        add(EvidenceItem("knowledge.worker_neurons:file", "worker_neurons", str(knowledge_payload.get("worker_neurons", {})), 0.92, 0.94, "fact", 0.98))
        add(EvidenceItem("status.signal_pipeline", "worker_neurons", str({"signal_summary": status.get("signal_summary", {}), "system_health": status.get("system_health", {})}), 0.82, 0.86, "fact", 0.82))
    if domain == "cognitive_core":
        add(EvidenceItem("brain.cognitive_core", "cognitive_core", str(knowledge_payload.get("cognitive_core", {})), 0.96, 0.95, "inference", 0.99))
        add(EvidenceItem("decision.evidence_separation", "cognitive_core", str(knowledge_payload.get("cognitive_core", {}).get("decision_understanding", {})), 0.92, 0.94, "fact", 0.95))
        add(EvidenceItem("knowledge.lessons_learned:file", "cognitive_core", str(knowledge_payload.get("lessons_learned", {})), 0.86, 0.9, "fact", 0.92))
    if domain == "market_knowledge":
        concept, reason = _pick_market_concept(q, dict(knowledge_payload.get("market_knowledge", {}) or {}))
        if concept:
            add(EvidenceItem(f"knowledge.market_knowledge:concept:{concept.get('name', 'unknown')}:{reason}", "market_knowledge", str(concept), 0.9, 0.92, "fact", 0.97))
        add(EvidenceItem("knowledge.market_knowledge:file", "market_knowledge", str(knowledge_payload.get("market_knowledge", {})), 0.7, 0.84, "fact", 0.95))
        add(EvidenceItem("self_model", "self_model", str(self_model), 0.8, 0.86, "fact", 0.72))
    if domain == "strategy_knowledge":
        strategy, reason = _pick_strategy(q, dict(knowledge_payload.get("strategy_knowledge", {}) or {}))
        if strategy:
            add(EvidenceItem(f"knowledge.strategy_knowledge:strategy:{strategy.get('name', 'unknown')}:{reason}", "strategy_knowledge", str(strategy), 0.9, 0.92, "fact", 0.97))
        add(EvidenceItem("knowledge.strategy_knowledge:file", "strategy_knowledge", str(knowledge_payload.get("strategy_knowledge", {})), 0.7, 0.84, "fact", 0.94))
    if domain == "risk_policy":
        add(EvidenceItem("knowledge.risk_policy:file", "risk_policy", str(knowledge_payload.get("risk_policy", {})), 0.9, 0.94, "fact", 0.97))
    if domain == "glossary":
        term, reason = _pick_glossary_term(q, dict(knowledge_payload.get("glossary", {}) or {}))
        if term:
            add(EvidenceItem(f"knowledge.glossary:term:{term.get('term', 'unknown')}:{reason}", "glossary", str(term), 0.9, 0.94, "fact", 0.98))
        add(EvidenceItem("knowledge.glossary:file", "glossary", str(knowledge_payload.get("glossary", {})), 0.8, 0.86, "fact", 0.9))
    if domain == "blockers":
        add(EvidenceItem("knowledge.blockers_catalog:file", "blockers", str(knowledge_payload.get("blockers_catalog", {})), 0.88, 0.9, "fact", 0.94))
    if domain == "recommendation":
        add(EvidenceItem("recommendation.engine", "recommendation", str(knowledge_payload.get("recommendation_result", {})), 0.95, 0.92, "inference", 0.98))
        add(EvidenceItem("knowledge.risk_policy:file", "risk_policy", str(knowledge_payload.get("risk_policy", {})), 0.9, 0.94, "fact", 0.94))
        add(EvidenceItem("knowledge.blockers_catalog:file", "blockers", str(knowledge_payload.get("blockers_catalog", {})), 0.88, 0.9, "fact", 0.92))
    if domain == "performance_feedback":
        add(EvidenceItem("performance.feedback", "performance_feedback", str(knowledge_payload.get("performance_feedback", {})), 0.95, 0.92, "inference", 0.99))
        add(EvidenceItem("recommendation.engine", "recommendation", str(knowledge_payload.get("recommendation_result", {})), 0.95, 0.9, "inference", 0.9))
        add(EvidenceItem("adaptation.monitor", "runtime_status", str(knowledge_payload.get("adaptation_summary", {})), 0.95, 0.9, "fact", 0.9))
    if domain == "paper_tuning":
        add(EvidenceItem("paper.tuning", "paper_tuning", str(knowledge_payload.get("paper_tuning", {})), 0.95, 0.93, "inference", 0.99))
        add(EvidenceItem("performance.feedback", "performance_feedback", str(knowledge_payload.get("performance_feedback", {})), 0.95, 0.9, "inference", 0.9))
        add(EvidenceItem("recommendation.engine", "recommendation", str(knowledge_payload.get("recommendation_result", {})), 0.95, 0.88, "inference", 0.86))
    if domain == "general_operator_help":
        add(EvidenceItem("knowledge.system_overview", "general_operator_help", str(knowledge_payload.get("system_overview", {})), 0.65, 0.82, "fact", 0.9))
    if domain == "experiment_recommendation":
        add(EvidenceItem("knowledge.readiness_rules", "experiment_recommendation", str(knowledge_payload.get("readiness_rules", {})), 0.65, 0.82, "fact", 0.9))
    if domain in {"runtime_status", "comparison_results"} and "adaptation_summary" in knowledge_payload:
        add(EvidenceItem("adaptation.monitor", "runtime_status", str(knowledge_payload.get("adaptation_summary", {})), 0.95, 0.9, "fact", 0.94))

    if any(k in q for k in ("what can you do", "how much do you know", "limitations", "intelligent")):
        out = [e for e in out if e.domain in {"self_model", "market_knowledge", "strategy_knowledge", "risk_policy", "general_operator_help", "worker_neurons"}]
    if any(k in q for k in ("worker", "workers", "neuron", "neurons")):
        out = [e for e in out if e.domain in {"worker_neurons", "self_model", "runtime_status", "general_operator_help"}]
    if any(k in q for k in ("own brain", "atlas brain", "one big brain", "giant brain", "one giant brain", "big brain", "processes everything", "understanding and learning", "brain to do all those things")):
        out = [e for e in out if e.domain in {"cognitive_core", "worker_neurons", "self_model", "runtime_status", "general_operator_help"}]
    if any(k in q for k in ("what is volatility", "what is a trend", "what does hold mean", "risk rules", "what strategies do you use")):
        out = [e for e in out if e.domain in {"market_knowledge", "strategy_knowledge", "risk_policy", "glossary", "blockers", "self_model"}]
    if any(k in q for k in ("recommend", "safe step", "safest move", "what should i watch", "reviewed first", "safe experiment")):
        out = [e for e in out if e.domain in {"recommendation", "risk_policy", "blockers", "runtime_status", "comparison_results", "experiment_recommendation"}]
    if any(k in q for k in ("how well are you adapting", "recommendations helping", "too reactive", "improving", "weakness")):
        out = [e for e in out if e.domain in {"performance_feedback", "recommendation", "runtime_status", "blockers"}]
    if any(k in q for k in ("small tuning", "paper-mode adjustment", "tuned first", "thresholds too strict", "paper experiment", "paper tuning")):
        out = [e for e in out if e.domain in {"paper_tuning", "performance_feedback", "recommendation", "runtime_status", "risk_policy"}]
    out.sort(key=lambda e: (e.relevance_score, e.confidence, e.recency), reverse=True)
    return out
