from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    content: str
    domain: str
    recency: float
    confidence: float
    relevance: float
    kind: str  # fact | inference


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def collect_candidate_evidence(
    *,
    query: str,
    domain: str,
    status: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]],
    comparison: dict[str, Any],
    self_model: dict[str, Any],
    knowledge: dict[str, dict[str, Any]],
) -> list[EvidenceItem]:
    q = str(query or "").lower()
    evidence: list[EvidenceItem] = []
    add = evidence.append

    status_message = str(status.get("message", "") or "")
    action = str(status.get("action", "") or "")
    run_id = str(status.get("comparison_run_id", "") or str(dict(status.get("comparison", {}) or {}).get("run_id", "")))
    mode = "LIVE" if bool(runtime.get("execute_trades", False)) else "PAPER"

    if domain in {"current_system_state", "blockers", "runtime_status", "comparison_results", "readiness_assessment"}:
        add(EvidenceItem("status", f"action={action}; message={status_message}", "current_system_state", 0.9, 0.95, 0.88, "fact"))
        add(EvidenceItem("runtime", f"mode={mode}", "runtime_status", 0.95, 0.95, 0.82, "fact"))
        if run_id:
            add(EvidenceItem("status", f"comparison_run_id={run_id}", "comparison_results", 0.9, 0.9, 0.74, "fact"))

    if domain in {"self_model", "market_knowledge", "strategy_knowledge", "general_operator_help"}:
        add(
            EvidenceItem(
                "self_model",
                f"identity={self_model.get('identity')}; purpose={self_model.get('purpose')}; role={self_model.get('current_role')}",
                "self_model",
                0.8,
                0.9,
                0.95 if domain == "self_model" else 0.72,
                "fact",
            )
        )
        add(
            EvidenceItem(
                "self_model",
                f"capabilities={self_model.get('capabilities')}; limitations={self_model.get('limitations')}",
                "self_model",
                0.8,
                0.9,
                0.94 if domain == "self_model" else 0.7,
                "fact",
            )
        )

    if domain == "market_knowledge":
        mk = knowledge.get("market_knowledge", {})
        add(EvidenceItem("knowledge.market_knowledge", str(mk), "market_knowledge", 0.7, 0.85, 0.96, "fact"))
    if domain == "strategy_knowledge":
        sk = knowledge.get("strategy_knowledge", {})
        add(EvidenceItem("knowledge.strategy_knowledge", str(sk), "strategy_knowledge", 0.7, 0.85, 0.95, "fact"))
    if domain == "general_operator_help":
        so = knowledge.get("system_overview", {})
        add(EvidenceItem("knowledge.system_overview", str(so), "general_operator_help", 0.6, 0.8, 0.9, "fact"))

    if domain == "comparison_results":
        add(EvidenceItem("comparison", str(comparison or status.get("comparison", {}) or {}), "comparison_results", 0.8, 0.9, 0.95, "fact"))

    if domain == "experiment_recommendation":
        rr = knowledge.get("readiness_rules", {})
        add(EvidenceItem("knowledge.readiness_rules", str(rr), "experiment_recommendation", 0.6, 0.82, 0.9, "fact"))
        add(EvidenceItem("status", f"message={status_message}", "experiment_recommendation", 0.9, 0.87, 0.8, "inference"))

    if _contains_any(q, ("how much do you know", "what can you do", "capabilities", "limitations", "intelligent")):
        evidence = [e for e in evidence if e.domain in {"self_model", "market_knowledge", "strategy_knowledge", "general_operator_help"}]

    evidence.sort(key=lambda e: (e.relevance, e.confidence, e.recency), reverse=True)
    return evidence

