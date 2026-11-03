from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any
import uuid
import hashlib
from collections import Counter

from atlas_ai.brain import AnswerEngine, BrainKnowledge, BrainMemory, classify_intent_candidates
from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
from atlas_ai.brain.ai_decision_reviewer import AIDecisionReviewer
from atlas_ai.brain.ai_promotion_intelligence import AIPromotionIntelligence
from atlas_ai.brain.nlu import parse_user_language
from atlas_ai.brain.control_layer import ControlLayer
from atlas_ai.brain.why_engine import build_why_trace
from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.decision_support import build_evidence_separation
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.runtime_heartbeat import get_backend_health
from atlas_ai.self_improvement import load_self_improvement_bundle
from atlas_ai.ui.structured_trading_analysis import (
    build_structured_trading_analysis_reply,
    format_trading_response_text,
    is_structured_trading_analysis_query,
)


@dataclass(frozen=True)
class AskAtlasReply:
    mode: str
    intent: str
    reasoning_mode: str
    acknowledge: str
    understanding: str
    data_sources: list[str]
    result: str
    approval: str
    response_kind: str = "question_answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "reasoning_mode": self.reasoning_mode,
            "acknowledge": self.acknowledge,
            "understanding": self.understanding,
            "data_sources": list(self.data_sources),
            "result": self.result,
            "approval": self.approval,
            "response_kind": self.response_kind,
        }


class AskAtlasService:
    def __init__(self, source: Any, *, execution_gate: ExecutionGate | None = None) -> None:
        self.source = source
        self.memory = BrainMemory()
        self.knowledge = BrainKnowledge()
        self.answer_engine = AnswerEngine(self.knowledge)
        self.control = ControlLayer()
        self.decision_journal = self._build_decision_journal(source)
        self.decision_reviewer = AIDecisionReviewer(self.decision_journal.base)
        self.promotion_intelligence = AIPromotionIntelligence(self.decision_journal.base)
        if execution_gate is not None:
            self.execution = execution_gate
        else:
            state_store_path = None
            if os.getenv("PYTEST_CURRENT_TEST"):
                state_store_path = str(Path(tempfile.gettempdir()) / f"atlas_automation_state_{uuid.uuid4().hex}.json")
            elif hasattr(source, "store") and getattr(source, "store", None) is not None:
                try:
                    state_store_path = str(Path(getattr(source.store, "base")) / "automation_state.json")
                except Exception:
                    state_store_path = None
            self.execution = ExecutionGate(state_store_path=state_store_path)

    @staticmethod
    def _build_decision_journal(source: Any) -> AIDecisionJournal:
        if hasattr(source, "store") and getattr(source, "store", None) is not None:
            try:
                return AIDecisionJournal(getattr(source.store, "base"))
            except Exception:
                pass
        if os.getenv("PYTEST_CURRENT_TEST"):
            return AIDecisionJournal(Path(tempfile.gettempdir()) / f"atlas_ai_decisions_{uuid.uuid4().hex}")
        return AIDecisionJournal("storage")

    def _last_confirmed_trade_outcome(self) -> dict[str, Any]:
        try:
            state = dict(self.source.load_learning_state().get("state", {}) or {})
        except Exception:
            state = {}
        intelligence = dict(state.get("trade_outcome_intelligence", {}) or {})
        trades = list(dict(intelligence.get("outcome_memory", {}) or {}).get("trades", []) or [])
        if not trades:
            loss_memory = dict(state.get("loss_memory", {}) or {})
            trades = list(loss_memory.get("recent_outcomes", []) or [])
        rows = [
            dict(row or {})
            for row in trades
            if str(dict(row or {}).get("outcome", "")).strip().lower() in {"win", "loss", "draw"}
        ]
        return rows[-1] if rows else {}

    def answer(self, query: str, mode: str = "standard") -> dict[str, Any]:
        q = " ".join(str(query or "").strip().split())
        nlu = parse_user_language(q)
        handled_as = self._classify_handling_mode(q)
        nlu_route_map = {
            "trading_analysis": "trading",
            "trade_explanation": "explanation",
            "system_status": "system",
            "system_command": "command",
            "trading_command": "trading",
            "learning_question": "brain",
            "general_question": "general",
            "ambiguous": "general",
        }
        handled_as = nlu_route_map.get(nlu.intent, handled_as)
        if nlu.intent == "ambiguous":
            return self._with_handling(
                {
                    "mode": "standard",
                    "intent": "ambiguous",
                    "reasoning_mode": "CLARIFY",
                    "acknowledge": "I need one clarification.",
                    "understanding": "The request is ambiguous.",
                    "data_sources": ["nlu"],
                    "result": nlu.clarification_question or "Please clarify what you want Atlas to do.",
                    "approval": "No action taken.",
                    "response_kind": "question_answer",
                    "nlu": nlu.to_dict(),
                },
                "general",
            )
        if nlu.safety_required and "confirm" not in nlu.normalized_text and nlu.intent == "system_command":
            return self._with_handling(
                {
                    "mode": "standard",
                    "intent": "system_command_confirmation_required",
                    "reasoning_mode": "SAFETY_CONFIRMATION",
                    "acknowledge": "Command understood.",
                    "understanding": "Destructive system command requires confirmation.",
                    "data_sources": ["nlu", "safety policy"],
                    "result": "Please confirm this command by saying or typing: confirm shutdown.",
                    "approval": "Confirmation required before command routing.",
                    "response_kind": "question_answer",
                    "nlu": nlu.to_dict(),
                },
                "command",
            )
        if nlu.safety_required and nlu.intent == "trading_command":
            return self._with_handling(
                {
                    "mode": "standard",
                    "intent": "safe_trading_command",
                    "reasoning_mode": "SAFETY_BOUNDARY",
                    "acknowledge": "Command understood.",
                    "understanding": "Trading command is safety-bound and cannot bypass gates.",
                    "data_sources": ["nlu", "execution safety"],
                    "result": "I can analyze the setup, but execution still requires ExecutionGate approval.",
                    "approval": "No direct execution from language command.",
                    "response_kind": "question_answer",
                    "nlu": nlu.to_dict(),
                },
                "trading",
            )
        if self._is_mutating_request(q):
            return self._with_handling(
                {
                "mode": "standard",
                "intent": "refusal_mutating_request",
                "reasoning_mode": "SAFETY_REFUSAL",
                "acknowledge": "I received a mutating request.",
                "understanding": "This asks for trade/runtime mutation which is blocked in Ask Atlas.",
                "data_sources": ["safety policy", "command gating"],
                "result": (
                    "I can explain state, blockers, readiness, and safe next steps, but I will not place trades, "
                    "enable live execution, or mutate runtime settings from Ask Atlas."
                ),
                "approval": "Execution and runtime changes require explicit approval-gated execution commands.",
                "response_kind": "question_answer",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
                },
                "command",
            )
        exec_cmd = self._parse_execution_command(q)
        if exec_cmd is not None:
            return self._with_handling(self._handle_execution_command(exec_cmd), "command")
        control_cmd = self.control.parse_command(q)
        if control_cmd is not None:
            return self._with_handling(self._handle_control_command(control_cmd), "command")
        if self._is_structured_trading_analysis_query(q):
            payload = self._with_handling(self._build_structured_trading_analysis_reply(q), "trading")
            self._remember_direct_payload(q, payload)
            return payload
        if self._is_volatility_query(q):
            return self._with_handling(self._build_volatility_reply(q, mode=mode), "trading")
        if self._is_safety_lock_query(q):
            payload = self._build_safety_lock_reply(q, mode=mode)
            self._remember_direct_payload(q, payload)
            return self._with_handling(payload, "system")
        exec_query = self._handle_execution_query(q)
        if exec_query is not None:
            return self._with_handling(exec_query, "system" if str(exec_query.get("response_kind", "")) == "control" else handled_as)
        if handled_as == "trading" and self._is_structured_trading_analysis_query(q):
            payload = self._with_handling(self._build_structured_trading_analysis_reply(q), "trading")
            self._remember_direct_payload(q, payload)
            return payload
        if self._is_operator_guidance_query(q):
            return self._with_handling(self._build_operator_guidance_reply(q), "general")
        if not q:
            return self._with_handling(AskAtlasReply(
                mode="standard",
                intent="unknown_operator_query",
                reasoning_mode="DIRECT",
                acknowledge="Ready.",
                understanding="Waiting for an operator question.",
                data_sources=["current status", "recent events", "comparison data when active"],
                result="Ask a direct question like: What state are you in? What is holding you back?",
                approval="Read-only assistant. No runtime or trading changes are executed here.",
            ).to_dict(), "general")
        if self._is_structured_trading_analysis_query(q):
            payload = self._with_handling(self._build_structured_trading_analysis_reply(q), "trading")
            self._remember_direct_payload(q, payload)
            return payload

        if handled_as == "system":
            payload = self._build_system_status_reply(q, mode=mode)
            self.memory.add(
                query=q,
                intent=str(payload.get("intent", "status") or "status"),
                domain=str(payload.get("selected_domain", "runtime_status") or "runtime_status"),
                plan="status_overview",
                answer=str(payload.get("result", "") or ""),
                confidence=0.9,
            )
            return self._with_handling(payload, "system")
        if handled_as == "explanation" and self._should_use_direct_explanation_reply(q):
            return self._with_handling(self._build_explanation_reply(q), "explanation")
        if handled_as == "general" and self._should_use_general_fallback_reply(q):
            return self._with_handling(self._build_general_question_reply(q, mode=mode), "general")
        if self._is_learning_lanes_query(q):
            payload = self._build_learning_lanes_reply(q, mode=mode)
            self._remember_direct_payload(q, payload)
            return self._with_handling(payload, "brain")

        response = self._with_handling(self._answer_via_brain(q, mode=mode), handled_as)
        response["nlu"] = nlu.to_dict()
        response.setdefault("debug", {})
        response["debug"]["detected_intent"] = nlu.intent
        response["debug"]["intent_confidence"] = nlu.confidence
        response["debug"]["entities"] = dict(nlu.entities)
        response["debug"]["routed_handler"] = handled_as
        response["debug"]["clarification_needed"] = bool(nlu.ambiguity_detected)
        response["debug"]["safety_required"] = bool(nlu.safety_required)
        if handled_as == "trading":
            self._ensure_trading_response_schema(response)
        return response

    def _answer_via_brain(self, query: str, mode: str = "standard") -> dict[str, Any]:
        q = " ".join(str(query or "").strip().split())
        status = dict(self.source.load_status() or {})
        self._enrich_status_with_self_improvement_context(status)
        events = list(self.source.load_events(limit=160) or [])
        runtime = dict(self.source.load_runtime_flags() or {})
        runtime["execution"] = self.execution.snapshot(runtime_flags=runtime)
        comparison = dict(getattr(self.source, "load_comparison", lambda: {})() or {})
        learning_state = dict(getattr(self.source, "load_learning_state", lambda: {})() or {})
        ai_decision_journal_recent = self.decision_journal.recent(limit=5)
        ai_decision_reviews_recent = self.decision_reviewer.recent_reviews(limit=5)
        ai_strategy_preferences = self.decision_reviewer.preferences.load()
        ai_promotion_intelligence = self.promotion_intelligence.load_latest()

        resolved_query = self.memory.build_follow_up_context(q) if self.memory.is_follow_up(q) else q
        provisional_intent = classify_intent_candidates(resolved_query)[0].name
        response_mode = self._select_mode(q, provisional_intent, mode)
        engine_result = self.answer_engine.answer(
            resolved_query,
            {
                "status": status,
                "runtime": runtime,
                "events": events,
                "comparison": comparison,
                "learning_state": learning_state,
                "ai_decision_journal_recent": ai_decision_journal_recent,
                "ai_decision_reviews_recent": ai_decision_reviews_recent,
                "ai_strategy_preferences": ai_strategy_preferences,
                "ai_promotion_intelligence": ai_promotion_intelligence,
            },
            mode=response_mode,
        )
        ai_trace = dict(engine_result.debug_summary.get("ai_decision_trace", {}) or {})
        if ai_trace:
            self.decision_journal.append(ai_trace, source="ask_atlas", query=q)
            self.decision_reviewer.review(learning_state=learning_state, events=events, limit=10)
        intent = engine_result.selected_intent
        result = engine_result.text
        self.memory.add(
            query=q,
            intent=intent,
            domain=engine_result.selected_domain,
            plan=engine_result.selected_plan,
            answer=result,
            confidence=engine_result.confidence,
        )
        created_control_items = (
            self.control.ingest_from_debug(dict(engine_result.debug_summary or {}))
            if self._should_create_control_items_from_brain(
                intent=intent,
                domain=engine_result.selected_domain,
                query=q,
            )
            else []
        )
        # Read-only default: generic Ask Atlas Q&A must not trigger execution side effects.
        auto_exec_result = None

        payload = AskAtlasReply(
            mode=response_mode,
            intent=intent,
            reasoning_mode="INTENT_SYNTHESIS",
            acknowledge="Answer ready.",
            understanding=f"Interpreted as: {intent}{' (follow-up resolved)' if self.memory.is_follow_up(q) else ''}",
            data_sources=[
                "current status",
                "runtime flags",
                "recent events",
                "comparison metadata",
                "structured operator knowledge",
            ],
            result=result,
            approval="Read-only assistant. System start/stop requires explicit system command intent.",
        )
        response = payload.to_dict()
        response["selected_domain"] = engine_result.selected_domain
        response["confidence"] = engine_result.confidence
        response["candidate_intents"] = [c.__dict__ for c in engine_result.candidate_intents]
        response["evidence_summary"] = [f"{e.source}:{e.domain}:{e.relevance_score:.2f}" for e in engine_result.evidence_used]
        response["debug"] = dict(engine_result.debug_summary)
        response["answer_trace"] = dict(response["debug"].get("answer_trace", {}) or {})
        response["control_items_created"] = [self.control.item_to_dict(x) for x in created_control_items]
        execution_state = self.execution.snapshot(runtime_flags=runtime)
        forced_item, _ = self._sync_forced_flat_control_item(runtime, execution_state)
        if forced_item is not None:
            response["control_items_created"].append(self.control.item_to_dict(forced_item))
        recon_items = self._sync_order_reconciliation_items(execution_state)
        if recon_items:
            response["control_items_created"].extend([self.control.item_to_dict(x) for x in recon_items])
        orphan_items = self._sync_orphan_order_items(execution_state)
        if orphan_items:
            response["control_items_created"].extend([self.control.item_to_dict(x) for x in orphan_items])
        response["control_state"] = self.control.to_debug()
        response["execution_state"] = execution_state
        response["debug"]["display_mode"] = str(execution_state.get("execution_environment_label", "Paper") or "Paper")
        self._ensure_trading_response_schema(response, status=status, runtime=runtime)
        return response

    @staticmethod
    def _should_create_control_items_from_brain(*, intent: str, domain: str, query: str) -> bool:
        _ = str(domain or "")
        q = str(query or "").strip().lower()
        if not q:
            return False
        allowed_intents = {
            "recommendation",
            "safe_next_step",
            "next_safe_experiment",
            "experiment_suggestion",
            "paper_experiment",
            "tuning_recommendation",
            "threshold_review",
            "filter_review",
            "tuning_comparison",
            "performance_review",
            "adaptation_review",
            "recommendation_review",
            "stability_review",
            "improvement_assessment",
            "weakness_analysis",
            "blocker_resolution",
        }
        if str(intent or "").strip().lower() not in allowed_intents:
            return False
        question_starts = (
            "what ",
            "why ",
            "how ",
            "when ",
            "where ",
            "which ",
            "who ",
            "are ",
            "is ",
            "can ",
            "could ",
            "should ",
            "would ",
            "do ",
            "does ",
            "did ",
        )
        explicit_instruction_phrases = (
            "create a control item",
            "create control item",
            "create a review item",
            "create review item",
            "add a control item",
            "add control item",
            "add this to the control queue",
            "add to control queue",
            "open a review item",
            "open review item",
            "queue a review",
            "queue review",
            "create a tuning proposal",
            "create tuning proposal",
            "propose a tuning",
            "propose tuning",
            "give me a safe experiment",
            "give me the next safe experiment",
            "create a safe experiment",
            "create safe experiment",
            "queue a safe experiment",
            "queue safe experiment",
        )
        if any(phrase in q for phrase in explicit_instruction_phrases):
            return True
        if q.endswith("?") or q.startswith(question_starts):
            return False
        return False

    def _enrich_status_with_self_improvement_context(self, status: dict[str, Any]) -> None:
        bundle: dict[str, Any] = {}
        if hasattr(self.source, "load_self_improvement_bundle"):
            try:
                bundle = dict(self.source.load_self_improvement_bundle() or {})
            except Exception:
                bundle = {}
        if not bundle and hasattr(self.source, "store") and getattr(self.source, "store", None) is not None:
            try:
                bundle = load_self_improvement_bundle(getattr(self.source, "store"))
            except Exception:
                bundle = {}
        proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
        pending = [
            dict(item)
            for item in proposals
            if str(dict(item).get("status", "") or "").strip().lower() == "awaiting_approval"
        ]
        if not pending:
            return
        latest = pending[-1]
        system_health = dict(status.get("system_health", {}) or {})
        self_improvement = dict(system_health.get("self_improvement", {}) or {})
        self_improvement["approval_pending_count"] = max(
            int(self_improvement.get("approval_pending_count", 0) or 0),
            len(pending),
        )
        self_improvement["latest_proposal_id"] = str(latest.get("proposal_id", "") or "")
        self_improvement["pending_proposal_id"] = str(latest.get("proposal_id", "") or "")
        self_improvement["latest_proposal_category"] = str(latest.get("category", "") or "")
        self_improvement["pending_proposal_category"] = str(latest.get("category", "") or "")
        self_improvement["latest_proposal_target_blocker"] = str(
            dict(latest.get("scope", {}) or {}).get("target_blocker", "") or ""
        )
        self_improvement["pending_proposal_target_blocker"] = self_improvement["latest_proposal_target_blocker"]
        proposal_id = self_improvement["pending_proposal_id"]
        packet = self._load_operator_review_packet(proposal_id)
        posture = dict(packet.get("operator_posture", {}) or {})
        if posture:
            self_improvement["pending_operator_posture"] = str(posture.get("posture", "") or "")
            self_improvement["pending_operator_posture_reason"] = str(posture.get("reason", "") or "")
            self_improvement["pending_operator_suggested_action"] = str(posture.get("suggested_operator_action", "") or "")
        system_health["self_improvement"] = self_improvement
        status["system_health"] = system_health

    def _load_operator_review_packet(self, proposal_id: str) -> dict[str, Any]:
        safe_id = str(proposal_id or "").strip()
        if not safe_id:
            return {}
        bases: list[Path] = []
        if hasattr(self.source, "store") and getattr(self.source, "store", None) is not None:
            with_context = getattr(self.source, "store")
            with_base = getattr(with_context, "base", "")
            if with_base:
                bases.append(Path(with_base))
        bases.append(Path("storage"))
        for base in bases:
            path = base / "proposal_reports" / f"{safe_id}_operator_review_packet.json"
            try:
                if path.exists():
                    return dict(json.loads(path.read_text(encoding="utf-8")) or {})
            except Exception:
                continue
        return {}

    def _ensure_trading_response_schema(
        self,
        response: dict[str, Any],
        *,
        status: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
    ) -> None:
        handled_as = str(response.get("handled_as", "") or "").strip().lower()
        if handled_as != "trading":
            return
        if response.get("trading_response"):
            return
        # Preserve brain-generated results; only enforce generic trading schema
        # when the response did not come from the brain answer engine.
        if response.get("candidate_intents") or response.get("selected_domain"):
            st = dict(status or self.source.load_status() or {})
            rt = dict(runtime or self.source.load_runtime_flags() or {})
            decision = build_structured_decision_analysis(status=st, runtime=rt)
            trading_response = dict(decision.get("trading_response", {}) or {})
            response.setdefault("trading_response", trading_response)
            return
        st = dict(status or self.source.load_status() or {})
        rt = dict(runtime or self.source.load_runtime_flags() or {})
        decision = build_structured_decision_analysis(status=st, runtime=rt)
        trading_response = dict(decision.get("trading_response", {}) or {})
        response["trading_response"] = trading_response
        response["result"] = self._format_trading_response_text(
            trading_response,
            fallback=str(response.get("result", "") or ""),
        )

    def _with_handling(self, payload: dict[str, Any], handled_as: str) -> dict[str, Any]:
        out = dict(payload or {})
        out["handled_as"] = str(handled_as or "general")
        if "answer_trace" not in out:
            selected_domain = str(out.get("selected_domain", "") or "")
            if not selected_domain:
                selected_domain = {
                    "command": "command_execution",
                    "system": "runtime_status",
                    "trading": "current_system_state",
                    "explanation": "blockers",
                    "general": "general_operator_help",
                }.get(str(handled_as or "general"), "general_operator_help")
            trace = {
                "normalized_question": "",
                "selected_intent": str(out.get("intent", "unknown_operator_query") or "unknown_operator_query"),
                "selected_domain": selected_domain,
                "selected_plan": str(dict(out.get("debug", {}) or {}).get("selected_plan", out.get("response_kind", "direct_handler")) or "direct_handler"),
                "evidence": [],
                "relevance_result": str(dict(out.get("debug", {}) or {}).get("relevance_guard", "direct_handler") or "direct_handler"),
                "handler": str(handled_as or "general"),
            }
            if out.get("evidence_separation"):
                trace["evidence_separation"] = dict(out.get("evidence_separation", {}) or {})
            out["answer_trace"] = trace
            out.setdefault("selected_domain", selected_domain)
            debug = dict(out.get("debug", {}) or {})
            debug.setdefault("answer_trace", trace)
            out["debug"] = debug
        self._attach_backend_staleness_warning(out)
        return out

    @staticmethod
    def _current_state_warning_scope(payload: dict[str, Any]) -> bool:
        intent = str(payload.get("intent", "") or "").strip().lower()
        domain = str(payload.get("selected_domain", "") or "").strip().lower()
        trace = dict(payload.get("answer_trace", {}) or {})
        trace_domain = str(trace.get("selected_domain", "") or "").strip().lower()
        handled_as = str(payload.get("handled_as", "") or "").strip().lower()
        current_state_intents = {
            "status",
            "blocker",
            "system_status_query",
            "execution_status",
            "automation_status",
            "volatility_state_query",
            "safety_lock_status",
            "blocker_resolution",
        }
        current_state_domains = {
            "runtime_status",
            "current_system_state",
            "blockers",
            "risk_policy",
            "market_volatility",
            "execution_status",
            "execution_control",
        }
        return (
            intent in current_state_intents
            or domain in current_state_domains
            or trace_domain in current_state_domains
            or handled_as in {"system", "explanation"}
        )

    @staticmethod
    def _format_backend_staleness_warning(health: dict[str, Any]) -> str:
        status = str(health.get("status", "") or "").strip().lower()
        threshold = int(health.get("staleness_threshold_seconds", 0) or 0)
        if status == "unknown":
            return (
                "Warning: Backend heartbeat is unknown because no heartbeat file is present. "
                "The status below is last-known state and may be out of date."
            )
        seconds = int(health.get("seconds_since_heartbeat", 0) or 0)
        minutes = seconds / 60.0
        if minutes >= 1.0:
            age = f"{minutes:.1f} minutes"
        else:
            age = f"{seconds} seconds"
        return (
            f"Warning: Backend has not reported in for {age} "
            f"(threshold {threshold} seconds). The status below is last-known state and may be out of date."
        )

    def _attach_backend_staleness_warning(self, payload: dict[str, Any]) -> None:
        if not self._current_state_warning_scope(payload):
            return
        store = getattr(self.source, "store", None)
        if store is None:
            return
        try:
            health = get_backend_health(store)
        except Exception:
            return
        status = str(health.get("status", "") or "").strip().lower()
        if status == "live":
            return
        if status not in {"stale", "unknown"}:
            return
        warning = self._format_backend_staleness_warning(health)
        result = str(payload.get("result", "") or "")
        if result.startswith("Warning: Backend"):
            return
        payload["result"] = f"{warning}\n\nLast-known answer:\n{result}" if result else warning
        payload["backend_health"] = dict(health)
        payload["backend_staleness_warning"] = warning
        debug = dict(payload.get("debug", {}) or {})
        debug["backend_health"] = dict(health)
        debug["backend_staleness_warning_attached"] = True
        payload["debug"] = debug

    def _remember_direct_payload(self, query: str, payload: dict[str, Any], *, confidence: float = 0.7) -> None:
        trace = dict(payload.get("answer_trace", {}) or {})
        debug = dict(payload.get("debug", {}) or {})
        self.memory.add(
            query=query,
            intent=str(payload.get("intent", trace.get("selected_intent", "unknown_operator_query")) or "unknown_operator_query"),
            domain=str(payload.get("selected_domain", trace.get("selected_domain", "general_operator_help")) or "general_operator_help"),
            plan=str(trace.get("selected_plan", debug.get("selected_plan", "direct_handler")) or "direct_handler"),
            answer=str(payload.get("result", "") or ""),
            confidence=float(payload.get("confidence", confidence) or confidence),
        )

    @staticmethod
    def _format_trading_response_text(trading_response: dict[str, Any], fallback: str = "") -> str:
        return format_trading_response_text(trading_response, fallback=fallback)

    def _classify_handling_mode(self, query: str) -> str:
        """Route query to the appropriate handling subsystem.

        All natural-language operator queries now route through the brain
        answer engine for consistent intent classification, debug fields,
        and follow-up context. Only explicit command patterns bypass the
        brain and go directly to execution/control routing.
        """
        q = str(query or "").strip().lower()
        if not q:
            return "general"
        latest = self.memory.latest()
        exact_command_phrases = {
            "atlas wake",
            "atlas wake up",
            "wake atlas",
            "wake up atlas",
            "atlas sleep",
            "go to sleep atlas",
            "close system",
            "start system",
        }
        if q in exact_command_phrases:
            return "command"
        if self._parse_execution_command(q) is not None or self.control.parse_command(q) is not None:
            return "command"
        if "best self" in q or "become your best self" in q:
            return "general"
        explanation_markers = (
            "why no trade",
            "why not trading",
            "why are you not trading",
            "why blocked",
            "why did you block",
            "explain the decision",
            "explain the blocker",
            "what is the main blocker",
            "what is holding you back",
        )
        if any(marker in q for marker in explanation_markers):
            return "explanation"
        if self._is_structured_trading_analysis_query(q):
            return "trading"
        intent_candidates = classify_intent_candidates(q)
        top_intent = str(intent_candidates[0].name if intent_candidates else "").strip().lower()
        if top_intent in {"system_command"}:
            return "command"
        if top_intent in {"blocker", "explanation"}:
            return "explanation"
        if top_intent in {"recommendation", "confidence", "market_knowledge", "strategy_knowledge", "risk_policy"}:
            return "trading"
        if top_intent in {"system_status_query", "status"}:
            return "system"
        if top_intent in {"readiness", "comparison"}:
            return "brain"
        if self.memory.is_follow_up(q) and latest is not None:
            if latest.intent in {"blocker", "explanation", "recommendation", "trading_analysis"}:
                return "brain"
            if latest.intent in {
                "status",
                "readiness",
                "comparison",
                "self_knowledge",
                "self_assessment",
                "capabilities",
                "limitations",
                "market_knowledge",
                "strategy_knowledge",
                "risk_policy",
                "glossary",
            }:
                return "brain"
        if top_intent in {"self_knowledge", "self_assessment", "capabilities", "limitations", "help", "unknown_operator_query"}:
            return "general"
        trade_tokens = {"buy", "sell", "trade", "entry", "setup", "bias", "long", "short"}
        if any(token in q.split() for token in trade_tokens):
            return "trading"
        return "brain"

    @staticmethod
    def _should_use_direct_explanation_reply(query: str) -> bool:
        q = str(query or "").strip().lower()
        return q in {"why no trade", "why no trade?", "why blocked", "why blocked?"}

    @staticmethod
    def _should_use_general_fallback_reply(query: str) -> bool:
        q = str(query or "").strip().lower()
        return "best self" in q or "become your best self" in q

    @staticmethod
    def _is_learning_lanes_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        if any(phrase in q for phrase in ("new brain", "cognitive core", "post trade lesson upgrade", "brain upgrade")):
            return False
        if "chart" in q or "charts" in q:
            return False
        phrases = (
            "does atlas study",
            "do you study",
            "are you studying",
            "study all the time",
            "what are you learning",
            "what did you learn",
            "what have you learned",
            "what is atlas learning",
            "how are you learning",
            "learning lanes",
            "learning status",
            "do you learn from losses",
        )
        return any(phrase in q for phrase in phrases)

    @staticmethod
    def _is_safety_lock_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        phrases = (
            "safety lock",
            "why are you locked",
            "why is atlas locked",
            "what clears the lock",
            "what clears safety",
            "clear the safety lock",
            "why is trading locked",
            "why are trades locked",
            "is this a pattern ban",
            "is the safety lock a pattern ban",
        )
        return any(phrase in q for phrase in phrases)

    def _build_safety_lock_reply(self, query: str, *, mode: str = "standard") -> dict[str, Any]:
        status = dict(self.source.load_status() or {})
        review = dict(status.get("safety_lock_review", {}) or {})
        if not review:
            review = self._derive_safety_lock_review_for_answer(status)
        active = bool(review.get("active", False))
        reason = str(review.get("reason", "") or ("safety_lock_active" if active else "none"))
        evidence = [dict(item or {}) for item in list(review.get("evidence", []) or [])[:4]]
        checklist = [dict(item or {}) for item in list(review.get("recovery_checklist", []) or [])[:6]]
        separate = dict(review.get("separate_from_learning", {}) or {})
        if str(mode or "").strip().lower() == "voice":
            result = (
                f"Safety lock is {'active' if active else 'clear'}."
                + (f" Reason: {reason}." if active else "")
                + " It is account protection, not a pattern ban, and it cannot be bypassed by conversation."
            )
        else:
            evidence_lines = [
                f"- {str(item.get('label', item.get('code', 'evidence')))}: {item.get('value', 'n/a')} - {str(item.get('detail', '') or '').strip()}"
                for item in evidence
            ]
            checklist_lines = [
                f"- {str(item.get('label', item.get('code', 'review required')))} [{str(item.get('status', 'review'))}]"
                for item in checklist
            ]
            result = (
                f"Safety lock: {'ACTIVE' if active else 'CLEAR'}.\n\n"
                f"Reason:\n- {reason}\n\n"
                "Evidence:\n"
                + ("\n".join(evidence_lines) if evidence_lines else "- No safety-lock evidence reported.")
                + "\n\nRecovery path:\n"
                + ("\n".join(checklist_lines) if checklist_lines else "- Normal market, risk, timing, and execution gates still apply.")
                + "\n\nLearning separation:\n"
                f"- not a pattern ban. {str(separate.get('note', 'Safety lock is account protection, not a pattern ban.'))}\n\n"
                "Boundary:\n"
                "- This answer is read-only. It does not unlock safety, change risk, force entries, or bypass execution gates."
            )
        return {
            "mode": "voice" if str(mode).lower() == "voice" else "standard",
            "intent": "safety_lock_review",
            "selected_domain": "risk_policy",
            "reasoning_mode": "SAFETY_LOCK_REVIEW",
            "acknowledge": "Safety-lock review ready.",
            "understanding": "Interpreted as a question about the active safety lock and recovery path.",
            "data_sources": ["current status", "safety lock review", "risk policy boundary"],
            "result": result,
            "approval": "Read-only answer. No unlock or execution action was taken.",
            "response_kind": "question_answer",
            "safety_lock_review": review,
        }

    @staticmethod
    def _derive_safety_lock_review_for_answer(status: dict[str, Any]) -> dict[str, Any]:
        system_health = dict(status.get("system_health", {}) or {})
        safety = dict(system_health.get("safety_layer", status.get("safety_layer", {})) or {})
        active = bool(safety.get("safety_lock_active", False))
        reason = str(safety.get("safety_lock_reason", "") or "")
        reasons = [part.strip() for part in reason.split(",") if part.strip()]
        evidence = [
            {
                "code": item,
                "label": item.replace("_", " ").title(),
                "value": "not_reported",
                "detail": "Safety layer reported this active lock reason.",
            }
            for item in reasons
        ]
        checklist = []
        if active:
            checklist = [
                {"code": "manual_operator_unlock_required", "label": "Manual operator review is required before unlock.", "status": "blocked"},
                {"code": "cooldown_expired", "label": "Cooldown must be expired or inactive.", "status": "review"},
                {"code": "no_active_hard_blocker", "label": "No hard market, timing, broker, or risk blocker may remain.", "status": "review"},
                {"code": "volatility_back_to_acceptable", "label": "Volatility must be acceptable for new entries.", "status": "review"},
                {"code": "next_setup_meets_high_confluence", "label": "The next setup must meet high-confluence normal gates.", "status": "review"},
            ]
        return {
            "enabled": True,
            "active": active,
            "status": "locked" if active else "clear",
            "reason": reason,
            "evidence": evidence,
            "recovery_checklist": checklist,
            "separate_from_learning": {
                "pattern_ban": False,
                "note": "Safety lock protects the account. It is separate from loss-pattern learning and does not permanently ban a setup family.",
            },
            "policy_note": "Safety-lock review is read-only observability.",
        }

    def _build_learning_lanes_reply(self, query: str, *, mode: str = "standard") -> dict[str, Any]:
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        try:
            learning_state = dict(self.source.load_learning_state().get("state", {}) or {})
        except Exception:
            learning_state = {}
        lanes = dict(status.get("learning_lanes", {}) or {})
        lane_rows = dict(lanes.get("lanes", {}) or {})
        if not lane_rows:
            lane_rows = self._derive_learning_lanes_for_answer(status, runtime, learning_state)
        active = [name for name, row in lane_rows.items() if bool(dict(row or {}).get("active", False))]
        active_count = len(active)
        remember = dict(lane_rows.get("remember", {}) or {})
        remember_evidence = dict(remember.get("evidence", {}) or {})
        caution = dict(lane_rows.get("bounded_caution", {}) or {})
        caution_evidence = dict(caution.get("evidence", {}) or {})
        proposals = dict(lane_rows.get("propose_improvements", {}) or {})
        proposal_evidence = dict(proposals.get("evidence", {}) or {})
        missing = [str(x) for x in list(caution_evidence.get("missing_confirmations", []) or [])[:3] if str(x)]
        lessons = int(remember_evidence.get("pattern_lessons", 0) or 0)
        outcomes = int(remember_evidence.get("recent_outcomes", 0) or 0)
        proposal_total = int(proposal_evidence.get("total", 0) or 0)
        if str(mode or "").strip().lower() == "voice":
            result = (
                f"Yes. Atlas is studying continuously while the stack is running: {active_count}/6 learning lanes are active. "
                "It observes, classifies, remembers, explains, and proposes review items, but it cannot change strategy, risk, or execution by itself."
            )
        else:
            result = (
                f"Yes. Atlas studies continuously while the live stack is running: {active_count}/6 learning lanes are active.\n\n"
                "Current learning lanes:\n"
                f"- Observe: {self._lane_status_text(lane_rows, 'observe')}\n"
                f"- Classify: {self._lane_status_text(lane_rows, 'classify')}\n"
                f"- Remember: {self._lane_status_text(lane_rows, 'remember')} ({lessons} pattern lessons, {outcomes} outcomes)\n"
                f"- Explain: {self._lane_status_text(lane_rows, 'explain')}\n"
                f"- Bounded caution: {self._lane_status_text(lane_rows, 'bounded_caution')}"
                + (f" (missing: {', '.join(missing)})" if missing else "")
                + f"\n- Propose improvements: {self._lane_status_text(lane_rows, 'propose_improvements')} ({proposal_total} proposals)\n\n"
                "Boundary:\n"
                "- Learning can observe, classify, remember, explain, apply only already-allowed bounded caution, and propose improvements for review.\n"
                "- It cannot silently rewrite strategy logic, risk policy, execution policy, or force trades."
            )
        return {
            "mode": str(mode or "standard"),
            "intent": "learning_status",
            "reasoning_mode": "LEARNING_LANES",
            "acknowledge": "Learning status checked.",
            "understanding": "Interpreted as a question about Atlas learning/studying continuously.",
            "data_sources": ["current status", "runtime flags", "learning memory", "learning lanes"],
            "result": result,
            "approval": "Read-only answer. No strategy, risk, or execution changes made.",
            "response_kind": "question_answer",
            "selected_domain": "learning_state",
            "confidence": 0.94,
            "learning_lanes": {
                "active_lanes": active_count,
                "lanes": lane_rows,
                "policy_note": "Learning status is explanatory only; it does not grant execution authority.",
            },
        }

    @staticmethod
    def _lane_status_text(lanes: dict[str, Any], name: str) -> str:
        row = dict(lanes.get(name, {}) or {})
        text = str(row.get("status", "waiting") or "waiting").replace("_", " ")
        return text

    @staticmethod
    def _derive_learning_lanes_for_answer(
        status: dict[str, Any],
        runtime: dict[str, Any],
        learning_state: dict[str, Any],
    ) -> dict[str, Any]:
        loss_memory = dict(learning_state.get("loss_memory", {}) or {})
        lessons = dict(loss_memory.get("pattern_lessons", {}) or {})
        outcomes = list(loss_memory.get("recent_outcomes", []) or [])
        visual = dict(status.get("visual_chart_read", {}) or {})
        market = dict(status.get("market", {}) or {})
        recent_close = list(market.get("recent_close", []) or [])
        blocker = dict(status.get("structured_blocker", {}) or status.get("blocker_classification", {}) or {})
        critic = dict(status.get("learning_critic", {}) or {})
        loss_review = dict(status.get("loss_learning_review", {}) or {})
        self_improvement = dict(status.get("self_improvement_state", {}) or dict(status.get("system_health", {}) or {}).get("self_improvement", {}) or {})
        proposal_count = int(self_improvement.get("proposal_count", self_improvement.get("approval_pending_count", 0)) or 0)
        observe_active = bool(visual.get("available", False)) or bool(recent_close) or bool(status.get("updated_at"))
        classify_active = bool(blocker) or bool(status.get("decision_trace")) or bool(status.get("decision_analysis"))
        remember_active = bool(lessons) or bool(outcomes)
        explain_active = bool(status.get("decision_explanation")) or bool(status.get("message")) or bool(loss_review)
        caution_active = bool(critic.get("applied", False)) or bool(dict(status.get("loss_learning", {}) or {}).get("applied", False))
        return {
            "observe": {
                "active": observe_active,
                "status": "active" if observe_active else "waiting",
                "evidence": {"recent_close_points": len(recent_close), "visual_available": bool(visual.get("available", False))},
            },
            "classify": {
                "active": classify_active,
                "status": "active" if classify_active else "waiting",
                "evidence": {"blocker_code": str(blocker.get("code", "") or ""), "final_action": str(status.get("action", "") or "")},
            },
            "remember": {
                "active": remember_active,
                "status": "active" if remember_active else "waiting",
                "evidence": {"pattern_lessons": len(lessons), "recent_outcomes": len(outcomes)},
            },
            "explain": {
                "active": explain_active,
                "status": "active" if explain_active else "waiting",
                "evidence": {"summary": str(dict(status.get("decision_explanation", {}) or {}).get("summary", status.get("message", "")) or "")},
            },
            "bounded_caution": {
                "active": caution_active,
                "status": "active" if caution_active else "watching",
                "evidence": {
                    "learning_critic_status": str(critic.get("status", "watching") or "watching"),
                    "missing_confirmations": list(loss_review.get("current_missing_confirmations", []) or [])[:8],
                },
            },
            "propose_improvements": {
                "active": bool(proposal_count),
                "status": "active" if proposal_count else "waiting",
                "evidence": {"total": proposal_count},
            },
        }

    def _build_system_status_reply(self, query: str, *, intent: str = "system_status_query", mode: str = "standard") -> dict[str, Any]:
        q = str(query or "")
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        execution_state = self.execution.snapshot()
        operator_health = dict(status.get("operator_health", {}) or runtime.get("operator_health", {}) or {})
        system_health = dict(status.get("system_health", {}) or {})
        backend_running = bool(status.get("updated_at")) and not bool((status.get("freshness_runtime", {}) or {}).get("is_stale", False))
        frontend_mode = str(operator_health.get("frontend_mode", runtime.get("ui_desktop_window", {}).get("frontend_mode", "")) or "unknown")
        desktop_ready = bool(operator_health.get("desktop_window_ready", runtime.get("ui_desktop_window", {}).get("desktop_window_ready", False)))
        exec_mode = str(operator_health.get("effective_mode", system_health.get("effective_mode", "unknown")) or "unknown")
        blockers: list[str] = []
        if bool(system_health.get("config_mismatch_detected", False)):
            blockers.append("config mismatch detected")
        if bool(system_health.get("routing_risk_detected", False)):
            blockers.append("routing risk detected")
        if bool(system_health.get("stale_data_detected", False)):
            blockers.append("stale data detected")
        mode_label = str(execution_state.get("execution_environment_label", "Paper") or "Paper")
        action = str(status.get("action", "hold") or "hold").lower()
        reason = str(status.get("message", "") or "No high-confidence trade setup.")
        summary = (
            f"Backend looks {'running' if backend_running else 'not healthy enough to confirm'}. "
            f"Status: {action}. Reason: {reason}. Mode: {mode_label}."
        )
        if mode == "voice":
            summary = f"Status: {action}. Mode: {mode_label}."
        if blockers:
            summary += f" Current system blockers: {', '.join(blockers[:3])}."
        intent_candidates = [c.__dict__ for c in classify_intent_candidates(q)]
        evidence_separation = build_evidence_separation(status, {**runtime, "execution": execution_state})
        return {
            "mode": "standard",
            "intent": "status",
            "selected_domain": "runtime_status",
            "confidence": 0.9,
            "reasoning_mode": "SYSTEM_STATUS",
            "acknowledge": "System status ready.",
            "understanding": "Interpreted as a system/runtime question.",
            "data_sources": ["status payload", "runtime flags", "execution snapshot"],
            "result": summary,
            "approval": "Read-only system answer. No runtime changes were made.",
            "response_kind": "question_answer",
            "debug": {
                "intent_candidates": intent_candidates,
                "domain_candidates": [{"name": "runtime_status", "confidence": 0.9, "reason": "system_status_reply"}],
                "selected_plan": "status_overview",
                "evidence_used": ["status", "runtime", "execution_snapshot"],
                "evidence_chosen": ["status", "runtime", "execution_snapshot"],
                "relevance_result": "ok",
                "relevance_guard": "ok",
                "display_mode": mode_label,
            },
            "control_state": self.control.to_debug(),
            "execution_state": execution_state,
            "evidence_separation": evidence_separation,
        }

    def _build_explanation_reply(self, query: str, *, intent: str = "explanation_request") -> dict[str, Any]:
        _ = str(query or "")
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        execution_state = self.execution.snapshot()
        decision = dict(status.get("decision_explanation", {}) or {})
        decision_trace = dict(status.get("decision_trace", {}) or {})
        evidence_separation = dict(decision_trace.get("evidence_separation", {}) or {})
        if not evidence_separation:
            evidence_separation = build_evidence_separation(
                status,
                {**runtime, "execution": execution_state},
            )
        structured = build_structured_decision_analysis(status=status, runtime=runtime)
        regime = dict(structured.get("market_regime", status.get("market_regime", {})) or {})
        reversal = dict(structured.get("reversal_detection", status.get("reversal_detection", {})) or {})
        reason_codes = list(decision.get("reason_codes", []) or [])
        setup_reasons = list(decision.get("setup_quality_reasons", []) or [])
        message = str(status.get("message", "") or "")
        final_decision = str(decision.get("final_decision", "") or status.get("action", "unknown"))
        timing_discipline = dict(status.get("timing_discipline", {}) or {})
        timing_result = dict(timing_discipline.get("structured_result", {}) or timing_discipline)
        timing_code = str(timing_result.get("reason_code", "") or "").strip()
        timing_detail = str(timing_result.get("detail", "") or "").strip()
        timing_values = dict(timing_result.get("condition_values", {}) or {})
        timing_observed = dict(timing_values.get("observed", {}) or {})
        timing_bits = []
        for key in ("location", "momentum", "session"):
            value = str(timing_observed.get(key, "") or "").strip()
            if value:
                timing_bits.append(f"{key}={value}")
        trace_filters = dict(decision_trace.get("filters", {}) or {})
        trace_hard_blockers = list(trace_filters.get("hard_blockers", []) or [])
        trace_blocker_names = [str(dict(item or {}).get("name", "") or "") for item in trace_hard_blockers if str(dict(item or {}).get("name", "") or "")]
        blockers = trace_blocker_names[:3] or reason_codes[:3] or setup_reasons[:3]
        if timing_result and timing_result.get("passed") is False and (timing_code or timing_detail):
            code_text = f" ({timing_code})" if timing_code else ""
            value_text = f" Conditions: {', '.join(timing_bits)}." if timing_bits else ""
            result = f"The current decision is {final_decision}. Blocked by timing discipline{code_text}: {timing_detail.rstrip('.')}.{value_text}"
        elif blockers:
            blocker_text = ", ".join(str(x).replace("_", " ") for x in blockers)
            result = f"The current decision is {final_decision}. Main blockers: {blocker_text}."
        elif message:
            result = f"The current decision is {final_decision}. {message}"
        else:
            result = "The current decision is cautious because Atlas does not see enough aligned evidence to act safely."
        trace_candidate = dict(decision_trace.get("selected_candidate", {}) or {})
        if trace_candidate:
            result += (
                f" Selected candidate: {str(trace_candidate.get('side', 'flat'))} "
                f"via {str(trace_candidate.get('strategy', 'none'))} "
                f"at {float(trace_candidate.get('confidence', 0.0) or 0.0):.2f} confidence."
            )
        bias = dict(decision_trace.get("bias_breakdown", {}) or {})
        if bias:
            result += (
                f" Bias read: signal={str(bias.get('signal_bias', 'neutral'))}, "
                f"chart={str(bias.get('chart_structure_bias', 'neutral'))}, "
                f"higher timeframe={str(bias.get('higher_timeframe_bias', 'unknown'))}, "
                f"final={str(bias.get('final_trade_bias', 'neutral'))}."
            )
        if evidence_separation:
            result += (
                f" Evidence lanes: market={str(evidence_separation.get('market_bias', 'neutral'))}, "
                f"signal={str(evidence_separation.get('signal_direction', 'neutral'))}, "
                f"setup={str(evidence_separation.get('setup_readiness', 'unknown'))}, "
                f"confidence={float(evidence_separation.get('confidence', 0.0) or 0.0):.2f}/"
                f"{float(evidence_separation.get('threshold', 0.0) or 0.0):.2f}, "
                f"execution={str(evidence_separation.get('execution_permission', 'not_allowed'))}."
            )
        if regime:
            result += (
                f" Current regime reads as {str(regime.get('regime', 'unknown') or 'unknown')} "
                f"with {int(regime.get('regime_confidence', 0) or 0)}/100 confidence."
            )
        if bool(reversal.get("reversal_candidate", False)):
            result += " Reversal conditions are being watched."
            if not bool(reversal.get("confirmed_reversal", False)):
                result += " They are not confirmed because structure and momentum still need to agree."
        freshness = dict(runtime.get("freshness_runtime", {}) or {})
        if bool(freshness.get("is_stale", False)):
            result += " I would treat this explanation cautiously because the runtime context is stale."
        system_health = dict(status.get("system_health", {}) or {})
        safety_layer = dict(system_health.get("safety_layer", {}) or {})
        if bool(safety_layer.get("safety_lock_active", False)):
            recovery = dict(safety_layer.get("recovery", {}) or {})
            required_actions = list(recovery.get("required_actions", []) or [])
            result += " Safety lock is active as a protective control."
            if required_actions:
                result += " Recovery steps: " + ", ".join(str(x) for x in required_actions[:4]) + "."
            result += " To request unlock workflow, use /api/safety/unlock with explicit confirmation."
        return {
            "mode": "standard",
            "intent": intent,
            "selected_domain": "current_system_state",
            "reasoning_mode": "DECISION_EXPLANATION",
            "acknowledge": "Explanation ready.",
            "understanding": "Interpreted as a request to explain the current decision or blocker.",
            "data_sources": ["decision trace", "decision explanation", "evidence separation", "status payload", "runtime flags"],
            "result": result,
            "approval": "Read-only explanation. No execution or runtime state was changed.",
            "response_kind": "question_answer",
            "control_state": self.control.to_debug(),
            "execution_state": execution_state,
            "evidence_separation": evidence_separation,
        }

    def _build_general_question_reply(self, query: str, mode: str = "standard") -> dict[str, Any]:
        q = str(query or "").strip()
        ql = q.lower()
        execution_state = self.execution.snapshot()
        if any(phrase in ql for phrase in ("best self", "become your best self")):
            result = (
                "Becoming your best self usually takes longer than people want and more consistency than they expect. "
                "In practice, think in months and years, not days. The real path is steady habits, honest self-review, "
                "and staying with the work when progress feels slow."
            )
        elif q.endswith("?"):
            result = (
                "That depends on the goal, but the safest answer is usually steady, repeatable progress over time. "
                "Small disciplined improvements tend to last longer than short bursts of motivation."
            )
        else:
            result = (
                "I read that as a general question, not a trading one. My short answer is: steady, honest, repeatable "
                "work is what usually compounds into real improvement."
            )
        if mode == "voice":
            result = result.split(". ")[0].strip()
        return {
            "mode": "voice" if mode == "voice" else "standard",
            "intent": "general_question",
            "selected_domain": "general_operator_help",
            "reasoning_mode": "GENERAL_ASSISTANT",
            "acknowledge": "General answer ready.",
            "understanding": "Interpreted as a general question, not a trading or system request.",
            "data_sources": ["general assistant fallback", "operator interaction rules"],
            "result": result,
            "approval": "No system or trading action was taken.",
            "response_kind": "question_answer",
            "control_state": self.control.to_debug(),
            "execution_state": execution_state,
        }

    @staticmethod
    def _is_operator_guidance_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        markers = (
            "improve yourself",
            "how can you get better",
            "optimize",
            "how do we improve",
            "make this better",
            "improve atlas",
        )
        return any(m in q for m in markers)

    @staticmethod
    def _is_structured_trading_analysis_query(query: str) -> bool:
        return is_structured_trading_analysis_query(query)

    @staticmethod
    def _is_volatility_query(query: str) -> bool:
        q = str(query or "").strip().lower()
        markers = (
            "what volatility",
            "market too volatile",
            "volatility class",
            "volatility do you trade best",
            "learn about volatility",
            "volatility state",
        )
        return any(m in q for m in markers)

    def _build_volatility_reply(self, query: str, mode: str = "standard") -> dict[str, Any]:
        # purpose: give explicit operator answer for volatility state and trading impact.
        # what it does: reads volatility intelligence fields from status and formats actionable explanation.
        _ = str(query or "")
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        execution_state = self.execution.snapshot()
        vi = dict(status.get("volatility_intelligence", {}) or {})
        klass = str(vi.get("class", "normal_volatility") or "normal_volatility")
        conf = float(vi.get("confidence", 0.0) or 0.0)
        response_rule = str(vi.get("response_rule", "normal_analysis") or "normal_analysis")
        explanation = str(vi.get("explanation", "") or "Volatility classifier is active.")
        blockers = list(vi.get("blockers", []) or [])
        tags = list(vi.get("tags", []) or [])
        perf = dict(status.get("volatility_learning_performance", {}) or {})
        class_perf = dict((perf.get("classes", {}) or {}).get(klass, {}) or {})
        trade_impact = "Execution still requires ExecutionGate approval."
        if response_rule in {"volatility_hard_block", "news_hard_block", "spread_slippage_block"}:
            trade_impact = "New entries are currently blocked by volatility safety controls."
        elif response_rule == "reduce_risk_require_higher_confluence":
            trade_impact = "Risk is reduced and confluence requirements are stricter."
        elif response_rule == "prefer_no_trade_reduce_confidence":
            trade_impact = "No-trade is preferred until structure and momentum become clean."
        elif response_rule == "allow_continuation_if_aligned":
            trade_impact = "Continuation candidates are allowed only when structure, momentum, spread, and news align."
        result = (
            f"Current volatility class: {klass}. Confidence: {int(round(conf * 100))}/100. "
            f"Response rule: {response_rule}. {explanation} {trade_impact}"
        )
        if blockers:
            result += " Active blockers: " + ", ".join(str(x) for x in blockers[:4]) + "."
        if tags:
            result += " Tags: " + ", ".join(str(x) for x in tags[:5]) + "."
        if class_perf:
            result += (
                f" Learned performance in this class: win rate {float(class_perf.get('win_rate', 0.0)) * 100:.1f}%, "
                f"expectancy {float(class_perf.get('expectancy_r', 0.0)):+.2f}R."
            )
        if mode == "voice":
            result = result.split(". ")[0] + "."
        return {
            "mode": "voice" if mode == "voice" else "standard",
            "intent": "volatility_state_query",
            "selected_domain": "market_volatility",
            "reasoning_mode": "VOLATILITY_EXPLANATION",
            "acknowledge": "Volatility state ready.",
            "understanding": "Interpreted as a volatility condition question.",
            "data_sources": ["status.volatility_intelligence", "status.volatility_learning_performance", "execution snapshot"],
            "result": result,
            "approval": "Analysis-only response. Execution remains gated.",
            "response_kind": "question_answer",
            "control_state": self.control.to_debug(),
            "execution_state": execution_state,
            "volatility": {
                "class": klass,
                "confidence": conf,
                "response_rule": response_rule,
                "blockers": blockers,
                "tags": tags,
            },
            "debug": {
                "routed_handler": "volatility_reply",
                "volatility_class": klass,
                "volatility_response_rule": response_rule,
                "volatility_blockers": blockers,
                "volatility_tags": tags,
                "runtime_has_vi": bool(vi),
                "runtime_has_perf": bool(perf),
            },
        }

    @staticmethod
    def _clamp_confidence_score(raw: Any) -> int:
        try:
            score = float(raw)
            if score <= 1.0:
                score = score * 100.0
            return int(max(0, min(100, round(score))))
        except Exception:
            return 0

    def _build_structured_trading_analysis_reply(self, query: str) -> dict[str, Any]:
        _ = str(query or "")
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        snap = self.execution.snapshot(runtime_flags=runtime)
        reply = build_structured_trading_analysis_reply(status=status, runtime=runtime, execution_state=snap)
        reply["control_state"] = self.control.to_debug()
        return reply

    def _build_operator_guidance_reply(self, query: str) -> dict[str, Any]:
        snap = self.execution.snapshot()
        options = [
            {
                "category": "system_status_analysis",
                "query": "show guarded demo readiness",
                "description": "Review gate, kill switch, freshness visibility, and current safety posture.",
            },
            {
                "category": "trading_performance_review",
                "query": "show last demo session summary",
                "description": "Inspect guarded attempts, blocked reasons, and recent execution outcomes.",
            },
            {
                "category": "safety_audit_suggestions",
                "query": "run guarded demo checklist",
                "description": "Run a read-only guarded workflow check and identify any missing prerequisites.",
            },
            {
                "category": "demo_workflow_guidance",
                "query": "check cross-surface consistency",
                "description": "Verify Ask/runtime execution surfaces are aligned before or after a demo run.",
            },
        ]
        return {
            "mode": "standard",
            "intent": "operator_guidance",
            "reasoning_mode": "GUIDANCE",
            "acknowledge": "Guidance ready.",
            "understanding": "Interpreted as a meta/operator-improvement request, not a trading command.",
            "data_sources": ["execution snapshot", "guarded demo workflow"],
            "result": (
                "This is a read-only guidance request. I did not execute or modify trading/runtime state. "
                "Choose one of the guided review options below."
            ),
            "guidance_options": options,
            "meta_intent": {
                "query": str(query or ""),
                "classified_as": "operator_guidance",
                "action_taken": "none_read_only",
            },
            "approval": "Execution remains command-gated and approval-gated.",
            "response_kind": "question_answer",
            "control_state": self.control.to_debug(),
            "execution_state": snap,
        }

    def _maybe_auto_execute_recommendation(
        self,
        *,
        query: str,
        status: dict[str, Any],
        runtime: dict[str, Any],
        selected_intent: str,
        selected_domain: str,
    ) -> dict[str, Any] | None:
        intent_key = str(selected_intent or "").strip().lower()
        domain_key = str(selected_domain or "").strip().lower()
        # Keep auto-execution bounded to recommendation-like paths only.
        if not (
            "recommend" in intent_key
            or intent_key in {"trading_decision", "trade_recommendation"}
            or "recommend" in domain_key
        ):
            return None

        action = str(status.get("action", "") or "").strip().lower()
        if action not in {"buy", "sell", "long", "short", "place_trade"}:
            return None

        chosen = dict(status.get("chosen", {}) or {})
        confidence = float(chosen.get("confidence", status.get("confidence", 0.0)) or 0.0)
        threshold = float(status.get("min_confidence_threshold", runtime.get("min_confidence_threshold", 0.0)) or 0.0)
        if threshold > 0.0 and confidence < threshold:
            return {"ok": False, "reason": "confidence_below_threshold", "confidence": confidence, "threshold": threshold}

        execution_snapshot = self.execution.snapshot()
        pilot = dict(execution_snapshot.get("tiny_live_pilot", {}) or {})
        instrument = str(
            status.get("instrument")
            or status.get("symbol")
            or runtime.get("symbol")
            or pilot.get("allowed_instrument")
            or "XAUUSD"
        ).strip().upper()
        setup_family = str(
            status.get("setup_family")
            or chosen.get("setup_family")
            or runtime.get("setup_family")
            or pilot.get("allowed_setup_family")
            or "continuation"
        ).strip().lower()
        size = float(
            status.get("size")
            or chosen.get("size")
            or runtime.get("size")
            or pilot.get("size_cap")
            or 0.01
        )
        session = str(
            execution_snapshot.get("session_control", {}).get("current_session")
            or runtime.get("session")
            or "none"
        ).strip().lower()
        evidence_strength = float(
            status.get("evidence_strength")
            or chosen.get("evidence_strength")
            or runtime.get("evidence_strength")
            or 0.0
        )
        side = str(chosen.get("side") or status.get("action") or "").strip().lower()
        if side not in {"buy", "sell"}:
            side = ""
        strategy = str(chosen.get("strategy") or status.get("strategy") or setup_family).strip()
        stop_loss_points = float(chosen.get("stop_loss_points") or status.get("stop_loss_points") or 0.0)
        take_profit_points = float(chosen.get("take_profit_points") or status.get("take_profit_points") or 0.0)
        signal_marker = str(
            status.get("signal_id")
            or status.get("cycle_id")
            or status.get("updated_at")
            or runtime.get("status_updated_at")
            or ""
        ).strip()
        if not signal_marker:
            sig_input = f"{query}|{action}|{instrument}|{setup_family}|{size:.4f}|{confidence:.4f}|{evidence_strength:.4f}|{status.get('message', '')}"
            signal_marker = hashlib.sha1(sig_input.encode("utf-8")).hexdigest()

        risk_policy = dict(self.knowledge.load("risk_policy") or {})
        out = self.execution.attempt_auto_execution_from_signal(
            signal_id=signal_marker,
            action_type="place_trade",
            environment=str(execution_snapshot.get("execution_environment", "paper") or "paper"),
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            confidence=confidence,
            evidence_strength=evidence_strength,
            side=side,
            strategy=strategy,
            stop_loss_points=stop_loss_points,
            take_profit_points=take_profit_points,
            runtime_flags=runtime,
            risk_policy=risk_policy,
            dry_run=False,
        )
        return dict(out or {})

    def _sync_forced_flat_control_item(
        self,
        runtime_flags: dict[str, Any] | None,
        execution_state: dict[str, Any] | None,
    ) -> tuple[Any | None, str]:
        snap = dict(execution_state or {})
        sc = dict(snap.get("session_control", {}) or {})
        open_pos = int((runtime_flags or {}).get("open_positions", 0) or 0)
        block_reason = str(sc.get("session_block_reason", "near_session_close_no_new_trades") or "near_session_close_no_new_trades")
        close_block = block_reason in {"near_session_close_no_new_trades", "early_close_no_new_trades", "insufficient_time_to_close"}
        forced = bool(sc.get("forced_flat_required", False)) or (close_block and open_pos > 0)
        if not forced:
            return None, "not_required"

        mins = int(sc.get("minutes_to_session_close", -1) or -1)
        cal_status = str(sc.get("session_calendar_status", "unknown") or "unknown")
        session_name = str(sc.get("current_session", "none") or "none")
        tz = str(sc.get("session_timezone", "UTC") or "UTC")
        urgency = "immediate_review" if mins >= 0 and mins <= 5 else ("urgent" if mins >= 0 and mins <= 15 else "high")
        priority = 1 if urgency == "immediate_review" else 2
        confidence = 0.95 if open_pos > 0 else 0.8
        dedupe_key = f"{session_name}|{tz}|{cal_status}|{block_reason}"
        summary = (
            f"forced-flat review ({urgency}): {session_name} closes in {mins}m"
            if mins >= 0
            else f"forced-flat review ({urgency}): {session_name} close window active"
        )
        detailed_reason = (
            "Session controls require flattening before close. "
            f"Reason={block_reason}. Session={session_name}. Time_to_close={mins}m. "
            f"Open_positions={open_pos}. Calendar={cal_status}."
        )
        supporting = [
            f"session={session_name}",
            f"timezone={tz}",
            f"time_to_close_minutes={mins}",
            f"open_positions={open_pos}",
            f"calendar_status={cal_status}",
            f"forced_flat_reason={block_reason}",
        ]
        constraints = [
            "no autonomous forced close",
            "operator approval/review required",
            "no execution bypass",
            "preserve session/day-risk discipline",
        ]
        item, action = self.control.upsert_forced_flat_review(
            summary=summary,
            detailed_reason=detailed_reason,
            supporting=supporting,
            constraints=constraints,
            confidence=confidence,
            priority=priority,
            related_domain="runtime_status",
            source_module="session_control",
            priority_reason=f"forced-flat {urgency} due to session-close discipline",
            trigger_conditions=[
                f"forced_flat_required=true",
                f"session_name={session_name}",
                f"time_to_close_minutes={mins}",
                f"block_reason={block_reason}",
                f"urgency={urgency}",
                f"calendar_status={cal_status}",
            ],
            approval_level="approval_required",
            approval_reason="Flattening decisions are operator-controlled and must be explicitly reviewed.",
            dedupe_key=dedupe_key,
        )
        return item, action

    def _sync_order_reconciliation_items(self, execution_state: dict[str, Any] | None) -> list[Any]:
        snap = dict(execution_state or {})
        pending = list(self.execution.pending_reconciliation_orders(limit=20))
        created_or_updated: list[Any] = []
        for row in pending:
            request_id = str(row.get("internal_request_id", "") or "unknown")
            broker_order_id = str(row.get("broker_order_id", "") or "unknown")
            internal_status = str(row.get("status", "unknown") or "unknown")
            recon_reason = str(row.get("rejection_reason", "") or "reconciliation_required")
            unresolved_since = float(row.get("submitted_at", 0.0) or 0.0)
            env = str(row.get("environment", "live") or "live")
            summary = f"order reconciliation review: {request_id} ({internal_status})"
            detailed_reason = (
                f"Order state is unresolved and requires explicit operator reconciliation. "
                f"internal_request_id={request_id}, broker_order_id={broker_order_id}, "
                f"internal_status={internal_status}, reason={recon_reason}."
            )
            supporting = [
                f"internal_request_id={request_id}",
                f"broker_order_id={broker_order_id}",
                f"current_internal_status={internal_status}",
                f"last_known_broker_status={snap.get('last_live_order_status', 'unknown')}",
                f"reconciliation_reason={recon_reason}",
                f"unresolved_since={unresolved_since}",
                f"affected_environment={env}",
            ]
            constraints = [
                "no auto-resubmission",
                "no auto-assumed fill/cancel",
                "query-only reconciliation actions",
                "operator-controlled resolution",
            ]
            item, _ = self.control.upsert_order_reconciliation_review(
                summary=summary,
                detailed_reason=detailed_reason,
                supporting=supporting,
                constraints=constraints,
                confidence=0.9,
                priority=1,
                related_domain="order_reconciliation",
                source_module="reconciliation",
                priority_reason="unresolved live order state requires immediate operator reconciliation review",
                trigger_conditions=[
                    "reconciliation_required=true",
                    f"internal_request_id={request_id}",
                    f"current_internal_status={internal_status}",
                    f"affected_environment={env}",
                ],
                approval_level="approval_required",
                approval_reason="Reconciliation can affect live order interpretation and requires explicit operator review.",
                dedupe_key=f"{request_id}|{broker_order_id}",
            )
            if item is not None:
                created_or_updated.append(item)
        return created_or_updated

    def _sync_orphan_order_items(self, execution_state: dict[str, Any] | None) -> list[Any]:
        _ = dict(execution_state or {})
        rows = self.execution.list_orphan_orders(include_reviewed=False, limit=20)
        created_or_updated: list[Any] = []
        for row in rows:
            bid = str(row.get("broker_order_id", "") or "unknown")
            summary = f"order orphan review: {bid}"
            detailed_reason = (
                f"Broker order exists but Atlas has no internal ownership record. "
                f"broker_order_id={bid}, instrument={row.get('instrument', 'n/a')}, "
                f"side={row.get('side', 'n/a')}, size={row.get('size', 0.0)}, broker_status={row.get('broker_status', 'unknown')}."
            )
            supporting = [
                f"broker_order_id={bid}",
                f"instrument={row.get('instrument', 'n/a')}",
                f"side={row.get('side', 'n/a')}",
                f"size={row.get('size', 0.0)}",
                f"broker_status={row.get('broker_status', 'unknown')}",
                f"detected_at={row.get('detected_at', 0.0)}",
                "reason=unknown_to_atlas",
            ]
            constraints = [
                "do not auto-cancel",
                "do not auto-assume ownership",
                "operator review required",
                "audit trail required",
            ]
            item, _ = self.control.upsert_order_orphan_review(
                summary=summary,
                detailed_reason=detailed_reason,
                supporting=supporting,
                constraints=constraints,
                confidence=0.92,
                priority=1,
                related_domain="order_reconciliation",
                source_module="broker_adapter",
                priority_reason="unknown broker order requires immediate operator orphan review",
                trigger_conditions=[f"broker_order_id={bid}", "reason=unknown_to_atlas"],
                approval_level="approval_required",
                approval_reason="Orphan orders are sensitive live-state mismatches and must be operator-reviewed.",
                dedupe_key=bid,
            )
            if item is not None:
                created_or_updated.append(item)
        return created_or_updated

    @staticmethod
    def _build_runbook_checklist_text(snap: dict[str, Any], runtime: dict[str, Any]) -> str:
        pilot = dict(snap.get("tiny_live_pilot", {}) or {})
        sc = dict(snap.get("session_control", {}) or {})
        open_pos = int(runtime.get("open_positions", 0) or 0)
        pre_lines = [
            f"- system status: {'OK' if bool(snap.get('execution_gate_enabled', False)) and not bool(snap.get('kill_switch_active', False)) else 'BLOCKED'} (gate={bool(snap.get('execution_gate_enabled', False))}, kill_switch={bool(snap.get('kill_switch_active', False))})",
            f"- live pilot enabled+approved: {'OK' if bool(pilot.get('enabled', False)) and bool(pilot.get('approved', False)) else 'BLOCKED'}",
            f"- session active: {'OK' if bool(sc.get('session_active', False)) else 'BLOCKED'}",
            f"- reconciliation clear: {'OK' if not bool(snap.get('reconciliation_required', False)) else 'BLOCKED'}",
            f"- orphan orders clear: {'OK' if int(snap.get('orphan_order_count', 0) or 0) == 0 else 'REVIEW'} (count={int(snap.get('orphan_order_count', 0) or 0)})",
            f"- risk counters reset check: {'OK' if int(snap.get('tiny_live_pilot_attempts_today', 0) or 0) == 0 else 'REVIEW'} (attempts={int(snap.get('tiny_live_pilot_attempts_today', 0) or 0)})",
            f"- no forced-flat pending: {'OK' if not bool(snap.get('forced_flat_pending', False)) else 'REVIEW'}",
        ]
        stop_flags: list[str] = []
        if bool(sc.get("daily_stop_triggered", False)):
            stop_flags.append("daily loss cap hit")
        if int(snap.get("tiny_live_pilot_consecutive_losses", 0) or 0) >= int(pilot.get("max_consecutive_losses", 9999) or 9999):
            stop_flags.append("consecutive loss limit hit")
        if bool(snap.get("forced_flat_pending", False)) and str(snap.get("forced_flat_urgency", "none")) in {"urgent", "immediate_review"}:
            stop_flags.append(f"forced-flat {snap.get('forced_flat_urgency', 'urgent')}")
        if bool(pilot.get("paused", False)):
            stop_flags.append(f"pilot paused ({pilot.get('pause_reason', 'unknown')})")
        during_lines = [
            f"- trades used / limit: {int(snap.get('tiny_live_pilot_trade_count', 0) or 0)} / {int(pilot.get('max_trades_per_day', 0) or 0)}",
            f"- daily loss used / cap: {float(snap.get('tiny_live_pilot_daily_loss_used', 0.0) or 0.0):.3f} / {float(pilot.get('daily_loss_cap', 0.0) or 0.0):.3f}",
            f"- consecutive losses: {int(snap.get('tiny_live_pilot_consecutive_losses', 0) or 0)} / {int(pilot.get('max_consecutive_losses', 0) or 0)}",
            f"- current session + time to close: {sc.get('current_session', 'none')} / {int(sc.get('minutes_to_session_close', -1) or -1)}m",
            f"- pilot paused: {bool(pilot.get('paused', False))} ({pilot.get('pause_reason', 'none')})",
            f"- stop conditions: {', '.join(stop_flags) if stop_flags else 'none active'}",
        ]
        post_lines = [
            f"- all positions closed: {open_pos <= 0} (open_positions={open_pos})",
            f"- pending live orders: {int(snap.get('pending_live_orders_count', 0) or 0)}",
            f"- reconciliation required: {bool(snap.get('reconciliation_required', False))}",
            f"- orphan orders: {int(snap.get('orphan_order_count', 0) or 0)}",
            (
                f"- today summary: attempts={int(snap.get('tiny_live_pilot_attempts_today', 0) or 0)} "
                f"last={snap.get('last_live_order_status', 'none')} "
                f"pauses={bool(pilot.get('paused', False))} "
                f"anomalies={'yes' if bool(snap.get('reconciliation_required', False)) or int(snap.get('orphan_order_count', 0) or 0) > 0 else 'no'}"
            ),
        ]
        return (
            "Pilot runbook checklist:\n"
            "Pre-session:\n" + "\n".join(pre_lines) + "\n"
            "During session:\n" + "\n".join(during_lines) + "\n"
            "Post-session:\n" + "\n".join(post_lines)
        )

    @staticmethod
    def _build_guarded_demo_readiness_payload(snap: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
        freshness = dict(runtime.get("freshness_runtime", {}) or {})
        is_stale = bool(freshness.get("is_stale", False))
        status_age_ms = freshness.get("status_age_ms", None)
        if status_age_ms is None:
            freshness_state = "unknown_runtime_context"
        else:
            freshness_state = "stale" if is_stale else "healthy"
        last_exec = dict(snap.get("last_execution_result", {}) or {})
        return {
            "execution_path_source": "execution_gate_guarded",
            "ask_query_mode": "read_only_by_default",
            "explicit_action_required": True,
            "freshness_state": freshness_state,
            "freshness_runtime": {
                "is_stale": is_stale,
                "status_age_ms": status_age_ms,
                "max_freshness_age_ms": runtime.get("max_freshness_age_ms", freshness.get("max_freshness_age_ms")),
            },
            "last_block_reason": str(
                snap.get("tiny_live_pilot_last_block_reason", "")
                or snap.get("automation_last_block_reason", "")
                or last_exec.get("reason", "")
                or ""
            ),
            "last_execution_result": last_exec,
            "last_request_origin": str((snap.get("automation_last_event", {}) or {}).get("request_origin", "")),
            "state_consistency_hint": "Use /api/runtime.execution and Ask Atlas execution_state from same shared gate instance.",
        }

    @staticmethod
    def _build_demo_session_review_payload(
        *, snap: dict[str, Any], runtime: dict[str, Any], audit_rows: list[dict[str, Any]], recent_limit: int = 8
    ) -> dict[str, Any]:
        attempt_events = {
            "execution_success",
            "execution_dry_run",
            "execution_failed",
            "execution_blocked",
            "automation_execution_success",
            "automation_execution_failed",
            "automation_execution_blocked",
        }
        blocked_events = {"execution_blocked", "execution_failed", "automation_execution_failed", "automation_execution_blocked"}
        allowed_events = {"execution_success", "execution_dry_run", "automation_execution_success"}
        safety_events = {
            "kill_switch_activated",
            "kill_switch_deactivated",
            "sleep_mode_activated",
            "sleep_mode_deactivated",
            "automation_paused",
            "session_rule_blocked",
            "forced_flat_required",
            "orphan_scan_completed",
        }

        attempts: list[dict[str, Any]] = []
        blocked_reason_counter: Counter[str] = Counter()
        freshness_reason_counter: Counter[str] = Counter()
        allowed_count = 0
        blocked_count = 0
        recent_safety: list[dict[str, Any]] = []

        for row in list(audit_rows or []):
            event = str(row.get("event", "") or "")
            details = dict(row.get("details", {}) or {})
            reason = str(details.get("reason", "") or "")
            req_id = str(row.get("request_id", "") or "")
            ts = float(row.get("ts", 0.0) or 0.0)
            if event in attempt_events:
                attempts.append({"ts": ts, "event": event, "request_id": req_id, "reason": reason})
                if event in allowed_events:
                    allowed_count += 1
                if event in blocked_events:
                    blocked_count += 1
                    if reason:
                        blocked_reason_counter[reason] += 1
                        if "freshness" in reason:
                            freshness_reason_counter[reason] += 1
            if event in safety_events:
                recent_safety.append({"ts": ts, "event": event, "details": details})

        freshness_runtime = dict(runtime.get("freshness_runtime", {}) or {})
        top_block_reasons = [{"reason": k, "count": v} for k, v in blocked_reason_counter.most_common(5)]
        freshness_incidents = [{"reason": k, "count": v} for k, v in freshness_reason_counter.most_common(5)]
        recent_blocked = [x for x in attempts if x.get("event") in blocked_events][-max(1, int(recent_limit)) :]
        recent_attempts = attempts[-max(1, int(recent_limit)) :]
        recent_safety = recent_safety[-max(1, int(recent_limit)) :]

        return {
            "attempt_count": len(attempts),
            "allowed_count": int(allowed_count),
            "blocked_count": int(blocked_count),
            "top_block_reasons": top_block_reasons,
            "freshness_incidents": freshness_incidents,
            "recent_attempts": recent_attempts,
            "recent_blocked_attempts": recent_blocked,
            "recent_safety_events": recent_safety,
            "last_execution_result": dict(snap.get("last_execution_result", {}) or {}),
            "last_auto_execution_result": dict(snap.get("last_auto_execution_result", {}) or {}),
            "runtime_freshness": {
                "is_stale": bool(freshness_runtime.get("is_stale", False)),
                "status_age_ms": freshness_runtime.get("status_age_ms", None),
                "max_freshness_age_ms": freshness_runtime.get("max_freshness_age_ms", runtime.get("max_freshness_age_ms", None)),
            },
        }

    @staticmethod
    def _is_mutating_request(query: str) -> bool:
        q = str(query or "").strip().lower()
        blocked = (
            "place a trade",
            "execute trade",
            "buy now",
            "sell now",
            "enable live",
            "switch to live",
            "disable guard",
            "change threshold",
            "modify runtime",
            "set execute_trades",
            "turn on live trading",
        )
        return any(k in q for k in blocked)

    def _handle_control_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = str(cmd.get("action", ""))
        item_id = int(cmd.get("item_id", 0) or 0)
        if action == "approve_all":
            updated_items = self.control.set_all_pending_status("approved")
            for updated in list(updated_items):
                if updated.type == "extended_paper_validation":
                    self.control.transition_workflow(
                        int(updated.id),
                        "under_observation",
                        note="approved_for_extended_paper_observation",
                    )
                elif updated.type == "paper_tuning_promotion_candidate":
                    self.control.transition_workflow(
                        int(updated.id),
                        "validated_in_paper",
                        note="approved_as_validated_paper_candidate",
                    )
            count = len(updated_items)
            if count == 0:
                text = "No pending control items were available to approve."
            else:
                high_risk = sum(
                    1
                    for item in updated_items
                    if item.approval_level == "high_risk_requires_explicit_confirmation"
                )
                text = f"Approved {count} pending control item{'s' if count != 1 else ''}."
                if high_risk:
                    text += f" High-risk approvals included: {high_risk}."
            return {
                "mode": "standard",
                "intent": "control_action",
                "result": text,
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action in {"approve", "reject", "defer"}:
            status_map = {"approve": "approved", "reject": "rejected", "defer": "deferred"}
            updated = self.control.set_status(item_id, status_map[action])
            if updated is None:
                text = f"Control item {item_id} not found or not pending."
            else:
                if action == "approve":
                    if updated.type == "extended_paper_validation":
                        transitioned = self.control.transition_workflow(item_id, "under_observation", note="approved_for_extended_paper_observation")
                        if transitioned is not None:
                            updated = transitioned
                    elif updated.type == "paper_tuning_promotion_candidate":
                        transitioned = self.control.transition_workflow(item_id, "validated_in_paper", note="approved_as_validated_paper_candidate")
                        if transitioned is not None:
                            updated = transitioned
                text = f"Item {item_id} is now {updated.status}."
            return {
                "mode": "standard",
                "intent": "control_action",
                "result": text,
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_pending":
            pending = self.control.pending()
            lines = [f"- item {i.id} [{i.type}] p{i.priority}: {i.summary}" for i in pending[:10]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Pending items:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_approved":
            items = self.control.by_status("approved", limit=10)
            lines = [f"- item {i.id} [{i.type}] p{i.priority}: {i.summary}" for i in items]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Recently approved:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_rejected":
            items = self.control.by_status("rejected", limit=10)
            lines = [f"- item {i.id} [{i.type}] p{i.priority}: {i.summary}" for i in items]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Rejected items:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_high_priority":
            pending = [i for i in self.control.pending() if i.priority <= 2]
            lines = [f"- item {i.id} [{i.type}] p{i.priority}: {i.summary}" for i in pending[:10]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "High-priority suggestions:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_needs_approval":
            items = [i for i in self.control.pending() if i.approval_level in {"approval_required", "high_risk_requires_explicit_confirmation"}]
            lines = [f"- item {i.id} [{i.type}] {i.approval_level} p{i.priority}: {i.summary}" for i in items[:12]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Items requiring approval:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_high_risk":
            items = [i for i in self.control.pending() if i.approval_level == "high_risk_requires_explicit_confirmation"]
            lines = [f"- item {i.id} [{i.type}] p{i.priority}: {i.summary}" for i in items[:12]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "High-risk items:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
            }
        if action == "show_active_workflows":
            state = self.control.to_debug()
            under_obs = list(state.get("under_observation", []) or [])
            validated = list(state.get("validated_in_paper", []) or [])
            lines = [f"- item {x.get('id')} [{x.get('type')}] under_observation: {x.get('summary')}" for x in under_obs[:8]]
            lines += [f"- item {x.get('id')} [{x.get('type')}] validated_in_paper: {x.get('summary')}" for x in validated[:8]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Active advanced workflows:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": state,
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_under_observation":
            state = self.control.to_debug()
            under_obs = list(state.get("under_observation", []) or [])
            lines = [f"- item {x.get('id')} [{x.get('type')}] p{x.get('priority')}: {x.get('summary')}" for x in under_obs[:12]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Under observation:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": state,
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_validated_paper":
            state = self.control.to_debug()
            validated = list(state.get("validated_in_paper", []) or [])
            lines = [f"- item {x.get('id')} [{x.get('type')}]: {x.get('summary')}" for x in validated[:12]]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Validated in paper mode:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": state,
                "execution_state": self.execution.snapshot(),
            }
        if action == "show_promotion_review":
            items = [i for i in self.control.pending() if i.type in {"paper_tuning_promotion_candidate", "extended_paper_validation", "tuning_candidate"}]
            lines = [
                f"- item {i.id} [{i.type}] {i.approval_level}: {i.summary} | constraints: {', '.join(i.safety_constraints[:2]) or 'n/a'}"
                for i in items[:12]
            ]
            return {
                "mode": "standard",
                "intent": "control_list",
                "result": "Review before any promotion:\n" + ("\n".join(lines) if lines else "- none"),
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "explain":
            item = self.control.get(item_id)
            if item is None:
                text = f"Control item {item_id} not found."
            else:
                text = (
                    f"Item {item.id}: {item.summary}\n"
                    f"Type: {item.type}\n"
                    f"Status: {item.status}\n"
                    f"Source module: {item.source_module or 'n/a'}\n"
                    f"Related domain: {item.related_domain or 'n/a'}\n"
                    f"Approval level: {item.approval_level}\n"
                    f"Approval reason: {item.approval_reason or 'n/a'}\n"
                    f"Workflow state: {item.workflow_state}\n"
                    f"Reason: {item.detailed_reason}\n"
                    f"Priority reason: {item.priority_reason or 'n/a'}\n"
                    f"Evidence: {', '.join(item.supporting_evidence[:4]) or 'n/a'}\n"
                    f"Safety constraints: {', '.join(item.safety_constraints[:4]) or 'n/a'}"
                )
            return {
                "mode": "standard",
                "intent": "control_explain",
                "result": text,
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        if action == "why":
            item = self.control.get(item_id)
            if item is None:
                text = f"Control item {item_id} not found."
            else:
                text = (
                    f"Why item {item.id} was suggested:\n"
                    f"- Trigger conditions: {', '.join(item.trigger_conditions[:6]) or 'n/a'}\n"
                    f"- Relevant signals/evidence: {', '.join(item.supporting_evidence[:6]) or 'n/a'}\n"
                    f"- Domain context: {item.related_domain or 'n/a'}\n"
                    f"- Source module: {item.source_module or 'n/a'}\n"
                    f"- Priority: p{item.priority} ({item.priority_reason or 'operator review required'})\n"
                    f"- Approval: {item.approval_level} ({item.approval_reason or 'n/a'})"
                )
            return {
                "mode": "standard",
                "intent": "control_explain",
                "result": text,
                "response_kind": "control",
                "control_state": self.control.to_debug(),
                "execution_state": self.execution.snapshot(),
            }
        return {
            "mode": "standard",
            "intent": "control_action",
            "result": "Unsupported control command.",
            "response_kind": "control",
            "control_state": self.control.to_debug(),
            "execution_state": self.execution.snapshot(),
        }

    def execution_snapshot(self) -> dict[str, Any]:
        return self.execution.snapshot()

    def why_trace(self) -> dict[str, Any]:
        status = dict(self.source.load_status() or {})
        runtime = dict(self.source.load_runtime_flags() or {})
        return build_why_trace(status, runtime, self.execution.snapshot(runtime_flags=runtime))

    def guarded_demo_workflow_snapshot(self) -> dict[str, Any]:
        snap = self.execution.snapshot()
        runtime = dict(self.source.load_runtime_flags() or {})
        readiness = self._build_guarded_demo_readiness_payload(snap, runtime)
        return {
            "readiness": readiness,
            "last_execution_result": dict(snap.get("last_execution_result", {}) or {}),
            "last_auto_execution_result": dict(snap.get("last_auto_execution_result", {}) or {}),
            "freshness_runtime": dict(runtime.get("freshness_runtime", {}) or {}),
            "clean_shutdown_ready": not bool(snap.get("sleep_mode_active", False)) and not bool(snap.get("kill_switch_active", False)),
        }

    def _parse_execution_command(self, query: str) -> dict[str, Any] | None:
        q = str(query or "").strip().lower()
        if not q:
            return None
        if q in {"enable execution gate", "execution gate on"}:
            return {"action": "enable_gate"}
        if q in {"disable execution gate", "execution gate off"}:
            return {"action": "disable_gate"}
        if q in {"activate kill switch", "enable kill switch"}:
            return {"action": "kill_on"}
        if q in {"deactivate kill switch", "disable kill switch"}:
            return {"action": "kill_off"}
        if q in {"activate sleep mode", "atlas sleep mode on", "sleep safety mode on"}:
            return {"action": "sleep_on", "force_flat_review": False}
        if q in {
            "activate sleep mode with forced-flat review",
            "activate sleep mode with forced flat review",
            "sleep safety mode on with forced-flat review",
        }:
            return {"action": "sleep_on", "force_flat_review": True}
        if q in {"deactivate sleep mode", "atlas sleep mode off", "sleep safety mode off"}:
            return {"action": "sleep_off"}
        if q in {"activate operator present mode", "operator present mode on"}:
            return {"action": "operator_present_on"}
        if q in {"deactivate operator present mode", "operator present mode off"}:
            return {"action": "operator_present_off"}
        if q in {"activate session override", "session override on"}:
            return {"action": "session_override_on"}
        if q in {"deactivate session override", "session override off"}:
            return {"action": "session_override_off"}
        if q.startswith("create execution request "):
            # format: create execution request <paper|demo|live|live_real> <action_type> <dry|real> <payload...>
            seg = query.strip().split()
            if len(seg) < 7:
                return {"action": "bad_create"}
            env = seg[3].lower()
            action_type = seg[4].lower()
            mode = seg[5].lower()
            payload = " ".join(seg[6:])
            return {"action": "create_request", "environment": env, "action_type": action_type, "dry_run": mode != "real", "payload": payload}
        if q.startswith("set execution environment "):
            seg = q.split()
            if len(seg) >= 4:
                return {"action": "set_execution_environment", "environment": seg[3]}
        if q.startswith("approve execution "):
            seg = q.split()
            if len(seg) >= 3:
                return {"action": "approve_execution", "id": seg[2], "live_confirm": ("live-confirm" in q or "live confirm" in q)}
        if q.startswith("reject execution "):
            seg = q.split()
            if len(seg) >= 3:
                return {"action": "reject_execution", "id": seg[2]}
        if q.startswith("run execution "):
            seg = q.split()
            if len(seg) >= 3:
                return {"action": "run_execution", "id": seg[2]}
        if q in {"show execution requests", "what execution requests are pending?"}:
            return {"action": "show_execution_requests"}
        if q in {"show execution audit", "what was the last execution attempt?"}:
            return {"action": "show_execution_audit"}
        if q in {"run guarded demo checklist", "run guarded demo workflow checklist"}:
            return {"action": "run_guarded_demo_checklist"}
        if q in {"validate blocked trade path", "run blocked trade validation"}:
            return {"action": "validate_blocked_trade_path"}
        if q in {"validate allowed guarded trade path", "run allowed guarded trade validation"}:
            return {"action": "validate_allowed_guarded_trade_path"}
        if q in {"validate freshness block", "run freshness block validation"}:
            return {"action": "validate_freshness_block"}
        if q in {"check cross-surface consistency", "run cross-surface consistency check"}:
            return {"action": "run_cross_surface_consistency_check"}
        if q in {"mark clean shutdown", "mark clean shutdown now"}:
            return {"action": "mark_clean_shutdown"}
        if q.startswith("set rollout stage "):
            tail = q[len("set rollout stage ") :].strip()
            approved = False
            for marker in (" approved", " confirm", " live-confirm", " live confirm"):
                if tail.endswith(marker):
                    approved = True
                    tail = tail[: -len(marker)].strip()
            stage = tail.replace(" ", "_")
            if stage:
                return {"action": "set_rollout_stage", "stage": stage, "approved": approved}
        if q.startswith("configure tiny live pilot "):
            seg = query.strip().split()
            if len(seg) >= 10:
                return {
                    "action": "configure_tiny_live_pilot",
                    "instrument": seg[4],
                    "setup_family": seg[5],
                    "size_cap": seg[6],
                    "max_trades": seg[7],
                    "daily_loss_cap": seg[8],
                    "max_consecutive_losses": seg[9],
                }
            return {"action": "bad_tiny_live_pilot"}
        if q in {"approve tiny live pilot", "approve tiny live pilot mode"}:
            return {"action": "approve_tiny_live_pilot"}
        if q in {"activate tiny live pilot", "activate tiny live pilot mode"}:
            return {"action": "activate_tiny_live_pilot"}
        if q in {"pause tiny live pilot", "pause tiny live pilot mode"}:
            return {"action": "pause_tiny_live_pilot"}
        if q in {"resume tiny live pilot", "resume tiny live pilot mode"}:
            return {"action": "resume_tiny_live_pilot"}
        if q.startswith("create automation policy "):
            # create automation policy <id> <paper|live|both> <setup_csv> <instrument_csv> <session_csv> <size_cap> <daily_loss_cap> <max_trades> [stage]
            seg = query.strip().split()
            if len(seg) >= 11:
                return {
                    "action": "create_automation_policy",
                    "policy_id": seg[3],
                    "environment": seg[4],
                    "setup_csv": seg[5],
                    "instrument_csv": seg[6],
                    "session_csv": seg[7],
                    "size_cap": seg[8],
                    "daily_loss_cap": seg[9],
                    "max_trades": seg[10],
                    "stage": (seg[11] if len(seg) >= 12 else "approved_paper_execution"),
                }
            return {"action": "bad_automation_create"}
        if q.startswith("approve automation policy "):
            seg = q.split()
            if len(seg) >= 4:
                return {"action": "approve_automation_policy", "policy_id": seg[3]}
        if q.startswith("activate automation policy "):
            seg = q.split()
            if len(seg) >= 4:
                return {"action": "activate_automation_policy", "policy_id": seg[3]}
        if q.startswith("deactivate automation policy "):
            seg = q.split()
            if len(seg) >= 4:
                return {"action": "deactivate_automation_policy", "policy_id": seg[3]}
        if q in {"resume automation", "resume paused automation"}:
            return {"action": "resume_automation"}
        if q.startswith("set autonomy mode "):
            mode = q[len("set autonomy mode ") :].strip().lower()
            if mode:
                return {"action": "set_autonomy_mode", "mode": mode}
        if q.startswith("run auto execution "):
            # run auto execution <paper|live> <action_type> <instrument> <setup> <size> <session> <confidence> <evidence> [dry|real]
            seg = query.strip().split()
            if len(seg) >= 11:
                return {
                    "action": "run_auto_execution",
                    "environment": seg[3],
                    "action_type": seg[4],
                    "instrument": seg[5],
                    "setup_family": seg[6],
                    "size": seg[7],
                    "session": seg[8],
                    "confidence": seg[9],
                    "evidence_strength": seg[10],
                    "dry_run": not (len(seg) >= 12 and str(seg[11]).lower() == "real"),
                }
        if q.startswith("reconcile order "):
            seg = q.split()
            if len(seg) >= 3:
                return {"action": "reconcile_order", "id": seg[2]}
        if q.startswith("reconcile orphan "):
            seg = q.split()
            if len(seg) >= 3:
                return {"action": "reconcile_orphan", "broker_order_id": seg[2]}
        if q in {"scan broker open orders", "scan broker orders"}:
            return {"action": "scan_broker_open_orders"}
        if q in {"reconcile now", "reconcile now live orders"}:
            return {"action": "reconcile_now"}
        return None

    def _handle_execution_query(self, query: str) -> dict[str, Any] | None:
        q = str(query or "").strip().lower()
        runtime = dict(self.source.load_runtime_flags() or {})
        snap = self.execution.snapshot(runtime_flags=runtime)
        self._sync_forced_flat_control_item(runtime, snap)
        self._sync_order_reconciliation_items(snap)
        self._sync_orphan_order_items(snap)
        if q in {"is execution enabled?", "is execution enabled"}:
            return {"mode": "standard", "intent": "execution_status", "response_kind": "control", "result": f"Execution gate is {'enabled' if snap['execution_gate_enabled'] else 'disabled'}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is backend running?", "is backend running"}:
            return {
                "mode": "standard",
                "intent": "system_status_query",
                "response_kind": "control",
                "result": f"Backend is {'running and reporting status normally' if bool(snap.get('execution_gate_enabled', False)) else 'not fully initialized'}. Execution mode: {snap.get('execution_environment_label', 'Paper')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "what autonomy mode is active?",
            "what autonomy mode is active",
            "is autonomy guarded right now?",
            "is autonomy guarded right now",
            "are you autonomous?",
            "are you autonomous",
            "are you an autonomous trader?",
            "are you an autonomous trader",
            "what is your autonomous trader mandate?",
            "what is your autonomous trader mandate",
        }:
            mandate = dict(snap.get("autonomous_trader_mandate", {}) or {})
            semantics = dict(dict(snap.get("autonomy_mode_semantics", {}) or {}).get(str(snap.get("autonomy_mode", "")), {}) or {})
            return {
                "mode": "standard",
                "intent": "automation_status",
                "response_kind": "control",
                "result": (
                    f"Autonomy mode is {str(snap.get('autonomy_mode', 'unknown'))}. "
                    f"Target role is {str(mandate.get('target_role', 'bounded_autonomous_trader'))}. "
                    f"Operating mode is {str((snap.get('mode_separation', {}) or {}).get('execution_mode', 'unknown'))}. "
                    f"Demo-only: {bool(semantics.get('demo_only', False))}. "
                    f"Real live trading: {str(mandate.get('real_live_trading', 'disabled_until_explicit_operator_approval'))}. "
                    f"Simulation market state active: {bool((snap.get('mode_separation', {}) or {}).get('market_state_mode') == 'simulation')}. "
                    "Atlas can simulate disciplined trader-like service intent, but it has no real feelings or private desires."
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "are we in sleep mode now?",
            "are we in sleep mode now",
            "are we in sleep mode?",
            "are we in sleep mode",
            "monitoring-only while sleeping",
            "monitoring only while sleeping",
        }:
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": (
                    f"Sleep Safety Mode is {'active' if bool(snap.get('sleep_mode_active', False)) else 'inactive'}. "
                    f"Execution blocked by sleep mode: {bool(snap.get('sleep_execution_blocked', False))}. "
                    f"Monitoring active: {bool(snap.get('monitoring_active_while_sleeping', True))}. "
                    f"Alert-only behavior: {bool(snap.get('alert_only_while_sleeping', False))}."
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "is session override active?",
            "is session override active",
            "are we overriding session control?",
            "are we overriding session control",
            "why is trading allowed outside normal session?",
            "why is trading allowed outside normal session",
            "what session is being simulated right now?",
            "what session is being simulated right now",
        }:
            active = bool(snap.get("session_override_active", False))
            note = str(snap.get("session_override_note", "off") or "off")
            sim_session = str(snap.get("session_override_effective_session", "none") or "none")
            sim_close = int(snap.get("session_override_effective_minutes_to_close", 0) or 0)
            block_reason = str(snap.get("session_override_last_block_reason", "") or "")
            return {
                "mode": "standard",
                "intent": "session_override_status",
                "response_kind": "control",
                "result": (
                    f"Session override is {'active' if active else 'inactive'}. "
                    f"{'TESTING ONLY. Session-time gating is temporarily relaxed.' if active else 'Normal session logic is active.'} "
                    f"Real session active: {bool(snap.get('session_override_real_session_active', False))}. "
                    f"Simulated session: {sim_session} ({sim_close}m to close). "
                    f"Non-persistent: true. "
                    f"{('Last override block reason: ' + block_reason + '. ') if block_reason else ''}"
                    f"Detail: {note}"
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "why is execution blocked?",
            "why is execution blocked",
            "why is trading blocked?",
            "why is trading blocked",
            "what is blocking trading right now?",
            "what is blocking trading right now",
            "what is blocking demo trading right now?",
            "what is blocking demo trading right now",
        }:
            reason = str(snap.get("tiny_live_pilot_last_block_reason", "") or "")
            last_exec = dict(snap.get("last_execution_result", {}) or {})
            last_reason = str(last_exec.get("reason", "") or "")
            if bool(snap.get("sleep_mode_active", False)):
                reason = "sleep_mode_active_execution_blocked"
            auto_reason = str(snap.get("auto_trade_reason", "") or "")
            resolved = reason or last_reason or auto_reason or "no_active_block_reason"
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": f"Execution block reason: {resolved}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "was the last executed trade a loss?",
            "was the last executed trade a loss",
            "does atlas know the last executed trade was a loss?",
            "does atlas know the last executed trade was a loss",
            "what was the last executed trade outcome?",
            "what was the last executed trade outcome",
            "show the last executed trade outcome",
            "show me the last executed trade outcome",
        }:
            row = self._last_confirmed_trade_outcome()
            if not row:
                result = "I do not have a broker-confirmed closed trade outcome in memory yet."
            else:
                outcome = str(row.get("outcome", "unknown") or "unknown").lower()
                result = (
                    f"Last broker-confirmed executed trade outcome: {outcome}. "
                    f"Ticket {row.get('ticket', '-')}, strategy {row.get('strategy', 'unknown')}, "
                    f"side {row.get('side', 'unknown')}, PnL {float(row.get('net_pnl_zar', 0.0) or 0.0):+.2f} ZAR. "
                    "Blocked or approval-missing attempts are not counted as trades."
                )
            return {
                "mode": "standard",
                "intent": "last_trade_outcome",
                "response_kind": "learning_status",
                "result": result,
                "last_confirmed_trade": row,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "are we in live demo mode now?",
            "are we in live demo mode now",
            "is this real-money live or demo?",
            "is this real money live or demo?",
            "what environment are execution requests using?",
            "what environment are execution requests using",
            "is real-money live disabled?",
            "is real money live disabled?",
        }:
            env_label = str(snap.get("execution_environment_label", "Paper") or "Paper")
            env_key = str(snap.get("execution_environment", "paper") or "paper")
            real_live_enabled = bool(snap.get("real_live_enabled", False))
            return {
                "mode": "standard",
                "intent": "execution_environment_status",
                "response_kind": "control",
                "result": (
                    f"Execution environment: {env_label} ({env_key}). "
                    f"Real Live: {'enabled' if real_live_enabled else 'disabled / not in use'}."
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "are you in operator-present mode?",
            "are you in operator present mode?",
            "are we in operator present mode?",
            "are we in operator-present mode?",
        }:
            return {
                "mode": "standard",
                "intent": "operator_presence_status",
                "response_kind": "control",
                "result": f"Operator-present mode is {'active' if bool(snap.get('operator_present', False)) else 'inactive'}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "can you auto-trade right now?",
            "can you auto trade right now?",
            "is auto-trading demo-only?",
            "is auto trading demo-only?",
            "is auto-trading demo only?",
            "is auto trading demo only?",
            "what must be true before auto-trading can run?",
            "what must be true before auto trading can run?",
            "why did you not take that trade automatically?",
            "why did you not take that trade automatically",
            "why is auto-trading blocked?",
            "why is auto trading blocked?",
            "can you auto-execute trades right now?",
            "can you auto execute trades right now?",
            "why are you not auto-executing?",
            "why are you not auto executing?",
            "did you auto-execute the last eligible trade?",
            "did you auto execute the last eligible trade?",
            "what blocked the last auto-execution?",
            "what blocked the last auto execution?",
            "is automatic execution demo-only?",
            "is automatic execution demo only?",
            "what conditions must be true before auto-execution runs?",
            "what conditions must be true before auto execution runs?",
        }:
            auto_allowed = bool(snap.get("auto_trade_allowed", False))
            auto_reason = str(snap.get("auto_trade_reason", "") or "eligible")
            last_attempt = dict(snap.get("last_auto_execution_result", {}) or snap.get("auto_trade_last_attempt", {}) or {})
            last_outcome = "none"
            if last_attempt:
                last_outcome = "success" if bool(last_attempt.get("ok", False)) else str(last_attempt.get("reason", "blocked") or "blocked")
            must = (
                "demo environment, operator-present active, sleep mode inactive, execution gate enabled, kill switch inactive, "
                "tiny live pilot active/approved/bounded, session/day-risk checks passing, no forced-flat urgency, "
                "no reconciliation required, no orphan orders pending review."
            )
            return {
                "mode": "standard",
                "intent": "auto_trading_status",
                "response_kind": "control",
                "result": (
                    f"Auto-trading is {'allowed' if auto_allowed else 'blocked'} right now. "
                    f"Reason: {auto_reason}. Demo-only: {bool(snap.get('auto_trade_demo_only', True))}. "
                    f"Last auto-execution outcome: {last_outcome}. Requirements: {must}"
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is the kill switch active?", "is the kill switch active"}:
            return {"mode": "standard", "intent": "execution_status", "response_kind": "control", "result": f"Kill switch is {'active' if snap['kill_switch_active'] else 'inactive'} ({snap.get('kill_switch_reason', 'n/a')}).", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what still needs explicit live approval?", "what still needs explicit live approval"}:
            pending_live = [x for x in self.execution.pending_requests(environment="live", limit=20) if x.get("approval_status") != "approved"]
            lines = [f"- {x['id']} [{x['action_type']}] {x['approval_status']}" for x in pending_live]
            return {"mode": "standard", "intent": "execution_status", "response_kind": "control", "result": "Live approval pending:\n" + ("\n".join(lines) if lines else "- none"), "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what is approved for paper only?", "what is approved for paper only"}:
            reqs = [x for x in self.execution.pending_requests(environment="paper", limit=20) if x.get("approval_status") == "approved"]
            lines = [f"- {x['id']} [{x['action_type']}] dry_run={x['dry_run']}" for x in reqs]
            return {"mode": "standard", "intent": "execution_status", "response_kind": "control", "result": "Approved for paper:\n" + ("\n".join(lines) if lines else "- none"), "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what rollout stage are you in?", "what rollout stage are you in"}:
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": f"Rollout stage: {snap.get('rollout_stage', 'paper_only')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is live execution allowed right now?", "is live execution allowed right now"}:
            allowed = bool(snap.get("live_execution_allowed", False))
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": f"Live execution is {'allowed' if allowed else 'not allowed'} in stage {snap.get('rollout_stage', 'paper_only')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what restrictions apply in limited live pilot?", "what restrictions apply in limited live pilot"}:
            restrictions = dict(snap.get("active_restrictions", {}) or {})
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": (
                    "Limited live pilot restrictions:\n"
                    f"- instruments: {', '.join(restrictions.get('allowed_instruments', []) or ['none'])}\n"
                    f"- setup families: {', '.join(restrictions.get('allowed_setup_families', []) or ['none'])}\n"
                    f"- max size: {restrictions.get('max_size', 0.0)}\n"
                    f"- daily loss cap %: {restrictions.get('daily_loss_cap_pct', 0.0)}"
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what is blocking promotion to the next stage?", "what is blocking promotion to the next stage"}:
            reason = str(snap.get("regression_reason", "") or snap.get("kill_switch_reason", "") or "No active blocker recorded.")
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": f"Promotion blocker: {reason}",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"should execution be scaled or reduced?", "should execution be scaled or reduced"}:
            if bool(snap.get("regression_triggered", False)) or bool(snap.get("kill_switch_active", False)):
                decision = "Reduced. Safety regression trigger is active."
            elif not bool(snap.get("live_execution_allowed", False)):
                decision = "Hold. Live execution is not allowed in current stage."
            else:
                decision = "Maintain current scope. Scale only after explicit reviewed promotion."
            return {
                "mode": "standard",
                "intent": "execution_status",
                "response_kind": "control",
                "result": decision,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "are we in tiny live pilot mode?",
            "are we in tiny live pilot mode",
            "is live pilot paused?",
            "is live pilot paused",
        }:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            return {
                "mode": "standard",
                "intent": "live_pilot_status",
                "response_kind": "control",
                "result": (
                    f"Tiny live pilot enabled={bool(pilot.get('enabled', False))}, "
                    f"approved={bool(pilot.get('approved', False))}, paused={bool(pilot.get('paused', False))}."
                ),
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what are the live pilot restrictions?", "what are the live pilot restrictions"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            text = (
                "Tiny live pilot restrictions:\n"
                f"- instrument: {pilot.get('allowed_instrument', 'n/a')}\n"
                f"- setup family: {pilot.get('allowed_setup_family', 'n/a')}\n"
                f"- size cap: {pilot.get('size_cap', 0.0)}\n"
                f"- max trades/day: {pilot.get('max_trades_per_day', 0)}\n"
                f"- daily loss cap: {pilot.get('daily_loss_cap', 0.0)}\n"
                f"- max consecutive losses: {pilot.get('max_consecutive_losses', 0)}\n"
                "- limited live pilot only"
            )
            return {"mode": "standard", "intent": "live_pilot_restrictions", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {
            "what is blocking live pilot?",
            "what is blocking live pilot",
            "what is blocking demo trading right now?",
            "what is blocking demo trading right now",
        }:
            return {
                "mode": "standard",
                "intent": "live_pilot_eligibility",
                "response_kind": "control",
                "result": f"Live pilot block reason: {snap.get('tiny_live_pilot_last_block_reason', 'none')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is live pilot paused?", "is live pilot paused"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            return {
                "mode": "standard",
                "intent": "live_pilot_pause_reason",
                "response_kind": "control",
                "result": f"Tiny live pilot paused={bool(pilot.get('paused', False))}; reason={pilot.get('pause_reason', 'none')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"why was live pilot paused?", "why was live pilot paused"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            return {
                "mode": "standard",
                "intent": "live_pilot_pause_reason",
                "response_kind": "control",
                "result": f"Live pilot pause reason: {pilot.get('pause_reason', 'not paused')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what needs approval before live pilot can continue?", "what needs approval before live pilot can continue"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            blockers: list[str] = []
            if not bool(pilot.get("approved", False)):
                blockers.append("tiny live pilot approval missing")
            if not bool(pilot.get("enabled", False)):
                blockers.append("tiny live pilot not activated")
            if bool(pilot.get("paused", False)):
                blockers.append(f"pilot paused: {pilot.get('pause_reason', 'unknown')}")
            if not bool(snap.get("execution_gate_enabled", False)):
                blockers.append("execution gate disabled")
            if bool(snap.get("kill_switch_active", False)):
                blockers.append("kill switch active")
            text = "Live pilot approval/continuation blockers:\n" + ("\n".join(f"- {x}" for x in blockers) if blockers else "- none")
            return {"mode": "standard", "intent": "live_pilot_eligibility", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"how many live pilot trades have we taken today?", "how many live pilot trades have we taken today"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            text = f"Live pilot trades today: {snap.get('tiny_live_pilot_trade_count', 0)} / {pilot.get('max_trades_per_day', 0)}."
            return {"mode": "standard", "intent": "live_pilot_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"how much live pilot risk is left today?", "how much live pilot risk is left today"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            used = float(snap.get("tiny_live_pilot_daily_loss_used", runtime.get("daily_loss_pct", 0.0)) or 0.0)
            cap = float(pilot.get("daily_loss_cap", 0.0) or 0.0)
            remaining = max(0.0, cap - used)
            text = f"Live pilot daily risk remaining: {remaining:.3f} (used={used:.3f}, cap={cap:.3f})."
            return {"mode": "standard", "intent": "live_pilot_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {
            "show me the live pilot review",
            "show me live pilot review",
            "what happened in the live pilot today?",
            "what happened in the live pilot today",
        }:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            text = (
                "Live pilot review:\n"
                f"- status: enabled={bool(pilot.get('enabled', False))}, approved={bool(pilot.get('approved', False))}, paused={bool(pilot.get('paused', False))}\n"
                f"- scope: {pilot.get('allowed_instrument', 'n/a')} / {pilot.get('allowed_setup_family', 'n/a')} size_cap={pilot.get('size_cap', 0.0)}\n"
                f"- activity: attempts_today={snap.get('tiny_live_pilot_attempts_today', 0)} trades_used={snap.get('tiny_live_pilot_trade_count', 0)}/{pilot.get('max_trades_per_day', 0)} approvals_used={snap.get('live_approvals_used', 0)}\n"
                f"- risk: daily_loss_used={snap.get('tiny_live_pilot_daily_loss_used', 0.0):.3f}/{float(pilot.get('daily_loss_cap', 0.0) or 0.0):.3f} consecutive_losses={snap.get('tiny_live_pilot_consecutive_losses', 0)}/{pilot.get('max_consecutive_losses', 0)}\n"
                f"- order health: last={snap.get('last_live_order_status', 'none')} pending={snap.get('pending_live_orders_count', 0)} partial={snap.get('partially_filled_count', 0)} reconciliation_required={bool(snap.get('reconciliation_required', False))}\n"
                f"- blocks: pilot_block={snap.get('tiny_live_pilot_last_block_reason', 'none')} session_block={dict(snap.get('session_control', {}) or {}).get('session_block_reason', 'none')} pause_reason={pilot.get('pause_reason', 'none')}\n"
                f"- forced flat: pending={bool(snap.get('forced_flat_pending', False))} urgency={snap.get('forced_flat_urgency', 'none')}"
            )
            return {"mode": "standard", "intent": "live_pilot_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is anything waiting for reconciliation?", "is there anything waiting for reconciliation?"}:
            req = bool(snap.get("reconciliation_required", False))
            text = "Yes, reconciliation is required." if req else "No reconciliation-needed live order state is currently active."
            return {"mode": "standard", "intent": "order_reconciliation", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"why is reconciliation required?", "why is reconciliation required"}:
            reason = str(snap.get("last_uncertain_execution_state", "") or snap.get("last_broker_reject_reason", "") or snap.get("tiny_live_pilot_last_block_reason", "") or "No unresolved reason available.")
            return {"mode": "standard", "intent": "reconciliation_status", "response_kind": "control", "result": f"Reconciliation is required because: {reason}", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what order needs review?", "what order needs review"}:
            pending = self.execution.pending_reconciliation_orders(limit=1)
            if not pending:
                text = "No order currently needs reconciliation review."
            else:
                x = pending[0]
                text = f"Order needing review: {x.get('internal_request_id', 'unknown')} (status={x.get('status', 'unknown')}, broker_order_id={x.get('broker_order_id', 'unknown')})."
            return {"mode": "standard", "intent": "reconciliation_review", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what is the mismatch?", "what is the mismatch"}:
            pending = self.execution.pending_reconciliation_orders(limit=1)
            if not pending:
                text = "No active mismatch is currently tracked."
            else:
                x = pending[0]
                text = (
                    f"Mismatch summary: internal_status={x.get('status', 'unknown')} "
                    f"broker_order_id={x.get('broker_order_id', 'unknown')} "
                    f"reason={x.get('rejection_reason', 'reconciliation_required')}."
                )
            return {"mode": "standard", "intent": "order_mismatch_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what was the last known broker status?", "what was the last known broker status"}:
            return {"mode": "standard", "intent": "reconciliation_status", "response_kind": "control", "result": f"Last known broker-related status: {snap.get('last_live_order_status', 'unknown')}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"can you reconcile now?", "can you reconcile now"}:
            out = self.execution.reconcile_now(limit=10)
            self._sync_order_reconciliation_items(self.execution.snapshot())
            return {"mode": "standard", "intent": "reconciliation_action", "response_kind": "control", "result": f"Reconciliation run complete: resolved={out.get('resolved', 0)} unresolved={out.get('unresolved', 0)}.", "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "reconciliation_result": out}
        if q in {"did reconciliation resolve the issue?", "did reconciliation resolve the issue"}:
            req = bool(self.execution.snapshot().get("reconciliation_required", False))
            text = "Reconciliation remains unresolved." if req else "Reconciliation is clear for currently tracked orders."
            return {"mode": "standard", "intent": "reconciliation_status", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug()}
        if q in {"show reconciliation items", "show reconciliation items"}:
            pending = self.execution.pending_reconciliation_orders(limit=20)
            lines = [f"- {x.get('internal_request_id', 'unknown')} status={x.get('status', 'unknown')} broker_order_id={x.get('broker_order_id', 'unknown')}" for x in pending]
            return {"mode": "standard", "intent": "reconciliation_review", "response_kind": "control", "result": "Reconciliation items:\n" + ("\n".join(lines) if lines else "- none"), "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"why is the pilot paused right now?", "why is the pilot paused right now"}:
            pilot = dict(snap.get("tiny_live_pilot", {}) or {})
            pause_reason = str(pilot.get("pause_reason", "") or "not_paused")
            text = f"Pilot pause reason: {pause_reason}. Last block reason: {snap.get('tiny_live_pilot_last_block_reason', 'none')}."
            return {"mode": "standard", "intent": "live_pilot_pause_reason", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what was the last live order result?", "what was the last live order result"}:
            text = (
                f"Last live order status: {snap.get('last_live_order_status', 'none')}. "
                f"Pending={snap.get('pending_live_orders_count', 0)}, partial={snap.get('partially_filled_count', 0)}, "
                f"reject_reason={snap.get('last_broker_reject_reason', 'none')}."
            )
            return {"mode": "standard", "intent": "order_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"are there any forced-flat issues pending?", "are there any forced-flat issues pending"}:
            text = f"Forced-flat pending: {bool(snap.get('forced_flat_pending', False))} (urgency={snap.get('forced_flat_urgency', 'none')})."
            return {"mode": "standard", "intent": "forced_flat_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what happened to the last live order?", "what happened to the last live order"}:
            return {
                "mode": "standard",
                "intent": "order_status",
                "response_kind": "control",
                "result": f"Last live order status: {snap.get('last_live_order_status', 'none')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"was the order acknowledged?", "was the order acknowledged"}:
            status = str(snap.get("last_live_order_status", "none"))
            yes = status in {"acknowledged", "partially_filled", "filled"}
            return {
                "mode": "standard",
                "intent": "order_status",
                "response_kind": "control",
                "result": f"Order acknowledged: {'yes' if yes else 'no'} (status={status}).",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"was anything partially filled?", "was anything partially filled"}:
            c = int(snap.get("partially_filled_count", 0) or 0)
            return {
                "mode": "standard",
                "intent": "order_status",
                "response_kind": "control",
                "result": f"Partially filled live orders: {c}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"why was the order rejected?", "why was the order rejected"}:
            reason = str(snap.get("last_broker_reject_reason", "") or "No recent broker rejection reason.")
            return {"mode": "standard", "intent": "broker_reject_reason", "response_kind": "control", "result": reason, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is reconciliation required?", "is reconciliation required"}:
            req = bool(snap.get("reconciliation_required", False))
            return {"mode": "standard", "intent": "order_reconciliation", "response_kind": "control", "result": f"Reconciliation required: {'yes' if req else 'no'}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"did the broker confirm the order?", "did the broker confirm the order"}:
            status = str(snap.get("last_live_order_status", "none"))
            confirmed = status in {"acknowledged", "partially_filled", "filled"}
            return {"mode": "standard", "intent": "execution_result", "response_kind": "control", "result": f"Broker confirmation: {'yes' if confirmed else 'no'} (status={status}).", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is there any uncertain live order state?", "is there any uncertain live order state"}:
            uncertain = str(snap.get("last_uncertain_execution_state", "") or "")
            return {"mode": "standard", "intent": "order_reconciliation", "response_kind": "control", "result": (f"Uncertain state detected: {uncertain}." if uncertain else "No uncertain live order state detected."), "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what orders are still pending?", "what orders are still pending"}:
            p = int(snap.get("pending_live_orders_count", 0) or 0)
            return {"mode": "standard", "intent": "pending_orders_status", "response_kind": "control", "result": f"Pending live orders: {p}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {
            "show last demo session summary",
            "what happened in the last demo run?",
            "what happened in the last demo run",
            "show recent blocked trades",
            "show recent guarded execution attempts",
            "summarize recent safety blocks",
            "did freshness block anything recently?",
            "did freshness block anything recently",
        }:
            audit_rows = self.execution.recent_audit(limit=200)
            review = self._build_demo_session_review_payload(snap=snap, runtime=runtime, audit_rows=audit_rows, recent_limit=8)
            freshness_hits = sum(int(x.get("count", 0) or 0) for x in list(review.get("freshness_incidents", []) or []))
            top_block = "none"
            reasons = list(review.get("top_block_reasons", []) or [])
            if reasons:
                top_block = str(reasons[0].get("reason", "none"))
            text = (
                "Demo session review:\n"
                f"- guarded attempts: {review.get('attempt_count', 0)}\n"
                f"- allowed: {review.get('allowed_count', 0)}\n"
                f"- blocked: {review.get('blocked_count', 0)}\n"
                f"- top block reason: {top_block}\n"
                f"- freshness incidents: {freshness_hits}\n"
                f"- recent safety events: {len(list(review.get('recent_safety_events', []) or []))}"
            )
            return {
                "mode": "standard",
                "intent": "demo_session_review",
                "response_kind": "control",
                "result": text,
                "demo_session_review": review,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {
            "are we ready to trade?",
            "are we ready to trade",
            "should we start the session?",
            "should we start the session",
            "should we stop trading now?",
            "should we stop trading now",
            "what needs to be checked before trading?",
            "what needs to be checked before trading",
            "give me the runbook checklist",
            "show guarded demo readiness",
            "guarded demo readiness",
            "show clean shutdown checklist",
            "clean shutdown checklist",
            "what should i review after session?",
            "what should i review after session",
        }:
            text = self._build_runbook_checklist_text(snap, runtime)
            readiness = self._build_guarded_demo_readiness_payload(snap, runtime)
            shutdown_checks = [
                f"- sleep mode inactive: {not bool(snap.get('sleep_mode_active', False))}",
                f"- kill switch inactive: {not bool(snap.get('kill_switch_active', False))}",
                f"- no pending live orders: {int(snap.get('pending_live_orders_count', 0) or 0) == 0}",
                f"- no reconciliation required: {not bool(snap.get('reconciliation_required', False))}",
                f"- no orphan orders pending: {int(snap.get('orphan_order_count', 0) or 0) == 0}",
                "- finalize clean shutdown explicitly: use 'mark clean shutdown'",
            ]
            return {
                "mode": "standard",
                "intent": "live_pilot_status",
                "response_kind": "control",
                "result": text + "\nClean shutdown checklist:\n" + "\n".join(shutdown_checks),
                "guarded_demo_readiness": readiness,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"are there any orphan orders?", "are there any orphan orders"}:
            c = int(snap.get("orphan_order_count", 0) or 0)
            return {"mode": "standard", "intent": "order_status", "response_kind": "control", "result": f"Orphan orders detected: {c}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what orders exist at the broker but not in atlas?", "what orders exist at the broker but not in atlas"}:
            rows = self.execution.list_orphan_orders(include_reviewed=False, limit=10)
            lines = [f"- {x.get('broker_order_id', 'unknown')} {x.get('instrument', 'n/a')} {x.get('broker_status', 'unknown')} detected_at={x.get('detected_at', 0.0)}" for x in rows]
            return {"mode": "standard", "intent": "order_status", "response_kind": "control", "result": "Broker orders unknown to Atlas:\n" + ("\n".join(lines) if lines else "- none"), "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"should i review any unknown orders?", "should i review any unknown orders"}:
            c = int(snap.get("orphan_order_count", 0) or 0)
            txt = "Yes. High-priority orphan order review is recommended." if c > 0 else "No unknown broker orders currently require review."
            return {"mode": "standard", "intent": "order_status", "response_kind": "control", "result": txt, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"when was this detected?", "when was this detected"}:
            rows = self.execution.list_orphan_orders(include_reviewed=False, limit=1)
            ts = rows[0].get("detected_at", 0.0) if rows else 0.0
            return {"mode": "standard", "intent": "order_status", "response_kind": "control", "result": f"Top orphan detected_at: {ts}.", "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is automation enabled?", "is automation enabled"}:
            return {
                "mode": "standard",
                "intent": "automation_status",
                "response_kind": "control",
                "result": f"Automation is {'enabled' if snap.get('automation_enabled', False) else 'disabled'} (paused={snap.get('automation_paused', False)}).",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what automation policy is active?", "what automation policy is active"}:
            policy = dict(snap.get("automation_policy", {}) or {})
            if not policy:
                text = "No active automation policy."
            else:
                text = (
                    f"Active automation policy: {policy.get('id', 'n/a')} | env={policy.get('environment', 'n/a')} | "
                    f"setups={','.join(policy.get('allowed_setup_families', []) or []) or 'none'} | "
                    f"instruments={','.join(policy.get('allowed_instruments', []) or []) or 'none'}"
                )
            return {"mode": "standard", "intent": "automation_policy", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"why was automation paused?", "why was automation paused"}:
            return {
                "mode": "standard",
                "intent": "automation_pause_reason",
                "response_kind": "control",
                "result": f"Automation pause reason: {snap.get('automation_pause_reason', 'not paused')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what is allowed to auto-execute right now?", "what is allowed to auto-execute right now"}:
            policy = dict(snap.get("automation_policy", {}) or {})
            if not policy or not snap.get("automation_enabled", False):
                text = "No auto-execution scope is currently active."
            else:
                text = (
                    "Allowed auto-execution scope:\n"
                    f"- environment: {policy.get('environment', 'n/a')}\n"
                    f"- setup families: {', '.join(policy.get('allowed_setup_families', []) or ['none'])}\n"
                    f"- instruments: {', '.join(policy.get('allowed_instruments', []) or ['none'])}\n"
                    f"- sessions: {', '.join(policy.get('allowed_sessions', []) or ['none'])}\n"
                    f"- size cap: {policy.get('size_cap', 0.0)}"
                )
            return {"mode": "standard", "intent": "automation_eligibility", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what still needs approval before automation can run?", "what still needs approval before automation can run"}:
            policy = dict(snap.get("automation_policy", {}) or {})
            blockers: list[str] = []
            if not policy:
                blockers.append("no active automation policy")
            else:
                if not bool(policy.get("approved", False)):
                    blockers.append("policy approval is missing")
                if not bool(policy.get("enabled", False)):
                    blockers.append("policy is not activated")
            if not snap.get("execution_gate_enabled", False):
                blockers.append("execution gate disabled")
            if snap.get("kill_switch_active", False):
                blockers.append("kill switch active")
            text = "Automation approval blockers:\n" + ("\n".join(f"- {x}" for x in blockers) if blockers else "- none")
            return {"mode": "standard", "intent": "automation_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"is live automation allowed right now?", "is live automation allowed right now"}:
            text = f"Live automation is {'allowed' if snap.get('automation_live_allowed', False) else 'not allowed'}."
            return {"mode": "standard", "intent": "automation_status", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"what are the current automation restrictions?", "what are the current automation restrictions"}:
            policy = dict(snap.get("automation_policy", {}) or {})
            if not policy:
                text = "No automation policy restrictions are active."
            else:
                text = (
                    "Current automation restrictions:\n"
                    f"- rollout stage: {snap.get('rollout_stage', 'paper_only')}\n"
                    f"- environment: {policy.get('environment', 'n/a')}\n"
                    f"- allowed setups: {', '.join(policy.get('allowed_setup_families', []) or ['none'])}\n"
                    f"- allowed instruments: {', '.join(policy.get('allowed_instruments', []) or ['none'])}\n"
                    f"- sessions: {', '.join(policy.get('allowed_sessions', []) or ['none'])}\n"
                    f"- size cap: {policy.get('size_cap', 0.0)}\n"
                    f"- daily loss cap: {policy.get('daily_loss_cap', 0.0)}"
                )
            return {"mode": "standard", "intent": "automation_policy", "response_kind": "control", "result": text, "execution_state": snap, "control_state": self.control.to_debug()}
        if q in {"why was this auto-execution blocked?", "why was this auto-execution blocked"}:
            return {
                "mode": "standard",
                "intent": "automation_pause_reason",
                "response_kind": "control",
                "result": f"Last automation block reason: {snap.get('automation_last_block_reason', 'n/a')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"are we in a trading session?", "are we in a trading session"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Session active: {bool(sc.get('session_active', False))}. "
                f"Current session: {sc.get('current_session', 'none')}. "
                f"Timezone: {sc.get('session_timezone', 'UTC')}."
            )
            return {
                "mode": "standard",
                "intent": "session_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"how much time is left in the session?", "how much time is left in the session"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Time to session close: {sc.get('minutes_to_session_close', -1)} minute(s). "
                f"Local session time: {sc.get('current_local_time', 'n/a')}."
            )
            return {
                "mode": "standard",
                "intent": "session_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what timezone are your session controls using?", "what timezone are your session controls using"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Session timezone: {sc.get('session_timezone', 'UTC')}. "
                f"Calendar source: {sc.get('session_source', 'n/a')}."
            )
            return {
                "mode": "standard",
                "intent": "timezone_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"when does the session close locally?", "when does the session close locally"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Session window (local): {sc.get('current_session_window_local', 'none')}. "
                f"Time to close: {sc.get('minutes_to_session_close', -1)} minute(s)."
            )
            return {
                "mode": "standard",
                "intent": "session_calendar_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is today a holiday or special close?", "is today a holiday or special close"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Calendar status: {sc.get('session_calendar_status', 'unknown')}. "
                f"Block reason: {sc.get('session_block_reason', 'none')}."
            )
            return {
                "mode": "standard",
                "intent": "holiday_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"can we open a new trade now?", "can we open a new trade now"}:
            sc = dict(snap.get("session_control", {}) or {})
            blocked = str(sc.get("session_block_reason", "") or "").strip()
            allowed = not blocked
            text = (
                "Yes, a new trade is currently allowed by session/day controls."
                if allowed
                else f"No, trading is blocked by session/day controls: {blocked}."
            )
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"do we need to flatten before close?", "do we need to flatten before close"}:
            sc = dict(snap.get("session_control", {}) or {})
            need = bool(sc.get("forced_flat_required", False))
            mins = int(sc.get("minutes_to_session_close", -1) or -1)
            text = (
                f"Forced flat required: {need}. "
                f"Time to close: {mins} minute(s). "
                f"Reason: {sc.get('session_block_reason', 'none')}."
            )
            return {
                "mode": "standard",
                "intent": "forced_flat_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"why is forced flat required?", "why is forced flat required"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Forced flat is required because {sc.get('session_block_reason', 'session close discipline')}. "
                f"Session={sc.get('current_session', 'none')} timezone={sc.get('session_timezone', 'UTC')} "
                f"calendar={sc.get('session_calendar_status', 'unknown')}."
            )
            return {
                "mode": "standard",
                "intent": "forced_flat_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"how much time is left before flatten is required?", "how much time is left before flatten is required"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = f"Time before flatten requirement: {sc.get('minutes_to_session_close', -1)} minute(s)."
            return {
                "mode": "standard",
                "intent": "flatten_urgency",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is there a forced-flat review item pending?", "is there a forced-flat review item pending"}:
            pending = [
                i for i in self.control.pending()
                if i.type == "forced_flat_review"
            ]
            if pending:
                top = pending[0]
                text = f"Yes. Pending forced-flat review item {top.id} (p{top.priority}): {top.summary}"
            else:
                text = "No pending forced-flat review item."
            return {
                "mode": "standard",
                "intent": "session_close_review",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what happens if we do not flatten before close?", "what happens if we do not flatten before close"}:
            text = (
                "Session discipline may be violated and overnight exposure risk can increase. "
                "Atlas raises a high-priority forced-flat review item but does not auto-close positions."
            )
            return {
                "mode": "standard",
                "intent": "flatten_urgency",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is this because of regular close or early close?", "is this because of regular close or early close"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = f"Close type: {sc.get('session_calendar_status', 'unknown')}."
            return {
                "mode": "standard",
                "intent": "session_close_review",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"when is the next valid trading session?", "when is the next valid trading session"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = f"Next valid session: {sc.get('next_session_local', 'unknown')}."
            return {
                "mode": "standard",
                "intent": "next_session_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"are we near session close?", "are we near session close"}:
            sc = dict(snap.get("session_control", {}) or {})
            mins = int(sc.get("minutes_to_session_close", -1) or -1)
            near = mins >= 0 and mins <= 15
            text = (
                f"Near close: {near}. "
                f"Minutes to close: {mins}. "
                f"Calendar status: {sc.get('session_calendar_status', 'unknown')}."
            )
            return {
                "mode": "standard",
                "intent": "session_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what market calendar is active?", "what market calendar is active"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Active calendar source: {sc.get('session_source', 'unknown')}. "
                f"Current session: {sc.get('current_session', 'none')}. "
                f"Timezone: {sc.get('session_timezone', 'UTC')}."
            )
            return {
                "mode": "standard",
                "intent": "session_calendar_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"why is trading blocked?", "why is trading blocked"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Trading block reason: {sc.get('session_block_reason', 'none')}. "
                f"Daily stop: {bool(sc.get('daily_stop_triggered', False))}. "
                f"Session pause: {bool(sc.get('session_pause', False))}."
            )
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"did we hit the daily limit?", "did we hit the daily limit"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = f"Daily stop triggered: {bool(sc.get('daily_stop_triggered', False))}."
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"how many trades have we taken today?", "how many trades have we taken today"}:
            sc = dict(snap.get("session_control", {}) or {})
            text = (
                f"Trades today: {sc.get('daily_trade_count', 0)} / {sc.get('daily_trade_limit', 0)}."
            )
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"are we paused due to losses?", "are we paused due to losses"}:
            sc = dict(snap.get("session_control", {}) or {})
            paused = bool(sc.get("session_pause", False))
            text = (
                f"Session pause due to losses: {paused}. "
                f"Consecutive losses: {sc.get('consecutive_losses', 0)} / {sc.get('consecutive_loss_limit', 0)}."
            )
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"when will trading resume?", "when will trading resume"}:
            sc = dict(snap.get("session_control", {}) or {})
            reason = str(sc.get("session_block_reason", "") or sc.get("session_pause_reason", "none"))
            mins = sc.get("minutes_to_session_close", -1)
            text = (
                "Trading can resume when session/day limits are clear and execution gates allow it. "
                f"Current block reason: {reason}. Time to session close: {mins} minute(s)."
            )
            return {
                "mode": "standard",
                "intent": "runtime_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"was automation restored after restart?", "was automation restored after restart"}:
            restored = str(snap.get("recovery_mode", "fresh_start")) != "fresh_start"
            text = (
                f"Automation was {'restored' if restored else 'not restored from prior state'} after restart. "
                f"Recovery mode: {snap.get('recovery_mode', 'fresh_start')}."
            )
            return {
                "mode": "standard",
                "intent": "automation_recovery_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"why is automation paused after restart?", "why is automation paused after restart"}:
            text = (
                f"Automation paused: {bool(snap.get('automation_paused', False))}. "
                f"Recovery reason: {snap.get('recovery_reason', 'n/a')}. "
                f"Pause reason: {snap.get('automation_pause_reason', 'n/a')}."
            )
            return {
                "mode": "standard",
                "intent": "recovery_pause_reason",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what policy was restored?", "what policy was restored"}:
            restored_id = str(snap.get("restored_policy_id", "") or "none")
            active_id = str(snap.get("automation_active_policy_id", "") or "none")
            text = f"Restored policy id: {restored_id}. Active automation policy id: {active_id}."
            return {
                "mode": "standard",
                "intent": "restored_policy_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"did the kill switch persist?", "did the kill switch persist"}:
            text = (
                f"Kill switch is {'active' if snap.get('kill_switch_active', False) else 'inactive'} "
                f"with reason: {snap.get('kill_switch_reason', 'n/a')}."
            )
            return {
                "mode": "standard",
                "intent": "automation_recovery_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what still needs review before automation can resume?", "what still needs review before automation can resume"}:
            blockers: list[str] = []
            if snap.get("recovery_requires_explicit_resume", False):
                blockers.append("explicit operator resume is required after recovery")
            if snap.get("kill_switch_active", False):
                blockers.append("kill switch is active")
            if not snap.get("execution_gate_enabled", False):
                blockers.append("execution gate is disabled")
            if not snap.get("automation_active_policy_id", ""):
                blockers.append("no active automation policy")
            text = "Recovery review requirements:\n" + ("\n".join(f"- {x}" for x in blockers) if blockers else "- none")
            return {
                "mode": "standard",
                "intent": "automation_recovery_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"is live automation allowed after restart?", "is live automation allowed after restart"}:
            text = (
                f"Live automation after restart is {'allowed' if snap.get('automation_live_allowed', False) else 'not allowed'}; "
                f"recovery mode={snap.get('recovery_mode', 'fresh_start')}."
            )
            return {
                "mode": "standard",
                "intent": "automation_recovery_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if q in {"what recovery mode are you in?", "what recovery mode are you in"}:
            text = (
                f"Recovery mode: {snap.get('recovery_mode', 'fresh_start')}. "
                f"Reason: {snap.get('recovery_reason', 'n/a')}. "
                f"Explicit resume required: {bool(snap.get('recovery_requires_explicit_resume', False))}."
            )
            return {
                "mode": "standard",
                "intent": "automation_recovery_status",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        return None

    def _handle_execution_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = str(cmd.get("action", ""))
        runtime = dict(self.source.load_runtime_flags() or {})
        risk_policy = dict(self.knowledge.load("risk_policy") or {})
        self._sync_forced_flat_control_item(runtime, self.execution.snapshot())
        self._sync_order_reconciliation_items(self.execution.snapshot())
        self._sync_orphan_order_items(self.execution.snapshot())

        if action == "run_guarded_demo_checklist":
            snap = self.execution.snapshot()
            readiness = self._build_guarded_demo_readiness_payload(snap, runtime)
            checks = {
                "execution_path_guarded": readiness.get("execution_path_source") == "execution_gate_guarded",
                "ask_read_only_default": readiness.get("ask_query_mode") == "read_only_by_default",
                "explicit_action_required": bool(readiness.get("explicit_action_required", False)),
                "freshness_visible": "freshness_runtime" in readiness,
                "last_result_visible": "last_execution_result" in readiness,
            }
            return {
                "mode": "standard",
                "intent": "guarded_demo_workflow",
                "response_kind": "control",
                "result": (
                    "Guarded demo checklist complete: "
                    + ", ".join([f"{k}={'ok' if bool(v) else 'fail'}" for k, v in checks.items()])
                ),
                "workflow_checks": checks,
                "guarded_demo_readiness": readiness,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "validate_blocked_trade_path":
            self.execution.set_gate_enabled(True)
            req = self.execution.create_request(
                environment="paper",
                action_type="place_trade",
                payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
                requested_by="operator_validation",
                dry_run=True,
                auto_generated=False,
                request_origin="operator_validation",
                strict_runtime_checks=False,
            )
            rid = str(req.get("id", ""))
            out = self.execution.execute(rid, runtime_flags=runtime, risk_policy=risk_policy)
            snap = self.execution.snapshot()
            return {
                "mode": "standard",
                "intent": "guarded_demo_validation",
                "response_kind": "control",
                "result": f"Blocked-trade validation reason: {out.get('reason', 'unknown')}.",
                "validation_result": {
                    "scenario": "blocked_trade_path",
                    "request_id": rid,
                    "blocked": not bool(out.get("ok", False)),
                    "block_reason": str(out.get("reason", "")),
                    "order_side_effect": False,
                },
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "validate_allowed_guarded_trade_path":
            self.execution.set_gate_enabled(True)
            req = self.execution.create_request(
                environment="paper",
                action_type="place_trade",
                payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
                requested_by="operator_validation",
                dry_run=True,
                auto_generated=False,
                request_origin="operator_validation",
                strict_runtime_checks=False,
            )
            rid = str(req.get("id", ""))
            self.execution.approve_request(rid, approved_by="operator_validation", explicit_live_confirmation=False)
            out = self.execution.execute(rid, runtime_flags=runtime, risk_policy=risk_policy)
            snap = self.execution.snapshot()
            return {
                "mode": "standard",
                "intent": "guarded_demo_validation",
                "response_kind": "control",
                "result": f"Allowed-guarded validation outcome: {'ok' if bool(out.get('ok', False)) else out.get('reason', 'blocked')}.",
                "validation_result": {
                    "scenario": "allowed_guarded_trade_path",
                    "request_id": rid,
                    "ok": bool(out.get("ok", False)),
                    "dry_run": bool(out.get("dry_run", False)),
                    "request_origin": "operator_validation",
                    "audit_present": bool(self.execution.snapshot().get("last_audit_event", None)),
                },
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "validate_freshness_block":
            self.execution.set_gate_enabled(True)
            req = self.execution.create_request(
                environment="paper",
                action_type="place_trade",
                payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
                requested_by="operator_validation",
                dry_run=False,
                auto_generated=False,
                request_origin="operator_validation",
                strict_runtime_checks=True,
            )
            rid = str(req.get("id", ""))
            self.execution.approve_request(rid, approved_by="operator_validation")
            stale_runtime = dict(runtime)
            stale_runtime["freshness_runtime"] = {
                "status_age_ms": 300000,
                "is_stale": True,
            }
            stale_runtime["max_freshness_age_ms"] = 120000
            out = self.execution.execute(
                rid,
                runtime_flags=stale_runtime,
                risk_policy=(risk_policy or {"daily_loss_cap_pct": 5.0}),
            )
            snap = self.execution.snapshot()
            return {
                "mode": "standard",
                "intent": "guarded_demo_validation",
                "response_kind": "control",
                "result": f"Freshness validation reason: {out.get('reason', 'unknown')}.",
                "validation_result": {
                    "scenario": "freshness_block",
                    "request_id": rid,
                    "blocked": not bool(out.get("ok", False)),
                    "block_reason": str(out.get("reason", "")),
                },
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "run_cross_surface_consistency_check":
            snap = self.execution.snapshot()
            runtime_exec = dict(runtime.get("execution", {}) or {})
            checks = {
                "execution_state_present": bool(snap),
                "runtime_freshness_present": isinstance(runtime.get("freshness_runtime", {}), dict),
                "ask_read_only_default": True,
                "execution_env_consistent_with_runtime_snapshot": (
                    True if not runtime_exec else str(runtime_exec.get("execution_environment", "")) == str(snap.get("execution_environment", ""))
                ),
            }
            return {
                "mode": "standard",
                "intent": "guarded_demo_workflow",
                "response_kind": "control",
                "result": "Cross-surface consistency check: " + ", ".join([f"{k}={'ok' if bool(v) else 'fail'}" for k, v in checks.items()]),
                "workflow_checks": checks,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "mark_clean_shutdown":
            self.execution.mark_clean_shutdown()
            snap = self.execution.snapshot()
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Clean shutdown marker persisted.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }

        if action == "enable_gate":
            snap = self.execution.set_gate_enabled(True)
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Execution gate enabled.", "execution_state": snap, "control_state": self.control.to_debug()}
        if action == "disable_gate":
            snap = self.execution.set_gate_enabled(False)
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Execution gate disabled.", "execution_state": snap, "control_state": self.control.to_debug()}
        if action == "kill_on":
            snap = self.execution.activate_kill_switch("operator_requested")
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Kill switch activated. All execution is blocked.", "execution_state": snap, "control_state": self.control.to_debug()}
        if action == "kill_off":
            snap = self.execution.deactivate_kill_switch("operator_requested")
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Kill switch deactivated.", "execution_state": snap, "control_state": self.control.to_debug()}
        if action == "sleep_on":
            snap = self.execution.activate_sleep_mode(
                reason="operator_requested",
                force_flat_review=bool(cmd.get("force_flat_review", False)),
            )
            if bool(cmd.get("force_flat_review", False)):
                self._sync_forced_flat_control_item(runtime, snap)
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Sleep Safety Mode activated. New execution requests and automation are blocked; monitoring and alerts remain active.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "sleep_off":
            snap = self.execution.deactivate_sleep_mode(reason="operator_requested")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Sleep Safety Mode deactivated.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "operator_present_on":
            snap = self.execution.set_operator_present(True, reason="operator_requested")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Operator-present mode activated.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "operator_present_off":
            snap = self.execution.set_operator_present(False, reason="operator_requested")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Operator-present mode deactivated.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "session_override_on":
            snap = self.execution.set_session_override(True, reason="operator_requested")
            if bool(snap.get("session_override_active", False)):
                text = "Session override activated (testing only). Session-time gating is temporarily relaxed in Live Demo and will reset on restart."
            else:
                text = (
                    "Session override not activated. Demo-only testing rule enforced. "
                    f"Reason: {snap.get('session_override_last_block_reason', 'session_override_demo_only')}."
                )
            return {
                "mode": "standard",
                "intent": "session_override_command",
                "response_kind": "control",
                "result": text,
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "session_override_off":
            snap = self.execution.set_session_override(False, reason="operator_requested")
            return {
                "mode": "standard",
                "intent": "session_override_command",
                "response_kind": "control",
                "result": "Session override deactivated. Normal session logic restored.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "bad_create":
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Invalid request format. Use: create execution request <paper|demo|live|live_real> <action_type> <dry|real> <payload>", "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug()}
        if action == "create_request":
            req = self.execution.create_request(
                environment=str(cmd.get("environment", "")),
                action_type=str(cmd.get("action_type", "")),
                payload_summary=str(cmd.get("payload", "")),
                requested_by="operator",
                dry_run=bool(cmd.get("dry_run", True)),
            )
            if not str(req.get("id", "")).strip():
                return {
                    "mode": "standard",
                    "intent": "execution_command",
                    "response_kind": "control",
                    "result": f"Execution request blocked: {req.get('last_block_reason', 'sleep_mode_active_no_new_requests')}.",
                    "execution_state": self.execution.snapshot(),
                    "control_state": self.control.to_debug(),
                    "execution_request": req,
                }
            live_like = str(req.get("environment", "")).lower() in {"live", "demo"}
            approval_level = "high_risk_requires_explicit_confirmation" if live_like else "approval_required"
            self.control.create_manual_item(
                item_type="execution_request",
                summary=f"execution request {req['id']} [{req['environment']}] {req['action_type']}",
                detailed_reason=f"Execution request created: {req['payload_summary']}",
                supporting=[f"dry_run={req['dry_run']}", f"approval_status={req['approval_status']}"],
                constraints=["no automatic execution", "must pass validation", "must pass risk policy checks", "must pass kill-switch gate"],
                confidence=0.7,
                priority=2 if live_like else 3,
                related_domain="command_execution",
                source_module="execution_gate",
                priority_reason="execution request requires strict governance review",
                trigger_conditions=[f"environment={req['environment']}", f"action_type={req['action_type']}"],
                approval_level=approval_level,
                approval_reason="Execution requests cannot run without explicit approval and safety validation.",
            )
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": f"Execution request created: {req['id']} (approval pending).", "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "execution_request": req}
        if action == "approve_execution":
            req = self.execution.approve_request(
                str(cmd.get("id", "")),
                approved_by="operator",
                note="explicit operator approval",
                explicit_live_confirmation=bool(cmd.get("live_confirm", False)),
            )
            if req is None:
                text = "Execution request not found."
            elif req.get("approval_status") != "approved":
                text = f"Execution request {req['id']} still pending: {req.get('last_block_reason', 'approval_incomplete')}."
            else:
                text = f"Execution request {req['id']} approved."
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "execution_request": req}
        if action == "reject_execution":
            req = self.execution.reject_request(str(cmd.get("id", "")), note="rejected_by_operator")
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": (f"Execution request {req['id']} rejected." if req else "Execution request not found."), "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "execution_request": req}
        if action == "run_execution":
            req_id = str(cmd.get("id", ""))
            out = self.execution.execute(req_id, runtime_flags=runtime, risk_policy=risk_policy)
            self._sync_forced_flat_control_item(runtime, self.execution.snapshot())
            self._sync_order_reconciliation_items(self.execution.snapshot())
            if out.get("ok") is True and out.get("dry_run"):
                text = f"Dry-run completed for {req_id}; no real action was executed."
            elif out.get("ok") is True:
                text = f"Execution completed for {req_id}."
            else:
                text = f"Execution blocked for {req_id}: {out.get('reason', 'unknown_reason')}."
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "execution_result": out}
        if action == "reconcile_order":
            rid = str(cmd.get("id", ""))
            out = self.execution.reconcile_order(rid)
            self._sync_order_reconciliation_items(self.execution.snapshot())
            text = (
                f"Reconciliation for {rid}: status={out.get('status', 'unknown')} reason={out.get('reason', 'n/a')}."
                if out.get("ok") or out.get("requires_review") is not None
                else f"Reconciliation failed for {rid}: {out.get('reason', 'unknown')}."
            )
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "reconciliation_result": out}
        if action == "reconcile_orphan":
            bid = str(cmd.get("broker_order_id", ""))
            out = self.execution.reconcile_orphan(bid, mark_reviewed=True)
            self._sync_orphan_order_items(self.execution.snapshot())
            text = f"Orphan reconciliation for {bid}: {'updated' if out.get('ok') else 'failed'} ({out.get('reason', 'ok')})."
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "reconciliation_result": out}
        if action == "scan_broker_open_orders":
            out = self.execution.scan_broker_open_orders()
            self._sync_orphan_order_items(self.execution.snapshot())
            text = f"Broker scan complete: scanned={out.get('scanned', 0)} orphans={out.get('orphans_detected', 0)} mismatches={out.get('mismatches_detected', 0)}."
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "scan_result": out}
        if action == "reconcile_now":
            out = self.execution.reconcile_now(limit=10)
            self._sync_order_reconciliation_items(self.execution.snapshot())
            text = f"Reconciliation run complete: resolved={out.get('resolved', 0)} unresolved={out.get('unresolved', 0)}."
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": text, "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "reconciliation_result": out}
        if action == "show_execution_requests":
            reqs = self.execution.pending_requests(limit=15)
            lines = [f"- {r['id']} [{r['environment']}] {r['action_type']} approval={r['approval_status']} dry_run={r['dry_run']}" for r in reqs]
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Pending execution requests:\n" + ("\n".join(lines) if lines else "- none"), "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug()}
        if action == "set_execution_environment":
            env = str(cmd.get("environment", "paper"))
            snap = self.execution.set_execution_environment(env)
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Execution environment set to {snap.get('execution_environment_label', 'Paper')} ({snap.get('execution_environment', 'paper')}).",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "show_execution_audit":
            last = self.execution.snapshot().get("last_audit_event")
            return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": ("Last execution audit event: " + str(last) if last else "No execution audit events yet."), "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug(), "execution_audit": self.execution.recent_audit(limit=20)}
        if action == "set_rollout_stage":
            stage = str(cmd.get("stage", "paper_only"))
            approved = bool(cmd.get("approved", False))
            snap = self.execution.set_rollout_stage(stage, approved=approved, note="operator_command")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Rollout stage now: {snap.get('rollout_stage', 'paper_only')}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "bad_tiny_live_pilot":
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": (
                    "Invalid tiny live pilot format. Use: configure tiny live pilot "
                    "<instrument> <setup_family> <size_cap> <max_trades> <daily_loss_cap> <max_consecutive_losses>"
                ),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "configure_tiny_live_pilot":
            cfg = self.execution.configure_tiny_live_pilot(
                allowed_instrument=str(cmd.get("instrument", "XAUUSD")),
                allowed_setup_family=str(cmd.get("setup_family", "continuation")),
                size_cap=float(cmd.get("size_cap", 0.01) or 0.01),
                max_trades_per_day=int(float(cmd.get("max_trades", 1) or 1)),
                daily_loss_cap=float(cmd.get("daily_loss_cap", 0.5) or 0.5),
                max_consecutive_losses=int(float(cmd.get("max_consecutive_losses", 1) or 1)),
            )
            self.control.create_manual_item(
                item_type="execution_request",
                summary=f"tiny live pilot activation review: {cfg.get('allowed_instrument', 'n/a')} / {cfg.get('allowed_setup_family', 'n/a')}",
                detailed_reason="Tiny live pilot configured. Activation remains approval-gated and tightly bounded.",
                supporting=[
                    f"size_cap={cfg.get('size_cap', 0.0)}",
                    f"max_trades_per_day={cfg.get('max_trades_per_day', 0)}",
                    f"daily_loss_cap={cfg.get('daily_loss_cap', 0.0)}",
                ],
                constraints=["limited live pilot only", "one instrument and one setup family", "no autonomous broad live trading"],
                confidence=0.75,
                priority=1,
                related_domain="command_execution",
                source_module="execution_gate",
                priority_reason="tiny live pilot is sensitive and requires explicit approval",
                trigger_conditions=["tiny_live_pilot_configured=true"],
                approval_level="high_risk_requires_explicit_confirmation",
                approval_reason="Live pilot activation is sensitive and requires explicit confirmation.",
            )
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": "Tiny live pilot configured. Approval required before activation.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "approve_tiny_live_pilot":
            cfg = self.execution.approve_tiny_live_pilot(note="explicit_operator_approval")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Tiny live pilot approved: {bool(cfg.get('approved', False))}.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "activate_tiny_live_pilot":
            cfg = self.execution.activate_tiny_live_pilot()
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Tiny live pilot enabled: {bool(cfg.get('enabled', False))}.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "pause_tiny_live_pilot":
            cfg = self.execution.pause_tiny_live_pilot("operator_requested")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Tiny live pilot paused: {bool(cfg.get('paused', False))}.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "resume_tiny_live_pilot":
            cfg = self.execution.resume_tiny_live_pilot()
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Tiny live pilot paused: {bool(cfg.get('paused', False))}.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "bad_automation_create":
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": (
                    "Invalid automation policy format. Use: create automation policy "
                    "<id> <paper|live|both> <setup_csv> <instrument_csv> <session_csv> "
                    "<size_cap> <daily_loss_cap> <max_trades> [stage]"
                ),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "create_automation_policy":
            setup = [x.strip() for x in str(cmd.get("setup_csv", "")).split(",") if x.strip()]
            inst = [x.strip() for x in str(cmd.get("instrument_csv", "")).split(",") if x.strip()]
            sess = [x.strip() for x in str(cmd.get("session_csv", "")).split(",") if x.strip()]
            policy = self.execution.create_automation_policy(
                policy_id=str(cmd.get("policy_id", "")),
                environment=str(cmd.get("environment", "paper")),
                allowed_setup_families=setup,
                allowed_instruments=inst,
                allowed_sessions=sess,
                size_cap=float(cmd.get("size_cap", 0.0) or 0.0),
                daily_loss_cap=float(cmd.get("daily_loss_cap", 0.0) or 0.0),
                max_trades_per_session=int(float(cmd.get("max_trades", 0) or 0)),
                active_rollout_stage=str(cmd.get("stage", "approved_paper_execution")),
            )
            self.control.create_manual_item(
                item_type="execution_request",
                summary=f"automation policy activation review: {policy.get('id', 'n/a')}",
                detailed_reason="Automation policy created. Activation requires explicit approval and bounded scope review.",
                supporting=[
                    f"environment={policy.get('environment', 'paper')}",
                    f"setups={','.join(policy.get('allowed_setup_families', []) or [])}",
                    f"instruments={','.join(policy.get('allowed_instruments', []) or [])}",
                ],
                constraints=["no unrestricted automation", "no live bypass", "kill switch and validation stay active"],
                confidence=0.7,
                priority=2,
                related_domain="command_execution",
                source_module="execution_gate",
                priority_reason="automation activation requires explicit governance approval",
                trigger_conditions=[f"policy_id={policy.get('id', 'n/a')}", "automation_activation_requested=true"],
                approval_level="high_risk_requires_explicit_confirmation",
                approval_reason="Automation activation can impact execution flow and requires explicit operator confirmation.",
            )
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Automation policy created: {policy.get('id', 'n/a')}. Approval required before activation.",
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
                "automation_policy": policy,
            }
        if action == "approve_automation_policy":
            policy = self.execution.approve_automation_policy(str(cmd.get("policy_id", "")), note="explicit_operator_approval")
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": (f"Automation policy approved: {policy.get('id', 'n/a')}." if policy else "Automation policy not found."),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
                "automation_policy": policy,
            }
        if action == "activate_automation_policy":
            policy = self.execution.activate_automation_policy(str(cmd.get("policy_id", "")))
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": (f"Automation policy activated: {policy.get('id', 'n/a')}." if policy else "Automation activation blocked. Approval may be missing."),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
                "automation_policy": policy,
            }
        if action == "deactivate_automation_policy":
            policy = self.execution.deactivate_automation_policy(str(cmd.get("policy_id", "")))
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": (f"Automation policy deactivated: {policy.get('id', 'n/a')}." if policy else "Automation policy not found."),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
                "automation_policy": policy,
            }
        if action == "resume_automation":
            ok = self.execution.resume_automation()
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": ("Automation resumed." if ok else "Automation resume blocked."),
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
            }
        if action == "set_autonomy_mode":
            snap = self.execution.set_autonomy_mode(str(cmd.get("mode", "")))
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": f"Autonomy mode set to {str(snap.get('autonomy_mode', 'unknown'))}.",
                "execution_state": snap,
                "control_state": self.control.to_debug(),
            }
        if action == "run_auto_execution":
            status = dict(self.source.load_status() or {})
            chosen = dict(status.get("chosen", {}) or {})
            side = str(cmd.get("side") or chosen.get("side") or status.get("action") or "").strip().lower()
            if side not in {"buy", "sell"}:
                side = ""
            out = self.execution.attempt_auto_execution(
                action_type=str(cmd.get("action_type", "place_trade")),
                environment=str(cmd.get("environment", "paper")),
                instrument=str(cmd.get("instrument", "")),
                setup_family=str(cmd.get("setup_family", "")),
                size=float(cmd.get("size", 0.0) or 0.0),
                session=str(cmd.get("session", "")),
                confidence=float(cmd.get("confidence", 0.0) or 0.0),
                evidence_strength=float(cmd.get("evidence_strength", 0.0) or 0.0),
                side=side,
                strategy=str(cmd.get("strategy") or chosen.get("strategy") or cmd.get("setup_family", "")),
                stop_loss_points=float(cmd.get("stop_loss_points") or chosen.get("stop_loss_points") or 0.0),
                take_profit_points=float(cmd.get("take_profit_points") or chosen.get("take_profit_points") or 0.0),
                runtime_flags=runtime,
                risk_policy=risk_policy,
                dry_run=bool(cmd.get("dry_run", True)),
            )
            self._sync_forced_flat_control_item(runtime, self.execution.snapshot())
            if not out.get("ok", False):
                self.control.create_manual_item(
                    item_type="execution_request",
                    summary="automation blocked - review required",
                    detailed_reason=f"Auto-execution was blocked: {out.get('reason', 'unknown_reason')}.",
                    supporting=[f"reason={out.get('reason', 'unknown_reason')}"],
                    constraints=["automation paused until reviewed", "no execution bypass", "operator approval required to resume automation"],
                    confidence=0.7,
                    priority=1,
                    related_domain="command_execution",
                    source_module="execution_gate",
                    priority_reason="automation block trigger requires explicit review",
                    trigger_conditions=[f"automation_block_reason={out.get('reason', 'unknown_reason')}"],
                    approval_level="approval_required",
                    approval_reason="Automation pause/escalation requires operator review before resume.",
                )
            text = "Auto-execution request processed successfully." if out.get("ok", False) else f"Auto-execution blocked: {out.get('reason', 'unknown_reason')}."
            return {
                "mode": "standard",
                "intent": "execution_command",
                "response_kind": "control",
                "result": text,
                "execution_state": self.execution.snapshot(),
                "control_state": self.control.to_debug(),
                "automation_result": out,
            }
        return {"mode": "standard", "intent": "execution_command", "response_kind": "control", "result": "Unsupported execution command.", "execution_state": self.execution.snapshot(), "control_state": self.control.to_debug()}

    @staticmethod
    def _select_mode(query: str, intent: str, requested: str) -> str:
        req = str(requested or "").strip().lower()
        if req in {"quick", "standard", "deep", "voice"}:
            return req
        q = query.lower()
        if intent == "status" and any(k in q for k in ("state", "status", "mode")):
            return "quick"
        return "standard"
