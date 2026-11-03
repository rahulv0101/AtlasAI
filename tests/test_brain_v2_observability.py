from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.21, "side": "flat"},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}},
            "comparison_run_id": "atlas_wake_latest",
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "No high-confidence trade found."}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}


def test_brain_v2_debug_fields_present_for_target_questions() -> None:
    svc = AskAtlasService(_Source())
    queries = [
        "What state are you in?",
        "What is holding you back?",
        "Why are you not taking trades?",
        "Are you ready for live?",
        "What changed in the latest comparison?",
        "What is the next safe experiment?",
        "How much do you know about the markets?",
        "What can you do?",
        "What are your limitations?",
        "Do you understand volatility?",
        "Help me understand what you can answer.",
        "Are you intelligent?",
    ]
    for q in queries:
        r = svc.answer(q)
        dbg = r.get("debug", {})
        assert isinstance(dbg.get("intent_candidates"), list)
        assert isinstance(dbg.get("domain_candidates"), list)
        assert isinstance(dbg.get("selected_plan"), str)
        assert isinstance(dbg.get("evidence_chosen"), list)
        assert isinstance(dbg.get("relevance_guard"), str)
        trace = dict(r.get("answer_trace", {}) or {})
        assert "normalized_question" in trace
        if trace.get("normalized_question"):
            assert trace["normalized_question"] == q
        assert trace["selected_intent"] == r["intent"]
        assert trace["selected_domain"] == r.get("selected_domain", trace["selected_domain"])
        assert trace["selected_plan"]
        assert isinstance(trace.get("evidence"), list)
        assert trace["relevance_result"]
        separation = dict(trace.get("evidence_separation", {}) or {})
        assert separation["signal_direction"] == "neutral"
        assert separation["confidence"] == 0.21
        assert separation["threshold"] == 0.58
        assert separation["execution_permission"] == "not_allowed"
        assert r.get("selected_domain")
        assert r.get("confidence") is not None


def test_answer_trace_keeps_operator_questions_out_of_execution_state() -> None:
    svc = AskAtlasService(_Source())
    before = svc.execution.snapshot()

    reply = svc.answer("How much do you know about the markets?")
    after = svc.execution.snapshot()

    assert reply["handled_as"] == "general"
    assert reply["answer_trace"]["selected_domain"] in {"self_model", "market_knowledge"}
    assert reply["answer_trace"]["selected_plan"] in {"self_knowledge_plan", "market_knowledge_plan"}
    assert before["execution_environment"] == after["execution_environment"]
    assert before["rollout_stage"] == after["rollout_stage"]
    assert before["execution_gate_enabled"] == after["execution_gate_enabled"]
