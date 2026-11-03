from __future__ import annotations

from atlas_ai.brain.ai_decision_core import AIDecisionCore
from atlas_ai.brain.cognitive_core import build_cognitive_core


def test_ai_decision_core_outputs_guarded_decision_trace() -> None:
    status = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "chosen": {"side": "sell", "confidence": 0.42},
        "min_confidence_threshold": 0.55,
        "primary_blocker": "confidence below threshold",
        "context_snapshot": {"regime": "trend", "session": "new_york"},
    }
    cognitive = build_cognitive_core(
        status=status,
        runtime={},
        learning_state={},
        adaptation_summary={"stability_score": 0.7, "overreaction_score": 0.1},
        evidence_separation={
            "market_bias": "sell",
            "signal_direction": "sell",
            "final_trade_bias": "sell",
            "setup_readiness": "developing_or_blocked",
            "execution_permission": "not_allowed",
        },
    )

    trace = AIDecisionCore().decide(
        status=status,
        runtime={},
        learning_state={},
        cognitive_core=cognitive,
    ).to_dict()

    assert trace["schema"] == "atlas_ai_decision_trace_v1"
    assert trace["recommendation"]["action"] == "wait_for_confirmation"
    assert "confidence_below_threshold" in trace["blocker_analysis"]["blockers"]
    assert trace["strategy_invention"]["candidate_count"] > 0
    assert trace["safety"]["can_execute"] is False
    assert trace["safety"]["can_bypass_execution_gate"] is False


def test_ai_decision_core_can_prepare_candidate_but_not_execute() -> None:
    status = {
        "symbol": "XAUUSD",
        "chosen": {"side": "buy", "confidence": 0.82},
        "min_confidence_threshold": 0.55,
        "context_snapshot": {"regime": "trend"},
    }
    cognitive = build_cognitive_core(
        status=status,
        runtime={},
        learning_state={},
        adaptation_summary={"stability_score": 0.85, "overreaction_score": 0.05},
        evidence_separation={
            "market_bias": "buy",
            "signal_direction": "buy",
            "final_trade_bias": "buy",
            "setup_readiness": "ready",
            "execution_permission": "allowed",
        },
    )

    core = AIDecisionCore()
    trace = core.decide(status=status, cognitive_core=cognitive).to_dict()

    assert trace["recommendation"]["action"] == "prepare_trade_candidate"
    assert trace["safety"]["can_execute"] is False
    assert not hasattr(core, "execute")
    assert not hasattr(core, "place_order")


def test_ask_atlas_journals_and_reports_latest_ai_decision(tmp_path) -> None:
    from atlas_ai.memory.knowledge_store import KnowledgeStore
    from atlas_ai.ui.ask_atlas import AskAtlasService

    class Source:
        def __init__(self) -> None:
            self.store = KnowledgeStore(str(tmp_path))

        def load_status(self) -> dict:
            return {
                "symbol": "XAUUSD",
                "chosen": {"side": "sell", "confidence": 0.42},
                "min_confidence_threshold": 0.55,
                "primary_blocker": "confidence below threshold",
                "context_snapshot": {"regime": "trend"},
            }

        def load_events(self, limit: int = 160) -> list[dict]:
            return []

        def load_runtime_flags(self) -> dict:
            return {}

        def load_learning_state(self) -> dict:
            return {"state": {}}

        def load_comparison(self) -> dict:
            return {}

    svc = AskAtlasService(Source())
    svc.answer("Does Atlas have its own brain for understanding and learning?")
    reply = svc.answer("What was your last decision?")

    assert "Latest AI decision" in reply["result"]
    assert (tmp_path / "ai_decision_journal.jsonl").exists()
