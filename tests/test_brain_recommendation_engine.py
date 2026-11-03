from __future__ import annotations

from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.brain.recommendation_engine import RecommendationEngine
from atlas_ai.brain.knowledge import BrainKnowledge
from atlas_ai.ui.ask_atlas import AskAtlasService


class _BaseSource:
    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}


class _NoisySource(_BaseSource):
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.25},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "chop", "trend_strength": 0.08},
            "context_snapshot": {"volatility_state": "high"},
            "market": {"recent_close": [100, 101.5, 99.0, 101.7, 98.8, 101.4, 99.2, 101.6]},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "risk", "message": "Breakout failed quickly in noisy chop."},
            {"type": "risk", "message": "Conservative filter blocked trade due to noise."},
        ]


class _StrongSource(_BaseSource):
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "Monitoring confirmation setup.",
            "chosen": {"confidence": 0.70},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "trend", "trend_strength": 0.52},
            "context_snapshot": {"volatility_state": "elevated"},
            "market": {"recent_close": [100, 100.2, 100.4, 100.7, 101.0, 101.3, 101.5, 101.8]},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "risk", "message": "Breakout follow-through confirmed across recent windows."},
            {"type": "risk", "message": "Aligned momentum confirmation observed."},
            {"type": "risk", "message": "Filter blocked one entry due to paper-only guard."},
        ]


class _NoisyExecSource(_NoisySource):
    def __init__(self) -> None:
        self.execute_called = False

    def execute_trade(self) -> None:
        self.execute_called = True


def _build_reco(source: _BaseSource, query: str = "What should you do next?") -> dict:
    knowledge = BrainKnowledge()
    status = source.load_status()
    events = source.load_events()
    runtime = source.load_runtime_flags()
    comparison = source.load_comparison()
    adaptation = build_adaptation_summary(status, events)
    result = RecommendationEngine().build(
        query=query,
        status=status,
        runtime=runtime,
        comparison=comparison,
        adaptation_summary=adaptation,
        blockers_catalog=knowledge.load("blockers_catalog"),
        risk_policy=knowledge.load("risk_policy"),
        readiness_rules=knowledge.load("readiness_rules"),
        strategy_knowledge=knowledge.load("strategy_knowledge"),
    )
    return RecommendationEngine.to_dict(result)


def _ask(source: _BaseSource, q: str) -> dict:
    return AskAtlasService(source).answer(q)


def test_noisy_market_prefers_caution_recommendation() -> None:
    reco = _build_reco(_NoisySource())
    top = reco["items"][0]
    assert top["category"] == "action_posture"
    assert "hold" in top["recommendation"].lower() or "wait" in top["recommendation"].lower()


def test_strong_evidence_still_bounded_recommendation() -> None:
    reco = _build_reco(_StrongSource())
    joined = " ".join(item["recommendation"].lower() for item in reco["items"][:3])
    assert "confirmation" in joined or "monitor" in joined
    assert all("no auto" in item["constraint"].lower() or "paper" in item["constraint"].lower() or "no execution" in item["constraint"].lower() for item in reco["items"])


def test_high_blocker_priority_outranks_opportunity() -> None:
    reco = _build_reco(_NoisySource(), query="What should be reviewed first?")
    categories = [x["category"] for x in reco["items"][:2]]
    assert "blocker_resolution" in categories or "action_posture" in categories
    if "blocker_resolution" in [x["category"] for x in reco["items"]]:
        blocker_idx = [x["category"] for x in reco["items"]].index("blocker_resolution")
        experiment_idx = [x["category"] for x in reco["items"]].index("experiment_suggestion")
        assert blocker_idx < experiment_idx


def test_low_readiness_forces_conservative_policy_note() -> None:
    reco = _build_reco(_StrongSource())
    assert "readiness_low_forced_non_live_recommendations" in reco["policy_notes"]


def test_ask_atlas_next_step_query() -> None:
    r = _ask(_NoisySource(), "What should you do next?")
    assert r["intent"] in {"safe_next_step", "recommendation"}
    text = r["result"].lower()
    assert ("recommended next step" in text) or ("no high-confidence trade found. recommendation: hold." in text)


def test_ask_atlas_market_recommendation_query() -> None:
    r = _ask(_NoisySource(), "What do you recommend in this market?")
    assert r["intent"] == "recommendation"
    assert "recommendation:" in r["result"].lower() or "recommended next step" in r["result"].lower()


def test_ask_atlas_review_first_query() -> None:
    r = _ask(_NoisySource(), "What should be reviewed first?")
    assert r["intent"] == "blocker_resolution"
    assert "main recommendation:" in r["result"].lower()


def test_ask_atlas_safe_experiment_query() -> None:
    r = _ask(_NoisySource(), "Give me a safe experiment.")
    assert r["intent"] == "experiment_suggestion"
    assert "safe experiment:" in r["result"].lower()
    assert "paper" in r["result"].lower()


def test_recommendation_engine_emits_caution_adjustment_when_noisy() -> None:
    reco = _build_reco(_NoisySource(), query="What do you recommend in this market?")
    categories = [x["category"] for x in reco["items"]]
    assert "caution_adjustment" in categories


def test_ask_atlas_why_recommending_caution_returns_explanation_assist() -> None:
    r = _ask(_NoisySource(), "Why are you recommending caution?")
    assert r["intent"] == "recommendation"
    assert "why:" in r["result"].lower()
    assert "evidence_strength" in r["result"] or "blocker_severity" in r["result"]


def test_conversational_trade_questions_route_to_safe_recommendation() -> None:
    questions = [
        "Should I buy or sell?",
        "What do you think I should do?",
        "Is this a buy?",
        "Should I take a trade right now?",
        "Is there a trade opportunity?",
        "What direction do you prefer?",
    ]
    for q in questions:
        r = _ask(_NoisySource(), q)
        assert r["intent"] in {"recommendation", "safe_next_step"}
        assert r["intent"] != "unknown_operator_query"
        assert "No high-confidence trade found. Recommendation: hold." in r["result"]


def test_conversational_trade_questions_have_no_execution_side_effects() -> None:
    source = _NoisyExecSource()
    svc = AskAtlasService(source)
    svc.answer("Should I buy or sell?")
    svc.answer("Is there a trade opportunity?")
    assert source.execute_called is False
