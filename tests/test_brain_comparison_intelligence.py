from __future__ import annotations

from atlas_ai.brain.comparison_intelligence import ComparisonIntelligence
from atlas_ai.ui.ask_atlas import AskAtlasService


def _adaptation(evidence: float, stability: float, regime_conf: float = 0.6) -> dict:
    return {
        "evidence_strength_score": evidence,
        "stability_score": stability,
        "regime_shift_confidence": regime_conf,
    }


def _performance(rec_q: float, adapt: float, stability: float) -> dict:
    return {
        "scores": {
            "recommendation_quality_score": rec_q,
            "adaptation_effectiveness_score": adapt,
            "stability_quality_score": stability,
        }
    }


def _paper(outcome: str) -> dict:
    return {"comparisons": [{"outcome": outcome}]}


def test_improved_recommendation_quality_positive_comparison() -> None:
    ci = ComparisonIntelligence()
    ci.evaluate(
        adaptation_summary=_adaptation(0.5, 0.6),
        performance_feedback=_performance(0.45, 0.5, 0.6),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[{"type": "risk", "message": "filter blocked setup"}],
    )
    result = ci.evaluate(
        adaptation_summary=_adaptation(0.6, 0.62),
        performance_feedback=_performance(0.72, 0.58, 0.64),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[{"type": "risk", "message": "monitoring event"}],
    )
    assert result.direction in {"improving", "mixed"}
    assert result.deltas["recommendation_quality_delta"] > 0


def test_faster_adaptation_with_worse_stability_mixed() -> None:
    ci = ComparisonIntelligence()
    ci.evaluate(
        adaptation_summary=_adaptation(0.55, 0.72),
        performance_feedback=_performance(0.6, 0.45, 0.72),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[],
    )
    result = ci.evaluate(
        adaptation_summary=_adaptation(0.7, 0.52),
        performance_feedback=_performance(0.62, 0.7, 0.52),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[],
    )
    assert result.deltas["adaptation_speed_delta"] > 0
    assert result.deltas["stability_delta"] < 0
    assert "mixed" in result.safe_interpretation.lower() or result.direction in {"mixed", "degrading"}


def test_lower_blocker_pressure_shows_improving_blocker_delta() -> None:
    ci = ComparisonIntelligence()
    ci.evaluate(
        adaptation_summary=_adaptation(0.5, 0.6),
        performance_feedback=_performance(0.5, 0.5, 0.6),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[
            {"type": "risk", "message": "filter blocked setup"},
            {"type": "risk", "message": "blocked by conservative filter"},
        ],
    )
    result = ci.evaluate(
        adaptation_summary=_adaptation(0.52, 0.62),
        performance_feedback=_performance(0.55, 0.52, 0.62),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[{"type": "risk", "message": "monitor only"}],
    )
    assert result.deltas["blocker_pressure_delta"] > 0


def test_tuning_help_but_noise_risk_produces_cautious_or_mixed() -> None:
    ci = ComparisonIntelligence()
    ci.evaluate(
        adaptation_summary=_adaptation(0.55, 0.7),
        performance_feedback=_performance(0.6, 0.6, 0.7),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[],
    )
    result = ci.evaluate(
        adaptation_summary=_adaptation(0.58, 0.52),
        performance_feedback=_performance(0.62, 0.62, 0.52),
        recommendation_result={"items": []},
        paper_tuning=_paper("helpful"),
        events=[],
    )
    assert result.deltas["tuning_effectiveness_delta"] >= 0
    assert result.direction in {"mixed", "degrading", "improving"}
    assert "safe" in result.safe_interpretation.lower() or "caution" in result.safe_interpretation.lower()


def test_sparse_history_returns_no_clear_comparison_yet() -> None:
    ci = ComparisonIntelligence()
    result = ci.evaluate(
        adaptation_summary=_adaptation(0.5, 0.5),
        performance_feedback=_performance(0.5, 0.5, 0.5),
        recommendation_result={"items": []},
        paper_tuning=_paper("inconclusive"),
        events=[],
    )
    assert result.sparse_data is True
    assert result.direction == "no clear comparison yet"


class _AskSource:
    def __init__(self) -> None:
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.31},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "trend", "trend_strength": 0.3},
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
            "market": {"recent_close": [100, 100.2, 100.4, 100.5, 100.7, 100.8, 101.0, 101.2]},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "risk", "message": "filter blocked setup"},
            {"type": "risk", "message": "breakout follow-through confirmed"},
        ]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest"}

    def execute_trade(self) -> None:
        self.execute_called = True


def test_ask_are_you_improving_returns_recent_vs_prior_context() -> None:
    svc = AskAtlasService(_AskSource())
    svc.answer("How well are you adapting?")
    r = svc.answer("Are you improving?")
    assert r["intent"] == "improvement_assessment"
    assert "Compared to:" in r["result"]


def test_ask_compared_to_what_explains_baseline_reference() -> None:
    svc = AskAtlasService(_AskSource())
    r = svc.answer("Compared to what?")
    assert r["intent"] == "baseline_comparison"
    assert "vs baseline" in r["result"].lower() or "Compared to:" in r["result"]


def test_ask_did_paper_tuning_improve_things_stays_tuning_comparison() -> None:
    source = _AskSource()
    svc = AskAtlasService(source)
    r = svc.answer("Did paper tuning improve things?")
    assert r["intent"] == "tuning_comparison"
    assert "Paper tuning result:" in r["result"]
    assert source.execute_called is False

