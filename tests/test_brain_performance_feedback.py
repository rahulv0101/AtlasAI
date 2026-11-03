from __future__ import annotations

from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.brain.performance_feedback import PerformanceFeedbackLoop
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self, *, noisy: bool = False, strong_shift_delayed: bool = False, flip_regime: bool = False) -> None:
        self.noisy = noisy
        self.strong_shift_delayed = strong_shift_delayed
        self.flip_regime = flip_regime

    def load_status(self) -> dict:
        if self.noisy:
            return {
                "action": "hold",
                "message": "No high-confidence trade found.",
                "chosen": {"confidence": 0.22},
                "min_confidence_threshold": 0.58,
                "regime": {"name": "chop", "trend_strength": 0.07},
                "context_snapshot": {"volatility_state": "high"},
                "market": {"recent_close": [100, 101.7, 99.1, 101.6, 98.9, 101.8, 99.0, 101.5]},
                "comparison": {"active": True, "run_id": "atlas_wake_latest"},
            }
        return {
            "action": "hold",
            "message": "Monitoring setup quality.",
            "chosen": {"confidence": 0.69},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "trend", "trend_strength": 0.48},
            "context_snapshot": {"volatility_state": "elevated"},
            "market": {"recent_close": [100, 100.2, 100.4, 100.6, 100.9, 101.1, 101.3, 101.6]},
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        if self.noisy:
            ev = [
                {"type": "risk", "message": "Breakout failed in noisy chop."},
                {"type": "risk", "message": "Conservative filter blocked trade due to noise."},
            ]
            if self.flip_regime:
                ev += [
                    {"type": "risk", "message": "Regime changed rapidly to trend then reverted."},
                    {"type": "risk", "message": "Regime flip detected under mixed evidence."},
                ]
            return ev
        ev = [
            {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
            {"type": "risk", "message": "Filter blocked one paper candidate due to guardrails."},
        ]
        if self.strong_shift_delayed:
            ev += [
                {"type": "risk", "message": "Strong persistence confirmed across windows."},
                {"type": "risk", "message": "Strong persistence confirmed but response delayed."},
            ]
        return ev

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}


def _ask(source: _Source, q: str) -> dict:
    return AskAtlasService(source).answer(q)


def test_strong_evidence_delayed_adaptation_flags_slowness() -> None:
    source = _Source(strong_shift_delayed=True)
    status = source.load_status()
    events = source.load_events()
    adaptation = build_adaptation_summary(status, events)
    # Force delayed behavior condition in bounded evaluation input.
    adaptation["detection_to_confidence_update_latency"] = 6.5
    feedback = PerformanceFeedbackLoop().evaluate(
        status=status,
        runtime=source.load_runtime_flags(),
        adaptation_summary=adaptation,
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait"}]},
        events=events,
    )
    assert feedback.scores["confidence_update_quality_score"] < 0.6
    assert "adaptation_review" in feedback.scoring_breakdown


def test_weak_noisy_with_aggressive_shift_flags_reactivity_concern() -> None:
    source = _Source(noisy=True)
    adaptation = build_adaptation_summary(source.load_status(), source.load_events())
    adaptation["overreaction_score"] = 0.82
    feedback = PerformanceFeedbackLoop().evaluate(
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=adaptation,
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait"}]},
        events=source.load_events(),
    )
    assert feedback.scores["stability_quality_score"] < 0.62
    assert "stability/reactivity" in feedback.biggest_weakness
    assert any("stability/reactivity" in x for x in feedback.weakness_ranking)


def test_stable_noisy_with_caution_recommendation_is_appropriate() -> None:
    source = _Source(noisy=True)
    adaptation = build_adaptation_summary(source.load_status(), source.load_events())
    feedback = PerformanceFeedbackLoop().evaluate(
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=adaptation,
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        events=source.load_events(),
    )
    assert feedback.scores["recommendation_quality_score"] >= 0.6


def test_frequent_regime_flips_flag_regime_stability_concern() -> None:
    source = _Source(noisy=True, flip_regime=True)
    feedback = PerformanceFeedbackLoop().evaluate(
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        events=source.load_events(),
    )
    assert feedback.scores["regime_interpretation_score"] < 0.7
    assert "regime_review" in feedback.scoring_breakdown


def test_high_blocker_ignored_flags_priority_issue() -> None:
    source = _Source(noisy=True)
    feedback = PerformanceFeedbackLoop().evaluate(
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "monitoring_priority", "recommendation": "Watch volatility only."}]},
        events=source.load_events(),
    )
    assert "blocker pressure" in feedback.biggest_weakness or feedback.scores["recommendation_quality_score"] < 0.65


def test_ask_how_well_adapting_returns_adaptation_review() -> None:
    r = _ask(_Source(noisy=True), "How well are you adapting?")
    assert r["intent"] == "adaptation_review"
    assert "Adaptation review:" in r["result"]
    assert "Recommendation review:" in r["result"]
    assert "Stability review:" in r["result"]
    assert "Main weakness:" in r["result"]


def test_ask_too_reactive_returns_stability_review() -> None:
    r = _ask(_Source(noisy=True), "Are you becoming too reactive?")
    assert r["intent"] == "stability_review"
    assert "Stability review:" in r["result"]


def test_ask_improving_returns_direction_review() -> None:
    service = AskAtlasService(_Source(noisy=True))
    service.answer("How well are you adapting?")
    r = service.answer("Are you improving?")
    assert r["intent"] == "improvement_assessment"
    assert "Recent direction:" in r["result"]


def test_ask_biggest_weakness_returns_safe_next_step() -> None:
    r = _ask(_Source(noisy=True), "What is your biggest current weakness?")
    assert r["intent"] == "weakness_analysis"
    assert "Main weakness:" in r["result"]
    assert "Safe next step:" in r["result"]


def test_feedback_remains_non_executing_and_conservative_noted() -> None:
    source = _Source(noisy=True)
    feedback = PerformanceFeedbackLoop().evaluate(
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        events=source.load_events(),
    )
    assert "no_trade_execution_from_feedback" in feedback.notes
    assert "no_strategy_or_risk_auto_mutation" in feedback.notes
