from __future__ import annotations

from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.brain.paper_tuning import PaperTuningEngine
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self, *, noisy: bool = False, stable_borderline: bool = False) -> None:
        self.noisy = noisy
        self.stable_borderline = stable_borderline

    def load_status(self) -> dict:
        if self.noisy:
            return {
                "action": "hold",
                "message": "No high-confidence trade found.",
                "chosen": {"confidence": 0.20},
                "min_confidence_threshold": 0.58,
                "regime": {"name": "chop", "trend_strength": 0.07},
                "context_snapshot": {"volatility_state": "high"},
                "market": {"recent_close": [100, 101.8, 99.0, 101.6, 98.8, 101.9, 99.1, 101.7]},
                "comparison": {"active": True, "run_id": "atlas_wake_latest"},
            }
        if self.stable_borderline:
            return {
                "action": "hold",
                "message": "Borderline rejections under stable trend.",
                "chosen": {"confidence": 0.56},
                "min_confidence_threshold": 0.58,
                "regime": {"name": "trend", "trend_strength": 0.45},
                "context_snapshot": {"volatility_state": "elevated"},
                "market": {"recent_close": [100, 100.2, 100.4, 100.6, 100.9, 101.1, 101.3, 101.5]},
                "comparison": {"active": True, "run_id": "atlas_wake_latest"},
            }
        return {
            "action": "hold",
            "message": "Monitoring setup quality.",
            "chosen": {"confidence": 0.50},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "transition", "trend_strength": 0.18},
            "context_snapshot": {"volatility_state": "elevated"},
            "market": {"recent_close": [100, 100.2, 100.1, 100.3, 100.2, 100.35, 100.25, 100.4]},
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        if self.noisy:
            return [
                {"type": "risk", "message": "Breakout failed in noisy chop."},
                {"type": "risk", "message": "Conservative filter blocked trade due to noise."},
            ]
        if self.stable_borderline:
            return [
                {"type": "risk", "message": "Filter blocked borderline setup near threshold."},
                {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
            ]
        return [
            {"type": "risk", "message": "Filter blocked trade due to mixed confirmation."},
            {"type": "risk", "message": "Breakout follow-through mixed across candles."},
        ]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest"}


def _ask(source: _Source, q: str) -> dict:
    return AskAtlasService(source).answer(q)


def test_borderline_rejections_stable_regime_suggest_threshold_tuning() -> None:
    source = _Source(stable_borderline=True)
    tuning = PaperTuningEngine().evaluate(
        query="What small tuning do you recommend?",
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.62, "recommendation_quality_score": 0.65}, "biggest_weakness": "adaptation speed"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    assert any(c.category == "threshold_tuning_candidate" for c in tuning.candidates)
    assert all(c.promotion_allowed is False for c in tuning.candidates)


def test_noisy_regime_rejects_aggressive_tuning() -> None:
    source = _Source(noisy=True)
    tuning = PaperTuningEngine().evaluate(
        query="What small tuning do you recommend?",
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.45, "recommendation_quality_score": 0.7}, "biggest_weakness": "stability/reactivity"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    assert "aggressive_threshold_relaxation_rejected_under_noise" in tuning.rejected


def test_high_blocker_with_safe_tuning_prioritizes_blocker_aligned_candidate() -> None:
    source = _Source(stable_borderline=True)
    tuning = PaperTuningEngine().evaluate(
        query="What should be tuned first?",
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "blocker_resolution", "recommendation": "Review blocker first."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.58, "recommendation_quality_score": 0.62}, "biggest_weakness": "blocker pressure"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    assert len(tuning.candidates) >= 1
    assert tuning.candidates[0].safe_to_paper_test is True


def test_improved_acceptance_but_worse_stability_not_recommended_for_promotion() -> None:
    source = _Source(noisy=True)
    tuning = PaperTuningEngine().evaluate(
        query="Did the last paper tuning help?",
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.4, "recommendation_quality_score": 0.65}, "biggest_weakness": "stability/reactivity"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    assert any(c.outcome in {"hurts_safety", "inconclusive"} for c in tuning.comparisons)
    assert "No live" in " ".join(c.recommendation for c in tuning.comparisons) or "Do not promote" in " ".join(c.recommendation for c in tuning.comparisons)


def test_tuning_deltas_remain_bounded() -> None:
    source = _Source(stable_borderline=True)
    tuning = PaperTuningEngine().evaluate(
        query="What small tuning do you recommend?",
        status=source.load_status(),
        runtime=source.load_runtime_flags(),
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.62, "recommendation_quality_score": 0.65}, "biggest_weakness": "adaptation speed"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    for c in tuning.candidates:
        lo, hi = c.bounded_range
        assert lo <= c.proposed_delta <= hi


def test_live_mode_rejects_tuning_generation() -> None:
    source = _Source(stable_borderline=True)
    runtime = source.load_runtime_flags()
    runtime["execute_trades"] = True
    tuning = PaperTuningEngine().evaluate(
        query="What small tuning do you recommend?",
        status=source.load_status(),
        runtime=runtime,
        adaptation_summary=build_adaptation_summary(source.load_status(), source.load_events()),
        recommendation_result={"items": [{"category": "action_posture", "recommendation": "Hold and wait for confirmation."}]},
        performance_feedback={"scores": {"adaptation_effectiveness_score": 0.62, "recommendation_quality_score": 0.65}, "biggest_weakness": "adaptation speed"},
        risk_policy={"rules": ["paper mode only"]},
        strategy_knowledge={"strategies": [{"name": "continuation"}]},
        events=source.load_events(),
    )
    assert tuning.candidates == []
    assert "tuning_rejected_non_paper_mode" in tuning.rejected
    assert "paper_mode_required" in tuning.safety_gates_triggered


def test_ask_small_tuning_recommendation() -> None:
    r = _ask(_Source(stable_borderline=True), "What small tuning do you recommend?")
    assert r["intent"] == "tuning_recommendation"
    assert "Recommended paper-mode tuning:" in r["result"]


def test_ask_paper_experiment_plan() -> None:
    r = _ask(_Source(stable_borderline=True), "What paper experiment would you run next?")
    assert r["intent"] == "paper_experiment"
    assert "Safe paper experiment:" in r["result"]


def test_ask_last_paper_tuning_help() -> None:
    r = _ask(_Source(noisy=True), "Did the last paper tuning help?")
    assert r["intent"] == "tuning_comparison"
    assert "Paper tuning result:" in r["result"]
