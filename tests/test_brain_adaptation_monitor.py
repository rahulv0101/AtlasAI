from __future__ import annotations

from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.18, "side": "flat"},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "trend", "trend_strength": 0.09},
            "context_snapshot": {"volatility_state": "high", "session": "london", "spread_state": "tight"},
            "market": {"recent_close": [100, 100.3, 99.7, 100.4, 99.8, 100.5, 99.9, 100.6]},
            "comparison": {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "risk", "message": "Conservative pre-trade filter blocked trade: poor liquidity."},
            {"type": "risk", "message": "Breakout follow-through failed in recent window."},
        ]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}


def _ask(q: str) -> dict:
    return AskAtlasService(_Source()).answer(q)


def test_adaptation_speed_question() -> None:
    r = _ask("How fast are you adapting to the market?")
    assert r["intent"] == "adaptation_status"
    assert "Adaptation status:" in r["result"]


def test_learning_from_charts_question() -> None:
    r = _ask("What are you learning from the charts?")
    assert r["intent"] in {"chart_learning", "adaptation_status"}
    assert "Observed changes:" in r["result"] or "Adaptation status:" in r["result"]


def test_current_regime_change_question() -> None:
    r = _ask("What changed in the current regime?")
    assert r["intent"] in {"regime_analysis", "comparison"}
    assert "regime" in r["result"].lower() or "comparison" in r["result"].lower() or "observed changes" in r["result"].lower()


def test_overreacting_question() -> None:
    r = _ask("Are you overreacting?")
    assert r["intent"] == "adaptation_status"
    assert "stability" in r["result"].lower() or "overreactive" in r["result"].lower()


def test_setup_quality_direction_question() -> None:
    r = _ask("Is setup quality getting better or worse?")
    assert r["intent"] in {"chart_learning", "adaptation_status"}
    assert "setup quality change" in r["result"].lower()


def test_learning_speed_metrics_exist() -> None:
    src = _Source()
    summary = build_adaptation_summary(src.load_status(), src.load_events())
    for key in (
        "active_learning_mode",
        "evidence_strength_score",
        "evidence_to_detection_latency",
        "detection_to_confidence_update_latency",
        "detection_to_behavior_change_latency",
        "trades_to_stable_adjustment",
        "regime_learning_speed",
        "setup_decay_detection_speed",
        "recovery_learning_speed",
        "stability_score",
    ):
        assert key in summary


def test_how_fast_are_you_learning_question() -> None:
    r = _ask("How fast are you learning?")
    assert r["intent"] == "adaptation_status"
    assert "learning speed" in r["result"].lower()


def test_detect_regime_change_speed_question() -> None:
    r = _ask("How quickly did you detect this regime change?")
    assert r["intent"] in {"adaptation_status", "regime_analysis"}
    assert "detection" in r["result"].lower() or "regime" in r["result"].lower()


def test_confidence_adjustment_latency_question() -> None:
    r = _ask("How long does it take you to adjust confidence?")
    assert r["intent"] == "adaptation_status"
    assert "confidence update latency" in r["result"].lower() or "confidence adjustment speed" in r["result"].lower()


def test_learning_slow_or_overreacting_question() -> None:
    r = _ask("Are you learning too slowly or overreacting?")
    assert r["intent"] == "adaptation_status"
    assert "overreaction" in r["result"].lower() or "stability" in r["result"].lower() or "adaptation speed" in r["result"].lower()


def _strong_persistent_status() -> dict:
    return {
        "action": "hold",
        "message": "Monitoring trend continuation quality.",
        "chosen": {"confidence": 0.72, "side": "long"},
        "min_confidence_threshold": 0.58,
        "regime": {"name": "trend", "trend_strength": 0.52},
        "context_snapshot": {"volatility_state": "high"},
        "market": {"recent_close": [100, 100.2, 100.4, 100.7, 101.0, 101.2, 101.4, 101.7]},
    }


def _noisy_fake_move_status() -> dict:
    return {
        "action": "hold",
        "message": "No clear setup.",
        "chosen": {"confidence": 0.31, "side": "flat"},
        "min_confidence_threshold": 0.58,
        "regime": {"name": "chop", "trend_strength": 0.06},
        "context_snapshot": {"volatility_state": "high"},
        "market": {"recent_close": [100, 101.5, 99.2, 101.8, 98.9, 101.6, 99.0, 101.7]},
    }


def _moderate_shift_status() -> dict:
    return {
        "action": "hold",
        "message": "Watching developing regime shift.",
        "chosen": {"confidence": 0.52, "side": "flat"},
        "min_confidence_threshold": 0.58,
        "regime": {"name": "transition", "trend_strength": 0.16},
        "context_snapshot": {"volatility_state": "elevated"},
        "market": {"recent_close": [100, 100.2, 100.05, 100.25, 100.1, 100.3, 100.22, 100.35]},
    }


def test_strong_persistent_trend_selects_fast_path() -> None:
    status = _strong_persistent_status()
    events = [
        {"type": "risk", "message": "Breakout follow-through confirmed on recent windows."},
        {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
        {"type": "risk", "message": "Filter blocked trade due to conservative paper guard."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["active_learning_mode"] == "fast_path"
    assert s["evidence_strength_score"] >= 0.8
    assert s["stability"] in {"stable", "noisy"}


def test_short_noisy_fake_move_selects_slow_path() -> None:
    status = _noisy_fake_move_status()
    events = [
        {"type": "risk", "message": "Breakout failed quickly after entry zone touched."},
        {"type": "risk", "message": "Conservative filter blocked trade: noisy range."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["active_learning_mode"] == "slow_path"
    assert s["evidence_strength_score"] < 0.5
    assert abs(float(s["bounded_confidence_delta"])) <= float(s["max_confidence_delta_per_update"])


def test_moderate_regime_shift_selects_medium_path() -> None:
    status = _moderate_shift_status()
    events = [
        {"type": "risk", "message": "Breakout follow-through mixed across recent candles."},
        {"type": "risk", "message": "Filter blocked one setup due to spread widening."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["active_learning_mode"] == "medium_path"
    assert 0.5 <= s["evidence_strength_score"] < 0.8


def test_repeated_breakout_failures_raise_decay_speed() -> None:
    status = _noisy_fake_move_status()
    events = [
        {"type": "risk", "message": "Breakout follow-through failed near resistance."},
        {"type": "risk", "message": "Breakout failed and setup was blocked."},
        {"type": "risk", "message": "Breakout failed again in noisy session."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["setup_decay_detection_speed"] >= 0.5


def test_mixed_contradictory_evidence_blocks_fast_path() -> None:
    status = _strong_persistent_status()
    events = [
        {"type": "risk", "message": "Breakout follow-through confirmed in trend window."},
        {"type": "risk", "message": "Breakout failed immediately in choppy move."},
        {"type": "risk", "message": "Conservative filter blocked trade due to conflicting structure."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["active_learning_mode"] != "fast_path"
    assert "fast_path_blocked_high_contradiction" in s["guardrails_triggered"] or s["evidence_strength_score"] < 0.8


def test_repeated_clean_operator_teaching_improves_persistence_without_overreaction() -> None:
    status = _strong_persistent_status()
    events = [
        {"type": "learning", "message": "Operator teaching review recorded: correct_hold"},
        {"type": "learning", "message": "Operator teaching review recorded: correct_hold"},
        {"type": "learning", "message": "Operator teaching review recorded: execution_issue"},
        {"type": "learning", "message": "Operator teaching review recorded: good_trade"},
        {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
    ]
    s = build_adaptation_summary(status, events)
    comps = dict(s["evidence_components"])
    assert comps["operator_teaching_persistence"] > 0.0
    assert comps["operator_teaching_confirmation"] > 0.0
    assert s["evidence_strength_score"] >= 0.65
    assert s["active_learning_mode"] in {"medium_path", "fast_path"}
    assert s["overreaction_score"] < 0.65


def test_operator_teaching_does_not_override_noisy_contradictory_context() -> None:
    status = _noisy_fake_move_status()
    events = [
        {"type": "learning", "message": "Operator teaching review recorded: missed_trade"},
        {"type": "learning", "message": "Operator teaching review recorded: missed_trade"},
        {"type": "learning", "message": "Operator teaching review recorded: too_cautious"},
        {"type": "risk", "message": "Breakout failed immediately in noisy range."},
        {"type": "risk", "message": "Conservative filter blocked trade due to conflicting structure."},
    ]
    s = build_adaptation_summary(status, events)
    assert s["active_learning_mode"] != "fast_path"
    assert "fast_path_blocked_high_noise" in s["guardrails_triggered"] or s["evidence_strength_score"] < 0.65


def test_learning_mode_question_reports_path() -> None:
    r = _ask("Are you on fast-path or slow-path right now?")
    assert r["intent"] in {"learning_speed", "adaptation_status"}
    assert "learning mode" in r["result"].lower()


def test_direct_learning_speed_question_reports_adaptation_metrics() -> None:
    r = _ask("How fast do you learn?")
    assert r["intent"] in {"learning_speed", "adaptation_status"}
    text = r["result"].lower()
    assert "learning mode" in text
    assert "learning speed" in text
    assert "no core strategy rule changed automatically" in text


def test_evidence_strength_question_reports_score() -> None:
    r = _ask("How strong is the current evidence?")
    assert r["intent"] in {"evidence_strength", "adaptation_status"}
    assert "evidence strength score" in r["result"].lower() or "evidence strength" in r["result"].lower()
