from __future__ import annotations

from atlas_ai.learning.online_learning import build_online_learning_observation, update_online_learning_state


def test_online_learning_observation_is_report_only_for_noisy_late_session() -> None:
    status = {
        "action": "hold",
        "chosen": {"strategy": "evolved_rule(fast=18,slow=87)", "side": "sell", "confidence": 0.38},
        "risk": {"spread_points": 45},
        "context_snapshot": {
            "session": "late_us",
            "momentum": "mixed",
            "location": "mid_range",
            "spread_points": 45,
        },
        "decision_analysis": {
            "confluence_score": 42,
            "confluence_threshold": 50,
            "hard_blockers": ["spread_liquidity_too_poor", "no_clear_setup"],
        },
    }
    sample = {
        "fingerprint": "regime=range|momentum=mixed|location=mid_range",
        "teaching_focus": ["wait for a cleaner edge"],
    }
    learning = {"total_samples": 3, "top_patterns": [{"fingerprint": sample["fingerprint"], "count": 3}]}

    out = build_online_learning_observation(status=status, near_setup_sample=sample, near_setup_learning=learning)

    assert out["allowed_influence"] == "report_only"
    assert out["mode"] == "slow_path_observe"
    assert out["overreaction_risk"] > 0.5
    assert "cannot rewrite strategies" in out["policy_note"]


def test_online_learning_can_classify_persistent_evidence_without_policy_change() -> None:
    status = {
        "action": "blocked",
        "chosen": {"strategy": "gold_breakout_retest", "side": "sell", "confidence": 0.6},
        "context_snapshot": {"session": "new_york", "momentum": "strong", "location": "breakout", "spread_points": 25},
        "decision_analysis": {
            "confluence_score": 47,
            "confluence_threshold": 50,
            "hard_blockers": ["no_clear_setup"],
        },
    }
    sample = {"fingerprint": "breakout_retest_gap", "teaching_focus": ["confirm retest"]}
    learning = {"total_samples": 18, "top_patterns": [{"fingerprint": "breakout_retest_gap", "count": 18}]}

    out = build_online_learning_observation(status=status, near_setup_sample=sample, near_setup_learning=learning)

    assert out["mode"] == "fast_path_observe"
    assert out["allowed_influence"] == "bounded_confidence_haircut"
    assert out["confidence_haircut_cap"] > 0.0
    assert out["evidence_strength"] >= 0.78
    assert out["lesson_digest"]["recommended_use"] == "promote_to_operator_lesson"
    assert "cannot rewrite strategies" in out["lesson_digest"]["policy_note"]


def test_online_learning_state_keeps_recent_observations_and_top_patterns() -> None:
    state = {}
    first = {"top_fingerprint": "a", "allowed_influence": "bounded_confidence_haircut", "updated_at": "2026-05-14T00:00:00+00:00"}
    second = {"top_fingerprint": "a", "allowed_influence": "bounded_confidence_haircut", "updated_at": "2026-05-14T00:01:00+00:00"}

    state = update_online_learning_state(state, first)
    state = update_online_learning_state(state, second)

    online = state["online_learning"]
    assert online["allowed_influence"] == "bounded_confidence_haircut"
    assert online["top_fingerprints"][0]["fingerprint"] == "a"
    assert online["top_fingerprints"][0]["count"] == 2
    assert "bounded confidence haircuts" in online["policy_note"]


def test_online_learning_state_exposes_latest_lesson_digest() -> None:
    obs = {
        "top_fingerprint": "mid_range_weak",
        "allowed_influence": "report_only",
        "updated_at": "2026-05-14T00:00:00+00:00",
        "lesson_digest": {"primary_lesson": "wait for pullback location", "confidence": "moderate"},
    }
    state = update_online_learning_state({}, obs)

    assert state["online_learning"]["lesson_digest"]["primary_lesson"] == "wait for pullback location"
