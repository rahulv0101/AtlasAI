from __future__ import annotations

from atlas_ai.learning.post_trade_lessons import build_post_trade_lesson


def test_post_trade_lesson_separates_timing_from_bad_idea() -> None:
    lesson = build_post_trade_lesson(
        {
            "outcome": "loss",
            "loss_classification": "late_entry",
            "duration_minutes": 18,
            "setup_family": "breakout",
            "side": "sell",
            "entry_session": "new_york",
            "regime": "trend",
            "pattern_fingerprint": "strategy=gold_micro_impulse|side=sell",
        },
        {
            "location": "breakout",
            "momentum": "strong",
            "higher_tf_bias": "sell",
            "htf_aligned": True,
            "spread_points": 18,
        },
    )

    assert lesson["schema"] == "atlas_post_trade_lesson_v1"
    assert lesson["failure_axis"] == "timing"
    assert lesson["idea_quality"] == "possibly_valid_but_poorly_executed"
    assert lesson["timing_quality"] == "early_or_late"
    assert lesson["next_time_action"] == "wait_for_fresher_entry_or_retest_before_repeating"
    assert "cannot rewrite strategy" in lesson["policy_note"]


def test_post_trade_lesson_marks_mid_range_loss_as_structure_issue() -> None:
    lesson = build_post_trade_lesson(
        {
            "outcome": "loss",
            "loss_classification": "weak_structure",
            "duration_minutes": 9,
            "setup_family": "continuation",
            "side": "buy",
            "session": "london",
            "regime": "range",
        },
        {"location": "mid_range", "momentum": "mixed", "spread_points": 20},
    )

    assert lesson["failure_axis"] == "location_or_structure"
    assert lesson["idea_quality"] == "unproven_or_weak"
    assert lesson["timing_quality"] == "poor_location_timing"
    assert "location=mid_range" in lesson["future_recognition"]


def test_post_trade_lesson_keeps_win_as_positive_fingerprint() -> None:
    lesson = build_post_trade_lesson(
        {
            "outcome": "win",
            "setup_family": "breakout",
            "side": "sell",
            "session": "new_york",
            "regime": "trend",
        },
        {"location": "breakout", "momentum": "strong", "higher_tf_bias": "sell"},
    )

    assert lesson["failure_axis"] == "none"
    assert lesson["idea_quality"] == "validated"
    assert lesson["next_time_action"] == "remember_as_positive_fingerprint_when_conditions_match"
