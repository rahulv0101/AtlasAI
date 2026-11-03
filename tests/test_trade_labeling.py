from __future__ import annotations

from atlas_ai.learning.trade_labeling import label_trade


def test_label_trade_marks_high_quality_winner_with_positive_reasons_only():
    label, reasons, confidence = label_trade(
        {
            "session": "asia",
            "regime": "trend",
            "post_spike_context": False,
            "london_extreme_volatility": False,
            "trend_low_volatility": False,
            "weak_consensus": False,
            "poor_liquidity": False,
            "repeated_reentry": False,
            "first_entry_quality": True,
            "strong_trend_clean_entry": True,
            "favorable_session": True,
            "high_quality_confirmation": True,
            "outcome": "win",
        }
    )
    assert label == "high-quality winner"
    assert "strong_trend_clean_entry" in reasons
    assert "post_spike" not in reasons
    assert confidence == "high"


def test_label_trade_marks_avoidable_loser_from_negative_context():
    label, reasons, confidence = label_trade(
        {
            "session": "london",
            "regime": "trend",
            "post_spike_context": False,
            "london_extreme_volatility": True,
            "trend_low_volatility": True,
            "weak_consensus": False,
            "poor_liquidity": True,
            "repeated_reentry": False,
            "first_entry_quality": False,
            "strong_trend_clean_entry": False,
            "favorable_session": False,
            "high_quality_confirmation": False,
            "outcome": "loss",
        }
    )
    assert label in {"avoidable loser", "low-quality setup"}
    assert "london_drag" in reasons or "low_vol_trend" in reasons
    assert confidence == "high"


def test_label_trade_marks_low_quality_setup_regardless_of_outcome_when_severe():
    label, reasons, confidence = label_trade(
        {
            "session": "late_us",
            "regime": "trend",
            "post_spike_context": True,
            "london_extreme_volatility": False,
            "trend_low_volatility": False,
            "weak_consensus": False,
            "poor_liquidity": False,
            "repeated_reentry": True,
            "first_entry_quality": False,
            "strong_trend_clean_entry": False,
            "favorable_session": False,
            "high_quality_confirmation": False,
            "outcome": "win",
        }
    )
    assert label == "low-quality setup"
    assert "late_us_trend" in reasons
    assert "post_spike" in reasons
    assert confidence == "high"
