from __future__ import annotations

import json
from pathlib import Path

from atlas_ai.learning.edge_building import (
    build_edge_building_report,
    score_forward_safe_setup,
    score_live_forward_safe_setup,
)
from atlas_ai.learning.setup_quality_score import build_offline_setup_quality_score, score_trade_record


def test_score_trade_record_keeps_acceptable_trade_near_neutral() -> None:
    scored = score_trade_record(
        {
            "label": "acceptable trade",
            "reason_codes": ["favorable_session", "mixed_context"],
            "trade_return": -0.0002,
        }
    )
    assert -0.15 < float(scored["setup_quality_score"]) < 0.15
    assert scored["setup_quality_bucket"] in {"neutral", "good"}


def test_score_trade_record_penalizes_low_quality_setup_even_when_winner() -> None:
    scored = score_trade_record(
        {
            "label": "low-quality setup",
            "reason_codes": ["post_spike"],
            "trade_return": 0.0071,
        }
    )
    assert float(scored["setup_quality_score"]) <= -0.35
    assert scored["setup_quality_bucket"] in {"weak", "avoid"}


def test_build_offline_setup_quality_score_creates_report(tmp_path: Path) -> None:
    label_report = {
        "ok": True,
        "records": [
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T01:39:00",
                "side": "buy",
                "strategy": "momentum_continuation",
                "label": "high-quality winner",
                "reason_codes": ["first_entry_quality", "high_quality_confirmation"],
                "trade_return": 0.0071,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T01:53:00",
                "side": "sell",
                "strategy": "momentum_continuation",
                "label": "avoidable loser",
                "reason_codes": ["poor_liquidity"],
                "trade_return": -0.00545,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T07:36:00",
                "side": "sell",
                "strategy": "evolved_rule",
                "label": "low-quality setup",
                "reason_codes": ["post_spike", "repeated_reentry"],
                "trade_return": -0.00545,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T15:00:00",
                "side": "buy",
                "strategy": "evolved_rule",
                "label": "acceptable trade",
                "reason_codes": ["mixed_context"],
                "trade_return": -0.0002,
            },
        ],
    }
    review_report = {
        "recommendations": {
            "scoring_readiness": "Labels are usable for a first offline hybrid score."
        }
    }
    label_path = tmp_path / "trade_labels_report.json"
    review_path = tmp_path / "trade_label_review.json"
    output_path = tmp_path / "trade_setup_quality_report.json"
    label_path.write_text(json.dumps(label_report), encoding="utf-8")
    review_path.write_text(json.dumps(review_report), encoding="utf-8")

    result = build_offline_setup_quality_score(label_path, review_report_path=review_path, output_path=output_path)

    assert result["ok"] is True
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["scoring_mode"] == "offline_hybrid_v1"
    assert saved["summary_by_label"]["high-quality winner"]["average_score"] > 0
    assert saved["summary_by_label"]["low-quality setup"]["average_score"] < 0


def test_forward_safe_score_is_outcome_independent() -> None:
    base = {
        "session": "late_us",
        "regime": "trend",
        "volatility_bucket": "low_volatility",
        "trend_strength_ratio": 9.1,
        "volume_ratio": 0.62,
        "repeat_entry": True,
        "post_spike_context": True,
        "reason_codes": ["late_us_trend", "poor_liquidity", "post_spike"],
    }
    winner = {**base, "trade_return": 0.0071, "outcome": "win"}
    loser = {**base, "trade_return": -0.00545, "outcome": "loss"}

    assert score_forward_safe_setup(winner) == score_forward_safe_setup(loser)


def test_build_edge_building_report_creates_report(tmp_path: Path) -> None:
    label_report = {
        "ok": True,
        "records": [
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T01:39:00",
                "session": "asia",
                "regime": "trend",
                "trend_strength_ratio": 5.2,
                "volatility_bucket": "normal_volatility",
                "volume_ratio": 1.2,
                "repeat_entry": False,
                "post_spike_context": False,
                "reason_codes": [],
                "trade_return": 0.0071,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T07:36:00",
                "session": "london",
                "regime": "trend",
                "trend_strength_ratio": 9.5,
                "volatility_bucket": "high_volatility",
                "volume_ratio": 0.7,
                "repeat_entry": True,
                "post_spike_context": True,
                "poor_liquidity": True,
                "reason_codes": ["post_spike", "poor_liquidity", "london_drag"],
                "trade_return": -0.00545,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-01T21:15:00",
                "session": "late_us",
                "regime": "trend",
                "trend_strength_ratio": 8.3,
                "volatility_bucket": "normal_volatility",
                "volume_ratio": 0.9,
                "repeat_entry": True,
                "post_spike_context": False,
                "reason_codes": ["late_us_trend", "repeated_reentry"],
                "trade_return": -0.00545,
            },
            {
                "symbol": "XAUUSD",
                "entry_time": "2026-04-02T15:00:00",
                "session": "new_york",
                "regime": "range",
                "trend_strength_ratio": 1.6,
                "volatility_bucket": "normal_volatility",
                "volume_ratio": 1.1,
                "repeat_entry": False,
                "post_spike_context": False,
                "reason_codes": [],
                "trade_return": 0.00755,
            },
        ],
    }
    label_path = tmp_path / "trade_labels_report.json"
    output_path = tmp_path / "edge_building_report.json"
    label_path.write_text(json.dumps(label_report), encoding="utf-8")

    result = build_edge_building_report(label_path, output_path=output_path, cutoff_score=74.0)

    assert result["ok"] is True
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["scoring_mode"] == "forward_safe_setup_quality_v1"
    assert saved["filter_experiment"]["trades_removed"] >= 1
    assert "top_loss_buckets" in saved["loss_breakdown"]


def test_score_live_forward_safe_setup_penalizes_known_bad_contexts() -> None:
    scored = score_live_forward_safe_setup(
        session="late_us",
        regime="trend",
        trend_strength=9.4,
        volatility_pct=0.02,
        volume_ratio=0.72,
        spike_ratio=1.9,
        same_side_support=3,
        setup_quality_reasons=["post_spike_entry", "thin_liquidity", "trend_low_volatility"],
    )

    assert scored["setup_quality_score"] < 76.0
    assert scored["setup_quality_bucket"] == "low"
    assert "late_us_trend" in scored["reasons"]
    assert "poor_liquidity" in scored["reasons"]
