from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.learning.outcome_intelligence import (
    build_trade_outcome_intelligence,
    build_trade_outcome_memory,
    refresh_trade_outcome_intelligence,
)
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _seed_decisions_and_outcomes(store: KnowledgeStore) -> None:
    base = datetime.now(timezone.utc) + timedelta(minutes=20)
    for i in range(18):
        decision_time = base + timedelta(minutes=i * 2)
        side = "buy" if i % 2 == 0 else "sell"
        strategy = "gold_micro_impulse" if i % 3 == 0 else "gold_liquidity_sweep"
        session = "london" if i < 9 else "new_york"
        regime = "trend" if i % 3 != 1 else "range"
        confidence = 0.72 if i % 2 == 0 else 0.26
        trend_strength = 0.31 if regime == "trend" else 0.08
        store.log_event(
            "decision_sample",
            "captured decision",
            {
                "decision_id": f"dec{i}",
                "strategy": strategy,
                "side": side,
                "confidence": confidence,
                "timeframe": "M1",
                "regime": regime,
                "higher_tf_bias": side if i % 4 != 0 else ("sell" if side == "buy" else "buy"),
                "session": session,
                "hour_utc": 8 + (i % 6),
                "spread_points": 16.0 + (i % 5),
                "same_side_support": 3 if i % 2 == 0 else 1,
                "same_side_ratio": 0.65 if i % 2 == 0 else 0.25,
                "volatility_pct": 0.041 if regime == "trend" else 0.018,
                "trend_strength": trend_strength,
                "m1_quality": {"factor": 1.04 if i % 2 == 0 else 0.93},
                "meta_model": {"probability": 0.62 if i % 2 == 0 else 0.38},
                "contextual_ranking": {"setup_quality": {"score": 0.14 if i % 2 == 0 else -0.08}},
                "execution_success": True,
                "time": decision_time.isoformat(),
            },
        )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": str(i),
                        "decision_id": f"dec{i}",
                        "time": (base + timedelta(minutes=(i * 2) + (8 if i % 2 == 0 else 5))).isoformat(),
                        "side": "buy" if i % 2 == 0 else "sell",
                        "hour_utc": 8 + (i % 6),
                        "net_pnl_zar": 120.0 if i % 2 == 0 else -90.0,
                        "outcome": "win" if i % 2 == 0 else "loss",
                        "session": "london" if i < 9 else "new_york",
                        "strategy": "gold_micro_impulse" if i % 3 == 0 else "gold_liquidity_sweep",
                    }
                    for i in range(18)
                ]
            }
        }
    )


def test_build_trade_outcome_memory_enriches_trade_conditions(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    _seed_decisions_and_outcomes(store)

    rows = build_trade_outcome_memory(store)

    assert len(rows) == 18
    assert rows[0]["duration_minutes"] > 0
    assert rows[0]["confidence_bucket"] in {"low", "medium", "good", "high"}
    assert rows[0]["volatility_bucket"] in {"low", "normal", "high"}
    assert "htf_aligned" in rows[0]
    assert rows[0]["post_trade_lesson"]["schema"] == "atlas_post_trade_lesson_v1"
    assert rows[0]["post_trade_lesson"]["bounded_learning_only"] is True


def test_build_trade_outcome_intelligence_produces_patterns_calibration_and_regime_memory(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    _seed_decisions_and_outcomes(store)

    intelligence = build_trade_outcome_intelligence(store)

    assert intelligence["sample_count"] == 18
    assert intelligence["winner_patterns"]["samples"] == 9
    assert intelligence["loser_patterns"]["samples"] == 9
    assert intelligence["confidence_calibration"]["samples"] == 18
    assert intelligence["signal_scoring_model"]["status"] == "trained"
    assert "by_regime" in intelligence["regime_memory"]
    assert intelligence["pattern_comparison"]["main_signal_gap"]


def test_refresh_trade_outcome_intelligence_persists_learning_state(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    _seed_decisions_and_outcomes(store)

    refresh_trade_outcome_intelligence(store)

    state = store.load_learning_state().get("state", {})
    assert "trade_outcome_intelligence" in state
    assert state["trade_outcome_intelligence"]["sample_count"] == 18
