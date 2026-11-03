from datetime import datetime, timedelta, timezone

from atlas_ai.config.settings import Settings
from atlas_ai.learning.experiment_runner import build_joined_decision_outcome_samples, readiness_report, run_meta_model_experiment
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _settings(storage_dir: str) -> Settings:
    return Settings(
        hfm_login=1,
        hfm_password="x",
        hfm_server="demo",
        storage_dir=storage_dir,
    )


def test_build_joined_decision_outcome_samples_matches_prior_decisions(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    base_time = datetime.now(timezone.utc) + timedelta(minutes=20)
    for i in range(30):
        strategy = "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep"
        side = "buy" if i % 2 == 0 else "sell"
        session = "london" if i % 2 == 0 else "new_york"
        store.log_event(
            "decision_sample",
            "test decision",
            {
                "strategy": strategy,
                "side": side,
                "confidence": 0.25 + (i % 3) * 0.05,
                "timeframe": "M1",
                "regime": "trend" if i % 2 == 0 else "range",
                "higher_tf_bias": side,
                "session": session,
                "hour_utc": 8 + (i % 6),
                "spread_points": 15 + (i % 4),
                "same_side_support": 2 + (i % 2),
                "m1_quality": {"factor": 1.02 if i % 2 == 0 else 0.94},
            },
        )

    learning_state = {
        "loss_memory": {
            "recent_outcomes": [
                {
                    "ticket": str(i),
                    "time": (base_time + timedelta(minutes=i)).isoformat(),
                    "side": "buy" if i % 2 == 0 else "sell",
                    "hour_utc": 8 + (i % 6),
                    "net_pnl_zar": 50.0 if i % 2 == 0 else -40.0,
                    "outcome": "win" if i % 2 == 0 else "loss",
                    "session": "london" if i % 2 == 0 else "new_york",
                    "strategy": "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep",
                }
                for i in range(30)
            ]
        }
    }
    store.save_learning_state(learning_state)

    samples = build_joined_decision_outcome_samples(store)
    assert len(samples) > 0
    assert any("timeframe:M1" in s.features for s in samples)


def test_run_meta_model_experiment_prefers_joined_dataset(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    base_time = datetime.now(timezone.utc) + timedelta(minutes=20)
    for i in range(30):
        store.log_event(
            "decision_sample",
            "test decision",
            {
                "strategy": "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep",
                "side": "buy" if i % 2 == 0 else "sell",
                "confidence": 0.3,
                "timeframe": "M1",
                "regime": "trend" if i % 2 == 0 else "range",
                "higher_tf_bias": "buy" if i % 2 == 0 else "sell",
                "session": "london",
                "hour_utc": 9,
                "spread_points": 18.0,
                "same_side_support": 3,
                "m1_quality": {"factor": 1.04},
            },
        )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": str(i),
                        "time": (base_time + timedelta(minutes=i)).isoformat(),
                        "side": "buy" if i % 2 == 0 else "sell",
                        "hour_utc": 9,
                        "net_pnl_zar": 80.0 if i % 2 == 0 else -60.0,
                        "outcome": "win" if i % 2 == 0 else "loss",
                        "session": "london",
                        "strategy": "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep",
                    }
                    for i in range(30)
                ]
            }
        }
    )

    result = run_meta_model_experiment(_settings(str(tmp_path)), store)
    assert result["ok"] is True
    assert result["dataset_source"] == "decision_outcome_joined"
    assert result["joined_samples"] >= 24


def test_joined_samples_ignore_failed_execution_attempts(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    event_time = datetime.now(timezone.utc)
    store.log_event(
        "decision_sample",
        "failed live decision",
        {
            "decision_id": "deadbeef",
            "strategy": "gold_micro_impulse",
            "side": "buy",
            "confidence": 0.3,
            "timeframe": "M1",
            "regime": "trend",
            "higher_tf_bias": "buy",
            "session": "london",
            "hour_utc": 9,
            "spread_points": 18.0,
            "same_side_support": 3,
            "m1_quality": {"factor": 1.04},
            "execution_success": False,
        },
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": "1",
                        "decision_id": "deadbeef",
                        "time": (event_time + timedelta(minutes=5)).isoformat(),
                        "side": "buy",
                        "hour_utc": 9,
                        "net_pnl_zar": 80.0,
                        "outcome": "win",
                        "session": "london",
                        "strategy": "gold_micro_impulse",
                    }
                ]
            }
        }
    )

    samples = build_joined_decision_outcome_samples(store)
    assert samples == []


def test_joined_samples_prefer_exact_decision_id_match(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    early = datetime.now(timezone.utc)
    late = early + timedelta(minutes=1)
    store.log_event(
        "decision_sample",
        "first live decision",
        {
            "decision_id": "wrong111",
            "strategy": "gold_micro_impulse",
            "side": "buy",
            "confidence": 0.2,
            "timeframe": "M1",
            "regime": "range",
            "higher_tf_bias": "sell",
            "session": "london",
            "hour_utc": 9,
            "spread_points": 20.0,
            "same_side_support": 2,
            "m1_quality": {"factor": 0.9},
            "execution_success": True,
        },
    )
    store.log_event(
        "decision_sample",
        "second live decision",
        {
            "decision_id": "match123",
            "strategy": "gold_micro_impulse",
            "side": "buy",
            "confidence": 0.4,
            "timeframe": "M1",
            "regime": "trend",
            "higher_tf_bias": "buy",
            "session": "london",
            "hour_utc": 9,
            "spread_points": 14.0,
            "same_side_support": 3,
            "m1_quality": {"factor": 1.08},
            "execution_success": True,
        },
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": "1",
                        "decision_id": "match123",
                        "time": (late + timedelta(minutes=5)).isoformat(),
                        "side": "buy",
                        "hour_utc": 9,
                        "net_pnl_zar": 80.0,
                        "outcome": "win",
                        "session": "london",
                        "strategy": "gold_micro_impulse",
                    }
                ]
            }
        }
    )

    samples = build_joined_decision_outcome_samples(store)
    assert len(samples) == 1
    assert "regime:trend" in samples[0].features


def test_readiness_report_counts_joined_progress(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    base_time = datetime.now(timezone.utc) + timedelta(minutes=20)
    for i in range(5):
        store.log_event(
            "decision_sample",
            "test decision",
            {
                "strategy": "gold_micro_impulse",
                "side": "buy",
                "confidence": 0.3,
                "timeframe": "M1",
                "regime": "trend",
                "higher_tf_bias": "buy",
                "session": "london",
                "hour_utc": 9,
                "spread_points": 18.0,
                "same_side_support": 3,
                "m1_quality": {"factor": 1.04},
            },
        )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": str(i),
                        "time": (base_time + timedelta(minutes=i)).isoformat(),
                        "side": "buy",
                        "hour_utc": 9,
                        "net_pnl_zar": 80.0,
                        "outcome": "win",
                        "session": "london",
                        "strategy": "gold_micro_impulse",
                    }
                    for i in range(5)
                ]
            }
        }
    )
    report = readiness_report(store)
    assert report["decision_samples"] == 5
    assert report["closed_outcomes"] == 5
    assert report["joined_samples"] == 5
    assert report["joined_training_ready"] is False
    assert report["samples_needed_for_joined_training"] == 19
    assert "learning_stability" in report


def test_joined_samples_can_fallback_when_outcome_strategy_is_unknown(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    base_time = datetime.now(timezone.utc)
    store.log_event(
        "decision_sample",
        "test decision",
        {
            "decision_id": "abc12345",
            "strategy": "gold_micro_impulse",
            "side": "buy",
            "confidence": 0.3,
            "timeframe": "M1",
            "regime": "trend",
            "higher_tf_bias": "buy",
            "session": "new_york",
            "hour_utc": 14,
            "spread_points": 18.0,
            "same_side_support": 3,
            "m1_quality": {"factor": 1.04},
            "execution_success": True,
        },
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": "1",
                        "time": (base_time + timedelta(minutes=5)).isoformat(),
                        "side": "buy",
                        "hour_utc": 14,
                        "net_pnl_zar": 80.0,
                        "outcome": "win",
                        "session": "new_york",
                        "strategy": "unknown",
                    }
                ]
            }
        }
    )

    samples = build_joined_decision_outcome_samples(store)

    assert len(samples) == 1
    assert "strategy:gold_micro_impulse" in samples[0].features


def test_run_meta_model_experiment_respects_promotion_cooldown(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    base_time = datetime.now(timezone.utc) + timedelta(minutes=20)
    for i in range(40):
        store.log_event(
            "decision_sample",
            "test decision",
            {
                "decision_id": f"case{i}",
                "strategy": "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep",
                "side": "buy" if i % 2 == 0 else "sell",
                "confidence": 0.3 + ((i % 4) * 0.02),
                "timeframe": "M1",
                "regime": "trend" if i % 2 == 0 else "range",
                "higher_tf_bias": "buy" if i % 2 == 0 else "sell",
                "session": "london",
                "hour_utc": 9,
                "spread_points": 18.0,
                "same_side_support": 3,
                "m1_quality": {"factor": 1.04},
                "execution_success": True,
            },
        )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": str(i),
                        "decision_id": f"case{i}",
                        "time": (base_time + timedelta(minutes=i)).isoformat(),
                        "side": "buy" if i % 2 == 0 else "sell",
                        "hour_utc": 9,
                        "net_pnl_zar": 80.0 if i % 2 == 0 else -60.0,
                        "outcome": "win" if i % 2 == 0 else "loss",
                        "session": "london",
                        "strategy": "gold_micro_impulse" if i % 2 == 0 else "gold_liquidity_sweep",
                    }
                    for i in range(40)
                ]
            },
            "model_registry": {
                "last_promoted_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    )

    result = run_meta_model_experiment(_settings(str(tmp_path)), store)

    assert result["ok"] is True
    assert result["promotion"]["cooldown_active"] is True
    assert result["promotion"]["promoted"] is False
