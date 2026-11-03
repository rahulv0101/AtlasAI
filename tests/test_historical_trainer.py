from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from atlas_ai.learning.historical_trainer import HistoricalTrainer
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _sample_df(rows: int = 900) -> pd.DataFrame:
    base = 2000.0
    data = []
    for i in range(rows):
        drift = 0.05 * i
        wave = ((i % 20) - 10) * 1.4
        open_px = base + drift + wave
        close_px = open_px + (1.5 if i % 3 else -1.1)
        high_px = max(open_px, close_px) + 1.2
        low_px = min(open_px, close_px) - 1.2
        data.append(
            {
                "time": pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(minutes=i),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tick_volume": 100 + (i % 7) * 8,
            }
        )
    return pd.DataFrame(data)


def _settings(storage_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        storage_dir=storage_dir,
        symbol="XAUUSD",
        timeframe="M1",
        historical_train_months=6,
        historical_train_max_bars=1200,
        historical_backtest_max_bars=800,
        historical_training_stage_timeout_seconds=5.0,
        evolved_strategy_train_max_bars=700,
        evolved_strategy_generations=8,
        evolved_strategy_population=10,
        evolved_strategy_survivors=3,
        evolved_candidate_min_trades=200,
        evolved_candidate_min_win_rate=0.52,
        evolved_candidate_min_profit_component=0.0,
        training_quality_validation_slices=3,
        training_quality_min_slice_win_rate=0.48,
        training_quality_min_slice_profit_component=-0.002,
        training_quality_validation_weight=0.0,
    )


def test_historical_trainer_logs_failure_event_on_fetch_error(tmp_path):
    settings = _settings(str(tmp_path))
    store = KnowledgeStore(settings.storage_dir)
    trainer = HistoricalTrainer(settings, store)

    class BrokenFeed:
        last_fetch_meta = {"ok": False, "error": "boom"}

        def get_rates_range(self, start, end):
            raise RuntimeError("range failed")

        def get_rates(self, bars):
            raise RuntimeError("fallback failed")

    trainer.feed = BrokenFeed()

    with pytest.raises(RuntimeError):
        trainer.train()

    events = store.tail_events(10)
    assert any(e.get("message") == "Historical training started" for e in events)
    failed = [e for e in events if e.get("message") == "Historical training failed"]
    assert failed
    assert "fallback failed" in str(failed[-1].get("meta", {}).get("error", ""))


def test_historical_trainer_logs_progress_and_falls_back_when_no_candidates(tmp_path, monkeypatch):
    settings = _settings(str(tmp_path))
    store = KnowledgeStore(settings.storage_dir)
    trainer = HistoricalTrainer(settings, store)

    class FakeFeed:
        last_fetch_meta = {"ok": True, "bars_received": 900}

        def get_rates_range(self, start, end):
            return _sample_df()

    trainer.feed = FakeFeed()

    class FakeBacktester:
        def __init__(self, store=None, settings=None):
            self.store = store
            self.settings = settings

        def run(self, df):
            return SimpleNamespace(
                total_return=0.12,
                trades=44,
                wins=24,
                losses=20,
                expectancy=0.01,
                profit_factor=1.1,
                max_drawdown=0.04,
                sharpe=0.8,
                sortino=1.0,
                average_win=0.03,
                average_loss=-0.02,
                risk_of_ruin=0.05,
                average_hold_bars=2.0,
                regime_performance={"trend": {"trades": 10}},
            )

    class FakeStrategy:
        fast = 7
        slow = 31
        rsi_buy = 48.0
        rsi_sell = 52.0

    class FakeLab:
        def __init__(self, seed=17):
            self.seed = seed

        def discover_candidates(self, df, generations, population, survivors, max_seconds=None, progress_callback=None):
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "discover_candidates",
                        "generation": 1,
                        "generations": generations,
                        "best_score": 0.42,
                        "elapsed_seconds": 0.1,
                    }
                )
            return []

        def optimize_for_profit(self, df, generations=20, population=30, max_seconds=None, progress_callback=None):
            return FakeStrategy(), 0.33

    monkeypatch.setattr("atlas_ai.learning.historical_trainer.Backtester", FakeBacktester)
    monkeypatch.setattr("atlas_ai.learning.historical_trainer.StrategyLab", FakeLab)

    result = trainer.train()

    assert result.trades == 44
    state = store.load_learning_state().get("state", {})
    assert state["best_evolved_strategy"]["source"] == "optimize_for_profit_fallback"
    events = store.tail_events(20)
    messages = [e.get("message") for e in events]
    assert "Historical training started" in messages
    assert "Historical training fetched history" in messages
    assert "Historical training backtest completed" in messages
    assert "Historical training progress" in messages
    assert "Historical training candidate search yielded no promoted pool; using fallback" in messages
    assert "Historical training completed" in messages


def test_historical_trainer_logs_weak_point_summary_for_promoted_candidates(tmp_path, monkeypatch):
    settings = _settings(str(tmp_path))
    settings.training_trend_robustness_weight = 0.45
    settings.training_range_dependency_penalty = 0.18
    settings.training_trend_loss_penalty_weight = 0.25
    store = KnowledgeStore(settings.storage_dir)
    trainer = HistoricalTrainer(settings, store)

    class FakeFeed:
        last_fetch_meta = {"ok": True, "bars_received": 900}

        def get_rates_range(self, start, end):
            return _sample_df()

    trainer.feed = FakeFeed()

    class FakeBacktester:
        def __init__(self, store=None, settings=None):
            self.store = store
            self.settings = settings

        def run(self, df):
            return SimpleNamespace(
                total_return=0.19,
                trades=80,
                wins=42,
                losses=38,
                expectancy=0.01,
                profit_factor=1.2,
                max_drawdown=0.04,
                sharpe=0.8,
                sortino=1.0,
                average_win=0.03,
                average_loss=-0.02,
                risk_of_ruin=0.05,
                average_hold_bars=2.0,
                regime_performance={"trend": {"trades": 40}},
            )

    class FakeCandidate:
        def __init__(self):
            self.fast = 34
            self.slow = 98
            self.rsi_buy = 49.8
            self.rsi_sell = 45.9
            self.score = 0.61
            self.trades = 220
            self.wins = 115
            self.losses = 105
            self.win_rate = 115 / 220
            self.profit_component = 0.23
            self.confidence_floor = 0.58
            self.generations = 8
            self.population = 10
            self.trend_trades = 140
            self.trend_wins = 60
            self.trend_losses = 80
            self.trend_win_rate = 60 / 140
            self.trend_profit_component = -0.03
            self.range_trades = 80
            self.range_wins = 55
            self.range_losses = 25
            self.range_win_rate = 55 / 80
            self.range_profit_component = 0.12
            self.trend_edge_score = -0.02
            self.range_dependency_score = 0.15
            self.stop_hunt_vulnerability = 0.09
            self.explainability_summary = "trend edge is weak; performance leans on range behavior; stop-hunt vulnerability is elevated"
            self.source = "mutation_tournament"

        def signature(self):
            return "f34_s98_rb49.8_rs45.9"

        def to_dict(self):
            return {
                "fast": self.fast,
                "slow": self.slow,
                "rsi_buy": self.rsi_buy,
                "rsi_sell": self.rsi_sell,
                "score": self.score,
                "trades": self.trades,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": self.win_rate,
                "profit_component": self.profit_component,
                "confidence_floor": self.confidence_floor,
                "generations": self.generations,
                "population": self.population,
                "trend_trades": self.trend_trades,
                "trend_wins": self.trend_wins,
                "trend_losses": self.trend_losses,
                "trend_win_rate": self.trend_win_rate,
                "trend_profit_component": self.trend_profit_component,
                "range_trades": self.range_trades,
                "range_wins": self.range_wins,
                "range_losses": self.range_losses,
                "range_win_rate": self.range_win_rate,
                "range_profit_component": self.range_profit_component,
                "trend_edge_score": self.trend_edge_score,
                "range_dependency_score": self.range_dependency_score,
                "stop_hunt_vulnerability": self.stop_hunt_vulnerability,
                "explainability_summary": self.explainability_summary,
                "signature": self.signature(),
                "source": self.source,
            }

    class FakeLab:
        def __init__(self, seed=17):
            self.seed = seed

        def discover_candidates(self, df, generations, population, survivors, max_seconds=None, progress_callback=None, score_adjuster=None):
            return [FakeCandidate()]

        def optimize_for_profit(self, df, generations=20, population=30, max_seconds=None, progress_callback=None):
            fallback = SimpleNamespace(fast=7, slow=31, rsi_buy=48.0, rsi_sell=52.0)
            return fallback, 0.33

    monkeypatch.setattr("atlas_ai.learning.historical_trainer.Backtester", FakeBacktester)
    monkeypatch.setattr("atlas_ai.learning.historical_trainer.StrategyLab", FakeLab)

    trainer.train()

    events = store.tail_events(12)
    completed = [e for e in events if e.get("message") == "Historical training completed"]
    fallbacks = [e for e in events if e.get("message") == "Historical training candidate search yielded no promoted pool; using fallback"]
    assert completed
    assert fallbacks
    promotion_meta = dict(fallbacks[-1].get("meta", {}).get("promotion_meta", {}))
    rejected = list(promotion_meta.get("rejected", []))
    assert rejected
    assert "weak_trend_edge" in rejected[0].get("reasons", [])
    assert "trend edge is still weak" in dict(completed[-1].get("meta", {}).get("weak_point_summary", {})).get("weaknesses", [])


def test_candidate_score_adjuster_can_reward_profit_and_penalize_regression(tmp_path):
    settings = _settings(str(tmp_path))
    settings.training_trend_robustness_weight = 0.0
    settings.training_range_dependency_penalty = 0.0
    settings.training_trend_loss_penalty_weight = 0.0
    settings.training_profit_alignment_weight = 0.8
    settings.training_profit_regression_penalty_weight = 1.2
    settings.training_min_win_rate_alignment = 0.4
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    adjuster = trainer._candidate_score_adjuster()

    better = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.10,
            "win_rate": 0.58,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.56,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
        }
    )
    worse = adjuster(
        {
            "score": 0.22,
            "profit_component": -0.03,
            "win_rate": 0.46,
            "trend_profit_component": 0.03,
            "trend_win_rate": 0.55,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.00,
        }
    )

    assert better > worse


def test_candidate_score_adjuster_can_reward_pullback_quality_and_penalize_wick_pressure(tmp_path):
    settings = _settings(str(tmp_path))
    settings.training_pullback_quality_weight = 0.8
    settings.training_wick_pressure_penalty_weight = 0.9
    settings.training_trend_maturity_weight = 0.6
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    adjuster = trainer._candidate_score_adjuster()

    cleaner = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.04,
            "win_rate": 0.53,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.54,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
            "pullback_quality_score": 0.70,
            "wick_pressure_score": 0.15,
            "trend_maturity_score": 0.62,
        }
    )
    noisier = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.04,
            "win_rate": 0.53,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.54,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
            "pullback_quality_score": 0.25,
            "wick_pressure_score": 0.60,
            "trend_maturity_score": 0.20,
        }
    )

    assert cleaner > noisier


def test_promote_candidates_accepts_near_miss_with_good_quality(tmp_path):
    settings = _settings(str(tmp_path))
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    class Candidate:
        def __init__(self):
            self.trades = 175
            self.win_rate = 0.505
            self.profit_component = 0.001
            self.trend_edge_score = 0.02
            self.range_dependency_score = 0.0
            self.stop_hunt_vulnerability = 0.0

        def signature(self):
            return "near_miss"

    promoted, meta = trainer._promote_candidates([Candidate()])

    assert len(promoted) == 1
    assert meta["mode"] == "near_miss"


def test_promote_candidates_rejects_low_quality_candidate(tmp_path):
    settings = _settings(str(tmp_path))
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    class Candidate:
        def __init__(self):
            self.trades = 120
            self.win_rate = 0.46
            self.profit_component = -0.02
            self.trend_edge_score = -0.01
            self.range_dependency_score = 0.08
            self.stop_hunt_vulnerability = 0.12

        def signature(self):
            return "bad_candidate"

    promoted, meta = trainer._promote_candidates([Candidate()])

    assert promoted == []
    assert meta["mode"] == "fallback_only"
    assert meta["rejected"][0]["signature"] == "bad_candidate"


def test_candidate_score_adjuster_can_reward_target_win_rate_and_penalize_overtrading(tmp_path):
    settings = _settings(str(tmp_path))
    settings.training_target_candidate_win_rate = 0.58
    settings.training_overtrade_penalty_weight = 0.8
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    adjuster = trainer._candidate_score_adjuster()

    cleaner = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.03,
            "win_rate": 0.60,
            "trades": 220,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.56,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
        }
    )
    noisier = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.03,
            "win_rate": 0.54,
            "trades": 420,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.56,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
        }
    )

    assert cleaner > noisier


def test_candidate_score_adjuster_can_reward_continuation_quality(tmp_path):
    settings = _settings(str(tmp_path))
    settings.training_continuation_quality_weight = 0.7
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    adjuster = trainer._candidate_score_adjuster()

    cleaner = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.03,
            "win_rate": 0.54,
            "trades": 220,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.56,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
            "continuation_quality_score": 0.72,
        }
    )
    faker = adjuster(
        {
            "score": 0.20,
            "profit_component": 0.03,
            "win_rate": 0.54,
            "trades": 220,
            "trend_profit_component": 0.02,
            "trend_win_rate": 0.56,
            "trend_losses": 45,
            "trend_wins": 55,
            "range_profit_component": 0.01,
            "continuation_quality_score": 0.20,
        }
    )

    assert cleaner > faker


def test_setup_family_summary_groups_promoted_candidates(tmp_path):
    settings = _settings(str(tmp_path))
    trainer = HistoricalTrainer(settings, KnowledgeStore(settings.storage_dir))

    summary = trainer._setup_family_summary(
        {
            "candidates": [
                {"fast": 37, "slow": 98, "rsi_buy": 48.8, "rsi_sell": 49.2, "score": 0.09, "win_rate": 0.53, "trend_edge_score": 0.028},
                {"fast": 35, "slow": 103, "rsi_buy": 49.9, "rsi_sell": 48.3, "score": 0.08, "win_rate": 0.52, "trend_edge_score": 0.018},
                {"fast": 8, "slow": 84, "rsi_buy": 37.8, "rsi_sell": 57.2, "score": 0.04, "win_rate": 0.55, "trend_edge_score": 0.009},
            ]
        }
    )

    assert summary["best_family"] in {"pullback", "mean_reversion", "continuation", "hybrid", "breakout"}
    assert summary["families"]


def test_recent_slice_quality_rejects_fragile_candidate(tmp_path):
    settings = _settings(str(tmp_path))
    store = KnowledgeStore(settings.storage_dir)
    trainer = HistoricalTrainer(settings, store)
    df = _sample_df(1200)

    class Candidate:
        fast = 34
        slow = 98
        rsi_buy = 49.8
        rsi_sell = 45.9

        def signature(self):
            return "fragile"

    class FakeLab:
        def __init__(self):
            self.calls = 0

        def evaluate_strategy(self, strategy, chunk):
            self.calls += 1
            if self.calls == 2:
                return {"trades": 80, "win_rate": 0.41, "profit_component": -0.02}
            return {"trades": 80, "win_rate": 0.55, "profit_component": 0.03}

    vetted, rejected = trainer._filter_candidates_by_recent_slice_quality([Candidate()], df, FakeLab())

    assert vetted == []
    assert rejected
    assert rejected[0]["reason"] == "fragile_recent_slices"
    assert len(rejected[0]["validated_slices"]) >= 2


def test_recent_slice_quality_keeps_stable_candidate(tmp_path):
    settings = _settings(str(tmp_path))
    store = KnowledgeStore(settings.storage_dir)
    trainer = HistoricalTrainer(settings, store)
    df = _sample_df(1200)

    class Candidate:
        fast = 34
        slow = 98
        rsi_buy = 49.8
        rsi_sell = 45.9

        def signature(self):
            return "stable"

    class FakeLab:
        def evaluate_strategy(self, strategy, chunk):
            return {"trades": 85, "win_rate": 0.56, "profit_component": 0.025}

    vetted, rejected = trainer._filter_candidates_by_recent_slice_quality([Candidate()], df, FakeLab())

    assert len(vetted) == 1
    assert rejected == []
    assert getattr(vetted[0], "quality_validation_score", 0.0) > 0.0
