from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from atlas_ai.backtest import Backtester
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal


def _sample_df(rows: int = 900) -> pd.DataFrame:
    base = 2000.0
    data = []
    for i in range(rows):
        drift = 0.08 * i
        wave = ((i % 20) - 10) * 1.7
        open_px = base + drift + wave
        close_px = open_px + (1.8 if i % 3 else -1.2)
        high_px = max(open_px, close_px) + 1.5
        low_px = min(open_px, close_px) - 1.5
        data.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tick_volume": 120 + (i % 9) * 8,
            }
        )
    return pd.DataFrame(data)


def test_backtester_refreshes_evolved_strategy_periodically(monkeypatch):
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=3,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=200,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=12,
        evolved_strategy_population=18,
        backtest_include_evolved=True,
        enabled_strategies=(),
        disabled_strategies=(),
    )
    backtester = Backtester(settings=settings)
    calls: list[int] = []

    class DummyStrategy:
        name = "evolved_rule"

        def generate_signal(self, window):
            return TradeSignal(self.name, "flat", 0.0, 100, 150)

    def fake_evolve(window, generations, population):
        calls.append(len(window))
        return DummyStrategy()

    monkeypatch.setattr(backtester.lab, "evolve", fake_evolve)

    result = backtester.run(_sample_df())

    assert result.trades >= 0
    assert calls[0] == 251
    assert all(x <= 500 for x in calls)
    assert result.max_drawdown >= 0.0
    assert result.average_hold_bars >= 0.0
    assert isinstance(result.regime_performance, dict)


def test_backtester_does_not_mutate_live_allocator_state(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "allocator": {
                "momentum_continuation": {"wins": 10, "losses": 5, "pnl": 123.0},
            }
        }
    )
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=3,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=200,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=12,
        evolved_strategy_population=18,
        backtest_include_evolved=False,
        enabled_strategies=(),
        disabled_strategies=(),
    )

    Backtester(store=store, settings=settings).run(_sample_df())

    allocator = store.load_learning_state().get("state", {}).get("allocator", {})
    assert allocator["momentum_continuation"]["wins"] == 10
    assert allocator["momentum_continuation"]["losses"] == 5
    assert allocator["momentum_continuation"]["pnl"] == 123.0


def test_backtester_can_persist_allocator_learning_when_opted_in(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=3,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=200,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=12,
        evolved_strategy_population=18,
        backtest_include_evolved=False,
        backtest_persist_learning_state=True,
        enabled_strategies=(),
        disabled_strategies=(),
    )

    Backtester(store=store, settings=settings).run(_sample_df())

    allocator = store.load_learning_state().get("state", {}).get("allocator", {})
    assert allocator


def test_backtester_persists_structured_loss_memory_when_opted_in(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=3,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=200,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=12,
        evolved_strategy_population=18,
        backtest_include_evolved=False,
        backtest_persist_learning_state=True,
        timeframe_minutes=1,
        enabled_strategies=(),
        disabled_strategies=(),
    )

    Backtester(store=store, settings=settings).run(_sample_df())

    state = store.load_learning_state().get("state", {})
    memory = dict(state.get("loss_memory", {}) or {})
    outcomes = list(memory.get("recent_outcomes", []) or [])
    assert outcomes
    sample = outcomes[-1]
    assert sample["post_trade_review"]["bounded_learning_only"] is True
    assert sample["post_trade_review"]["source"] == "historical_backtest"
    assert sample["outcome"] in {"win", "loss", "draw"}
    assert "side_hour_stats" in memory
    assert "setup_family" in sample
    assert "best_mfe_r" in sample


def test_backtester_classifies_false_continuation_and_late_entry():
    backtester = Backtester()

    false_cont = backtester._classify_backtest_loss(
        outcome="loss",
        exit_reason="stop_loss",
        duration_minutes=3.0,
        session="london",
        regime="trend",
        best_mfe_r=0.1,
        setup_family="continuation",
    )
    late_entry = backtester._classify_backtest_loss(
        outcome="loss",
        exit_reason="stop_loss",
        duration_minutes=8.0,
        session="london",
        regime="trend",
        best_mfe_r=0.6,
        setup_family="pullback",
    )

    assert false_cont == "false_continuation"
    assert late_entry == "late_entry"


def test_backtester_reports_risk_metrics():
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=4,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=250,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=8,
        evolved_strategy_population=8,
        backtest_include_evolved=False,
        enabled_strategies=(),
        disabled_strategies=(),
    )
    result = Backtester(settings=settings).run(_sample_df())
    assert isinstance(result.expectancy, float)
    assert isinstance(result.profit_factor, float)
    assert isinstance(result.max_drawdown, float)
    assert isinstance(result.sharpe, float)
    assert isinstance(result.sortino, float)
    assert isinstance(result.risk_of_ruin, float)
    assert isinstance(result.win_rate, float)


def test_backtester_respects_custom_initial_balance():
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=4,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=250,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=8,
        evolved_strategy_population=8,
        backtest_include_evolved=False,
        backtest_initial_balance=100.0,
        enabled_strategies=(),
        disabled_strategies=(),
    )
    result = Backtester(settings=settings).run(_sample_df())
    assert isinstance(result.total_return, float)


def test_backtester_win_rate_uses_only_closed_wins_and_losses():
    backtester = Backtester()

    total_trades, wins, losses, win_rate = backtester._closed_trade_counts([0.01, -0.02, 0.0, 0.03, -0.01, 0.0])

    assert total_trades == 6
    assert wins == 2
    assert losses == 2
    assert win_rate == 0.5


def test_backtester_strategy_allowlist_filters_base_universe():
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=4,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=250,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=8,
        evolved_strategy_population=8,
        backtest_include_evolved=False,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
    )
    backtester = Backtester(settings=settings)
    assert backtester._strategy_enabled("momentum_continuation") is True
    assert backtester._strategy_enabled("ema_crossover") is False


def test_simulate_trade_supports_break_even_and_stale_exit_management():
    settings = SimpleNamespace(
        exploration_epsilon=0.0,
        max_risk_per_trade=0.005,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=6,
        backtest_point_size=0.01,
        backtest_break_even_r_trigger=0.5,
        backtest_break_even_lock_r=0.0,
        backtest_trailing_r_trigger=1.2,
        backtest_trailing_buffer_r=0.8,
        backtest_stale_exit_bars=3,
        backtest_stale_min_progress_r=0.1,
        backtest_evolved_refresh_bars=250,
        backtest_evolved_lookback_bars=500,
        evolved_strategy_generations=8,
        evolved_strategy_population=8,
        backtest_include_evolved=False,
        enabled_strategies=(),
        disabled_strategies=(),
    )
    backtester = Backtester(settings=settings)
    df = pd.DataFrame(
        [
            {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "tick_volume": 100}
            for _ in range(260)
        ]
        + [
            {"open": 100.0, "high": 100.8, "low": 99.9, "close": 100.7, "tick_volume": 150},
            {"open": 100.7, "high": 100.75, "low": 100.0, "close": 100.1, "tick_volume": 120},
            {"open": 100.1, "high": 100.2, "low": 99.95, "close": 100.05, "tick_volume": 110},
        ]
    )
    signal = TradeSignal("test", "buy", 0.5, 100.0, 200.0)
    r_mult, exit_idx, best_mfe_r = backtester._simulate_trade(df, signal, 259)
    assert r_mult >= 0.0
    assert exit_idx >= 260
    assert best_mfe_r >= 0.0


def test_backtester_research_regime_overlay_is_opt_in(monkeypatch):
    class DummyStrategy:
        name = "momentum_continuation"

        def generate_signal(self, window):
            return TradeSignal(self.name, "buy", 0.9, 100.0, 150.0)

    base_kwargs = {
        "exploration_epsilon": 0.0,
        "max_risk_per_trade": 0.005,
        "backtest_spread_cost_bps": 0.0,
        "backtest_slippage_cost_bps": 0.0,
        "backtest_max_hold_bars": 3,
        "backtest_point_size": 0.01,
        "backtest_evolved_refresh_bars": 250,
        "backtest_evolved_lookback_bars": 500,
        "evolved_strategy_generations": 8,
        "evolved_strategy_population": 8,
        "backtest_include_evolved": False,
        "enabled_strategies": (),
        "disabled_strategies": (),
        "research_confidence_threshold": 0.5,
    }

    settings = SimpleNamespace(**base_kwargs, research_regime_filter_enabled=False)
    backtester = Backtester(settings=settings)
    backtester.base = [DummyStrategy()]
    monkeypatch.setattr(backtester.allocator, "choose", lambda signals: signals[0])
    monkeypatch.setattr(backtester, "_simulate_trade", lambda df, signal, i: (0.5, i + 1))
    result_unfiltered = backtester.run(_sample_df())
    assert result_unfiltered.trades > 0

    settings_filtered = SimpleNamespace(**base_kwargs, research_regime_filter_enabled=True)
    backtester_filtered = Backtester(settings=settings_filtered)
    backtester_filtered.base = [DummyStrategy()]
    monkeypatch.setattr(backtester_filtered.allocator, "choose", lambda signals: signals[0])
    monkeypatch.setattr(backtester_filtered, "_simulate_trade", lambda df, signal, i: (0.5, i + 1))
    monkeypatch.setattr("atlas_ai.backtest.research_regime_filter_reason", lambda settings, context: "blocked by overlay")
    result_filtered = backtester_filtered.run(_sample_df())
    assert result_filtered.trades == 0


def test_backtester_conservative_filter_is_opt_in(monkeypatch):
    class DummyStrategy:
        name = "momentum_continuation"

        def generate_signal(self, window):
            return TradeSignal(self.name, "buy", 0.9, 100.0, 150.0)

    base_kwargs = {
        "exploration_epsilon": 0.0,
        "max_risk_per_trade": 0.005,
        "backtest_spread_cost_bps": 0.0,
        "backtest_slippage_cost_bps": 0.0,
        "backtest_max_hold_bars": 3,
        "backtest_point_size": 0.01,
        "backtest_evolved_refresh_bars": 250,
        "backtest_evolved_lookback_bars": 500,
        "evolved_strategy_generations": 8,
        "evolved_strategy_population": 8,
        "backtest_include_evolved": False,
        "enabled_strategies": (),
        "disabled_strategies": (),
        "research_confidence_threshold": 0.5,
        "research_regime_filter_enabled": False,
    }

    settings = SimpleNamespace(**base_kwargs, conservative_filter_enabled=False)
    backtester = Backtester(settings=settings)
    backtester.base = [DummyStrategy()]
    monkeypatch.setattr(backtester.allocator, "choose", lambda signals: signals[0])
    monkeypatch.setattr(backtester, "_simulate_trade", lambda df, signal, i: (0.5, i + 1))
    result_unfiltered = backtester.run(_sample_df())
    assert result_unfiltered.trades > 0

    settings_filtered = SimpleNamespace(**base_kwargs, conservative_filter_enabled=True)
    backtester_filtered = Backtester(settings=settings_filtered)
    backtester_filtered.base = [DummyStrategy()]
    monkeypatch.setattr(backtester_filtered.allocator, "choose", lambda signals: signals[0])
    monkeypatch.setattr(backtester_filtered, "_simulate_trade", lambda df, signal, i: (0.5, i + 1))
    monkeypatch.setattr("atlas_ai.backtest.conservative_pretrade_filter_reason", lambda settings, context: "blocked by conservative filter")
    result_filtered = backtester_filtered.run(_sample_df())
    assert result_filtered.trades == 0
