from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

import atlas_ai.paper_compare as paper_compare
from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.paper_compare import (
    ComparisonTracker,
    PAPER_PRESET_CONSERVATIVE_FILTER_V1,
    PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A,
    PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A,
    PAPER_PRESET_SETUP_QUALITY_FILTER_V1,
    PaperRolloutConfigError,
    ShadowPaperPortfolio,
    _candidate_key_from_status,
    _enforce_paper_runtime_flags,
    _paper_preset_settings,
    prepare_comparison_run,
    _validate_rollout_pair,
)
from atlas_ai.strategies.minute_pack import PullbackTrendM15MidExitStrategy
from atlas_ai.trader import AtlasTrader


def test_shadow_paper_portfolio_metrics(tmp_path):
    settings = SimpleNamespace(
        storage_dir=str(tmp_path),
        symbol="XAUUSD",
        max_open_trades=2,
        max_risk_per_trade=0.005,
        backtest_max_hold_bars=12,
        backtest_spread_cost_bps=3.0,
        backtest_slippage_cost_bps=1.5,
    )
    store = KnowledgeStore(str(tmp_path))
    portfolio = ShadowPaperPortfolio(settings, store, "control")
    portfolio.closed_trades = [
        {
            "entry_time": "2026-04-03T10:00:00+00:00",
            "exit_time": "2026-04-03T10:05:00+00:00",
            "pnl_return": 0.01,
            "regime": "trend",
        },
        {
            "entry_time": "2026-04-03T10:06:00+00:00",
            "exit_time": "2026-04-03T10:10:00+00:00",
            "pnl_return": -0.005,
            "regime": "range",
        },
    ]
    portfolio.filtered_trades = 3
    portfolio.missed_trades = 1

    metrics = portfolio._compute_metrics()

    assert metrics["total_trades"] == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["expectancy"] > 0.0
    assert metrics["profit_factor"] > 1.0
    assert metrics["filtered_trades"] == 3
    assert metrics["missed_trades"] == 1
    assert metrics["regime_distribution"]["trend"] == 1
    assert metrics["regime_distribution"]["range"] == 1


def test_paper_preset_settings_only_flip_conservative_filter(tmp_path):
    base = SimpleNamespace(
        storage_dir=str(tmp_path),
        symbol="XAUUSD",
        timeframe="M1",
        execute_trades=True,
        meta_model_enabled=True,
        gemini_influence_enabled=True,
        research_regime_filter_enabled=True,
        conservative_filter_enabled=False,
        conservative_filter_block_late_us_trend=False,
        conservative_filter_block_post_spike=False,
        conservative_filter_block_poor_liquidity=False,
        conservative_filter_poor_liquidity_volume_ratio=0.5,
        conservative_filter_post_spike_atr_multiple=2.0,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    control = _paper_preset_settings(base, "control", tmp_path / "control", PAPER_PRESET_CONSERVATIVE_FILTER_V1)
    experiment = _paper_preset_settings(base, "experiment", tmp_path / "experiment", PAPER_PRESET_CONSERVATIVE_FILTER_V1)

    assert control.execute_trades is False
    assert experiment.execute_trades is False
    assert control.meta_model_enabled is False
    assert experiment.meta_model_enabled is False
    assert control.gemini_influence_enabled is False
    assert experiment.gemini_influence_enabled is False
    assert control.research_regime_filter_enabled is False
    assert experiment.research_regime_filter_enabled is False
    assert control.conservative_filter_enabled is False
    assert experiment.conservative_filter_enabled is True


def test_paper_preset_settings_only_flip_setup_quality_filter(tmp_path):
    base = SimpleNamespace(
        storage_dir=str(tmp_path),
        symbol="XAUUSD",
        timeframe="M1",
        execute_trades=True,
        meta_model_enabled=True,
        gemini_influence_enabled=True,
        research_regime_filter_enabled=True,
        conservative_filter_enabled=True,
        conservative_filter_block_late_us_trend=False,
        conservative_filter_block_post_spike=False,
        conservative_filter_block_poor_liquidity=False,
        conservative_filter_poor_liquidity_volume_ratio=0.5,
        conservative_filter_post_spike_atr_multiple=2.0,
        setup_quality_enabled=True,
        setup_quality_filter_enabled=False,
        setup_quality_filter_min_score=55.0,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    control = _paper_preset_settings(base, "control", tmp_path / "control", PAPER_PRESET_SETUP_QUALITY_FILTER_V1)
    experiment = _paper_preset_settings(base, "experiment", tmp_path / "experiment", PAPER_PRESET_SETUP_QUALITY_FILTER_V1)

    assert control.execute_trades is False
    assert experiment.execute_trades is False
    assert control.meta_model_enabled is False
    assert experiment.meta_model_enabled is False
    assert control.gemini_influence_enabled is False
    assert experiment.gemini_influence_enabled is False
    assert control.conservative_filter_enabled is False
    assert experiment.conservative_filter_enabled is False
    assert control.setup_quality_enabled is False
    assert experiment.setup_quality_enabled is False
    assert control.setup_quality_filter_enabled is False
    assert experiment.setup_quality_filter_enabled is True
    assert control.setup_quality_filter_min_score == 76.0
    assert experiment.setup_quality_filter_min_score == 76.0


def test_paper_preset_settings_isolates_m15_ema_stage_a_candidate(tmp_path):
    base = SimpleNamespace(
        storage_dir=str(tmp_path),
        symbol="XAUUSD",
        timeframe="M15",
        execute_trades=True,
        meta_model_enabled=True,
        gemini_influence_enabled=True,
        research_regime_filter_enabled=True,
        conservative_filter_enabled=True,
        conservative_filter_block_late_us_trend=True,
        conservative_filter_block_post_spike=True,
        conservative_filter_block_poor_liquidity=True,
        conservative_filter_poor_liquidity_volume_ratio=0.80,
        conservative_filter_post_spike_atr_multiple=1.6,
        setup_quality_enabled=True,
        setup_quality_filter_enabled=True,
        setup_quality_filter_min_score=55.0,
        enabled_strategies=("ema_crossover", "pullback_trend"),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    control = _paper_preset_settings(base, "control", tmp_path / "control", PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A)
    experiment = _paper_preset_settings(base, "experiment", tmp_path / "experiment", PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A)

    assert control.execute_trades is False
    assert experiment.execute_trades is False
    assert control.timeframe == "M15"
    assert experiment.timeframe == "M15"
    assert control.enabled_strategies == ("ema_crossover", "pullback_trend")
    assert experiment.enabled_strategies == ("ema_crossover_closed_confirm_mid_exit",)
    assert control.backtest_include_evolved is False
    assert experiment.backtest_include_evolved is False
    _validate_rollout_pair(control, experiment, PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A)


def test_paper_preset_settings_isolates_m15_pullback_mid_exit_candidate(tmp_path):
    base = SimpleNamespace(
        storage_dir=str(tmp_path),
        symbol="XAUUSD",
        timeframe="M15",
        execute_trades=True,
        meta_model_enabled=True,
        gemini_influence_enabled=True,
        research_regime_filter_enabled=True,
        conservative_filter_enabled=True,
        conservative_filter_block_late_us_trend=True,
        conservative_filter_block_post_spike=True,
        conservative_filter_block_poor_liquidity=True,
        conservative_filter_poor_liquidity_volume_ratio=0.80,
        conservative_filter_post_spike_atr_multiple=1.6,
        setup_quality_enabled=True,
        setup_quality_filter_enabled=True,
        setup_quality_filter_min_score=55.0,
        enabled_strategies=("ema_crossover", "pullback_trend"),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    control = _paper_preset_settings(base, "control", tmp_path / "control", PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A)
    experiment = _paper_preset_settings(base, "experiment", tmp_path / "experiment", PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A)

    assert control.execute_trades is False
    assert experiment.execute_trades is False
    assert control.timeframe == "M15"
    assert experiment.timeframe == "M15"
    assert control.enabled_strategies == ("ema_crossover", "pullback_trend")
    assert experiment.enabled_strategies == ("pullback_trend_m15_mid_exit",)
    assert control.backtest_include_evolved is False
    assert experiment.backtest_include_evolved is False
    _validate_rollout_pair(control, experiment, PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A)


def test_recalibrated_ema_strategy_is_paper_only_m15_variant(tmp_path, monkeypatch):
    class DummyExecutor:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setattr("atlas_ai.trader.TradeExecutor", DummyExecutor)
    settings = Settings(
        hfm_login=0,
        hfm_password="",
        hfm_server="",
        storage_dir=str(tmp_path),
        timeframe="M15",
        execute_trades=False,
        symbol="XAUUSD",
        exploration_epsilon=0.0,
    )

    trader = AtlasTrader(settings, store=KnowledgeStore(str(tmp_path)))
    names = [str(getattr(strategy, "name", "")) for strategy in trader.base_strategies]

    assert "ema_crossover_closed_confirm_mid_exit" in names
    assert "pullback_trend_m15_mid_exit" in names


def test_pullback_m15_mid_exit_uses_bar_range_scaled_exits():
    df = pd.DataFrame(
        [
            {
                "open": 2400.0 + i * 0.2,
                "high": 2400.7 + i * 0.2,
                "low": 2399.7 + i * 0.2,
                "close": 2400.1 + i * 0.2,
                "tick_volume": 100 + i,
            }
            for i in range(40)
        ]
    )

    signal = PullbackTrendM15MidExitStrategy().generate_signal(df)

    assert signal.strategy == "pullback_trend_m15_mid_exit"
    assert signal.stop_loss_points >= 1100.0
    assert signal.take_profit_points >= 1450.0


def test_validate_rollout_pair_rejects_non_filter_differences(tmp_path):
    control = SimpleNamespace(
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
        conservative_filter_enabled=False,
        conservative_filter_block_late_us_trend=True,
        conservative_filter_block_post_spike=True,
        conservative_filter_block_poor_liquidity=True,
        conservative_filter_poor_liquidity_volume_ratio=0.80,
        conservative_filter_post_spike_atr_multiple=1.6,
        symbol="XAUUSD",
        timeframe="M1",
        research_confidence_threshold=0.5,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
    )
    experiment = SimpleNamespace(**{**control.__dict__, "conservative_filter_enabled": True, "max_open_trades": 3})

    try:
        _validate_rollout_pair(control, experiment, PAPER_PRESET_CONSERVATIVE_FILTER_V1)
        assert False, "expected PaperRolloutConfigError"
    except PaperRolloutConfigError:
        pass


def test_enforce_paper_runtime_flags_rejects_live_runtime(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": True})

    try:
        _enforce_paper_runtime_flags(store, "control", "run123", PAPER_PRESET_CONSERVATIVE_FILTER_V1)
        assert False, "expected PaperRolloutConfigError"
    except PaperRolloutConfigError:
        pass


def test_candidate_key_from_status_is_lane_stable():
    settings = SimpleNamespace(symbol="XAUUSD", timeframe="M1")
    status = {
        "updated_at": "2026-04-04T11:46:23+00:00",
        "market": {"recent_time": ["2026-04-02 23:59:00"]},
        "chosen": {
            "strategy": "momentum_continuation",
            "side": "buy",
            "stop_loss_points": 115.0,
            "take_profit_points": 175.0,
        },
    }

    key = _candidate_key_from_status(status, settings)

    assert key == "XAUUSD|M1|2026-04-02 23:59:00|momentum_continuation|buy|115.0|175.0"


def test_comparison_tracker_reports_removed_winners_and_losers(tmp_path):
    tracker = ComparisonTracker(tmp_path)
    settings = SimpleNamespace(symbol="XAUUSD", timeframe="M1")
    control_status = {
        "action": "paper",
        "message": "control paper",
        "updated_at": "2026-04-04T11:46:23+00:00",
        "chosen": {
            "strategy": "momentum_continuation",
            "side": "buy",
            "confidence": 0.31,
            "stop_loss_points": 115.0,
            "take_profit_points": 175.0,
        },
        "market": {"recent_time": ["2026-04-02 23:59:00"]},
        "regime": {"name": "trend", "trend_strength": 0.24},
        "research_regime_overlay": {"session": "late_us"},
        "decision_explanation": {"final_decision": "allow"},
        "context_snapshot": {"session": "late_us", "regime": "trend"},
        "shadow_advice": {"recommendation": "caution"},
        "system_health": {"warnings": ["stale_data_detected"]},
    }
    experiment_status = {
        **control_status,
        "action": "blocked",
        "message": "Conservative pre-trade filter blocked trade: late_us trend regime.",
        "decision_explanation": {"final_decision": "block"},
        "shadow_advice": {"recommendation": "observe"},
        "system_health": {"warnings": ["routing_risk_detected"]},
    }

    tracker.record_cycle(control_status, experiment_status, settings)

    control_portfolio = SimpleNamespace(
        closed_trades=[
            {
                "candidate_key": _candidate_key_from_status(control_status, settings),
                "outcome": "loss",
                "pnl_return": -0.005,
                "exit_reason": "stop_loss",
            }
        ],
        open_trades=[],
        filtered_trades=0,
        metrics={"win_rate": 0.0, "expectancy": -0.005, "profit_factor": 0.0, "drawdown": 0.01, "average_win": 0.0, "average_loss": -0.005},
    )
    experiment_portfolio = SimpleNamespace(
        closed_trades=[],
        open_trades=[],
        filtered_trades=1,
        metrics={"win_rate": 0.0, "expectancy": 0.0, "profit_factor": 0.0, "drawdown": 0.0, "average_win": 0.0, "average_loss": 0.0},
    )

    tracker.attach_closed_outcomes(control_portfolio, experiment_portfolio)
    report = tracker.build_report(control_portfolio, experiment_portfolio)

    assert report["cross_lane"]["filtered_by_experiment"] == 1
    assert report["winners_vs_losers_removed"]["losers_avoided"] == 1
    assert report["winners_vs_losers_removed"]["winners_lost"] == 0
    assert report["winners_vs_losers_removed"]["removed_trade_expectancy"] == -0.005
    assert report["winners_vs_losers_removed"]["removed_trade_pnl_sum"] == -0.005
    assert report["winners_vs_losers_removed"]["removed_trade_outcomes_known"] == 1
    assert report["filter_effectiveness"]["assessment"] == "positive"
    assert report["lane_diagnostics"]["latest_control"]["system_health"]["warnings"] == ["stale_data_detected"]
    assert report["lane_diagnostics"]["latest_experiment"]["shadow_advice"]["recommendation"] == "observe"
    assert report["decision_checklist"]["classification"] == "CONTINUE_TESTING"
    assert "minimum_sample_reached" in report["decision_checklist"]["checks"]


def test_comparison_tracker_rejects_when_expectancy_and_trade_count_fail(tmp_path):
    tracker = ComparisonTracker(tmp_path)
    tracker.state["filtered_by_experiment"] = {
        f"f{i}": {
            "candidate_key": f"f{i}",
            "control_outcome": {"outcome": "loss" if i < 6 else "win", "pnl_return": -0.004 if i < 6 else 0.01},
            "experiment_outcome": {},
            "session": "london",
            "regime": "trend",
        }
        for i in range(10)
    }
    control_portfolio = SimpleNamespace(
        closed_trades=[
            {
                "candidate_key": f"c{i}",
                "outcome": "win" if i % 2 == 0 else "loss",
                "pnl_return": 0.01 if i % 2 == 0 else -0.004,
                "exit_reason": "mixed",
                "entry_time": f"2026-04-03T10:{i:02d}:00+00:00",
                "exit_time": f"2026-04-03T10:{i:02d}:30+00:00",
                "regime": "trend",
                "session": "london",
            }
            for i in range(120)
        ],
        open_trades=[],
        filtered_trades=0,
        metrics={
            "win_rate": 0.55,
            "expectancy": 0.002,
            "profit_factor": 1.25,
            "drawdown": 0.04,
            "average_win": 0.01,
            "average_loss": -0.004,
        },
    )
    experiment_portfolio = SimpleNamespace(
        closed_trades=[
            {
                "candidate_key": f"e{i}",
                "outcome": "win" if i < 20 else "loss",
                "pnl_return": 0.004 if i < 20 else -0.006,
                "exit_reason": "mixed",
                "entry_time": f"2026-04-03T11:{i:02d}:00+00:00",
                "exit_time": f"2026-04-03T11:{i:02d}:30+00:00",
                "regime": "trend",
                "session": "london",
            }
            for i in range(40)
        ],
        open_trades=[],
        filtered_trades=60,
        metrics={
            "win_rate": 0.50,
            "expectancy": -0.0015,
            "profit_factor": 0.82,
            "drawdown": 0.06,
            "average_win": 0.004,
            "average_loss": -0.006,
        },
    )

    report = tracker.build_report(control_portfolio, experiment_portfolio)

    assert report["decision_checklist"]["classification"] == "REJECT"
    assert report["decision_checklist"]["checks"]["trade_count_meaningful"]["status"] == "fail"
    assert report["decision_checklist"]["checks"]["expectancy_change"]["status"] == "fail"


def test_comparison_tracker_accepts_when_quality_improves_at_minimum_sample(tmp_path):
    tracker = ComparisonTracker(tmp_path)
    filtered = {}
    for i in range(25):
        outcome = "loss" if i < 18 else "win"
        filtered[f"f{i}"] = {
            "candidate_key": f"f{i}",
            "control_outcome": {"outcome": outcome, "pnl_return": -0.005 if outcome == "loss" else 0.006},
            "experiment_outcome": {},
            "session": "london" if i % 2 == 0 else "ny_open",
            "regime": "trend" if i % 3 else "range",
        }
    tracker.state["filtered_by_experiment"] = filtered

    control_portfolio = SimpleNamespace(
        closed_trades=[
            {
                "candidate_key": f"c{i}",
                "outcome": "win" if i < 48 else "loss",
                "pnl_return": 0.008 if i < 48 else -0.006,
                "exit_reason": "mixed",
                "entry_time": f"2026-04-03T09:{i%60:02d}:00+00:00",
                "exit_time": f"2026-04-03T09:{i%60:02d}:30+00:00",
                "regime": "trend" if i % 3 else "range",
                "session": "london" if i % 2 == 0 else "ny_open",
            }
            for i in range(100)
        ],
        open_trades=[],
        filtered_trades=0,
        metrics={
            "win_rate": 0.48,
            "expectancy": 0.0008,
            "profit_factor": 1.08,
            "drawdown": 0.06,
            "average_win": 0.008,
            "average_loss": -0.006,
        },
    )
    experiment_portfolio = SimpleNamespace(
        closed_trades=[
            {
                "candidate_key": f"e{i}",
                "outcome": "win" if i < 41 else "loss",
                "pnl_return": 0.009 if i < 41 else -0.0055,
                "exit_reason": "mixed",
                "entry_time": f"2026-04-03T10:{i%60:02d}:00+00:00",
                "exit_time": f"2026-04-03T10:{i%60:02d}:30+00:00",
                "regime": "trend" if i % 3 else "range",
                "session": "london" if i % 2 == 0 else "ny_open",
            }
            for i in range(75)
        ],
        open_trades=[],
        filtered_trades=25,
        metrics={
            "win_rate": 0.5467,
            "expectancy": 0.0016,
            "profit_factor": 1.32,
            "drawdown": 0.05,
            "average_win": 0.009,
            "average_loss": -0.0055,
        },
    )

    report = tracker.build_report(control_portfolio, experiment_portfolio)

    assert report["decision_checklist"]["classification"] == "ACCEPT"
    assert report["decision_checklist"]["checks"]["minimum_sample_reached"]["status"] == "pass"
    assert report["decision_checklist"]["checks"]["losers_avoided_vs_winners_removed"]["status"] == "pass"


def test_prepare_comparison_run_retries_transient_stale_market_data(tmp_path, monkeypatch):
    now = datetime(2026, 4, 6, 11, 5, tzinfo=timezone.utc)
    stale_time = now - timedelta(days=3)
    fresh_time = now - timedelta(minutes=1)

    class FakeFeed:
        def __init__(self, settings):
            self.calls = 0

        def get_rates(self, bars=50):
            self.calls += 1
            ts = stale_time if self.calls == 1 else fresh_time
            return pd.DataFrame([{"time": pd.Timestamp(ts), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}])

        def diagnostics(self, bars=50):
            ts = stale_time if self.calls == 1 else fresh_time
            return {
                "terminal_connected": True,
                "trade_allowed": True,
                "account_login": 123,
                "account_server": "demo",
                "symbol": "XAUUSD",
                "symbol_selected": True,
                "symbol_visible": True,
                "timeframe_requested": "M1",
                "latest_tick_utc": now.isoformat(),
                "latest_bar_utc": ts.isoformat(),
                "bars_received": 50,
                "timestamp_source": "live_feed",
                "mt5_last_error": (1, "Success"),
            }

    monkeypatch.setattr(paper_compare, "MarketDataFeed", FakeFeed)
    monkeypatch.setattr(paper_compare, "_now_utc", lambda: now)
    monkeypatch.setattr(paper_compare, "comparison_version_hash", lambda settings: "testhash")

    base = Settings(
        hfm_login=123,
        hfm_password="pw",
        hfm_server="demo",
        storage_dir=str(tmp_path / "storage"),
        symbol="XAUUSD",
        timeframe="M1",
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
        conservative_filter_enabled=False,
        setup_quality_enabled=False,
        setup_quality_filter_enabled=False,
        setup_quality_filter_min_score=76.0,
        conservative_filter_block_late_us_trend=True,
        conservative_filter_block_post_spike=True,
        conservative_filter_block_poor_liquidity=True,
        conservative_filter_poor_liquidity_volume_ratio=0.8,
        conservative_filter_post_spike_atr_multiple=1.6,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    prepared = prepare_comparison_run(
        settings=base,
        comparison_root=str(tmp_path / "compare"),
        run_id="retry_ok",
        preset=PAPER_PRESET_SETUP_QUALITY_FILTER_V1,
    )

    diagnostics = json.loads((tmp_path / "compare" / "retry_ok" / "preflight_diagnostics.json").read_text(encoding="utf-8"))
    assert prepared["manifest"]["market_open"] is True
    assert prepared["manifest"]["latest_bar_utc"] == fresh_time.isoformat()
    assert diagnostics["market_data"]["freshness_passed"] is True
    assert diagnostics["market_data"]["attempt"] == 2
    assert diagnostics["market_data"]["timestamp_source"] == "live_feed"


def test_prepare_comparison_run_writes_diagnostics_on_stale_failure(tmp_path, monkeypatch):
    now = datetime(2026, 4, 6, 11, 5, tzinfo=timezone.utc)
    stale_time = now - timedelta(days=3)

    class FakeFeed:
        def __init__(self, settings):
            self.calls = 0

        def get_rates(self, bars=50):
            self.calls += 1
            return pd.DataFrame([{"time": pd.Timestamp(stale_time), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0}])

        def diagnostics(self, bars=50):
            return {
                "terminal_connected": True,
                "trade_allowed": True,
                "account_login": 123,
                "account_server": "demo",
                "symbol": "XAUUSD",
                "symbol_selected": True,
                "symbol_visible": True,
                "timeframe_requested": "M1",
                "latest_tick_utc": now.isoformat(),
                "latest_bar_utc": stale_time.isoformat(),
                "bars_received": 50,
                "timestamp_source": "live_feed",
                "mt5_last_error": (1, "Success"),
            }

    monkeypatch.setattr(paper_compare, "MarketDataFeed", FakeFeed)
    monkeypatch.setattr(paper_compare, "_now_utc", lambda: now)
    monkeypatch.setattr(paper_compare, "comparison_version_hash", lambda settings: "testhash")

    base = Settings(
        hfm_login=123,
        hfm_password="pw",
        hfm_server="demo",
        storage_dir=str(tmp_path / "storage"),
        symbol="XAUUSD",
        timeframe="M1",
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
        conservative_filter_enabled=False,
        setup_quality_enabled=False,
        setup_quality_filter_enabled=False,
        setup_quality_filter_min_score=76.0,
        conservative_filter_block_late_us_trend=True,
        conservative_filter_block_post_spike=True,
        conservative_filter_block_poor_liquidity=True,
        conservative_filter_poor_liquidity_volume_ratio=0.8,
        conservative_filter_post_spike_atr_multiple=1.6,
        enabled_strategies=("momentum_continuation",),
        disabled_strategies=(),
        max_risk_per_trade=0.005,
        max_open_trades=2,
        max_spread_points=80.0,
        min_same_side_signals=2,
        backtest_max_hold_bars=12,
        backtest_include_evolved=True,
        research_confidence_threshold=0.5,
        max_trades_per_day=12,
        max_consecutive_losses=3,
    )

    try:
        prepare_comparison_run(
            settings=base,
            comparison_root=str(tmp_path / "compare"),
            run_id="retry_fail",
            preset=PAPER_PRESET_SETUP_QUALITY_FILTER_V1,
        )
        assert False, "expected stale-data failure"
    except RuntimeError as exc:
        msg = str(exc)
        assert "Threshold=480s" in msg
        assert "Source=live_feed" in msg
        assert stale_time.isoformat() in msg

    diagnostics = json.loads((tmp_path / "compare" / "retry_fail" / "preflight_diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["market_data"]["freshness_passed"] is False
    assert diagnostics["market_data"]["latest_bar_utc"] == stale_time.isoformat()
    assert diagnostics["market_data"]["latest_tick_utc"] == now.isoformat()
