"""Tests for the EvolutionScheduler daemon.

Coverage:
- Scheduler runs evolve_generation on interval
- Scheduler does not execute trades
- Scheduler writes reporter artifacts
- Scheduler applies PromotionGovernor rules
- Scheduler handles evolution failure without crashing live trading
- Scheduler does not start duplicates
- CLI evolution --once runs successfully
- Runtime/status exposes scheduler state
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool
from atlas_ai.strategies.evolution_scheduler import (
    EvolutionScheduler,
    SchedulerState,
    start_evolution_scheduler,
    stop_evolution_scheduler,
)
from atlas_ai.strategies.evolution_drawdown_guard import EvolutionDrawdownGuard
from atlas_ai.strategies.evolution_engine import EvolutionEngine
from atlas_ai.strategies.evolution_reporter import EvolutionReporter
from atlas_ai.strategies.promotion_governor import PromotionGovernor


@pytest.fixture
def mock_settings():
    """Mock settings with evolution enabled."""
    settings = MagicMock()
    settings.evolution_enabled = True
    settings.evolution_interval_hours = 0.01  # 36 seconds for fast tests
    settings.max_demo_strategies = 3
    settings.evolved_max_risk_pct = 0.005
    settings.evolution_min_shadow_trades = 5
    settings.evolution_max_shadow_drawdown_pct = 0.05
    settings.evolution_min_demo_trades = 5
    settings.evolution_min_demo_win_rate = 0.50
    settings.evolution_min_demo_profit_factor = 1.10
    settings.live_promotion_requires_approval = True
    settings.evolution_max_demo_drawdown_pct = 0.05
    settings.backtest_evolved_refresh_bars = 50
    settings.backtest_evolved_lookback_bars = 500
    settings.evolved_strategy_population = 8
    settings.symbol = "XAUUSD"
    settings.timeframe = "M15"
    settings.higher_timeframe_confirm = "M5"
    settings.backtest_spread_cost_bps = 3.0
    settings.backtest_slippage_cost_bps = 1.5
    settings.backtest_max_hold_bars = 12
    settings.backtest_point_size = 0.01
    settings.backtest_break_even_r_trigger = 0.9
    settings.backtest_break_even_lock_r = 0.05
    settings.backtest_trailing_r_trigger = 1.4
    settings.backtest_trailing_buffer_r = 0.9
    settings.backtest_stale_exit_bars = 4
    settings.backtest_stale_min_progress_r = 0.15
    settings.evolved_candidate_min_trades = 200
    settings.evolved_candidate_min_win_rate = 0.52
    return settings


@pytest.fixture
def mock_store(tmp_path: Path):
    """Mock knowledge store backed by temp directory."""
    store = MagicMock()
    store.base = str(tmp_path)
    store.log_event = MagicMock()
    store.load_runtime_flags = MagicMock(return_value={})
    store.save_runtime_flags = MagicMock()
    store.load_status = MagicMock(return_value={})
    store.save_status = MagicMock()
    store.load_learning_state = MagicMock(return_value={"state": {}})
    store.update_learning_state = MagicMock()
    store.tail_events = MagicMock(return_value=[])
    store.load_events = MagicMock(return_value=[])
    store.maintain_events = MagicMock(return_value={})
    return store


@pytest.fixture
def sample_df():
    """Minimal market data dataframe for testing."""
    return pd.DataFrame({
        "open": [2000.0 + i * 0.5 for i in range(60)],
        "high": [2001.0 + i * 0.5 for i in range(60)],
        "low": [1999.0 + i * 0.5 for i in range(60)],
        "close": [2000.5 + i * 0.5 for i in range(60)],
        "tick_volume": [100 + i for i in range(60)],
        "time": pd.date_range("2024-01-01", periods=60, freq="15min"),
    })


# ------------------------------------------------------------------
# Scheduler construction and state
# ------------------------------------------------------------------


def test_scheduler_state_roundtrip():
    """SchedulerState can be serialized and deserialized."""
    state = SchedulerState(
        scheduler_status="running",
        current_generation=5,
        best_fitness=72.5,
        best_uid="abc123",
        active_demo_count=2,
        total_runs=10,
    )
    serialized = state.to_dict()
    restored = SchedulerState.from_dict(serialized)
    assert restored.scheduler_status == "running"
    assert restored.current_generation == 5
    assert restored.best_fitness == 72.5
    assert restored.best_uid == "abc123"
    assert restored.active_demo_count == 2
    assert restored.total_runs == 10


def test_scheduler_persists_state(mock_settings, mock_store):
    """Scheduler writes state to disk on cycle completion."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    state_file = Path(mock_store.base) / "evolution_scheduler_state.json"
    # State is not persisted on init; only after a cycle runs
    assert not state_file.exists()
    scheduler._save_state()
    assert state_file.exists()
    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["scheduler_status"] == "idle"
    assert "current_generation" in payload


# ------------------------------------------------------------------
# Safety: scheduler never executes trades
# ------------------------------------------------------------------


def test_scheduler_has_no_execute_method(mock_settings, mock_store):
    """Pure architecture regression: EvolutionScheduler has no execute() method."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    assert not hasattr(scheduler, "execute")
    assert not hasattr(scheduler, "place_order")
    assert not hasattr(scheduler, "send_to_broker")


def test_scheduler_engine_never_executes_trades(mock_settings, mock_store):
    """EvolutionEngine wrapped by scheduler has no execute capability."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    assert not hasattr(scheduler.engine, "execute")
    assert not hasattr(scheduler.engine, "place_order")


# ------------------------------------------------------------------
# Scheduler cycle execution
# ------------------------------------------------------------------


def test_scheduler_run_once_dry_run(mock_settings, mock_store, sample_df):
    """run_once() completes without error in dry-run mode."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=True)
    summary = scheduler.run_once(df=sample_df)
    assert isinstance(summary, dict)
    assert "generation" in summary
    assert "best_fitness" in summary
    assert summary["dry_run"] is True
    assert summary["duration_seconds"] >= 0.0


def test_scheduler_run_once_writes_artifacts(mock_settings, mock_store, sample_df):
    """run_once() writes leaderboard.json and evolution_log.json (non-dry-run)."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=False)
    # Pre-seed population so evolve_generation has something to work with
    scheduler.engine.seed_population(size=4)
    scheduler.run_once(df=sample_df)

    leaderboard_file = Path(mock_store.base) / "leaderboard.json"
    log_file = Path(mock_store.base) / "evolution_log.json"
    assert leaderboard_file.exists()
    assert log_file.exists()

    lb = json.loads(leaderboard_file.read_text(encoding="utf-8"))
    assert "leaderboard" in lb
    assert "stages" in lb
    assert "top_candidate" in lb

    el = json.loads(log_file.read_text(encoding="utf-8"))
    assert "events" in el


def test_scheduler_run_once_seeds_ai_invented_strategies(mock_settings, mock_store, sample_df):
    """Non-dry evolution seeds AI-designed proposals before mutation/backtest."""
    mock_store.load_status.return_value = {
        "primary_blocker": "confidence below threshold",
        "context_snapshot": {"regime": "trend"},
    }
    mock_store.load_runtime_flags.return_value = {}
    mock_store.load_learning_state.return_value = {"state": {}}
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=False)

    summary = scheduler.run_once(df=sample_df)

    assert summary["ai_invented_seeded"] > 0
    assert any(
        call.args
        and call.args[0] == "evolution"
        and "AI-invented strategy proposals" in str(call.args[1])
        for call in mock_store.log_event.call_args_list
    )
    assert not hasattr(scheduler, "execute")
    assert not hasattr(scheduler.engine, "execute")
    assert (Path(mock_store.base) / "ai_decision_journal.jsonl").exists()
    assert (Path(mock_store.base) / "ai_promotion_recommendations.json").exists()
    assert "ai_promotion_top" in summary


# ------------------------------------------------------------------
# PromotionGovernor integration
# ------------------------------------------------------------------


def test_scheduler_applies_promotion_governor(mock_settings, mock_store, sample_df):
    """Scheduler respects PromotionGovernor rules for demo cap."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=True)

    # Pre-seed population with max_demo strategies already active
    for i in range(mock_settings.max_demo_strategies):
        c = StrategyChromosome(
            uid=f"demo_{i}", generation=1, genes=make_standard_gene_pool(),
            promotion_stage="demo", demo_active=True,
            backtest_completed=True, backtest_score=70.0,
        )
        scheduler.engine.population.append(c)

    # Add a shadow candidate
    shadow = StrategyChromosome(
        uid="shadow_1", generation=1, genes=make_standard_gene_pool(),
        promotion_stage="shadow", demo_active=False,
        backtest_completed=True, backtest_score=65.0, shadow_passed=True,
    )
    scheduler.engine.population.append(shadow)

    summary = scheduler.run_once(df=sample_df)

    # Because demo cap is already reached, no new promotion should happen
    assert summary["promotions"] == 0
    assert shadow.promotion_stage == "shadow"


def test_scheduler_dry_run_does_not_persist_backtested_to_shadow(mock_settings, mock_store, sample_df):
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=True)
    candidate = StrategyChromosome(
        uid="backtested_1",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="backtested",
        backtest_completed=True,
        backtest_score=65.0,
    )
    scheduler.engine.population.append(candidate)

    summary = scheduler.run_once(df=sample_df)

    assert summary["dry_run"] is True
    assert candidate.promotion_stage == "backtested"
    assert candidate.promotion_reason == ""


def test_scheduler_drawdown_guard_pauses_strategy(mock_settings, mock_store, sample_df):
    """Drawdown guard pauses demo strategies exceeding threshold."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=True)

    # Create a demo strategy with bad shadow outcomes
    c = StrategyChromosome(
        uid="bad_demo", generation=1, genes=make_standard_gene_pool(),
        promotion_stage="demo", demo_active=True,
        backtest_completed=True, backtest_score=60.0,
    )
    scheduler.engine.population.append(c)

    # Inject fake shadow outcomes with large drawdown
    from atlas_ai.strategies.shadow_tester import ShadowOutcome
    scheduler._shadow_outcomes["bad_demo"] = [
        ShadowOutcome("bad_demo", "s", "buy", 0.5, 100.0, 105.0, 5.0, 1, "tp"),
        ShadowOutcome("bad_demo", "s", "buy", 0.5, 105.0, 90.0, -15.0, 1, "sl"),
    ]

    scheduler.run_once(df=sample_df)
    assert not c.demo_active
    assert "paused" in c.promotion_reason


# ------------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------------


def test_scheduler_handles_failure_gracefully(mock_settings, mock_store):
    """Scheduler _loop catches exceptions, logs them, and updates state."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    # Override interval to avoid 300s minimum clamp
    scheduler.interval_seconds = 0.01

    # Force an error by providing an invalid dataframe via patched _run_cycle
    call_count = 0
    def failing_run_cycle(df=None):
        nonlocal call_count
        call_count += 1
        # Set stop immediately so loop doesn't wait 300s
        stop_event.set()
        raise RuntimeError("simulated df failure")

    scheduler._run_cycle = failing_run_cycle

    # Run loop once — it should catch the exception and update state
    scheduler._loop()

    assert call_count >= 1
    assert scheduler.state.last_error != ""
    assert scheduler.state.scheduler_status in {"idle", "failed"}


def test_scheduler_loop_recovers_from_failure(mock_settings, mock_store, sample_df):
    """The _loop catches exceptions and continues to the next iteration."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    # Override interval to avoid 300s minimum clamp
    scheduler.interval_seconds = 0.01

    call_count = 0
    original_run_cycle = scheduler._run_cycle

    def flaky_run_cycle(df=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated failure")
        # On second call, succeed but also signal stop so loop exits
        stop_event.set()
        return original_run_cycle(df)

    scheduler._run_cycle = flaky_run_cycle

    # Run loop — first call fails, second call succeeds and sets stop
    scheduler._loop()

    # The loop should have attempted at least twice
    assert call_count >= 2
    assert scheduler.state.last_error != ""


# ------------------------------------------------------------------
# Singleton / duplicate prevention
# ------------------------------------------------------------------


def test_scheduler_singleton_prevents_duplicates(mock_settings, mock_store):
    """start_evolution_scheduler returns None on duplicate attempt."""
    stop_event = threading.Event()
    scheduler1 = start_evolution_scheduler(mock_settings, mock_store, stop_event)
    assert scheduler1 is not None

    # Second start should return None
    scheduler2 = start_evolution_scheduler(mock_settings, mock_store, stop_event)
    assert scheduler2 is None

    stop_evolution_scheduler(scheduler1)


def test_stop_evolution_scheduler_releases_lock(mock_settings, mock_store):
    """Stopping a scheduler allows a new one to start."""
    stop_event = threading.Event()
    scheduler1 = start_evolution_scheduler(mock_settings, mock_store, stop_event)
    assert scheduler1 is not None

    stop_evolution_scheduler(scheduler1)

    # After stopping, a new scheduler can start
    stop_event2 = threading.Event()
    scheduler2 = start_evolution_scheduler(mock_settings, mock_store, stop_event2)
    assert scheduler2 is not None
    stop_evolution_scheduler(scheduler2)


# ------------------------------------------------------------------
# Background thread behavior
# ------------------------------------------------------------------


def test_scheduler_background_thread_runs(mock_settings, mock_store, sample_df):
    """Scheduler background thread executes at least one cycle."""
    stop_event = threading.Event()
    # Very short interval for testing
    mock_settings.evolution_interval_hours = 0.001  # ~3.6 seconds
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, interval_hours=0.001)

    # Pre-populate to avoid seeding delay
    scheduler.engine.seed_population(size=4)

    # Patch MarketDataFeed so background thread doesn't hit real MT5
    with patch("atlas_ai.strategies.evolution_scheduler.MarketDataFeed") as MockFeed:
        instance = MockFeed.return_value
        instance.get_rates.return_value = sample_df
        scheduler.start()
        # Wait for at least one cycle to complete
        time.sleep(0.5)
        stop_event.set()
        scheduler.shutdown()

    assert scheduler.state.total_runs >= 1
    assert scheduler.state.scheduler_status in {"idle", "running"}


def test_scheduler_shutdown_is_clean(mock_settings, mock_store):
    """Scheduler shuts down without hanging."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    # Override interval to avoid 300s minimum clamp
    scheduler.interval_seconds = 0.01

    # Patch MarketDataFeed so background thread doesn't hit real MT5
    with patch("atlas_ai.strategies.evolution_scheduler.MarketDataFeed") as MockFeed:
        instance = MockFeed.return_value
        instance.get_rates.return_value = pd.DataFrame({
            "open": [2000.0], "high": [2001.0], "low": [1999.0],
            "close": [2000.5], "tick_volume": [100],
            "time": pd.date_range("2024-01-01", periods=1, freq="15min"),
        })
        scheduler.start()
        # Give thread a moment to start
        time.sleep(0.1)
        # Shutdown should complete quickly
        scheduler.shutdown()

    assert not scheduler._thread.is_alive() if scheduler._thread else True
    assert scheduler.state.scheduler_status == "idle"


# ------------------------------------------------------------------
# CLI mode: evolution --once
# ------------------------------------------------------------------


def test_cli_evolution_once_runs_successfully(mock_settings, mock_store, sample_df):
    """Simulate CLI `python -m atlas_ai.main evolution --once` behavior."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event, dry_run=True)

    # Pre-seed population to avoid empty-population seeding
    scheduler.engine.seed_population(size=4)

    summary = scheduler.run_once(df=sample_df)
    assert summary["population_size"] > 0
    assert "generation" in summary
    assert "best_fitness" in summary
    assert "promotions" in summary
    assert "rejections" in summary


# ------------------------------------------------------------------
# State exposure for UI/runtime
# ------------------------------------------------------------------


def test_scheduler_state_exposes_required_fields(mock_settings, mock_store):
    """Scheduler state contains all fields expected by UI/runtime."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    state = scheduler.state

    required_fields = [
        "scheduler_status",
        "current_generation",
        "last_evolution_run_at",
        "next_evolution_run_at",
        "best_fitness",
        "best_uid",
        "active_demo_count",
        "active_shadow_count",
        "promotions_this_cycle",
        "rejections_this_cycle",
        "last_error",
        "total_runs",
    ]
    for field in required_fields:
        assert hasattr(state, field), f"Missing field: {field}"


def test_scheduler_state_json_file_format(mock_settings, mock_store):
    """Persisted state file has the expected structure."""
    stop_event = threading.Event()
    scheduler = EvolutionScheduler(mock_settings, mock_store, stop_event)
    scheduler._save_state()

    state_file = Path(mock_store.base) / "evolution_scheduler_state.json"
    payload = json.loads(state_file.read_text(encoding="utf-8"))

    assert "scheduler_status" in payload
    assert "current_generation" in payload
    assert "last_evolution_run_at" in payload
    assert "next_evolution_run_at" in payload
    assert "best_fitness" in payload
    assert "best_uid" in payload
    assert "active_demo_count" in payload
    assert "active_shadow_count" in payload
    assert "last_error" in payload
    assert "total_runs" in payload

