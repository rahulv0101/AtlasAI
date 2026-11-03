from __future__ import annotations

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace

from atlas_ai.risk.risk_manager import RiskSnapshot
from atlas_ai.system_health import build_system_health
from atlas_ai.trader import AtlasTrader


def _trader() -> AtlasTrader:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        max_consecutive_losses=3,
        max_daily_drawdown=0.03,
        timeframe_minutes=1,
        execute_trades=True,
        gemini_influence_enabled=False,
        meta_model_enabled=True,
        research_regime_filter_enabled=False,
    )
    return trader


def _snapshot(*, daily_loss_pct: float = 0.0) -> RiskSnapshot:
    return RiskSnapshot(
        equity=1000.0,
        balance=1000.0,
        daily_loss_pct=daily_loss_pct,
        weekly_loss_pct=0.0,
        open_positions=0,
        spread_points=20.0,
        consecutive_losses=0,
    )


def test_consecutive_loss_circuit_breaker_activation() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 4},
        loss_memory={"recent_outcomes": []},
    )
    assert out["safety_lock_active"] is True
    assert "consecutive_loss_threshold" in out["safety_lock_reason"]


def test_drawdown_based_circuit_breaker_activation() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(daily_loss_pct=0.03),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 0},
        loss_memory={"recent_outcomes": []},
    )
    assert out["safety_lock_active"] is True
    assert "rolling_drawdown_limit" in out["safety_lock_reason"]


def test_expectancy_degradation_triggers_safety_response() -> None:
    trader = _trader()
    now = datetime.now(timezone.utc)
    outcomes = [
        {
            "outcome": "loss",
            "net_pnl_zar": -120.0,
            "closed_at": now.isoformat(),
        }
        for _ in range(10)
    ]
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 0},
        loss_memory={"recent_outcomes": outcomes},
    )
    assert out["safety_lock_active"] is True
    assert "rolling_expectancy_degradation" in out["safety_lock_reason"]


def test_single_loss_causes_short_cooldown_not_full_lock() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 1, "cooldown_until": ""},
        loss_memory={"recent_outcomes": [{"outcome": "loss", "net_pnl_zar": -20.0}]},
    )
    assert out["safety_lock_active"] is False
    assert out["cooldown_active"] is True
    assert out["cooldown_level"] == "short"


def test_stale_time_dated_loss_memory_does_not_lock_on_current_session_pressure() -> None:
    trader = _trader()
    stale_time = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    outcomes = [
        {"outcome": "loss", "net_pnl_zar": -20.0, "time": stale_time}
        for _ in range(8)
    ]

    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(daily_loss_pct=0.001),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 0, "trades_today": 0},
        loss_memory={"recent_outcomes": outcomes},
    )

    assert out["safety_lock_active"] is False
    assert out["loss_streak"] == 0
    assert out["supporting_win_rate"] == 0.0


def test_current_time_dated_loss_memory_still_locks_when_recent() -> None:
    trader = _trader()
    current_time = datetime.now(timezone.utc).isoformat()
    outcomes = [
        {"outcome": "loss", "net_pnl_zar": -20.0, "time": current_time}
        for _ in range(8)
    ]

    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(daily_loss_pct=0.001),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 0, "trades_today": 0},
        loss_memory={"recent_outcomes": outcomes},
    )

    assert out["safety_lock_active"] is True
    assert "consecutive_loss_threshold" in out["safety_lock_reason"]


def test_single_loss_cooldown_does_not_rearm_after_same_marker_expires() -> None:
    trader = _trader()
    expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    guard_state = {
        "consecutive_losses": 1,
        "cooldown_until": expired,
        "cooldown_level": "short",
        "cooldown_reason": "single_loss_cooldown",
        "cooldown_marker": "losses=1|level=short|reason=single_loss_cooldown",
    }

    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state=guard_state,
        loss_memory={"recent_outcomes": [{"outcome": "loss", "net_pnl_zar": -20.0}]},
    )

    assert out["cooldown_active"] is False
    assert guard_state["cooldown_until"] == ""
    assert guard_state["cooldown_level"] == "none"
    assert guard_state["cooldown_consumed_marker"] == "losses=1|level=short|reason=single_loss_cooldown"

    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state=guard_state,
        loss_memory={"recent_outcomes": [{"outcome": "loss", "net_pnl_zar": -20.0}]},
    )

    assert out["cooldown_active"] is False
    assert guard_state["cooldown_until"] == ""


def test_existing_unmarked_cooldown_is_not_extended_during_migration() -> None:
    trader = _trader()
    existing = (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()
    guard_state = {
        "consecutive_losses": 1,
        "cooldown_until": existing,
        "cooldown_level": "short",
        "cooldown_reason": "single_loss_cooldown",
    }

    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state=guard_state,
        loss_memory={"recent_outcomes": [{"outcome": "loss", "net_pnl_zar": -20.0}]},
    )

    assert out["cooldown_active"] is True
    assert guard_state["cooldown_until"] == existing
    assert guard_state["cooldown_marker"] == "losses=1|level=short|reason=single_loss_cooldown"


def test_guard_state_load_preserves_cooldown_marker() -> None:
    trader = _trader()
    trader.store = SimpleNamespace(
        load_runtime_flags=lambda: {
            "risk_guard_state": {
                "date": datetime.now(timezone.utc).date().isoformat(),
                "consecutive_losses": 1,
                "cooldown_until": "2026-05-12T10:12:58+02:00",
                "cooldown_level": "short",
                "cooldown_reason": "single_loss_cooldown",
                "cooldown_marker": "losses=1|level=short|reason=single_loss_cooldown",
                "cooldown_consumed_marker": "older",
            }
        }
    )

    state = trader._load_guard_state()

    assert state["cooldown_marker"] == "losses=1|level=short|reason=single_loss_cooldown"
    assert state["cooldown_consumed_marker"] == "older"


def test_repeated_losses_escalate_cooldown() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"consecutive_losses": 2, "cooldown_until": ""},
        loss_memory={
            "recent_outcomes": [
                {"outcome": "loss", "net_pnl_zar": -20.0},
                {"outcome": "loss", "net_pnl_zar": -25.0},
            ]
        },
    )
    assert out["cooldown_active"] is True
    assert out["cooldown_level"] in {"medium", "high"}


def test_extreme_volatility_blocks_entries() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "high_volatility_range", "volatility_pct": 0.08},
        context_snapshot={"vix_regime": "extreme"},
        guard_state={"consecutive_losses": 0},
        loss_memory={"recent_outcomes": []},
    )
    assert out["volatility_state"] == "extreme"
    assert out["volatility_action"] == "block_new_entries"


def test_elevated_volatility_reduces_aggressiveness_without_forcing_stop() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.05},
        context_snapshot={"vix_regime": "elevated"},
        guard_state={"consecutive_losses": 0},
        loss_memory={"recent_outcomes": []},
    )
    assert out["volatility_state"] == "elevated"
    assert out["volatility_action"] == "reduce_aggressiveness"
    assert out["safety_lock_active"] is False


def test_safety_lock_state_visible_in_system_health_surface() -> None:
    status = {
        "execute_trades": True,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "safety_layer": {
            "safety_lock_active": True,
            "safety_lock_reason": "consecutive_loss_threshold",
            "cooldown_active": True,
            "cooldown_level": "high",
            "volatility_state": "extreme",
            "volatility_action": "block_new_entries",
            "recovery_ready": False,
            "recovery_mode": "lockdown",
            "recovery_blockers": ["safety_lock_active"],
        },
    }
    health = build_system_health(status, settings=SimpleNamespace(timeframe_minutes=1, execute_trades=True, gemini_influence_enabled=False, meta_model_enabled=True, research_regime_filter_enabled=False), runtime_flags={})
    assert health["safety_layer"]["safety_lock_active"] is True
    assert "safety_lock_active" in health["protective_locks"]
    assert "safety_lock_active" in health["classification"]["PROTECTIVE_LOCK"]


def test_recovery_requires_conditions_before_resume() -> None:
    trader = _trader()
    out = trader._evaluate_safety_layer(
        snapshot=_snapshot(),
        regime={"name": "trend", "volatility_pct": 0.03},
        context_snapshot={"vix_regime": "normal"},
        guard_state={"safety_lock_active": True, "safety_lock_reason": "manual"},
        loss_memory={"recent_outcomes": []},
    )
    assert out["recovery_ready"] is False
    assert "safety_lock_active" in out["recovery_blockers"]


def test_daily_target_does_not_override_safety_logic() -> None:
    constants = AtlasTrader.step.__code__.co_consts
    assert all("Trading paused for the day." not in str(x) for x in constants)
