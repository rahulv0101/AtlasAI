"""Isolated diagnostic: injected candidate through trader and real ExecutionGate."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from atlas_ai.execution.broker_adapter import TestBrokerAdapter
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.risk.risk_manager import RiskManager
from atlas_ai.testing.replay_harness import _build_trader, default_replay_scenarios
from atlas_ai.trader import AtlasTrader


def test_injected_valid_candidate_reaches_real_gate_and_simulated_broker(tmp_path):
    scenario = replace(default_replay_scenarios()[2], base_volume=0.01, min_confidence=0.44)
    trader, store = _build_trader(scenario)
    gate = ExecutionGate(state_store_path=str(tmp_path / "automation_state.json"))
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    gate.set_broker_adapter(TestBrokerAdapter())
    gate.configure_tiny_live_pilot(
        allowed_instrument="XAUUSD", allowed_setup_family="continuation",
        size_cap=0.01, max_trades_per_day=8, daily_loss_cap=0.6, max_consecutive_losses=8,
    )
    gate.approve_tiny_live_pilot(note="isolated diagnostic")
    gate.activate_tiny_live_pilot()
    now = datetime.now(timezone.utc)
    gate._session_control._sessions = [SessionWindow(
        name="test", start="00:00", end="23:59", timezone_name="UTC",
        no_trade_buffer_minutes=0, weekdays=tuple(range(7)),
    )]
    trader.execution_gate = gate
    trader.settings.max_weekly_drawdown = 0.05
    trader.settings.max_open_trades = 1
    trader.risk.evaluate = RiskManager(trader.settings).evaluate
    # Controlled entry evidence for the injected candidate; real timing check below.
    trader._derive_runtime_chart_features = lambda **kwargs: {
        "location_state": "pullback", "momentum_state": "strong",
        "trend_state": "uptrend", "trend_structure": "HH/HL", "setup_type_hint": "continuation",
    }
    def timing(signal, context, status=None):
        context = {**context, "entry_location_state": "pullback", "entry_momentum_state": "strong", "entry_trend_state": "uptrend"}
        return AtlasTrader._timing_discipline_snapshot(trader, signal, context, status)
    trader._timing_discipline_snapshot = timing
    with patch("atlas_ai.trader.mt5.account_info", return_value=SimpleNamespace(login=123456)), \
         patch("atlas_ai.trader.mt5.symbol_info", return_value=SimpleNamespace(trade_tick_value=1.0)), \
         patch("atlas_ai.trader.mt5.order_send", side_effect=AssertionError("Real broker submission forbidden")), \
         patch("atlas_ai.trader.evaluate_heartbeat_state", return_value={"is_stale": False}):
        status = trader.step()
    print({"action": status.get("action"), "message": status.get("message"), "chosen": status.get("chosen"), "approval": status.get("guarded_demo_auto_approval"), "execution": status.get("execution")})
    assert status["timing_discipline"]["passed"] is True
    assert status["execution"]["success"] is True
    assert status["guarded_demo_auto_approval"]["approved"] is True
    assert any(row["event"] == "order_submit_attempt" for row in gate.recent_audit(limit=100))
