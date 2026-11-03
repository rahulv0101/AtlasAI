from __future__ import annotations

from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {"execute_trades": False, "read_only_ui": False, "open_positions": 0}

    def load_status(self) -> dict:
        return {"action": "hold", "message": "hold", "chosen": {"confidence": 0.6}, "comparison": {"active": False}}

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _force_closed_session(gate: ExecutionGate) -> None:
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="closed", start="00:00", end="00:01", timezone_name="UTC", no_trade_buffer_minutes=1)
    ]


def _disable_pilot_constraints(gate: ExecutionGate) -> None:
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    gate._tiny_live_pilot.configured = False  # type: ignore[attr-defined]
    gate._tiny_live_pilot.enabled = False  # type: ignore[attr-defined]
    gate._tiny_live_pilot.approved = False  # type: ignore[attr-defined]
    gate._tiny_live_pilot.paused = False  # type: ignore[attr-defined]


def test_override_enables_trading_outside_session() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate.set_execution_environment("demo")
    _disable_pilot_constraints(gate)
    _force_closed_session(gate)
    gate.set_session_override(True, reason="test")
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is True
    assert out.get("dry_run") is True


def test_override_off_restores_normal_session_blocking() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate.set_execution_environment("demo")
    _disable_pilot_constraints(gate)
    _force_closed_session(gate)
    gate.set_session_override(False, reason="test")
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "outside_active_session"


def test_override_does_not_bypass_other_safety_gates() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate.set_execution_environment("demo")
    _disable_pilot_constraints(gate)
    gate.activate_kill_switch("test")
    _force_closed_session(gate)
    gate.set_session_override(True, reason="test")
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "kill_switch_active"


def test_override_not_persisted_across_restart() -> None:
    gate = ExecutionGate()
    gate.set_execution_environment("demo")
    gate.set_session_override(True, reason="test")
    # New instance must default to false even if state file exists.
    gate2 = ExecutionGate()
    assert gate2.snapshot()["session_override_active"] is False


def test_override_blocked_outside_demo_environment() -> None:
    gate = ExecutionGate()
    gate.set_execution_environment("paper")
    snap = gate.set_session_override(True, reason="test")
    assert snap["session_override_active"] is False
    assert snap["session_override_last_block_reason"] == "session_override_demo_only"


def test_ask_atlas_reports_override_state() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    svc.answer("activate session override")
    out = svc.answer("Are we overriding session control?")
    assert "active" in out["result"].lower()
    assert out["execution_state"]["session_override_active"] is True


def test_session_override_exact_commands_are_recognized() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")

    activate = svc.answer("activate session override")
    assert activate["intent"] == "session_override_command"
    assert "Session override activated (testing only)" in activate["result"]
    assert activate["execution_state"]["session_override_active"] is True
    assert activate["intent"] != "unknown_operator_query"

    deactivate = svc.answer("deactivate session override")
    assert deactivate["intent"] == "session_override_command"
    assert "deactivated" in deactivate["result"].lower()
    assert deactivate["execution_state"]["session_override_active"] is False
    assert deactivate["intent"] != "unknown_operator_query"


def test_session_override_exact_queries_return_state() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    svc.answer("activate session override")

    q1 = svc.answer("is session override active?")
    assert q1["intent"] == "session_override_status"
    assert q1["execution_state"]["session_override_active"] is True
    assert "Session override is active" in q1["result"]
    assert q1["intent"] != "unknown_operator_query"

    q2 = svc.answer("are we overriding session control?")
    assert q2["intent"] == "session_override_status"
    assert q2["execution_state"]["session_override_active"] is True
    assert "TESTING ONLY" in q2["result"]
    assert q2["intent"] != "unknown_operator_query"
