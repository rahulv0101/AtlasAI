from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {"execute_trades": True, "read_only_ui": False, "open_positions": 0}

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


def _set_active_session_for_now(gate: ExecutionGate, *, minutes_to_close: int = 90) -> None:
    now = datetime.now(timezone.utc)
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=20)).strftime("%H:%M"),
            end=(now + timedelta(minutes=minutes_to_close)).strftime("%H:%M"),
            timezone_name="UTC",
            no_trade_buffer_minutes=0,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def _setup_demo_pilot(svc: AskAtlasService) -> None:
    svc.answer("set execution environment demo")
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    _set_active_session_for_now(svc.execution)


def test_sleep_mode_blocks_new_execution_request_creation() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    svc.answer("activate sleep mode")
    out = svc.answer("create execution request demo place_trade real instrument=XAUUSD setup=continuation size=0.01")
    assert "sleep_mode_active_no_new_requests" in out["result"]
    req = out["execution_request"]
    assert req["id"] == ""
    assert req["execution_status"] == "blocked"


def test_sleep_mode_blocks_execution_attempts() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    created = svc.answer("create execution request demo place_trade real instrument=XAUUSD setup=continuation size=0.01")
    rid = created["execution_request"]["id"]
    svc.answer(f"approve execution {rid} live-confirm")
    svc.answer("activate sleep mode")
    out = svc.answer(f"run execution {rid}")
    assert "sleep_mode_active_execution_blocked" in out["result"]


def test_sleep_mode_blocks_automation() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    svc.answer("create automation policy p1 demo continuation XAUUSD test 0.01 0.6 2 limited_live_pilot")
    svc.answer("approve automation policy p1")
    svc.answer("activate automation policy p1")
    svc.answer("activate sleep mode")
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "sleep_mode_active_automation_blocked" in out["result"]


def test_monitoring_and_alerts_remain_visible_while_sleeping() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    svc.answer("activate sleep mode")
    svc.answer("create execution request demo place_trade real instrument=XAUUSD setup=continuation size=0.01")
    status = svc.answer("Are we in sleep mode now?")
    snap = status["execution_state"]
    assert "Sleep Safety Mode is active" in status["result"]
    assert snap["monitoring_active_while_sleeping"] is True
    assert snap["alert_only_while_sleeping"] is True
    assert int(snap["sleep_alert_count"]) >= 1


def test_sleep_mode_queries_have_no_execution_side_effects() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup_demo_pilot(svc)
    svc.answer("activate sleep mode")
    svc.answer("are we in sleep mode?")
    svc.answer("why is execution blocked?")
    svc.answer("monitoring-only while sleeping")
    assert source.execute_called is False
