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


def test_environment_switches_to_demo_explicitly() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("set execution environment demo")
    snap = out["execution_state"]
    assert snap["execution_environment"] == "demo"
    assert snap["execution_environment_label"] == "Live Demo"


def test_demo_mode_visible_in_query_and_runtime_state() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    out = svc.answer("Are we in live demo mode now?")
    snap = out["execution_state"]
    assert out["intent"] == "execution_environment_status"
    assert "live demo" in out["result"].lower()
    assert snap["execution_environment"] == "demo"
    assert snap["real_live_enabled"] is False


def test_live_requests_route_to_demo_environment() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    created = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")
    req = created["execution_request"]
    assert req["environment"] == "demo"
    assert req["requested_environment"] == "live"


def test_real_money_live_path_remains_disabled() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    req_id = svc.answer("create execution request live_real place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    out = svc.answer(f"run execution {req_id}")
    assert "real_live_disabled" in out["result"]


def test_tiny_live_pilot_restrictions_still_apply_in_demo() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    req_id = svc.answer("create execution request demo place_trade real instrument=EURUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_instrument_restricted" in out["result"]


def test_safety_gates_still_block_in_demo() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    req_id = svc.answer("create execution request demo place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    missing = svc.answer(f"run execution {req_id}")
    assert "approval_missing" in missing["result"]
    svc.answer(f"approve execution {req_id} live-confirm")
    svc.answer("activate kill switch")
    blocked = svc.answer(f"run execution {req_id}")
    assert "kill_switch_active" in blocked["result"]


def test_demo_mode_queries_are_safe_and_structured() -> None:
    svc = AskAtlasService(_Source())
    _setup_demo_pilot(svc)
    out = svc.answer("Is this real-money live or demo?")
    assert out["response_kind"] == "control"
    assert out["intent"] == "execution_environment_status"
    assert "disabled / not in use" in out["result"]


def test_environment_status_queries_have_no_execution_side_effects() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("set execution environment demo")
    svc.answer("Are we in live demo mode now?")
    svc.answer("What environment are execution requests using?")
    assert source.execute_called is False
