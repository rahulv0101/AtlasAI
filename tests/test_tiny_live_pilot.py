from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {"execute_trades": True, "read_only_ui": False}

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.61},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold under weak evidence"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _set_active_session_for_now(svc: AskAtlasService, *, minutes_to_close: int = 120, buffer: int = 5) -> None:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=15)).strftime("%H:%M")
    end = (now + timedelta(minutes=minutes_to_close)).strftime("%H:%M")
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="test", start=start, end=end, timezone_name="UTC", no_trade_buffer_minutes=buffer, weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]


def _setup_tiny_live_pilot(svc: AskAtlasService) -> None:
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    _set_active_session_for_now(svc)


def _live_request(svc: AskAtlasService, *, instrument: str = "XAUUSD", setup: str = "continuation", size: str = "0.01") -> str:
    created = svc.answer(f"create execution request live place_trade real instrument={instrument} setup={setup} size={size}")
    req_id = created["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    return req_id


def test_live_pilot_requires_approval_before_activation() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    svc.answer("activate tiny live pilot")
    snap = svc.execution_snapshot()
    assert snap["tiny_live_pilot"]["enabled"] is False


def test_only_allowed_instrument_can_be_traded() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc, instrument="EURUSD")
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_instrument_restricted" in out["result"]


def test_only_allowed_setup_family_can_be_used() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc, setup="breakout")
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_setup_restricted" in out["result"]


def test_size_cap_enforced() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc, size="0.02")
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_size_cap_exceeded" in out["result"]


def test_daily_trade_cap_enforced() -> None:
    source = _Source()
    source.runtime["day_trade_count"] = 2
    svc = AskAtlasService(source)
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc)
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_max_trades_reached" in out["result"]
    assert svc.execution_snapshot()["tiny_live_pilot"]["paused"] is True


def test_daily_loss_cap_pauses_live_pilot() -> None:
    source = _Source()
    source.runtime["daily_loss_pct"] = 0.7
    svc = AskAtlasService(source)
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc)
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_daily_loss_cap_hit" in out["result"]
    assert svc.execution_snapshot()["tiny_live_pilot"]["paused"] is True


def test_consecutive_loss_cap_pauses_live_pilot() -> None:
    source = _Source()
    source.runtime["consecutive_losses"] = 1
    svc = AskAtlasService(source)
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req_id = _live_request(svc)
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_consecutive_loss_cap_hit" in out["result"]
    assert svc.execution_snapshot()["tiny_live_pilot"]["paused"] is True


def test_forced_flat_urgency_pauses_live_pilot() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    _set_active_session_for_now(svc, minutes_to_close=60, buffer=5)
    svc.execution._session_control.to_dict = lambda runtime_flags=None: {  # type: ignore[method-assign]
        "forced_flat_required": True,
        "minutes_to_session_close": 5,
    }
    req_id = _live_request(svc)
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_forced_flat_urgency" in out["result"]
    assert svc.execution_snapshot()["tiny_live_pilot"]["paused"] is True


def test_blocked_live_pilot_does_not_execute() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup_tiny_live_pilot(svc)
    # keep unapproved/unactivated so it blocks
    req_id = _live_request(svc)
    out = svc.answer(f"run execution {req_id}")
    assert "tiny_live_pilot_disabled" in out["result"] or "tiny_live_pilot_not_approved" in out["result"]
    assert source.execute_called is False


def test_ask_atlas_reports_live_pilot_status_and_restrictions() -> None:
    svc = AskAtlasService(_Source())
    _setup_tiny_live_pilot(svc)
    status = svc.answer("Are we in tiny live pilot mode?")
    restrictions = svc.answer("What are the live pilot restrictions?")
    blocked = svc.answer("What is blocking live pilot?")
    assert status["intent"] == "live_pilot_status"
    assert restrictions["intent"] == "live_pilot_restrictions"
    assert blocked["intent"] == "live_pilot_eligibility"
    assert "execution_state" in status
