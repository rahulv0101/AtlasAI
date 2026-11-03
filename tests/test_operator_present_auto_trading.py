from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {
            "execute_trades": True,
            "read_only_ui": False,
            "open_positions": 0,
            "day_trade_count": 0,
            "daily_loss_pct": 0.0,
            "consecutive_losses": 0,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False},
        }

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


def _set_active_session_for_now(svc: AskAtlasService, *, minutes_to_close: int = 90, buffer: int = 0) -> None:
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=20)).strftime("%H:%M"),
            end=(now + timedelta(minutes=minutes_to_close)).strftime("%H:%M"),
            timezone_name="UTC",
            no_trade_buffer_minutes=buffer,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def _setup_auto_demo(svc: AskAtlasService) -> None:
    svc.answer("set execution environment demo")
    svc.answer("set autonomy mode guarded_execution")
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    svc.answer("create automation policy p1 live continuation XAUUSD test 0.01 0.6 2 limited_live_pilot")
    svc.answer("approve automation policy p1")
    svc.answer("activate automation policy p1")
    svc.answer("activate operator present mode")
    _set_active_session_for_now(svc)


def test_auto_trading_allowed_only_in_demo() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    out = svc.answer("run auto execution paper place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "auto_trading_demo_only" in out["result"]


def test_auto_trading_blocked_when_operator_not_present() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    svc.answer("deactivate operator present mode")
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "operator_not_present" in out["result"]


def test_auto_trading_blocked_when_sleep_mode_active() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    svc.answer("activate sleep mode")
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "sleep_mode_active_automation_blocked" in out["result"]


def test_auto_trading_blocked_by_forced_flat_urgency() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    _set_active_session_for_now(svc, minutes_to_close=10, buffer=0)
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "forced_flat_urgency" in out["result"]


def test_auto_trading_blocked_by_reconciliation_required() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    svc.execution._order_tracker.upsert_submission(  # type: ignore[attr-defined]
        request_id="r1",
        broker_order_id="BRK_X",
        environment="demo",
        instrument="XAUUSD",
        side="place_trade",
        size=0.01,
        status="reconciliation_required",
        rejection_reason="pending_recon",
        reconciliation_status="required",
    )
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "reconciliation_required" in out["result"]


def test_auto_trading_blocked_by_orphan_orders() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    svc.execution._orphan_orders["BRK_ORPHAN"] = {  # type: ignore[attr-defined]
        "broker_order_id": "BRK_ORPHAN",
        "instrument": "XAUUSD",
        "side": "buy",
        "size": 0.01,
        "broker_status": "acknowledged",
        "detected_at": datetime.now(timezone.utc).timestamp(),
        "reason": "unknown_to_atlas",
        "reviewed": False,
    }
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "orphan_orders_pending_review" in out["result"]


def test_auto_trading_blocked_by_session_day_risk_controls() -> None:
    src = _Source()
    src.runtime["daily_loss_pct"] = 2.0
    svc = AskAtlasService(src)
    _setup_auto_demo(svc)
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "daily_loss_limit_reached" in out["result"]


def test_auto_trading_respects_tiny_pilot_restrictions() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    out = svc.answer("run auto execution demo place_trade EURUSD continuation 0.01 test 0.8 0.8 real")
    assert "tiny_live_pilot_instrument_restricted" in out["result"]


def test_real_money_live_auto_trading_remains_disabled() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    out = svc.answer("run auto execution live_real place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "real_live_auto_trading_disabled" in out["result"]


def test_ask_auto_trading_queries_return_structured_answers() -> None:
    svc = AskAtlasService(_Source())
    _setup_auto_demo(svc)
    q1 = svc.answer("Can you auto-trade right now?")
    q2 = svc.answer("Why is auto-trading blocked?")
    q3 = svc.answer("Are you in operator-present mode?")
    assert q1["response_kind"] == "control"
    assert q2["response_kind"] == "control"
    assert q3["intent"] == "operator_presence_status"


def test_status_queries_have_no_execution_side_effects() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_auto_demo(svc)
    svc.answer("Can you auto-trade right now?")
    svc.answer("Why is auto-trading blocked?")
    svc.answer("Are we in operator present mode?")
    assert src.execute_called is False
