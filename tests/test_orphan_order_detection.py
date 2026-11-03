from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": True, "read_only_ui": False}
        self.execute_called = False

    def load_status(self) -> dict:
        return {"action": "hold", "message": "hold", "chosen": {"confidence": 0.6}, "min_confidence_threshold": 0.58, "comparison": {"active": False}}

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _setup(svc: AskAtlasService) -> None:
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="test", start=(now - timedelta(minutes=20)).strftime("%H:%M"), end=(now + timedelta(minutes=90)).strftime("%H:%M"), timezone_name="UTC", no_trade_buffer_minutes=0, weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]


def _inject_orphan(svc: AskAtlasService) -> None:
    svc.execution._broker_adapter.submit_order(  # type: ignore[attr-defined]
        idempotency_key="external_orphan_1",
        request={
            "instrument": "XAUUSD",
            "action_type": "place_trade",
            "side": "buy",
            "size": 0.01,
        },
    )


def test_orphan_detection_creates_control_item() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _inject_orphan(svc)
    svc.answer("scan broker open orders")
    pending = svc.answer("show pending items")
    assert "order_orphan_review" in pending["result"].lower()


def test_orphan_dedupe_does_not_spam() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _inject_orphan(svc)
    svc.answer("scan broker open orders")
    svc.answer("scan broker open orders")
    pending = svc.answer("show pending items")
    lines = [x for x in str(pending["result"]).split("\n") if "order_orphan_review" in x.lower()]
    assert len(lines) <= 1


def test_orphan_queries_return_expected_output() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _inject_orphan(svc)
    svc.answer("scan broker open orders")
    q1 = svc.answer("Are there any orphan orders?")
    q2 = svc.answer("What orders exist at the broker but not in Atlas?")
    assert "orphan orders detected" in q1["result"].lower()
    assert "unknown to atlas" in q2["result"].lower() or "broker orders unknown to atlas" in q2["result"].lower()


def test_no_execution_side_effects_from_orphan_flow() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup(svc)
    _inject_orphan(svc)
    svc.answer("scan broker open orders")
    svc.answer("reconcile orphan BRK_00000001")
    assert source.execute_called is False
