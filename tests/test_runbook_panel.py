from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": True, "read_only_ui": False, "open_positions": 0}
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


def test_pre_session_checklist_reflects_blocked_conditions() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    svc.answer("activate kill switch")
    out = svc.answer("Are we ready to trade?")
    assert "BLOCKED" in out["result"]
    assert "kill_switch=True" in out["result"] or "kill switch" in out["result"].lower()


def test_reconciliation_orphan_shows_blocked_or_review() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    # Inject orphan and scan.
    svc.execution._broker_adapter.submit_order(idempotency_key="orphan_r", request={"instrument": "XAUUSD", "size": 0.01, "action_type": "place_trade"})  # type: ignore[attr-defined]
    svc.answer("scan broker open orders")
    out = svc.answer("Give me the runbook checklist")
    text = out["result"]
    assert "orphan orders clear: REVIEW" in text


def test_stop_conditions_show_when_limits_hit() -> None:
    source = _Source()
    source.runtime["consecutive_losses"] = 2
    svc = AskAtlasService(source)
    _setup(svc)
    out = svc.answer("Should we stop trading now?")
    assert "stop conditions" in out["result"].lower()
    assert "consecutive loss limit hit" in out["result"].lower() or "pilot paused" in out["result"].lower()


def test_runbook_queries_return_structured_checklist() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    out = svc.answer("What needs to be checked before trading?")
    assert "Pilot runbook checklist:" in out["result"]
    assert "Pre-session:" in out["result"]
    assert "During session:" in out["result"]
    assert "Post-session:" in out["result"]


def test_runbook_queries_have_no_execution_side_effects() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup(svc)
    svc.answer("Give me the runbook checklist")
    svc.answer("What should I review after session?")
    assert source.execute_called is False


def test_post_session_checklist_reflects_open_pending_reconciliation() -> None:
    source = _Source()
    source.runtime["open_positions"] = 1
    svc = AskAtlasService(source)
    _setup(svc)
    # Force reconciliation-required state.
    req = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")
    rid = req["execution_request"]["id"]
    svc.answer(f"approve execution {rid} live-confirm")
    svc.execution._broker_adapter.set_submit_override(rid, svc.execution._broker_adapter.query_order_status(broker_order_id="MISSING"))  # type: ignore[attr-defined]
    svc.answer(f"run execution {rid}")
    out = svc.answer("What should I review after session?")
    assert "all positions closed: False" in out["result"]
    assert "reconciliation required: True" in out["result"]
