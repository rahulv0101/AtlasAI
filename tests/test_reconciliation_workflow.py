from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.broker_adapter import BrokerOrderResponse
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


def _open_session(svc: AskAtlasService) -> None:
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="test", start=(now - timedelta(minutes=20)).strftime("%H:%M"), end=(now + timedelta(minutes=90)).strftime("%H:%M"), timezone_name="UTC", no_trade_buffer_minutes=0, weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]


def _setup(svc: AskAtlasService) -> None:
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    _open_session(svc)


def _create_uncertain(svc: AskAtlasService) -> str:
    req = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")
    rid = req["execution_request"]["id"]
    svc.answer(f"approve execution {rid} live-confirm")
    svc.execution._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid, BrokerOrderResponse(ok=False, status="unknown", broker_order_id="BRK_UNCERTAIN", message="timeout")
    )
    svc.answer(f"run execution {rid}")
    return rid


def test_reconciliation_required_creates_high_priority_control_item() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _create_uncertain(svc)
    pending = svc.answer("show pending items")
    assert "order reconciliation review" in pending["result"].lower()


def test_duplicate_mismatch_does_not_spam_duplicates() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _create_uncertain(svc)
    svc.answer("show me the live pilot review")
    svc.answer("show me the live pilot review")
    c = svc.answer("show pending items")
    lines = [x for x in str(c["result"]).split("\n") if "order_reconciliation_review" in x.lower() or "order reconciliation review" in x.lower()]
    assert len(lines) <= 1


def test_reconcile_action_queries_safely_and_updates_when_clear() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    rid = _create_uncertain(svc)
    svc.execution._broker_adapter.set_status_override(  # type: ignore[attr-defined]
        "BRK_UNCERTAIN", BrokerOrderResponse(ok=True, status="acknowledged", broker_order_id="BRK_UNCERTAIN", message="ok")
    )
    out = svc.answer(f"reconcile order {rid}")
    assert "status=acknowledged" in out["result"].lower() or "reconciliation for" in out["result"].lower()


def test_unresolved_query_keeps_item_pending() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    rid = _create_uncertain(svc)
    svc.execution._broker_adapter.set_status_override(  # type: ignore[attr-defined]
        "BRK_UNCERTAIN", BrokerOrderResponse(ok=False, status="unknown", broker_order_id="BRK_UNCERTAIN", message="still_unknown")
    )
    svc.answer(f"reconcile order {rid}")
    p = svc.answer("show pending items")
    assert "reconciliation" in p["result"].lower()


def test_ask_atlas_reconciliation_queries_are_structured() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _create_uncertain(svc)
    r = svc.answer("Why is reconciliation required?")
    assert r["intent"] in {"reconciliation_status", "order_reconciliation"}
    assert "execution_state" in r
    assert "control_state" in r


def test_runtime_flow_surfaces_reconciliation_review_fields() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _create_uncertain(svc)
    snap = svc.execution_snapshot()
    assert "top_reconciliation_request_id" in snap
    assert "top_reconciliation_reason" in snap


def test_reconcile_action_does_not_trigger_new_order_submission() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup(svc)
    rid = _create_uncertain(svc)
    svc.answer(f"reconcile order {rid}")
    assert source.execute_called is False


def test_explain_item_includes_mismatch_reason_and_statuses() -> None:
    svc = AskAtlasService(_Source())
    _setup(svc)
    _create_uncertain(svc)
    pending = svc.answer("show pending items")
    line = next((x for x in str(pending["result"]).split("\n") if "order_reconciliation_review" in x.lower()), "")
    item_id = int(line.split("- item ")[1].split()[0]) if line else 0
    assert item_id > 0
    expl = svc.answer(f"explain item {item_id}")
    text = str(expl["result"]).lower()
    assert "reconciliation" in text
    assert ("internal_request_id" in text) or ("internal status" in text)
