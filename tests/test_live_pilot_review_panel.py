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
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.61},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _open_session(svc: AskAtlasService, *, mins_to_close: int = 120, buffer: int = 0) -> None:
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=30)).strftime("%H:%M"),
            end=(now + timedelta(minutes=mins_to_close)).strftime("%H:%M"),
            timezone_name="UTC",
            no_trade_buffer_minutes=buffer,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def _setup_live_pilot(svc: AskAtlasService) -> None:
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    _open_session(svc)


def _submit_live(svc: AskAtlasService) -> str:
    req = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")
    rid = req["execution_request"]["id"]
    svc.answer(f"approve execution {rid} live-confirm")
    return rid


def test_runtime_payload_contains_live_review_fields() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    snap = svc.execution_snapshot()
    assert "tiny_live_pilot_attempts_today" in snap
    assert "live_approvals_used" in snap
    assert "forced_flat_pending" in snap
    assert "forced_flat_urgency" in snap
    assert "reconciliation_required" in snap


def test_paused_pilot_shows_pause_reason() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    svc.answer("pause tiny live pilot")
    out = svc.answer("Why is the pilot paused right now?")
    assert out["intent"] == "live_pilot_pause_reason"
    assert "operator_requested" in out["result"]


def test_reconciliation_required_state_is_surfaced() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    rid = _submit_live(svc)
    svc.execution._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=False, status="unknown", broker_order_id="BRK_UNK_2", message="timeout"),
    )
    svc.answer(f"run execution {rid}")
    out = svc.answer("Is anything waiting for reconciliation?")
    assert out["intent"] == "order_reconciliation"
    assert "yes" in out["result"].lower()


def test_forced_flat_pending_state_is_surfaced() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    svc.execution._session_control.to_dict = lambda runtime_flags=None: {  # type: ignore[method-assign]
        "forced_flat_required": True,
        "minutes_to_session_close": 4,
        "session_block_reason": "near_session_close_no_new_trades",
    }
    out = svc.answer("Are there any forced-flat issues pending?")
    assert out["intent"] == "forced_flat_status"
    assert "true" in out["result"].lower() or "yes" in out["result"].lower()


def test_last_broker_reject_reason_is_visible() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    rid = _submit_live(svc)
    svc.execution._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=False, status="rejected", broker_order_id="BRK_REJ_2", message="insufficient_margin"),
    )
    svc.answer(f"run execution {rid}")
    out = svc.answer("Why was the order rejected?")
    assert out["intent"] == "broker_reject_reason"
    assert "insufficient_margin" in out["result"]


def test_risk_used_remaining_appears_correctly() -> None:
    source = _Source()
    source.runtime["daily_loss_pct"] = 0.2
    svc = AskAtlasService(source)
    _setup_live_pilot(svc)
    out = svc.answer("How much live pilot risk is left today?")
    assert out["intent"] == "live_pilot_status"
    assert "remaining" in out["result"].lower()


def test_live_pilot_review_query_returns_structured_safe_answer() -> None:
    svc = AskAtlasService(_Source())
    _setup_live_pilot(svc)
    out = svc.answer("Show me the live pilot review")
    assert out["intent"] == "live_pilot_status"
    assert "Live pilot review:" in out["result"]
    assert "execution_state" in out
    assert "control_state" in out


def test_live_pilot_review_query_does_not_trigger_execution() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _setup_live_pilot(svc)
    svc.answer("Show me the live pilot review")
    svc.answer("What happened in the live pilot today?")
    assert source.execute_called is False
