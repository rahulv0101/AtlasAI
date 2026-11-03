from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {"execute_trades": False, "open_positions": 1, "read_only_ui": False}

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "Near close risk controls active.",
            "chosen": {"confidence": 0.4},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "session close discipline active"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _set_near_close_session(svc: AskAtlasService, *, early_close: bool = False) -> None:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=45)).strftime("%H:%M")
    end = (now + timedelta(minutes=180)).strftime("%H:%M")
    overrides = None
    if early_close:
        overrides = {now.date().isoformat(): (now + timedelta(minutes=3)).strftime("%H:%M")}
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="london",
            start=start,
            end=end,
            timezone_name="UTC",
            no_trade_buffer_minutes=10,
            early_close_overrides=overrides,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def test_forced_flat_required_creates_high_priority_control_item() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc)
    out = svc.answer("Do we need to flatten before close?")
    assert out["intent"] == "forced_flat_status"
    pending = [x for x in svc.control.pending() if x.type == "forced_flat_review"]
    assert pending
    assert pending[0].priority <= 2
    assert "forced-flat" in pending[0].summary.lower()


def test_repeated_trigger_does_not_spam_duplicates() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc)
    svc.answer("Do we need to flatten before close?")
    svc.answer("Do we need to flatten before close?")
    pending = [x for x in svc.control.pending() if x.type == "forced_flat_review"]
    assert len(pending) == 1


def test_early_close_produces_correct_reason_and_urgency() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc, early_close=True)
    svc.answer("Do we need to flatten before close?")
    pending = [x for x in svc.control.pending() if x.type == "forced_flat_review"]
    assert pending
    item = pending[0]
    assert any("forced_flat_reason=early_close_no_new_trades" in s for s in item.supporting_evidence)
    assert any("urgency=immediate_review" in s or "urgency=urgent" in s for s in item.trigger_conditions)


def test_time_to_close_included_in_item_details() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc)
    svc.answer("Do we need to flatten before close?")
    item = [x for x in svc.control.pending() if x.type == "forced_flat_review"][0]
    assert "time_to_close" in item.detailed_reason.lower()
    assert any("time_to_close_minutes=" in s for s in item.supporting_evidence)


def test_ask_atlas_forced_flat_questions() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc)
    q1 = svc.answer("Why is forced flat required?")
    q2 = svc.answer("How much time is left before flatten is required?")
    q3 = svc.answer("Is there a forced-flat review item pending?")
    assert q1["intent"] == "forced_flat_status"
    assert q2["intent"] == "flatten_urgency"
    assert q3["intent"] == "session_close_review"
    assert "pending forced-flat review item" in q3["result"].lower()


def test_no_automatic_position_close_occurs() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _set_near_close_session(svc)
    svc.answer("Do we need to flatten before close?")
    assert src.execute_called is False


def test_explain_item_includes_flatten_reason_and_urgency() -> None:
    svc = AskAtlasService(_Source())
    _set_near_close_session(svc)
    svc.answer("Do we need to flatten before close?")
    item = [x for x in svc.control.pending() if x.type == "forced_flat_review"][0]
    explain = svc.answer(f"explain item {item.id}")
    why = svc.answer(f"why did you suggest item {item.id}")
    assert "forced flat" in explain["result"].lower() or "flattening" in explain["result"].lower()
    assert "urgency" in why["result"].lower()
