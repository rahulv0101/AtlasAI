from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _StubSource:
    def __init__(self) -> None:
        self._status = {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.22, "side": "flat", "strategy": "momentum_continuation"},
            "min_confidence_threshold": 0.58,
            "context_snapshot": {"regime": "trend", "session": "london", "liquidity_state": "normal", "spread_state": "tight"},
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
        }
        self._runtime = {"execute_trades": False}
        self._events: list[dict] = []

    def load_status(self) -> dict:
        return dict(self._status)

    def load_events(self, limit: int = 160) -> list[dict]:
        return list(self._events)[-limit:]

    def load_runtime_flags(self) -> dict:
        return dict(self._runtime)

    def load_comparison(self) -> dict:
        return dict(self._status.get("comparison", {}) or {})


def test_how_much_do_you_know_routes_to_self_knowledge() -> None:
    reply = AskAtlasService(_StubSource()).answer("How much do you know about markets?")
    assert reply["intent"] == "self_knowledge"
    assert "Knowledge level:" in reply["result"]
    assert "I understand:" in reply["result"]
    assert "Limitations:" in reply["result"]
    assert "Current role:" in reply["result"]
    assert "STATUS:" not in reply["result"]


def test_capabilities_question_routes_to_capabilities_intent() -> None:
    reply = AskAtlasService(_StubSource()).answer("What are your capabilities?")
    assert reply["intent"] == "capabilities"
    assert "Capabilities:" in reply["result"]
    assert "Current role:" in reply["result"]
    assert "Best grounded answer:" not in reply["result"]


def test_readiness_question_stays_direct_and_grounded() -> None:
    reply = AskAtlasService(_StubSource()).answer("Are you ready for live trading?")
    assert reply["intent"] == "readiness"
    assert reply["result"].startswith("Readiness:")
    assert "Why:" in reply["result"]
    assert "Confidence gate:" in reply["result"]


def test_what_can_you_do_routes_to_capabilities_without_unknown_fallback() -> None:
    reply = AskAtlasService(_StubSource()).answer("What can you do?")
    assert reply["intent"] == "capabilities"
    assert "unknown_operator_query" not in reply["intent"]
    assert "Best grounded answer:" not in reply["result"]


def test_autonomous_trader_question_reports_bounded_mandate() -> None:
    svc = AskAtlasService(_StubSource())
    svc.answer("set autonomy mode guarded_autonomous_demo")
    reply = svc.answer("Are you an autonomous trader?")

    assert reply["intent"] == "automation_status"
    text = reply["result"].lower()
    assert "bounded_autonomous_trader" in text
    assert "real live trading" in text
    assert "no real feelings" in text
    assert reply["execution_state"]["autonomy_mode"] == "guarded_autonomous_demo"
