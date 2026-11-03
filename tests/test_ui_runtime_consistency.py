from __future__ import annotations

from pathlib import Path

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


def test_operational_queries_do_not_fall_back_to_unknown() -> None:
    svc = AskAtlasService(_Source())
    queries = [
        "What is blocking trading right now?",
        "Why is trading blocked?",
        "Are we in tiny live pilot mode?",
        "Is live pilot paused?",
        "What is blocking demo trading right now?",
        "Is the kill switch active?",
        "Can you auto-trade right now?",
        "Are we overriding session control?",
        "What environment are execution requests using?",
        "Show me live pilot review",
        "Give me the runbook checklist",
    ]
    for q in queries:
        out = svc.answer(q)
        assert out["intent"] != "unknown_operator_query"
        assert "execution_state" in out


def test_live_demo_label_comes_from_backend_environment() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    out = svc.answer("What environment are execution requests using?")
    assert out["intent"] == "execution_environment_status"
    assert "Live Demo" in out["result"]


def test_session_override_is_explicitly_testing_only() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    svc.answer("activate session override")
    out = svc.answer("Are we overriding session control?")
    assert out["intent"] == "session_override_status"
    assert "TESTING ONLY" in out["result"]
    assert "Real session active:" in out["result"]


def test_status_queries_do_not_trigger_execution_side_effects() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("Can you auto-trade right now?")
    svc.answer("Show me the live pilot review")
    svc.answer("Give me the runbook checklist")
    assert source.execute_called is False


def test_ui_mode_fallback_is_unknown_not_paper() -> None:
    app_js = (Path(__file__).resolve().parents[1] / "atlas_ai" / "ui" / "app.js").read_text(encoding="utf-8")
    assert "Unknown (safe fallback)" in app_js
    assert "execution_environment_label || dbg.display_mode" in app_js
