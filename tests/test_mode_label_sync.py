from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": False, "read_only_ui": False}

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.21},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
            "system_health": {"effective_mode": "paper", "paper_mode_confirmed": True},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "No high-confidence trade found."}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}


def test_switching_to_demo_shows_live_demo_mode_label() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    out = svc.answer("What state are you in?")
    assert "Mode: Live Demo" in out["result"]
    assert out["debug"]["display_mode"] == "Live Demo"


def test_switching_to_paper_shows_paper_mode_label() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment paper")
    out = svc.answer("What state are you in?")
    assert "Mode: Paper" in out["result"]
    assert out["debug"]["display_mode"] == "Paper"


def test_voice_status_reflects_correct_mode_label() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("set execution environment demo")
    out = svc.answer("What state are you in?", mode="voice")
    assert "Mode: Live Demo." in out["result"]

