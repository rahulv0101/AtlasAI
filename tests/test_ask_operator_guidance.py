from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": False, "read_only_ui": False}
        self.events: list[dict] = []

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.42},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return list(self.events)[-limit:]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}


def test_operator_guidance_query_returns_structured_meta_intent_response() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("improve yourself")
    assert out["intent"] == "operator_guidance"
    assert out["meta_intent"]["classified_as"] == "operator_guidance"
    assert out["meta_intent"]["action_taken"] == "none_read_only"
    assert isinstance(out.get("guidance_options"), list)
    assert len(out["guidance_options"]) >= 4
    assert "Closest likely interpretation answered" not in out["result"]


def test_operator_guidance_query_has_no_execution_request_side_effects() -> None:
    svc = AskAtlasService(_Source())
    before = svc.execution.pending_requests(limit=50)
    _ = svc.answer("how can you get better")
    _ = svc.answer("what should I do")
    after = svc.execution.pending_requests(limit=50)
    assert before == after
    assert svc.execution.last_request() is None
