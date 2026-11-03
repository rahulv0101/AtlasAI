from __future__ import annotations

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.ui_server import create_ui_app


class _Source:
    def __init__(self) -> None:
        self.runtime = {
            "execute_trades": False,
            "read_only_ui": False,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False, "max_freshness_age_ms": 120000},
            "max_freshness_age_ms": 120000,
            "open_positions": 0,
        }

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.55},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold under weak evidence"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}


def test_run_guarded_demo_checklist_is_read_only() -> None:
    svc = AskAtlasService(_Source())
    before = svc.execution.pending_requests(limit=50)
    out = svc.answer("run guarded demo checklist")
    after = svc.execution.pending_requests(limit=50)
    assert out["intent"] == "guarded_demo_workflow"
    assert "workflow_checks" in out
    assert before == after


def test_generic_queries_remain_non_executing_during_workflow() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("should i buy or sell?")
    svc.answer("can you trade now?")
    svc.answer("what is the market bias?")
    assert svc.execution.last_request() is None


def test_blocked_trade_validation_returns_block_reason_without_order_side_effect() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("validate blocked trade path")
    val = dict(out.get("validation_result", {}) or {})
    assert val.get("scenario") == "blocked_trade_path"
    assert val.get("blocked") is True
    assert val.get("block_reason") == "approval_missing"
    assert int(out["execution_state"].get("pending_live_orders_count", 0) or 0) == 0


def test_allowed_guarded_trade_validation_uses_guarded_dry_run_path() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("validate allowed guarded trade path")
    val = dict(out.get("validation_result", {}) or {})
    req_id = str(val.get("request_id", ""))
    req = svc.execution.get_request(req_id)
    assert val.get("scenario") == "allowed_guarded_trade_path"
    assert val.get("ok") is True
    assert val.get("dry_run") is True
    assert req is not None
    assert req.get("request_origin") == "operator_validation"


def test_freshness_validation_reports_stale_block() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("validate freshness block")
    val = dict(out.get("validation_result", {}) or {})
    assert val.get("scenario") == "freshness_block"
    assert val.get("blocked") is True
    assert val.get("block_reason") == "freshness_stale"


def test_cross_surface_consistency_check_reflects_shared_execution_truth(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": False, "read_only_ui": False})
    app = create_ui_app(store)
    client = app.test_client()
    ask = client.post("/api/ask-atlas", json={"query": "check cross-surface consistency"}).get_json()
    runtime = client.get("/api/runtime").get_json()
    assert ask["intent"] == "guarded_demo_workflow"
    assert "guarded_demo_workflow" in runtime
    assert runtime["execution"]["execution_environment"] == ask["execution_state"]["execution_environment"]


def test_post_run_review_query_is_read_only() -> None:
    svc = AskAtlasService(_Source())
    _ = svc.answer("validate blocked trade path")
    before = svc.execution.pending_requests(limit=50)
    _ = svc.answer("show last demo session summary")
    after = svc.execution.pending_requests(limit=50)
    assert before == after


def test_clean_shutdown_checklist_non_executing_and_mark_shutdown_explicit() -> None:
    svc = AskAtlasService(_Source())
    before = svc.execution.pending_requests(limit=50)
    out = svc.answer("clean shutdown checklist")
    after = svc.execution.pending_requests(limit=50)
    assert out["intent"] == "live_pilot_status"
    assert before == after
    mark = svc.answer("mark clean shutdown")
    assert mark["execution_state"]["last_shutdown_clean"] is True
