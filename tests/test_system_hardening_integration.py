from __future__ import annotations

from atlas_ai.brain.control_layer import ControlLayer
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.ui_server import create_ui_app
from atlas_ai.memory.knowledge_store import KnowledgeStore


class _Source:
    def __init__(self) -> None:
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.31, "side": "flat"},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}},
            "comparison_run_id": "atlas_wake_latest",
            "regime": {"name": "chop", "confidence": 0.42},
            "adaptation": {
                "active_learning_mode": "slow_path",
                "evidence_strength_score": 0.41,
                "stability_score": 0.73,
                "overreaction_score": 0.21,
            },
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "Conservative filter blocked borderline setup near threshold."}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False, "read_only_ui": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}

    def execute_trade(self) -> None:
        self.execute_called = True


class _SparseSource(_Source):
    def load_status(self) -> dict:
        return {"action": "hold", "message": "No fresh status details available."}

    def load_events(self, limit: int = 160) -> list[dict]:
        return []

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {}


def test_ask_atlas_end_to_end_includes_debug_control_execution() -> None:
    svc = AskAtlasService(_Source())
    resp = svc.answer("What state are you in?")
    assert resp["intent"] in {"status", "system_status_query"}
    assert resp["handled_as"] == "system"
    assert isinstance(resp.get("debug", {}), dict)
    assert isinstance(resp.get("control_state", {}), dict)
    assert isinstance(resp.get("execution_state", {}), dict)
    assert "Backend looks" in resp["result"]


def test_recommendation_to_control_item_to_explain_flow() -> None:
    svc = AskAtlasService(_Source())
    created = svc.answer("Give me a safe experiment.").get("control_items_created", [])
    assert created
    item_id = int(created[0]["id"])
    explained = svc.answer(f"explain item {item_id}")
    assert explained["response_kind"] == "control"
    assert f"Item {item_id}" in explained["result"]
    assert "Source module" in explained["result"]


def test_approved_execution_still_blocked_by_kill_switch() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    req = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")["execution_request"]
    req_id = req["id"]
    svc.answer(f"approve execution {req_id}")
    svc.answer("activate kill switch")
    blocked = svc.answer(f"run execution {req_id}")
    assert "kill_switch_active" in blocked["result"]


def test_dry_run_has_no_execution_side_effects() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    req_id = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id}")
    out = svc.answer(f"run execution {req_id}")
    assert "Dry-run completed" in out["result"]
    assert source.execute_called is False


def test_voice_query_cannot_trigger_execution_request() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    voice_resp = svc.answer("What should you do next?", mode="voice")
    assert voice_resp["intent"] in {"recommendation", "safe_next_step"}
    assert svc.execution.pending_requests(limit=10) == []
    assert source.execute_called is False


def test_sparse_data_returns_safe_conservative_response() -> None:
    svc = AskAtlasService(_SparseSource())
    resp = svc.answer("Are you improving?")
    assert resp["result"]
    assert "traceback" not in resp["result"].lower()
    assert resp["selected_domain"] in {"performance_feedback", "general_operator_help", "runtime_status", "comparison_results"}


def test_dashboard_runtime_endpoint_includes_execution_state(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    assert "execution" in payload
    execution = payload["execution"]
    assert "execution_gate_enabled" in execution
    assert "kill_switch_active" in execution
    assert "pending_count" in execution


def test_invalid_workflow_transition_is_blocked_integration() -> None:
    c = ControlLayer()
    created = c.ingest_from_debug(
        {
            "paper_tuning_debug": {
                "comparisons": [
                    {
                        "candidate_target": "persistence_requirement_offset",
                        "baseline_summary": "baseline stable",
                        "tuned_summary": "tuned inconclusive",
                        "outcome": "inconclusive",
                        "recommendation": "Gather more evidence.",
                    }
                ]
            }
        }
    )
    target = next(x for x in created if x.type == "extended_paper_validation")
    blocked = c.transition_workflow(target.id, "validated_in_paper")
    assert blocked is None


def test_paper_tuning_workflow_approval_does_not_create_live_execution() -> None:
    svc = AskAtlasService(_Source())
    created = svc.control.ingest_from_debug(
        {
            "paper_tuning_debug": {
                "comparisons": [
                    {
                        "candidate_target": "confidence_threshold_offset",
                        "baseline_summary": "baseline stable",
                        "tuned_summary": "tuned improved",
                        "outcome": "helpful",
                        "recommendation": "Keep in paper validation.",
                    }
                ]
            }
        }
    )
    item = next(x for x in created if x.type == "paper_tuning_promotion_candidate")
    svc.answer(f"approve item {item.id}")
    assert svc.execution.pending_requests(environment="live", limit=10) == []


def test_comparison_sparse_history_returns_conservative_direction() -> None:
    svc = AskAtlasService(_SparseSource())
    out = svc.answer("Compared to what?")
    assert out["result"]
    assert out["selected_domain"] == "comparison_results"
    assert "Compared to" in out["result"] or "safe interpretation" in out["result"].lower()
