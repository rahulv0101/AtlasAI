from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {
            "execute_trades": False,
            "read_only_ui": False,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False, "max_freshness_age_ms": 120000},
            "max_freshness_age_ms": 120000,
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


def test_demo_summary_query_is_read_only() -> None:
    svc = AskAtlasService(_Source())
    before_requests = svc.execution.pending_requests(limit=50)
    before_control = svc.control.to_debug()
    out = svc.answer("show last demo session summary")
    after_requests = svc.execution.pending_requests(limit=50)
    after_control = svc.control.to_debug()
    assert out["intent"] == "demo_session_review"
    assert before_requests == after_requests
    assert before_control == after_control


def test_blocked_and_allowed_counts_reflect_actual_audit_state() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")

    ok_req = svc.answer("create execution request paper place_trade dry instrument=XAUUSD setup_family=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {ok_req}")
    svc.answer(f"run execution {ok_req}")

    blocked_req = svc.answer("create execution request paper place_trade dry instrument=XAUUSD setup_family=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {blocked_req}")
    svc.answer("activate kill switch")
    svc.answer(f"run execution {blocked_req}")

    out = svc.answer("show recent guarded execution attempts")
    review = dict(out.get("demo_session_review", {}) or {})
    assert int(review.get("allowed_count", 0) or 0) >= 1
    assert int(review.get("blocked_count", 0) or 0) >= 1
    top = [str(x.get("reason", "")) for x in list(review.get("top_block_reasons", []) or [])]
    assert "kill_switch_active" in top


def test_freshness_incident_appears_in_demo_summary() -> None:
    svc = AskAtlasService(_Source())
    gate = svc.execution
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=False,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    rid = str(req.get("id", ""))
    gate.approve_request(rid, approved_by="trader_engine")
    out = gate.execute(
        rid,
        runtime_flags={"execute_trades": False, "freshness_runtime": {"status_age_ms": 240000, "is_stale": True}},
        risk_policy={"daily_loss_cap_pct": 5.0},
        execute_fn=lambda _req: {"ok": True},
    )
    assert out.get("reason") == "freshness_stale"

    review_out = svc.answer("did freshness block anything recently?")
    review = dict(review_out.get("demo_session_review", {}) or {})
    freshness = [str(x.get("reason", "")) for x in list(review.get("freshness_incidents", []) or [])]
    assert "freshness_stale" in freshness


def test_review_queries_have_no_execution_side_effects() -> None:
    svc = AskAtlasService(_Source())
    for q in (
        "show last demo session summary",
        "what happened in the last demo run?",
        "show recent blocked trades",
        "summarize recent safety blocks",
        "did freshness block anything recently?",
    ):
        svc.answer(q)
    assert svc.execution.last_request() is None
