from __future__ import annotations

from atlas_ai.brain.control_layer import ControlLayer
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.54},
            "min_confidence_threshold": 0.58,
            "regime": {"name": "trend", "trend_strength": 0.42},
            "context_snapshot": {"volatility_state": "elevated"},
            "market": {"recent_close": [100, 100.2, 100.4, 100.6, 100.9, 101.0, 101.2, 101.4]},
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "risk", "message": "Filter blocked borderline setup near threshold."},
            {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
        ]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest"}

    def execute_trade(self) -> None:
        self.execute_called = True
        raise RuntimeError("Should never be called by control layer tests")


def test_suggestion_creation_becomes_control_item() -> None:
    svc = AskAtlasService(_Source())
    resp = svc.answer("Give me a safe experiment.")
    created = list(resp.get("control_items_created", []) or [])
    assert len(created) >= 1
    assert created[0]["status"] == "pending"


def test_general_self_questions_do_not_create_control_items() -> None:
    svc = AskAtlasService(_Source())
    before = svc.control.to_debug()
    resp = svc.answer("What can you do?")
    after = svc.control.to_debug()
    assert list(resp.get("control_items_created", []) or []) == []
    assert before == after


def test_read_only_questions_do_not_create_control_items() -> None:
    svc = AskAtlasService(_Source())
    queries = [
        "market bias",
        "what should I do",
        "buy or sell",
        "what setup is this?",
        "What should you do next?",
        "What small tuning do you recommend?",
        "Are you improving?",
        "How well are you adapting?",
    ]
    for query in queries:
        before = int(svc.control.to_debug()["pending_count"])
        resp = svc.answer(query)
        after = int(svc.control.to_debug()["pending_count"])
        assert list(resp.get("control_items_created", []) or []) == []
        assert after == before == 0


def test_approve_updates_state() -> None:
    svc = AskAtlasService(_Source())
    first = svc.answer("Give me a safe experiment.")
    created = list(first.get("control_items_created", []) or [])
    item_id = int(created[0]["id"])
    approve = svc.answer(f"approve item {item_id}")
    assert "approved" in approve["result"].lower()
    assert int(approve["control_state"]["approved_count"]) >= 1
    second = svc.answer(f"approve item {item_id}")
    assert "not found or not pending" in second["result"].lower()


def test_reject_stored_and_not_repeated_blindly() -> None:
    svc = AskAtlasService(_Source())
    first = svc.answer("Give me a safe experiment.")
    created1 = list(first.get("control_items_created", []) or [])
    item_id = int(created1[0]["id"])
    svc.answer(f"reject item {item_id}")
    second = svc.answer("Give me a safe experiment.")
    created2 = list(second.get("control_items_created", []) or [])
    # Dedup keeps already-seen suggestion from being blindly recreated.
    assert len(created2) == 0


def test_defer_keeps_item_trackable() -> None:
    svc = AskAtlasService(_Source())
    first = svc.answer("Give me a safe experiment.")
    created = list(first.get("control_items_created", []) or [])
    item_id = int(created[0]["id"])
    defer = svc.answer(f"defer item {item_id}")
    assert "deferred" in defer["result"].lower()
    pending = svc.answer("show pending items")
    # deferred items are not pending, but command still works safely
    assert "pending items" in pending["result"].lower()


def test_command_parsing_works_safely() -> None:
    c = ControlLayer.parse_command("approve item 3")
    assert c is not None and c["action"] == "approve" and c["item_id"] == 3
    c2 = ControlLayer.parse_command("show pending items")
    assert c2 is not None and c2["action"] == "show_pending"
    c3 = ControlLayer.parse_command("approve all pending items")
    assert c3 is not None and c3["action"] == "approve_all"
    c4 = ControlLayer.parse_command("approve all")
    assert c4 is not None and c4["action"] == "approve_all"
    c5 = ControlLayer.parse_command("approve all now")
    assert c5 is None
    c6 = ControlLayer.parse_command("why did you suggest item 11")
    assert c6 is not None and c6["action"] == "why" and c6["item_id"] == 11


def test_approve_all_updates_each_pending_item() -> None:
    svc = AskAtlasService(_Source())
    first = svc.answer("Give me a safe experiment.")
    second = svc.answer("Propose a tuning: what small tuning do you recommend?")
    created = [
        *list(first.get("control_items_created", []) or []),
        *list(second.get("control_items_created", []) or []),
    ]
    assert len(created) >= 2
    resp = svc.answer("approve all pending items")
    assert "approved" in resp["result"].lower()
    assert int(resp["control_state"]["pending_count"]) == 0
    assert int(resp["control_state"]["approved_count"]) >= len(created)


def test_approve_all_handles_empty_queue_safely() -> None:
    svc = AskAtlasService(_Source())
    resp = svc.answer("approve all pending items")
    assert resp["response_kind"] == "control"
    assert "no pending control items" in resp["result"].lower()


def test_set_all_pending_status_only_touches_pending_items() -> None:
    c = ControlLayer()
    created = c.ingest_from_debug(
        {
            "recommendation_debug": {
                "items": [
                    {
                        "recommendation": "one",
                        "reason": "r1",
                        "supporting_evidence": ["e1"],
                        "constraint": "safe",
                        "confidence": 0.6,
                        "priority": 4,
                    },
                    {
                        "recommendation": "two",
                        "reason": "r2",
                        "supporting_evidence": ["e2"],
                        "constraint": "safe",
                        "confidence": 0.7,
                        "priority": 3,
                    },
                ]
            }
        }
    )
    assert len(created) == 2
    c.set_status(int(created[0].id), "rejected")
    updated = c.set_all_pending_status("approved")
    assert len(updated) == 1
    assert updated[0].status == "approved"
    assert c.get(int(created[0].id)).status == "rejected"
    assert c.get(int(created[1].id)).status == "approved"


def test_no_automatic_execution_occurs() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("What small tuning do you recommend?")
    svc.answer("show pending items")
    assert source.execute_called is False


def test_invalid_id_is_handled_safely() -> None:
    svc = AskAtlasService(_Source())
    resp = svc.answer("approve item 999999")
    assert resp["response_kind"] == "control"
    assert "not found or not pending" in resp["result"].lower()


def test_priority_ordering_is_deterministic() -> None:
    c = ControlLayer()
    c.ingest_from_debug(
        {
            "recommendation_debug": {
                "items": [
                    {
                        "recommendation": "low priority",
                        "reason": "r1",
                        "supporting_evidence": ["e1"],
                        "constraint": "safe",
                        "confidence": 0.5,
                        "priority": 7,
                    },
                    {
                        "recommendation": "high priority",
                        "reason": "r2",
                        "supporting_evidence": ["e2"],
                        "constraint": "safe",
                        "confidence": 0.5,
                        "priority": 2,
                    },
                ]
            }
        }
    )
    pending = c.pending()
    assert len(pending) == 2
    assert pending[0].summary == "high priority"
    assert pending[1].summary == "low priority"


def test_explain_and_why_return_structured_detail() -> None:
    svc = AskAtlasService(_Source())
    first = svc.answer("Give me a safe experiment.")
    created = list(first.get("control_items_created", []) or [])
    item_id = int(created[0]["id"])
    explain = svc.answer(f"explain item {item_id}")
    why = svc.answer(f"why did you suggest item {item_id}")
    assert "source module" in explain["result"].lower()
    assert "priority reason" in explain["result"].lower()
    assert "trigger conditions" in why["result"].lower()
    assert "domain context" in why["result"].lower()


def test_workflow_items_require_approval_levels() -> None:
    c = ControlLayer()
    c.ingest_from_debug(
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
    pending = c.pending()
    promotion_items = [x for x in pending if x.type == "paper_tuning_promotion_candidate"]
    assert promotion_items
    assert promotion_items[0].approval_level == "high_risk_requires_explicit_confirmation"
    assert "approval" in promotion_items[0].approval_reason.lower()


def test_approve_can_progress_to_under_observation_state() -> None:
    svc = AskAtlasService(_Source())
    created = svc.control.ingest_from_debug(
        {
            "paper_tuning_debug": {
                "comparisons": [
                    {
                        "candidate_target": "persistence_requirement_offset",
                        "baseline_summary": "baseline stable",
                        "tuned_summary": "tuned inconclusive",
                        "outcome": "inconclusive",
                        "recommendation": "Gather more paper evidence.",
                    }
                ]
            }
        }
    )
    item = next(
        (
            x
            for x in created
            if x.type in {"extended_paper_validation", "paper_tuning_promotion_candidate"}
        ),
        None,
    )
    assert item is not None
    item_id = int(item.id)
    svc.answer(f"approve item {item_id}")
    item = svc.control.get(item_id)
    assert item is not None
    assert item.workflow_state in {"under_observation", "validated_in_paper"}
    assert item.status == "approved"
    assert item.approval_timestamp is not None


def test_invalid_workflow_transition_is_blocked() -> None:
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
    approved = c.set_status(target.id, "approved")
    assert approved is not None
    moved = c.transition_workflow(target.id, "under_observation", note="manual_review")
    assert moved is not None
    final = c.transition_workflow(target.id, "validated_in_paper", note="paper_confirmation")
    assert final is not None
    assert final.workflow_state == "validated_in_paper"


def test_repeated_high_risk_signals_raise_priority() -> None:
    c = ControlLayer()
    c.ingest_from_debug(
        {
            "paper_tuning_debug": {
                "comparisons": [
                    {
                        "candidate_target": "target_a",
                        "baseline_summary": "b1",
                        "tuned_summary": "t1",
                        "outcome": "helpful",
                        "recommendation": "r1",
                    },
                    {
                        "candidate_target": "target_b",
                        "baseline_summary": "b2",
                        "tuned_summary": "t2",
                        "outcome": "helpful",
                        "recommendation": "r2",
                    },
                ]
            }
        }
    )
    items = [x for x in c.pending() if x.type == "paper_tuning_promotion_candidate"]
    assert len(items) == 2
    # second item should get at least same-or-higher priority ranking (lower number) due to repeated trigger boost
    assert items[1].priority <= items[0].priority


def test_phase8_control_queries_are_supported() -> None:
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
    needs = svc.answer("What needs approval right now?")
    assert "requiring approval" in needs["result"].lower()
    item = next(
        (
            x
            for x in created
            if x.type in {"extended_paper_validation", "paper_tuning_promotion_candidate"}
        ),
        None,
    )
    assert item is not None
    svc.answer(f"approve item {int(item.id)}")
    under_obs = svc.answer("What is under observation?")
    assert "under observation" in under_obs["result"].lower() or "none" in under_obs["result"].lower()
