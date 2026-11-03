from __future__ import annotations

import json

from atlas_ai.learning.outcome_intelligence import sync_self_improvement_from_outcome_intelligence
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import load_self_improvement_bundle
from atlas_ai.ui_server import create_ui_app


def _sample_outcome_intelligence() -> dict:
    return {
        "sample_count": 72,
        "loser_patterns": {
            "top_patterns": {
                "trend_bucket": [
                    {"value": "weak", "samples": 44, "win_rate": 0.25, "avg_pnl_zar": -320.0}
                ],
                "strategy": [
                    {"value": "momentum_continuation", "samples": 46, "win_rate": 0.32, "avg_pnl_zar": -140.0}
                ],
            }
        },
        "pattern_comparison": {
            "winner_vs_loser_deltas": {
                "confidence": 0.11,
                "trend_strength": 0.09,
                "duration_minutes": 3.5,
            }
        },
    }


def _seed_proposal(store: KnowledgeStore) -> str:
    sync_self_improvement_from_outcome_intelligence(store, _sample_outcome_intelligence())
    bundle = load_self_improvement_bundle(store)
    return str(bundle["proposal_queue.json"]["proposals"][-1]["proposal_id"])


def _transition(client, rollout_id: str, target_status: str):
    return client.post(
        f"/api/self-improvement/rollouts/{rollout_id}/transition",
        json={
            "target_status": target_status,
            "confirmation": f"transition {rollout_id} {target_status}",
        },
    )


def test_validate_endpoint_transitions_proposal_to_awaiting_approval(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    payload = client.post(f"/api/self-improvement/proposals/{proposal_id}/validate").get_json()

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    audit = store.tail_self_improvement_audit(limit=20)

    assert payload["ok"] is True
    assert proposal["status"] == "awaiting_approval"
    assert "validation_report_generated" in proposal["validation"]["completed"]
    assert any(row.get("event") == "proposal_status_changed" for row in audit)


def test_approve_endpoint_creates_rollout_and_snapshot_link(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    payload = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "note": "guarded demo review passed", "confirmation": f"approve {proposal_id}"},
    ).get_json()

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    rollout = bundle["rollout_registry.json"]["rollouts"][-1]
    snapshot_id = str(proposal["rollback_plan"]["restore_snapshot_id"])

    assert payload["ok"] is True
    assert proposal["status"] == "approved_for_rollout"
    assert proposal["approval"]["approved"] is True
    assert "operator_review" in proposal["validation"]["completed"]
    assert "operator_review" not in proposal["validation"]["missing"]
    assert rollout["proposal_id"] == proposal_id
    assert rollout["status"] == "approved_for_rollout"
    assert snapshot_id
    assert store.load_rule_snapshot(snapshot_id)["proposal_id"] == proposal_id


def test_rollout_transition_updates_rollout_and_proposal_status(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])
    transition = client.post(
        f"/api/self-improvement/rollouts/{rollout_id}/transition",
        json={
            "target_status": "shadow_active",
            "reason": "start shadow comparison",
            "confirmation": f"transition {rollout_id} shadow_active",
        },
    ).get_json()

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    rollout = bundle["rollout_registry.json"]["rollouts"][-1]

    assert transition["ok"] is True
    assert rollout["status"] == "shadow_active"
    assert proposal["status"] == "shadow_active"


def test_rollout_transition_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])

    response = client.post(
        f"/api/self-improvement/rollouts/{rollout_id}/transition",
        json={"target_status": "shadow_active"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == f"transition {rollout_id} shadow_active"


def test_self_improvement_mutations_are_blocked_in_read_only_comparison_mode(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    root = tmp_path / "comparison_runs" / "run123"
    (root / "experiment").mkdir(parents=True)
    (root / "control").mkdir(parents=True)
    (root / "comparison_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    app = create_ui_app(store, comparison_root=str(tmp_path / "comparison_runs"), comparison_run_id="run123")
    client = app.test_client()
    response = client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")

    assert response.status_code == 409
    assert "read-only" in response.get_json()["error"].lower()


def test_approve_endpoint_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    response = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == f"approve {proposal_id}"


def test_close_endpoint_defers_proposal_without_rollout(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    payload = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/close",
        json={
            "target_status": "deferred",
            "reason": "operator_wants_more_time",
            "confirmation": f"defer {proposal_id}",
        },
    ).get_json()

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    rollouts = bundle["rollout_registry.json"]["rollouts"]

    assert payload["ok"] is True
    assert payload["policy"]["does_not_create_rollout"] is True
    assert proposal["status"] == "deferred"
    assert proposal["operator_review"]["decision"] == "deferred"
    assert "operator_review" in proposal["validation"]["completed"]
    assert "operator_review" not in proposal["validation"]["missing"]
    assert rollouts == []


def test_close_endpoint_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    response = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/close",
        json={"target_status": "rejected", "reason": "operator_rejected"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == f"reject {proposal_id}"


def test_close_endpoint_rejects_unsupported_status(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    response = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/close",
        json={"target_status": "approved_for_rollout"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "unsupported_close_status"


def test_promote_endpoint_updates_active_rule_registry(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])
    _transition(client, rollout_id, "shadow_active")
    _transition(client, rollout_id, "demo_limited_active")
    _transition(client, rollout_id, "promotion_candidate")

    promote = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/promote",
        json={
            "approved_by": "operator",
            "note": "promote approved demo candidate",
            "confirmation": f"promote {proposal_id}",
        },
    ).get_json()

    bundle = load_self_improvement_bundle(store)
    registry = bundle["active_rule_registry.json"]
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    rollout = bundle["rollout_registry.json"]["rollouts"][-1]

    assert promote["ok"] is True
    assert registry["version"] == "atlas_rules_v2"
    assert registry["active_profiles"]["guarded_demo"]["source_proposal_id"] == proposal_id
    assert proposal["status"] == "promoted"
    assert rollout["status"] == "promoted"


def test_promote_endpoint_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])
    _transition(client, rollout_id, "shadow_active")
    _transition(client, rollout_id, "demo_limited_active")
    _transition(client, rollout_id, "promotion_candidate")

    response = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/promote",
        json={"approved_by": "operator"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == f"promote {proposal_id}"


def test_rollback_endpoint_restores_prior_active_rule_registry(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])
    _transition(client, rollout_id, "shadow_active")
    _transition(client, rollout_id, "demo_limited_active")
    _transition(client, rollout_id, "promotion_candidate")
    client.post(
        f"/api/self-improvement/proposals/{proposal_id}/promote",
        json={"approved_by": "operator", "confirmation": f"promote {proposal_id}"},
    )

    rollback = client.post(
        f"/api/self-improvement/rollouts/{rollout_id}/rollback",
        json={"reason": "operator_requested_rollback", "confirmation": f"rollback {rollout_id}"},
    ).get_json()

    bundle = load_self_improvement_bundle(store)
    registry = bundle["active_rule_registry.json"]
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    rollout = bundle["rollout_registry.json"]["rollouts"][-1]

    assert rollback["ok"] is True
    assert registry["version"] == "atlas_rules_v1"
    assert proposal["status"] == "rolled_back"
    assert rollout["status"] == "rolled_back"


def test_rollback_endpoint_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    proposal_id = _seed_proposal(store)

    app = create_ui_app(store)
    client = app.test_client()
    client.post(f"/api/self-improvement/proposals/{proposal_id}/validate")
    approve = client.post(
        f"/api/self-improvement/proposals/{proposal_id}/approve",
        json={"approved_by": "operator", "confirmation": f"approve {proposal_id}"},
    ).get_json()
    rollout_id = str(approve["rollout"]["rollout_id"])

    response = client.post(
        f"/api/self-improvement/rollouts/{rollout_id}/rollback",
        json={"reason": "operator_requested_rollback"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == f"rollback {rollout_id}"


def test_runtime_endpoint_exposes_self_improvement_runtime_guard(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    app = create_ui_app(store)
    client = app.test_client()

    payload = client.get("/api/runtime").get_json()

    assert payload["self_improvement_runtime"]["execution_rule_source"] == "approved_registry_only"
    assert payload["self_improvement_runtime"]["approved_registry_loaded"] is True
