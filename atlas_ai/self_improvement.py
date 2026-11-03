from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def knowledge_dir() -> Path:
    path = project_root() / "knowledge"
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_SELF_IMPROVEMENT_POLICY: dict[str, Any] = {
    "version": "1.0",
    "updated_at": "",
    "zones": {
        "allowed_auto_apply": [
            "observability",
            "diagnostics",
            "ui_clarity",
            "voice_response_wording",
            "test_coverage",
            "read_only_analysis",
        ],
        "approval_required": [
            "strategy_logic",
            "confidence_thresholds",
            "filters",
            "execution_gate_behavior",
            "automation_policy",
            "session_rules",
            "voice_command_actions",
        ],
        "forbidden_auto_apply": [
            "risk_policy_loosening",
            "execution_policy_loosening",
            "mode_switching",
            "kill_switch_semantics",
            "live_authority_expansion",
            "silent_retraining_affecting_live",
        ],
    },
    "promotion_requirements": {
        "min_evidence_strength": 0.7,
        "min_sample_count": 40,
        "requires_backtest": True,
        "requires_shadow_validation": True,
        "requires_operator_approval": True,
    },
    "guarded_rollout_defaults": {
        "environment": "demo",
        "max_trades": 2,
        "max_daily_loss_pct": 0.6,
        "allowed_instruments": ["XAUUSD"],
        "approval_required": True,
    },
}

DEFAULT_PROPOSAL_QUEUE: dict[str, Any] = {
    "version": "1.0",
    "updated_at": "",
    "proposals": [],
}

DEFAULT_LEARNED_INSIGHTS: dict[str, Any] = {
    "version": "1.0",
    "updated_at": "",
    "insights": [],
}

DEFAULT_ACTIVE_RULE_REGISTRY: dict[str, Any] = {
    "version": "atlas_rules_v1",
    "updated_at": "",
    "active_profiles": {},
    "rule_versions": [],
}

DEFAULT_ROLLOUT_REGISTRY: dict[str, Any] = {
    "version": "1.0",
    "updated_at": "",
    "rollouts": [],
}

PROPOSAL_STATUS_DRAFT = "draft"
PROPOSAL_STATUS_VALIDATED = "validated"
PROPOSAL_STATUS_AWAITING_APPROVAL = "awaiting_approval"
PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT = "approved_for_rollout"
PROPOSAL_STATUS_SHADOW_ACTIVE = "shadow_active"
PROPOSAL_STATUS_DEMO_LIMITED_ACTIVE = "demo_limited_active"
PROPOSAL_STATUS_PROMOTION_CANDIDATE = "promotion_candidate"
PROPOSAL_STATUS_PROMOTED = "promoted"
PROPOSAL_STATUS_ROLLED_BACK = "rolled_back"
PROPOSAL_STATUS_REJECTED = "rejected"

ACTIVE_ROLLOUT_STATUSES = {
    PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT,
    PROPOSAL_STATUS_SHADOW_ACTIVE,
    PROPOSAL_STATUS_DEMO_LIMITED_ACTIVE,
    PROPOSAL_STATUS_PROMOTION_CANDIDATE,
}

ALLOWED_ROLLOUT_TRANSITIONS: dict[str, set[str]] = {
    PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT: {
        PROPOSAL_STATUS_SHADOW_ACTIVE,
        PROPOSAL_STATUS_ROLLED_BACK,
        PROPOSAL_STATUS_REJECTED,
    },
    PROPOSAL_STATUS_SHADOW_ACTIVE: {
        PROPOSAL_STATUS_DEMO_LIMITED_ACTIVE,
        PROPOSAL_STATUS_ROLLED_BACK,
    },
    PROPOSAL_STATUS_DEMO_LIMITED_ACTIVE: {
        PROPOSAL_STATUS_PROMOTION_CANDIDATE,
        PROPOSAL_STATUS_ROLLED_BACK,
    },
    PROPOSAL_STATUS_PROMOTION_CANDIDATE: {
        PROPOSAL_STATUS_PROMOTED,
        PROPOSAL_STATUS_ROLLED_BACK,
    },
}


SELF_IMPROVEMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    "self_improvement_policy.json": DEFAULT_SELF_IMPROVEMENT_POLICY,
    "proposal_queue.json": DEFAULT_PROPOSAL_QUEUE,
    "learned_insights.json": DEFAULT_LEARNED_INSIGHTS,
    "active_rule_registry.json": DEFAULT_ACTIVE_RULE_REGISTRY,
    "rollout_registry.json": DEFAULT_ROLLOUT_REGISTRY,
}


def default_payload(filename: str) -> dict[str, Any]:
    payload = SELF_IMPROVEMENT_DEFAULTS.get(filename, {})
    return copy.deepcopy(payload)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def knowledge_path(filename: str) -> Path:
    return knowledge_dir() / filename


def ensure_self_improvement_knowledge(store: Any | None = None) -> None:
    for filename, payload in SELF_IMPROVEMENT_DEFAULTS.items():
        if store is not None and hasattr(store, "load_structured_knowledge_file") and hasattr(store, "save_structured_knowledge_file"):
            current = store.load_structured_knowledge_file(filename, None)
            if current is None:
                store.save_structured_knowledge_file(filename, copy.deepcopy(payload))
            continue
        path = knowledge_path(filename)
        if path.exists():
            continue
        path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")


def load_self_improvement_bundle(store: Any | None = None) -> dict[str, Any]:
    ensure_self_improvement_knowledge(store)
    bundle: dict[str, Any] = {}
    for filename in SELF_IMPROVEMENT_DEFAULTS:
        if store is not None and hasattr(store, "load_structured_knowledge_file"):
            bundle[filename] = store.load_structured_knowledge_file(filename, default_payload(filename))
        else:
            path = knowledge_path(filename)
            try:
                bundle[filename] = __import__("json").loads(path.read_text(encoding="utf-8"))
            except Exception:
                bundle[filename] = default_payload(filename)
    return bundle


def _save_bundle_item(store: Any, filename: str, payload: dict[str, Any]) -> None:
    if store is not None and hasattr(store, "save_structured_knowledge_file"):
        store.save_structured_knowledge_file(filename, payload)
    else:
        knowledge_path(filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_learned_insights(store: Any, insights_payload: dict[str, Any]) -> None:
    payload = dict(insights_payload or {})
    payload["updated_at"] = _now_iso()
    _save_bundle_item(store, "learned_insights.json", payload)


def save_proposal_queue(store: Any, proposals_payload: dict[str, Any]) -> None:
    payload = dict(proposals_payload or {})
    payload["updated_at"] = _now_iso()
    _save_bundle_item(store, "proposal_queue.json", payload)


def save_rollout_registry(store: Any, rollouts_payload: dict[str, Any]) -> None:
    payload = dict(rollouts_payload or {})
    payload["updated_at"] = _now_iso()
    _save_bundle_item(store, "rollout_registry.json", payload)


def save_active_rule_registry(store: Any, active_rules_payload: dict[str, Any]) -> None:
    payload = dict(active_rules_payload or {})
    payload["updated_at"] = _now_iso()
    _save_bundle_item(store, "active_rule_registry.json", payload)


def append_or_update_insight(store: Any, insight: dict[str, Any]) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    payload = dict(bundle.get("learned_insights.json", {}) or default_payload("learned_insights.json"))
    insights = list(payload.get("insights", []) or [])
    insight_id = str(insight.get("insight_id", "") or "").strip()
    updated = False
    for idx, current in enumerate(insights):
        if str(current.get("insight_id", "") or "").strip() == insight_id:
            merged = dict(current)
            merged.update(insight)
            merged["updated_at"] = _now_iso()
            insights[idx] = merged
            updated = True
            insight = merged
            break
    if not updated:
        insight = dict(insight)
        insight.setdefault("created_at", _now_iso())
        insight["updated_at"] = _now_iso()
        insights.append(insight)
    payload["insights"] = insights
    save_learned_insights(store, payload)
    return dict(insight)


def append_or_update_proposal(store: Any, proposal: dict[str, Any]) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposals = list(payload.get("proposals", []) or [])
    proposal_id = str(proposal.get("proposal_id", "") or "").strip()
    updated = False
    for idx, current in enumerate(proposals):
        if str(current.get("proposal_id", "") or "").strip() == proposal_id:
            merged = dict(current)
            merged.update(proposal)
            merged["updated_at"] = _now_iso()
            proposals[idx] = merged
            updated = True
            proposal = merged
            break
    if not updated:
        proposal = dict(proposal)
        proposal.setdefault("created_at", _now_iso())
        proposal["updated_at"] = _now_iso()
        proposals.append(proposal)
    payload["proposals"] = proposals
    save_proposal_queue(store, payload)
    return dict(proposal)


def append_or_update_rollout(store: Any, rollout: dict[str, Any]) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    payload = dict(bundle.get("rollout_registry.json", {}) or default_payload("rollout_registry.json"))
    rollouts = list(payload.get("rollouts", []) or [])
    rollout_id = str(rollout.get("rollout_id", "") or "").strip()
    updated = False
    for idx, current in enumerate(rollouts):
        if str(current.get("rollout_id", "") or "").strip() == rollout_id:
            merged = dict(current)
            merged.update(rollout)
            merged["updated_at"] = _now_iso()
            rollouts[idx] = merged
            updated = True
            rollout = merged
            break
    if not updated:
        rollout = dict(rollout)
        rollout.setdefault("created_at", _now_iso())
        rollout["updated_at"] = _now_iso()
        rollouts.append(rollout)
    payload["rollouts"] = rollouts
    save_rollout_registry(store, payload)
    return dict(rollout)


def load_validation_report(store: Any, proposal_id: str) -> dict[str, Any]:
    safe_id = str(proposal_id or "").strip()
    if not safe_id:
        return {}
    storage_base = Path(getattr(store, "base", project_root() / "storage"))
    path = storage_base / "proposal_reports" / f"{safe_id}.json"
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def _find_proposal(payload: dict[str, Any], proposal_id: str) -> dict[str, Any] | None:
    safe_id = str(proposal_id or "").strip()
    for proposal in list(payload.get("proposals", []) or []):
        if str(proposal.get("proposal_id", "") or "").strip() == safe_id:
            return dict(proposal)
    return None


def _status_for_rollout(target_status: str) -> str:
    status = str(target_status or "").strip().lower()
    return status or PROPOSAL_STATUS_DRAFT


def _complete_validation_step(proposal: dict[str, Any], step: str) -> dict[str, Any]:
    updated = dict(proposal or {})
    validation = dict(updated.get("validation", {}) or {})
    required = list(validation.get("required", []) or [])
    completed = list(validation.get("completed", []) or [])
    if step and step not in completed:
        completed.append(step)
    validation["required"] = required
    validation["completed"] = completed
    validation["missing"] = [item for item in required if item not in completed]
    updated["validation"] = validation
    return updated


def record_proposal_validation(store: Any, proposal_id: str, report: dict[str, Any], actor: str = "atlas") -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    proposal_payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposal = _find_proposal(proposal_payload, proposal_id)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": str(proposal_id or "")}

    readiness = dict(report.get("promotion_readiness", {}) or {})
    validation = dict(proposal.get("validation", {}) or {})
    completed = list(validation.get("completed", []) or [])
    if "validation_report_generated" not in completed:
        completed.append("validation_report_generated")
    target_status = PROPOSAL_STATUS_AWAITING_APPROVAL if bool(readiness.get("ready_for_rollout_review", False)) else PROPOSAL_STATUS_VALIDATED
    updated = append_or_update_proposal(
        store,
        {
            **proposal,
            "status": target_status,
            "validation": {
                "required": list(validation.get("required", []) or []),
                "completed": completed,
                "last_report_generated_at": str(report.get("generated_at", "") or _now_iso()),
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "proposal_status_changed",
                "actor": actor,
                "proposal_id": str(updated.get("proposal_id", "") or ""),
                "status": target_status,
                "reason": "validation_completed",
                "meta": {
                    "ready_for_rollout_review": bool(readiness.get("ready_for_rollout_review", False)),
                },
            }
        )
    return {"ok": True, "proposal": updated, "status": target_status}


def approve_proposal_for_rollout(
    store: Any,
    proposal_id: str,
    approved_by: str = "operator",
    note: str = "",
    actor: str = "operator",
) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    proposal_payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposal = _find_proposal(proposal_payload, proposal_id)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": str(proposal_id or "")}

    current_status = _status_for_rollout(proposal.get("status", ""))
    if current_status not in {PROPOSAL_STATUS_VALIDATED, PROPOSAL_STATUS_AWAITING_APPROVAL}:
        return {
            "ok": False,
            "reason": "proposal_not_ready_for_approval",
            "proposal_id": str(proposal_id or ""),
            "status": current_status,
        }

    active_rules = dict(bundle.get("active_rule_registry.json", {}) or default_payload("active_rule_registry.json"))
    snapshot_id = f"snapshot_{str(proposal_id).strip()}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    if hasattr(store, "save_rule_snapshot"):
        store.save_rule_snapshot(
            snapshot_id,
            {
                "proposal_id": str(proposal.get("proposal_id", "") or ""),
                "captured_at": _now_iso(),
                "active_rule_registry": active_rules,
            },
        )
    policy = dict(bundle.get("self_improvement_policy.json", {}) or default_payload("self_improvement_policy.json"))
    rollout_defaults = dict(policy.get("guarded_rollout_defaults", {}) or {})
    rollout_id = str(dict(proposal.get("rollout", {}) or {}).get("rollout_id", "") or f"rollout_{str(proposal_id).strip()}")
    rollout = append_or_update_rollout(
        store,
        {
            "rollout_id": rollout_id,
            "proposal_id": str(proposal.get("proposal_id", "") or ""),
            "status": PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT,
            "environment": str(rollout_defaults.get("environment", "demo") or "demo"),
            "max_trades": int(rollout_defaults.get("max_trades", 2) or 2),
            "max_daily_loss_pct": float(rollout_defaults.get("max_daily_loss_pct", 0.6) or 0.6),
            "allowed_instruments": list(rollout_defaults.get("allowed_instruments", ["XAUUSD"]) or ["XAUUSD"]),
            "approval_required": bool(rollout_defaults.get("approval_required", True)),
            "approved_by": str(approved_by or "operator"),
            "approved_at": _now_iso(),
            "snapshot_id": snapshot_id,
            "note": str(note or ""),
        },
    )
    updated = append_or_update_proposal(
        store,
        {
            **_complete_validation_step(proposal, "operator_review"),
            "status": PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT,
            "approval": {
                "required": True,
                "approved": True,
                "approved_by": str(approved_by or "operator"),
                "approved_at": _now_iso(),
                "note": str(note or ""),
            },
            "rollout": {
                "required": True,
                "rollout_id": rollout_id,
                "promotion_ready": False,
            },
            "rollback_plan": {
                "restore_snapshot_id": snapshot_id,
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "approval_granted",
                "actor": actor,
                "proposal_id": str(updated.get("proposal_id", "") or ""),
                "status": PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT,
                "reason": "operator_approved_for_rollout",
                "meta": {
                    "rollout_id": rollout_id,
                    "snapshot_id": snapshot_id,
                    "approved_by": str(approved_by or "operator"),
                },
            }
        )
        store.append_self_improvement_audit(
            {
                "event": "rollout_started",
                "actor": actor,
                "proposal_id": str(updated.get("proposal_id", "") or ""),
                "rollout_id": rollout_id,
                "status": PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT,
                "reason": "approved_for_guarded_rollout",
            }
        )
    return {"ok": True, "proposal": updated, "rollout": rollout, "snapshot_id": snapshot_id}


def close_proposal_without_rollout(
    store: Any,
    proposal_id: str,
    *,
    target_status: str,
    actor: str = "operator",
    reason: str = "",
    note: str = "",
) -> dict[str, Any]:
    normalized = _status_for_rollout(target_status)
    if normalized not in {PROPOSAL_STATUS_REJECTED, "deferred"}:
        return {"ok": False, "reason": "unsupported_close_status", "target_status": normalized}
    bundle = load_self_improvement_bundle(store)
    proposal_payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposal = _find_proposal(proposal_payload, proposal_id)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": str(proposal_id or "")}

    updated = append_or_update_proposal(
        store,
        {
            **_complete_validation_step(proposal, "operator_review"),
            "status": normalized,
            "operator_review": {
                "decision": normalized,
                "reviewed_by": str(actor or "operator"),
                "reviewed_at": _now_iso(),
                "reason": str(reason or ""),
                "note": str(note or ""),
            },
            "rollout": {
                **dict(proposal.get("rollout", {}) or {}),
                "promotion_ready": False,
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "proposal_status_changed",
                "actor": actor,
                "proposal_id": str(updated.get("proposal_id", "") or ""),
                "status": normalized,
                "reason": str(reason or f"operator_{normalized}"),
                "meta": {"note": str(note or ""), "does_not_create_rollout": True},
            }
        )
    return {
        "ok": True,
        "proposal": updated,
        "status": normalized,
        "policy": {
            "does_not_execute": True,
            "does_not_create_rollout": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
        },
    }


def transition_rollout_status(
    store: Any,
    rollout_id: str,
    target_status: str,
    actor: str = "operator",
    reason: str = "",
    note: str = "",
) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    rollout_payload = dict(bundle.get("rollout_registry.json", {}) or default_payload("rollout_registry.json"))
    rollouts = list(rollout_payload.get("rollouts", []) or [])
    current = next(
        (dict(item) for item in rollouts if str(item.get("rollout_id", "") or "").strip() == str(rollout_id or "").strip()),
        None,
    )
    if current is None:
        return {"ok": False, "reason": "rollout_not_found", "rollout_id": str(rollout_id or "")}

    current_status = _status_for_rollout(current.get("status", ""))
    normalized_target = _status_for_rollout(target_status)
    if normalized_target != PROPOSAL_STATUS_ROLLED_BACK and normalized_target not in ALLOWED_ROLLOUT_TRANSITIONS.get(current_status, set()):
        return {
            "ok": False,
            "reason": "invalid_rollout_transition",
            "rollout_id": str(rollout_id or ""),
            "current_status": current_status,
            "target_status": normalized_target,
        }

    rollout = append_or_update_rollout(
        store,
        {
            **current,
            "status": normalized_target,
            "note": str(note or current.get("note", "") or ""),
            "last_transition_reason": str(reason or ""),
            "last_transition_at": _now_iso(),
            "last_transition_actor": str(actor or "operator"),
        },
    )

    proposal_id = str(current.get("proposal_id", "") or "")
    proposal = None
    if proposal_id:
        proposal_bundle = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
        existing = _find_proposal(proposal_bundle, proposal_id)
        if existing is not None:
            proposal = append_or_update_proposal(
                store,
                {
                    **existing,
                    "status": normalized_target,
                    "rollout": {
                        "required": True,
                        "rollout_id": str(rollout.get("rollout_id", "") or ""),
                        "promotion_ready": normalized_target == PROPOSAL_STATUS_PROMOTION_CANDIDATE,
                    },
                },
            )

    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "rollout_transitioned",
                "actor": actor,
                "proposal_id": proposal_id,
                "rollout_id": str(rollout.get("rollout_id", "") or ""),
                "status": normalized_target,
                "reason": str(reason or ""),
                "meta": {
                    "from_status": current_status,
                    "note": str(note or ""),
                },
            }
        )
        if normalized_target == PROPOSAL_STATUS_ROLLED_BACK:
            store.append_self_improvement_audit(
                {
                    "event": "rollback_completed",
                    "actor": actor,
                    "proposal_id": proposal_id,
                    "rollout_id": str(rollout.get("rollout_id", "") or ""),
                    "status": normalized_target,
                    "reason": str(reason or "operator_requested_rollback"),
                    "meta": {
                        "snapshot_id": str(rollout.get("snapshot_id", "") or ""),
                    },
                }
            )
    return {"ok": True, "rollout": rollout, "proposal": proposal}


def _next_registry_version_label(active_rules: dict[str, Any]) -> str:
    current = str(active_rules.get("version", "atlas_rules_v1") or "atlas_rules_v1").strip()
    prefix = "atlas_rules_v"
    if current.startswith(prefix):
        try:
            return f"{prefix}{int(current[len(prefix):]) + 1}"
        except Exception:
            return f"{prefix}2"
    return f"{prefix}2"


def _next_rule_version_id(active_rules: dict[str, Any]) -> str:
    versions = list(active_rules.get("rule_versions", []) or [])
    return f"sp_v{len(versions) + 1}"


def promote_proposal_to_active_rules(
    store: Any,
    proposal_id: str,
    approved_by: str = "operator",
    actor: str = "operator",
    note: str = "",
) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    proposal_payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposal = _find_proposal(proposal_payload, proposal_id)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": str(proposal_id or "")}

    status = _status_for_rollout(proposal.get("status", ""))
    if status != PROPOSAL_STATUS_PROMOTION_CANDIDATE:
        return {
            "ok": False,
            "reason": "proposal_not_ready_for_promotion",
            "proposal_id": str(proposal_id or ""),
            "status": status,
        }

    rollout_id = str(dict(proposal.get("rollout", {}) or {}).get("rollout_id", "") or "")
    rollout_bundle = dict(bundle.get("rollout_registry.json", {}) or default_payload("rollout_registry.json"))
    rollout = next(
        (dict(item) for item in list(rollout_bundle.get("rollouts", []) or []) if str(item.get("rollout_id", "") or "").strip() == rollout_id),
        None,
    )
    if rollout is None:
        return {"ok": False, "reason": "rollout_not_found", "proposal_id": str(proposal_id or ""), "rollout_id": rollout_id}

    active_rules = dict(bundle.get("active_rule_registry.json", {}) or default_payload("active_rule_registry.json"))
    next_registry_version = _next_registry_version_label(active_rules)
    rule_version_id = _next_rule_version_id(active_rules)
    snapshot_id = str(dict(proposal.get("rollback_plan", {}) or {}).get("restore_snapshot_id", "") or rollout.get("snapshot_id", "") or "")
    rule_versions = list(active_rules.get("rule_versions", []) or [])
    rule_versions.append(
        {
            "rule_version_id": rule_version_id,
            "source_proposal_id": str(proposal.get("proposal_id", "") or ""),
            "rollout_id": rollout_id,
            "approved_by": str(approved_by or "operator"),
            "approved_at": _now_iso(),
            "snapshot_id": snapshot_id,
            "changes": [dict(proposal.get("proposed_change", {}) or {})],
            "status": PROPOSAL_STATUS_PROMOTED,
            "note": str(note or ""),
        }
    )
    updated_registry = {
        **active_rules,
        "version": next_registry_version,
        "active_profiles": {
            **dict(active_rules.get("active_profiles", {}) or {}),
            "guarded_demo": {
                "strategy_params_version": rule_version_id,
                "source_proposal_id": str(proposal.get("proposal_id", "") or ""),
                "rollout_id": rollout_id,
            },
        },
        "rule_versions": rule_versions,
    }
    save_active_rule_registry(store, updated_registry)

    updated_rollout = append_or_update_rollout(
        store,
        {
            **rollout,
            "status": PROPOSAL_STATUS_PROMOTED,
            "promoted_at": _now_iso(),
            "promoted_rule_version_id": rule_version_id,
            "note": str(note or rollout.get("note", "") or ""),
        },
    )
    updated_proposal = append_or_update_proposal(
        store,
        {
            **proposal,
            "status": PROPOSAL_STATUS_PROMOTED,
            "rollout": {
                "required": True,
                "rollout_id": rollout_id,
                "promotion_ready": True,
                "promoted_rule_version_id": rule_version_id,
            },
            "approval": {
                **dict(proposal.get("approval", {}) or {}),
                "approved": True,
                "approved_by": str(approved_by or "operator"),
                "approved_at": _now_iso(),
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "promotion_completed",
                "actor": actor,
                "proposal_id": str(updated_proposal.get("proposal_id", "") or ""),
                "rollout_id": rollout_id,
                "status": PROPOSAL_STATUS_PROMOTED,
                "reason": "approved_rule_registry_updated",
                "meta": {
                    "rule_registry_version": next_registry_version,
                    "rule_version_id": rule_version_id,
                    "snapshot_id": snapshot_id,
                },
            }
        )
    return {
        "ok": True,
        "proposal": updated_proposal,
        "rollout": updated_rollout,
        "active_rule_registry": updated_registry,
        "rule_version_id": rule_version_id,
    }


def rollback_rollout_promotion(
    store: Any,
    rollout_id: str,
    actor: str = "operator",
    reason: str = "",
    note: str = "",
) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    rollout_bundle = dict(bundle.get("rollout_registry.json", {}) or default_payload("rollout_registry.json"))
    rollout = next(
        (dict(item) for item in list(rollout_bundle.get("rollouts", []) or []) if str(item.get("rollout_id", "") or "").strip() == str(rollout_id or "").strip()),
        None,
    )
    if rollout is None:
        return {"ok": False, "reason": "rollout_not_found", "rollout_id": str(rollout_id or "")}

    snapshot_id = str(rollout.get("snapshot_id", "") or "")
    if not snapshot_id or not hasattr(store, "load_rule_snapshot"):
        return {"ok": False, "reason": "rollback_snapshot_missing", "rollout_id": str(rollout_id or "")}

    snapshot = store.load_rule_snapshot(snapshot_id)
    prior_registry = dict(snapshot.get("active_rule_registry", {}) or {})
    if not prior_registry:
        return {"ok": False, "reason": "rollback_snapshot_invalid", "rollout_id": str(rollout_id or ""), "snapshot_id": snapshot_id}

    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "rollback_triggered",
                "actor": actor,
                "proposal_id": str(rollout.get("proposal_id", "") or ""),
                "rollout_id": str(rollout.get("rollout_id", "") or ""),
                "reason": str(reason or "operator_requested_rollback"),
                "meta": {"snapshot_id": snapshot_id},
            }
        )

    save_active_rule_registry(store, prior_registry)
    result = transition_rollout_status(
        store,
        str(rollout.get("rollout_id", "") or ""),
        target_status=PROPOSAL_STATUS_ROLLED_BACK,
        actor=actor,
        reason=str(reason or "operator_requested_rollback"),
        note=note,
    )
    result["active_rule_registry"] = prior_registry
    result["snapshot_id"] = snapshot_id
    return result


def build_self_improvement_runtime_guard(store: Any) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    summary = build_self_improvement_summary(bundle)
    proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    active_rules = dict(bundle.get("active_rule_registry.json", {}) or {})
    draft_count = sum(1 for item in proposals if _status_for_rollout(item.get("status", "")) == PROPOSAL_STATUS_DRAFT)
    promoted_count = sum(
        1 for item in list(active_rules.get("rule_versions", []) or []) if _status_for_rollout(item.get("status", "")) == PROPOSAL_STATUS_PROMOTED
    )
    latest_rule = dict(list(active_rules.get("rule_versions", []) or [])[-1] if list(active_rules.get("rule_versions", []) or []) else {})
    return {
        "execution_rule_source": "approved_registry_only",
        "approved_registry_loaded": True,
        "draft_rules_loaded_into_execution": False,
        "active_rule_registry_version": str(active_rules.get("version", "") or ""),
        "active_rule_version_id": str(latest_rule.get("rule_version_id", "") or ""),
        "latest_promoted_proposal_id": str(latest_rule.get("source_proposal_id", "") or ""),
        "proposal_count": int(summary.get("proposal_count", 0) or 0),
        "approval_pending_count": int(summary.get("approval_pending_count", 0) or 0),
        "rollout_active": bool(summary.get("rollout_active", False)),
        "latest_proposal_id": str(summary.get("latest_proposal_id", "") or ""),
        "pending_proposal_id": str(summary.get("pending_proposal_id", "") or ""),
        "pending_proposal_category": str(summary.get("pending_proposal_category", "") or ""),
        "pending_proposal_target_blocker": str(summary.get("pending_proposal_target_blocker", "") or ""),
        "draft_count": draft_count,
        "promoted_rule_count": promoted_count,
        "checked_at": _now_iso(),
    }


def save_validation_report(store: Any, proposal_id: str, report: dict[str, Any]) -> Path:
    safe_id = str(proposal_id or "").strip() or "unknown_proposal"
    storage_base = Path(getattr(store, "base", project_root() / "storage"))
    report_dir = storage_base / "proposal_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{safe_id}.json"
    payload = dict(report or {})
    payload.setdefault("proposal_id", safe_id)
    payload["updated_at"] = _now_iso()
    if store is not None and hasattr(store, "_write_json_atomic"):
        store._write_json_atomic(path, payload)  # type: ignore[attr-defined]
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def build_self_improvement_summary(bundle: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(bundle or {})
    proposals = list(dict(data.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    insights = list(dict(data.get("learned_insights.json", {}) or {}).get("insights", []) or [])
    rollouts = list(dict(data.get("rollout_registry.json", {}) or {}).get("rollouts", []) or [])
    active_rules = dict(data.get("active_rule_registry.json", {}) or {})
    approval_pending = sum(1 for item in proposals if str(item.get("status", "")).strip().lower() == "awaiting_approval")
    pending_proposals = [
        dict(item)
        for item in proposals
        if str(dict(item).get("status", "") or "").strip().lower() == PROPOSAL_STATUS_AWAITING_APPROVAL
    ]
    latest_pending = pending_proposals[-1] if pending_proposals else {}
    validated = sum(1 for item in proposals if str(item.get("status", "")).strip().lower() == PROPOSAL_STATUS_VALIDATED)
    rollout_active = any(
        str(item.get("status", "")).strip().lower() in ACTIVE_ROLLOUT_STATUSES
        for item in rollouts
    )
    rollback_available = any(str(item.get("snapshot_id", "") or "").strip() for item in rollouts)
    return {
        "policy_version": str(dict(data.get("self_improvement_policy.json", {}) or {}).get("version", "1.0") or "1.0"),
        "proposal_count": len(proposals),
        "validated_count": validated,
        "approval_pending_count": approval_pending,
        "insight_count": len(insights),
        "rollout_count": len(rollouts),
        "rollout_active": rollout_active,
        "rollback_available": rollback_available,
        "active_rule_version": str(active_rules.get("version", "") or ""),
        "latest_proposal_id": str(proposals[-1].get("proposal_id", "") or "") if proposals else "",
        "pending_proposal_id": str(latest_pending.get("proposal_id", "") or ""),
        "pending_proposal_category": str(latest_pending.get("category", "") or ""),
        "pending_proposal_target_blocker": str(dict(latest_pending.get("scope", {}) or {}).get("target_blocker", "") or "") if latest_pending else "",
        "latest_insight_id": str(insights[-1].get("insight_id", "") or "") if insights else "",
        "latest_rollout_id": str(rollouts[-1].get("rollout_id", "") or "") if rollouts else "",
    }


def reconcile_proposal_queue_with_rollouts(store: Any) -> dict[str, Any]:
    """
    Reconcile stale awaiting_approval proposals against rollout/approval state.
    FIX: prevent already-approved proposals from reappearing as pending after restart.
    """
    bundle = load_self_improvement_bundle(store)
    proposal_payload = dict(bundle.get("proposal_queue.json", {}) or default_payload("proposal_queue.json"))
    proposals = list(proposal_payload.get("proposals", []) or [])
    rollout_payload = dict(bundle.get("rollout_registry.json", {}) or default_payload("rollout_registry.json"))
    rollouts = list(rollout_payload.get("rollouts", []) or [])
    rollout_by_proposal = {
        str(item.get("proposal_id", "") or "").strip(): dict(item)
        for item in rollouts
        if str(item.get("proposal_id", "") or "").strip()
    }
    changed = 0
    reconciled: list[str] = []
    new_proposals: list[dict[str, Any]] = []
    for item in proposals:
        proposal = dict(item or {})
        proposal_id = str(proposal.get("proposal_id", "") or "").strip()
        status = str(proposal.get("status", "") or "").strip().lower()
        approval = dict(proposal.get("approval", {}) or {})
        has_approved_flag = bool(approval.get("approved", False))
        rollout = rollout_by_proposal.get(proposal_id, {})
        rollout_status = str(rollout.get("status", "") or "").strip().lower()
        should_reconcile = (
            status == PROPOSAL_STATUS_AWAITING_APPROVAL
            and (
                has_approved_flag
                or rollout_status in ACTIVE_ROLLOUT_STATUSES
                or rollout_status in {
                    PROPOSAL_STATUS_PROMOTED,
                    PROPOSAL_STATUS_ROLLED_BACK,
                    PROPOSAL_STATUS_REJECTED,
                }
            )
        )
        if should_reconcile:
            proposal["status"] = rollout_status or PROPOSAL_STATUS_APPROVED_FOR_ROLLOUT
            if rollout:
                proposal["rollout"] = {
                    "required": True,
                    "rollout_id": str(rollout.get("rollout_id", "") or ""),
                    "promotion_ready": proposal["status"] == PROPOSAL_STATUS_PROMOTION_CANDIDATE,
                }
            proposal["updated_at"] = _now_iso()
            changed += 1
            if proposal_id:
                reconciled.append(proposal_id)
        new_proposals.append(proposal)
    if changed:
        proposal_payload["proposals"] = new_proposals
        save_proposal_queue(store, proposal_payload)
        if hasattr(store, "append_self_improvement_audit"):
            store.append_self_improvement_audit(
                {
                    "event": "proposal_queue_reconciled",
                    "actor": "system",
                    "status": "ok",
                    "reason": "startup_runtime_reconcile",
                    "meta": {"reconciled_count": changed, "proposal_ids": reconciled},
                }
            )
    return {"ok": True, "reconciled_count": changed, "proposal_ids": reconciled}
