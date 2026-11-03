from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from atlas_ai.memory.jsonl_tail import read_jsonl_tail

from atlas_ai.self_improvement import append_or_update_proposal, load_self_improvement_bundle, save_validation_report


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_side(row: dict[str, Any]) -> str:
    side = str(row.get("side", "") or "").strip().lower()
    return side if side in {"buy", "sell"} else "flat"


def _side_from_blocker_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    for side in ("buy", "sell"):
        if f"side={side}" in lowered or f"side: {side}" in lowered:
            return side
    return "flat"


def build_missed_entry_audit(status: dict[str, Any]) -> dict[str, Any] | None:
    """Build an audit-only record for strong-looking setups that did not execute."""
    st = dict(status or {})
    action = str(st.get("action", "") or "").strip().lower()
    if action not in {"hold", "blocked"}:
        return None

    chosen = dict(st.get("chosen", {}) or {})
    best = dict(st.get("best_actionable_candidate", {}) or {})
    decision = dict(st.get("decision_explanation", {}) or {})
    trace = dict(st.get("decision_trace", {}) or {})
    analysis = dict(st.get("decision_analysis", {}) or {})
    context = dict(st.get("context_snapshot", {}) or {})
    signal_summary = dict(st.get("signal_summary", {}) or {})
    learning_status = dict(st.get("learning_status", st.get("learning_status_snapshot", {})) or {})
    market = dict(st.get("market", {}) or {})
    primary_blocker = str(
        st.get("primary_blocker")
        or decision.get("primary_blocker")
        or dict(trace.get("blocker_summary", {}) or {}).get("primary")
        or st.get("message", "")
        or ""
    )
    blocker_class = str(
        st.get("blocker_class")
        or decision.get("blocker_class")
        or dict(trace.get("blocker_summary", {}) or {}).get("class")
        or "unknown"
    )

    advisory_score = _safe_float(
        best.get(
            "advisory_setup_quality_score",
            dict(st.get("setup_quality_observation", {}) or {}).get("score", 0.0),
        )
    )
    best_score = _safe_float(best.get("score", 0.0))
    chosen_conf = _safe_float(chosen.get("confidence", 0.0))
    best_conf = _safe_float(best.get("confidence", 0.0))
    best_side = _safe_side(best)
    chosen_side = _safe_side(chosen)
    blocker_side = _side_from_blocker_text(primary_blocker or str(st.get("message", "") or ""))
    audit_side = best_side if best_side in {"buy", "sell"} else chosen_side if chosen_side in {"buy", "sell"} else blocker_side
    confluence_score = _safe_float(analysis.get("confluence_score", 0.0))
    confluence_threshold = _safe_float(analysis.get("confluence_threshold", 0.0))
    consensus_blocked_candidate = bool(
        "consensus guard blocked trade" in f"{primary_blocker} {st.get('message', '')}".lower()
        and audit_side in {"buy", "sell"}
    )

    has_meaningful_setup = bool(
        best_side in {"buy", "sell"}
        or consensus_blocked_candidate
        or advisory_score >= 65.0
        or best_score >= 0.45
        or (chosen_side in {"buy", "sell"} and chosen_conf >= 0.40)
        or (confluence_threshold > 0 and confluence_score >= confluence_threshold - 8)
    )
    if not has_meaningful_setup:
        return None

    fingerprint = _fingerprint(
        symbol=str(st.get("symbol", "XAUUSD") or "XAUUSD"),
        timeframe=str(st.get("timeframe", "") or ""),
        side=audit_side,
        strategy=str(best.get("strategy", "") or chosen.get("strategy", "") or ""),
        primary_blocker=primary_blocker,
        context=context,
    )
    record = {
        "schema": "atlas_missed_entry_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "symbol": str(st.get("symbol", "XAUUSD") or "XAUUSD"),
        "timeframe": str(st.get("timeframe", "") or ""),
        "action": action,
        "message": str(st.get("message", "") or ""),
        "reference": {
            "price": _safe_float(market.get("last_close", st.get("last_close", 0.0))),
            "status_updated_at": str(st.get("updated_at", "") or ""),
        },
        "primary_blocker": primary_blocker,
        "blocker_class": blocker_class,
        "chosen": {
            "strategy": str(chosen.get("strategy", "") or ""),
            "side": chosen_side,
            "confidence": round(chosen_conf, 4),
        },
        "best_actionable_candidate": {
            "strategy": str(best.get("strategy", "") or ""),
            "side": best_side,
            "confidence": round(best_conf, 4),
            "score": round(best_score, 4),
            "advisory_setup_quality_score": round(advisory_score, 4),
            "reasons": list(best.get("reasons", []) or [])[:6],
        },
        "decision_metrics": {
            "confluence_score": round(confluence_score, 4),
            "confluence_threshold": round(confluence_threshold, 4),
            "execution_request_eligible": bool(
                dict(analysis.get("decision_protocol", {}) or {}).get("act", {}).get(
                    "execution_request_eligible",
                    False,
                )
            ),
            "reason_codes": list(decision.get("reason_codes", []) or [])[:8],
        },
        "market_context": {
            "regime": str(context.get("regime", "") or ""),
            "trend_state": str(context.get("trend_state", "") or ""),
            "trend_structure": str(context.get("trend_structure", "") or ""),
            "momentum": str(context.get("momentum", "") or ""),
            "location": str(context.get("location", "") or ""),
            "session": str(context.get("session", "") or ""),
            "spread_points": _safe_float(context.get("spread_points", 0.0)),
        },
        "collection_context": _collection_context(
            status=st,
            context=context,
            side=audit_side,
            primary_blocker=primary_blocker,
        ),
        "signal_visibility": {
            "raw_count": int(signal_summary.get("raw_count", 0) or 0),
            "active_count": int(signal_summary.get("active_count", 0) or 0),
            "evolved_active_count": int(signal_summary.get("evolved_active_count", 0) or 0),
            "demo_chromosome_active_count": int(learning_status.get("demo_chromosome_active_count", 0) or 0),
            "demo_chromosome_active_ids": list(learning_status.get("demo_chromosome_active_ids", []) or [])[:5],
        },
        "learning_focus": _learning_focus(
            chosen_side=chosen_side,
            best_side=best_side,
            blocker_text=f"{primary_blocker} {st.get('message', '')}",
            blocker_class=blocker_class,
            context=context,
        ),
        "policy": {
            "audit_only": True,
            "does_not_execute": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
        },
    }
    return record


def record_missed_entry_audit(
    storage_dir: str | Path,
    status: dict[str, Any],
    *,
    dedupe_window_seconds: float = 600.0,
) -> dict[str, Any]:
    record = build_missed_entry_audit(status)
    if record is None:
        return {"ok": False, "reason": "not_a_missed_entry_candidate", "recorded": False}
    base = Path(storage_dir)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "missed_entry_audits.jsonl"
    duplicate = _recent_duplicate(record, recent_missed_entry_audits(base, limit=20), window_seconds=dedupe_window_seconds)
    if duplicate:
        return {
            "ok": True,
            "reason": "duplicate_recent_missed_entry_audit",
            "recorded": False,
            "deduped": True,
            "audit": record,
        }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {"ok": True, "recorded": True, "path": str(path), "audit": record}


def recent_missed_entry_audits(storage_dir: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    path = Path(storage_dir) / "missed_entry_audits.jsonl"
    return read_jsonl_tail(path, limit)


def review_missed_entry_outcomes(
    storage_dir: str | Path,
    status: dict[str, Any],
    *,
    min_move_points: float = 80.0,
) -> dict[str, Any]:
    """Review recent missed-entry audits against current price without changing policy."""
    base = Path(storage_dir)
    audits = recent_missed_entry_audits(base, limit=50)
    if not audits:
        return {"ok": False, "reason": "no_missed_entry_audits", "reviewed": False}
    current_market = dict(dict(status or {}).get("market", {}) or {})
    current_price = _safe_float(current_market.get("last_close", dict(status or {}).get("last_close", 0.0)))
    if current_price <= 0:
        return {"ok": False, "reason": "current_price_unavailable", "reviewed": False}

    reviewed: list[dict[str, Any]] = []
    for audit in audits[-10:]:
        side = _safe_side(dict(audit.get("best_actionable_candidate", {}) or {}))
        if side == "flat":
            side = _safe_side(dict(audit.get("chosen", {}) or {}))
        reference = dict(audit.get("reference", {}) or {})
        ref_price = _safe_float(reference.get("price", 0.0))
        if ref_price <= 0:
            outcome = "reference_price_unavailable"
            move_points = 0.0
        else:
            move_points = (current_price - ref_price) * 100.0
            directional_move = move_points if side == "buy" else -move_points if side == "sell" else 0.0
            if side == "flat":
                outcome = "direction_unavailable"
            elif directional_move >= float(min_move_points):
                outcome = "missed_opportunity"
            elif directional_move <= -float(min_move_points):
                outcome = "correct_hold_or_block"
            else:
                outcome = "inconclusive"
        reviewed.append({
            "audit_created_at": str(audit.get("created_at", "") or ""),
            "side": side,
            "reference_price": ref_price,
            "current_price": current_price,
            "move_points": round(move_points, 2),
            "outcome": outcome,
            "primary_blocker": str(audit.get("primary_blocker", "") or ""),
            "policy": {"review_only": True, "does_not_execute": True, "does_not_change_policy": True},
        })

    result = {
        "schema": "atlas_missed_entry_outcome_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "reviewed": True,
        "current_price": current_price,
        "min_move_points": float(min_move_points),
        "reviews": reviewed,
        "summary": {
            "missed_opportunity": sum(1 for row in reviewed if row["outcome"] == "missed_opportunity"),
            "correct_hold_or_block": sum(1 for row in reviewed if row["outcome"] == "correct_hold_or_block"),
            "inconclusive": sum(1 for row in reviewed if row["outcome"] == "inconclusive"),
            "direction_unavailable": sum(1 for row in reviewed if row["outcome"] == "direction_unavailable"),
        },
    }
    path = base / "missed_entry_outcome_reviews.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    return {**result, "path": str(path)}


def summarize_missed_entry_learning(storage_dir: str | Path, *, limit: int = 120) -> dict[str, Any]:
    """Summarize whether blockers are preventing bad trades or missing good ones."""
    base = Path(storage_dir)
    reviews_path = base / "missed_entry_outcome_reviews.jsonl"
    if not reviews_path.exists():
        return {
            "schema": "atlas_missed_entry_learning_summary_v1",
            "available": False,
            "reason": "no_outcome_reviews",
            "policy": _summary_policy(),
        }
    rows: list[dict[str, Any]] = []
    for review in read_jsonl_tail(reviews_path, limit):
        rows.extend(dict(item) for item in list(review.get("reviews", []) or []) if isinstance(item, dict))

    by_blocker: dict[str, dict[str, Any]] = {}
    for row in rows:
        outcome = str(row.get("outcome", "") or "")
        if outcome in {"reference_price_unavailable", "direction_unavailable"}:
            continue
        blocker = str(row.get("primary_blocker", "") or "unknown").strip() or "unknown"
        bucket = by_blocker.setdefault(
            blocker,
            {
                "primary_blocker": blocker,
                "samples": 0,
                "missed_opportunity": 0,
                "correct_hold_or_block": 0,
                "inconclusive": 0,
                "total_favorable_missed_points": 0.0,
                "total_adverse_blocked_points": 0.0,
            },
        )
        bucket["samples"] += 1
        if outcome == "missed_opportunity":
            bucket["missed_opportunity"] += 1
            bucket["total_favorable_missed_points"] += abs(_safe_float(row.get("move_points", 0.0)))
        elif outcome == "correct_hold_or_block":
            bucket["correct_hold_or_block"] += 1
            bucket["total_adverse_blocked_points"] += abs(_safe_float(row.get("move_points", 0.0)))
        else:
            bucket["inconclusive"] += 1

    blocker_rows = []
    for row in by_blocker.values():
        samples = max(1, int(row["samples"]))
        missed = int(row["missed_opportunity"])
        correct = int(row["correct_hold_or_block"])
        row["missed_rate"] = round(missed / samples, 4)
        row["correct_block_rate"] = round(correct / samples, 4)
        if missed >= 2 and missed > correct:
            row["repair_priority"] = "high"
            row["suggested_action"] = "review_blocker_selectivity"
        elif missed == 1 and missed >= correct:
            row["repair_priority"] = "medium"
            row["suggested_action"] = "collect_more_confirmations_before_tuning"
        elif correct > missed:
            row["repair_priority"] = "low"
            row["suggested_action"] = "keep_guard_and_monitor"
        else:
            row["repair_priority"] = "watch"
            row["suggested_action"] = "continue_forward_validation"
        blocker_rows.append(row)
    blocker_rows.sort(
        key=lambda row: (
            {"high": 3, "medium": 2, "watch": 1, "low": 0}.get(str(row.get("repair_priority")), 0),
            int(row.get("missed_opportunity", 0)),
            int(row.get("samples", 0)),
        ),
        reverse=True,
    )
    total_missed = sum(int(row.get("missed_opportunity", 0)) for row in blocker_rows)
    total_correct = sum(int(row.get("correct_hold_or_block", 0)) for row in blocker_rows)
    return {
        "schema": "atlas_missed_entry_learning_summary_v1",
        "available": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": sum(int(row.get("samples", 0)) for row in blocker_rows),
        "missed_opportunity": total_missed,
        "correct_hold_or_block": total_correct,
        "top_blockers": blocker_rows[:8],
        "top_repair_candidate": blocker_rows[0] if blocker_rows else {},
        "policy": _summary_policy(),
    }


def build_blocker_selectivity_review(
    storage_dir: str | Path,
    *,
    primary_blocker: str | None = None,
    limit: int = 240,
) -> dict[str, Any]:
    """Build an offline review plan for a blocker that may be too selective."""
    base = Path(storage_dir)
    summary = summarize_missed_entry_learning(base, limit=limit)
    target = str(primary_blocker or "").strip()
    if not target:
        target = str(dict(summary.get("top_repair_candidate", {}) or {}).get("primary_blocker", "") or "").strip()
    if not target:
        return {
            "schema": "atlas_blocker_selectivity_review_v1",
            "available": False,
            "reason": "no_repair_candidate",
            "policy": _selectivity_review_policy(),
        }

    reviews_path = base / "missed_entry_outcome_reviews.jsonl"
    rows: list[dict[str, Any]] = []
    if reviews_path.exists():
        for review in read_jsonl_tail(reviews_path, limit):
            rows.extend(
                dict(item)
                for item in list(review.get("reviews", []) or [])
                if isinstance(item, dict) and str(item.get("primary_blocker", "") or "").strip() == target
            )

    usable = [
        row
        for row in rows
        if str(row.get("outcome", "") or "") not in {"reference_price_unavailable", "direction_unavailable"}
    ]
    if not usable:
        return {
            "schema": "atlas_blocker_selectivity_review_v1",
            "available": False,
            "reason": "no_reviewed_rows_for_target_blocker",
            "target_blocker": target,
            "policy": _selectivity_review_policy(),
        }

    outcome_counts = {
        "missed_opportunity": sum(1 for row in usable if str(row.get("outcome", "")) == "missed_opportunity"),
        "correct_hold_or_block": sum(1 for row in usable if str(row.get("outcome", "")) == "correct_hold_or_block"),
        "inconclusive": sum(1 for row in usable if str(row.get("outcome", "")) == "inconclusive"),
    }
    samples = len(usable)
    missed_rate = outcome_counts["missed_opportunity"] / max(1, samples)
    correct_rate = outcome_counts["correct_hold_or_block"] / max(1, samples)
    favorable_points = sum(abs(_safe_float(row.get("move_points", 0.0))) for row in usable if str(row.get("outcome", "")) == "missed_opportunity")
    adverse_points = sum(abs(_safe_float(row.get("move_points", 0.0))) for row in usable if str(row.get("outcome", "")) == "correct_hold_or_block")
    sides = sorted({str(row.get("side", "unknown") or "unknown") for row in usable})

    if outcome_counts["missed_opportunity"] >= 2 and outcome_counts["missed_opportunity"] > outcome_counts["correct_hold_or_block"]:
        review_priority = "high"
        proposed_focus = "inspect whether the blocker needs more context-aware confirmation, not a lower global threshold"
    elif outcome_counts["missed_opportunity"] >= outcome_counts["correct_hold_or_block"]:
        review_priority = "medium"
        proposed_focus = "collect more confirmations and replay borderline cases before proposing any tuning"
    else:
        review_priority = "low"
        proposed_focus = "keep the blocker unchanged and continue monitoring"

    result = {
        "schema": "atlas_blocker_selectivity_review_v1",
        "available": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_blocker": target,
        "samples": samples,
        "sides": sides,
        "outcome_counts": outcome_counts,
        "missed_rate": round(missed_rate, 4),
        "correct_block_rate": round(correct_rate, 4),
        "total_favorable_missed_points": round(favorable_points, 2),
        "total_adverse_blocked_points": round(adverse_points, 2),
        "review_priority": review_priority,
        "proposal": {
            "type": "offline_selectivity_review",
            "focus": proposed_focus,
            "candidate_change_class": "context_filter_or_confirmation_rule",
            "forbidden_change_class": "global_guard_disable_or_unapproved_threshold_loosen",
            "next_validation": "replay reviewed cases against a shadow-only candidate rule before any operator approval step",
        },
        "policy": _selectivity_review_policy(),
    }
    path = base / "blocker_selectivity_review.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "path": str(path)}


def build_blocker_selectivity_replay(
    storage_dir: str | Path,
    *,
    primary_blocker: str | None = None,
    limit: int = 240,
) -> dict[str, Any]:
    """Replay a blocker repair candidate as a shadow-only review lane."""
    base = Path(storage_dir)
    review = build_blocker_selectivity_review(base, primary_blocker=primary_blocker, limit=limit)
    if not bool(review.get("available")):
        return {
            "schema": "atlas_blocker_selectivity_replay_v1",
            "available": False,
            "reason": str(review.get("reason", "selectivity_review_unavailable") or "selectivity_review_unavailable"),
            "policy": _selectivity_replay_policy(),
        }

    counts = dict(review.get("outcome_counts", {}) or {})
    missed = int(counts.get("missed_opportunity", 0) or 0)
    correct = int(counts.get("correct_hold_or_block", 0) or 0)
    inconclusive = int(counts.get("inconclusive", 0) or 0)
    samples = max(1, int(review.get("samples", 0) or 0))
    opportunity_capture_rate = missed / samples
    false_unlock_risk_rate = correct / samples
    unknown_rate = inconclusive / samples
    net_review_edge = opportunity_capture_rate - false_unlock_risk_rate
    if missed >= 2 and missed > correct and net_review_edge >= 0.1:
        verdict = "shadow_review_candidate"
        next_step = "create a proposal-only context confirmation rule and replay it against richer stored setup context"
    elif missed >= correct:
        verdict = "collect_more_context"
        next_step = "keep collecting reviewed blocker cases before proposing any rule"
    else:
        verdict = "keep_blocker_unchanged"
        next_step = "continue monitoring; reviewed evidence does not justify even a shadow repair lane yet"

    result = {
        "schema": "atlas_blocker_selectivity_replay_v1",
        "available": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_blocker": str(review.get("target_blocker", "") or ""),
        "mode": "shadow_review_only",
        "candidate_rule": {
            "name": "context_aware_consensus_selectivity_review",
            "action": "flag_blocked_case_for_shadow_context_review",
            "does_not_allow_trade": True,
            "does_not_lower_min_same_side_signals": True,
            "requires_context_confirmation_fields": [
                "higher_timeframe_alignment",
                "directional_structure",
                "momentum_quality",
                "spread_state",
                "loss_pattern_state",
            ],
        },
        "samples": samples,
        "metrics": {
            "missed_opportunity_capture_count": missed,
            "correct_block_false_unlock_risk_count": correct,
            "inconclusive_count": inconclusive,
            "opportunity_capture_rate": round(opportunity_capture_rate, 4),
            "false_unlock_risk_rate": round(false_unlock_risk_rate, 4),
            "unknown_rate": round(unknown_rate, 4),
            "net_review_edge": round(net_review_edge, 4),
        },
        "verdict": verdict,
        "next_step": next_step,
        "source_review": {
            "review_priority": str(review.get("review_priority", "") or ""),
            "proposal": dict(review.get("proposal", {}) or {}),
        },
        "policy": _selectivity_replay_policy(),
    }
    path = base / "blocker_selectivity_replay.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "path": str(path)}


def sync_blocker_selectivity_proposal(store: Any, *, primary_blocker: str | None = None) -> dict[str, Any]:
    """Create a draft self-improvement proposal from the shadow replay result."""
    base = Path(getattr(store, "base", store))
    replay = build_blocker_selectivity_replay(base, primary_blocker=primary_blocker)
    if not bool(replay.get("available")):
        return {"ok": False, "reason": str(replay.get("reason", "replay_unavailable") or "replay_unavailable"), "created_proposals": []}
    if str(replay.get("verdict", "") or "") != "shadow_review_candidate":
        return {
            "ok": False,
            "reason": "replay_verdict_not_proposal_ready",
            "verdict": str(replay.get("verdict", "") or ""),
            "created_proposals": [],
        }

    target_blocker = str(replay.get("target_blocker", "") or "").strip()
    metrics = dict(replay.get("metrics", {}) or {})
    proposal_id = "proposal_consensus_selectivity_context_review"
    bundle = load_self_improvement_bundle(store)
    existing = next(
        (
            dict(row)
            for row in list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
            if str(dict(row).get("proposal_id", "") or "") == proposal_id
        ),
        {},
    )
    proposal = append_or_update_proposal(
        store,
        {
            "proposal_id": proposal_id,
            "status": str(existing.get("status", "draft") or "draft"),
            "category": "filters",
            "scope": {
                "strategies": ["consensus_guard"],
                "instruments": ["XAUUSD"],
                "sessions": [],
                "target_blocker": target_blocker,
            },
            "source": {
                "generator": "missed_entry_blocker_selectivity_replay",
                "artifact": "storage/blocker_selectivity_replay.json",
                "linked_review_artifact": "storage/blocker_selectivity_review.json",
            },
            "current_behavior": {
                "summary": "Consensus guard requires same-side support before a candidate can proceed.",
                "observed_blocker": target_blocker,
            },
            "proposed_change": {
                "type": "context_confirmation_candidate",
                "target": "consensus_guard_selectivity_review",
                "suggested_direction": "add shadow-only context confirmation review lane",
                "candidate_rule_name": dict(replay.get("candidate_rule", {}) or {}).get("name", "context_aware_consensus_selectivity_review"),
                "does_not_lower_min_same_side_signals": True,
                "does_not_allow_trade": True,
            },
            "reason": str(replay.get("next_step", "") or "Shadow replay indicates blocker selectivity should be reviewed."),
            "evidence": {
                "strength": min(0.95, max(0.0, 0.55 + float(metrics.get("net_review_edge", 0.0) or 0.0))),
                "grade": "proposal_candidate",
                "sample_count": int(replay.get("samples", 0) or 0),
                "metrics": metrics,
                "verdict": str(replay.get("verdict", "") or ""),
            },
            "validation": {
                "required": ["richer_context_replay", "shadow_compare", "regression_tests", "operator_review"],
                "completed": list(dict(existing.get("validation", {}) or {}).get("completed", []) or []),
            },
            "approval": {
                "required": True,
                "approved": bool(dict(existing.get("approval", {}) or {}).get("approved", False)),
                "approved_by": str(dict(existing.get("approval", {}) or {}).get("approved_by", "") or ""),
                "approved_at": str(dict(existing.get("approval", {}) or {}).get("approved_at", "") or ""),
            },
            "rollout": {
                "required": True,
                "rollout_id": str(dict(existing.get("rollout", {}) or {}).get("rollout_id", "") or ""),
                "promotion_ready": bool(dict(existing.get("rollout", {}) or {}).get("promotion_ready", False)),
            },
            "rollback_plan": {
                "restore_snapshot_id": str(dict(existing.get("rollback_plan", {}) or {}).get("restore_snapshot_id", "") or ""),
            },
            "policy": {
                "proposal_only": True,
                "does_not_execute": True,
                "does_not_change_strategy_logic": True,
                "does_not_change_risk_policy": True,
                "does_not_change_execution_policy": True,
                "requires_operator_approval": True,
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "proposal_created" if not existing else "proposal_updated",
                "actor": "atlas",
                "proposal_id": proposal_id,
                "reason": "blocker_selectivity_replay_promoted_to_draft_proposal",
                "meta": {"target_blocker": target_blocker, "metrics": metrics},
            }
        )
    return {"ok": True, "created_proposals": [proposal_id], "proposal": proposal, "replay": replay}


def validate_blocker_selectivity_richer_context_replay(
    store: Any,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
) -> dict[str, Any]:
    """Validate only the richer-context replay step for the selectivity proposal."""
    bundle = load_self_improvement_bundle(store)
    proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    proposal = next((dict(row) for row in proposals if str(dict(row).get("proposal_id", "") or "") == proposal_id), None)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": proposal_id}

    scope = dict(proposal.get("scope", {}) or {})
    replay = build_blocker_selectivity_replay(
        Path(getattr(store, "base", "storage")),
        primary_blocker=str(scope.get("target_blocker", "") or "") or None,
    )
    metrics = dict(replay.get("metrics", {}) or {})
    required_context = list(dict(replay.get("candidate_rule", {}) or {}).get("requires_context_confirmation_fields", []) or [])
    replay_passed = (
        bool(replay.get("available"))
        and str(replay.get("verdict", "") or "") == "shadow_review_candidate"
        and int(replay.get("samples", 0) or 0) >= 40
        and float(metrics.get("net_review_edge", 0.0) or 0.0) > 0.0
    )

    validation = dict(proposal.get("validation", {}) or {})
    required = list(validation.get("required", []) or [])
    completed = list(validation.get("completed", []) or [])
    if replay_passed and "richer_context_replay" not in completed:
        completed.append("richer_context_replay")
    missing = [item for item in required if item not in completed]

    report = {
        "proposal_id": proposal_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": str(proposal.get("category", "") or ""),
        "status": str(proposal.get("status", "") or ""),
        "validation_step": "richer_context_replay",
        "ok": replay_passed,
        "evidence": {
            "sample_count": int(replay.get("samples", 0) or 0),
            "target_blocker": str(replay.get("target_blocker", "") or ""),
            "metrics": metrics,
            "required_context_fields": required_context,
        },
        "validation": {
            "required": required,
            "completed": completed,
            "missing": missing,
        },
        "promotion_readiness": {
            "operator_approval_required": True,
            "ready_for_rollout_review": False,
            "reason": "richer_context_replay_only; shadow_compare, regression_tests, and operator_review still required",
        },
        "policy": {
            "does_not_execute": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
            "does_not_lower_thresholds": True,
        },
    }
    save_validation_report(store, proposal_id, report)
    updated = append_or_update_proposal(
        store,
        {
            **proposal,
            "status": str(proposal.get("status", "draft") or "draft"),
            "validation": {
                "required": required,
                "completed": completed,
                "missing": missing,
                "last_report_generated_at": str(report.get("generated_at", "") or ""),
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "proposal_validation_step_completed" if replay_passed else "proposal_validation_step_failed",
                "actor": "atlas",
                "proposal_id": proposal_id,
                "reason": "richer_context_replay_validation",
                "meta": {
                    "replay_passed": replay_passed,
                    "missing": missing,
                    "metrics": metrics,
                },
            }
        )
    return {"ok": replay_passed, "proposal": updated, "report": report, "missing": missing}


def validate_blocker_selectivity_shadow_compare(
    store: Any,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
) -> dict[str, Any]:
    """Validate the proposal's shadow comparison step without changing execution."""
    bundle = load_self_improvement_bundle(store)
    proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    proposal = next((dict(row) for row in proposals if str(dict(row).get("proposal_id", "") or "") == proposal_id), None)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": proposal_id}

    scope = dict(proposal.get("scope", {}) or {})
    replay = build_blocker_selectivity_replay(
        Path(getattr(store, "base", "storage")),
        primary_blocker=str(scope.get("target_blocker", "") or "") or None,
    )
    metrics = dict(replay.get("metrics", {}) or {})
    capture_rate = _safe_float(metrics.get("opportunity_capture_rate", 0.0))
    false_unlock_rate = _safe_float(metrics.get("false_unlock_risk_rate", 0.0))
    unknown_rate = _safe_float(metrics.get("unknown_rate", 1.0), 1.0)
    net_edge = _safe_float(metrics.get("net_review_edge", 0.0))
    samples = int(replay.get("samples", 0) or 0)
    compare_passed = (
        bool(replay.get("available"))
        and samples >= 40
        and capture_rate > false_unlock_rate
        and net_edge >= 0.1
        and unknown_rate <= 0.6
    )

    validation = dict(proposal.get("validation", {}) or {})
    required = list(validation.get("required", []) or [])
    completed = list(validation.get("completed", []) or [])
    if compare_passed and "shadow_compare" not in completed:
        completed.append("shadow_compare")
    missing = [item for item in required if item not in completed]

    report = {
        "proposal_id": proposal_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": str(proposal.get("category", "") or ""),
        "status": str(proposal.get("status", "") or ""),
        "validation_step": "shadow_compare",
        "ok": compare_passed,
        "comparison": {
            "control_behavior": "keep consensus guard behavior unchanged",
            "draft_behavior": "shadow-flag context-confirmed consensus blocks for review only",
            "capture_rate": round(capture_rate, 4),
            "false_unlock_risk_rate": round(false_unlock_rate, 4),
            "unknown_rate": round(unknown_rate, 4),
            "net_review_edge": round(net_edge, 4),
            "samples": samples,
            "decision": "pass" if compare_passed else "continue_collecting",
        },
        "validation": {
            "required": required,
            "completed": completed,
            "missing": missing,
        },
        "promotion_readiness": {
            "operator_approval_required": True,
            "ready_for_rollout_review": False,
            "reason": "shadow_compare_only; regression_tests and operator_review still required",
        },
        "policy": {
            "shadow_compare_only": True,
            "does_not_execute": True,
            "does_not_allow_trade": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
            "does_not_lower_thresholds": True,
        },
    }
    save_validation_report(store, proposal_id, report)
    updated = append_or_update_proposal(
        store,
        {
            **proposal,
            "status": str(proposal.get("status", "draft") or "draft"),
            "validation": {
                "required": required,
                "completed": completed,
                "missing": missing,
                "last_report_generated_at": str(report.get("generated_at", "") or ""),
            },
        },
    )
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "proposal_validation_step_completed" if compare_passed else "proposal_validation_step_failed",
                "actor": "atlas",
                "proposal_id": proposal_id,
                "reason": "shadow_compare_validation",
                "meta": {
                    "compare_passed": compare_passed,
                    "missing": missing,
                    "metrics": metrics,
                },
            }
        )
    return {"ok": compare_passed, "proposal": updated, "report": report, "missing": missing}


def build_blocker_selectivity_collection_plan(
    store: Any,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
) -> dict[str, Any]:
    """Plan what evidence is needed before retrying failed shadow comparison."""
    base = Path(getattr(store, "base", store))
    report_path = base / "proposal_reports" / f"{proposal_id}.json"
    try:
        report = dict(json.loads(report_path.read_text(encoding="utf-8")) or {}) if report_path.exists() else {}
    except Exception:
        report = {}
    comparison = dict(report.get("comparison", {}) or {})
    replay = build_blocker_selectivity_replay(base)
    candidate_rule = dict(replay.get("candidate_rule", {}) or {})
    capture_rate = _safe_float(comparison.get("capture_rate", dict(replay.get("metrics", {}) or {}).get("opportunity_capture_rate", 0.0)))
    false_unlock_rate = _safe_float(comparison.get("false_unlock_risk_rate", dict(replay.get("metrics", {}) or {}).get("false_unlock_risk_rate", 0.0)))
    unknown_rate = _safe_float(comparison.get("unknown_rate", dict(replay.get("metrics", {}) or {}).get("unknown_rate", 1.0)), 1.0)
    net_edge = _safe_float(comparison.get("net_review_edge", dict(replay.get("metrics", {}) or {}).get("net_review_edge", 0.0)))
    blockers: list[str] = []
    if capture_rate <= false_unlock_rate:
        blockers.append("false_unlock_risk_exceeds_capture")
    if net_edge < 0.1:
        blockers.append("net_review_edge_below_retry_threshold")
    if unknown_rate > 0.45:
        blockers.append("too_many_inconclusive_reviews")
    if not blockers:
        blockers.append("shadow_compare_not_ready_for_approval")

    result = {
        "schema": "atlas_blocker_selectivity_collection_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_id": proposal_id,
        "source_validation_step": str(report.get("validation_step", "shadow_compare") or "shadow_compare"),
        "source_decision": str(comparison.get("decision", "continue_collecting") or "continue_collecting"),
        "failure_reasons": blockers,
        "current_metrics": {
            "capture_rate": round(capture_rate, 4),
            "false_unlock_risk_rate": round(false_unlock_rate, 4),
            "unknown_rate": round(unknown_rate, 4),
            "net_review_edge": round(net_edge, 4),
            "samples": int(comparison.get("samples", dict(replay.get("metrics", {}) or {}).get("samples", 0)) or 0),
        },
        "retry_requirements": {
            "min_samples": 120,
            "max_unknown_rate": 0.45,
            "min_net_review_edge": 0.1,
            "capture_must_exceed_false_unlock": True,
        },
        "capture_fields": list(candidate_rule.get("requires_context_confirmation_fields", []) or [
            "higher_timeframe_alignment",
            "directional_structure",
            "momentum_quality",
            "spread_state",
            "loss_pattern_state",
        ]),
        "collection_focus": [
            "capture blocker-side context at the moment consensus blocks",
            "label whether later movement favored the blocked side",
            "separate correct protective blocks from missed opportunity blocks",
            "keep draft proposal parked until shadow_compare passes",
        ],
        "policy": {
            "collection_only": True,
            "does_not_execute": True,
            "does_not_allow_trade": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
            "does_not_lower_thresholds": True,
        },
    }
    path = base / "blocker_selectivity_collection_plan.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "path": str(path)}


def build_blocker_selectivity_collection_readiness(
    storage_dir: str | Path,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
    limit: int = 240,
) -> dict[str, Any]:
    """Report whether enough enriched audit rows exist to retry shadow_compare."""
    base = Path(storage_dir)
    plan_path = base / "blocker_selectivity_collection_plan.json"
    try:
        plan = dict(json.loads(plan_path.read_text(encoding="utf-8")) or {}) if plan_path.exists() else {}
    except Exception:
        plan = {}
    required_fields = list(plan.get("capture_fields", []) or [
        "higher_timeframe_alignment",
        "directional_structure",
        "momentum_quality",
        "spread_state",
        "loss_pattern_state",
    ])
    retry = dict(plan.get("retry_requirements", {}) or {})
    min_samples = int(retry.get("min_samples", 120) or 120)
    target_blocker = _proposal_target_blocker(base, proposal_id)

    audits = recent_missed_entry_audits(base, limit=limit)
    if target_blocker:
        audits = [row for row in audits if str(row.get("primary_blocker", "") or "").strip() == target_blocker]
    enriched = []
    missing_rows = []
    for audit in audits:
        ctx = dict(audit.get("collection_context", {}) or {})
        missing = []
        for field in required_fields:
            value = ctx.get(field)
            if value is None or str(value).strip().lower() in {"", "unknown"}:
                missing.append(field)
        if missing:
            missing_rows.append({"fingerprint": str(audit.get("fingerprint", "") or ""), "missing": missing})
        else:
            enriched.append(audit)

    ready = len(enriched) >= min_samples
    result = {
        "schema": "atlas_blocker_selectivity_collection_readiness_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ready_to_retry_shadow_compare": ready,
        "required_fields": required_fields,
        "target_blocker": target_blocker,
        "required_samples": min_samples,
        "total_recent_audits": len(audits),
        "enriched_samples": len(enriched),
        "missing_context_samples": len(missing_rows),
        "remaining_enriched_samples_needed": max(0, min_samples - len(enriched)),
        "top_missing_fields": _top_missing_fields(missing_rows),
        "policy": {
            "readiness_only": True,
            "does_not_execute": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
        },
    }
    path = base / "blocker_selectivity_collection_readiness.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "path": str(path)}


def build_blocker_selectivity_offline_collector(
    storage_dir: str | Path,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
    min_samples: int | None = None,
) -> dict[str, Any]:
    """Mine stored Atlas audit/review artifacts for target-blocker evidence.

    This is intentionally collection-only: it never creates synthetic samples and
    never changes the live blocker, strategy, risk, or execution path.
    """
    base = Path(storage_dir)
    backfill = backfill_missed_entry_collection_context(base, limit=1_000_000)
    plan_path = base / "blocker_selectivity_collection_plan.json"
    try:
        plan = dict(json.loads(plan_path.read_text(encoding="utf-8")) or {}) if plan_path.exists() else {}
    except Exception:
        plan = {}
    required_fields = list(plan.get("capture_fields", []) or [
        "higher_timeframe_alignment",
        "directional_structure",
        "momentum_quality",
        "spread_state",
        "loss_pattern_state",
    ])
    retry = dict(plan.get("retry_requirements", {}) or {})
    required_samples = int(min_samples or retry.get("min_samples", 120) or 120)
    target_blocker = _proposal_target_blocker(base, proposal_id)

    audits = _read_jsonl_dicts(base / "missed_entry_audits.jsonl")
    if target_blocker:
        target_audits = [row for row in audits if str(row.get("primary_blocker", "") or "").strip() == target_blocker]
    else:
        target_audits = list(audits)

    enriched = []
    missing_rows = []
    for audit in target_audits:
        ctx = dict(audit.get("collection_context", {}) or {})
        missing = [
            field
            for field in required_fields
            if ctx.get(field) is None or str(ctx.get(field, "")).strip().lower() in {"", "unknown"}
        ]
        if missing:
            missing_rows.append({"fingerprint": str(audit.get("fingerprint", "") or ""), "missing": missing})
        else:
            enriched.append(audit)

    # These are all-history totals, not a recent window. Keep one batch in memory.
    outcome_counts = {"missed_opportunity": 0, "correct_hold_or_block": 0, "inconclusive": 0}
    reviewed_samples = 0
    for batch in _iter_jsonl_dicts(base / "missed_entry_outcome_reviews.jsonl"):
        for item in batch.get("reviews", []) or []:
            if not isinstance(item, dict):
                continue
            if target_blocker and str(item.get("primary_blocker", "") or "").strip() != target_blocker:
                continue
            if str(item.get("outcome", "") or "") in {"reference_price_unavailable", "direction_unavailable"}:
                continue
            reviewed_samples += 1
            outcome = str(item.get("outcome", ""))
            if outcome in outcome_counts:
                outcome_counts[outcome] += 1
    missed_rate = outcome_counts["missed_opportunity"] / max(1, reviewed_samples)
    correct_rate = outcome_counts["correct_hold_or_block"] / max(1, reviewed_samples)
    unknown_rate = outcome_counts["inconclusive"] / max(1, reviewed_samples)
    net_review_edge = missed_rate - correct_rate
    ready = len(enriched) >= required_samples

    result = {
        "schema": "atlas_blocker_selectivity_offline_collector_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_id": proposal_id,
        "target_blocker": target_blocker,
        "mode": "offline_collection_only",
        "source_artifacts": [
            "storage/missed_entry_audits.jsonl",
            "storage/missed_entry_outcome_reviews.jsonl",
        ],
        "required_fields": required_fields,
        "required_samples": required_samples,
        "all_time_audit_samples": len(target_audits),
        "enriched_samples": len(enriched),
        "missing_context_samples": len(missing_rows),
        "remaining_enriched_samples_needed": max(0, required_samples - len(enriched)),
        "reviewed_outcome_samples": reviewed_samples,
        "outcome_counts": outcome_counts,
        "metrics": {
            "missed_rate": round(missed_rate, 4),
            "correct_block_rate": round(correct_rate, 4),
            "unknown_rate": round(unknown_rate, 4),
            "net_review_edge": round(net_review_edge, 4),
        },
        "ready_to_retry_shadow_compare": ready,
        "top_missing_fields": _top_missing_fields(missing_rows),
        "backfill": {
            "ok": bool(backfill.get("ok", False)),
            "scanned": int(backfill.get("scanned", 0) or 0),
            "updated": int(backfill.get("updated", 0) or 0),
        },
        "policy": {
            "offline_collection_only": True,
            "synthetic_samples_created": False,
            "does_not_execute": True,
            "does_not_allow_trade": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
            "does_not_lower_thresholds": True,
        },
    }
    path = base / "blocker_selectivity_offline_collector.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {**result, "path": str(path)}


def backfill_missed_entry_audits_from_events(
    storage_dir: str | Path,
    *,
    include_archives: bool = False,
    limit_files: int = 40,
) -> dict[str, Any]:
    """Backfill missed-entry audits from real stored decision/risk event metadata."""
    base = Path(storage_dir)
    audit_path = base / "missed_entry_audits.jsonl"
    existing = _read_jsonl_dicts(audit_path)
    existing_keys = {
        (
            str(row.get("fingerprint", "") or ""),
            str(dict(row.get("source_event", {}) or {}).get("time", "") or str(row.get("reference", {}).get("status_updated_at", "") or "")),
        )
        for row in existing
    }
    event_paths = [base / "events.jsonl"]
    if include_archives:
        archive = base / "archive"
        if archive.exists():
            archived = sorted(archive.glob("events_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            event_paths.extend(archived[: max(0, int(limit_files))])

    scanned_events = 0
    candidate_events = 0
    appended = 0
    skipped_duplicate = 0
    skipped_not_candidate = 0
    malformed = 0
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as out:
        for event_path in event_paths:
            if not event_path.exists():
                continue
            try:
                lines = event_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except Exception:
                    malformed += 1
                    continue
                if not isinstance(event, dict):
                    malformed += 1
                    continue
                scanned_events += 1
                event_type = str(event.get("type", "") or "").strip().lower()
                if event_type not in {"decision", "risk"}:
                    continue
                meta = dict(event.get("meta", {}) or {})
                if not meta:
                    continue
                text = f"{event.get('message', '')} {meta.get('message', '')} {meta.get('primary_blocker', '')}"
                if "consensus guard blocked trade" not in text.lower() and str(meta.get("action", "") or "").lower() not in {"hold", "blocked"}:
                    continue
                candidate_events += 1
                status = dict(meta)
                status.setdefault("message", str(event.get("message", "") or ""))
                status.setdefault("action", "blocked" if event_type == "risk" else "hold")
                status.setdefault("updated_at", str(event.get("time", "") or ""))
                record = build_missed_entry_audit(status)
                if record is None:
                    skipped_not_candidate += 1
                    continue
                record["created_at"] = str(event.get("time", "") or record.get("created_at", ""))
                record["source_event"] = {
                    "time": str(event.get("time", "") or ""),
                    "type": event_type,
                    "path": str(event_path),
                    "backfilled": True,
                }
                key = (str(record.get("fingerprint", "") or ""), str(record["source_event"]["time"]))
                if key in existing_keys:
                    skipped_duplicate += 1
                    continue
                existing_keys.add(key)
                out.write(json.dumps(record, sort_keys=True) + "\n")
                appended += 1

    result = {
        "schema": "atlas_missed_entry_event_backfill_v1",
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "include_archives": bool(include_archives),
        "event_files_scanned": sum(1 for path in event_paths if path.exists()),
        "scanned_events": scanned_events,
        "candidate_events": candidate_events,
        "appended": appended,
        "skipped_duplicate": skipped_duplicate,
        "skipped_not_candidate": skipped_not_candidate,
        "malformed": malformed,
        "path": str(audit_path),
        "policy": {
            "event_backfill_only": True,
            "synthetic_samples_created": False,
            "does_not_execute": True,
            "does_not_allow_trade": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
        },
    }
    (base / "missed_entry_event_backfill.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def build_consensus_selectivity_operator_review_packet(
    store: Any,
    *,
    proposal_id: str = "proposal_consensus_selectivity_context_review",
) -> dict[str, Any]:
    """Build the final operator-review packet without approving or activating it."""
    base = Path(getattr(store, "base", "storage"))
    bundle = load_self_improvement_bundle(store)
    proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    proposal = next((dict(row) for row in proposals if str(dict(row).get("proposal_id", "") or "") == proposal_id), None)
    if proposal is None:
        return {"ok": False, "reason": "proposal_not_found", "proposal_id": proposal_id}

    collector = build_blocker_selectivity_offline_collector(base)
    replay = build_blocker_selectivity_replay(
        base,
        primary_blocker=str(dict(proposal.get("scope", {}) or {}).get("target_blocker", "") or "") or None,
    )
    report_path = base / "proposal_reports" / f"{proposal_id}.json"
    try:
        validation_report = dict(json.loads(report_path.read_text(encoding="utf-8")) or {}) if report_path.exists() else {}
    except Exception:
        validation_report = {}
    validation = dict(proposal.get("validation", {}) or {})
    approval = dict(proposal.get("approval", {}) or {})
    required = list(validation.get("required", []) or [])
    completed = list(validation.get("completed", []) or [])
    missing = [item for item in required if item not in completed]
    recommendation = "review_for_operator_approval" if missing == ["operator_review"] else "finish_missing_validation_before_review"
    collector_metrics = dict(collector.get("metrics", {}) or {})
    replay_metrics = dict(replay.get("metrics", {}) or {})
    shadow_compare = dict(validation_report.get("comparison", {}) or {})
    collector_net = float(collector_metrics.get("net_review_edge", 0.0) or 0.0)
    replay_net = float(replay_metrics.get("net_review_edge", 0.0) or 0.0)
    shadow_net = float(shadow_compare.get("net_review_edge", 0.0) or 0.0)
    shadow_decision = str(shadow_compare.get("decision", "") or "").strip().lower()
    regression_result = dict(validation.get("last_regression_test_result", {}) or {})
    regression_ok = bool(regression_result.get("ok", False)) or "regression_tests" in completed
    ready_for_review = missing == ["operator_review"]
    if not ready_for_review or not regression_ok:
        posture = "defer_until_validation_complete"
        posture_reason = "Required validation is incomplete or regression tests have not passed."
        suggested_operator_action = "defer"
    elif shadow_decision == "pass" and min(collector_net, replay_net, shadow_net) >= 0.1:
        posture = "guarded_rollout_candidate"
        posture_reason = "Collector, replay, and shadow-compare evidence all show positive net review edge."
        suggested_operator_action = "consider_guarded_rollout_approval"
    elif shadow_decision == "pass" and shadow_net >= 0.1 and collector_net >= 0.1:
        posture = "guarded_rollout_candidate_with_replay_caution"
        posture_reason = "Shadow compare and collector evidence are positive, but replay evidence still needs caution."
        suggested_operator_action = "review_packet_then_consider_guarded_rollout_approval"
    elif max(collector_net, replay_net, shadow_net) < 0.1:
        posture = "reject_or_defer_candidate"
        posture_reason = "Evidence does not show enough positive review edge for a guarded rollout."
        suggested_operator_action = "reject_or_defer"
    else:
        posture = "defer_for_more_evidence"
        posture_reason = "Evidence is mixed; more review evidence is safer before approval."
        suggested_operator_action = "defer"

    packet = {
        "schema": "atlas_consensus_selectivity_operator_review_packet_v1",
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_id": proposal_id,
        "status": str(proposal.get("status", "") or ""),
        "recommendation": recommendation,
        "operator_decision_required": "approve_guarded_rollout_or_reject",
        "proposal_summary": {
            "category": str(proposal.get("category", "") or ""),
            "target_blocker": str(dict(proposal.get("scope", {}) or {}).get("target_blocker", "") or ""),
            "current_behavior": dict(proposal.get("current_behavior", {}) or {}),
            "proposed_change": dict(proposal.get("proposed_change", {}) or {}),
            "reason": str(proposal.get("reason", "") or ""),
        },
        "evidence": {
            "collector": {
                "enriched_samples": int(collector.get("enriched_samples", 0) or 0),
                "required_samples": int(collector.get("required_samples", 0) or 0),
                "remaining_enriched_samples_needed": int(collector.get("remaining_enriched_samples_needed", 0) or 0),
                "reviewed_outcome_samples": int(collector.get("reviewed_outcome_samples", 0) or 0),
                "outcome_counts": dict(collector.get("outcome_counts", {}) or {}),
                "metrics": dict(collector.get("metrics", {}) or {}),
            },
            "shadow_compare": dict(validation_report.get("comparison", {}) or {}),
            "replay_metrics": dict(replay.get("metrics", {}) or {}),
            "regression_tests": regression_result
            or {
                "ok": regression_ok,
                "source": "proposal_validation_completed_marker",
                "completed_marker_present": "regression_tests" in completed,
            },
        },
        "operator_posture": {
            "posture": posture,
            "reason": posture_reason,
            "suggested_operator_action": suggested_operator_action,
            "approval_still_requires_explicit_operator_command": True,
            "does_not_approve": True,
            "inputs": {
                "collector_net_review_edge": collector_net,
                "replay_net_review_edge": replay_net,
                "shadow_net_review_edge": shadow_net,
                "shadow_decision": shadow_decision or "unknown",
                "regression_ok": regression_ok,
                "ready_for_operator_review": ready_for_review,
            },
        },
        "validation": {
            "required": required,
            "completed": completed,
            "missing": missing,
            "ready_for_operator_review": missing == ["operator_review"],
        },
        "approval": {
            "required": bool(approval.get("required", True)),
            "approved": bool(approval.get("approved", False)),
            "approved_by": str(approval.get("approved_by", "") or ""),
            "approved_at": str(approval.get("approved_at", "") or ""),
        },
        "operator_options": [
            {
                "decision": "approve_guarded_rollout",
                "effect": "records operator approval and creates a guarded demo rollout entry",
                "requires_explicit_operator_command": True,
            },
            {
                "decision": "reject_or_defer",
                "effect": "keeps proposal non-executing and leaves current consensus behavior unchanged",
                "requires_explicit_operator_command": True,
            },
        ],
        "safety": {
            "does_not_execute": True,
            "does_not_approve": True,
            "does_not_activate_rollout": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
            "does_not_lower_thresholds": True,
        },
    }
    path = base / "proposal_reports" / f"{proposal_id}_operator_review_packet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
    if hasattr(store, "append_self_improvement_audit"):
        store.append_self_improvement_audit(
            {
                "event": "operator_review_packet_generated",
                "actor": "atlas",
                "proposal_id": proposal_id,
                "reason": recommendation,
                "meta": {"path": str(path), "missing": missing},
            }
        )
    return {**packet, "path": str(path)}


def backfill_missed_entry_collection_context(storage_dir: str | Path, *, limit: int = 1000) -> dict[str, Any]:
    """Backfill collection_context on existing missed-entry audits where stored context permits it."""
    base = Path(storage_dir)
    path = base / "missed_entry_audits.jsonl"
    if not path.exists():
        return {"ok": False, "reason": "no_missed_entry_audits", "updated": 0}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {"ok": False, "reason": "read_failed", "updated": 0}

    rows: list[dict[str, Any]] = []
    updated = 0
    scanned = 0
    start = max(0, len(lines) - max(1, int(limit)))
    for index, line in enumerate(lines):
        try:
            row = dict(json.loads(line))
        except Exception:
            rows.append({})
            continue
        if index >= start:
            scanned += 1
            existing = dict(row.get("collection_context", {}) or {})
            if not existing or any(str(existing.get(field, "")).strip().lower() in {"", "unknown"} for field in ("directional_structure", "momentum_quality", "spread_state", "loss_pattern_state")):
                candidate = dict(row.get("best_actionable_candidate", {}) or {}) or dict(row.get("chosen", {}) or {})
                side = _safe_side(candidate)
                context = _context_from_audit_row(row)
                status = {
                    "higher_tf_bias": _extract_htf_bias(row),
                    "loss_learning": {"state": _infer_loss_pattern_state(row)},
                }
                merged = _collection_context(status=status, context=context, side=side, primary_blocker=str(row.get("primary_blocker", "") or ""))
                merged["backfilled"] = True
                merged["backfill_source"] = "missed_entry_audit_market_context"
                row["collection_context"] = {**existing, **merged}
                updated += 1
        rows.append(row)

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    result = {
        "schema": "atlas_missed_entry_collection_context_backfill_v1",
        "ok": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scanned": scanned,
        "updated": updated,
        "path": str(path),
        "policy": {
            "audit_backfill_only": True,
            "does_not_execute": True,
            "does_not_change_strategy_logic": True,
            "does_not_change_risk_policy": True,
            "does_not_change_execution_policy": True,
        },
    }
    (base / "missed_entry_collection_context_backfill.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _iter_jsonl_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """Stream a full-history aggregation without retaining the entire journal."""
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(row, dict):
                yield row


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl_dicts(path))


def _learning_focus(
    *,
    chosen_side: str,
    best_side: str,
    blocker_text: str,
    blocker_class: str,
    context: dict[str, Any],
) -> list[str]:
    blocker = blocker_text.lower()
    focus: list[str] = []
    if chosen_side == "flat" and best_side in {"buy", "sell"}:
        focus.append("study_why_actionable_candidate_lost_to_flat_selection")
    if "loss-pattern" in blocker or "loss pattern" in blocker:
        focus.append("validate_loss_pattern_block_against_next_price_move")
    if "no directional" in blocker:
        focus.append("improve_direction_confirmation_before_entry")
    if "confidence" in blocker or blocker_class == "signal_quality":
        focus.append("compare_setup_quality_against_signal_confidence")
    if str(context.get("momentum", "")).lower() in {"mixed", "weak"}:
        focus.append("wait_for_momentum_confirmation_or_mark_as_correct_hold")
    if str(context.get("location", "")).lower() in {"mid_range", "consolidation"}:
        focus.append("separate_good_area_from_executable_entry_timing")
    return list(dict.fromkeys(focus or ["review_missed_entry_context"]))[:6]


def _collection_context(
    *,
    status: dict[str, Any],
    context: dict[str, Any],
    side: str,
    primary_blocker: str,
) -> dict[str, Any]:
    htf = dict(status.get("higher_timeframe", {}) or {})
    htf_bias = str(
        status.get("higher_tf_bias")
        or context.get("higher_tf_bias")
        or context.get("higher_timeframe_bias")
        or htf.get("bias")
        or "unknown"
    ).strip().lower()
    directional_structure = str(
        context.get("trend_structure")
        or context.get("directional_structure")
        or context.get("trend_state")
        or "unknown"
    ).strip().lower()
    momentum_quality = str(
        context.get("momentum_quality")
        or context.get("momentum")
        or context.get("momentum_state")
        or "unknown"
    ).strip().lower()
    spread_state = str(context.get("spread_state") or "unknown").strip().lower()
    loss_state = str(
        context.get("loss_pattern_state")
        or dict(status.get("loss_learning", {}) or {}).get("state")
        or ("active" if "loss" in str(primary_blocker or "").lower() else "unknown")
    ).strip().lower()
    return {
        "schema": "atlas_blocker_collection_context_v1",
        "side": side if side in {"buy", "sell"} else "flat",
        "higher_timeframe_bias": htf_bias,
        "higher_timeframe_alignment": bool(side in {"buy", "sell"} and htf_bias in {"buy", "sell"} and side == htf_bias),
        "directional_structure": directional_structure,
        "directional_structure_aligned": _structure_aligns(side, directional_structure),
        "momentum_quality": momentum_quality,
        "spread_state": spread_state,
        "loss_pattern_state": loss_state,
        "required_for_selectivity_retry": [
            "higher_timeframe_alignment",
            "directional_structure",
            "momentum_quality",
            "spread_state",
            "loss_pattern_state",
        ],
    }


def _context_from_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    market = dict(row.get("market_context", {}) or {})
    spread_points = _safe_float(market.get("spread_points", 0.0))
    spread_state = str(market.get("spread_state", "") or "").strip().lower()
    if not spread_state:
        spread_state = "normal" if spread_points > 0 else "unknown"
    return {
        "trend_structure": str(market.get("trend_structure", "") or ""),
        "trend_state": str(market.get("trend_state", "") or ""),
        "momentum": str(market.get("momentum", "") or ""),
        "spread_state": spread_state,
        "loss_pattern_state": _infer_loss_pattern_state(row),
    }


def _extract_htf_bias(row: dict[str, Any]) -> str:
    text = f"{row.get('primary_blocker', '')} {row.get('message', '')}".lower()
    if "htf_bias=buy" in text or "htf bias buy" in text:
        return "buy"
    if "htf_bias=sell" in text or "htf bias sell" in text:
        return "sell"
    return "unknown"


def _infer_loss_pattern_state(row: dict[str, Any]) -> str:
    text = f"{row.get('primary_blocker', '')} {row.get('message', '')}".lower()
    if "loss-pattern" in text or "loss pattern" in text or "loss_learning" in text:
        return "active"
    if text.strip():
        return "inactive"
    return "unknown"


def _structure_aligns(side: str, structure: str) -> bool:
    text = str(structure or "").lower()
    if side == "buy":
        return any(token in text for token in ("hh", "hl", "uptrend", "bull"))
    if side == "sell":
        return any(token in text for token in ("lh", "ll", "downtrend", "bear"))
    return False


def _top_missing_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        for field in list(row.get("missing", []) or []):
            key = str(field or "").strip()
            if key:
                counts[key] = counts.get(key, 0) + 1
    return [
        {"field": field, "count": count}
        for field, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
    ]


def _proposal_target_blocker(base: Path, proposal_id: str) -> str:
    project_base = Path(__file__).resolve().parents[2]
    if base.resolve() != (project_base / "storage").resolve():
        return ""
    try:
        payload = json.loads((project_base / "knowledge" / "proposal_queue.json").read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    for proposal in list(dict(payload or {}).get("proposals", []) or []):
        if str(dict(proposal).get("proposal_id", "") or "") == str(proposal_id or ""):
            return str(dict(dict(proposal).get("scope", {}) or {}).get("target_blocker", "") or "").strip()
    return ""


def _summary_policy() -> dict[str, bool]:
    return {
        "summary_only": True,
        "does_not_execute": True,
        "does_not_change_strategy_logic": True,
        "does_not_change_risk_policy": True,
        "does_not_change_execution_policy": True,
    }


def _selectivity_review_policy() -> dict[str, bool]:
    return {
        "review_only": True,
        "does_not_execute": True,
        "does_not_change_strategy_logic": True,
        "does_not_change_risk_policy": True,
        "does_not_change_execution_policy": True,
        "requires_operator_approval_for_any_candidate_rule": True,
    }


def _selectivity_replay_policy() -> dict[str, bool]:
    return {
        "shadow_review_only": True,
        "does_not_execute": True,
        "does_not_allow_trade": True,
        "does_not_change_strategy_logic": True,
        "does_not_change_risk_policy": True,
        "does_not_change_execution_policy": True,
        "does_not_lower_thresholds": True,
        "requires_operator_approval_for_any_candidate_rule": True,
    }


def _fingerprint(
    *,
    symbol: str,
    timeframe: str,
    side: str,
    strategy: str,
    primary_blocker: str,
    context: dict[str, Any],
) -> str:
    parts = [
        f"symbol={str(symbol or '').upper()}",
        f"tf={str(timeframe or '').upper()}",
        f"side={str(side or 'flat').lower()}",
        f"strategy={str(strategy or '').split('(', 1)[0].strip().lower()}",
        f"blocker={str(primary_blocker or '').strip().lower()}",
        f"regime={str(context.get('regime', '') or '').lower()}",
        f"momentum={str(context.get('momentum', '') or '').lower()}",
        f"location={str(context.get('location', '') or '').lower()}",
    ]
    return "|".join(parts)


def _parse_dt(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recent_duplicate(record: dict[str, Any], recent: list[dict[str, Any]], *, window_seconds: float) -> bool:
    fingerprint = str(record.get("fingerprint", "") or "")
    created = _parse_dt(record.get("created_at"))
    if not fingerprint or created is None:
        return False
    for row in reversed(list(recent or [])):
        if str(row.get("fingerprint", "") or "") != fingerprint:
            continue
        previous = _parse_dt(row.get("created_at"))
        if previous is None:
            continue
        if abs((created - previous).total_seconds()) <= max(1.0, float(window_seconds)):
            return True
    return False
