from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from atlas_ai.memory.knowledge_store import KnowledgeStore


def _label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _top_counts(rows: list[dict[str, Any]], field: str, limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter(_label(row.get(field)) for row in rows)
    return [{"value": value, "count": int(count)} for value, count in counts.most_common(limit)]


def _lesson_template(classification: str) -> dict[str, Any]:
    code = _label(classification, "weak_structure")
    table: dict[str, dict[str, Any]] = {
        "weak_pullback": {
            "failed_when": ["pullback entered before M1 momentum turned", "entry quality was not clean enough"],
            "corrected_entry": "Only treat the pullback as tradable after price reaches a valid reaction zone and M1 momentum turns back with the setup.",
            "confirmation_required": ["valid_pullback_location", "strong_or_turning_m1_momentum", "controlled_wick_pressure"],
            "avoid_until": ["mid_range_removed", "momentum_not_mixed"],
            "failure_axis": "location_or_structure",
        },
        "stop_hunt": {
            "failed_when": ["entry was exposed to wick pressure", "price swept before confirming direction"],
            "corrected_entry": "Wait for sweep rejection, structure reclaim, or calmer wick pressure before treating the move as confirmed.",
            "confirmation_required": ["rejection_after_sweep", "structure_reclaim_or_hold", "reduced_wick_pressure"],
            "avoid_until": ["wick_pressure_normalizes", "confirmation_after_sweep"],
            "failure_axis": "timing",
        },
        "late_entry": {
            "failed_when": ["entry chased expansion", "price was already stretched or near an extreme"],
            "corrected_entry": "Skip the chase and wait for a retest, pullback, or fresh structure entry before executing.",
            "confirmation_required": ["no_overextension", "retest_or_pullback", "fresh_entry_location"],
            "avoid_until": ["price_not_stretched", "fresh_location_available"],
            "failure_axis": "timing",
        },
        "false_continuation": {
            "failed_when": ["continuation was assumed before momentum and structure confirmed"],
            "corrected_entry": "Require structure to hold and momentum to confirm before accepting continuation.",
            "confirmation_required": ["structure_hold", "strong_m1_momentum", "same_side_support"],
            "avoid_until": ["first_push_confirmed", "momentum_strong"],
            "failure_axis": "confirmation_quality",
        },
        "weak_structure": {
            "failed_when": ["structure was not clean enough", "location or momentum did not support execution"],
            "corrected_entry": "Require clearer HH/HL or LH/LL structure, non-mid-range location, and momentum alignment.",
            "confirmation_required": ["clear_structure", "not_mid_range", "momentum_alignment"],
            "avoid_until": ["structure_clear", "location_clean"],
            "failure_axis": "location_or_structure",
        },
    }
    lesson = dict(table.get(code, table["weak_structure"]))
    lesson["classification"] = code
    lesson["bounded_learning_only"] = True
    lesson["can_boost_confidence"] = False
    lesson["policy_note"] = (
        "Loss-to-win lesson is teaching evidence only. It may clear a loss-pattern caution when "
        "the corrected confirmations are present, but it cannot force entries, bypass guards, "
        "or change strategy, risk, or execution policy."
    )
    return lesson


def build_loss_to_win_playbook(store: KnowledgeStore) -> dict[str, Any]:
    state = dict(store.load_learning_state().get("state", {}) or {})
    analysis = dict(state.get("historical_loss_analysis", {}) or {})
    losses = [dict(row or {}) for row in list(analysis.get("loss_samples", []) or [])]
    if not losses:
        playbook = {
            "schema": "atlas_loss_to_win_playbook_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "historical_loss_analysis",
            "status": "no_historical_losses_available",
            "lessons": {},
            "policy_note": "No playbook created because no historical loss samples were available.",
        }
        store.update_learning_state(lambda current: {**current, "loss_to_win_playbook": playbook})
        return playbook

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in losses:
        grouped[_label(row.get("loss_classification"), "weak_structure")].append(row)

    lessons: dict[str, dict[str, Any]] = {}
    for classification, rows in grouped.items():
        template = _lesson_template(classification)
        lessons[classification] = {
            **template,
            "samples": int(len(rows)),
            "strategies": _top_counts(rows, "strategy"),
            "sessions": _top_counts(rows, "session"),
            "regimes": _top_counts(rows, "regime"),
            "sides": _top_counts(rows, "side"),
            "average_loss_zar": round(sum(float(row.get("net_pnl_zar", 0.0) or 0.0) for row in rows) / max(1, len(rows)), 4),
            "example_tickets": [str(row.get("ticket", "") or "") for row in rows[-5:] if row.get("ticket")],
        }

    playbook = {
        "schema": "atlas_loss_to_win_playbook_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "historical_loss_analysis",
        "status": "active",
        "symbol": str(analysis.get("symbol", "unknown") or "unknown"),
        "timeframe": str(analysis.get("timeframe", "unknown") or "unknown"),
        "bars": int(analysis.get("bars", 0) or 0),
        "total_trades": int(analysis.get("total_trades", 0) or 0),
        "losses": int(analysis.get("losses", len(losses)) or len(losses)),
        "win_rate": float(analysis.get("win_rate", 0.0) or 0.0),
        "lessons": lessons,
        "dominant_lessons": [
            {"classification": key, "samples": int(value.get("samples", 0) or 0)}
            for key, value in sorted(lessons.items(), key=lambda item: int(item[1].get("samples", 0) or 0), reverse=True)
        ],
        "policy_note": (
            "This playbook converts repeated historical losses into corrected-entry teaching. "
            "It does not rewrite strategies, lower thresholds, alter risk, or bypass execution gates."
        ),
    }
    store.update_learning_state(lambda current: {**current, "loss_to_win_playbook": playbook})
    store.log_event("learning", "Loss-to-win playbook rebuilt from historical losses.", {
        "schema": playbook["schema"],
        "lessons": len(lessons),
        "losses": playbook["losses"],
        "dominant_lessons": playbook["dominant_lessons"][:5],
    })
    return playbook
