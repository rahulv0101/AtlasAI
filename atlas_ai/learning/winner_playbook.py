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


def _winner_confirmation_template(setup_family: str) -> dict[str, Any]:
    family = _label(setup_family, "unknown")
    table: dict[str, dict[str, Any]] = {
        "pullback": {
            "winning_signature": "Pullback winners need a clean reaction location and momentum turning with the setup.",
            "confirmation_required": ["valid_pullback_location", "momentum_turn_or_strong_momentum", "not_mid_range"],
        },
        "breakout": {
            "winning_signature": "Breakout winners need follow-through or a retest instead of a raw chase.",
            "confirmation_required": ["breakout_follow_through_or_retest", "no_overextension", "execution_friction_acceptable"],
        },
        "continuation": {
            "winning_signature": "Continuation winners need structure to hold and momentum to support the same side.",
            "confirmation_required": ["structure_hold", "strong_m1_momentum", "same_side_support"],
        },
        "reversal": {
            "winning_signature": "Reversal winners need sweep/rejection evidence and a structure reclaim.",
            "confirmation_required": ["rejection_after_sweep", "structure_reclaim", "momentum_confirmation"],
        },
        "unknown": {
            "winning_signature": "Winners need clean location, momentum confirmation, and acceptable execution friction.",
            "confirmation_required": ["clean_location", "momentum_confirmation", "execution_friction_acceptable"],
        },
    }
    return dict(table.get(family, table["unknown"]))


def build_winner_playbook(store: KnowledgeStore) -> dict[str, Any]:
    state = dict(store.load_learning_state().get("state", {}) or {})
    analysis = dict(state.get("historical_winner_analysis", {}) or {})
    wins = [dict(row or {}) for row in list(analysis.get("win_samples", []) or [])]
    if not wins:
        playbook = {
            "schema": "atlas_winner_playbook_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "historical_winner_analysis",
            "status": "no_historical_winners_available",
            "lessons": {},
            "policy_note": "No winner playbook created because no historical winner samples were available.",
        }
        store.update_learning_state(lambda current: {**current, "winner_playbook": playbook})
        return playbook

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in wins:
        strategy = _label(row.get("strategy"))
        setup_family = _label(row.get("setup_family"))
        grouped[f"{strategy}|{setup_family}"].append(row)

    lessons: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        strategy, setup_family = key.split("|", 1)
        template = _winner_confirmation_template(setup_family)
        lessons[key] = {
            "key": key,
            "strategy": strategy,
            "setup_family": setup_family,
            "samples": int(len(rows)),
            "average_win_zar": round(sum(float(row.get("net_pnl_zar", 0.0) or 0.0) for row in rows) / max(1, len(rows)), 4),
            "average_hold_minutes": round(sum(float(row.get("duration_minutes", 0.0) or 0.0) for row in rows) / max(1, len(rows)), 2),
            "winning_signature": template["winning_signature"],
            "confirmation_required": list(template["confirmation_required"]),
            "sessions": _top_counts(rows, "session"),
            "regimes": _top_counts(rows, "regime"),
            "sides": _top_counts(rows, "side"),
            "example_tickets": [str(row.get("ticket", "") or "") for row in rows[-5:] if row.get("ticket")],
            "bounded_learning_only": True,
            "can_force_trade": False,
            "policy_note": "Winner playbook is positive teaching evidence only; it cannot force entries, bypass guards, or change risk/execution policy.",
        }

    playbook = {
        "schema": "atlas_winner_playbook_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "historical_winner_analysis",
        "status": "active",
        "symbol": str(analysis.get("symbol", "unknown") or "unknown"),
        "timeframe": str(analysis.get("timeframe", "unknown") or "unknown"),
        "bars": int(analysis.get("bars", 0) or 0),
        "total_trades": int(analysis.get("total_trades", 0) or 0),
        "wins": int(analysis.get("wins", len(wins)) or len(wins)),
        "losses": int(analysis.get("losses", 0) or 0),
        "win_rate": float(analysis.get("win_rate", 0.0) or 0.0),
        "lessons": lessons,
        "dominant_lessons": [
            {"key": key, "samples": int(value.get("samples", 0) or 0)}
            for key, value in sorted(lessons.items(), key=lambda item: int(item[1].get("samples", 0) or 0), reverse=True)
        ],
        "policy_note": (
            "This playbook teaches what historical winning setup versions looked like. "
            "It does not rewrite strategies, lower thresholds, alter risk, or bypass execution gates."
        ),
    }
    store.update_learning_state(lambda current: {**current, "winner_playbook": playbook})
    store.log_event("learning", "Winner playbook rebuilt from historical winners.", {
        "schema": playbook["schema"],
        "lessons": len(lessons),
        "wins": playbook["wins"],
        "dominant_lessons": playbook["dominant_lessons"][:5],
    })
    return playbook
