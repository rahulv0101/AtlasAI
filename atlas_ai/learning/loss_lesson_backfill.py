from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from atlas_ai.learning.post_trade_lessons import build_post_trade_lesson
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategy_aliases import canonical_strategy_name


def _label(value: Any, default: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    return text or default


def _parse_iso(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _strategy_key(value: Any) -> str:
    raw = str(value or "unknown").strip()
    if raw.startswith("evolved_rule("):
        return "evolved_rule"
    return canonical_strategy_name(raw).split("(", 1)[0].strip().lower() or "unknown"


def _volatility_state(value: Any) -> str:
    try:
        vol = float(value or 0.0)
    except Exception:
        return "unknown"
    if vol <= 0.0:
        return "unknown"
    if vol < 0.02:
        return "low"
    if vol < 0.05:
        return "normal"
    return "high"


def _momentum_state(context: dict[str, Any]) -> str:
    if _label(context.get("entry_momentum_state"), ""):
        return _label(context.get("entry_momentum_state"))
    try:
        body_ratio = float(context.get("body_ratio", 0.0) or 0.0)
        trend_strength = float(context.get("trend_strength", 0.0) or 0.0)
    except Exception:
        return "unknown"
    if body_ratio >= 0.65 and trend_strength >= 0.12:
        return "strong"
    if body_ratio >= 0.35 or trend_strength >= 0.08:
        return "mixed"
    return "weak"


def _location_state(context: dict[str, Any]) -> str:
    existing = _label(context.get("entry_location_state") or context.get("location"), "")
    if existing:
        return existing
    reasons = " ".join(str(x).lower() for x in list(context.get("setup_quality_reasons", []) or []))
    if "mid" in reasons and "range" in reasons:
        return "mid_range"
    if "pullback" in reasons:
        return "pullback"
    if "breakout" in reasons or "retest" in reasons:
        return "breakout"
    return "unknown"


def _context_from_outcome(row: dict[str, Any]) -> dict[str, Any]:
    context = dict(row.get("context", {}) or {})
    for source_key, target_key in [
        ("session", "session"),
        ("entry_session", "session"),
        ("regime", "regime"),
        ("spread_points", "spread_points"),
        ("higher_tf_bias", "higher_tf_bias"),
        ("higher_timeframe_bias", "higher_tf_bias"),
        ("entry_location_state", "entry_location_state"),
        ("location", "entry_location_state"),
        ("entry_momentum_state", "entry_momentum_state"),
        ("momentum", "entry_momentum_state"),
        ("trend_state", "trend_state"),
        ("entry_trend_state", "entry_trend_state"),
        ("volatility_state", "volatility_state"),
        ("volatility_bucket", "volatility_state"),
        ("volatility_pct", "volatility_pct"),
        ("trend_strength", "trend_strength"),
        ("body_ratio", "body_ratio"),
        ("volume_ratio", "volume_ratio"),
        ("same_side_support", "same_side_support"),
        ("htf_aligned", "htf_aligned"),
    ]:
        if source_key in row and target_key not in context:
            context[target_key] = row.get(source_key)
    if not _label(context.get("volatility_state"), ""):
        context["volatility_state"] = _volatility_state(context.get("volatility_pct"))
    if not _label(context.get("entry_momentum_state"), ""):
        context["entry_momentum_state"] = _momentum_state(context)
    if not _label(context.get("entry_location_state"), ""):
        context["entry_location_state"] = _location_state(context)
    return context


def _decision_context(decision_meta: dict[str, Any]) -> dict[str, Any]:
    meta = dict(decision_meta or {})
    contextual = dict(meta.get("contextual_ranking", {}) or {})
    ctx = dict(contextual.get("context", {}) or {})
    setup_quality = dict(contextual.get("setup_quality", {}) or {})
    for key in [
        "session",
        "regime",
        "higher_tf_bias",
        "spread_points",
        "confidence",
        "same_side_support",
        "same_side_ratio",
        "m1_quality_factor",
        "timeframe",
        "volatility_pct",
        "trend_strength",
        "body_ratio",
        "volume_ratio",
        "spike_ratio",
        "htf_aligned",
        "meta_probability",
        "entry_location_state",
        "entry_momentum_state",
        "entry_trend_state",
        "volatility_state",
    ]:
        if key not in ctx and key in meta:
            ctx[key] = meta.get(key)
    if "setup_quality_reasons" not in ctx:
        ctx["setup_quality_reasons"] = list(setup_quality.get("reasons", []) or [])
    if "setup_quality_score" not in ctx:
        ctx["setup_quality_score"] = float(setup_quality.get("score", 0.0) or 0.0)
    ctx["volatility_state"] = _label(ctx.get("volatility_state"), _volatility_state(ctx.get("volatility_pct")))
    ctx["entry_momentum_state"] = _momentum_state(ctx)
    ctx["entry_location_state"] = _location_state(ctx)
    ctx.setdefault("entry_trend_state", _label(ctx.get("regime"), "unknown"))
    return ctx


def _decision_events(store: KnowledgeStore, *, event_limit: int = 50000) -> list[dict[str, Any]]:
    events = list(store.tail_events(limit=event_limit, event_type="decision_sample"))
    archive_dir = Path(store.base) / "archive"
    if archive_dir.exists():
        for path in sorted(archive_dir.glob("events_*.jsonl")):
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if '"decision_sample"' not in line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        if str(row.get("type", "")).strip().lower() == "decision_sample":
                            events.append(row)
            except Exception:
                continue
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        meta = dict(event.get("meta", {}) or {})
        key = (str(event.get("time", "") or ""), str(meta.get("decision_id", "") or ""))
        deduped[key] = event
    rows = list(deduped.values())
    rows.sort(key=lambda event: str(event.get("time", "") or ""))
    return rows[-event_limit:]


def _decision_pool(store: KnowledgeStore, *, event_limit: int = 50000) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for event in _decision_events(store, event_limit=event_limit):
        event_time = _parse_iso(event.get("time"))
        meta = dict(event.get("meta", {}) or {})
        side = _label(meta.get("side"), "flat")
        if event_time is None or side not in {"buy", "sell"}:
            continue
        pool.append(
            {
                "time": event_time,
                "strategy_key": _strategy_key(meta.get("strategy")),
                "side": side,
                "session": _label(meta.get("session")),
                "decision_id": str(meta.get("decision_id", "") or "").strip().lower(),
                "meta": meta,
            }
        )
    pool.sort(key=lambda row: row["time"])
    return pool


def _find_decision_for_outcome(
    outcome: dict[str, Any],
    pool: list[dict[str, Any]],
    *,
    max_gap_hours: float = 18.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcome_time = _parse_iso(outcome.get("time"))
    if outcome_time is None:
        return {}, {"linked": False, "reason": "missing_outcome_time"}
    decision_id = str(outcome.get("decision_id", "") or "").strip().lower()
    strategy = _strategy_key(outcome.get("strategy"))
    side = _label(outcome.get("side"), "flat")
    session = _label(outcome.get("entry_session") or outcome.get("session"))
    max_gap = timedelta(hours=max(1.0, float(max_gap_hours)))
    best: dict[str, Any] = {}
    best_rank: tuple[int, int, float] | None = None
    for candidate in pool:
        cand_time = candidate["time"]
        if cand_time > outcome_time:
            continue
        gap = outcome_time - cand_time
        if gap > max_gap:
            continue
        if side in {"buy", "sell"} and str(candidate.get("side")) != side:
            continue
        if decision_id and candidate.get("decision_id") == decision_id:
            return dict(candidate.get("meta", {}) or {}), {
                "linked": True,
                "method": "decision_id",
                "gap_minutes": round(gap.total_seconds() / 60.0, 2),
            }
        if strategy not in {"", "unknown"} and str(candidate.get("strategy_key")) != strategy:
            continue
        same_session_rank = 0 if str(candidate.get("session")) == session else 1
        rank = (same_session_rank, 0, gap.total_seconds())
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = dict(candidate.get("meta", {}) or {})
    if not best:
        return {}, {"linked": False, "reason": "no_prior_matching_decision"}
    return best, {
        "linked": True,
        "method": "nearest_prior_strategy_side",
        "gap_minutes": round(float(best_rank[2]) / 60.0, 2) if best_rank else 0.0,
    }


def enrich_outcomes_with_decision_context(
    outcomes: list[dict[str, Any]],
    store: KnowledgeStore,
    *,
    max_gap_hours: float = 18.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = _decision_pool(store)
    enriched_rows: list[dict[str, Any]] = []
    linked = 0
    for row in outcomes:
        outcome = dict(row or {})
        decision, link = _find_decision_for_outcome(outcome, pool, max_gap_hours=max_gap_hours)
        if decision:
            linked += 1
            ctx = _decision_context(decision)
            existing_context = dict(outcome.get("context", {}) or {})
            outcome["context"] = {**ctx, **existing_context}
            outcome["decision_context_link"] = {
                **link,
                "decision_id": str(decision.get("decision_id", outcome.get("decision_id", "")) or ""),
                "decision_time": str(decision.get("time", "") or ""),
                "strategy": str(decision.get("strategy", outcome.get("strategy", "")) or ""),
                "bounded_learning_only": True,
            }
            for source_key in [
                "decision_id",
                "strategy",
                "side",
                "session",
                "regime",
                "higher_tf_bias",
                "spread_points",
                "confidence",
                "trend_strength",
                "volatility_pct",
                "setup_family",
            ]:
                if decision.get(source_key) not in {None, ""}:
                    outcome[source_key] = decision.get(source_key)
            outcome["entry_session"] = str(decision.get("session", outcome.get("entry_session", outcome.get("session", ""))) or "")
            outcome["volatility_state"] = ctx.get("volatility_state")
            outcome["entry_momentum_state"] = ctx.get("entry_momentum_state")
            outcome["entry_location_state"] = ctx.get("entry_location_state")
            outcome["entry_trend_state"] = ctx.get("entry_trend_state")
            outcome["htf_aligned"] = ctx.get("htf_aligned")
        else:
            outcome["decision_context_link"] = {**link, "bounded_learning_only": True}
        enriched_rows.append(outcome)
    return enriched_rows, {
        "decision_samples_scanned": len(pool),
        "outcomes_scanned": len(outcomes),
        "outcomes_linked": linked,
        "link_rate": round(linked / max(1, len(outcomes)), 4),
        "policy_note": "Decision context enrichment is bounded teaching memory only; it does not change strategy, risk, guards, or execution policy.",
    }


def build_trade_pattern_fingerprint(outcome_row: dict[str, Any]) -> str:
    row = dict(outcome_row or {})
    context = _context_from_outcome(row)
    strategy = _strategy_key(row.get("strategy", "unknown"))
    side = _label(row.get("side"), "flat")
    setup_family = _label(row.get("setup_family"), "unknown")
    session = _label(row.get("entry_session") or row.get("session"), "unknown")
    regime = _label(row.get("regime") or context.get("regime"), "unknown")
    location = _label(
        context.get("entry_location_state")
        or context.get("location")
        or context.get("entry_location"),
        "unknown",
    )
    momentum = _label(
        context.get("entry_momentum_state")
        or context.get("momentum")
        or context.get("entry_momentum"),
        "unknown",
    )
    trend_state = _label(
        context.get("entry_trend_state")
        or context.get("trend_state")
        or regime,
        "unknown",
    )
    volatility_state = _label(context.get("volatility_state") or context.get("volatility_bucket"), "unknown")
    parts = {
        "strategy": strategy,
        "side": side,
        "setup": setup_family,
        "session": session,
        "regime": regime,
        "trend": trend_state,
        "location": location,
        "momentum": momentum,
        "volatility": volatility_state,
    }
    return "|".join(f"{key}={value}" for key, value in parts.items())


def _execution_lesson(classification: str, setup_family: str) -> dict[str, Any]:
    code = _label(classification, "weak_structure")
    table: dict[str, dict[str, Any]] = {
        "false_continuation": {
            "correction": "Wait for momentum to confirm in the trade direction after structure holds.",
            "confirmation_required": ["strong_m1_momentum", "structure_hold", "avoid_first_push"],
        },
        "weak_pullback": {
            "correction": "Wait for pullback location and M1 momentum to turn back with the setup.",
            "confirmation_required": ["valid_pullback_location", "m1_momentum_turn", "controlled_wick_pressure"],
        },
        "late_entry": {
            "correction": "Avoid chasing expansion; wait for a retest, pullback, or fresh structure entry.",
            "confirmation_required": ["no_overextension", "retest_or_pullback", "fresh_entry_location"],
        },
        "spread_liquidity_distortion": {
            "correction": "Wait until execution friction normalizes before repeating the setup.",
            "confirmation_required": ["spread_normalized", "liquidity_normal"],
        },
        "stop_hunt": {
            "correction": "Wait for sweep rejection or structure reclaim before treating the move as confirmed.",
            "confirmation_required": ["rejection_after_sweep", "structure_reclaim", "momentum_confirmation"],
        },
    }
    lesson = dict(table.get(code, {}))
    if not lesson:
        lesson = {
            "correction": "Treat the repeated loss fingerprint as developing until cleaner confirmation appears.",
            "confirmation_required": ["cleaner_structure", "momentum_confirmation", "valid_entry_location"],
        }
    return {
        "classification": code,
        "setup_family": _label(setup_family),
        "correction": lesson["correction"],
        "confirmation_required": list(lesson["confirmation_required"]),
        "bounded_learning_only": True,
        "policy_note": "Backfilled execution lesson can inform bounded confidence evidence only; it cannot rewrite strategy, risk, guards, or execution policy.",
    }


def _enrich_outcome(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row or {})
    outcome = _label(enriched.get("outcome"), "draw")
    if outcome not in {"win", "loss", "draw"}:
        outcome = "draw"
        enriched["outcome"] = outcome
    fingerprint = build_trade_pattern_fingerprint(enriched)
    enriched["pattern_fingerprint"] = fingerprint
    review = dict(enriched.get("post_trade_review", {}) or {})
    classification = _label(enriched.get("loss_classification") or review.get("classification"), "none")
    if outcome == "loss" and classification in {"none", "unknown"}:
        classification = "weak_structure"
        enriched["loss_classification"] = classification
    if "execution_lesson" not in enriched or not isinstance(enriched.get("execution_lesson"), dict):
        enriched["execution_lesson"] = _execution_lesson(classification, str(enriched.get("setup_family", "unknown") or "unknown"))
    enriched["post_trade_lesson"] = build_post_trade_lesson(enriched, _context_from_outcome(enriched))
    review.setdefault("classification", classification)
    review.setdefault("bounded_learning_only", True)
    review.setdefault("validation_required", True)
    review["post_trade_lesson"] = dict(enriched.get("post_trade_lesson", {}) or {})
    enriched["post_trade_review"] = review
    return enriched


def rebuild_loss_pattern_lessons(store: KnowledgeStore, *, max_outcomes: int = 200) -> dict[str, Any]:
    state = dict(store.load_learning_state().get("state", {}) or {})
    memory = dict(state.get("loss_memory", {}) or {})
    raw_recent = [dict(row or {}) for row in list(memory.get("recent_outcomes", []) or [])[-max_outcomes:]]
    linked_recent, link_summary = enrich_outcomes_with_decision_context(raw_recent, store)
    recent = [_enrich_outcome(row) for row in linked_recent]

    lessons: dict[str, dict[str, Any]] = {}
    for row in recent:
        fingerprint = str(row.get("pattern_fingerprint", "") or "")
        if not fingerprint:
            continue
        existing = lessons.setdefault(
            fingerprint,
            {
                "fingerprint": fingerprint,
                "samples": 0,
                "wins": 0,
                "losses": 0,
                "bounded_learning_only": True,
            },
        )
        outcome = _label(row.get("outcome"), "draw")
        existing["samples"] = int(existing.get("samples", 0) or 0) + 1
        if outcome == "win":
            existing["wins"] = int(existing.get("wins", 0) or 0) + 1
        elif outcome == "loss":
            existing["losses"] = int(existing.get("losses", 0) or 0) + 1
            existing["execution_lesson"] = dict(row.get("execution_lesson", {}) or {})
            existing["post_trade_lesson"] = dict(row.get("post_trade_lesson", {}) or {})
            existing["failure_axis"] = str(dict(row.get("post_trade_lesson", {}) or {}).get("failure_axis", "") or "")
            existing["idea_quality"] = str(dict(row.get("post_trade_lesson", {}) or {}).get("idea_quality", "") or "")
            existing["timing_quality"] = str(dict(row.get("post_trade_lesson", {}) or {}).get("timing_quality", "") or "")
        existing["last_seen"] = str(row.get("time", "") or existing.get("last_seen", ""))
        existing["last_outcome"] = outcome
        existing["strategy"] = str(row.get("strategy", existing.get("strategy", "")) or "")
        existing["side"] = str(row.get("side", existing.get("side", "")) or "")
        existing["setup_family"] = str(row.get("setup_family", existing.get("setup_family", "")) or "")
        existing["context_quality"] = {
            "location_known": _label(dict(row.get("context", {}) or {}).get("entry_location_state"), "") not in {"", "unknown"},
            "momentum_known": _label(dict(row.get("context", {}) or {}).get("entry_momentum_state"), "") not in {"", "unknown"},
            "volatility_known": _label(dict(row.get("context", {}) or {}).get("volatility_state"), "") not in {"", "unknown"},
            "decision_context_linked": bool(dict(row.get("decision_context_link", {}) or {}).get("linked", False)),
        }

    for lesson in lessons.values():
        samples = int(lesson.get("samples", 0) or 0)
        losses = int(lesson.get("losses", 0) or 0)
        lesson["loss_rate"] = round(losses / max(1, samples), 4)

    memory["recent_outcomes"] = recent[-max_outcomes:]
    memory["pattern_lessons"] = dict(list(lessons.items())[-250:])
    memory["lesson_backfill"] = {
        "schema": "atlas_loss_lesson_backfill_v1",
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "recent_outcomes": len(recent),
        "pattern_lessons": len(memory["pattern_lessons"]),
        "loss_dominant_lessons": sum(
            1
            for lesson in memory["pattern_lessons"].values()
            if int(lesson.get("losses", 0) or 0) > 0 and float(lesson.get("loss_rate", 0.0) or 0.0) >= 0.5
        ),
        "decision_context": link_summary,
        "context_quality": {
            "location_known": sum(
                1
                for row in recent
                if _label(dict(row.get("context", {}) or {}).get("entry_location_state"), "") not in {"", "unknown"}
            ),
            "momentum_known": sum(
                1
                for row in recent
                if _label(dict(row.get("context", {}) or {}).get("entry_momentum_state"), "") not in {"", "unknown"}
            ),
            "volatility_known": sum(
                1
                for row in recent
                if _label(dict(row.get("context", {}) or {}).get("volatility_state"), "") not in {"", "unknown"}
            ),
        },
        "policy_note": "Backfill rebuilds structured teaching memory from recorded outcomes only; it does not alter strategy, risk, guards, or execution policy.",
    }
    state["loss_memory"] = memory
    store.save_learning_state(state)
    store.log_event("learning", "Loss-pattern lesson memory backfilled.", memory["lesson_backfill"])
    return dict(memory["lesson_backfill"])


def main() -> None:
    summary = rebuild_loss_pattern_lessons(KnowledgeStore("storage"))
    print(summary)


if __name__ == "__main__":
    main()
