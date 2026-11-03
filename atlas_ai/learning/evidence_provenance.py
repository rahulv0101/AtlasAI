"""Source labels and exclusion of operator plumbing tests from strategy evidence."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def is_manual_broker_test(row: dict[str, Any]) -> bool:
    text = " ".join(str(row.get(key, "") or "").lower() for key in
                    ("strategy", "decision_id", "comment", "requested_by", "request_origin"))
    return "manual_broker_test" in text or (
        str(row.get("strategy", "")).lower() == "mabr"
        and str(row.get("decision_id", "")).lower() == "manual"
    )


def outcome_source(row: dict[str, Any]) -> str:
    if is_manual_broker_test(row):
        return "manual_test"
    review = dict(row.get("post_trade_review", {}) or {})
    if review.get("source") == "historical_backtest" or str(row.get("ticket", "")).startswith("bt_"):
        return "historical_backtest"
    if row.get("forced_demo_entry") or row.get("forced_learning_probe"):
        return "forced_probe"
    if str(row.get("ticket", "")).isdigit():
        return "broker_recorded"
    return "unknown"


def source_counts(memory: dict[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(("historical_backtest", "broker_recorded", "manual_test", "forced_probe", "unknown"), 0)
    for row in memory.get("recent_outcomes", []):
        counts[outcome_source(row)] += 1
    counts["excluded_manual_tests"] = len(memory.get("excluded_manual_test_outcomes", []))
    return counts


def exclude_manual_tests(state: dict[str, Any]) -> dict[str, Any]:
    """Remove only retained test contributions; repeated calls are idempotent."""
    memory = dict(state.get("loss_memory", {}) or {})
    removed = [row for row in memory.get("recent_outcomes", []) if is_manual_broker_test(row)]
    if not removed:
        return state
    state = deepcopy(state)
    memory = state["loss_memory"]
    archive = memory.setdefault("excluded_manual_test_outcomes", [])
    archived = {str(row.get("ticket")) for row in archive}
    for row in removed:
        ticket = str(row.get("ticket"))
        if ticket in archived:
            continue
        archive.append({**row, "exclusion_reason": "manual_broker_test_excluded_from_strategy_learning"})
        archived.add(ticket)
        # These aggregates contain more history than recent_outcomes: subtract the
        # known contribution instead of rebuilding them from the truncated window.
        key = f"{row.get('side')}:{row.get('hour_utc')}"
        aggregates = [memory.get("side_hour_stats", {}).get(key),
                      state.get("allocator", {}).get(row.get("strategy"))]
        for aggregate in aggregates:
            if aggregate is None:
                continue
            field = {"loss": "losses", "win": "wins"}.get(row.get("outcome"))
            if field:
                aggregate[field] = max(0, int(aggregate.get(field, 0)) - 1)
            aggregate["pnl"] = float(aggregate.get("pnl", 0)) - float(row.get("net_pnl_zar", 0))
        fingerprint = row.get("pattern_fingerprint")
        lesson = memory.get("pattern_lessons", {}).get(fingerprint)
        if lesson:
            lesson["samples"] = max(0, int(lesson.get("samples", 0)) - 1)
            field = {"loss": "losses", "win": "wins"}.get(row.get("outcome"))
            if field:
                lesson[field] = max(0, int(lesson.get(field, 0)) - 1)
            if not lesson["samples"]:
                del memory["pattern_lessons"][fingerprint]
            else:
                lesson["loss_rate"] = lesson.get("losses", 0) / lesson["samples"]
    memory["recent_outcomes"] = [row for row in memory["recent_outcomes"] if not is_manual_broker_test(row)]
    lessons = memory.get("pattern_lessons", {})
    summary = memory.setdefault("lesson_backfill", {})
    summary.update(recent_outcomes=len(memory["recent_outcomes"]), pattern_lessons=len(lessons),
                   loss_dominant_lessons=sum(1 for row in lessons.values() if row.get("losses", 0) > 0 and row.get("loss_rate", 0) >= .5))
    return state
