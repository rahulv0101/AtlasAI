from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.brain.ai_decision_journal import AIDecisionJournal, summarize_decision_record
from atlas_ai.brain.ai_strategy_preferences import AIStrategyPreferences
from atlas_ai.memory.jsonl_tail import read_jsonl_tail


class AIDecisionReviewer:
    """Review AI decisions against later outcomes and blocked events."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.journal = AIDecisionJournal(self.base)
        self.preferences = AIStrategyPreferences(self.base)
        self.path = self.base / "ai_decision_reviews.jsonl"

    def review(
        self,
        *,
        learning_state: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        records = self.journal.recent(limit=limit)
        outcomes = _recent_outcomes(dict(learning_state or {}), limit=limit)
        events = [dict(e or {}) for e in list(events or [])[-limit:]]
        reviews = [self._review_record(record, outcomes, events) for record in records]
        for review in reviews:
            self._append_review(review)
        preferences = self.preferences.update_from_reviews(reviews) if reviews else self.preferences.load()
        return {
            "schema": "atlas_ai_decision_review_batch_v1",
            "reviewed": len(reviews),
            "outcomes_seen": len(outcomes),
            "events_seen": len(events),
            "latest_review": reviews[-1] if reviews else {},
            "summary": self._summary(reviews),
            "strategy_preferences": AIStrategyPreferences.summarize(preferences),
        }

    def recent_reviews(self, limit: int = 5) -> list[dict[str, Any]]:
        return read_jsonl_tail(self.path, limit)

    def _review_record(self, record: dict[str, Any], outcomes: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_decision_record(record)
        action = summary["action"]
        latest_outcome = outcomes[-1] if outcomes else {}
        outcome_label = str(latest_outcome.get("outcome", "") or "").lower()
        blocked_events = [
            e for e in events
            if str(e.get("type", "") or "").lower() in {"risk", "blocked", "paper_filtered"}
        ]
        blocker_quality = self._blocker_quality(summary, blocked_events)
        decision_outcome = self._decision_outcome(action, outcome_label, blocker_quality, bool(blocked_events))
        archetypes = list(summary.get("strategy_archetypes", []) or [])
        return {
            "schema": "atlas_ai_decision_review_v1",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "decision_recorded_at": summary["recorded_at"],
            "source": summary["source"],
            "query": summary["query"],
            "action": action,
            "decision_confidence": summary["confidence"],
            "trade_bias": summary["trade_bias"],
            "primary_blocker": summary["primary_blocker"],
            "outcome": decision_outcome,
            "latest_trade_outcome": outcome_label or "none",
            "blocker_quality": blocker_quality,
            "confidence_quality": self._confidence_quality(summary["confidence"], outcome_label, decision_outcome),
            "strategy_archetype_review": {
                "archetypes": archetypes,
                "status": "pending_outcome" if archetypes and not outcome_label else ("observed_with_trade_outcome" if archetypes else "none"),
            },
            "lesson": self._lesson(action, decision_outcome, summary["primary_blocker"]),
            "safety": {
                "review_only": True,
                "does_not_mutate_strategy": True,
                "does_not_mutate_risk": True,
                "does_not_execute": True,
            },
        }

    @staticmethod
    def _blocker_quality(summary: dict[str, Any], blocked_events: list[dict[str, Any]]) -> str:
        blocker = str(summary.get("primary_blocker", "") or "").lower()
        if not blocker or blocker == "none":
            return "no_blocker_claimed"
        if any(blocker in str(e.get("message", "") or "").lower() for e in blocked_events):
            return "confirmed_by_event"
        if blocked_events and any(token in str(blocked_events[-1].get("message", "") or "").lower() for token in blocker.split()[:2]):
            return "partially_confirmed"
        return "unconfirmed"

    @staticmethod
    def _decision_outcome(action: str, outcome_label: str, blocker_quality: str, has_blocked_events: bool) -> str:
        if action in {"wait_for_confirmation", "wait_and_reduce_frequency", "hold_and_observe"}:
            if blocker_quality in {"confirmed_by_event", "partially_confirmed"} or has_blocked_events:
                return "validated_wait"
            if outcome_label == "loss":
                return "validated_caution_after_loss"
            if outcome_label == "win":
                return "possible_overcaution"
            return "pending"
        if action == "prepare_trade_candidate":
            if outcome_label == "win":
                return "validated_prepare"
            if outcome_label == "loss":
                return "prepare_needs_review"
            return "pending"
        return "pending"

    @staticmethod
    def _confidence_quality(confidence: float, outcome_label: str, decision_outcome: str) -> str:
        if decision_outcome.startswith("validated"):
            return "aligned"
        if confidence >= 0.7 and decision_outcome in {"prepare_needs_review", "possible_overcaution"}:
            return "overconfident_or_miscalibrated"
        if not outcome_label:
            return "pending"
        return "unclear"

    @staticmethod
    def _lesson(action: str, decision_outcome: str, blocker: str) -> str:
        if decision_outcome == "validated_wait":
            return f"Waiting was justified because {blocker or 'a later blocker confirmed caution'}."
        if decision_outcome == "possible_overcaution":
            return "Caution may have skipped a winning condition; require more samples before loosening gates."
        if decision_outcome == "prepare_needs_review":
            return "Prepared trade candidate led into a loss; review timing, location, and confirmation quality."
        if decision_outcome == "validated_prepare":
            return "Prepared trade candidate aligned with a winning outcome; keep validation requirements intact."
        return "Outcome is still pending; do not adjust policy from this record alone."

    @staticmethod
    def _summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(r.get("outcome", "pending")) for r in reviews)
        return {
            "validated_wait": counts.get("validated_wait", 0),
            "validated_prepare": counts.get("validated_prepare", 0),
            "possible_overcaution": counts.get("possible_overcaution", 0),
            "prepare_needs_review": counts.get("prepare_needs_review", 0),
            "pending": counts.get("pending", 0),
        }

    def _append_review(self, review: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(review, sort_keys=True) + "\n")


def _recent_outcomes(learning_state: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    state = dict(learning_state.get("state", learning_state) or {})
    intelligence = dict(state.get("trade_outcome_intelligence", {}) or {})
    trades = list(dict(intelligence.get("outcome_memory", {}) or {}).get("trades", []) or [])
    if not trades:
        trades = list(dict(state.get("loss_memory", {}) or {}).get("recent_outcomes", []) or [])
    return [dict(row or {}) for row in trades[-max(1, int(limit)) :]]
