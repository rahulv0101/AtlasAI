from __future__ import annotations

import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.brain.ai_decision_journal import AIDecisionJournal, summarize_decision_record
from atlas_ai.brain.ai_strategy_preferences import AIStrategyPreferences
from atlas_ai.brain.ai_review_ledger import AIReviewLedger
from atlas_ai.memory.jsonl_tail import read_jsonl_tail


class AIDecisionReviewer:
    """Review AI decisions against later outcomes and blocked events."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.journal = AIDecisionJournal(self.base)
        self.preferences = AIStrategyPreferences(self.base)
        self.path = self.base / "ai_decision_reviews.jsonl"
        self.ledger = AIReviewLedger(self.base)

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
        appended = 0
        for review in reviews:
            fingerprint = hashlib.sha256(json.dumps(
                {key: value for key, value in review.items() if key != "reviewed_at"},
                sort_keys=True,
            ).encode("utf-8")).hexdigest()
            with self.ledger.transaction() as connection:
                previous = connection.execute(
                    "SELECT fingerprint FROM reviews WHERE decision_id = ?", (review["decision_id"],)
                ).fetchone()
                if previous and previous[0] == fingerprint:
                    continue
                self._append_review(review)
                connection.execute(
                    "INSERT OR REPLACE INTO reviews VALUES (?, ?, ?)",
                    (review["decision_id"], fingerprint, json.dumps(review, sort_keys=True)),
                )
                appended += 1
        preferences = self.preferences.update_from_reviews(reviews) if reviews else self.preferences.load()
        return {
            "schema": "atlas_ai_decision_review_batch_v1",
            "reviewed": len(reviews),
            "new_or_changed": appended,
            "outcomes_seen": len(outcomes),
            "events_seen": len(events),
            "latest_review": reviews[-1] if reviews else {},
            "summary": self._summary(reviews),
            "strategy_preferences": AIStrategyPreferences.summarize(preferences),
        }

    def recent_reviews(self, limit: int = 5) -> list[dict[str, Any]]:
        current = self.ledger.recent(limit)
        if current:
            return current
        # Preserve old records for audit, but never present their conclusions as verified.
        return [
            {**row, "legacy_unverified": True, "outcome": "unverified_legacy",
             "confidence_quality": "unverified", "blocker_quality": "unverified",
             "lesson": "Legacy review: outcome attribution and duplicate counting were not verified."}
            for row in read_jsonl_tail(self.path, limit)
        ]

    def _review_record(self, record: dict[str, Any], outcomes: list[dict[str, Any]], events: list[dict[str, Any]]) -> dict[str, Any]:
        summary = summarize_decision_record(record)
        action = summary["action"]
        latest_outcome = self._matched_outcome(record, summary, outcomes)
        outcome_label = str(latest_outcome.get("outcome", "") or "").lower()
        blocked_events = [
            e for e in events
            if str(e.get("type", "") or "").lower() in {"risk", "blocked", "paper_filtered"}
        ]
        blocker_quality = self._blocker_quality(summary, blocked_events)
        decision_outcome = self._decision_outcome(action, outcome_label, blocker_quality, bool(blocked_events))
        archetypes = list(summary.get("strategy_archetypes", []) or [])
        matched_archetype = str(latest_outcome.get("archetype", "") or "")
        if matched_archetype not in archetypes:
            matched_archetype = ""
        return {
            "schema": "atlas_ai_decision_review_v2",
            "decision_id": summary["decision_id"],
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
            "outcome_evidence": {
                "status": "linked_outcome" if latest_outcome else "unverified",
                "ticket": str(latest_outcome.get("ticket", "") or ""),
                "time": str(latest_outcome.get("time", "") or ""),
                "decision_id": str(latest_outcome.get("decision_id", "") or ""),
            },
            "blocker_quality": blocker_quality,
            "confidence_quality": self._confidence_quality(summary["confidence"], outcome_label, decision_outcome),
            "strategy_archetype_review": {
                "archetypes": [matched_archetype] if matched_archetype else archetypes,
                # An executed strategy does not validate every invented archetype.
                "status": "matched_executed_archetype" if matched_archetype else ("pending_outcome" if archetypes else "none"),
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
    def _matched_outcome(record: dict[str, Any], summary: dict[str, Any],
                         outcomes: list[dict[str, Any]]) -> dict[str, Any]:
        if summary["action"] != "prepare_trade_candidate":
            return {}
        audit = dict(dict(record.get("trace", {}) or {}).get("audit", {}) or {})
        execution_id = str(audit.get("execution_decision_id", "") or "")
        if not execution_id:
            return {}
        try:
            decision_time = datetime.fromisoformat(summary["recorded_at"].replace("Z", "+00:00"))
            matches = []
            for row in outcomes:
                if str(row.get("decision_id", "")) != execution_id or not row.get("ticket"):
                    continue
                if row.get("outcome") not in {"win", "loss"}:
                    continue
                closed = datetime.fromisoformat(str(row.get("time", "")).replace("Z", "+00:00"))
                if closed > decision_time:
                    matches.append(row)
            # Ambiguous attribution must not become evidence.
            return matches[0] if len(matches) == 1 else {}
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _blocker_quality(summary: dict[str, Any], blocked_events: list[dict[str, Any]]) -> str:
        blocker = str(summary.get("primary_blocker", "") or "").lower()
        if not blocker or blocker == "none":
            return "no_blocker_claimed"
        if any(blocker in str(e.get("message", "") or "").lower() for e in blocked_events):
            return "blocker_text_observed_not_outcome_validation"
        if blocked_events and any(token in str(blocked_events[-1].get("message", "") or "").lower() for token in blocker.split()[:2]):
            return "similar_blocker_text_observed_not_outcome_validation"
        return "unconfirmed"

    @staticmethod
    def _decision_outcome(action: str, outcome_label: str, blocker_quality: str, has_blocked_events: bool) -> str:
        if action in {"wait_for_confirmation", "wait_and_reduce_frequency", "hold_and_observe"}:
            # A veto is not a counterfactual outcome of the rejected candidate.
            return "pending"
        if action == "prepare_trade_candidate":
            if outcome_label == "win":
                return "linked_prepare_win"
            if outcome_label == "loss":
                return "linked_prepare_loss"
            return "pending"
        return "pending"

    @staticmethod
    def _confidence_quality(confidence: float, outcome_label: str, decision_outcome: str) -> str:
        return "not_calibrated"

    @staticmethod
    def _lesson(action: str, decision_outcome: str, blocker: str) -> str:
        if decision_outcome in {"linked_prepare_win", "linked_prepare_loss"}:
            return "A trade outcome is linked by execution decision ID and later close time; this does not validate invented archetypes or calibrate confidence."
        return "Outcome unverified: a blocker records why Atlas waited, not whether waiting improved the result. No preference credit awarded."

    @staticmethod
    def _summary(reviews: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(r.get("outcome", "pending")) for r in reviews)
        return {
            "linked_prepare_win": counts.get("linked_prepare_win", 0),
            "linked_prepare_loss": counts.get("linked_prepare_loss", 0),
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
