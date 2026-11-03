from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from atlas_ai.brain.ai_review_ledger import AIReviewLedger


DEFAULT_ARCHETYPES = [
    "trend_pullback_continuation",
    "range_reversion_filter",
    "low_frequency_volatility_guard",
    "balanced_confirmation",
]


class AIStrategyPreferences:
    """Preference memory for AI-invented strategy archetypes."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "ai_strategy_preferences.json"
        self.ledger = AIReviewLedger(self.base)

    def load(self) -> dict[str, Any]:
        # Legacy scores mixed duplicate decisions and unlinked outcomes. Keep the
        # original JSON untouched for audit; only correlated v2 evidence can score.
        return self._from_evidence(self.ledger.preference_evidence())

    def update_from_reviews(self, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = []
        for review in reviews or []:
            link = dict(review.get("outcome_evidence", {}) or {})
            archetype_review = dict(review.get("strategy_archetype_review", {}) or {})
            if (review.get("schema") != "atlas_ai_decision_review_v2"
                    or not review.get("decision_id")
                    or link.get("status") != "linked_outcome"
                    or not all(link.get(key) for key in ("ticket", "time", "decision_id"))
                    or archetype_review.get("status") != "matched_executed_archetype"
                    or not self._delta_for_outcome(str(review.get("outcome", "")))):
                continue
            eligible.append(review)
        if eligible:
            with self.ledger.transaction() as connection:
                for review in eligible:
                    link = review["outcome_evidence"]
                    evidence_id = json.dumps([link["decision_id"], str(link["ticket"])])
                    connection.execute(
                        "INSERT OR IGNORE INTO preference_evidence VALUES (?, ?)",
                        (evidence_id, json.dumps(review, sort_keys=True)),
                    )
        return self.load()

    def _from_evidence(self, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        scores = dict(self._default_payload()["scores"])
        evidence: dict[str, Any] = {}
        for review in reviews:
            outcome = str(review.get("outcome", "") or "")
            archetypes = list(dict(review.get("strategy_archetype_review", {}) or {}).get("archetypes", []) or [])
            delta = self._delta_for_outcome(outcome)
            for archetype in set(archetypes):
                key = str(archetype or "").strip()
                if not key:
                    continue
                scores[key] = max(-1.0, min(1.0, float(scores.get(key, 0.0) or 0.0) + delta))
                row = dict(evidence.get(key, {}) or {})
                row["samples"] = int(row.get("samples", 0) or 0) + 1
                row[outcome or "pending"] = int(row.get(outcome or "pending", 0) or 0) + 1
                row["last_outcome"] = outcome or "pending"
                evidence[key] = row

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        payload = {
            "schema": "atlas_ai_strategy_preferences_v2",
            "updated_at": max((str(r.get("reviewed_at", "")) for r in reviews), default=""),
            "scores": scores,
            "evidence": evidence,
            "ranking": [{"archetype": key, "score": round(float(score), 4)} for key, score in ranked],
            "policy": {
                "preference_only": True,
                "does_not_mutate_strategy": True,
                "does_not_mutate_risk": True,
                "does_not_execute": True,
            },
            "evidence_status": "linked_outcomes_only",
            "legacy_scores_excluded": self.path.exists(),
        }
        return payload

    def rank(self, candidates: list[str]) -> list[str]:
        payload = self.load()
        scores = dict(payload.get("scores", {}) or {})
        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda item: (float(scores.get(item[1], 0.0) or 0.0), -item[0]), reverse=True)
        return [name for _, name in indexed]

    @staticmethod
    def summarize(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = AIStrategyPreferences._normalize(payload)
        ranking = list(normalized.get("ranking", []) or [])
        top = dict(ranking[0]) if ranking else {}
        return {
            "top_archetype": str(top.get("archetype", "none") or "none"),
            "top_score": float(top.get("score", 0.0) or 0.0),
            "ranking": ranking[:4],
            "policy": dict(normalized.get("policy", {}) or {}),
        }

    @staticmethod
    def _delta_for_outcome(outcome: str) -> float:
        if outcome == "linked_prepare_win":
            return 0.18
        if outcome == "linked_prepare_loss":
            return -0.18
        return 0.0

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {
            "schema": "atlas_ai_strategy_preferences_v1",
            "updated_at": "",
            "scores": {name: 0.0 for name in DEFAULT_ARCHETYPES},
            "evidence": {},
            "ranking": [{"archetype": name, "score": 0.0} for name in DEFAULT_ARCHETYPES],
            "policy": {
                "preference_only": True,
                "does_not_mutate_strategy": True,
                "does_not_mutate_risk": True,
                "does_not_execute": True,
            },
        }

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        base = AIStrategyPreferences._default_payload()
        raw = dict(payload or {})
        scores = {**dict(base["scores"]), **dict(raw.get("scores", {}) or {})}
        ranking = list(raw.get("ranking", []) or [])
        if not ranking:
            ranking = [{"archetype": key, "score": round(float(value), 4)} for key, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
        return {
            "schema": "atlas_ai_strategy_preferences_v1",
            "updated_at": str(raw.get("updated_at", "") or ""),
            "scores": {str(k): float(v or 0.0) for k, v in scores.items()},
            "evidence": dict(raw.get("evidence", {}) or {}),
            "ranking": ranking,
            "policy": {**dict(base["policy"]), **dict(raw.get("policy", {}) or {})},
        }
