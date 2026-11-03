from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_payload()
        return self._normalize(payload)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(payload)
        self.path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        return normalized

    def update_from_reviews(self, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        current = self.load()
        scores = dict(current.get("scores", {}) or {})
        evidence = dict(current.get("evidence", {}) or {})
        for review in list(reviews or []):
            outcome = str(review.get("outcome", "") or "")
            archetypes = list(dict(review.get("strategy_archetype_review", {}) or {}).get("archetypes", []) or [])
            delta = self._delta_for_outcome(outcome)
            for archetype in archetypes:
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
            "schema": "atlas_ai_strategy_preferences_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
            "evidence": evidence,
            "ranking": [{"archetype": key, "score": round(float(score), 4)} for key, score in ranked],
            "policy": {
                "preference_only": True,
                "does_not_mutate_strategy": True,
                "does_not_mutate_risk": True,
                "does_not_execute": True,
            },
        }
        return self.save(payload)

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
        if outcome == "validated_prepare":
            return 0.18
        if outcome == "validated_wait":
            return 0.08
        if outcome == "possible_overcaution":
            return -0.06
        if outcome == "prepare_needs_review":
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
