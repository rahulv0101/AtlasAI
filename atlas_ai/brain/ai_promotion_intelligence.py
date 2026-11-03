from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.brain.ai_strategy_preferences import AIStrategyPreferences
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.promotion_governor import PromotionGovernor


@dataclass(frozen=True)
class PromotionRecommendation:
    uid: str
    origin: str
    stage: str
    recommendation: str
    confidence: float
    reasons: list[str]
    blockers: list[str]
    preference_score: float
    safety: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "origin": self.origin,
            "stage": self.stage,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "blockers": list(self.blockers),
            "preference_score": self.preference_score,
            "safety": dict(self.safety),
        }


class AIPromotionIntelligence:
    """Recommend strategy research/promotion actions without promoting directly."""

    def __init__(
        self,
        storage_dir: str | Path = "storage",
        *,
        governor: PromotionGovernor | None = None,
    ) -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.governor = governor or PromotionGovernor()
        self.preferences = AIStrategyPreferences(self.base)
        self.path = self.base / "ai_promotion_recommendations.json"

    def evaluate(
        self,
        population: list[StrategyChromosome],
        *,
        strategy_preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        preferences = strategy_preferences or self.preferences.load()
        recommendations = [
            self._evaluate_chromosome(c, population, preferences).to_dict()
            for c in list(population or [])
        ]
        recommendations.sort(key=lambda row: (float(row.get("confidence", 0.0)), float(row.get("preference_score", 0.0))), reverse=True)
        payload = {
            "schema": "atlas_ai_promotion_intelligence_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "population_size": len(population or []),
            "recommendations": recommendations,
            "top_recommendation": recommendations[0] if recommendations else {},
            "safety": {
                "recommendation_only": True,
                "does_not_promote": True,
                "does_not_execute": True,
                "requires_promotion_governor": True,
                "requires_human_approval_for_live": True,
            },
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def load_latest(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _evaluate_chromosome(
        self,
        chromosome: StrategyChromosome,
        population: list[StrategyChromosome],
        preferences: dict[str, Any],
    ) -> PromotionRecommendation:
        pref_score = self._preference_score(chromosome, preferences)
        reasons: list[str] = []
        blockers: list[str] = []
        recommendation = "keep_researching"
        confidence = 0.35 + max(0.0, pref_score) * 0.2

        if chromosome.reject_reason or chromosome.promotion_stage == "rejected":
            recommendation = "reject_or_archive"
            blockers.append(chromosome.reject_reason or "marked_rejected")
            confidence = 0.8
        elif not chromosome.backtest_completed and chromosome.promotion_stage not in {"demo", "proposed_live", "live"}:
            recommendation = "run_offline_backtest"
            reasons.append("backtest_pending")
            confidence = 0.62 + max(0.0, pref_score) * 0.1
        elif chromosome.backtest_score <= 0 and chromosome.promotion_stage not in {"demo", "proposed_live", "live"}:
            recommendation = "reject_or_archive"
            blockers.append(chromosome.reject_reason or "backtest_score_nonpositive")
            confidence = 0.76
        elif chromosome.promotion_stage in {"backtested", "shadow"}:
            ok, reason = self.governor.can_promote_to_demo(chromosome, population, shadow_outcomes=None)
            if ok:
                recommendation = "eligible_for_shadow_or_demo_review"
                reasons.append(f"backtest_score:{chromosome.backtest_score:.1f}")
                if pref_score > 0:
                    reasons.append(f"ai_preference_positive:{pref_score:.2f}")
                confidence = min(0.95, 0.58 + chromosome.backtest_score / 250.0 + max(0.0, pref_score) * 0.15)
            else:
                recommendation = "continue_shadow_or_retest"
                blockers.append(reason)
                confidence = 0.55
        elif chromosome.promotion_stage == "demo":
            ok, reason = self.governor.can_promote_to_live(chromosome, human_approved=False)
            if ok:
                recommendation = "operator_review_for_live_proposal"
                reasons.append("demo_metrics_meet_thresholds")
                confidence = 0.72 + max(0.0, pref_score) * 0.1
            else:
                recommendation = "continue_demo_validation"
                blockers.append(reason)
                confidence = 0.58
        elif chromosome.promotion_stage in {"proposed_live", "live"}:
            recommendation = "monitor_only"
            reasons.append(f"stage:{chromosome.promotion_stage}")
            confidence = 0.5

        return PromotionRecommendation(
            uid=chromosome.uid,
            origin=chromosome.origin,
            stage=chromosome.promotion_stage,
            recommendation=recommendation,
            confidence=round(max(0.05, min(0.99, confidence)), 4),
            reasons=reasons,
            blockers=blockers,
            preference_score=round(pref_score, 4),
            safety={
                "recommendation_only": True,
                "does_not_promote": True,
                "does_not_execute": True,
                "execution_gate_required": True,
            },
        )

    @staticmethod
    def _preference_score(chromosome: StrategyChromosome, preferences: dict[str, Any]) -> float:
        scores = dict(preferences.get("scores", {}) or {})
        origin = str(chromosome.origin or "")
        for archetype, score in scores.items():
            if str(archetype) in origin:
                try:
                    return float(score)
                except Exception:
                    return 0.0
        return 0.0
