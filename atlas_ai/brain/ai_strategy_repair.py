from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.gold_validator import GoldValidator


class AIStrategyRepairer:
    """Create bounded repair variants from failed shadow-review candidates."""

    def __init__(self, store: KnowledgeStore, validator: GoldValidator | None = None) -> None:
        self.store = store
        self.base = Path(store.base)
        self.population_path = self.base / "evolution_population.json"
        self.output_path = self.base / "ai_strategy_repairs.json"
        self.validator = validator or GoldValidator()

    def repair_shadow_failure(self, uid: str, *, limit: int = 3) -> dict[str, Any]:
        population, generation = self._load_population()
        parent = next((c for c in population if c.uid == uid), None)
        if parent is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}
        if parent.shadow_passed:
            return {"ok": False, "uid": uid, "reason": "candidate_not_shadow_failed", "executed": False}

        variants: list[StrategyChromosome] = []
        for idx, spec in enumerate(self._repair_specs(parent)[: max(1, int(limit))]):
            child = parent.clone()
            child.uid = self._repair_uid(parent.uid, idx, spec)
            child.generation = parent.generation + 1
            child.origin = f"ai_repair:{spec['name']}"
            child.parent_uids = [parent.uid]
            child.created_at = datetime.now(timezone.utc).isoformat()
            child.promotion_stage = "untested"
            child.promotion_reason = "shadow_failure_repair_awaiting_offline_backtest"
            child.reject_reason = ""
            child.backtest_completed = False
            child.backtest_score = 0.0
            child.shadow_signals = 0
            child.shadow_agreement_rate = 0.0
            child.shadow_degradation_pct = 0.0
            child.shadow_passed = False
            for key, value in dict(spec.get("genes", {}) or {}).items():
                child.set_gene(key, float(value))
            ok, reason = self.validator.validate_silent(child)
            if not ok:
                child.reject_reason = f"repair_validation_failed:{reason}"
                continue
            variants.append(child)

        existing = {c.uid for c in population}
        for variant in variants:
            if variant.uid not in existing:
                population.append(variant)
                existing.add(variant.uid)
        self._save_population(population, max(generation, parent.generation + 1))
        payload = {
            "schema": "atlas_ai_strategy_repairs_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "parent_uid": parent.uid,
            "failure": {
                "promotion_reason": parent.promotion_reason,
                "shadow_signals": parent.shadow_signals,
                "shadow_win_rate": parent.shadow_agreement_rate,
                "shadow_degradation_pct": parent.shadow_degradation_pct,
            },
            "variants": [
                {
                    "uid": child.uid,
                    "origin": child.origin,
                    "stage": child.promotion_stage,
                    "genes": {gene.name: gene.value for gene in child.genes},
                }
                for child in variants
            ],
            "safety": {
                "proposal_only": True,
                "does_not_execute": True,
                "does_not_promote": True,
                "requires_backtest": True,
                "requires_shadow_review": True,
            },
            "executed": False,
        }
        self.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.store.log_event("evolution", "AI strategy repair variants generated.", payload)
        return {"ok": True, **payload}

    def micro_repair_below_threshold(self, uid: str, *, limit: int = 4) -> dict[str, Any]:
        population, generation = self._load_population()
        parent = next((c for c in population if c.uid == uid), None)
        if parent is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}
        if parent.backtest_score <= 0 or parent.backtest_score >= 60.0:
            return {"ok": False, "uid": uid, "reason": "candidate_not_near_threshold", "executed": False}

        variants: list[StrategyChromosome] = []
        for idx, spec in enumerate(self._micro_specs(parent)[: max(1, int(limit))]):
            child = parent.clone()
            child.uid = self._repair_uid(parent.uid, idx, {"micro": spec})
            child.generation = parent.generation + 1
            child.origin = f"ai_micro_repair:{spec['name']}"
            child.parent_uids = [parent.uid]
            child.created_at = datetime.now(timezone.utc).isoformat()
            child.promotion_stage = "untested"
            child.promotion_reason = "near_threshold_micro_repair_awaiting_offline_backtest"
            child.reject_reason = ""
            child.backtest_completed = False
            child.backtest_score = 0.0
            child.shadow_signals = 0
            child.shadow_agreement_rate = 0.0
            child.shadow_degradation_pct = 0.0
            child.shadow_passed = False
            for key, value in dict(spec.get("genes", {}) or {}).items():
                child.set_gene(key, float(value))
            ok, reason = self.validator.validate_silent(child)
            if not ok:
                child.reject_reason = f"micro_repair_validation_failed:{reason}"
                continue
            variants.append(child)

        existing = {c.uid for c in population}
        for variant in variants:
            if variant.uid not in existing:
                population.append(variant)
                existing.add(variant.uid)
        self._save_population(population, max(generation, parent.generation + 1))
        payload = {
            "schema": "atlas_ai_strategy_micro_repairs_v1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "parent_uid": parent.uid,
            "parent_score": parent.backtest_score,
            "variants": [
                {
                    "uid": child.uid,
                    "origin": child.origin,
                    "stage": child.promotion_stage,
                    "genes": {gene.name: gene.value for gene in child.genes},
                }
                for child in variants
            ],
            "safety": {
                "proposal_only": True,
                "does_not_execute": True,
                "does_not_promote": True,
                "requires_backtest": True,
                "requires_shadow_review": True,
            },
            "executed": False,
        }
        self.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.store.log_event("evolution", "AI strategy micro-repair variants generated.", payload)
        return {"ok": True, **payload}

    @staticmethod
    def _repair_specs(parent: StrategyChromosome) -> list[dict[str, Any]]:
        base_conf = parent.gene_value("risk_min_confidence", 0.55)
        base_body = parent.gene_value("confirm_min_body_ratio", 0.35)
        base_vol = parent.gene_value("confirm_volume_min_ratio", 0.85)
        base_sl = parent.gene_value("exit_sl_atr_multiple", 1.6)
        base_tp = parent.gene_value("exit_tp_atr_multiple", 2.8)
        base_hold = parent.gene_value("exit_max_hold_bars", 8)
        return [
            {
                "name": "higher_confirmation_shorter_hold",
                "genes": {
                    "risk_min_confidence": min(0.85, base_conf + 0.08),
                    "confirm_min_body_ratio": min(0.60, base_body + 0.08),
                    "confirm_volume_min_ratio": min(1.30, base_vol + 0.12),
                    "exit_max_hold_bars": max(3.0, base_hold - 3.0),
                },
            },
            {
                "name": "tighter_risk_reward_filter",
                "genes": {
                    "risk_min_confidence": min(0.85, base_conf + 0.06),
                    "exit_sl_atr_multiple": max(0.8, base_sl * 0.9),
                    "exit_tp_atr_multiple": max(base_sl * 1.35, base_tp * 0.9),
                    "risk_max_trades_per_day": min(parent.gene_value("risk_max_trades_per_day", 8), 4.0),
                },
            },
            {
                "name": "trend_strength_guard",
                "genes": {
                    "risk_min_confidence": min(0.85, base_conf + 0.10),
                    "confirm_min_trend_strength": min(0.45, parent.gene_value("confirm_min_trend_strength", 0.22) + 0.08),
                    "regime_prefer_trend": 1.0,
                    "confirm_volume_min_ratio": min(1.30, base_vol + 0.08),
                },
            },
        ]

    @staticmethod
    def _micro_specs(parent: StrategyChromosome) -> list[dict[str, Any]]:
        conf = parent.gene_value("risk_min_confidence", 0.55)
        body = parent.gene_value("confirm_min_body_ratio", 0.35)
        vol = parent.gene_value("confirm_volume_min_ratio", 0.85)
        trend = parent.gene_value("confirm_min_trend_strength", 0.22)
        hold = parent.gene_value("exit_max_hold_bars", 8)
        tp = parent.gene_value("exit_tp_atr_multiple", 2.8)
        return [
            {
                "name": "tiny_confidence_lift",
                "genes": {
                    "risk_min_confidence": min(0.85, conf + 0.02),
                    "confirm_min_body_ratio": min(0.60, body + 0.02),
                },
            },
            {
                "name": "tiny_volume_lift",
                "genes": {
                    "confirm_volume_min_ratio": min(1.30, vol + 0.04),
                    "risk_min_confidence": min(0.85, conf + 0.01),
                },
            },
            {
                "name": "tiny_hold_reduction",
                "genes": {
                    "exit_max_hold_bars": max(3.0, hold - 1.0),
                    "risk_min_confidence": min(0.85, conf + 0.01),
                },
            },
            {
                "name": "tiny_trend_and_reward_lift",
                "genes": {
                    "confirm_min_trend_strength": min(0.45, trend + 0.02),
                    "exit_tp_atr_multiple": min(5.0, tp + 0.1),
                },
            },
        ]

    @staticmethod
    def _repair_uid(parent_uid: str, idx: int, spec: dict[str, Any]) -> str:
        digest = hashlib.sha1(f"{parent_uid}:{idx}:{spec}".encode("utf-8")).hexdigest()[:12]
        return f"repair_{digest}"

    def _load_population(self) -> tuple[list[StrategyChromosome], int]:
        if not self.population_path.exists():
            return [], 0
        try:
            payload = json.loads(self.population_path.read_text(encoding="utf-8"))
        except Exception:
            return [], 0
        return [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])], int(payload.get("generation", 0) or 0)

    def _save_population(self, population: list[StrategyChromosome], generation: int) -> None:
        payload = {
            "generation": int(generation),
            "chromosomes": [c.to_dict() for c in population],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.population_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
