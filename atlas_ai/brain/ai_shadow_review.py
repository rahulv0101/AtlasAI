from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.memory.jsonl_tail import read_jsonl_tail
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_reporter import EvolutionReporter


class AIShadowReviewPreparer:
    """Restore extended-backtest winners for the existing shadow pipeline."""

    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store
        self.base = Path(store.base)
        self.population_path = self.base / "evolution_population.json"
        self.results_path = self.base / "ai_extended_backtests.jsonl"
        self.archive_path = self.base / "evolution_archive.jsonl"

    def prepare(self, uid: str) -> dict[str, Any]:
        result = self._latest_passed_result(uid)
        if result is None:
            return {"ok": False, "uid": uid, "reason": "passed_extended_backtest_not_found", "executed": False}
        chromosome = self._load_archived(uid)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "archived_candidate_not_found", "executed": False}

        metrics = dict(result.get("result", {}) or {})
        chromosome.backtest_completed = True
        chromosome.backtest_score = float(result.get("score", 0.0) or 0.0)
        chromosome.backtest_trades = int(metrics.get("trades", 0) or 0)
        chromosome.backtest_wins = int(metrics.get("wins", 0) or 0)
        chromosome.backtest_losses = int(metrics.get("losses", 0) or 0)
        chromosome.backtest_win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        chromosome.backtest_profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        chromosome.backtest_sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        chromosome.backtest_expectancy = float(metrics.get("expectancy", 0.0) or 0.0)
        chromosome.backtest_max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        chromosome.promotion_stage = "backtested"
        chromosome.promotion_reason = "extended_backtest_passed_awaiting_shadow_review"
        chromosome.reject_reason = ""
        chromosome.demo_active = False

        population = self._load_population()
        population = [c for c in population if c.uid != chromosome.uid]
        population.append(chromosome)
        population.sort(key=lambda c: c.backtest_score, reverse=True)
        generation = max([c.generation for c in population] + [0])
        self._save_population(population, generation)
        EvolutionReporter(self.base).write_leaderboard(population)

        payload = {
            "ok": True,
            "uid": chromosome.uid,
            "stage": chromosome.promotion_stage,
            "score": chromosome.backtest_score,
            "trades": chromosome.backtest_trades,
            "win_rate": chromosome.backtest_win_rate,
            "profit_factor": chromosome.backtest_profit_factor,
            "max_drawdown": chromosome.backtest_max_drawdown,
            "population_size": len(population),
            "safety": {
                "research_only_restore": True,
                "does_not_execute": True,
                "does_not_promote_live": True,
                "requires_existing_shadow_pipeline": True,
                "requires_promotion_governor": True,
            },
            "executed": False,
        }
        self.store.log_event("evolution", "AI candidate prepared for shadow review.", payload)
        return payload

    def _latest_passed_result(self, uid: str) -> dict[str, Any] | None:
        rows = read_jsonl_tail(
            self.results_path,
            1,
            matcher=lambda row: str(row.get("uid", "")) == str(uid) and bool(row.get("passed_extended_sample", False)),
        )
        return rows[-1] if rows else None

    def _load_archived(self, uid: str) -> StrategyChromosome | None:
        rows = read_jsonl_tail(self.archive_path, 1, matcher=lambda row: str(row.get("uid", "")) == str(uid))
        if not rows:
            return None
        return StrategyChromosome.from_dict(rows[-1])

    def _load_population(self) -> list[StrategyChromosome]:
        if not self.population_path.exists():
            return []
        try:
            payload = json.loads(self.population_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])]

    def _save_population(self, population: list[StrategyChromosome], generation: int) -> None:
        payload = {
            "generation": int(generation),
            "chromosomes": [c.to_dict() for c in population],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.population_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
