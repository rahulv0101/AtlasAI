"""EvolutionReporter: generates leaderboard and evolution logs for UI consumption.

Writes JSON snapshots that the web UI can poll for strategy evolution status.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.strategy_explainer import StrategyExplainer


class EvolutionReporter:
    """Produces leaderboard.json and evolution_log.json artifacts."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_leaderboard(self, population: list[StrategyChromosome]) -> Path:
        """Write leaderboard.json with ranked strategies."""
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_population": len(population),
            "stages": self._stage_counts(population),
            "leaderboard": self._build_leaderboard_rows(population),
            "top_candidate": self._top_candidate(population),
        }
        path = self.output_dir / "leaderboard.json"
        import json
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_evolution_log(self, events: list[dict[str, Any]]) -> Path:
        """Append/write evolution_log.json with generational events."""
        path = self.output_dir / "evolution_log.json"
        import json
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(events),
            "events": events,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def strategy_detail(self, chromosome: StrategyChromosome) -> dict[str, Any]:
        """Return a detailed dict for a single strategy (suitable for API response)."""
        explainer = StrategyExplainer()
        explanation = explainer.explain(chromosome)
        return {
            "uid": chromosome.uid,
            "generation": chromosome.generation,
            "origin": chromosome.origin,
            "parent_uids": list(chromosome.parent_uids),
            "created_at": chromosome.created_at,
            "promotion_stage": chromosome.promotion_stage,
            "promotion_reason": chromosome.promotion_reason,
            "reject_reason": chromosome.reject_reason,
            "signature": chromosome.signature(),
            "genes": {g.name: g.value for g in chromosome.genes},
            "explanation": explanation,
            "compact_summary": explainer.explain_compact(chromosome),
            "backtest": {
                "completed": chromosome.backtest_completed,
                "score": chromosome.backtest_score,
                "trades": chromosome.backtest_trades,
                "wins": chromosome.backtest_wins,
                "losses": chromosome.backtest_losses,
                "win_rate": chromosome.backtest_win_rate,
                "profit_factor": chromosome.backtest_profit_factor,
                "sharpe": chromosome.backtest_sharpe,
                "expectancy": chromosome.backtest_expectancy,
                "max_drawdown": chromosome.backtest_max_drawdown,
            },
            "shadow": {
                "signals": chromosome.shadow_signals,
                "agreement_rate": chromosome.shadow_agreement_rate,
                "degradation_pct": chromosome.shadow_degradation_pct,
                "passed": chromosome.shadow_passed,
            },
            "demo": {
                "active": chromosome.demo_active,
                "trades": chromosome.demo_trades,
                "wins": chromosome.demo_wins,
                "losses": chromosome.demo_losses,
                "win_rate": chromosome.demo_win_rate,
                "pnl_pct": chromosome.demo_pnl_pct,
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_counts(population: list[StrategyChromosome]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in population:
            stage = c.promotion_stage or "unknown"
            counts[stage] = counts.get(stage, 0) + 1
        return counts

    @staticmethod
    def _build_leaderboard_rows(population: list[StrategyChromosome]) -> list[dict[str, Any]]:
        explainer = StrategyExplainer()
        scored = [
            {
                "rank": 0,
                "uid": c.uid,
                "generation": c.generation,
                "stage": c.promotion_stage,
                "backtest_score": c.backtest_score,
                "win_rate": c.backtest_win_rate,
                "profit_factor": c.backtest_profit_factor,
                "sharpe": c.backtest_sharpe,
                "max_drawdown": c.backtest_max_drawdown,
                "trades": c.backtest_trades,
                "compact_summary": explainer.explain_compact(c),
            }
            for c in population
            if c.backtest_completed
        ]
        scored.sort(key=lambda x: x["backtest_score"], reverse=True)
        for i, row in enumerate(scored, start=1):
            row["rank"] = i
        return scored

    @staticmethod
    def _top_candidate(population: list[StrategyChromosome]) -> dict[str, Any] | None:
        candidates = [c for c in population if c.backtest_completed and c.backtest_score > 0]
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.backtest_score)
        return {
            "uid": best.uid,
            "score": best.backtest_score,
            "win_rate": best.backtest_win_rate,
            "stage": best.promotion_stage,
            "compact_summary": StrategyExplainer.explain_compact(best),
        }

