"""EvolutionDrawdownGuard: pauses evolved demo strategies if drawdown exceeds threshold.

Operates on demo outcome history. Never executes trades.
"""
from __future__ import annotations
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.shadow_tester import ShadowOutcome


class EvolutionDrawdownGuard:
    """Monitors demo outcomes and pauses strategies that exceed drawdown limits."""

    def __init__(
        self,
        *,
        max_demo_drawdown_pct: float = 0.05,
        max_shadow_drawdown_pct: float = 0.05,
        pause_duration_hours: float = 24.0,
    ) -> None:
        self.max_demo_drawdown_pct = max_demo_drawdown_pct
        self.max_shadow_drawdown_pct = max_shadow_drawdown_pct
        self.pause_duration_hours = pause_duration_hours

    def check_drawdown(
        self,
        chromosome: StrategyChromosome,
        outcomes: list[ShadowOutcome],
        stage: str = "demo",
    ) -> tuple[bool, str]:
        """Return (ok, reason_or_empty) for the given chromosome and outcomes."""
        if not outcomes:
            return True, ""

        limit = (
            self.max_demo_drawdown_pct
            if stage == "demo"
            else self.max_shadow_drawdown_pct
        )

        dd_pct = self._compute_max_drawdown_pct(outcomes)
        if dd_pct > limit:
            return (
                False,
                f"{stage}_drawdown_exceeded:{dd_pct:.2%}>{limit:.2%}",
            )
        return True, ""

    def pause_strategy(self, chromosome: StrategyChromosome, reason: str) -> None:
        """Set strategy to inactive and record why."""
        chromosome.demo_active = False
        chromosome.promotion_reason = f"paused:{reason}"

    def resume_strategy(self, chromosome: StrategyChromosome) -> None:
        """Re-activate a paused strategy (operator or automated decision)."""
        chromosome.demo_active = True
        chromosome.promotion_reason = "resumed_by_operator_or_guard"

    @staticmethod
    def _compute_max_drawdown_pct(outcomes: list[ShadowOutcome]) -> float:
        """Compute max drawdown from an ordered list of outcomes."""
        peak = 0.0
        running = 0.0
        max_dd = 0.0
        for o in outcomes:
            running += o.pnl
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        if peak <= 0:
            return 0.0
        return max_dd / peak

    @staticmethod
    def summarize_outcomes(outcomes: list[ShadowOutcome]) -> dict[str, Any]:
        """Return summary stats for a list of outcomes."""
        if not outcomes:
            return {
                "trades": 0,
                "net_pnl": 0.0,
                "win_rate": 0.0,
                "max_drawdown_pct": 0.0,
            }
        wins = sum(1 for o in outcomes if o.pnl > 0)
        net_pnl = sum(o.pnl for o in outcomes)
        dd = EvolutionDrawdownGuard._compute_max_drawdown_pct(outcomes)
        return {
            "trades": len(outcomes),
            "net_pnl": net_pnl,
            "win_rate": wins / len(outcomes),
            "max_drawdown_pct": dd,
        }

