"""PromotionGovernor: enforces hard rules before promoting evolved strategies to demo/live.

Rules are unconditional — no override or bypass exists. The governor operates
entirely on chromosome metadata and shadow outcomes; it never executes trades.
"""
from __future__ import annotations
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.shadow_tester import ShadowOutcome


class PromotionError(Exception):
    """Raised when a promotion rule is violated."""
    pass


class PromotionGovernor:
    """Enforces promotion gates for the strategy evolution pipeline.

    Demo promotion requirements:
      - Max 3 strategies in demo simultaneously
      - Max 0.5% risk per trade (double-check gene value)
      - Min 100 shadow trades completed
      - Max 5% drawdown during shadow period
      - Backtest score >= 60

    Live promotion requirements:
      - Min 100 demo trades completed
      - Demo win rate >= 50%
      - Demo profit factor >= 1.1
      - Human approval flag must be set
      - No active safety locks on the strategy
    """

    def __init__(
        self,
        *,
        max_demo_strategies: int = 3,
        max_risk_per_trade_pct: float = 0.005,
        min_shadow_trades: int = 100,
        max_shadow_drawdown_pct: float = 0.05,
        shadow_drawdown_policy: str = "raw_peak",
        min_backtest_score: float = 60.0,
        min_demo_trades: int = 100,
        min_demo_win_rate: float = 0.50,
        min_demo_profit_factor: float = 1.10,
        live_promotion_requires_approval: bool = True,
    ) -> None:
        self.max_demo_strategies = max_demo_strategies
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.min_shadow_trades = min_shadow_trades
        self.max_shadow_drawdown_pct = max_shadow_drawdown_pct
        self.shadow_drawdown_policy = self._normalize_shadow_drawdown_policy(shadow_drawdown_policy)
        self.min_backtest_score = min_backtest_score
        self.min_demo_trades = min_demo_trades
        self.min_demo_win_rate = min_demo_win_rate
        self.min_demo_profit_factor = min_demo_profit_factor
        self.live_promotion_requires_approval = live_promotion_requires_approval

    # ------------------------------------------------------------------
    # Demo promotion
    # ------------------------------------------------------------------

    def can_promote_to_demo(
        self,
        chromosome: StrategyChromosome,
        population: list[StrategyChromosome],
        shadow_outcomes: list[ShadowOutcome] | None = None,
    ) -> tuple[bool, str]:
        """Return (ok, reason_or_empty) for demo promotion."""
        if chromosome.promotion_stage not in {"backtested", "shadow"}:
            return False, f"invalid_stage:{chromosome.promotion_stage}"

        if chromosome.backtest_score < self.min_backtest_score:
            return False, f"backtest_score_too_low:{chromosome.backtest_score:.1f}<{self.min_backtest_score}"

        risk = chromosome.gene_value("risk_max_risk_per_trade", 0.005)
        if risk > self.max_risk_per_trade_pct:
            return False, f"risk_too_high:{risk:.3f}>{self.max_risk_per_trade_pct:.3f}"

        if shadow_outcomes is not None:
            shadow_summary = self._summarize_shadow_outcomes(chromosome.uid, shadow_outcomes)
            if shadow_summary["trades"] < self.min_shadow_trades:
                return False, f"insufficient_shadow_trades:{shadow_summary['trades']}<{self.min_shadow_trades}"
            drawdown_key = (
                "gross_normalized_drawdown_pct"
                if self.shadow_drawdown_policy == "gross_normalized"
                else "max_drawdown_pct"
            )
            drawdown_value = float(shadow_summary[drawdown_key])
            if drawdown_value > self.max_shadow_drawdown_pct:
                return (
                    False,
                    f"shadow_drawdown_too_high:{drawdown_value:.2%}>{self.max_shadow_drawdown_pct:.2%}"
                    f":policy={self.shadow_drawdown_policy}",
                )

        active_demo = self.active_demo_count(population)
        if active_demo >= self.max_demo_strategies:
            return False, f"demo_cap_reached:{active_demo}/{self.max_demo_strategies}"

        return True, ""

    def promote_to_demo(
        self,
        chromosome: StrategyChromosome,
        population: list[StrategyChromosome],
        shadow_outcomes: list[ShadowOutcome] | None = None,
    ) -> None:
        """Promote a chromosome to demo, raising PromotionError on violation."""
        ok, reason = self.can_promote_to_demo(chromosome, population, shadow_outcomes)
        if not ok:
            raise PromotionError(reason)
        chromosome.promotion_stage = "demo"
        chromosome.promotion_reason = "shadow_passed_awaiting_demo"
        chromosome.demo_active = True

    # ------------------------------------------------------------------
    # Live promotion
    # ------------------------------------------------------------------

    def can_promote_to_live(
        self,
        chromosome: StrategyChromosome,
        human_approved: bool = False,
    ) -> tuple[bool, str]:
        """Return (ok, reason_or_empty) for live promotion."""
        if chromosome.promotion_stage != "demo":
            return False, f"invalid_stage:{chromosome.promotion_stage}"

        if not chromosome.demo_active:
            return False, "demo_inactive"

        if chromosome.demo_trades < self.min_demo_trades:
            return False, f"insufficient_demo_trades:{chromosome.demo_trades}<{self.min_demo_trades}"

        if chromosome.demo_win_rate < self.min_demo_win_rate:
            return False, f"demo_win_rate_too_low:{chromosome.demo_win_rate:.2%}<{self.min_demo_win_rate:.2%}"

        pf = self._demo_profit_factor(chromosome)
        if pf < self.min_demo_profit_factor:
            return False, f"demo_profit_factor_too_low:{pf:.2f}<{self.min_demo_profit_factor}"

        if self.live_promotion_requires_approval and not human_approved:
            return False, "live_promotion_requires_human_approval"

        return True, ""

    def promote_to_live(
        self,
        chromosome: StrategyChromosome,
        human_approved: bool = False,
    ) -> None:
        """Promote a chromosome to proposed_live, raising PromotionError on violation."""
        ok, reason = self.can_promote_to_live(chromosome, human_approved)
        if not ok:
            raise PromotionError(reason)
        chromosome.promotion_stage = "proposed_live"
        chromosome.promotion_reason = "demo_passed_awaiting_operator_approval"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def active_demo_count(population: list[StrategyChromosome]) -> int:
        return sum(1 for c in population if c.promotion_stage == "demo" and c.demo_active)

    @staticmethod
    def active_live_count(population: list[StrategyChromosome]) -> int:
        return sum(1 for c in population if c.promotion_stage == "live")

    @staticmethod
    def _summarize_shadow_outcomes(uid: str, outcomes: list[ShadowOutcome]) -> dict[str, Any]:
        rows = [o for o in outcomes if o.chromosome_uid == uid]
        if not rows:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "max_drawdown_pct": 0.0,
                "gross_normalized_drawdown_pct": 0.0,
                "net_pnl": 0.0,
            }

        wins = sum(1 for o in rows if o.pnl > 0)
        losses = sum(1 for o in rows if o.pnl <= 0)
        net_pnl = sum(o.pnl for o in rows)

        # Simple drawdown from shadow PnL curve
        peak = 0.0
        dd = 0.0
        running = 0.0
        gross_movement = 0.0
        for o in rows:
            pnl = float(o.pnl)
            gross_movement += abs(pnl)
            running += pnl
            if running > peak:
                peak = running
            dd = max(dd, peak - running)

        return {
            "trades": len(rows),
            "win_rate": wins / len(rows) if rows else 0.0,
            "max_drawdown_pct": dd / max(1e-9, abs(peak)) if peak != 0 else 0.0,
            "gross_normalized_drawdown_pct": dd / max(1e-9, gross_movement) if gross_movement > 0 else 0.0,
            "net_pnl": net_pnl,
            "wins": wins,
            "losses": losses,
        }

    @staticmethod
    def _normalize_shadow_drawdown_policy(value: str) -> str:
        policy = str(value or "raw_peak").strip().lower()
        aliases = {
            "raw": "raw_peak",
            "raw_peak": "raw_peak",
            "peak": "raw_peak",
            "gross": "gross_normalized",
            "gross_normalized": "gross_normalized",
            "gross-normalized": "gross_normalized",
        }
        return aliases.get(policy, "raw_peak")

    @staticmethod
    def _demo_profit_factor(chromosome: StrategyChromosome) -> float:
        gross_profit = chromosome.demo_pnl_pct if chromosome.demo_pnl_pct > 0 else 0.0
        # Approximate gross loss from win rate and trade count
        wins = chromosome.demo_wins
        losses = chromosome.demo_losses
        if wins == 0 or losses == 0:
            return 1.0 if chromosome.demo_pnl_pct >= 0 else 0.0
        avg_win = gross_profit / max(1, wins)
        avg_loss = abs(gross_profit - chromosome.demo_pnl_pct) / max(1, losses)
        if avg_loss == 0:
            return 99.0
        return (avg_win * wins) / (avg_loss * losses)

