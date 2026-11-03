"""Fitness scoring for evolved strategies.

Scores backtest results using a composite metric that balances win rate,
profit factor, Sharpe, expectancy, and drawdown. No trade execution here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome


@dataclass
class BacktestResult:
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    sharpe: float
    expectancy: float
    max_drawdown: float
    total_return: float
    average_win: float
    average_loss: float
    average_hold_bars: float


class FitnessScorer:
    """Composite fitness scorer with configurable weights.

    The composite score is normalized to 0-100 for easy interpretation.
    """

    def __init__(
        self,
        *,
        weight_win_rate: float = 0.25,
        weight_profit_factor: float = 0.20,
        weight_sharpe: float = 0.20,
        weight_expectancy: float = 0.15,
        weight_drawdown: float = 0.20,
        min_trades: int = 30,
        min_win_rate: float = 0.50,
        min_profit_factor: float = 1.10,
        max_drawdown_pct: float = 0.08,
    ) -> None:
        self.weight_win_rate = weight_win_rate
        self.weight_profit_factor = weight_profit_factor
        self.weight_sharpe = weight_sharpe
        self.weight_expectancy = weight_expectancy
        self.weight_drawdown = weight_drawdown
        self.min_trades = min_trades
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor
        self.max_drawdown_pct = max_drawdown_pct

    def score(self, result: BacktestResult) -> tuple[float, str]:
        """Return (composite_score_0_100, reason_or_empty)."""
        if result.trades < self.min_trades:
            return 0.0, f"insufficient_trades:{result.trades}<{self.min_trades}"
        if result.win_rate < self.min_win_rate:
            return 0.0, f"win_rate_too_low:{result.win_rate:.2f}<{self.min_win_rate}"
        if result.profit_factor < self.min_profit_factor:
            return 0.0, f"profit_factor_too_low:{result.profit_factor:.2f}<{self.min_profit_factor}"
        if result.max_drawdown > self.max_drawdown_pct:
            return 0.0, f"drawdown_too_high:{result.max_drawdown:.2%}>{self.max_drawdown_pct:.2%}"

        # Normalize each component to 0-100
        win_rate_score = min(100.0, result.win_rate * 100.0)
        pf_score = min(100.0, (result.profit_factor / 2.0) * 100.0)  # PF=2.0 → 100
        sharpe_score = min(100.0, max(0.0, result.sharpe * 25.0))  # Sharpe=4.0 → 100
        exp_score = min(100.0, max(0.0, (result.expectancy / 0.005) * 100.0))  # 0.5% → 100
        dd_score = max(0.0, 100.0 - (result.max_drawdown / self.max_drawdown_pct) * 100.0)

        composite = (
            win_rate_score * self.weight_win_rate
            + pf_score * self.weight_profit_factor
            + sharpe_score * self.weight_sharpe
            + exp_score * self.weight_expectancy
            + dd_score * self.weight_drawdown
        )
        return round(composite, 2), ""

    def update_chromosome(self, chromosome: StrategyChromosome, result: BacktestResult) -> None:
        chromosome.backtest_trades = result.trades
        chromosome.backtest_wins = result.wins
        chromosome.backtest_losses = result.losses
        chromosome.backtest_win_rate = result.win_rate
        chromosome.backtest_profit_factor = result.profit_factor
        chromosome.backtest_sharpe = result.sharpe
        chromosome.backtest_expectancy = result.expectancy
        chromosome.backtest_max_drawdown = result.max_drawdown
        score, reason = self.score(result)
        chromosome.backtest_score = score
        chromosome.backtest_completed = True
        if score > 0:
            chromosome.promotion_stage = "backtested"
            chromosome.promotion_reason = f"fitness_score={score}"
        else:
            chromosome.promotion_stage = "rejected"
            chromosome.reject_reason = reason

