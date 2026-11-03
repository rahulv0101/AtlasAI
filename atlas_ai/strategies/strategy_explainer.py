"""Plain-English explainer for evolved strategy chromosomes.

Generates human-readable descriptions of what a strategy does,
its parameters, and its current performance state.
"""
from __future__ import annotations
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome


class StrategyExplainer:
    """Produces plain-English summaries of StrategyChromosomes."""

    @staticmethod
    def explain(chromosome: StrategyChromosome) -> dict[str, str]:
        """Return a structured plain-English description by gene category."""
        g = chromosome.gene_value

        entry = (
            f"Entry uses EMA({g('entry_ema_fast', 20):.0f}) vs EMA({g('entry_ema_slow', 50):.0f}) "
            f"crossover with RSI confirmation. Buy when RSI < {g('rsi_buy_threshold', 42):.1f}; "
            f"sell when RSI > {g('rsi_sell_threshold', 58):.1f}."
        )

        confirmation = (
            f"Requires trend strength >= {g('confirm_min_trend_strength', 0.22):.2f}, "
            f"candle body ratio >= {g('confirm_min_body_ratio', 0.35):.2f}, "
            f"volume >= {g('confirm_volume_min_ratio', 0.85):.2f}x average. "
            f"HTF alignment {'enabled' if g('confirm_htf_alignment', 1) > 0.5 else 'disabled'}."
        )

        exit = (
            f"Stop-loss at {g('exit_sl_atr_multiple', 1.6):.1f}x ATR; "
            f"take-profit at {g('exit_tp_atr_multiple', 2.8):.1f}x ATR. "
            f"{'Trailing' if g('exit_trail_enable', 1) > 0.5 else 'Fixed'} stop. "
            f"Max hold {g('exit_max_hold_bars', 8):.0f} bars."
        )

        risk = (
            f"Risk per trade: {g('risk_max_risk_per_trade', 0.005):.3f} ({g('risk_max_risk_per_trade', 0.005)*100:.2f}%). "
            f"Min confidence: {g('risk_min_confidence', 0.55):.2f}. "
            f"Confidence multiplier: {g('risk_confidence_multiplier', 1.0):.2f}. "
            f"Max {g('risk_max_trades_per_day', 8):.0f} trades/day."
        )

        regime = (
            f"{'Trend-preferring' if g('regime_prefer_trend', 0) > 0.5 else 'Regime-agnostic'} filter. "
            f"Min trend strength: {g('regime_min_trend_strength', 0.18):.2f}. "
            f"Max volatility: {g('regime_max_volatility_pct', 0.07):.2%}. "
            f"Sessions allowed: {_session_label(int(g('regime_allowed_sessions', 2)))}."
        )

        performance = StrategyExplainer._performance_summary(chromosome)
        lineage = StrategyExplainer._lineage_summary(chromosome)

        return {
            "entry": entry,
            "confirmation": confirmation,
            "exit": exit,
            "risk": risk,
            "regime": regime,
            "performance": performance,
            "lineage": lineage,
        }

    @staticmethod
    def explain_compact(chromosome: StrategyChromosome) -> str:
        """One-line summary suitable for UI lists."""
        g = chromosome.gene_value
        perf = StrategyExplainer._performance_summary(chromosome)
        return (
            f"EMA({g('entry_ema_fast', 20):.0f}/{g('entry_ema_slow', 50):.0f}) "
            f"RSI({g('rsi_buy_threshold', 42):.0f}/{g('rsi_sell_threshold', 58):.0f}) | "
            f"SL/TP={g('exit_sl_atr_multiple', 1.6):.1f}x/{g('exit_tp_atr_multiple', 2.8):.1f}x | "
            f"stage={chromosome.promotion_stage} | {perf}"
        )

    @staticmethod
    def explain_markdown(chromosome: StrategyChromosome) -> str:
        """Formatted markdown suitable for operator review panels."""
        sections = StrategyExplainer.explain(chromosome)
        lines = [
            f"### Strategy `{chromosome.uid[:8]}` (gen {chromosome.generation})",
            "",
            f"**Origin:** {sections['lineage']}",
            f"**Stage:** {chromosome.promotion_stage}",
            "",
            "#### Entry Logic",
            sections["entry"],
            "",
            "#### Confirmation Filters",
            sections["confirmation"],
            "",
            "#### Exit Logic",
            sections["exit"],
            "",
            "#### Risk Controls",
            sections["risk"],
            "",
            "#### Regime Filters",
            sections["regime"],
            "",
            "#### Performance Record",
            sections["performance"],
        ]
        if chromosome.reject_reason:
            lines.extend(["", f"**Rejection Reason:** {chromosome.reject_reason}"])
        return "\n".join(lines)

    @staticmethod
    def _performance_summary(chromosome: StrategyChromosome) -> str:
        if not chromosome.backtest_completed:
            return "Backtest pending."
        if chromosome.backtest_score <= 0:
            return f"Backtest rejected: {chromosome.reject_reason or 'unknown'}"
        return (
            f"Backtest: score={chromosome.backtest_score:.1f} "
            f"wr={chromosome.backtest_win_rate:.2%} "
            f"pf={chromosome.backtest_profit_factor:.2f} "
            f"sharpe={chromosome.backtest_sharpe:.2f} "
            f"dd={chromosome.backtest_max_drawdown:.2%} "
            f"trades={chromosome.backtest_trades}"
        )

    @staticmethod
    def _lineage_summary(chromosome: StrategyChromosome) -> str:
        if chromosome.origin == "random":
            return "Random seed"
        if chromosome.origin == "mutation":
            return f"Mutated from {', '.join(chromosome.parent_uids)}"
        if chromosome.origin == "crossover":
            return f"Crossover of {', '.join(chromosome.parent_uids)}"
        return f"Origin: {chromosome.origin}"


def _session_label(flag: int) -> str:
    mapping = {
        0: "Asia only",
        1: "London only",
        2: "New York only",
        3: "London + New York",
    }
    return mapping.get(flag, "Unknown")

