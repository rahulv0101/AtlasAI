"""XAUUSD-specific strategy validator with hard safety rules.

Rejects dangerous parameter combinations before they reach backtesting.
Never mutates ExecutionGate or safety logic — only validates strategy DNA.
"""
from __future__ import annotations
from typing import Any, Callable

from atlas_ai.strategies.dna import StrategyChromosome, StrategyGene


class GoldValidationError(Exception):
    """Raised when a strategy chromosome violates XAUUSD safety rules."""
    pass


class GoldValidator:
    """Validates StrategyChromosomes against hard XAUUSD constraints.

    All rejections are unconditional — no override or bypass exists.
    """

    def __init__(self) -> None:
        self._rules: list[Callable[[StrategyChromosome], str]] = [
            self._rule_ema_fast_lt_slow,
            self._rule_ema_not_equal,
            self._rule_rsi_buy_lt_sell,
            self._rule_min_spread_viable,
            self._rule_max_hold_not_zero,
            self._rule_risk_not_excessive,
            self._rule_exit_tp_gt_sl,
            self._rule_confidence_in_range,
            self._rule_sessions_valid,
            self._rule_excluded_hours_valid,
            self._rule_loss_cooldown_valid,
        ]

    def validate(self, chromosome: StrategyChromosome) -> None:
        """Validate a chromosome; raise GoldValidationError on violation."""
        for rule in self._rules:
            reason = rule(chromosome)
            if reason:
                raise GoldValidationError(reason)

    def validate_silent(self, chromosome: StrategyChromosome) -> tuple[bool, str]:
        """Validate without raising; returns (ok, reason)."""
        try:
            self.validate(chromosome)
            return True, ""
        except GoldValidationError as exc:
            return False, str(exc)

    def _get(self, chromosome: StrategyChromosome, name: str, default: float = 0.0) -> float:
        return chromosome.gene_value(name, default)

    # ------------------------------------------------------------------
    # Hard XAUUSD rules
    # ------------------------------------------------------------------

    def _rule_ema_fast_lt_slow(self, c: StrategyChromosome) -> str:
        fast = self._get(c, "entry_ema_fast")
        slow = self._get(c, "entry_ema_slow")
        if fast >= slow:
            return f"EMA fast ({fast}) must be strictly less than slow ({slow})"
        if slow - fast < 8:
            return f"EMA span too tight ({slow - fast} < 8) — crossover noise risk"
        return ""

    def _rule_ema_not_equal(self, c: StrategyChromosome) -> str:
        fast = self._get(c, "entry_ema_fast")
        slow = self._get(c, "entry_ema_slow")
        if fast == slow:
            return "EMA periods cannot be identical"
        return ""

    def _rule_rsi_buy_lt_sell(self, c: StrategyChromosome) -> str:
        buy = self._get(c, "rsi_buy_threshold")
        sell = self._get(c, "rsi_sell_threshold")
        if buy >= sell:
            return f"RSI buy threshold ({buy}) must be below sell threshold ({sell})"
        gap = sell - buy
        if gap < 6:
            return f"RSI buy/sell gap too narrow ({gap} < 6) — will double-trigger"
        return ""

    def _rule_min_spread_viable(self, c: StrategyChromosome) -> str:
        sl = self._get(c, "exit_sl_atr_multiple")
        tp = self._get(c, "exit_tp_atr_multiple")
        # On XAUUSD a typical 15m ATR is ~80-200 pts; SL must allow for
        # at least a few pips of effective stop distance.
        if sl < 0.5:
            return f"Stop-loss ATR multiple too small ({sl}) for XAUUSD volatility"
        return ""

    def _rule_max_hold_not_zero(self, c: StrategyChromosome) -> str:
        hold = self._get(c, "exit_max_hold_bars")
        if hold < 2:
            return f"Max hold bars ({hold}) too short — would force premature exits"
        if hold > 40:
            return f"Max hold bars ({hold}) too long — exposes strategy to overnight/rollover risk on short timeframe"
        return ""

    def _rule_risk_not_excessive(self, c: StrategyChromosome) -> str:
        risk = self._get(c, "risk_max_risk_per_trade")
        if risk > 0.02:
            return f"Risk per trade ({risk}) exceeds 2.0% hard ceiling for XAUUSD"
        return ""

    def _rule_exit_tp_gt_sl(self, c: StrategyChromosome) -> str:
        sl = self._get(c, "exit_sl_atr_multiple")
        tp = self._get(c, "exit_tp_atr_multiple")
        if tp <= sl:
            return f"TP ATR multiple ({tp}) must exceed SL ATR multiple ({sl})"
        return ""

    def _rule_confidence_in_range(self, c: StrategyChromosome) -> str:
        min_conf = self._get(c, "risk_min_confidence")
        if min_conf < 0.25:
            return f"Min confidence ({min_conf}) too permissive — would admit low-conviction entries"
        if min_conf > 0.90:
            return f"Min confidence ({min_conf}) too restrictive — strategy will never fire"
        return ""

    def _rule_sessions_valid(self, c: StrategyChromosome) -> str:
        sessions = int(self._get(c, "regime_allowed_sessions"))
        if sessions < 0 or sessions > 3:
            return f"Allowed sessions flag ({sessions}) out of range 0-3"
        return ""

    def _rule_excluded_hours_valid(self, c: StrategyChromosome) -> str:
        mask = int(self._get(c, "regime_excluded_hours_mask", 0.0))
        if mask < 0 or mask >= (1 << 24):
            return f"Excluded hours mask ({mask}) out of range 0-{(1 << 24) - 1}"
        sessions = int(self._get(c, "regime_allowed_sessions"))
        allowed_hours = {
            1: set(range(7, 13)),
            2: set(range(13, 20)),
            3: set(range(7, 20)),
        }.get(sessions, set(range(24)))
        excluded_allowed = {hour for hour in allowed_hours if mask & (1 << hour)}
        if allowed_hours and len(excluded_allowed) >= len(allowed_hours):
            return "Excluded hours mask blocks every allowed session hour"
        return ""

    def _rule_loss_cooldown_valid(self, c: StrategyChromosome) -> str:
        cooldown = self._get(c, "risk_loss_cooldown_bars", 0.0)
        lift = self._get(c, "risk_loss_cooldown_confidence_lift", 0.0)
        if cooldown < 0 or cooldown > 12:
            return f"Loss cooldown bars ({cooldown}) out of range 0-12"
        if lift < 0 or lift > 0.25:
            return f"Loss cooldown confidence lift ({lift}) out of range 0.00-0.25"
        if cooldown >= 10 and lift >= 0.20:
            return "Loss cooldown too restrictive for research flow"
        return ""

