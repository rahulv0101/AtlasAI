"""Factory that materializes a StrategyChromosome into an executable Strategy subclass.

The factory reads gene values and produces a live strategy object that
generates TradeSignals. It does NOT execute trades — ExecutionGate handles that.
"""
from __future__ import annotations
from typing import Any

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.base import Strategy
from atlas_ai.strategies.dna import StrategyChromosome


class _EvolvedGoldStrategy(Strategy):
    """Runtime strategy generated from a StrategyChromosome.

    Combines mean-reversion and trend-momentum entries with regime-aware
    filtering. All tunable parameters come from the chromosome genes.
    """
    name: str
    chromosome: StrategyChromosome

    def __init__(self, chromosome: StrategyChromosome) -> None:
        self.chromosome = chromosome
        self.name = f"evolved_{chromosome.uid[:6]}"

    # --- helpers --------------------------------------------------------
    def _ema(self, series: pd.Series, period: float) -> pd.Series:
        return series.ewm(span=max(1, int(period)), adjust=False).mean()

    def _rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    def _atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high = df["high"] if "high" in df.columns else pd.Series([], dtype=float)
        low = df["low"] if "low" in df.columns else pd.Series([], dtype=float)
        close = df["close"]
        if "high" not in df.columns or "low" not in df.columns:
            return close.diff().abs().ewm(span=period, adjust=False).mean()
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def _gene(self, key: str, default: float = 0.0) -> float:
        return self.chromosome.gene_value(key, default)

    def _body_ratio(self, df: pd.DataFrame) -> float:
        h = float(df["high"].iloc[-1])
        l = float(df["low"].iloc[-1])
        o = float(df["open"].iloc[-1])
        c = float(df["close"].iloc[-1])
        rng = max(1e-9, h - l)
        return abs(c - o) / rng

    def _trend_strength(self, closes: pd.Series) -> float:
        if len(closes) < 30:
            return 0.0
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if first == 0:
            return 0.0
        drift = ((last - first) / first) * 100.0
        abs_moves = (closes.diff().abs()).dropna()
        avg_abs = abs_moves.mean()
        vol = (avg_abs / max(1e-9, abs(last))) * 100.0
        if abs(drift) >= max(0.12, vol * 2.2):
            return abs(drift)
        return 0.0

    def _session_label(self, df: pd.DataFrame) -> str:
        """Infer session from the latest bar time."""
        t = pd.to_datetime(df["time"].iloc[-1])
        h = t.hour
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 13:
            return "london"
        if 13 <= h < 20:
            return "new_york"
        return "late_us"

    # --- signal generator -----------------------------------------------
    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        # Need enough bars
        if len(df) < max(50, int(self._gene("entry_ema_slow", 50)) + 5):
            return TradeSignal(strategy=self.name, side="flat", confidence=0.0, stop_loss_points=0.0, take_profit_points=0.0)

        closes = df["close"]
        fast_period = int(self._gene("entry_ema_fast", 20))
        slow_period = int(self._gene("entry_ema_slow", 50))
        ema_fast = self._ema(closes, fast_period)
        ema_slow = self._ema(closes, slow_period)
        rsi = self._rsi(closes, period=14)
        atr_series = self._atr(df, period=14)
        atr_last = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

        # Latest values
        fast_val = float(ema_fast.iloc[-1])
        slow_val = float(ema_slow.iloc[-1])
        prev_fast = float(ema_fast.iloc[-2])
        prev_slow = float(ema_slow.iloc[-2])
        rsi_val = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])

        # Regime context
        regime_trend_strength = self._trend_strength(closes.tail(60))
        regime_vol = 0.0
        if len(closes) >= 30:
            abs_moves = closes.tail(30).diff().abs()
            avg_abs = abs_moves.mean()
            if avg_abs is not None and abs(float(closes.iloc[-1])) > 1e-9:
                regime_vol = (float(avg_abs) / abs(float(closes.iloc[-1]))) * 100.0

        # --- Session filter ---
        session = self._session_label(df)
        allowed_sessions = int(self._gene("regime_allowed_sessions", 2))
        session_map = {0: ("asia",), 1: ("london",), 2: ("new_york",), 3: ("london", "new_york")}
        allowed = session_map.get(allowed_sessions, ("london", "new_york"))
        if session not in allowed:
            return TradeSignal(strategy=self.name, side="flat", confidence=0.0, stop_loss_points=0.0, take_profit_points=0.0)
        excluded_hours_mask = int(self._gene("regime_excluded_hours_mask", 0.0))
        latest_hour = int(pd.to_datetime(df["time"].iloc[-1]).hour) if "time" in df.columns else -1
        if latest_hour >= 0 and excluded_hours_mask & (1 << latest_hour):
            return TradeSignal(strategy=self.name, side="flat", confidence=0.0, stop_loss_points=0.0, take_profit_points=0.0)

        # --- Trend / regime filter ---
        prefer_trend = bool(self._gene("regime_prefer_trend", 0.0) > 0.5)
        regime_min_ts = self._gene("regime_min_trend_strength", 0.18)
        regime_max_vol = self._gene("regime_max_volatility_pct", 0.07)
        strong_trend = regime_trend_strength >= regime_min_ts
        low_vol = regime_vol < regime_max_vol
        if not strong_trend and prefer_trend:
            return TradeSignal(strategy=self.name, side="flat", confidence=0.0, stop_loss_points=0.0, take_profit_points=0.0)

        # --- Signal logic ---
        side: str = "flat"
        raw_conf = 0.0

        # EMA crossover
        fast_above_now = fast_val > slow_val
        fast_above_prev = prev_fast > prev_slow
        if trend_cross := (fast_above_now and not fast_above_prev):
            side = "buy"
            raw_conf = 0.72
        elif reversal_cross := (not fast_above_now and fast_above_prev):
            side = "sell"
            raw_conf = 0.72
        elif fast_above_now:
            side = "buy"
            raw_conf = 0.35
        else:
            side = "sell"
            raw_conf = 0.35

        # RSI confirmation
        rsi_buy = self._gene("rsi_buy_threshold", 42.0)
        rsi_sell = self._gene("rsi_sell_threshold", 58.0)
        if side == "buy":
            if rsi_val < rsi_buy:
                raw_conf = min(1.0, raw_conf + 0.12)
            elif rsi_val > rsi_sell:
                raw_conf = max(0.0, raw_conf - 0.22)
        if side == "sell":
            if rsi_val > rsi_sell:
                raw_conf = min(1.0, raw_conf + 0.12)
            elif rsi_val < rsi_buy:
                raw_conf = max(0.0, raw_conf - 0.22)

        # RSI divergence boost (simple form)
        if side == "buy":
            if prev_rsi > 50 and rsi_val < prev_rsi and closes.iloc[-1] > closes.iloc[-2]:
                raw_conf = min(1.0, raw_conf + 0.08)
        if side == "sell":
            if prev_rsi < 50 and rsi_val > prev_rsi and closes.iloc[-1] < closes.iloc[-2]:
                raw_conf = min(1.0, raw_conf + 0.08)

        # Candle body confidence
        min_body = self._gene("confirm_min_body_ratio", 0.35)
        body = self._body_ratio(df)
        if body < min_body * 0.5:
            raw_conf = max(0.0, raw_conf - 0.12)
        elif body >= min_body:
            raw_conf = min(1.0, raw_conf + 0.06)

        # Volume confirmation (using tick_volume as proxy)
        vol_col = "tick_volume" if "tick_volume" in df.columns else None
        if vol_col:
            last_vol = float(df[vol_col].iloc[-1])
            avg_vol = float(df[vol_col].tail(8).mean())
            vol_ratio = last_vol / max(avg_vol, 1.0)
            min_vol = self._gene("confirm_volume_min_ratio", 0.85)
            if vol_ratio < min_vol:
                raw_conf = max(0.0, raw_conf - 0.08)
            elif vol_ratio >= 1.0:
                raw_conf = min(1.0, raw_conf + 0.04)

        # Minimum confidence gate
        min_conf = self._gene("risk_min_confidence", 0.55)
        if raw_conf < min_conf:
            return TradeSignal(strategy=self.name, side="flat", confidence=0.0, stop_loss_points=0.0, take_profit_points=0.0)

        # Confidence multiplier
        conf_mult = self._gene("risk_confidence_multiplier", 1.0)
        confidence = max(0.0, min(1.0, raw_conf * conf_mult))

        # Prevent evolved variants from bunching at 1.0 before live evidence
        if confidence > 0.82:
            confidence = (confidence * 0.72) + 0.08

        # Stop loss / take profit in points
        sl_mult = self._gene("exit_sl_atr_multiple", 1.6)
        tp_mult = self._gene("exit_tp_atr_multiple", 2.8)
        sl_points = atr_last * sl_mult
        tp_points = atr_last * tp_mult

        return TradeSignal(
            strategy=self.name,
            side=side,  # type: ignore[arg-type]
            confidence=confidence,
            stop_loss_points=sl_points,
            take_profit_points=tp_points,
        )


class StrategyFactory:
    """Creates executable Strategy objects from StrategyChromosomes."""

    @staticmethod
    def build(chromosome: StrategyChromosome) -> Strategy:
        return _EvolvedGoldStrategy(chromosome)

