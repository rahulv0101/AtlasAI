from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.base import Strategy
from atlas_ai.strategies.indicators import atr, ema, rsi


@dataclass
class EMACrossoverStrategy(Strategy):
    fast: int = 20
    slow: int = 50
    name: str = "ema_crossover"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < max(self.fast, self.slow) + 3:
            return TradeSignal(self.name, "flat", 0.0, 120.0, 180.0)

        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        vol = atr(df).iloc[-1]

        prev_fast = float(e_fast.iloc[-2])
        prev_slow = float(e_slow.iloc[-2])
        curr_fast = float(e_fast.iloc[-1])
        curr_slow = float(e_slow.iloc[-1])

        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow

        if curr_fast > curr_slow:
            side = "buy"
        elif curr_fast < curr_slow:
            side = "sell"
        else:
            side = "flat"

        price_now = max(float(close.iloc[-1]), 1e-9)
        spread_now = abs(curr_fast - curr_slow)
        slope_fast = abs(curr_fast - prev_fast)
        spread_component = (spread_now / price_now) * 700.0
        slope_component = (slope_fast / price_now) * 320.0
        cross_bonus = 0.18 if (bullish_cross or bearish_cross) else 0.0
        confidence = min(1.0, spread_component + slope_component + cross_bonus) if side != "flat" else 0.0
        return TradeSignal(self.name, side, confidence, max(vol * 2, 100), max(vol * 3, 150))


@dataclass
class EMACrossoverClosedConfirmMidExitStrategy(Strategy):
    fast: int = 8
    slow: int = 21
    spread_multiplier: float = 450.0
    slope_multiplier: float = 180.0
    cross_bonus: float = 0.12
    sl_atr_multiple: float = 2.0
    sl_floor_points: float = 1100.0
    tp_atr_multiple: float = 2.6
    tp_floor_points: float = 1450.0
    name: str = "ema_crossover_closed_confirm_mid_exit"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < max(self.fast, self.slow) + 4:
            return TradeSignal(self.name, "flat", 0.0, self.sl_floor_points, self.tp_floor_points)

        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        vol = atr(df).iloc[-1]

        fast_before_cross = float(e_fast.iloc[-3])
        slow_before_cross = float(e_slow.iloc[-3])
        fast_cross_bar = float(e_fast.iloc[-2])
        slow_cross_bar = float(e_slow.iloc[-2])
        curr_fast = float(e_fast.iloc[-1])
        curr_slow = float(e_slow.iloc[-1])

        closed_bullish_cross = fast_before_cross <= slow_before_cross and fast_cross_bar > slow_cross_bar
        closed_bearish_cross = fast_before_cross >= slow_before_cross and fast_cross_bar < slow_cross_bar

        side = "flat"
        if closed_bullish_cross and curr_fast > curr_slow:
            side = "buy"
        elif closed_bearish_cross and curr_fast < curr_slow:
            side = "sell"

        price_now = max(float(close.iloc[-1]), 1e-9)
        spread_component = (abs(curr_fast - curr_slow) / price_now) * self.spread_multiplier
        slope_component = (abs(curr_fast - fast_cross_bar) / price_now) * self.slope_multiplier
        confidence = min(1.0, spread_component + slope_component + self.cross_bonus) if side != "flat" else 0.0
        return TradeSignal(
            self.name,
            side,
            confidence,
            max(float(vol) * self.sl_atr_multiple, self.sl_floor_points),
            max(float(vol) * self.tp_atr_multiple, self.tp_floor_points),
        )


@dataclass
class RSIMeanReversionStrategy(Strategy):
    lower: float = 30.0
    upper: float = 70.0
    name: str = "rsi_mean_reversion"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        value = rsi(close).iloc[-1]
        vol = atr(df).iloc[-1]

        if value <= self.lower:
            side = "buy"
            confidence = min((self.lower - value) / self.lower, 1.0)
        elif value >= self.upper:
            side = "sell"
            confidence = min((value - self.upper) / (100 - self.upper), 1.0)
        else:
            side = "flat"
            confidence = 0.0

        return TradeSignal(self.name, side, confidence, max(vol * 1.8, 100), max(vol * 2.2, 140))


@dataclass
class BreakoutStrategy(Strategy):
    lookback: int = 30
    name: str = "breakout"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        recent = df.tail(self.lookback + 1)
        threshold_high = recent["high"].iloc[:-1].max()
        threshold_low = recent["low"].iloc[:-1].min()
        price = recent["close"].iloc[-1]
        vol = atr(df).iloc[-1]

        if price > threshold_high:
            side = "buy"
            confidence = min((price - threshold_high) / max(price, 1), 1.0)
        elif price < threshold_low:
            side = "sell"
            confidence = min((threshold_low - price) / max(price, 1), 1.0)
        else:
            side = "flat"
            confidence = 0.0

        return TradeSignal(self.name, side, confidence * 4, max(vol * 2.0, 120), max(vol * 3.0, 180))


@dataclass
class CandlestickPatternStrategy(Strategy):
    name: str = "candlestick_patterns"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 6:
            return TradeSignal(self.name, "flat", 0.0, 120.0, 180.0)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        vol = atr(df).iloc[-1]

        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        po = float(prev["open"])
        pc = float(prev["close"])

        body = abs(c - o)
        rng = max(h - l, 1e-9)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        bullish_engulf = (pc < po) and (c > o) and (c >= po) and (o <= pc)
        bearish_engulf = (pc > po) and (c < o) and (o >= pc) and (c <= po)
        hammer = (lower_wick >= body * 2.2) and (upper_wick <= body * 0.8) and (c > o)
        shooting_star = (upper_wick >= body * 2.2) and (lower_wick <= body * 0.8) and (c < o)

        side = "flat"
        confidence = 0.0
        if bullish_engulf or hammer:
            side = "buy"
            confidence = min(1.0, (body / rng) * 1.8 + (lower_wick / rng) * 0.5)
        elif bearish_engulf or shooting_star:
            side = "sell"
            confidence = min(1.0, (body / rng) * 1.8 + (upper_wick / rng) * 0.5)

        return TradeSignal(self.name, side, confidence, max(vol * 1.7, 110), max(vol * 2.5, 170))
