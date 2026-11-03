from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.base import Strategy
from atlas_ai.strategies.indicators import atr, bollinger_bands, ema, macd, roc, rsi, stochastic_kd


def _safe_vol(df: pd.DataFrame) -> float:
    v = float(atr(df).iloc[-1])
    if not v or v != v:
        return 120.0
    return v


def _recent_regime_stats(close: pd.Series, lookback: int = 30) -> tuple[str, float, float]:
    recent = close.tail(lookback)
    if len(recent) < max(12, lookback // 2):
        return "unknown", 0.0, 0.0
    first = float(recent.iloc[0])
    last = float(recent.iloc[-1])
    if first == 0.0:
        return "unknown", 0.0, 0.0
    drift_pct = ((last - first) / first) * 100.0
    abs_moves = [abs(float(recent.iloc[i]) - float(recent.iloc[i - 1])) for i in range(1, len(recent))]
    avg_abs = sum(abs_moves) / max(1, len(abs_moves))
    vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
    if abs(drift_pct) >= max(0.12, vol_pct * 2.2):
        return "trend", abs(drift_pct), vol_pct
    if vol_pct >= 0.06:
        return "high_volatility_range", abs(drift_pct), vol_pct
    return "range", abs(drift_pct), vol_pct


@dataclass
class MACDTrendStrategy(Strategy):
    name: str = "macd_trend"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        line, sig, hist = macd(close)
        m = float(line.iloc[-1] - sig.iloc[-1])
        h = float(hist.iloc[-1])
        vol = _safe_vol(df)
        if m > 0 and h > 0:
            side = "buy"
        elif m < 0 and h < 0:
            side = "sell"
        else:
            side = "flat"
        confidence = min(1.0, abs(m) / max(1e-9, float(close.iloc[-1])) * 9000)
        return TradeSignal(self.name, side, confidence, max(vol * 1.8, 110), max(vol * 2.6, 170))


@dataclass
class BollingerMeanReversionStrategy(Strategy):
    period: int = 20
    name: str = "bollinger_mean_reversion"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        lower, mid, upper = bollinger_bands(close, self.period, 2.0)
        price = float(close.iloc[-1])
        vol = _safe_vol(df)
        side = "flat"
        confidence = 0.0
        if price < float(lower.iloc[-1]):
            side = "buy"
            confidence = min(1.0, (float(lower.iloc[-1]) - price) / max(price, 1.0) * 20)
        elif price > float(upper.iloc[-1]):
            side = "sell"
            confidence = min(1.0, (price - float(upper.iloc[-1])) / max(price, 1.0) * 20)
        return TradeSignal(self.name, side, confidence, max(vol * 1.5, 100), max(vol * 2.1, 140))


@dataclass
class StochasticReversalStrategy(Strategy):
    name: str = "stochastic_reversal"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        k, d = stochastic_kd(df, period=14, smooth_k=3, smooth_d=3)
        kv = float(k.iloc[-1])
        dv = float(d.iloc[-1])
        prev_k = float(k.iloc[-2]) if len(k) > 1 else kv
        prev_d = float(d.iloc[-2]) if len(d) > 1 else dv
        vol = _safe_vol(df)

        side = "flat"
        confidence = 0.0
        if kv < 25 and dv < 25 and prev_k <= prev_d and kv > dv:
            side = "buy"
            confidence = min(1.0, (25 - kv) / 25 + 0.15)
        elif kv > 75 and dv > 75 and prev_k >= prev_d and kv < dv:
            side = "sell"
            confidence = min(1.0, (kv - 75) / 25 + 0.15)
        return TradeSignal(self.name, side, confidence, max(vol * 1.7, 105), max(vol * 2.4, 160))


@dataclass
class VWAPProxyReversionStrategy(Strategy):
    length: int = 40
    name: str = "vwap_proxy_reversion"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        recent = df.tail(self.length).copy()
        if len(recent) < 10:
            return TradeSignal(self.name, "flat", 0.0, 120.0, 170.0)
        typical = (recent["high"] + recent["low"] + recent["close"]) / 3.0
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        vol = recent[vol_col].replace(0, 1.0)
        vwap = float((typical * vol).sum() / max(1.0, float(vol.sum())))
        price = float(recent["close"].iloc[-1])
        dist = (price - vwap) / max(1.0, price)
        a = _safe_vol(df)
        if dist < -0.0012:
            side = "buy"
            confidence = min(1.0, abs(dist) * 450)
        elif dist > 0.0012:
            side = "sell"
            confidence = min(1.0, abs(dist) * 450)
        else:
            side = "flat"
            confidence = 0.0
        return TradeSignal(self.name, side, confidence, max(a * 1.6, 105), max(a * 2.3, 155))


@dataclass
class MomentumContinuationStrategy(Strategy):
    name: str = "momentum_continuation"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        r = roc(close, 12).iloc[-1]
        fast = ema(close, 9).iloc[-1]
        slow = ema(close, 21).iloc[-1]
        vol = _safe_vol(df)
        regime, trend_strength, volatility_pct = _recent_regime_stats(close, lookback=30)
        side = "flat"
        confidence = 0.0
        if regime != "trend" or trend_strength < 0.18 or volatility_pct < 0.03:
            return TradeSignal(self.name, "flat", 0.0, max(vol * 1.9, 115), max(vol * 2.8, 175))

        trend_weight = min(1.25, max(0.8, trend_strength / 0.24))
        if r > 0.08 and fast > slow:
            side = "buy"
            confidence = min(1.0, (float(r) / 0.6) * trend_weight)
        elif r < -0.08 and fast < slow:
            side = "sell"
            confidence = min(1.0, (abs(float(r)) / 0.6) * trend_weight)
        return TradeSignal(self.name, side, confidence, max(vol * 1.9, 115), max(vol * 2.8, 175))


@dataclass
class PullbackTrendStrategy(Strategy):
    name: str = "pullback_trend"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        e20 = ema(close, 20)
        e50 = ema(close, 50)
        cur = float(close.iloc[-1])
        vol = _safe_vol(df)
        side = "flat"
        confidence = 0.0
        if float(e20.iloc[-1]) > float(e50.iloc[-1]) and cur <= float(e20.iloc[-1]) * 1.0003:
            side = "buy"
            confidence = min(1.0, abs(float(e20.iloc[-1]) - cur) / max(cur, 1.0) * 300)
        elif float(e20.iloc[-1]) < float(e50.iloc[-1]) and cur >= float(e20.iloc[-1]) * 0.9997:
            side = "sell"
            confidence = min(1.0, abs(float(e20.iloc[-1]) - cur) / max(cur, 1.0) * 300)
        return TradeSignal(self.name, side, confidence, max(vol * 1.8, 110), max(vol * 2.6, 170))


@dataclass
class PullbackTrendM15MidExitStrategy(Strategy):
    fast: int = 8
    slow: int = 21
    proximity_band: float = 0.0006
    confidence_multiplier: float = 180.0
    sl_atr_multiple: float = 2.0
    sl_floor_points: float = 1100.0
    tp_atr_multiple: float = 2.6
    tp_floor_points: float = 1450.0
    name: str = "pullback_trend_m15_mid_exit"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        if len(close) < max(self.fast, self.slow) + 3:
            return TradeSignal(self.name, "flat", 0.0, self.sl_floor_points, self.tp_floor_points)

        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        cur = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        vol = _safe_vol(df)
        side = "flat"
        confidence = 0.0
        fast_now = float(e_fast.iloc[-1])
        slow_now = float(e_slow.iloc[-1])

        if fast_now > slow_now and cur <= fast_now * (1.0 + self.proximity_band) and cur >= prev:
            side = "buy"
            confidence = min(1.0, abs(fast_now - cur) / max(cur, 1.0) * self.confidence_multiplier)
        elif fast_now < slow_now and cur >= fast_now * (1.0 - self.proximity_band) and cur <= prev:
            side = "sell"
            confidence = min(1.0, abs(fast_now - cur) / max(cur, 1.0) * self.confidence_multiplier)
        return TradeSignal(
            self.name,
            side,
            confidence,
            max(vol * self.sl_atr_multiple, self.sl_floor_points),
            max(vol * self.tp_atr_multiple, self.tp_floor_points),
        )


@dataclass
class RSITrendHybridStrategy(Strategy):
    name: str = "rsi_trend_hybrid"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        rv = float(rsi(close, 14).iloc[-1])
        e8 = float(ema(close, 8).iloc[-1])
        e34 = float(ema(close, 34).iloc[-1])
        vol = _safe_vol(df)

        side = "flat"
        confidence = 0.0
        if rv > 52 and e8 > e34:
            side = "buy"
            confidence = min(1.0, (rv - 50) / 25)
        elif rv < 48 and e8 < e34:
            side = "sell"
            confidence = min(1.0, (50 - rv) / 25)
        return TradeSignal(self.name, side, confidence, max(vol * 1.7, 105), max(vol * 2.4, 160))


@dataclass
class GoldMicroImpulseStrategy(Strategy):
    name: str = "gold_micro_impulse"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 30:
            return TradeSignal(self.name, "flat", 0.0, 90.0, 130.0)
        recent = df.tail(12).copy()
        close = df["close"]
        e5 = ema(close, 5)
        e13 = ema(close, 13)
        last = recent.iloc[-1]
        prev = recent.iloc[-2]
        vol = _safe_vol(df)
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        last_vol = float(last.get(vol_col, 0.0) or 0.0)
        avg_vol = float(recent[vol_col].tail(8).mean() or 1.0)
        body = abs(float(last["close"]) - float(last["open"]))
        rng = max(1e-9, float(last["high"]) - float(last["low"]))
        body_ratio = body / rng
        breakout_high = float(recent["high"].iloc[:-1].max())
        breakout_low = float(recent["low"].iloc[:-1].min())

        side = "flat"
        confidence = 0.0
        if (
            float(last["close"]) > breakout_high
            and float(e5.iloc[-1]) > float(e13.iloc[-1])
            and float(last["close"]) > float(prev["close"])
            and last_vol >= avg_vol * 1.15
            and body_ratio >= 0.55
        ):
            side = "buy"
            confidence = min(1.0, ((last_vol / max(avg_vol, 1.0)) - 1.0) * 0.45 + body_ratio * 0.7)
        elif (
            float(last["close"]) < breakout_low
            and float(e5.iloc[-1]) < float(e13.iloc[-1])
            and float(last["close"]) < float(prev["close"])
            and last_vol >= avg_vol * 1.15
            and body_ratio >= 0.55
        ):
            side = "sell"
            confidence = min(1.0, ((last_vol / max(avg_vol, 1.0)) - 1.0) * 0.45 + body_ratio * 0.7)
        return TradeSignal(self.name, side, confidence, max(vol * 1.25, 85), max(vol * 1.9, 120))


@dataclass
class GoldBreakoutRetestStrategy(Strategy):
    lookback: int = 28
    confirmation_window: int = 6
    name: str = "gold_breakout_retest"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < self.lookback + self.confirmation_window + 10:
            return TradeSignal(self.name, "flat", 0.0, 105.0, 155.0)
        recent = df.tail(self.lookback + self.confirmation_window).copy()
        base = recent.iloc[: self.lookback]
        confirm = recent.iloc[self.lookback :]
        last = confirm.iloc[-1]
        prev = confirm.iloc[-2]
        close = df["close"]
        e5 = ema(close, 5)
        e13 = ema(close, 13)
        vol = _safe_vol(df)
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        avg_vol = float(recent[vol_col].tail(18).mean() or 1.0)
        last_vol = float(last.get(vol_col, 0.0) or 0.0)
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        rng = max(1e-9, h - l)
        body_ratio = abs(c - o) / rng
        prior_high = float(base["high"].max())
        prior_low = float(base["low"].min())
        tolerance = max(vol * 0.28, 0.8)
        closes = [float(x) for x in confirm["close"].tolist()]
        lows = [float(x) for x in confirm["low"].tolist()]
        highs = [float(x) for x in confirm["high"].tolist()]

        buy_break = any(x > prior_high + tolerance * 0.15 for x in closes[:-1])
        buy_retest = any(low <= prior_high + tolerance and close_at >= prior_high - tolerance * 0.45 for low, close_at in zip(lows[1:], closes[1:]))
        sell_break = any(x < prior_low - tolerance * 0.15 for x in closes[:-1])
        sell_retest = any(high >= prior_low - tolerance and close_at <= prior_low + tolerance * 0.45 for high, close_at in zip(highs[1:], closes[1:]))

        side = "flat"
        confidence = 0.0
        volume_factor = min(0.18, max(0.0, (last_vol / max(avg_vol, 1.0)) - 1.0) * 0.12)
        if (
            buy_break
            and buy_retest
            and c > max(prior_high + tolerance * 0.25, float(prev["close"]))
            and float(e5.iloc[-1]) > float(e13.iloc[-1])
            and body_ratio >= 0.38
        ):
            side = "buy"
            follow_through = min(0.28, max(0.0, (c - prior_high) / max(vol, 1.0)) * 0.18)
            confidence = min(0.72, 0.30 + body_ratio * 0.34 + follow_through + volume_factor)
        elif (
            sell_break
            and sell_retest
            and c < min(prior_low - tolerance * 0.25, float(prev["close"]))
            and float(e5.iloc[-1]) < float(e13.iloc[-1])
            and body_ratio >= 0.38
        ):
            side = "sell"
            follow_through = min(0.28, max(0.0, (prior_low - c) / max(vol, 1.0)) * 0.18)
            confidence = min(0.72, 0.30 + body_ratio * 0.34 + follow_through + volume_factor)
        return TradeSignal(self.name, side, confidence, max(vol * 1.45, 100), max(vol * 2.15, 145))


@dataclass
class GoldLiquiditySweepStrategy(Strategy):
    lookback: int = 8
    name: str = "gold_liquidity_sweep"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < self.lookback + 3:
            return TradeSignal(self.name, "flat", 0.0, 95.0, 135.0)
        recent = df.tail(self.lookback + 2).copy()
        last = recent.iloc[-1]
        prior = recent.iloc[:-1]
        vol = _safe_vol(df)
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        rng = max(1e-9, h - l)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        side = "flat"
        confidence = 0.0
        if h > prior_high and c < prior_high and upper_wick / rng >= 0.42:
            side = "sell"
            confidence = min(1.0, ((h - prior_high) / max(c, 1.0)) * 250 + (upper_wick / rng))
        elif l < prior_low and c > prior_low and lower_wick / rng >= 0.42:
            side = "buy"
            confidence = min(1.0, ((prior_low - l) / max(c, 1.0)) * 250 + (lower_wick / rng))
        return TradeSignal(self.name, side, confidence, max(vol * 1.3, 90), max(vol * 2.0, 125))


@dataclass
class GoldMicroStructureReversionStrategy(Strategy):
    name: str = "gold_micro_structure_reversion"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < 26:
            return TradeSignal(self.name, "flat", 0.0, 90.0, 120.0)
        recent = df.tail(20).copy()
        close = recent["close"]
        low_band, mid_band, high_band = bollinger_bands(close, period=14, num_std=1.6)
        last = recent.iloc[-1]
        prev = recent.iloc[-2]
        vol = _safe_vol(df)
        rng = max(1e-9, float(last["high"]) - float(last["low"]))
        body = abs(float(last["close"]) - float(last["open"]))
        body_ratio = body / rng
        pre_trigger = recent.iloc[:-2].tail(6)
        recent_range = float(pre_trigger["high"].max() - pre_trigger["low"].min()) if len(pre_trigger) >= 3 else 0.0
        baseline_range = float(recent["high"].max() - recent["low"].min())
        compressed = recent_range <= max(1e-9, baseline_range * 0.42)

        side = "flat"
        confidence = 0.0
        if (
            compressed
            and float(prev["close"]) < float(low_band.iloc[-2])
            and float(last["close"]) > float(mid_band.iloc[-1])
            and body_ratio >= 0.45
        ):
            side = "buy"
            confidence = min(1.0, body_ratio * 0.9 + 0.18)
        elif (
            compressed
            and float(prev["close"]) > float(high_band.iloc[-2])
            and float(last["close"]) < float(mid_band.iloc[-1])
            and body_ratio >= 0.45
        ):
            side = "sell"
            confidence = min(1.0, body_ratio * 0.9 + 0.18)
        return TradeSignal(self.name, side, confidence, max(vol * 1.2, 85), max(vol * 1.7, 115))
