from __future__ import annotations

import json
import argparse
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import load_settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.regime_filters import (
    build_research_regime_overlay_context,
    conservative_pretrade_filter_reason,
    research_regime_filter_reason,
)
from atlas_ai.strategies.base import Strategy
from atlas_ai.strategies.indicators import atr, ema
from atlas_ai.strategies.minute_pack import PullbackTrendM15MidExitStrategy
from scripts.capture_baseline import BASELINE_DIR, _parse_dt, _result_payload


START = "2026-08-03T00:00:00+00:00"
END = "2026-08-21T23:59:59+00:00"


@dataclass
class M15EMACrossoverProposal(Strategy):
    fast: int = 8
    slow: int = 21
    spread_multiplier: float = 450.0
    slope_multiplier: float = 180.0
    cross_bonus: float = 0.12
    sl_atr_multiple: float = 1.4
    sl_floor: float = 150.0
    tp_atr_multiple: float = 2.2
    tp_floor: float = 240.0
    name: str = "ema_crossover"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < max(self.fast, self.slow) + 3:
            return TradeSignal(self.name, "flat", 0.0, self.sl_floor, self.tp_floor)
        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        vol = float(atr(df).iloc[-1])
        prev_fast = float(e_fast.iloc[-2])
        prev_slow = float(e_slow.iloc[-2])
        curr_fast = float(e_fast.iloc[-1])
        curr_slow = float(e_slow.iloc[-1])
        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow
        side = "buy" if curr_fast > curr_slow else ("sell" if curr_fast < curr_slow else "flat")
        price_now = max(float(close.iloc[-1]), 1e-9)
        spread_component = (abs(curr_fast - curr_slow) / price_now) * self.spread_multiplier
        slope_component = (abs(curr_fast - prev_fast) / price_now) * self.slope_multiplier
        bonus = self.cross_bonus if (bullish_cross or bearish_cross) else 0.0
        confidence = min(1.0, spread_component + slope_component + bonus) if side != "flat" else 0.0
        return TradeSignal(
            self.name,
            side,
            confidence,
            max(vol * self.sl_atr_multiple, self.sl_floor),
            max(vol * self.tp_atr_multiple, self.tp_floor),
        )


@dataclass
class M15EMACrossoverClosedConfirmProposal(M15EMACrossoverProposal):
    name: str = "ema_crossover"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < max(self.fast, self.slow) + 4:
            return TradeSignal(self.name, "flat", 0.0, self.sl_floor, self.tp_floor)
        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        vol = float(atr(df).iloc[-1])

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
            max(vol * self.sl_atr_multiple, self.sl_floor),
            max(vol * self.tp_atr_multiple, self.tp_floor),
        )


@dataclass
class M15PullbackTrendProposal(Strategy):
    fast: int = 8
    slow: int = 21
    buy_band: float = 1.0006
    sell_band: float = 0.9994
    confidence_scale: float = 180.0
    sl_atr_multiple: float = 1.3
    sl_floor: float = 150.0
    tp_atr_multiple: float = 2.1
    tp_floor: float = 240.0
    name: str = "pullback_trend"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        if len(df) < max(self.fast, self.slow) + 3:
            return TradeSignal(self.name, "flat", 0.0, self.sl_floor, self.tp_floor)
        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        cur = float(close.iloc[-1])
        vol = float(atr(df).iloc[-1])
        side = "flat"
        confidence = 0.0
        if float(e_fast.iloc[-1]) > float(e_slow.iloc[-1]) and cur <= float(e_fast.iloc[-1]) * self.buy_band:
            side = "buy"
            confidence = min(1.0, abs(float(e_fast.iloc[-1]) - cur) / max(cur, 1.0) * self.confidence_scale)
        elif float(e_fast.iloc[-1]) < float(e_slow.iloc[-1]) and cur >= float(e_fast.iloc[-1]) * self.sell_band:
            side = "sell"
            confidence = min(1.0, abs(float(e_fast.iloc[-1]) - cur) / max(cur, 1.0) * self.confidence_scale)
        return TradeSignal(
            self.name,
            side,
            confidence,
            max(vol * self.sl_atr_multiple, self.sl_floor),
            max(vol * self.tp_atr_multiple, self.tp_floor),
        )


def _skip_reasons(strategy: Strategy, df: pd.DataFrame, settings: Any) -> dict[str, int]:
    counts = {
        "insufficient_history": 0,
        "flat_signal": 0,
        "below_confidence_threshold": 0,
        "research_regime_filter": 0,
        "conservative_filter": 0,
        "backtest_trade_taken": 0,
    }
    min_conf = float(getattr(settings, "research_confidence_threshold", 0.1))
    i = 250
    while i < len(df) - 2:
        window = df.iloc[: i + 1]
        if len(window) < max(int(getattr(strategy, "fast", 0) or 0), int(getattr(strategy, "slow", 0) or 0)) + 3:
            counts["insufficient_history"] += 1
            i += 1
            continue
        signal = strategy.generate_signal(window)
        if signal.side == "flat":
            counts["flat_signal"] += 1
            i += 1
            continue
        if float(signal.confidence) < min_conf:
            counts["below_confidence_threshold"] += 1
            i += 1
            continue
        overlay_context = build_research_regime_overlay_context(
            window,
            window.iloc[-1]["time"] if "time" in window.columns else None,
            spread_points=0.0,
        )
        if research_regime_filter_reason(settings, overlay_context):
            counts["research_regime_filter"] += 1
            i += 1
            continue
        if conservative_pretrade_filter_reason(settings, overlay_context):
            counts["conservative_filter"] += 1
            i += 1
            continue
        counts["backtest_trade_taken"] += 1
        i += 1
    return counts


def _run_one(strategy: Strategy, df: pd.DataFrame, settings: Any) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="atlas_m15_proposal_") as tmp:
        run_settings = replace(
            settings,
            storage_dir=tmp,
            execute_trades=False,
            timeframe="M15",
            enabled_strategies=(strategy.name,),
            backtest_include_evolved=False,
            backtest_persist_learning_state=False,
        )
        backtester = Backtester(store=KnowledgeStore(tmp), settings=run_settings)
        backtester.base = [strategy]
        result = backtester.run(df)
    return {
        "parameters": dict(strategy.__dict__),
        "metrics": _result_payload(result),
        "skip_reasons": _skip_reasons(strategy, df, run_settings),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest M15 parameter proposal variants.")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--label", default="m15_ema_pullback_proposals")
    args = parser.parse_args()

    settings = replace(load_settings(), execute_trades=False, timeframe="M15", backtest_include_evolved=False)
    start = _parse_dt(args.start)
    end = _parse_dt(args.end)
    feed = MarketDataFeed(settings)
    df = feed.get_rates_range(start, end)
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": "M15",
            "symbol": settings.symbol,
            "bars": int(len(df)),
            "reason": "Same fixed recent multi-session window as the M1 reference baseline.",
        },
        "safety": {
            "execute_trades": False,
            "note": "Offline backtest harness only; production strategy code and live config are not modified.",
        },
        "proposals": {
            "ema_crossover": _run_one(M15EMACrossoverProposal(), df, settings),
            "ema_crossover_closed_confirm": _run_one(M15EMACrossoverClosedConfirmProposal(), df, settings),
            "ema_crossover_closed_confirm_bar_sized_exit": _run_one(
                M15EMACrossoverClosedConfirmProposal(
                    sl_atr_multiple=2.5,
                    sl_floor=1470.0,
                    tp_atr_multiple=4.0,
                    tp_floor=2350.0,
                ),
                df,
                settings,
            ),
            "ema_crossover_closed_confirm_mid_exit": _run_one(
                M15EMACrossoverClosedConfirmProposal(
                    sl_atr_multiple=2.0,
                    sl_floor=1100.0,
                    tp_atr_multiple=2.6,
                    tp_floor=1450.0,
                ),
                df,
                settings,
            ),
            "ema_crossover_wide_exit": _run_one(
                M15EMACrossoverProposal(
                    sl_atr_multiple=2.5,
                    sl_floor=250.0,
                    tp_atr_multiple=4.0,
                    tp_floor=400.0,
                ),
                df,
                settings,
            ),
            "pullback_trend": _run_one(M15PullbackTrendProposal(), df, settings),
            "pullback_trend_corrected_exit": _run_one(
                M15PullbackTrendProposal(
                    sl_atr_multiple=2.0,
                    sl_floor=1100.0,
                    tp_atr_multiple=2.6,
                    tp_floor=1450.0,
                ),
                df,
                settings,
            ),
            "pullback_trend_m15_mid_exit": _run_one(PullbackTrendM15MidExitStrategy(), df, settings),
        },
    }
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in args.label).strip("_")
    path = BASELINE_DIR / f"baseline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_label or 'm15_ema_pullback_proposals'}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(str(path))


if __name__ == "__main__":
    main()
