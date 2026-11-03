from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import load_settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.market_context import build_context_snapshot
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.regime_filters import (
    build_research_regime_overlay_context,
    conservative_pretrade_filter_reason,
    research_regime_filter_reason,
)
from atlas_ai.trader import AtlasTrader
from atlas_ai.models import TradeSignal


BASELINE_DIR = PROJECT_ROOT / "docs" / "baselines"


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "timing_discipline_matched"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(ordered[int(k)])
    return float(ordered[lower] * (upper - k) + ordered[upper] * (k - lower))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_r": 0.0,
            "max_drawdown_r": 0.0,
            "max_drawdown_pct_at_configured_risk": 0.0,
            "median_r": 0.0,
            "exit_reasons": {},
            "hold_bars_distribution": {},
            "reason_codes": {},
            "by_strategy": {},
        }
    wins = [float(row["r_multiple"]) for row in rows if float(row["r_multiple"]) > 0.0]
    losses = [abs(float(row["r_multiple"])) for row in rows if float(row["r_multiple"]) < 0.0]
    classified = len(wins) + len(losses)
    r_values = [float(row["r_multiple"]) for row in rows]
    cumulative_r = 0.0
    peak_r = 0.0
    max_drawdown_r = 0.0
    for value in r_values:
        cumulative_r += value
        peak_r = max(peak_r, cumulative_r)
        max_drawdown_r = max(max_drawdown_r, peak_r - cumulative_r)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row.get("strategy") or "unknown")].append(row)
    return {
        "count": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "flat_or_zero": len(rows) - classified,
        "win_rate": (len(wins) / classified) if classified else 0.0,
        "profit_factor": (sum(wins) / sum(losses)) if losses else (999.0 if wins else 0.0),
        "average_r": sum(r_values) / len(r_values),
        "max_drawdown_r": max_drawdown_r,
        "max_drawdown_pct_at_configured_risk": max_drawdown_r * 0.005 * 100.0,
        "median_r": _percentile(r_values, 0.5),
        "exit_reasons": dict(Counter(str(row.get("exit_reason") or "unknown") for row in rows)),
        "hold_bars_distribution": dict(Counter(str(int(row.get("hold_bars") or 0)) for row in rows)),
        "reason_codes": dict(Counter(str(row.get("reason_code") or "PASSED") for row in rows)),
        "by_strategy": {
            strategy: {
                "count": len(strategy_rows),
                "win_rate": _summarize(strategy_rows)["win_rate"] if len(strategy_rows) != len(rows) else None,
                "average_r": sum(float(row["r_multiple"]) for row in strategy_rows) / len(strategy_rows),
                "exit_reasons": dict(Counter(str(row.get("exit_reason") or "unknown") for row in strategy_rows)),
                "reason_codes": dict(Counter(str(row.get("reason_code") or "PASSED") for row in strategy_rows)),
            }
            for strategy, strategy_rows in sorted(by_strategy.items())
        },
    }


def _precompute_replay_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    for span in (5, 9, 13, 20, 21, 50):
        out[f"_replay_ema_{span}"] = close.ewm(span=span, adjust=False).mean()
    out["_replay_roc_12"] = ((close - close.shift(12)) / close.shift(12).replace(0, math.nan)) * 100.0
    high_low = out["high"] - out["low"]
    high_close = (out["high"] - out["close"].shift()).abs()
    low_close = (out["low"] - out["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["_replay_atr_14"] = true_range.rolling(14).mean()
    out["_replay_regime_first_30"] = close.shift(29)
    out["_replay_regime_abs_move_avg_30"] = close.diff().abs().rolling(29).mean()
    return out


def _safe_precomputed_vol(df: pd.DataFrame, idx: int) -> float:
    try:
        vol = float(df.iloc[idx]["_replay_atr_14"])
    except Exception:
        return 120.0
    if not vol or vol != vol:
        return 120.0
    return vol


def _precomputed_regime_stats(df: pd.DataFrame, idx: int, lookback: int = 30) -> tuple[str, float, float]:
    if idx + 1 < max(12, lookback // 2):
        return "unknown", 0.0, 0.0
    row = df.iloc[idx]
    try:
        first = float(row["_replay_regime_first_30"])
        last = float(row["close"])
        avg_abs = float(row["_replay_regime_abs_move_avg_30"])
    except Exception:
        return "unknown", 0.0, 0.0
    if first == 0.0 or first != first or avg_abs != avg_abs:
        return "unknown", 0.0, 0.0
    drift_pct = ((last - first) / first) * 100.0
    vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
    if abs(drift_pct) >= max(0.12, vol_pct * 2.2):
        return "trend", abs(drift_pct), vol_pct
    if vol_pct >= 0.06:
        return "high_volatility_range", abs(drift_pct), vol_pct
    return "range", abs(drift_pct), vol_pct


def _fast_signal_for_strategy(strategy: Any, df: pd.DataFrame, idx: int) -> TradeSignal | None:
    name = str(getattr(strategy, "name", "") or "")
    row = df.iloc[idx]
    close = float(row["close"])
    vol = _safe_precomputed_vol(df, idx)

    if name == "ema_crossover":
        fast = int(getattr(strategy, "fast", 20))
        slow = int(getattr(strategy, "slow", 50))
        if idx + 1 < max(fast, slow) + 3:
            return TradeSignal(name, "flat", 0.0, 120.0, 180.0)
        prev = df.iloc[idx - 1]
        prev_fast = float(prev[f"_replay_ema_{fast}"])
        prev_slow = float(prev[f"_replay_ema_{slow}"])
        curr_fast = float(row[f"_replay_ema_{fast}"])
        curr_slow = float(row[f"_replay_ema_{slow}"])
        bullish_cross = prev_fast <= prev_slow and curr_fast > curr_slow
        bearish_cross = prev_fast >= prev_slow and curr_fast < curr_slow
        if curr_fast > curr_slow:
            side = "buy"
        elif curr_fast < curr_slow:
            side = "sell"
        else:
            side = "flat"
        price_now = max(close, 1e-9)
        spread_component = (abs(curr_fast - curr_slow) / price_now) * 700.0
        slope_component = (abs(curr_fast - prev_fast) / price_now) * 320.0
        cross_bonus = 0.18 if (bullish_cross or bearish_cross) else 0.0
        confidence = min(1.0, spread_component + slope_component + cross_bonus) if side != "flat" else 0.0
        return TradeSignal(name, side, confidence, max(vol * 2, 100), max(vol * 3, 150))

    if name == "momentum_continuation":
        regime, trend_strength, volatility_pct = _precomputed_regime_stats(df, idx, lookback=30)
        if regime != "trend" or trend_strength < 0.18 or volatility_pct < 0.03:
            return TradeSignal(name, "flat", 0.0, max(vol * 1.9, 115), max(vol * 2.8, 175))
        r = float(row["_replay_roc_12"])
        fast = float(row["_replay_ema_9"])
        slow = float(row["_replay_ema_21"])
        side = "flat"
        confidence = 0.0
        trend_weight = min(1.25, max(0.8, trend_strength / 0.24))
        if r > 0.08 and fast > slow:
            side = "buy"
            confidence = min(1.0, (float(r) / 0.6) * trend_weight)
        elif r < -0.08 and fast < slow:
            side = "sell"
            confidence = min(1.0, (abs(float(r)) / 0.6) * trend_weight)
        return TradeSignal(name, side, confidence, max(vol * 1.9, 115), max(vol * 2.8, 175))

    if name == "pullback_trend":
        e20 = float(row["_replay_ema_20"])
        e50 = float(row["_replay_ema_50"])
        side = "flat"
        confidence = 0.0
        if e20 > e50 and close <= e20 * 1.0003:
            side = "buy"
            confidence = min(1.0, abs(e20 - close) / max(close, 1.0) * 300)
        elif e20 < e50 and close >= e20 * 0.9997:
            side = "sell"
            confidence = min(1.0, abs(e20 - close) / max(close, 1.0) * 300)
        return TradeSignal(name, side, confidence, max(vol * 1.8, 110), max(vol * 2.6, 170))

    if name == "gold_breakout_retest":
        lookback = int(getattr(strategy, "lookback", 28))
        confirmation_window = int(getattr(strategy, "confirmation_window", 6))
        if idx + 1 < lookback + confirmation_window + 10:
            return TradeSignal(name, "flat", 0.0, 105.0, 155.0)
        start = idx + 1 - (lookback + confirmation_window)
        recent = df.iloc[start : idx + 1]
        base = recent.iloc[:lookback]
        confirm = recent.iloc[lookback:]
        last = confirm.iloc[-1]
        prev = confirm.iloc[-2]
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        avg_vol = float(recent[vol_col].tail(18).mean() or 1.0)
        last_vol = float(last.get(vol_col, 0.0) or 0.0)
        o = float(last["open"])
        h = float(last["high"])
        low = float(last["low"])
        c = float(last["close"])
        rng = max(1e-9, h - low)
        body_ratio = abs(c - o) / rng
        prior_high = float(base["high"].max())
        prior_low = float(base["low"].min())
        tolerance = max(vol * 0.28, 0.8)
        closes = [float(x) for x in confirm["close"].tolist()]
        lows = [float(x) for x in confirm["low"].tolist()]
        highs = [float(x) for x in confirm["high"].tolist()]
        buy_break = any(x > prior_high + tolerance * 0.15 for x in closes[:-1])
        buy_retest = any(
            low_i <= prior_high + tolerance and close_i >= prior_high - tolerance * 0.45
            for low_i, close_i in zip(lows[1:], closes[1:])
        )
        sell_break = any(x < prior_low - tolerance * 0.15 for x in closes[:-1])
        sell_retest = any(
            high_i >= prior_low - tolerance and close_i <= prior_low + tolerance * 0.45
            for high_i, close_i in zip(highs[1:], closes[1:])
        )
        side = "flat"
        confidence = 0.0
        volume_factor = min(0.18, max(0.0, (last_vol / max(avg_vol, 1.0)) - 1.0) * 0.12)
        e5 = float(row["_replay_ema_5"])
        e13 = float(row["_replay_ema_13"])
        if buy_break and buy_retest and c > max(prior_high + tolerance * 0.25, float(prev["close"])) and e5 > e13 and body_ratio >= 0.38:
            side = "buy"
            follow_through = min(0.28, max(0.0, (c - prior_high) / max(vol, 1.0)) * 0.18)
            confidence = min(0.72, 0.30 + body_ratio * 0.34 + follow_through + volume_factor)
        elif sell_break and sell_retest and c < min(prior_low - tolerance * 0.25, float(prev["close"])) and e5 < e13 and body_ratio >= 0.38:
            side = "sell"
            follow_through = min(0.28, max(0.0, (prior_low - c) / max(vol, 1.0)) * 0.18)
            confidence = min(0.72, 0.30 + body_ratio * 0.34 + follow_through + volume_factor)
        return TradeSignal(name, side, confidence, max(vol * 1.45, 100), max(vol * 2.15, 145))

    return None


def _generate_replay_signal(strategy: Any, full_df: pd.DataFrame, idx: int) -> TradeSignal:
    fast = _fast_signal_for_strategy(strategy, full_df, idx)
    if fast is not None:
        return fast
    return strategy.generate_signal(full_df.iloc[: idx + 1])


def _simulate_row(backtester: Backtester, df: pd.DataFrame, signal: Any, signal_idx: int, timing: dict[str, Any], group: str) -> dict[str, Any]:
    r_multiple, exit_idx, best_mfe_r = backtester._normalize_simulated_trade(backtester._simulate_trade(df, signal, signal_idx))
    details = dict(backtester._last_simulated_trade_details or {})
    entry_idx = int(details.get("entry_idx", signal_idx + 1) or signal_idx + 1)
    values = dict(timing.get("condition_values", {}) or {})
    observed = dict(values.get("observed", {}) or {})
    return {
        "group": group,
        "signal_bar": int(signal_idx),
        "entry_bar": int(entry_idx),
        "exit_bar": int(exit_idx),
        "signal_time": pd.Timestamp(df.iloc[signal_idx]["time"]).isoformat() if "time" in df.columns else "",
        "entry_time": pd.Timestamp(df.iloc[entry_idx]["time"]).isoformat() if "time" in df.columns and entry_idx < len(df) else "",
        "exit_time": pd.Timestamp(df.iloc[exit_idx]["time"]).isoformat() if "time" in df.columns and exit_idx < len(df) else "",
        "strategy": str(getattr(signal, "strategy", "unknown") or "unknown"),
        "side": str(getattr(signal, "side", "flat") or "flat"),
        "confidence": round(float(getattr(signal, "confidence", 0.0) or 0.0), 6),
        "stop_loss_points": float(getattr(signal, "stop_loss_points", 0.0) or 0.0),
        "take_profit_points": float(getattr(signal, "take_profit_points", 0.0) or 0.0),
        "r_multiple": round(float(r_multiple), 6),
        "best_mfe_r": round(float(best_mfe_r), 6),
        "hold_bars": max(1, int(exit_idx - signal_idx)),
        "exit_reason": str(details.get("exit_reason", "unknown") or "unknown"),
        "timing_passed": bool(timing.get("passed", False)),
        "variant_match": bool(timing.get("variant_match", False)),
        "variant_admitted": bool(timing.get("variant_admitted", False)),
        "reason_code": str(timing.get("reason_code") or ""),
        "timing_score": int(timing.get("score", values.get("score", 0)) or 0),
        "timing_missing": list(timing.get("missing", values.get("missing", [])) or []),
        "location": str(observed.get("location", "unknown") or "unknown"),
        "momentum": str(observed.get("momentum", "unknown") or "unknown"),
        "session": str(observed.get("session", "unknown") or "unknown"),
        "trend_state": str(observed.get("trend_state", "unknown") or "unknown"),
    }


def _apply_backtest_location_variant(timing: dict[str, Any], *, allow_breakout_location: bool) -> dict[str, Any]:
    if not allow_breakout_location:
        return timing
    out = dict(timing)
    condition_values = dict(out.get("condition_values", {}) or {})
    observed = dict(condition_values.get("observed", {}) or out.get("observed", {}) or {})
    if str(observed.get("location", "") or "").strip().lower() != "breakout":
        return timing

    missing = [str(item) for item in list(out.get("missing", []) or []) if str(item) != "fresh_entry_or_retest"]
    reasons = [
        str(item)
        for item in list(out.get("reasons", []) or [])
        if str(item) != "entry_location_is_not_fresh_retest_or_reaction_zone"
    ]
    out["missing"] = missing
    out["reasons"] = reasons
    out["hard_block"] = bool(str(out.get("side", "") or "").lower() in {"buy", "sell"} and missing)
    out["passed"] = not bool(out["hard_block"])
    out["status"] = "timing_block" if out["hard_block"] else "timing_clear_for_existing_gates"
    score = int(out.get("score", condition_values.get("score", 0)) or 0)
    out["score"] = min(100, score + 15)
    condition_values["score"] = out["score"]
    condition_values["missing"] = missing
    condition_values["reasons"] = reasons
    out["condition_values"] = condition_values
    if not out["hard_block"]:
        out["reason_code"] = None
        out["detail"] = (
            "Timing discipline passed in breakout-location backtest variant: breakout location "
            f"accepted with momentum={observed.get('momentum', 'unknown')}."
        )
    elif "strong_or_turning_m1_momentum" in missing:
        out["reason_code"] = "AWAITING_M1_MOMENTUM_CONFIRMATION"
    elif missing:
        out["reason_code"] = AtlasTrader._timing_reason_code(missing)
    structured = dict(out.get("structured_result", {}) or {})
    if structured:
        structured["passed"] = bool(out["passed"])
        structured["reason_code"] = out.get("reason_code")
        structured["detail"] = out.get("detail", structured.get("detail", ""))
        structured["condition_values"] = dict(condition_values)
        out["structured_result"] = structured
    return out


def _apply_mid_range_mixed_asia_variant(timing: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    """Replay-only relaxation for one exact observed timing combination."""
    if not enabled:
        return timing
    out = dict(timing)
    condition_values = dict(out.get("condition_values", {}) or {})
    observed = dict(condition_values.get("observed", {}) or out.get("observed", {}) or {})
    actual = tuple(str(observed.get(key, "") or "").strip().lower() for key in ("location", "momentum", "session"))
    if actual != ("mid_range", "mixed", "asia"):
        return timing

    relaxed_missing = {"tradable_location", "strong_or_turning_m1_momentum"}
    relaxed_reasons = {"entry_location_is_mid_range_or_unclear", "m1_momentum_not_confirmed"}
    missing = [str(item) for item in list(out.get("missing", []) or []) if str(item) not in relaxed_missing]
    reasons = [str(item) for item in list(out.get("reasons", []) or []) if str(item) not in relaxed_reasons]
    reasons.append("mid_range_mixed_accepted_in_asia_replay_variant")
    out["missing"] = missing
    out["reasons"] = reasons
    out["hard_block"] = bool(str(out.get("side", "") or "").lower() in {"buy", "sell"} and missing)
    out["passed"] = not out["hard_block"]
    out["status"] = "timing_block" if out["hard_block"] else "timing_clear_for_existing_gates"
    out["variant_match"] = True
    out["variant_admitted"] = bool(out["passed"] and not timing.get("passed", False))
    condition_values["missing"] = missing
    condition_values["reasons"] = reasons
    out["condition_values"] = condition_values
    if out["passed"]:
        out["reason_code"] = None
        out["detail"] = "Timing discipline passed in replay variant for location=mid_range, momentum=mixed, session=asia."
    elif missing:
        out["reason_code"] = AtlasTrader._timing_reason_code(missing)
    structured = dict(out.get("structured_result", {}) or {})
    if structured:
        structured.update(passed=bool(out["passed"]), reason_code=out.get("reason_code"),
                          detail=out.get("detail", structured.get("detail", "")), condition_values=condition_values)
        out["structured_result"] = structured
    return out


def run_matched_backtest(
    start: datetime,
    end: datetime,
    label: str,
    *,
    sample_limit: int = 250,
    allow_breakout_location: bool = False,
    allow_mid_range_mixed_asia: bool = False,
    confidence_threshold: float | None = None,
) -> Path:
    settings = replace(load_settings(), timeframe="M1", execute_trades=False, backtest_persist_learning_state=False)
    feed = MarketDataFeed(settings)
    df = feed.get_rates_range(start, end)
    if df.empty:
        raise SystemExit("No M1 bars returned for requested window.")
    replay_df = _precompute_replay_indicators(df)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = BASELINE_DIR / f"baseline_{timestamp}_{_safe_label(label)}.json"

    with tempfile.TemporaryDirectory(prefix="atlas_timing_matched_") as tmp:
        store = KnowledgeStore(tmp)
        backtester = Backtester(store=store, settings=settings)
        trader = AtlasTrader(settings=settings, store=store)
        evolved = None
        include_evolved = bool(getattr(settings, "backtest_include_evolved", False))
        refresh_bars = max(0, int(getattr(settings, "backtest_evolved_refresh_bars", 500) or 500))
        min_conf = float(confidence_threshold if confidence_threshold is not None else getattr(settings, "research_confidence_threshold", 0.1) or 0.1)

        blocked_rows: list[dict[str, Any]] = []
        allowed_rows: list[dict[str, Any]] = []
        skipped = Counter()

        i = 250
        while i < len(replay_df) - 2:
            window = replay_df.iloc[: i + 1]
            if include_evolved and (evolved is None or (refresh_bars and (i - 250) % refresh_bars == 0)):
                evolved = backtester._evolved_strategy(window)

            strategies = [s for s in backtester.base if backtester._strategy_enabled(getattr(s, "name", ""))]
            if include_evolved and evolved is not None:
                strategies.append(evolved)
            signals = [_generate_replay_signal(s, replay_df, i) for s in strategies]
            best = backtester.allocator.choose(signals)
            if str(best.side) == "flat":
                skipped["flat"] += 1
                i += 1
                continue
            if float(best.confidence or 0.0) < min_conf:
                skipped["below_research_confidence"] += 1
                i += 1
                continue

            when = window.iloc[-1]["time"] if "time" in window.columns else None
            overlay = build_research_regime_overlay_context(window, when, spread_points=0.0)
            if research_regime_filter_reason(settings, overlay):
                skipped["research_regime_filter"] += 1
                i += 1
                continue
            if conservative_pretrade_filter_reason(settings, overlay):
                skipped["conservative_pretrade_filter"] += 1
                i += 1
                continue

            regime = {
                "name": str(overlay.get("regime_name", "unknown") or "unknown"),
                "trend_strength": float(overlay.get("trend_strength_pct", 0.0) or 0.0),
                "volatility_pct": float(overlay.get("volatility_pct", 0.0) or 0.0),
            }
            chart_features = trader._derive_runtime_chart_features(df=window, chosen=best, regime=regime)
            status = {
                "symbol": settings.symbol,
                "timeframe": settings.timeframe,
                "session": str(overlay.get("session", "") or ""),
                "regime": regime,
                "research_regime_overlay": overlay,
                "chart_features": chart_features,
                "chosen": {
                    "strategy": str(best.strategy),
                    "side": str(best.side),
                    "confidence": float(best.confidence),
                    "stop_loss_points": float(best.stop_loss_points),
                    "take_profit_points": float(best.take_profit_points),
                },
                "risk": {"spread_points": 0.0},
            }
            context = build_context_snapshot(status)
            timing = trader._timing_discipline_snapshot(best, context, status)
            timing = _apply_backtest_location_variant(timing, allow_breakout_location=allow_breakout_location)
            timing = _apply_mid_range_mixed_asia_variant(timing, enabled=allow_mid_range_mixed_asia)
            group = "allowed" if bool(timing.get("passed", False)) else "blocked"
            row = _simulate_row(backtester, replay_df, best, i, timing, group)
            if group == "blocked":
                blocked_rows.append(row)
                i += 1
                continue

            allowed_rows.append(row)
            risk_fraction = float(getattr(settings, "max_risk_per_trade", 0.005) or 0.005)
            trade_return = float(row["r_multiple"]) * risk_fraction
            spread_cost = float(getattr(settings, "backtest_spread_cost_bps", 0.0) or 0.0) / 10000.0
            slippage_cost = float(getattr(settings, "backtest_slippage_cost_bps", 0.0) or 0.0) / 10000.0
            backtester.allocator.update(str(best.strategy), trade_return - spread_cost - slippage_cost)
            i += 1

    payload = {
        "schema": "atlas_timing_discipline_matched_backtest_v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": "M1",
            "symbol": settings.symbol,
            "bars": int(len(replay_df)),
        },
        "method": {
            "source": "single offline backtest candidate stream with AtlasTrader timing discipline applied",
            "population": "same generated best candidates after normal backtest confidence and regime filters",
            "split": "timing_discipline passed vs timing_discipline blocked",
            "blocked_handling": "blocked rows are simulated hypothetically but do not update allocator/equity",
            "allowed_handling": "allowed rows are simulated and update allocator, matching the gated sequence",
            "safety": "offline analysis only; execute_trades=false; no live execution path invoked",
            "variant": {
                "allow_breakout_location": bool(allow_breakout_location),
                "allow_mid_range_mixed_asia": bool(allow_mid_range_mixed_asia),
                "confidence_threshold": min_conf,
                "note": "Backtest-only timing interpretation; live timing-discipline code is unchanged.",
            },
        },
        "skipped_pre_timing": dict(skipped),
        "summaries": {
            "allowed": _summarize(allowed_rows),
            "variant_added": _summarize([row for row in allowed_rows if row.get("variant_admitted")]),
            "blocked_hypothetical": _summarize(blocked_rows),
        },
        "samples": {
            "allowed": allowed_rows[:sample_limit],
            "blocked_hypothetical": blocked_rows[:sample_limit],
        },
        "data_fetch": dict(getattr(feed, "last_fetch_meta", {}) or {}),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched timing-discipline offline comparison.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--label", default="timing_discipline_matched")
    parser.add_argument("--sample-limit", type=int, default=250)
    parser.add_argument("--allow-breakout-location", action="store_true")
    parser.add_argument("--allow-mid-range-mixed-asia", action="store_true")
    parser.add_argument("--confidence-threshold", type=float)
    args = parser.parse_args()
    path = run_matched_backtest(
        _parse_dt(args.start),
        _parse_dt(args.end),
        args.label,
        sample_limit=max(0, int(args.sample_limit)),
        allow_breakout_location=bool(args.allow_breakout_location),
        allow_mid_range_mixed_asia=bool(args.allow_mid_range_mixed_asia),
        confidence_threshold=args.confidence_threshold,
    )
    print(path)


if __name__ == "__main__":
    main()
