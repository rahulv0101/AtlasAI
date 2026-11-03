from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import Settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.execution.broker import HFMConnector
from atlas_ai.regime_filters import session_label


@dataclass
class LabeledTrade:
    symbol: str
    entry_time: str
    exit_time: str
    session: str
    regime: str
    trend_strength_ratio: float
    volatility_pct: float
    volatility_bucket: str
    confidence: float
    confidence_bucket: str
    side: str
    strategy: str
    repeat_entry: bool
    same_side_support: int
    same_side_ratio: float
    body_ratio: float
    volume_ratio: float
    spike_ratio: float
    spread_cost_bps_assumed: float
    slippage_cost_bps_assumed: float
    hold_bars: int
    mfe_r: float
    mae_r: float
    trade_return: float
    outcome: str
    exit_type: str
    label: str
    label_confidence: str
    reason_codes: list[str]


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.60:
        return "mid"
    return "low"


def _volatility_metrics(window: pd.DataFrame) -> tuple[float, str]:
    closes = [float(x) for x in window["close"].tail(60).tolist()]
    if len(closes) < 10:
        return 0.0, "unknown"
    abs_moves = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    avg_abs = sum(abs_moves) / max(1, len(abs_moves))
    last = abs(float(closes[-1]))
    vol_pct = (avg_abs / max(1e-9, last)) * 100.0
    if vol_pct < 0.03:
        return float(vol_pct), "low_volatility"
    if vol_pct >= 0.07:
        return float(vol_pct), "high_volatility"
    return float(vol_pct), "normal_volatility"


def _trend_strength_ratio(window: pd.DataFrame) -> float:
    closes = [float(x) for x in window["close"].tail(60).tolist()]
    if len(closes) < 30:
        return 0.0
    first = float(closes[0])
    last = float(closes[-1])
    if abs(first) < 1e-9:
        return 0.0
    drift_pct = abs((last - first) / first) * 100.0
    abs_moves = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    avg_abs = sum(abs_moves) / max(1, len(abs_moves))
    vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
    return float(drift_pct / max(vol_pct, 1e-9))


def _signal_micro_context(window: pd.DataFrame) -> tuple[float, float]:
    recent = window.tail(12).copy()
    if len(recent) < 8:
        return 0.0, 1.0
    last = recent.iloc[-1]
    candle_range = max(1e-9, float(last["high"]) - float(last["low"]))
    candle_body = abs(float(last["close"]) - float(last["open"]))
    body_ratio = candle_body / candle_range
    vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
    last_vol = float(last.get(vol_col, 0.0) or 0.0)
    avg_vol = float(recent[vol_col].tail(8).mean() or 1.0)
    volume_ratio = last_vol / max(avg_vol, 1.0)
    return round(body_ratio, 4), round(volume_ratio, 4)


def _spike_ratio(window: pd.DataFrame) -> float:
    recent = window.tail(14).copy()
    if len(recent) < 5:
        return 0.0
    ranges = [float(row.high) - float(row.low) for row in recent.itertuples(index=False)]
    atr_like = sum(ranges) / max(1, len(ranges))
    return float(ranges[-1] / max(1e-9, atr_like))


def _same_side_support_count(signals, side: str) -> int:
    target = str(side)
    return sum(1 for s in signals if str(getattr(s, "side", "")) == target and float(getattr(s, "confidence", 0.0) or 0.0) > 0.0)


def _simulate_trade_path(df: pd.DataFrame, signal, signal_idx: int, settings: Settings) -> dict[str, Any]:
    point_size = max(1e-6, float(getattr(settings, "backtest_point_size", 0.01) or 0.01))
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return {"exit_idx": signal_idx, "r_multiple": 0.0, "mfe_r": 0.0, "mae_r": 0.0, "exit_type": "no_entry"}
    entry_price = float(df.iloc[entry_idx]["open"])
    stop_points = max(1.0, float(getattr(signal, "stop_loss_points", 0.0) or 0.0))
    take_points = max(stop_points * 0.8, float(getattr(signal, "take_profit_points", 0.0) or 0.0))
    stop_distance = stop_points * point_size
    take_distance = take_points * point_size
    max_hold_bars = max(1, int(getattr(settings, "backtest_max_hold_bars", 12) or 12))
    break_even_r = max(0.1, float(getattr(settings, "backtest_break_even_r_trigger", 0.9) or 0.9))
    break_even_lock = max(0.0, float(getattr(settings, "backtest_break_even_lock_r", 0.05) or 0.05))
    trailing_r = max(0.2, float(getattr(settings, "backtest_trailing_r_trigger", 1.4) or 1.4))
    trailing_buffer = max(0.1, float(getattr(settings, "backtest_trailing_buffer_r", 0.9) or 0.9))
    stale_exit_bars = max(1, int(getattr(settings, "backtest_stale_exit_bars", 4) or 4))
    stale_min_progress = max(0.0, float(getattr(settings, "backtest_stale_min_progress_r", 0.15) or 0.15))

    if str(signal.side) == "buy":
        stop_price = entry_price - stop_distance
        take_price = entry_price + take_distance
    else:
        stop_price = entry_price + stop_distance
        take_price = entry_price - take_distance

    best_mfe_r = 0.0
    worst_mae_r = 0.0
    r_multiple = 0.0
    exit_idx = min(len(df) - 1, entry_idx + max_hold_bars - 1)
    exit_price = float(df.iloc[exit_idx]["close"])
    exit_type = "time_exit"

    for j in range(entry_idx, min(len(df), entry_idx + max_hold_bars)):
        row = df.iloc[j]
        open_px = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        if j > entry_idx:
            if best_mfe_r >= break_even_r:
                if str(signal.side) == "buy":
                    stop_price = max(stop_price, entry_price + (break_even_lock * stop_distance))
                else:
                    stop_price = min(stop_price, entry_price - (break_even_lock * stop_distance))
            if best_mfe_r >= trailing_r:
                locked_r = max(break_even_lock, best_mfe_r - trailing_buffer)
                if str(signal.side) == "buy":
                    stop_price = max(stop_price, entry_price + (locked_r * stop_distance))
                else:
                    stop_price = min(stop_price, entry_price - (locked_r * stop_distance))
        if str(signal.side) == "buy":
            favorable_r = max(0.0, (high - entry_price) / max(stop_distance, 1e-9))
            adverse_r = max(0.0, (entry_price - low) / max(stop_distance, 1e-9))
            hit_stop = low <= stop_price
            hit_take = high >= take_price
            stop_r = (stop_price - entry_price) / max(stop_distance, 1e-9)
            take_r = (take_price - entry_price) / max(stop_distance, 1e-9)
            if hit_stop and hit_take:
                bullish_bar = close >= open_px
                exit_price = take_price if bullish_bar else stop_price
                r_multiple = take_r if bullish_bar else stop_r
                exit_type = "take_profit" if bullish_bar else ("breakeven_stop" if stop_r >= 0 else "stop_loss")
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
            if hit_stop:
                exit_price = stop_price
                r_multiple = stop_r
                exit_type = "breakeven_stop" if stop_r >= 0 else "stop_loss"
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
            if hit_take:
                exit_price = take_price
                r_multiple = take_r
                exit_type = "take_profit"
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
        else:
            favorable_r = max(0.0, (entry_price - low) / max(stop_distance, 1e-9))
            adverse_r = max(0.0, (high - entry_price) / max(stop_distance, 1e-9))
            hit_stop = high >= stop_price
            hit_take = low <= take_price
            stop_r = (entry_price - stop_price) / max(stop_distance, 1e-9)
            take_r = (entry_price - take_price) / max(stop_distance, 1e-9)
            if hit_stop and hit_take:
                bearish_bar = close <= open_px
                exit_price = take_price if bearish_bar else stop_price
                r_multiple = take_r if bearish_bar else stop_r
                exit_type = "take_profit" if bearish_bar else ("breakeven_stop" if stop_r >= 0 else "stop_loss")
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
            if hit_stop:
                exit_price = stop_price
                r_multiple = stop_r
                exit_type = "breakeven_stop" if stop_r >= 0 else "stop_loss"
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
            if hit_take:
                exit_price = take_price
                r_multiple = take_r
                exit_type = "take_profit"
                exit_idx = j
                best_mfe_r = max(best_mfe_r, favorable_r)
                worst_mae_r = max(worst_mae_r, adverse_r)
                break
        best_mfe_r = max(best_mfe_r, favorable_r)
        worst_mae_r = max(worst_mae_r, adverse_r)
        if abs(r_multiple) < 1e-12 and ((j - entry_idx) + 1) >= stale_exit_bars and best_mfe_r < stale_min_progress:
            exit_price = close
            move = (exit_price - entry_price) if str(signal.side) == "buy" else (entry_price - exit_price)
            r_multiple = move / max(stop_distance, 1e-9)
            exit_type = "stale_exit"
            exit_idx = j
            break
        if j == min(len(df), entry_idx + max_hold_bars) - 1 and abs(r_multiple) < 1e-12:
            exit_price = close
            move = (exit_price - entry_price) if str(signal.side) == "buy" else (entry_price - exit_price)
            r_multiple = move / max(stop_distance, 1e-9)
            exit_type = "time_exit"
            exit_idx = j

    return {
        "exit_idx": int(exit_idx),
        "r_multiple": float(r_multiple),
        "mfe_r": float(best_mfe_r),
        "mae_r": float(worst_mae_r),
        "exit_type": str(exit_type),
    }


def label_trade(trade: dict[str, Any]) -> tuple[str, list[str], str]:
    negative_reasons: list[str] = []
    positive_reasons: list[str] = []
    severe_quality = False

    if str(trade.get("session", "")) == "late_us" and str(trade.get("regime", "")) == "trend":
        negative_reasons.append("late_us_trend")
        severe_quality = True
    if bool(trade.get("post_spike_context", False)):
        negative_reasons.append("post_spike")
        severe_quality = True
    if bool(trade.get("london_extreme_volatility", False)):
        negative_reasons.append("london_drag")
    if bool(trade.get("trend_low_volatility", False)):
        negative_reasons.append("low_vol_trend")
    if bool(trade.get("weak_consensus", False)):
        negative_reasons.append("weak_confirmation")
    if bool(trade.get("poor_liquidity", False)):
        negative_reasons.append("poor_liquidity")
    if bool(trade.get("repeated_reentry", False)):
        negative_reasons.append("repeated_reentry")
    if bool(trade.get("first_entry_quality", False)):
        positive_reasons.append("first_entry_quality")
    if bool(trade.get("strong_trend_clean_entry", False)):
        positive_reasons.append("strong_trend_clean_entry")
    if bool(trade.get("favorable_session", False)):
        positive_reasons.append("favorable_session")
    if bool(trade.get("high_quality_confirmation", False)):
        positive_reasons.append("high_quality_confirmation")

    outcome = str(trade.get("outcome", "flat"))
    negative_count = len(set(negative_reasons))
    positive_count = len(set(positive_reasons))

    if severe_quality or negative_count >= 2:
        label = "low-quality setup"
        reasons = list(dict.fromkeys(negative_reasons))
    elif outcome == "loss" and negative_count >= 1:
        label = "avoidable loser"
        reasons = list(dict.fromkeys(negative_reasons))
    elif outcome == "win" and positive_count >= 2 and negative_count == 0:
        label = "high-quality winner"
        reasons = list(dict.fromkeys(positive_reasons))
    else:
        label = "acceptable trade"
        reasons = ["mixed_context"] if (negative_count > 0 and positive_count > 0) else (list(dict.fromkeys(positive_reasons))[:1] or list(dict.fromkeys(negative_reasons))[:1] or ["neutral_context"])

    confidence = "medium"
    if label == "low-quality setup" and len(reasons) >= 2:
        confidence = "high"
    elif label == "avoidable loser" and len(reasons) >= 1:
        confidence = "high"
    elif label == "high-quality winner" and len(reasons) >= 2:
        confidence = "high"
    elif not reasons:
        confidence = "low"
    return label, reasons, confidence


def _trade_records(df: pd.DataFrame, settings: Settings) -> list[dict[str, Any]]:
    bt = Backtester(settings=settings)
    include_evolved = bool(getattr(settings, "backtest_include_evolved", False))
    refresh_bars = max(0, int(getattr(settings, "backtest_evolved_refresh_bars", 500)))
    evolved = None
    i = 250
    records: list[dict[str, Any]] = []
    prev_side = ""
    prev_regime = ""
    while i < len(df) - 2:
        window = df.iloc[: i + 1]
        if include_evolved and (evolved is None or (refresh_bars and (i - 250) % refresh_bars == 0)):
            evolved = bt._evolved_strategy(window)
        strategies = [s for s in bt.base if bt._strategy_enabled(getattr(s, "name", ""))]
        if include_evolved and evolved is not None:
            strategies.append(evolved)
        signals = [s.generate_signal(window) for s in strategies]
        best = bt.allocator.choose(signals)
        if best.side == "flat" or best.confidence < float(getattr(settings, "research_confidence_threshold", 0.1)):
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= len(df):
            break
        entry_time = pd.Timestamp(df.iloc[entry_idx]["time"]).to_pydatetime()
        exit_path = _simulate_trade_path(df, best, i, settings)
        exit_idx = int(exit_path["exit_idx"])
        hold_bars = max(1, int(exit_idx - i))
        trade_return = (float(exit_path["r_multiple"]) * float(getattr(settings, "max_risk_per_trade", 0.005)))
        trade_return -= float(getattr(settings, "backtest_spread_cost_bps", 0.0)) / 10000.0
        trade_return -= float(getattr(settings, "backtest_slippage_cost_bps", 0.0)) / 10000.0

        regime = bt._regime_name(list(window["close"].tail(60)))
        trend_strength_ratio = _trend_strength_ratio(window)
        volatility_pct, volatility_bucket = _volatility_metrics(window)
        body_ratio, volume_ratio = _signal_micro_context(window)
        spike_ratio = _spike_ratio(window)
        support = _same_side_support_count(signals, str(best.side))
        same_side_ratio = float(support / max(1, len(signals)))
        confidence = float(best.confidence)
        confidence_bucket = _confidence_bucket(confidence)
        outcome = "win" if trade_return > 0 else ("loss" if trade_return < 0 else "flat")
        exit_time = pd.Timestamp(df.iloc[min(exit_idx, len(df) - 1)]["time"]).to_pydatetime()

        record = {
            "symbol": settings.symbol,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "session": session_label(entry_time),
            "regime": regime,
            "trend_strength_ratio": round(trend_strength_ratio, 4),
            "volatility_pct": round(volatility_pct, 4),
            "volatility_bucket": volatility_bucket,
            "confidence": round(confidence, 4),
            "confidence_bucket": confidence_bucket,
            "side": str(best.side),
            "strategy": str(best.strategy),
            "repeat_entry": bool(prev_side == str(best.side) and prev_regime == regime),
            "same_side_support": int(support),
            "same_side_ratio": round(same_side_ratio, 4),
            "body_ratio": float(body_ratio),
            "volume_ratio": float(volume_ratio),
            "spike_ratio": round(float(spike_ratio), 4),
            "spread_cost_bps_assumed": float(getattr(settings, "backtest_spread_cost_bps", 0.0)),
            "slippage_cost_bps_assumed": float(getattr(settings, "backtest_slippage_cost_bps", 0.0)),
            "hold_bars": int(hold_bars),
            "mfe_r": round(float(exit_path["mfe_r"]), 4),
            "mae_r": round(float(exit_path["mae_r"]), 4),
            "trade_return": round(float(trade_return), 6),
            "outcome": outcome,
            "exit_type": str(exit_path["exit_type"]),
        }
        record["post_spike_context"] = bool(spike_ratio >= 1.6)
        record["poor_liquidity"] = bool(volume_ratio < 0.8)
        record["london_extreme_volatility"] = bool(
            record["session"] == "london" and (volatility_pct < 0.03 or volatility_pct >= 0.07)
        )
        record["trend_low_volatility"] = bool(regime == "trend" and volatility_pct < 0.03)
        record["weak_consensus"] = bool(same_side_ratio < 0.34)
        record["repeated_reentry"] = bool(record["repeat_entry"] and record["post_spike_context"])
        record["first_entry_quality"] = bool(not record["repeat_entry"] and same_side_ratio >= 0.5)
        record["favorable_session"] = bool(record["session"] in {"asia", "new_york"})
        record["strong_trend_clean_entry"] = bool(
            regime == "trend"
            and trend_strength_ratio >= 8.0
            and volatility_pct >= 0.03
            and not record["post_spike_context"]
        )
        record["high_quality_confirmation"] = bool(confidence_bucket in {"mid", "high"} and same_side_ratio >= 0.5)
        label, reasons, label_confidence = label_trade(record)
        record["label"] = label
        record["label_confidence"] = label_confidence
        record["reason_codes"] = reasons
        records.append(record)
        bt.allocator.update(best.strategy, float(trade_return))
        prev_side = str(best.side)
        prev_regime = regime
        i = max(i + 1, exit_idx)
    return records


def _label_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label, grp in df.groupby("label"):
        wins = grp[grp["trade_return"] > 0]["trade_return"].tolist()
        losses = grp[grp["trade_return"] < 0]["trade_return"].tolist()
        summary[str(label)] = {
            "count": int(len(grp)),
            "win_rate": float((grp["trade_return"] > 0).mean()) if len(grp) else 0.0,
            "expectancy": float(grp["trade_return"].mean()) if len(grp) else 0.0,
            "average_win": float(sum(wins) / len(wins)) if wins else 0.0,
            "average_loss": float(sum(losses) / len(losses)) if losses else 0.0,
        }
    return summary


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _reason_counts(df: pd.DataFrame, label: str) -> dict[str, int]:
    rows = df[df["label"] == label]
    counts: dict[str, int] = {}
    for reasons in rows["reason_codes"].tolist():
        for reason in list(reasons or []):
            counts[str(reason)] = int(counts.get(str(reason), 0)) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def run_trade_labeling(settings: Settings, output_path: str | None = None) -> dict[str, Any]:
    frozen = settings
    connector = HFMConnector(frozen)
    connector.connect()
    try:
        df = MarketDataFeed(frozen).get_rates(3000)
        records = _trade_records(df, frozen)
    finally:
        connector.shutdown()

    trades = pd.DataFrame(records)
    if trades.empty:
        result = {"ok": False, "reason": "No historical trades were generated for labeling."}
        if output_path:
            Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    output_file = Path(output_path or (Path(frozen.storage_dir) / "trade_labels_report.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "ok": True,
        "dataset_path": str(output_file),
        "label_definitions": {
            "high-quality winner": "Winning trade in favorable context with clean structure, confirmation, and no major weak-context flags.",
            "acceptable trade": "Trade that fits system rules and is not clearly ideal or clearly avoidable.",
            "avoidable loser": "Losing trade taken in a context that historical analysis suggests should often be skipped.",
            "low-quality setup": "Poor-context setup that should be downgraded or blocked regardless of outcome.",
        },
        "feature_set": [
            "symbol",
            "entry_time",
            "exit_time",
            "session",
            "regime",
            "trend_strength_ratio",
            "volatility_pct",
            "volatility_bucket",
            "confidence",
            "confidence_bucket",
            "side",
            "strategy",
            "repeat_entry",
            "same_side_support",
            "same_side_ratio",
            "body_ratio",
            "volume_ratio",
            "spike_ratio",
            "spread_cost_bps_assumed",
            "slippage_cost_bps_assumed",
            "hold_bars",
            "mfe_r",
            "mae_r",
            "trade_return",
            "outcome",
            "exit_type",
        ],
        "reason_code_taxonomy": {
            "late_us_trend": "Trend trade taken in the late US session.",
            "post_spike": "Entry immediately after an overextended bar/spike.",
            "london_drag": "Trade taken in a historically weak London volatility pocket.",
            "low_vol_trend": "Trend trade during low-volatility drift conditions.",
            "poor_liquidity": "Thin-volume / poor-liquidity context.",
            "weak_confirmation": "Weak same-side support / confirmation.",
            "repeated_reentry": "Repeated re-entry into the same move under degraded context.",
            "first_entry_quality": "Early, cleaner first-entry structure.",
            "strong_trend_clean_entry": "Strong trend with cleaner structure and non-spike entry.",
            "favorable_session": "Session historically associated with stronger expectancy.",
            "high_quality_confirmation": "Mid/high confidence with stronger confirmation.",
        },
        "summary_by_label": _label_summary(trades),
        "label_distribution_by_regime": {
            str(regime): dict(grp["label"].value_counts()) for regime, grp in trades.groupby("regime")
        },
        "label_distribution_by_session": {
            str(session): dict(grp["label"].value_counts()) for session, grp in trades.groupby("session")
        },
        "label_distribution_by_trend_strength": {
            "bottom_quartile": dict(trades[trades["trend_strength_ratio"] <= trades["trend_strength_ratio"].quantile(0.25)]["label"].value_counts()),
            "upper_quartile": dict(trades[trades["trend_strength_ratio"] >= trades["trend_strength_ratio"].quantile(0.75)]["label"].value_counts()),
        },
        "top_reason_codes": {
            "avoidable_loser": _reason_counts(trades, "avoidable loser"),
            "high_quality_winner": _reason_counts(trades, "high-quality winner"),
            "low_quality_setup": _reason_counts(trades, "low-quality setup"),
        },
        "records": records,
    }
    safe_result = _json_safe(result)
    output_file.write_text(json.dumps(safe_result, indent=2), encoding="utf-8")
    return safe_result
