from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def session_label(when_utc: datetime | None) -> str:
    if when_utc is None:
        when_utc = datetime.now(timezone.utc)
    hour = int(when_utc.hour)
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "late_us"


def _volume_ratio(df) -> float:
    recent = df.tail(20).copy()
    if len(recent) < 5:
        return 1.0
    vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
    if vol_col not in recent.columns:
        return 1.0
    values = [float(x or 0.0) for x in recent[vol_col].tolist()]
    current = values[-1]
    average = sum(values[:-1]) / max(1, len(values) - 1)
    return current / max(1e-9, average)


def _atr_like(df, bars: int = 14) -> float:
    recent = df.tail(bars).copy()
    if len(recent) < 5:
        return 0.0
    ranges = [float(row.high) - float(row.low) for row in recent.itertuples(index=False)]
    return sum(ranges) / max(1, len(ranges))


def _spike_ratio(df) -> float:
    if len(df) < 5:
        return 0.0
    last = df.iloc[-1]
    bar_range = float(last["high"]) - float(last["low"])
    return bar_range / max(1e-9, _atr_like(df, 14))


def _regime_metrics(df) -> dict[str, float | str]:
    closes = [float(x) for x in df.tail(60)["close"].tolist()]
    if len(closes) < 30:
        return {"name": "unknown", "trend_strength_pct": 0.0, "volatility_pct": 0.0}
    first = float(closes[0])
    last = float(closes[-1])
    if abs(first) < 1e-9:
        return {"name": "unknown", "trend_strength_pct": 0.0, "volatility_pct": 0.0}
    drift_pct = abs((last - first) / first) * 100.0
    abs_moves = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    avg_abs = sum(abs_moves) / max(1, len(abs_moves))
    vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
    if drift_pct >= max(0.12, vol_pct * 2.2):
        name = "trend"
    elif vol_pct >= 0.06:
        name = "high_volatility_range"
    else:
        name = "range"
    return {
        "name": name,
        "trend_strength_pct": round(float(drift_pct), 4),
        "volatility_pct": round(float(vol_pct), 4),
    }


def build_research_regime_overlay_context(
    df,
    when_utc: datetime | None,
    spread_points: float = 0.0,
) -> dict[str, Any]:
    ts = when_utc
    if ts is None:
        if "time" in df.columns and len(df) > 0:
            raw = df.iloc[-1]["time"]
            ts = raw.to_pydatetime() if hasattr(raw, "to_pydatetime") else raw
        else:
            ts = datetime.now(timezone.utc)
    volume_ratio = _volume_ratio(df)
    spike_ratio = _spike_ratio(df)
    regime = _regime_metrics(df)
    return {
        "session": session_label(ts),
        "regime_name": str(regime["name"]),
        "trend_strength_pct": float(regime["trend_strength_pct"]),
        "volatility_pct": float(regime["volatility_pct"]),
        "volume_ratio": round(float(volume_ratio), 4),
        "spike_ratio": round(float(spike_ratio), 4),
        "spread_points": round(float(spread_points or 0.0), 2),
        "low_liquidity": bool(volume_ratio < 0.75),
        "abnormal_spike": bool(spike_ratio >= 2.2),
    }


def research_regime_filter_reason(settings, context: dict[str, Any]) -> str:
    if not bool(getattr(settings, "research_regime_filter_enabled", False)):
        return ""
    low_liquidity_threshold = float(
        getattr(settings, "research_regime_low_liquidity_volume_ratio", 0.75) or 0.75
    )
    abnormal_spike_threshold = float(
        getattr(settings, "research_regime_abnormal_spike_atr_multiple", 2.2) or 2.2
    )
    post_spike_threshold = float(
        getattr(settings, "research_regime_post_spike_atr_multiple", 1.6) or 1.6
    )
    trend_strength_min = float(
        getattr(settings, "research_regime_trend_min_strength_pct", 0.20) or 0.20
    )
    low_volatility_max = float(
        getattr(settings, "research_regime_trend_low_volatility_max_pct", 0.03) or 0.03
    )
    london_low_volatility_max = float(
        getattr(settings, "research_regime_london_low_volatility_max_pct", 0.03) or 0.03
    )
    london_high_volatility_min = float(
        getattr(settings, "research_regime_london_high_volatility_min_pct", 0.07) or 0.07
    )
    regime_name = str(context.get("regime_name", "unknown") or "unknown")
    trend_strength = float(context.get("trend_strength_pct", 0.0) or 0.0)
    volatility_pct = float(context.get("volatility_pct", 0.0) or 0.0)
    session = str(context.get("session", "unknown") or "unknown")
    volume_ratio = float(context.get("volume_ratio", 1.0) or 1.0)
    spike_ratio = float(context.get("spike_ratio", 0.0) or 0.0)
    if bool(getattr(settings, "research_regime_filter_low_liquidity", True)) and volume_ratio < low_liquidity_threshold:
        return (
            "Research regime overlay blocked trade: low liquidity "
            f"(volume_ratio={volume_ratio:.2f} < {low_liquidity_threshold:.2f})."
        )
    if bool(getattr(settings, "research_regime_filter_abnormal_spike", True)) and spike_ratio >= abnormal_spike_threshold:
        return (
            "Research regime overlay blocked trade: abnormal spike "
            f"(spike_ratio={spike_ratio:.2f} >= {abnormal_spike_threshold:.2f})."
        )
    if bool(getattr(settings, "research_regime_filter_post_spike", False)) and spike_ratio >= post_spike_threshold:
        return (
            "Research regime overlay blocked trade: post-spike entry "
            f"(spike_ratio={spike_ratio:.2f} >= {post_spike_threshold:.2f})."
        )
    if bool(getattr(settings, "research_regime_filter_london_extreme_volatility", False)) and session == "london":
        if volatility_pct < london_low_volatility_max or volatility_pct >= london_high_volatility_min:
            return (
                "Research regime overlay blocked trade: london extreme volatility "
                f"(volatility_pct={volatility_pct:.4f}%, allowed=[{london_low_volatility_max:.4f}, "
                f"{london_high_volatility_min:.4f}))."
            )
    if regime_name == "trend":
        if bool(getattr(settings, "research_regime_filter_weak_trend", False)) and trend_strength < trend_strength_min:
            return (
                "Research regime overlay blocked trade: weak trend "
                f"(trend_strength={trend_strength:.4f}% < {trend_strength_min:.4f}%)."
            )
        if bool(getattr(settings, "research_regime_filter_trend_low_volatility", False)) and volatility_pct < low_volatility_max:
            return (
                "Research regime overlay blocked trade: low-volatility trend "
                f"(volatility_pct={volatility_pct:.4f}% < {low_volatility_max:.4f}%)."
            )
        if bool(getattr(settings, "research_regime_filter_trend_late_us", False)) and session == "late_us":
            return "Research regime overlay blocked trade: late_us trend regime."
    return ""


def conservative_pretrade_filter_reason(settings, context: dict[str, Any]) -> str:
    if not bool(getattr(settings, "conservative_filter_enabled", False)):
        return ""

    regime_name = str(context.get("regime_name", "unknown") or "unknown")
    session = str(context.get("session", "unknown") or "unknown")
    volume_ratio = float(context.get("volume_ratio", 1.0) or 1.0)
    spike_ratio = float(context.get("spike_ratio", 0.0) or 0.0)

    if bool(getattr(settings, "conservative_filter_block_late_us_trend", True)):
        if regime_name == "trend" and session == "late_us":
            return "Conservative pre-trade filter blocked trade: late_us trend regime."

    poor_liquidity_threshold = float(
        getattr(settings, "conservative_filter_poor_liquidity_volume_ratio", 0.80) or 0.80
    )
    if bool(getattr(settings, "conservative_filter_block_poor_liquidity", True)):
        if volume_ratio < poor_liquidity_threshold:
            return (
                "Conservative pre-trade filter blocked trade: poor liquidity "
                f"(volume_ratio={volume_ratio:.2f} < {poor_liquidity_threshold:.2f})."
            )

    post_spike_threshold = float(
        getattr(settings, "conservative_filter_post_spike_atr_multiple", 1.6) or 1.6
    )
    if bool(getattr(settings, "conservative_filter_block_post_spike", True)):
        if spike_ratio >= post_spike_threshold:
            return (
                "Conservative pre-trade filter blocked trade: post-spike entry "
                f"(spike_ratio={spike_ratio:.2f} >= {post_spike_threshold:.2f})."
            )

    return ""
