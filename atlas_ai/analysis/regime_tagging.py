from __future__ import annotations

from typing import Any


REGIME_LABELS = ("trending", "ranging", "high_volatility", "low_volatility")


def _floats(values: Any) -> list[float]:
    out: list[float] = []
    for value in list(values or []):
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def classify_regime_label(
    *,
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    trend_threshold_pct: float = 0.12,
    high_volatility_atr_pct: float = 0.06,
    low_volatility_atr_pct: float = 0.02,
) -> dict[str, Any]:
    """Rule-based regime tag for observability only.

    The label is intentionally simple and explainable. It does not mutate
    strategy weights, thresholds, or execution policy.
    """
    close_values = _floats(closes)
    high_values = _floats(highs or [])
    low_values = _floats(lows or [])
    if len(close_values) < 10:
        return {
            "label": "ranging",
            "confidence": 0.2,
            "reason": "insufficient_price_history_defaulted_to_ranging",
            "metrics": {"samples": len(close_values), "slope_pct": 0.0, "atr_pct": 0.0, "range_pct": 0.0},
            "policy_note": "regime label is observational only; ranking weights are unchanged",
        }

    first = close_values[0]
    last = close_values[-1]
    denom = max(1e-9, abs(first))
    slope_pct = ((last - first) / denom) * 100.0

    true_ranges: list[float] = []
    for idx in range(1, len(close_values)):
        prev_close = close_values[idx - 1]
        high = high_values[idx] if idx < len(high_values) else max(close_values[idx], prev_close)
        low = low_values[idx] if idx < len(low_values) else min(close_values[idx], prev_close)
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    atr_pct = (_avg(true_ranges) / max(1e-9, abs(last))) * 100.0
    range_pct = ((max(close_values) - min(close_values)) / max(1e-9, abs(last))) * 100.0

    abs_slope = abs(slope_pct)
    if abs_slope >= max(trend_threshold_pct, atr_pct * 2.0):
        label = "trending"
        reason = f"close-to-close slope {slope_pct:.4f}% exceeds trend threshold"
    elif atr_pct >= high_volatility_atr_pct:
        label = "high_volatility"
        reason = f"ATR proxy {atr_pct:.4f}% >= high volatility threshold {high_volatility_atr_pct:.4f}%"
    elif atr_pct <= low_volatility_atr_pct and abs_slope < trend_threshold_pct:
        label = "low_volatility"
        reason = f"ATR proxy {atr_pct:.4f}% <= low volatility threshold {low_volatility_atr_pct:.4f}% and slope is muted"
    else:
        label = "ranging"
        reason = "slope and ATR proxy do not confirm trend or volatility extreme"

    confidence = min(0.95, max(0.35, abs_slope / max(0.01, trend_threshold_pct) * 0.25 + atr_pct / max(0.01, high_volatility_atr_pct) * 0.25))
    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "reason": reason,
        "metrics": {
            "samples": len(close_values),
            "slope_pct": round(float(slope_pct), 6),
            "atr_pct": round(float(atr_pct), 6),
            "range_pct": round(float(range_pct), 6),
            "trend_threshold_pct": round(float(trend_threshold_pct), 6),
            "high_volatility_atr_pct": round(float(high_volatility_atr_pct), 6),
            "low_volatility_atr_pct": round(float(low_volatility_atr_pct), 6),
        },
        "policy_note": "regime label is observational only; ranking weights are unchanged",
    }
