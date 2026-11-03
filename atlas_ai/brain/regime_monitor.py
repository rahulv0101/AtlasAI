from __future__ import annotations

from typing import Any


def _trend_strength(status: dict[str, Any]) -> float:
    regime = dict(status.get("regime", {}) or {})
    context = dict(status.get("context_snapshot", {}) or {})
    return float(regime.get("trend_strength", context.get("trend_strength", 0.0)) or 0.0)


def _vol_state(status: dict[str, Any]) -> str:
    context = dict(status.get("context_snapshot", {}) or {})
    return str(context.get("volatility_state", "unknown") or "unknown")


def _noise_score(status: dict[str, Any]) -> float:
    market = dict(status.get("market", {}) or {})
    closes = list(market.get("recent_close", []) or [])
    if len(closes) < 6:
        return 0.5
    signed_deltas = [float(closes[i]) - float(closes[i - 1]) for i in range(1, len(closes))]
    abs_deltas = [abs(d) for d in signed_deltas]
    avg = sum(abs_deltas) / len(abs_deltas) if abs_deltas else 0.0
    variance = sum((d - avg) ** 2 for d in abs_deltas) / max(1, len(abs_deltas))
    sig_threshold = max(1e-6, avg * 0.6)
    sign_flips = sum(
        1
        for i in range(1, len(signed_deltas))
        if abs(signed_deltas[i]) >= sig_threshold
        and abs(signed_deltas[i - 1]) >= sig_threshold
        and (signed_deltas[i] > 0) != (signed_deltas[i - 1] > 0)
    )
    flip_ratio = sign_flips / max(1, len(signed_deltas) - 1)
    # Combine amplitude inconsistency + frequent directional flips to approximate choppy/noisy behavior.
    variance_component = max(0.0, min(1.0, variance / (avg + 1e-6) / 6.0))
    return max(0.0, min(1.0, variance_component * 0.35 + flip_ratio * 0.65))


def detect_regime_shift(status: dict[str, Any]) -> dict[str, Any]:
    trend = _trend_strength(status)
    vol = _vol_state(status)
    noise = _noise_score(status)

    shift_conf = 0.25
    observed: list[str] = []
    if vol in {"high", "elevated"}:
        shift_conf += 0.25
        observed.append("volatility increased")
    if trend < 0.12:
        shift_conf += 0.2
        observed.append("trend strength weakened")
    if noise > 0.55:
        shift_conf += 0.2
        observed.append("chart noise/choppiness increased")
    shift_conf = max(0.0, min(1.0, shift_conf))

    state = "stable"
    if shift_conf >= 0.7:
        state = "shifting"
    elif shift_conf >= 0.5:
        state = "watching"
    return {
        "regime_state": state,
        "shift_confidence": shift_conf,
        "observed_changes": observed or ["no strong regime shift signal detected"],
        "trend_strength": trend,
        "volatility_state": vol,
        "noise_score": noise,
    }
