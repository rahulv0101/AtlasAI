from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _state(value: Any) -> str:
    return str(value or "").strip().lower()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _confidence_label(score: int) -> str:
    if score >= 80:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _trend_side(trend_structure: str, trend_state: str) -> str:
    ts = str(trend_structure or "")
    st = _state(trend_state)
    if ts in {"HH/HL", "hh/hl"} or st == "uptrend":
        return "buy"
    if ts in {"LH/LL", "lh/ll"} or st == "downtrend":
        return "sell"
    return "neutral"


def build_market_regime_and_reversal(
    *,
    status: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
    chart_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    st = dict(status or {})
    rt = dict(runtime or {})
    context = dict(st.get("context_snapshot", {}) or {})
    chart = dict(chart_input or st.get("chart_features") or st.get("chart_analysis_input") or {})
    structure_tf = str(st.get("timeframe") or chart.get("timeframe") or "current").strip().upper() or "current"

    trend_structure = str(context.get("trend_structure", chart.get("trend_structure", "mixed")) or "mixed")
    trend_state = _state(context.get("trend_state", chart.get("trend_state", "unclear")))
    momentum_state = _state(context.get("momentum", chart.get("momentum_state", "mixed")))
    spread_state = _state(context.get("spread_state"))
    liquidity_state = _state(context.get("liquidity_state"))
    volatility_state = _state(context.get("volatility_state"))
    session = _state(context.get("session"))
    setup_type = _state(st.get("setup_family", chart.get("setup_type_hint", "no_trade")))

    expansion = _f(chart.get("candle_expansion", context.get("candle_expansion", 0.0)))
    compression = _f(chart.get("compression_score", context.get("compression_score", 0.0)))
    wickiness = _f(chart.get("wickiness", context.get("wickiness", 0.0)))
    chop_score = _f(chart.get("chop_score", context.get("chop_score", 0.0)))
    atr_proxy = _f(chart.get("atr_ratio", context.get("atr_ratio", context.get("trend_strength", 0.0))))
    repeated_rejection = bool(chart.get("repeated_rejection_at_extremes", context.get("repeated_rejection_at_extremes", False)))
    unstable_spread = spread_state == "wide" or liquidity_state in {"poor", "illiquid"} or volatility_state == "high"

    trend_points = 0
    range_points = 0
    volatile_points = 0
    trend_reasons: list[str] = []
    range_reasons: list[str] = []
    volatile_reasons: list[str] = []

    if trend_structure in {"HH/HL", "hh/hl", "LH/LL", "lh/ll"}:
        trend_points += 34
        trend_reasons.append(f"clear {structure_tf} directional structure")
    elif trend_state in {"uptrend", "downtrend"}:
        trend_points += 22
        trend_reasons.append("directional trend state is present")

    if momentum_state == "strong":
        trend_points += 18
        trend_reasons.append("momentum is directional")
    elif momentum_state == "weak":
        range_points += 12
        range_reasons.append("momentum is weak")
    elif momentum_state == "mixed" and trend_points >= 34 and volatility_state in {"normal", "low", ""}:
        trend_points += 12
        trend_reasons.append("clear structure persists while momentum is mixed")

    if compression >= 0.6 or _state(chart.get("compression_state")) == "compressed":
        range_points += 22
        range_reasons.append("structure is compressed")
    if repeated_rejection or wickiness >= 0.62 or chop_score >= 0.6:
        range_points += 24
        range_reasons.append("price is rejecting and choppy near extremes")
    if setup_type in {"consolidation", "mean_reversion"}:
        range_points += 10
    if trend_points >= 34 and range_points >= 22:
        range_reasons.append("range pressure is present inside a directional structure")

    if volatility_state == "high" or expansion >= 0.7 or atr_proxy >= 0.07:
        volatile_points += 28
        volatile_reasons.append("candle expansion / volatility is elevated")
    if wickiness >= 0.7:
        volatile_points += 18
        volatile_reasons.append("wickiness is elevated")
    if unstable_spread:
        volatile_points += 24
        volatile_reasons.append("spread or liquidity is unstable")
    if session in {"late_us", "rollover", "unknown"} and volatility_state == "high":
        volatile_points += 10
        volatile_reasons.append("session quality weakens volatile conditions")

    scores = {
        "trending": min(100, trend_points),
        "ranging": min(100, range_points),
        "volatile": min(100, volatile_points),
    }
    regime = "unknown"
    regime_reasons: list[str] = []
    confidence = 28
    dominant = max(scores, key=scores.get)
    if scores[dominant] >= 45:
        regime = dominant
        confidence = scores[dominant]
        if dominant == "trending":
            regime_reasons = trend_reasons
        elif dominant == "ranging":
            regime_reasons = range_reasons
        else:
            regime_reasons = volatile_reasons
    if dominant == "volatile" and scores["volatile"] >= 55:
        regime = "volatile"
        confidence = scores["volatile"]
        regime_reasons = volatile_reasons
    if regime == "unknown" and trend_points >= 34 and trend_structure in {"HH/HL", "hh/hl", "LH/LL", "lh/ll"}:
        regime = "trending"
        confidence = min(58, max(45, trend_points))
        regime_reasons = trend_reasons[:]

    trend_side = _trend_side(trend_structure, trend_state)
    dxy_alignment = _state(context.get("dxy_alignment"))
    us10y_alignment = _state(context.get("us10y_alignment"))
    macro_conflict = dxy_alignment in {"conflict", "conflicting", "opposed"} or us10y_alignment in {"conflict", "conflicting", "opposed"}
    exhaustion_wick = wickiness >= 0.68 or bool(chart.get("exhaustion_wick", context.get("exhaustion_wick", False)))
    new_local_high = bool(chart.get("new_local_high", context.get("new_local_high", False)))
    new_local_low = bool(chart.get("new_local_low", context.get("new_local_low", False)))
    structure_break = bool(chart.get("structure_break_confirmed", context.get("structure_break_confirmed", False)))
    momentum_confirm = _state(chart.get("reversal_momentum_confirmation", context.get("reversal_momentum_confirmation", ""))) in {"yes", "confirmed", "true", "strong"}
    if not momentum_confirm:
        momentum_confirm = bool(chart.get("momentum_confirmation", context.get("momentum_confirmation", False)))

    reversal_candidate = False
    confirmed_reversal = False
    reversal_confidence = 0
    reversal_bias = "neutral"
    reversal_reasons: list[str] = []

    if macro_conflict:
        reversal_reasons.append("macro conflict is present")
    if exhaustion_wick:
        reversal_reasons.append("exhaustion wick / rejection is present")
    if new_local_high or new_local_low:
        reversal_reasons.append("local extreme sweep is present")

    if macro_conflict and exhaustion_wick and (new_local_high or new_local_low):
        reversal_candidate = True
        reversal_confidence = 58
        reversal_bias = "sell" if new_local_high else "buy"

    if reversal_candidate and structure_break:
        reversal_confidence += 18
        reversal_reasons.append("structure break confirmed after sweep")
    if reversal_candidate and momentum_confirm:
        reversal_confidence += 16
        reversal_reasons.append("momentum confirms reversal")
    if reversal_candidate and regime == "volatile":
        reversal_confidence = max(0, reversal_confidence - 12)
        reversal_reasons.append("volatile regime reduces reversal confidence")
    if reversal_candidate and structure_break and momentum_confirm:
        confirmed_reversal = True
    else:
        confirmed_reversal = False
        if reversal_candidate and not structure_break:
            reversal_reasons.append("structure confirmation missing")
        if reversal_candidate and not momentum_confirm:
            reversal_reasons.append("momentum confirmation missing")

    reversal_confidence = max(0, min(100, reversal_confidence))

    return {
        "market_regime": {
            "regime": regime,
            "regime_confidence": confidence,
            "regime_confidence_level": _confidence_label(confidence),
            "regime_reasons": regime_reasons[:6],
            "scores": scores,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "behavior": {
                "trending": "prioritize continuation/pullback and avoid blind counter-trend entries",
                "ranging": "prefer mean reversion only at defined extremes with confirmation",
                "volatile": "defensive mode; reduce size or block entries depending on severity",
            },
        },
        "reversal_detection": {
            "reversal_candidate": reversal_candidate,
            "reversal_confidence": reversal_confidence,
            "reversal_confidence_level": _confidence_label(reversal_confidence),
            "reversal_bias": reversal_bias,
            "reversal_reasons": reversal_reasons[:8],
            "confirmed_reversal": confirmed_reversal,
            "requires": [
                "macro divergence/conflict",
                "structure confirmation",
                "momentum confirmation",
            ],
        },
    }
