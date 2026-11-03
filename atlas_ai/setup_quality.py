from __future__ import annotations

from typing import Any


def _win_rate(rows: list[dict[str, Any]]) -> tuple[float, int]:
    labeled = [row for row in rows if str(row.get("outcome", "")).lower() in {"win", "loss"}]
    if not labeled:
        return 0.5, 0
    wins = sum(1 for row in labeled if str(row.get("outcome", "")).lower() == "win")
    return wins / max(1, len(labeled)), len(labeled)


def score_setup_quality(
    *,
    settings: Any,
    context: dict[str, Any],
    recent_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    if not bool(getattr(settings, "setup_quality_enabled", False)):
        return {"enabled": False, "factor": 1.0, "score": 0.0, "reasons": [], "evidence": {}}

    side = str(context.get("side", "flat") or "flat").lower()
    if side not in {"buy", "sell"}:
        return {"enabled": True, "factor": 1.0, "score": 0.0, "reasons": ["flat_side"], "evidence": {}}

    strategy = str(context.get("strategy", "unknown") or "unknown").split("(", 1)[0].strip().lower()
    session = str(context.get("session", "unknown") or "unknown").lower()
    hour = int(context.get("hour_utc", -1) or -1)
    regime = str(context.get("regime", "unknown") or "unknown").lower()
    higher_tf_bias = str(context.get("higher_tf_bias", "unknown") or "unknown").lower()
    volatility_pct = float(context.get("volatility_pct", 0.0) or 0.0)
    trend_strength = float(context.get("trend_strength", 0.0) or 0.0)
    spread_points = float(context.get("spread_points", 0.0) or 0.0)
    volume_ratio = float(context.get("volume_ratio", 1.0) or 1.0)
    body_ratio = float(context.get("body_ratio", 0.0) or 0.0)
    same_side_ratio = float(context.get("same_side_ratio", 0.0) or 0.0)
    spike_ratio = float(context.get("spike_ratio", 0.0) or 0.0)
    htf_aligned = bool(context.get("htf_aligned", False))

    penalty = 0.0
    boost = 0.0
    reasons: list[str] = []
    evidence: dict[str, Any] = {}

    low_vol_max = float(getattr(settings, "setup_quality_trend_low_volatility_max_pct", 0.03) or 0.03)
    strong_trend_min = float(getattr(settings, "setup_quality_strong_trend_min_pct", 0.20) or 0.20)
    london_high_vol_min = float(getattr(settings, "setup_quality_london_high_volatility_min_pct", 0.07) or 0.07)
    spike_block = float(getattr(settings, "setup_quality_post_spike_atr_multiple", 1.6) or 1.6)
    min_bucket = max(4, int(getattr(settings, "setup_quality_min_bucket_samples", 6) or 6))

    if regime == "trend":
        if trend_strength < strong_trend_min:
            penalty += 0.07
            reasons.append("weak_trend_structure")
        elif trend_strength >= (strong_trend_min * 1.8):
            boost += 0.03
            reasons.append("strong_trend_structure")
        if volatility_pct < low_vol_max:
            penalty += 0.08
            reasons.append("trend_low_volatility")
    elif regime == "range" and 0.03 <= volatility_pct < london_high_vol_min:
        boost += 0.02
        reasons.append("stable_range_volatility")

    if session == "late_us" and regime == "trend":
        penalty += 0.10
        reasons.append("late_us_trend_drag")
    if session == "london" and (volatility_pct < low_vol_max or volatility_pct >= london_high_vol_min):
        penalty += 0.05
        reasons.append("london_extreme_volatility")
    if spike_ratio >= spike_block:
        penalty += 0.08
        reasons.append("post_spike_entry")
    if spread_points >= min(float(getattr(settings, "max_spread_points", 80.0)) * 0.5, 40.0):
        penalty += 0.04
        reasons.append("spread_friction")
    if volume_ratio < 0.8:
        penalty += 0.04
        reasons.append("thin_liquidity")
    if body_ratio < 0.32:
        penalty += 0.03
        reasons.append("weak_momentum_body")
    if same_side_ratio < 0.34:
        penalty += 0.03
        reasons.append("weak_consensus")
    if htf_aligned:
        boost += 0.03
        reasons.append("htf_alignment")
    elif higher_tf_bias in {"buy", "sell"}:
        penalty += 0.03
        reasons.append("htf_misaligned")

    labeled_outcomes = [row for row in list(recent_outcomes or []) if str(row.get("side", "")).lower() == side]
    strategy_rows = [row for row in labeled_outcomes if str(row.get("strategy", "")).split("(", 1)[0].strip().lower() == strategy]
    strategy_session_rows = [row for row in strategy_rows if str(row.get("session", "")).lower() == session]
    hour_rows = [row for row in labeled_outcomes if int(row.get("hour_utc", -1) or -1) == hour]
    session_rows = [row for row in labeled_outcomes if str(row.get("session", "")).lower() == session]

    wr_strategy, n_strategy = _win_rate(strategy_rows)
    wr_strategy_session, n_strategy_session = _win_rate(strategy_session_rows)
    wr_session, n_session = _win_rate(session_rows)
    wr_hour, n_hour = _win_rate(hour_rows)

    evidence["strategy_side"] = {"samples": n_strategy, "win_rate": round(wr_strategy, 3)}
    evidence["strategy_session_side"] = {"samples": n_strategy_session, "win_rate": round(wr_strategy_session, 3)}
    evidence["session_side"] = {"samples": n_session, "win_rate": round(wr_session, 3)}
    evidence["hour_side"] = {"samples": n_hour, "win_rate": round(wr_hour, 3)}

    if n_strategy >= min_bucket:
        if wr_strategy < 0.42:
            penalty += min(0.06, (0.42 - wr_strategy) * 0.5)
            reasons.append("strategy_side_underperforming")
        elif wr_strategy > 0.58:
            boost += min(0.04, (wr_strategy - 0.58) * 0.3)
            reasons.append("strategy_side_strength")

    if n_strategy_session >= min_bucket:
        if wr_strategy_session < 0.42:
            penalty += min(0.07, (0.42 - wr_strategy_session) * 0.55)
            reasons.append("strategy_session_drag")
        elif wr_strategy_session > 0.58:
            boost += min(0.05, (wr_strategy_session - 0.58) * 0.35)
            reasons.append("strategy_session_edge")

    if n_session >= min_bucket:
        if wr_session < 0.43:
            penalty += min(0.05, (0.43 - wr_session) * 0.35)
            reasons.append("session_drag")
        elif wr_session > 0.57:
            boost += min(0.03, (wr_session - 0.57) * 0.2)
            reasons.append("session_edge")

    if n_hour >= min_bucket:
        if wr_hour < 0.40:
            penalty += min(0.04, (0.40 - wr_hour) * 0.3)
            reasons.append("hour_drag")
        elif wr_hour > 0.60:
            boost += min(0.02, (wr_hour - 0.60) * 0.2)
            reasons.append("hour_edge")

    penalty_cap = max(0.0, min(0.35, float(getattr(settings, "setup_quality_penalty_cap", 0.18) or 0.18)))
    boost_cap = max(0.0, min(0.25, float(getattr(settings, "setup_quality_boost_cap", 0.10) or 0.10)))
    penalty = min(penalty_cap, penalty)
    boost = min(boost_cap, boost)
    score = boost - penalty
    factor = max(0.72, min(1.18, 1.0 + score))
    return {
        "enabled": True,
        "factor": round(factor, 4),
        "score": round(score, 4),
        "penalty": round(penalty, 4),
        "boost": round(boost, 4),
        "reasons": reasons[:8],
        "evidence": evidence,
    }
