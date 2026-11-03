from __future__ import annotations

"""Volatility classification and response policy for XAUUSD (analysis-only)."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


VOLATILITY_CLASSES = {
    "compressed_volatility",
    "normal_volatility",
    "elevated_volatility",
    "extreme_volatility",
    "news_spike_volatility",
    "liquidity_gap_volatility",
    "chop_volatility",
    "trend_expansion_volatility",
    "reversal_exhaustion_volatility",
    "session_transition_volatility",
}


@dataclass
class VolatilityFeatures:
    # FIX: unified volatility feature schema for each scanner cycle.
    atr_current: float = 0.0
    atr_14: float = 0.0
    atr_baseline: float = 0.0
    atr_ratio: float = 0.0
    candle_body_ratio: float = 0.0
    wick_ratio: float = 0.0
    average_range: float = 0.0
    range_expansion_ratio: float = 0.0
    spread_points: float = 0.0
    spread_pips: float = 0.0
    spread_status: str = "unknown"
    session: str = "unknown"
    session_transition_flag: bool = False
    news_window_flag: bool = False
    high_impact_news_hard_block: bool = False
    dxy_alignment_state: str = "unavailable"
    us10y_alignment_state: str = "unavailable"
    vix_state: str = "unavailable"
    price_structure_state: str = "unclear"
    momentum_state: str = "mixed"
    macro_available: bool = False


@dataclass
class VolatilityAssessment:
    # FIX: explicit class + confidence + response rule exposed to scanner/UI/Ask Atlas.
    volatility_class: str = "normal_volatility"
    volatility_confidence: float = 0.5
    response_rule: str = "normal_analysis"
    tags: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    explanation: str = ""
    feature_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VolatilityOutcomeRecord:
    timestamp: str
    volatility_class_at_entry: str
    atr_ratio_at_entry: float
    spread_status_at_entry: str
    session: str
    outcome: str
    r_multiple: float
    mfe: float
    mae: float
    response_was_correct: bool
    blocker_prevented_loss: bool
    blocker_missed_profit: bool


def classify_volatility(features: VolatilityFeatures) -> VolatilityAssessment:
    # FIX: deterministic taxonomy classification for major XAUUSD volatility states.
    atr_ratio = float(features.atr_ratio or 0.0)
    wick_ratio = float(features.wick_ratio or 0.0)
    body_ratio = float(features.candle_body_ratio or 0.0)
    spread_status = str(features.spread_status or "unknown")
    momentum = str(features.momentum_state or "mixed")
    structure = str(features.price_structure_state or "unclear")
    tags: list[str] = []
    blockers: list[str] = []
    confidence = 0.55
    klass = "normal_volatility"
    explanation = "ATR and spread are within baseline range."

    if bool(features.high_impact_news_hard_block) or bool(features.news_window_flag):
        klass = "news_spike_volatility"
        confidence = 0.92
        tags.extend(["news_window", "volatility_hard_block"])
        blockers.append("high_impact_news_window")
        explanation = "High-impact news window active."
    elif atr_ratio >= 2.2 or (spread_status in {"high", "blocked"} and atr_ratio >= 1.6):
        klass = "extreme_volatility"
        confidence = 0.90
        tags.extend(["extreme_move", "volatility_hard_block"])
        blockers.append("extreme_volatility")
        explanation = "ATR spike and/or spread stress indicates extreme volatility."
    elif spread_status in {"high", "blocked"} and atr_ratio >= 1.2:
        klass = "liquidity_gap_volatility"
        confidence = 0.84
        tags.extend(["liquidity_risk", "slippage_risk"])
        blockers.append("spread_liquidity_too_poor")
        explanation = "Spread expansion plus range expansion indicates liquidity gap risk."
    elif bool(features.session_transition_flag):
        klass = "session_transition_volatility"
        confidence = 0.70
        tags.append("session_transition")
        explanation = "Session transition window: volatility can reprice quickly."
    elif atr_ratio < 0.75 and body_ratio < 0.35 and wick_ratio < 0.45:
        klass = "compressed_volatility"
        confidence = 0.78
        tags.append("breakout_watch")
        explanation = "Compressed candles and low ATR suggest coiling market."
    elif wick_ratio >= 0.62 and momentum == "mixed":
        klass = "chop_volatility"
        confidence = 0.82
        tags.append("chop_risk")
        explanation = "High wick ratio and mixed momentum indicate chop."
    elif atr_ratio >= 1.35 and structure in {"trend_up", "trend_down"} and momentum in {"bullish", "bearish"}:
        klass = "trend_expansion_volatility"
        confidence = 0.83
        tags.append("trend_expansion")
        explanation = "ATR expansion aligns with directional structure and momentum."
    elif atr_ratio >= 1.2 and body_ratio < 0.35 and wick_ratio >= 0.55 and structure in {"trend_up", "trend_down"}:
        klass = "reversal_exhaustion_volatility"
        confidence = 0.66
        tags.append("reversal_watch")
        explanation = "Expansion with rejection wicks suggests possible exhaustion."
    elif atr_ratio >= 1.2:
        klass = "elevated_volatility"
        confidence = 0.76
        tags.append("elevated_risk")
        explanation = "ATR above baseline; risk should be reduced."
    else:
        klass = "normal_volatility"
        confidence = 0.74
        tags.append("normal_conditions")
        explanation = "Volatility is near baseline."

    response_rule = response_for_class(klass)
    if not features.macro_available:
        tags.append("macro_unavailable_price_inferred")
        confidence = max(0.45, confidence - 0.08)
    return VolatilityAssessment(
        volatility_class=klass,
        volatility_confidence=round(confidence, 3),
        response_rule=response_rule,
        tags=tags,
        blockers=blockers,
        explanation=explanation,
        feature_snapshot=asdict(features),
    )


def response_for_class(volatility_class: str) -> str:
    # FIX: class-to-response policy map keeps scanner behavior explicit and safe.
    mapping = {
        "compressed_volatility": "breakout_watch",
        "normal_volatility": "normal_analysis",
        "elevated_volatility": "reduce_risk_require_higher_confluence",
        "extreme_volatility": "volatility_hard_block",
        "news_spike_volatility": "news_hard_block",
        "liquidity_gap_volatility": "spread_slippage_block",
        "chop_volatility": "prefer_no_trade_reduce_confidence",
        "trend_expansion_volatility": "allow_continuation_if_aligned",
        "reversal_exhaustion_volatility": "reversal_only_with_full_confirmation",
        "session_transition_volatility": "reduce_confidence_until_stable",
    }
    return str(mapping.get(str(volatility_class), "normal_analysis"))


def summarize_performance_by_volatility(records: list[VolatilityOutcomeRecord]) -> dict[str, Any]:
    # FIX: summarize volatility-class outcomes every 50+ completed demo trades.
    grouped: dict[str, list[VolatilityOutcomeRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.volatility_class_at_entry, []).append(rec)
    out: dict[str, Any] = {}
    for klass, rows in grouped.items():
        n = len(rows)
        if n == 0:
            continue
        wins = len([r for r in rows if r.outcome == "win"])
        avg_r = sum(float(r.r_multiple) for r in rows) / max(1, n)
        expectancy = avg_r
        avg_spread_cost = sum(1.0 if r.spread_status_at_entry in {"high", "blocked"} else 0.2 for r in rows) / max(1, n)
        out[klass] = {
            "trades": n,
            "win_rate": round(wins / max(1, n), 4),
            "expectancy_r": round(expectancy, 4),
            "average_r": round(avg_r, 4),
            "max_drawdown_r_proxy": round(min(0.0, min(float(r.r_multiple) for r in rows)), 4),
            "average_spread_cost_proxy": round(avg_spread_cost, 4),
        }
    recommendations: list[str] = []
    for klass, stats in out.items():
        if float(stats.get("expectancy_r", 0.0)) < 0:
            recommendations.append(f"Tighten filters for {klass}; expectancy is negative.")
        elif float(stats.get("win_rate", 0.0)) >= 0.58 and float(stats.get("expectancy_r", 0.0)) > 0:
            recommendations.append(f"Allow more candidates in {klass}; class is currently profitable.")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_size": len(records),
        "classes": out,
        "recommendations": recommendations[:8],
    }
