from __future__ import annotations

"""Consistent Buy/Sell/No-Trade decision framework for XAUUSD."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from atlas_ai.config.settings import get_active_thresholds, load_settings

logger = logging.getLogger(__name__)


@dataclass
class DecisionInput:
    # purpose: carry normalized market/context inputs into decision scoring.
    # what it does: stores structure, momentum, location, macro, and risk context.
    higher_highs: bool = False
    higher_lows: bool = False
    lower_highs: bool = False
    lower_lows: bool = False
    near_support: bool = False
    near_resistance: bool = False
    in_middle_of_range: bool = False
    bullish_candle_strength: float = 0.0
    bearish_candle_strength: float = 0.0
    momentum_direction: str = "neutral"
    dxy_direction: str = "flat"
    us10y_direction: str = "flat"
    session: str = "unknown"
    session_tradeable: bool = False
    spread_pips: float = 0.0
    regime: str = "unknown"
    in_news_window: bool = False
    adx: float = 0.0
    atr_pips: float = 0.0
    volatility_class: str = "normal_volatility"
    volatility_confidence: float = 0.0
    volatility_response_rule: str = "normal_analysis"
    volatility_tags: list[str] = field(default_factory=list)
    high_impact_news_hard_block: bool = False
    spread_status: str = "unknown"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    open_trades_count: int = 0
    daily_trades_count: int = 0
    daily_drawdown_pct: float = 0.0


@dataclass
class DecisionOutput:
    # purpose: provide a complete, consistent analysis result for all consumers.
    # what it does: returns bias, decision, scores, blockers, and a clear reason.
    bias: str = "neutral"
    bias_strength: str = "weak"
    trade_decision: str = "NO_TRADE"
    confidence: float = 0.0
    structure_read: str = ""
    momentum_read: str = ""
    location_read: str = ""
    confirmation_read: str = ""
    buy_score: float = 0.0
    sell_score: float = 0.0
    no_trade_score: float = 0.0
    blockers: list[str] = field(default_factory=list)
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session: str = "unknown"
    can_create_execution_request: bool = False
    volatility_class: str = "normal_volatility"
    volatility_confidence: float = 0.0
    volatility_response_rule: str = "normal_analysis"
    volatility_tags: list[str] = field(default_factory=list)


class DecisionFramework:
    # purpose: evaluate buy/sell/no-trade candidates from confluence.
    # what it does: scores five pillars and never performs execution.
    NO_TRADE_DOMINANCE = 0.10

    def evaluate(self, inp: DecisionInput) -> DecisionOutput:
        # purpose: compute deterministic decision output from decision input.
        # what it does: applies structure, momentum, location, confirmation, blockers.
        settings = load_settings()
        thresholds = get_active_thresholds(settings)
        out = DecisionOutput(
            timestamp=inp.timestamp,
            session=inp.session,
            volatility_class=str(inp.volatility_class or "normal_volatility"),
            volatility_confidence=float(inp.volatility_confidence or 0.0),
            volatility_response_rule=str(inp.volatility_response_rule or "normal_analysis"),
            volatility_tags=list(inp.volatility_tags or []),
        )
        buy_score = 0.0
        sell_score = 0.0
        no_trade_score = 0.0
        blockers: list[str] = []

        if inp.higher_highs and inp.higher_lows:
            buy_score += 0.25
            out.structure_read = "Uptrend confirmed: higher highs + higher lows"
        elif inp.lower_highs and inp.lower_lows:
            sell_score += 0.25
            out.structure_read = "Downtrend confirmed: lower highs + lower lows"
        elif inp.higher_highs and inp.lower_lows:
            no_trade_score += 0.15
            out.structure_read = "Expanding range: conflicting structure"
        else:
            no_trade_score += 0.20
            out.structure_read = "No clear structure: ranging or choppy market"

        if inp.bullish_candle_strength >= 0.65:
            buy_score += 0.20
            out.momentum_read = f"Strong bullish momentum: {inp.bullish_candle_strength:.2f}"
        elif inp.bearish_candle_strength >= 0.65:
            sell_score += 0.20
            out.momentum_read = f"Strong bearish momentum: {inp.bearish_candle_strength:.2f}"
        elif inp.bullish_candle_strength >= 0.40:
            buy_score += 0.08
            out.momentum_read = f"Moderate bullish momentum: {inp.bullish_candle_strength:.2f}"
        elif inp.bearish_candle_strength >= 0.40:
            sell_score += 0.08
            out.momentum_read = f"Moderate bearish momentum: {inp.bearish_candle_strength:.2f}"
        else:
            no_trade_score += 0.15
            out.momentum_read = "Weak or mixed momentum: no directional conviction"

        if inp.near_support and not inp.near_resistance:
            buy_score += 0.20
            out.location_read = "Price near support: buy setup possible"
        elif inp.near_resistance and not inp.near_support:
            sell_score += 0.20
            out.location_read = "Price near resistance: sell setup possible"
        elif inp.in_middle_of_range:
            no_trade_score += 0.25
            out.location_read = "Price in middle of range: no edge, no trade"
        elif inp.near_support and inp.near_resistance:
            no_trade_score += 0.20
            out.location_read = "Compressed between support and resistance: avoid"
        else:
            no_trade_score += 0.10
            out.location_read = "Location unclear: insufficient level context"

        notes: list[str] = []
        if inp.dxy_direction == "falling":
            buy_score += 0.12
            notes.append("DXY falling: gold buy confirmation")
        elif inp.dxy_direction == "rising":
            sell_score += 0.12
            notes.append("DXY rising: gold sell confirmation")
        else:
            no_trade_score += 0.05
            notes.append("DXY flat: no macro confirmation")

        gold_bias_up = buy_score > sell_score
        if (gold_bias_up and inp.dxy_direction == "rising") or ((not gold_bias_up) and inp.dxy_direction == "falling"):
            buy_score *= 0.80
            sell_score *= 0.80
            notes.append("WARNING: Gold and DXY conflict: confidence reduced")

        if inp.us10y_direction == "falling":
            buy_score += 0.08
            notes.append("US10Y falling: supports gold long")
        elif inp.us10y_direction == "rising":
            sell_score += 0.08
            notes.append("US10Y rising: supports gold short")

        if inp.adx >= 35:
            if buy_score > sell_score:
                buy_score += 0.08
            else:
                sell_score += 0.08
            notes.append(f"ADX {inp.adx:.1f}: strong trend confirmed")
        elif inp.adx < 20:
            no_trade_score += 0.10
            notes.append(f"ADX {inp.adx:.1f}: weak trend, ranging conditions")
        out.confirmation_read = " | ".join(notes)

        if not inp.session_tradeable:
            blockers.append(f"Session blocked: {inp.session} is not a tradeable session")
            no_trade_score = 1.0
        if inp.in_news_window:
            blockers.append("News window active: no entries allowed")
            no_trade_score = 1.0
        # LEARNING MODE: threshold from get_active_thresholds()
        max_spread = float(thresholds["spread_max"])
        if inp.spread_pips > max_spread:
            blockers.append(f"Spread too wide: {inp.spread_pips:.2f} pips (max: {max_spread:.2f})")
            no_trade_score = 1.0
        # LEARNING MODE: threshold from get_active_thresholds()
        max_trades = int(thresholds["max_concurrent"])
        if inp.open_trades_count >= max_trades:
            blockers.append(f"Max concurrent trades reached: {inp.open_trades_count}/{max_trades}")
            no_trade_score = 1.0
        # LEARNING MODE: threshold from get_active_thresholds()
        max_daily = int(thresholds["max_trades_day"])
        if inp.daily_trades_count >= max_daily:
            blockers.append(f"Daily trade cap reached: {inp.daily_trades_count}/{max_daily}")
            no_trade_score = 1.0
        if inp.daily_drawdown_pct >= 5.0:
            blockers.append(f"Daily drawdown limit: {inp.daily_drawdown_pct:.1f}% (max: 5.0%)")
            no_trade_score = 1.0

        # FIX: volatility taxonomy response integration (analysis influence only).
        vol_class = str(inp.volatility_class or "normal_volatility")
        if bool(inp.high_impact_news_hard_block) or vol_class == "news_spike_volatility":
            blockers.append("high_impact_news_window")
            no_trade_score = 1.0
        elif vol_class == "extreme_volatility":
            blockers.append("volatility_hard_block")
            no_trade_score = 1.0
        elif vol_class == "liquidity_gap_volatility":
            blockers.append("spread_liquidity_too_poor")
            no_trade_score = 1.0
        elif vol_class == "chop_volatility":
            no_trade_score += 0.22
            buy_score *= 0.85
            sell_score *= 0.85
            out.confirmation_read = f"{out.confirmation_read} | Chop volatility: confidence reduced".strip(" |")
        elif vol_class == "elevated_volatility":
            buy_score *= 0.92
            sell_score *= 0.92
            no_trade_score += 0.05
            out.confirmation_read = f"{out.confirmation_read} | Elevated volatility: stricter quality".strip(" |")
        elif vol_class == "compressed_volatility":
            no_trade_score += 0.08
            out.confirmation_read = f"{out.confirmation_read} | Compressed volatility: breakout confirmation needed".strip(" |")
        elif vol_class == "session_transition_volatility":
            no_trade_score += 0.08
            buy_score *= 0.94
            sell_score *= 0.94
            out.confirmation_read = f"{out.confirmation_read} | Session transition: temporary confidence trim".strip(" |")
        elif vol_class == "trend_expansion_volatility":
            if inp.momentum_direction in {"bullish", "bearish"}:
                if buy_score > sell_score:
                    buy_score += 0.04
                else:
                    sell_score += 0.04
            out.confirmation_read = f"{out.confirmation_read} | Trend expansion: continuation allowed only if aligned".strip(" |")
        elif vol_class == "reversal_exhaustion_volatility":
            no_trade_score += 0.10
            buy_score *= 0.90
            sell_score *= 0.90
            out.confirmation_read = f"{out.confirmation_read} | Reversal exhaustion: requires full confirmation".strip(" |")

        out.buy_score = round(buy_score, 4)
        out.sell_score = round(sell_score, 4)
        out.no_trade_score = round(no_trade_score, 4)
        out.blockers = blockers
        if blockers:
            out.trade_decision = "NO_TRADE"
            out.bias = "neutral"
            out.bias_strength = "blocked"
            out.confidence = 0.0
            out.reason = f"Hard blockers present: {'; '.join(blockers)}"
            return out

        gap = abs(buy_score - sell_score)
        if gap < self.NO_TRADE_DOMINANCE:
            out.trade_decision = "NO_TRADE"
            out.reason = f"Buy score ({buy_score:.3f}) and sell score ({sell_score:.3f}) too close (gap: {gap:.3f}). No clear edge."
            out.confidence = max(buy_score, sell_score)
            return out

        # LEARNING MODE: threshold from get_active_thresholds()
        min_score = float(thresholds["min_score"])
        if buy_score > sell_score and buy_score >= min_score:
            out.trade_decision = "BUY_CANDIDATE"
            out.bias = "bullish"
            out.confidence = buy_score
            out.reason = f"Buy confluence confirmed. Score: {buy_score:.3f} vs sell: {sell_score:.3f}."
        elif sell_score > buy_score and sell_score >= min_score:
            out.trade_decision = "SELL_CANDIDATE"
            out.bias = "bearish"
            out.confidence = sell_score
            out.reason = f"Sell confluence confirmed. Score: {sell_score:.3f} vs buy: {buy_score:.3f}."
        else:
            out.trade_decision = "NO_TRADE"
            out.reason = f"Scores below threshold. Buy: {buy_score:.3f} Sell: {sell_score:.3f} (threshold: {min_score:.3f})"
            out.confidence = max(buy_score, sell_score)
            return out

        dominant = max(buy_score, sell_score)
        out.bias_strength = "strong" if dominant >= 0.75 else ("moderate" if dominant >= 0.60 else "weak")
        # LEARNING MODE: threshold from get_active_thresholds()
        threshold = float(thresholds["confidence"])
        out.can_create_execution_request = out.confidence >= threshold and out.trade_decision != "NO_TRADE"
        return out
