from __future__ import annotations

import os
from typing import Any
from datetime import datetime, timezone

from atlas_ai.analysis.chart_analysis import analyze_chart_features
from atlas_ai.analysis.market_regime import build_market_regime_and_reversal
from atlas_ai.analysis.setup_playbook import build_setup_radar, score_setup_playbook


def _clamp_score(raw: Any) -> int:
    try:
        score = float(raw)
        if score <= 1.0:
            score *= 100.0
        return int(max(0, min(100, round(score))))
    except Exception:
        return 0


def _confidence_level(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 65:
        return "medium"
    return "low"


def _bias_strength(score: int) -> str:
    if score >= 75:
        return "strong"
    if score >= 55:
        return "moderate"
    return "weak"


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _normalized_state(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _display_state(raw: Any, default: str = "unavailable") -> str:
    state = _normalized_state(raw)
    if state in {"", "unknown", "none", "n/a"}:
        return default
    return state


def _availability_state(raw: Any) -> str:
    state = _normalized_state(raw)
    if state in {"stale", "aging", "delayed"}:
        return "stale"
    if state in {"inferred", "derived", "estimated", "proxy"}:
        return "inferred"
    if state in {"", "unknown", "none", "n/a", "unavailable"}:
        return "unavailable"
    return "available"


def _strategy_key(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if "(" in text:
        text = text.split("(", 1)[0].strip()
    return text


def _strategy_family(strategy: str) -> str:
    key = _strategy_key(strategy)
    if key in {"rsi_mean_reversion", "bollinger_mean_reversion", "vwap_proxy_reversion", "stochastic_reversal", "gold_micro_structure_reversion"}:
        return "range_edge_reversal"
    if key in {"breakout", "gold_breakout_retest", "gold_liquidity_sweep"}:
        return "breakout_retest"
    if key in {"ema_crossover", "pullback_trend", "momentum_continuation", "macd_trend", "rsi_trend_hybrid", "gold_micro_impulse", "evolved_rule"}:
        return "pullback_continuation"
    return "unknown"


def _build_strategy_coverage(status: dict[str, Any], setup_radar: dict[str, Any]) -> dict[str, Any]:
    signals = [dict(row or {}) for row in list(status.get("signals", []) or []) if isinstance(row, dict)]
    top = dict(setup_radar.get("top_watch", {}) or setup_radar.get("top_candidate", {}) or {})
    playbook = str(top.get("name", "") or "")
    matching = [row for row in signals if _strategy_family(str(row.get("strategy", ""))) == playbook]
    directional = [
        row
        for row in matching
        if str(row.get("side", "flat") or "flat").lower() in {"buy", "sell"}
        and _safe_float(row.get("confidence", 0.0)) > 0.0
    ]
    best = max(directional, key=lambda row: _safe_float(row.get("confidence", 0.0)), default={})
    readiness = str(top.get("readiness", "") or "")
    coverage_gap = bool(readiness in {"ready_for_existing_gates", "developing"} and playbook and not directional)
    weak_coverage = bool(directional and _safe_float(best.get("confidence", 0.0)) < 0.15)
    return {
        "mode": "observability_only",
        "top_playbook": playbook or "unknown",
        "top_playbook_readiness": readiness or "unknown",
        "matching_signal_count": len(matching),
        "directional_signal_count": len(directional),
        "best_matching_signal": {
            "strategy": str(best.get("strategy", "") or ""),
            "side": str(best.get("side", "") or "flat"),
            "confidence": round(_safe_float(best.get("confidence", 0.0)), 4),
        } if best else {},
        "coverage_gap": coverage_gap,
        "weak_coverage": weak_coverage,
        "suggested_focus": (
            f"Top playbook {playbook} is visible, but no enabled strategy is producing a directional signal for it."
            if coverage_gap
            else (
                f"Top playbook {playbook} has only weak strategy confirmation."
                if weak_coverage
                else "Enabled strategies cover the current top playbook well enough for normal gates."
            )
        ),
        "policy_note": "Coverage diagnostic is observability only; it does not enable strategies or bypass execution gates.",
    }


def _derive_session_from_utc() -> str:
    h = datetime.now(timezone.utc).hour
    if 13 <= h < 17:
        return "overlap"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 20:
        return "new_york"
    if 0 <= h < 7:
        return "asia"
    return "off_hours"


def _normalize_symbol(raw: Any) -> str:
    return str(raw or "").strip().upper().replace("/", "")


def _timeframe_logic(status: dict[str, Any]) -> dict[str, Any]:
    primary_tf = str(status.get("timeframe") or "").strip().upper() or "UNKNOWN"
    htf = dict(status.get("higher_timeframe", {}) or {})
    confirm_tf = str(htf.get("confirm_tf") or status.get("higher_timeframe_confirm") or "").strip().upper() or "UNKNOWN"
    return {
        "structure_trend": primary_tf,
        "entry_confirmation": primary_tf,
        "execution_timing": primary_tf,
        "higher_timeframe_confirmation": confirm_tf,
        "higher_timeframe_bias": str(htf.get("bias", "unknown") or "unknown"),
        "data_source": "mt5_rates_api",
        "visible_terminal_chart_required": False,
    }


def _env_threshold(name: str, default: float) -> int:
    try:
        return int(max(0, min(100, round(float(os.getenv(name, str(default)))))))
    except Exception:
        return int(default)


def _feature_level(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 60:
        return "acceptable"
    if score >= 40:
        return "mixed"
    return "weak"


def _bounded_adjustment(total: float, value: float, *, low: float = -30.0, high: float = 30.0) -> float:
    return max(low, min(high, float(total) + float(value)))


def _humanize_reason(token: str) -> str:
    text = str(token or "").strip().replace("_", " ")
    if not text:
        return "unspecified risk"
    return text


def _alignment_state(score: int) -> str:
    if score >= 80:
        return "aligned"
    if score < 30:
        return "conflict"
    return "mixed"


def _account_tier_profile(balance: float) -> dict[str, Any]:
    if balance <= 0:
        return {
            "account_tier": "micro_account",
            "tier_reason": "balance unavailable; defaulting to safest tier",
            "risk_profile": "capital_preservation_strict",
        }
    if balance < 2500:
        return {
            "account_tier": "micro_account",
            "tier_reason": "small balance requires strict survival-first controls",
            "risk_profile": "capital_preservation_strict",
        }
    if balance < 25000:
        return {
            "account_tier": "small_account",
            "tier_reason": "balance still sensitive to drawdown and friction",
            "risk_profile": "capital_preservation_moderate",
        }
    return {
        "account_tier": "standard_account",
        "tier_reason": "balance supports standard guarded risk profile",
        "risk_profile": "capital_preservation_standard",
    }


def _size_tier_profile(balance: float) -> dict[str, Any]:
    tiers = [
        ("tier_0_0p01", 0.01, 750.0),
        ("tier_1_0p02", 0.02, 2500.0),
        ("tier_2_0p03", 0.03, 7500.0),
        ("tier_3_0p05", 0.05, 20000.0),
        ("tier_4_0p08", 0.08, 50000.0),
        ("tier_5_0p10", 0.10, None),
    ]
    current = tiers[0]
    for item in tiers:
        label, _max_lot, requirement = item
        if requirement is None or balance >= requirement:
            current = item
        else:
            break
    idx = tiers.index(current)
    next_tier = tiers[min(idx + 1, len(tiers) - 1)]
    next_req = next_tier[2]
    return {
        "current_size_tier": current[0],
        "current_size_tier_max_lot": current[1],
        "next_size_tier_requirement": f"balance>={next_req:.2f}" if next_req is not None else "max_tier_reached",
        "size_tier_reason": "bounded_stepwise_scaling",
    }


def _bounded_size_band(
    *,
    recommended_action: str,
    confluence_score: int,
    confluence_threshold: int,
    hard_blockers: list[str],
    contradictory_factors: list[str],
    macro_conflict: bool,
    event_watch: bool,
    spread_caution: bool,
    slippage_caution: bool,
    vix_caution: bool,
) -> dict[str, Any]:
    if hard_blockers:
        return {
            "band": "blocked",
            "cap_multiplier": 0.0,
            "risk_fraction_of_max": 0.0,
            "rationale": [
                "No size allocated because hard blockers are active.",
            ],
        }
    if recommended_action == "hold":
        return {
            "band": "blocked",
            "cap_multiplier": 0.0,
            "risk_fraction_of_max": 0.0,
            "rationale": [
                "No size allocated because confluence is below the active execution-quality threshold.",
            ],
        }

    rationale: list[str] = []
    if recommended_action.startswith("guarded_") and confluence_score >= confluence_threshold + 8:
        band = "full_guarded"
        cap = 1.0
        risk_fraction = 1.0
        rationale.append("High confluence passed the guarded threshold with room to spare.")
    elif recommended_action.startswith("guarded_"):
        band = "guarded"
        cap = 0.82
        risk_fraction = 0.82
        rationale.append("Qualified setup, but size stays below the max risk band by design.")
    else:
        band = "probe"
        cap = 0.62
        risk_fraction = 0.62
        rationale.append("Setup is developing, so only a reduced probe-size band is allowed.")

    caution_pairs = (
        (macro_conflict, "Macro context is mixed, so size is trimmed."),
        (event_watch, "Event/news risk is on watch, so size is trimmed."),
        (spread_caution, "Spread quality is not ideal, so size is trimmed."),
        (slippage_caution, "Slippage risk is elevated, so size is trimmed."),
        (vix_caution, "VIX risk regime is elevated, so size is trimmed."),
    )
    for active, note in caution_pairs:
        if active:
            cap = min(cap, 0.65)
            risk_fraction = min(risk_fraction, 0.65)
            rationale.append(note)

    if len(contradictory_factors) >= 3:
        cap = min(cap, 0.58)
        risk_fraction = min(risk_fraction, 0.58)
        rationale.append("Several contradictory factors are present, so size is held in a defensive band.")

    return {
        "band": band,
        "cap_multiplier": round(max(0.0, min(1.0, cap)), 3),
        "risk_fraction_of_max": round(max(0.0, min(1.0, risk_fraction)), 3),
        "rationale": rationale[:5],
    }


def build_structured_decision_analysis(
    *,
    status: dict[str, Any] | None,
    runtime: dict[str, Any] | None,
    chart_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    st = dict(status or {})
    rt = dict(runtime or {})
    chosen = dict(st.get("chosen", {}) or {})
    context = dict(st.get("context_snapshot", {}) or {})
    chart = analyze_chart_features(chart_input if chart_input is not None else (st.get("chart_features") if isinstance(st.get("chart_features"), dict) else None))
    regime_payload = build_market_regime_and_reversal(status=st, runtime=rt, chart_input=chart_input)
    market_regime = dict(regime_payload.get("market_regime", {}) or {})
    reversal_detection = dict(regime_payload.get("reversal_detection", {}) or {})
    setup_playbook = score_setup_playbook(status=st, runtime=rt, chart_input=chart)
    timeframe_logic = _timeframe_logic(st)
    structure_tf = str(timeframe_logic.get("structure_trend", "UNKNOWN"))

    action = _normalized_state(st.get("action"))
    symbol = _normalize_symbol(
        st.get("symbol")
        or dict(st.get("chosen", {}) or {}).get("symbol")
        or dict(st.get("context_snapshot", {}) or {}).get("symbol")
        or "XAUUSD"
    )
    side = _normalized_state(chosen.get("side"))
    bias_raw = side or action
    if bias_raw in {"buy", "long", "paper"}:
        market_bias = "buy" if side in {"buy", "long"} else "neutral"
    elif bias_raw in {"sell", "short"}:
        market_bias = "sell"
    else:
        market_bias = "neutral"

    base_score = _clamp_score(chosen.get("signal_confidence_seed", chosen.get("confidence", st.get("confidence", 0.0))))
    reasoning: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    supporting_factors: list[str] = []
    contradictory_factors: list[str] = []

    trend_state = _normalized_state(context.get("trend_state", chart.get("trend_state", "unclear")))
    trend_structure = str(context.get("trend_structure", chart.get("trend_structure", "mixed")) or "mixed")
    if trend_state == "unclear":
        if market_bias == "buy":
            trend_state = "uptrend"
        elif market_bias == "sell":
            trend_state = "downtrend"

    structure_side = "neutral"
    if trend_structure in {"HH/HL", "hh/hl"}:
        structure_side = "buy"
        trend_state = "uptrend"
        if market_bias in {"neutral", "buy"}:
            xau_structure_score = 92
            supporting_factors.append(f"higher highs and higher lows on {structure_tf}")
        else:
            xau_structure_score = 35
            warnings.append("structure_against_signal")
            contradictory_factors.append(f"{structure_tf} structure is bullish while the signal is sell")
    elif trend_structure in {"LH/LL", "lh/ll"}:
        structure_side = "sell"
        trend_state = "downtrend"
        if market_bias in {"neutral", "sell"}:
            xau_structure_score = 92
            supporting_factors.append(f"lower highs and lower lows on {structure_tf}")
        else:
            xau_structure_score = 35
            warnings.append("structure_against_signal")
            contradictory_factors.append(f"{structure_tf} structure is bearish while the signal is buy")
    elif trend_state in {"uptrend", "downtrend"}:
        xau_structure_score = 72
        supporting_factors.append(f"{trend_state} structure is present but less explicit")
    elif trend_state == "range":
        xau_structure_score = 28
        warnings.append("unclear_or_range_structure")
        contradictory_factors.append("market structure is ranging")
    else:
        xau_structure_score = 18
        warnings.append("unclear_or_range_structure")
        contradictory_factors.append("market structure is unclear")

    momentum_state = _normalized_state(context.get("momentum", chart.get("momentum_state", "mixed")))
    if momentum_state == "strong":
        xau_momentum_score = 88
        supporting_factors.append("momentum is supporting continuation")
    elif momentum_state == "weak":
        xau_momentum_score = 24
        warnings.append("weak_momentum")
        contradictory_factors.append("momentum is weak")
    else:
        xau_momentum_score = 50
        warnings.append("mixed_momentum")
        contradictory_factors.append("momentum is mixed")

    location_state = _normalized_state(context.get("location", chart.get("location_state", "unclear")))
    trend_strength = _safe_float(context.get("trend_strength", chart.get("trend_strength", 0.0)))
    if location_state == "pullback":
        location_bonus = 10
        supporting_factors.append("price is pulling back into structure")
    elif location_state == "breakout":
        location_bonus = 8
        supporting_factors.append("price is pressing a breakout location")
    elif location_state in {"consolidation", "mid_range", "unclear"}:
        location_bonus = -10
        warnings.append("suboptimal_location")
        contradictory_factors.append("entry location is suboptimal")
    else:
        location_bonus = 0

    setup_type = _normalized_state(st.get("setup_family", chart.get("setup_type_hint", "no_trade")))
    if setup_type not in {"pullback", "breakout", "consolidation", "continuation", "mean_reversion", "no_trade"}:
        setup_type = "no_trade"
    # purpose: when the regime engine reads ranging conditions, prefer
    # range-aware setup classification over blind continuation labeling.
    chart_setup_hint = _normalized_state(chart.get("setup_type_hint", ""))
    if (
        setup_type == "continuation"
        and str(market_regime.get("regime", "") or "").lower() == "ranging"
        and chart_setup_hint in {"mean_reversion", "consolidation"}
    ):
        setup_type = chart_setup_hint
    top_playbook = dict(setup_playbook.get("top_candidate", {}) or {})
    top_playbook_side = _normalized_state(top_playbook.get("side"))
    range_edge_playbook_ready = bool(
        str(top_playbook.get("name", "") or "") == "range_edge_reversal"
        and int(top_playbook.get("score", 0) or 0) >= 72
        and str(top_playbook.get("readiness", "") or "") == "ready_for_existing_gates"
        and top_playbook_side in {"buy", "sell"}
        and location_state not in {"mid_range", "unclear"}
        and setup_type in {"consolidation", "mean_reversion"}
    )

    risk_snapshot = dict(st.get("risk", {}) or {})
    balance = _safe_float(risk_snapshot.get("balance", st.get("balance", 0.0)))
    daily_loss_pct = _safe_float(risk_snapshot.get("daily_loss_pct", 0.0))
    trades_today = int(_safe_float(risk_snapshot.get("trades_today", 0)))
    account_profile = _account_tier_profile(balance)
    size_tier = _size_tier_profile(balance)
    account_tier = str(account_profile.get("account_tier", "standard_account"))

    spread_state = _normalized_state(context.get("spread_state"))
    liquidity_state = _normalized_state(context.get("liquidity_state"))
    spread_points = _safe_float(context.get("spread_points", rt.get("spread_points", 0.0)))
    max_spread_points = _safe_float(rt.get("max_spread_points", 35.0), default=35.0)
    slippage_state = _normalized_state(context.get("slippage_state"))
    slippage_points = _safe_float(context.get("slippage_points", rt.get("slippage_points", 0.0)))
    max_slippage_points = _safe_float(rt.get("max_slippage_points", 20.0), default=20.0)
    if max_spread_points <= 0:
        max_spread_points = 35.0
    if max_slippage_points <= 0:
        max_slippage_points = 20.0
    if spread_state == "wide" or (spread_points > 0 and spread_points >= max_spread_points):
        spread_safety_score = 10
        warnings.append("spread_liquidity_caution")
        contradictory_factors.append("spread conditions are unsafe")
    elif spread_state == "tight":
        spread_safety_score = 92
        supporting_factors.append("spread conditions are favorable")
    else:
        spread_safety_score = 68
    if liquidity_state in {"poor", "below_normal", "illiquid"}:
        spread_safety_score = min(spread_safety_score, 14 if liquidity_state == "poor" else 30)
        warnings.append("spread_liquidity_caution")
        contradictory_factors.append("liquidity is weak")
    elif liquidity_state == "normal":
        spread_safety_score = max(spread_safety_score, 82)
    if slippage_state in {"high", "elevated"} or (slippage_points > 0 and slippage_points >= max_slippage_points):
        spread_safety_score = min(spread_safety_score, 16 if slippage_state == "high" else 34)
        warnings.append("slippage_caution")
        contradictory_factors.append("slippage risk is elevated")
    elif slippage_state == "normal":
        spread_safety_score = max(spread_safety_score, 76)

    session_state = _normalized_state(context.get("session"))
    if session_state in {"", "unknown"}:
        session_state = _derive_session_from_utc()
    if session_state in {"london", "new_york", "ny_open"}:
        session_quality_score = 86
        supporting_factors.append(f"{session_state} session quality is acceptable")
    elif session_state in {"asia", "late_us"}:
        session_quality_score = 42
        warnings.append("session_behavior_caution")
        contradictory_factors.append(f"{session_state} session quality is weaker")
    elif session_state in {"rollover", "unknown", ""}:
        session_quality_score = 22
        warnings.append("session_behavior_caution")
        contradictory_factors.append("session quality is unclear")
    else:
        session_quality_score = 58

    dxy_alignment_raw = _normalized_state(
        context.get("dxy_alignment")
        or context.get("dxy_context_alignment")
        or context.get("dxy_conflict")
        or "unknown"
    )
    dxy_alignment_availability_raw = _availability_state(context.get("dxy_alignment", ""))
    # purpose: missing macro feed should remain explicitly unavailable; we keep
    # source metadata separate to show when downstream logic used inferred context.
    dxy_alignment_availability = dxy_alignment_availability_raw
    dxy_source = "macro_feed" if dxy_alignment_availability_raw in {"available", "stale"} else "price_only_inferred"
    dxy_trend = _display_state(context.get("dxy_trend", context.get("dxy_bias", "unknown")))
    if dxy_alignment_raw in {"aligned", "supportive", "favorable"}:
        dxy_alignment_score = 84
        supporting_factors.append("DXY context is aligned")
    elif dxy_alignment_raw in {"conflict", "conflicting", "opposed"}:
        dxy_alignment_score = 18
        warnings.append("dxy_conflict")
        contradictory_factors.append("DXY context is conflicting")
    else:
        dxy_alignment_score = 52

    us10y_alignment_raw = _normalized_state(
        context.get("us10y_alignment")
        or context.get("yield_alignment")
        or context.get("us10y_conflict")
        or "unknown"
    )
    us10y_alignment_availability_raw = _availability_state(context.get("us10y_alignment", ""))
    us10y_alignment_availability = us10y_alignment_availability_raw
    us10y_source = "macro_feed" if us10y_alignment_availability_raw in {"available", "stale"} else "price_only_inferred"
    us10y_trend = _display_state(context.get("us10y_trend", context.get("yield_trend", "unknown")))
    if us10y_alignment_raw in {"aligned", "supportive", "favorable"}:
        yield_alignment_score = 84
        supporting_factors.append("US10Y context is aligned")
    elif us10y_alignment_raw in {"conflict", "conflicting", "opposed"}:
        yield_alignment_score = 18
        warnings.append("us10y_conflict")
        contradictory_factors.append("US10Y yields are conflicting")
    else:
        yield_alignment_score = 52

    vix_regime = _display_state(context.get("vix_regime", context.get("vix_state", "unknown")))
    vix_alignment_raw = _normalized_state(context.get("vix_alignment", context.get("vix_context_alignment", "unknown")))
    vix_availability_raw = _availability_state(context.get("vix_regime", context.get("vix_state", "")))
    vix_availability = vix_availability_raw
    vix_source = "macro_feed" if vix_availability_raw in {"available", "stale"} else "price_only_inferred"
    if vix_alignment_raw in {"aligned", "supportive", "favorable"}:
        vix_alignment_score = 74
        supporting_factors.append("VIX risk regime is supportive")
    elif vix_alignment_raw in {"conflict", "conflicting", "opposed"}:
        vix_alignment_score = 26
        warnings.append("vix_risk_conflict")
        contradictory_factors.append("VIX risk regime is conflicting")
    elif vix_regime in {"high", "elevated", "stress", "fear"}:
        vix_alignment_score = 36
        warnings.append("vix_risk_elevated")
        contradictory_factors.append("VIX risk regime is elevated")
    elif vix_regime in {"calm", "normal", "low"}:
        vix_alignment_score = 68
        supporting_factors.append("VIX risk regime is calm enough")
    else:
        vix_alignment_score = 54

    oil_backdrop = _display_state(context.get("oil_backdrop", context.get("oil_regime", "unknown")))
    oil_alignment_raw = _normalized_state(context.get("oil_alignment", context.get("oil_context_alignment", "unknown")))
    oil_availability_raw = _availability_state(context.get("oil_backdrop", context.get("oil_regime", "")))
    oil_availability = oil_availability_raw
    oil_source = "macro_feed" if oil_availability_raw in {"available", "stale"} else "price_only_inferred"
    if oil_alignment_raw in {"aligned", "supportive", "favorable"}:
        oil_context_score = 62
        supporting_factors.append("oil backdrop is mildly supportive")
    elif oil_alignment_raw in {"conflict", "conflicting", "opposed"}:
        oil_context_score = 38
        contradictory_factors.append("oil backdrop is mildly conflicting")
    elif oil_backdrop in {"inflationary", "supportive"}:
        oil_context_score = 58
    else:
        oil_context_score = 50

    event_risk_state = _normalized_state(
        context.get("event_risk_state")
        or context.get("news_risk_state")
        or context.get("news_risk_classification")
        or context.get("event_risk")
        or "unknown"
    )
    in_high_impact_news_window = bool(
        context.get("in_high_impact_news_window")
        or context.get("high_impact_news_window")
        or context.get("news_window_active")
    )
    if event_risk_state in {"none", "low", "clear", "calm"}:
        event_risk_score = 92
        supporting_factors.append("event risk is low")
    elif event_risk_state in {"medium", "moderate", "watch"}:
        event_risk_score = 55
        contradictory_factors.append("event risk needs caution")
    elif event_risk_state in {"high", "elevated", "locked", "imminent", "extreme"}:
        event_risk_score = 12
        warnings.append("event_risk_high")
        contradictory_factors.append("event risk is too high")
    else:
        event_risk_score = 62
    if in_high_impact_news_window:
        event_risk_score = min(event_risk_score, 8)
        warnings.append("high_impact_news_window")
        contradictory_factors.append("high-impact news window is active")

    news_context_raw = _normalized_state(context.get("news_context_score", context.get("news_sentiment", context.get("macro_classification", "unknown"))))
    if news_context_raw in {"supportive", "aligned", "risk_on_supportive", "gold_supportive"}:
        news_context_score = 74
        supporting_factors.append("macro/news context is supportive")
    elif news_context_raw in {"conflict", "mixed", "conflicting", "risk_off_conflict"}:
        news_context_score = 34
        contradictory_factors.append("macro/news context is mixed")
    elif news_context_raw in {"negative", "high_risk"}:
        news_context_score = 18
        contradictory_factors.append("macro/news context is adverse")
    else:
        news_context_score = 52

    features = {
        "xau_structure_score": xau_structure_score,
        "xau_momentum_score": xau_momentum_score,
        "dxy_alignment_score": dxy_alignment_score,
        "yield_alignment_score": yield_alignment_score,
        "vix_alignment_score": vix_alignment_score,
        "oil_context_score": oil_context_score,
        "spread_safety_score": spread_safety_score,
        "session_quality_score": session_quality_score,
        "event_risk_score": event_risk_score,
        "news_context_score": news_context_score,
    }

    weighted_score = (
        xau_structure_score * 0.24
        + xau_momentum_score * 0.18
        + dxy_alignment_score * 0.10
        + yield_alignment_score * 0.10
        + vix_alignment_score * 0.07
        + oil_context_score * 0.03
        + spread_safety_score * 0.12
        + session_quality_score * 0.10
        + event_risk_score * 0.10
        + news_context_score * 0.06
        + max(0, location_bonus) * 0.5
    )
    runtime_confluence_threshold = rt.get("confluence_trade_threshold")
    if runtime_confluence_threshold is not None:
        confluence_threshold = int(max(0, min(100, round(_safe_float(runtime_confluence_threshold, 85)))))
    else:
        # LEARNING MODE: when runtime does not explicitly override the threshold,
        # derive a realistic analysis threshold from active learning-profile values.
        confluence_default = int(round(_safe_float(_env_threshold("ATLAS_CONFLUENCE_TRADE_THRESHOLD", 85), 85)))
        learning_mode_active = bool(st.get("learning_mode", False)) or str(st.get("mode_label", "")).upper().startswith("LEARNING")
        if learning_mode_active:
            threshold_snapshot = dict(st.get("thresholds", {}) or {})
            conf_floor = _safe_float(threshold_snapshot.get("confidence", 0.44), 0.44)
            min_score_floor = _safe_float(threshold_snapshot.get("min_score", conf_floor), conf_floor)
            derived_learning_threshold = int(round(max(min_score_floor * 100.0, (conf_floor * 100.0) + 8.0)))
            confluence_default = max(50, min(75, derived_learning_threshold))
            if bool(rt.get("analysis_confidence_margin_relaxed", False)):
                confluence_default = max(50, confluence_default - 8)
        confluence_threshold = int(max(0, min(100, confluence_default)))
    if account_tier == "micro_account":
        confluence_threshold = min(99, confluence_threshold + 10)
    elif account_tier == "small_account":
        confluence_threshold = min(99, confluence_threshold + 4)
    confluence_score = int(max(0, min(100, round((weighted_score * 0.82) + (base_score * 0.18) + location_bonus))))
    macro_conflict_threshold = int(max(0, min(100, round(_safe_float(rt.get("macro_conflict_block_threshold", _env_threshold("ATLAS_MACRO_CONFLICT_BLOCK_THRESHOLD", 30)), 30)))))
    event_risk_block_threshold = int(max(0, min(100, round(_safe_float(rt.get("event_risk_block_threshold", _env_threshold("ATLAS_EVENT_RISK_BLOCK_THRESHOLD", 25)), 25)))))

    if xau_structure_score < 40 and not range_edge_playbook_ready:
        blockers.append("structure_too_weak")
    elif xau_structure_score < 40 and range_edge_playbook_ready:
        warnings.append("range_structure_context")
        reasoning.append("range-edge setup is being judged by range/rejection context instead of trend structure")
    if symbol != "XAUUSD":
        blockers.append("xauusd_only_scope")
    if spread_safety_score < 35:
        blockers.append("spread_liquidity_too_poor")
    if slippage_state == "high" or (slippage_points > 0 and slippage_points >= max_slippage_points):
        blockers.append("slippage_risk_too_high")
    if event_risk_score < event_risk_block_threshold:
        blockers.append("event_risk_too_high")
    if in_high_impact_news_window:
        blockers.append("high_impact_news_window")
    if dxy_alignment_score < macro_conflict_threshold and yield_alignment_score < macro_conflict_threshold:
        blockers.append("macro_context_conflict_too_strong")
    if setup_type in {"no_trade", ""}:
        blockers.append("no_clear_setup")
    spread_block_reasons: list[str] = []
    expected_move_points = max(
        _safe_float(dict(st.get("chosen", {}) or {}).get("take_profit_points", 0.0)),
        _safe_float(dict(st.get("chosen", {}) or {}).get("stop_loss_points", 0.0)),
    )
    spread_to_reward = (expected_move_points / spread_points) if spread_points > 0 else 0.0
    min_reward_to_spread = 4.0 if account_tier == "micro_account" else (3.0 if account_tier == "small_account" else 2.0)
    if spread_points > 0 and spread_to_reward < min_reward_to_spread:
        blockers.append("spread_cost_too_high_for_expected_move")
        spread_block_reasons.append("reward_to_spread_below_minimum")
    reward_vs_risk_ratio = 0.0
    sl_points = _safe_float(dict(st.get("chosen", {}) or {}).get("stop_loss_points", 0.0))
    tp_points = _safe_float(dict(st.get("chosen", {}) or {}).get("take_profit_points", 0.0))
    if sl_points > 0:
        reward_vs_risk_ratio = max(0.0, tp_points / sl_points)

    if account_tier == "micro_account":
        if setup_type in {"mean_reversion", "consolidation"}:
            blockers.append("reversal_restricted_for_micro_account")
        if bool(reversal_detection.get("reversal_candidate", False)) and (
            not bool(reversal_detection.get("confirmed_reversal", False)) or confluence_score < 93
        ):
            blockers.append("reversal_confirmation_not_strong_enough_for_micro_account")
        if daily_loss_pct >= 0.012:
            blockers.append("micro_daily_drawdown_limit_reached")
        if trades_today >= 3:
            blockers.append("micro_frequency_guard")
    elif account_tier == "small_account":
        if daily_loss_pct >= 0.02:
            blockers.append("small_account_daily_drawdown_limit_reached")
        if trades_today >= 5:
            blockers.append("small_account_frequency_guard")
    if setup_type == "consolidation" and xau_structure_score < 70 and not range_edge_playbook_ready:
        blockers.append("no_clear_setup")
    if bool(rt.get("freshness_runtime", {}).get("is_stale", False)):
        blockers.append("stale_context")
        warnings.append("stale_context")
    if str(market_regime.get("regime", "unknown")) == "volatile" and int(market_regime.get("regime_confidence", 0) or 0) >= 70:
        blockers.append("volatile_regime_defensive_block")
    if str(market_regime.get("regime", "unknown")) == "trending" and setup_type in {"mean_reversion", "consolidation"}:
        warnings.append("counter_trend_setup_caution")
        contradictory_factors.append("setup leans counter-trend in a trending regime")
    if str(market_regime.get("regime", "unknown")) == "ranging" and setup_type in {"continuation", "pullback", "breakout"}:
        warnings.append("range_regime_continuation_caution")
        contradictory_factors.append("trend-continuation setup is less reliable in a ranging regime")

    if xau_structure_score >= 80:
        reasoning.append(f"{structure_tf} structure is clear")
    if xau_momentum_score >= 80:
        reasoning.append("momentum is strong enough to support continuation")
    if dxy_alignment_score >= 80:
        reasoning.append("DXY is aligned with the idea")
    if yield_alignment_score >= 80:
        reasoning.append("US10Y context is aligned with the idea")
    if event_risk_score < 55:
        reasoning.append("event/news risk requires caution")
    if news_context_score < 45:
        reasoning.append("macro/news context is mixed rather than decisive")
    if vix_alignment_score < 45:
        reasoning.append("VIX is adding risk-off caution")
    if spread_safety_score < 60:
        reasoning.append("execution friction is elevated")
    if market_regime:
        reasoning.append(
            f"market regime reads as {str(market_regime.get('regime', 'unknown'))} with {int(market_regime.get('regime_confidence', 0) or 0)}/100 confidence"
        )
    if bool(reversal_detection.get("reversal_candidate", False)) and not bool(reversal_detection.get("confirmed_reversal", False)):
        reasoning.append("reversal idea is present but still unconfirmed")

    hard_blockers = list(dict.fromkeys(blockers))
    conviction_adjustments: list[dict[str, Any]] = []

    def _adjust_conviction(label: str, delta: float, reason: str) -> None:
        nonlocal confluence_score
        conviction_adjustments.append(
            {
                "label": str(label),
                "delta": round(float(delta), 2),
                "reason": str(reason),
            }
        )
        confluence_score = int(round(_bounded_adjustment(confluence_score, delta, low=0.0, high=100.0)))

    same_side_support = int(dict(st.get("same_side_support", {}) or {}).get("count", 0) or 0)
    if same_side_support >= 3:
        _adjust_conviction("same_side_support", 4.0, "multiple active signals support the chosen direction")
    elif same_side_support == 2:
        _adjust_conviction("same_side_support", 2.0, "more than one active signal supports the chosen direction")
    elif side in {"buy", "sell"} and same_side_support <= 0:
        _adjust_conviction("same_side_support", -6.0, "the chosen direction lacks same-side confirmation support")

    freshness_runtime = dict(rt.get("freshness_runtime", st.get("freshness_runtime", {})) or {})
    freshness_state = _normalized_state(freshness_runtime.get("state"))
    if bool(freshness_runtime.get("is_stale", False)) or freshness_state == "stale":
        _adjust_conviction("freshness", -12.0, "runtime context is stale")
    elif freshness_state in {"aging", "delayed"}:
        _adjust_conviction("freshness", -5.0, "runtime context is aging")

    safety_layer = dict(st.get("safety_layer", {}) or {})
    if bool(safety_layer.get("safety_lock_active", False)):
        _adjust_conviction("safety_lock", -25.0, "safety lock is active")
    elif bool(safety_layer.get("cooldown_active", False)):
        _adjust_conviction("cooldown", -10.0, "cooldown is active")
    elif not bool(safety_layer.get("recovery_ready", True)):
        _adjust_conviction("recovery", -8.0, "recovery conditions are not fully ready")

    risk = dict(st.get("risk", {}) or {})
    consecutive_losses = int(risk.get("consecutive_losses", 0) or 0)
    if consecutive_losses > 0:
        _adjust_conviction(
            "recent_losses",
            -min(8.0, float(consecutive_losses) * 2.5),
            "recent losses reduce conviction until the read stabilizes",
        )

    if side == "flat":
        confluence_score = min(confluence_score, 45)
    if hard_blockers:
        blocker_cap = 59 if side in {"buy", "sell"} else 49
        confluence_score = min(confluence_score, blocker_cap)

    conviction_score = int(max(0, min(100, round(confluence_score))))
    conviction_level = _confidence_level(conviction_score)

    if hard_blockers:
        entry_quality = "avoid"
    elif conviction_score >= confluence_threshold:
        entry_quality = "good"
    elif conviction_score >= max(55, confluence_threshold - 15):
        entry_quality = "risky"
    else:
        entry_quality = "avoid"

    if market_bias == "neutral" and trend_state == "uptrend" and xau_structure_score >= 70 and xau_momentum_score >= 70:
        market_bias = "buy"
    if market_bias == "neutral" and trend_state == "downtrend" and xau_structure_score >= 70 and xau_momentum_score >= 70:
        market_bias = "sell"
    if hard_blockers:
        market_bias = "neutral" if any(x in hard_blockers for x in {"spread_liquidity_too_poor", "event_risk_too_high", "macro_context_conflict_too_strong", "no_clear_setup"}) else market_bias

    recommended_action = "hold"
    if not hard_blockers and conviction_score >= confluence_threshold and market_bias in {"buy", "sell"}:
        recommended_action = f"guarded_{market_bias}"
    elif not hard_blockers and conviction_score >= max(55, confluence_threshold - 15) and market_bias in {"buy", "sell"}:
        recommended_action = f"wait_for_m1_confirmation_{market_bias}"

    explanation = "No-trade: setup quality is insufficient right now."
    if hard_blockers:
        if "no_clear_setup" in hard_blockers:
            explanation = "No-trade: no valid setup is confirmed yet; structure/location confirmation is missing."
        else:
            explanation = f"No-trade: {', '.join(x.replace('_', ' ') for x in hard_blockers[:3])}."
    elif recommended_action.startswith("wait_for_m1_confirmation_"):
        explanation = "Setup is developing, but confirmation is still required."
    elif recommended_action.startswith("guarded_"):
        explanation = "Setup quality is acceptable for a guarded review."

    spread_explanation = "score_consistent_with_spread_and_liquidity"
    if (spread_state in {"tight", "normal"} and spread_safety_score <= 20) or liquidity_state in {"unknown", ""}:
        spread_explanation = "weak_due_to_missing_liquidity_context"
    context_alignment = {
        "dxy": {
            "state": _alignment_state(dxy_alignment_score),
            "trend": dxy_trend or "unavailable",
            "score": dxy_alignment_score,
            "availability": dxy_alignment_availability,
            "source": dxy_source,
        },
        "us10y": {
            "state": _alignment_state(yield_alignment_score),
            "trend": us10y_trend or "unavailable",
            "score": yield_alignment_score,
            "availability": us10y_alignment_availability,
            "source": us10y_source,
        },
        "vix": {
            "state": _alignment_state(vix_alignment_score),
            "regime": vix_regime or "unavailable",
            "score": vix_alignment_score,
            "availability": vix_availability,
            "source": vix_source,
        },
        "oil": {
            "state": _alignment_state(oil_context_score),
            "backdrop": oil_backdrop or "unavailable",
            "score": oil_context_score,
            "availability": oil_availability,
            "source": oil_source,
        },
        "spread_liquidity": {
            "spread_state": spread_state or "unknown",
            "liquidity_state": liquidity_state or "unknown",
            "slippage_state": slippage_state or "unknown",
            "spread_points": round(spread_points, 2) if spread_points > 0 else 0.0,
            "spread_pips": round(spread_points / 10.0, 2) if spread_points > 0 else 0.0,
            "slippage_points": round(slippage_points, 2) if slippage_points > 0 else 0.0,
            "score": spread_safety_score,
            "state": "caution" if spread_safety_score < 60 else "acceptable",
            "spread_status": "blocked" if spread_safety_score < 35 else ("elevated" if spread_safety_score < 60 else "normal"),
            "spread_explanation": spread_explanation,
            "reward_to_spread_ratio": round(spread_to_reward, 3) if spread_points > 0 else 0.0,
            "min_reward_to_spread_ratio": float(min_reward_to_spread),
            "spread_block_reasons": spread_block_reasons[:4],
        },
        "session": {
            "state": session_state or "unknown",
            "score": session_quality_score,
            "state_class": "caution" if session_quality_score < 60 else "acceptable",
            "availability": "available",
        },
        "regime": {
            "state": str(market_regime.get("regime", "unknown") or "unknown"),
            "score": int(market_regime.get("regime_confidence", 0) or 0),
            "reasons": list(market_regime.get("regime_reasons", []) or []),
        },
        "reversal": {
            "candidate": bool(reversal_detection.get("reversal_candidate", False)),
            "confirmed": bool(reversal_detection.get("confirmed_reversal", False)),
            "bias": str(reversal_detection.get("reversal_bias", "neutral") or "neutral"),
            "score": int(reversal_detection.get("reversal_confidence", 0) or 0),
        },
        "event_news": {
            "event_risk_state": event_risk_state or "unknown",
            "event_risk_score": event_risk_score,
            "news_context_score": news_context_score,
            "news_risk_classification": str(context.get("news_risk_classification", "unknown") or "unknown"),
            "news_sentiment": str(context.get("news_sentiment", "unknown") or "unknown"),
            "macro_classification": str(context.get("macro_classification", "unknown") or "unknown"),
            "availability": "unavailable" if all(_availability_state(context.get(k, "")) == "unavailable" for k in ("event_risk_state", "news_sentiment", "macro_classification")) else "available",
        },
    }
    size_band_rationale = _bounded_size_band(
        recommended_action=recommended_action,
        confluence_score=confluence_score,
        confluence_threshold=confluence_threshold,
        hard_blockers=hard_blockers,
        contradictory_factors=contradictory_factors,
        macro_conflict=bool("macro_context_conflict_too_strong" in hard_blockers or dxy_alignment_score < 45 or yield_alignment_score < 45),
        event_watch=event_risk_state in {"medium", "moderate", "watch"},
        spread_caution=spread_safety_score < 60,
        slippage_caution=slippage_state in {"elevated", "high"},
        vix_caution=vix_regime in {"high", "elevated", "stress", "fear"} or vix_alignment_score < 45,
    )
    macro_context = {
        "xau_symbol": symbol,
        "dxy": dict(context_alignment["dxy"]),
        "us10y": dict(context_alignment["us10y"]),
        "vix": dict(context_alignment["vix"]),
        "oil": dict(context_alignment["oil"]),
        "event_news": dict(context_alignment["event_news"]),
        "execution_friction": dict(context_alignment["spread_liquidity"]),
        "session": dict(context_alignment["session"]),
        "regime": dict(context_alignment["regime"]),
        "reversal": dict(context_alignment["reversal"]),
        "operator_terminal": {
            "reads": [
                f"DXY={context_alignment['dxy']['state']}",
                f"US10Y={context_alignment['us10y']['state']}",
                f"VIX={context_alignment['vix']['regime']}",
                f"oil={context_alignment['oil']['backdrop']}",
                f"event={context_alignment['event_news']['event_risk_state']}",
                f"spread={context_alignment['spread_liquidity']['spread_state']}",
                f"slippage={context_alignment['spread_liquidity']['slippage_state']}",
                f"session={context_alignment['session']['state']}",
                f"regime={context_alignment['regime']['state']}",
                f"reversal={'confirmed' if context_alignment['reversal']['confirmed'] else ('candidate' if context_alignment['reversal']['candidate'] else 'none')}",
            ],
            "allow_reason": supporting_factors[:4],
            "block_reason": hard_blockers[:4] or contradictory_factors[:4],
        },
    }
    # FIX: explicit confluence scoring framework for BUY/SELL/NO_TRADE consistency.
    buy_score = int(max(0, min(100, round(
        (xau_structure_score * (0.55 if trend_state != "uptrend" else 0.95))
        + (xau_momentum_score * 0.45)
        + max(0, location_bonus) * 1.6
        + (8 if market_bias == "buy" else 0)
        + max(0, dxy_alignment_score - 55) * 0.20
        + max(0, yield_alignment_score - 55) * 0.15
    ))))
    sell_score = int(max(0, min(100, round(
        (xau_structure_score * (0.55 if trend_state != "downtrend" else 0.95))
        + (xau_momentum_score * 0.45)
        + max(0, location_bonus) * 1.6
        + (8 if market_bias == "sell" else 0)
        + max(0, dxy_alignment_score - 55) * 0.20
        + max(0, yield_alignment_score - 55) * 0.15
    ))))
    if trend_state == "uptrend":
        sell_score = max(0, sell_score - 18)
    elif trend_state == "downtrend":
        buy_score = max(0, buy_score - 18)
    if market_bias == "buy":
        sell_score = max(0, sell_score - 10)
    elif market_bias == "sell":
        buy_score = max(0, buy_score - 10)
    signal_bias = side if side in {"buy", "sell"} else "neutral"
    chart_structure_bias = structure_side if structure_side in {"buy", "sell"} else "neutral"
    higher_timeframe_bias = str(timeframe_logic.get("higher_timeframe_bias", "unknown") or "unknown")
    conflict_penalty = 20 if ("macro_context_conflict_too_strong" in hard_blockers) else 0
    no_trade_score = int(max(0, min(100, round(
        50
        + (18 if "no_clear_setup" in hard_blockers else 0)
        + (18 if "structure_too_weak" in hard_blockers else 0)
        + (14 if "spread_liquidity_too_poor" in hard_blockers else 0)
        + (14 if "event_risk_too_high" in hard_blockers else 0)
        + conflict_penalty
        + (12 if trend_state in {"range", "unclear"} else 0)
        + (10 if momentum_state in {"mixed", "weak"} else 0)
        + (10 if location_state in {"mid_range", "unclear", "consolidation"} else 0)
    ))))
    if hard_blockers:
        buy_score = max(0, buy_score - 25)
        sell_score = max(0, sell_score - 25)

    trade_decision = "NO_TRADE"
    decision_threshold = max(60, confluence_threshold - 8)
    directional_quality_gate = conviction_score >= max(55, confluence_threshold - 15)
    directional_allowed = ("no_clear_setup" not in hard_blockers and "structure_too_weak" not in hard_blockers)
    if directional_allowed and directional_quality_gate and buy_score >= sell_score + 8 and buy_score >= no_trade_score and buy_score >= decision_threshold:
        trade_decision = "BUY"
    elif directional_allowed and directional_quality_gate and sell_score >= buy_score + 8 and sell_score >= no_trade_score and sell_score >= decision_threshold:
        trade_decision = "SELL"

    decision_bias = "bullish" if market_bias == "buy" else ("bearish" if market_bias == "sell" else "neutral")
    if trade_decision == "BUY":
        decision_bias = "bullish"
    elif trade_decision == "SELL":
        decision_bias = "bearish"
    elif trend_state in {"range", "unclear"} or hard_blockers:
        decision_bias = "neutral"
    final_trade_bias = "buy" if trade_decision == "BUY" else ("sell" if trade_decision == "SELL" else "neutral")

    bias_breakdown = {
        "signal_bias": signal_bias,
        "chart_structure_bias": chart_structure_bias,
        "higher_timeframe_bias": higher_timeframe_bias,
        "final_trade_bias": final_trade_bias,
        "legacy_market_bias": market_bias,
        "conflict": bool(
            signal_bias in {"buy", "sell"}
            and chart_structure_bias in {"buy", "sell"}
            and signal_bias != chart_structure_bias
        ),
        "explanation": (
            "Signal direction conflicts with chart structure; final trade bias stays neutral until alignment improves."
            if signal_bias in {"buy", "sell"} and chart_structure_bias in {"buy", "sell"} and signal_bias != chart_structure_bias
            else "Signal, structure, and final decision are separated so candidate direction is not mistaken for market direction."
        ),
    }

    # purpose: enforce one coherent directional outcome across recommendation,
    # protocol, and trading response surfaces.
    if hard_blockers:
        recommended_action = "hold"
    elif trade_decision == "BUY":
        recommended_action = "guarded_buy" if conviction_score >= confluence_threshold else "wait_for_m1_confirmation_buy"
    elif trade_decision == "SELL":
        recommended_action = "guarded_sell" if conviction_score >= confluence_threshold else "wait_for_m1_confirmation_sell"
    else:
        recommended_action = "hold"

    if hard_blockers:
        if "no_clear_setup" in hard_blockers:
            explanation = "No-trade: no valid setup is confirmed yet; structure/location confirmation is missing."
        else:
            explanation = f"No-trade: {', '.join(x.replace('_', ' ') for x in hard_blockers[:3])}."
    elif recommended_action.startswith("wait_for_m1_confirmation_"):
        explanation = "Setup is developing, but confirmation is still required."
    elif recommended_action.startswith("guarded_"):
        explanation = "Setup quality is acceptable for a guarded review."
    else:
        explanation = "No-trade: setup quality is insufficient right now."

    # These are filled after critique synthesis and mutated in-place so
    # all response surfaces share the same blocker ranking object.
    top_blockers: list[dict[str, Any]] = []
    top_blocker_summary: list[str] = []

    trading_response = {
        "bias": decision_bias,
        "bias_breakdown": bias_breakdown,
        "bias_strength": _bias_strength(max(buy_score, sell_score) if decision_bias != "neutral" else no_trade_score),
        "trade_decision": trade_decision,
        "confidence": {
            "score": conviction_score,
            "level": conviction_level,
        },
        "buy_score": buy_score,
        "sell_score": sell_score,
        "no_trade_score": no_trade_score,
        "structure_read": {
            "state": trend_state or "unclear",
            "structure": trend_structure or "mixed",
            "score": xau_structure_score,
        },
        "momentum_read": {
            "state": momentum_state or "mixed",
            "score": xau_momentum_score,
        },
        "location_read": {
            "state": location_state or "unclear",
            "bonus": location_bonus,
        },
        "confirmation_read": {
            "dxy_alignment_score": dxy_alignment_score,
            "us10y_alignment_score": yield_alignment_score,
            "session_state": session_state or "unknown",
            "spread_state": spread_state or "unknown",
            "regime_state": str(market_regime.get("regime", "unknown") or "unknown"),
            "event_risk_state": event_risk_state or "unknown",
        },
        "reward_vs_risk": {
            "ratio": round(reward_vs_risk_ratio, 3),
            "take_profit_points": round(tp_points, 3),
            "stop_loss_points": round(sl_points, 3),
        },
        "blockers": hard_blockers[:8],
        "top_blockers": top_blockers,
        "top_blocker_summary": top_blocker_summary,
        "context_alignment": context_alignment,
        "recommended_action": recommended_action,
        "size_band_rationale": size_band_rationale,
        "safety_note": "Capital preservation first. No forced trade if structure, momentum, spread, session, or macro context are not aligned.",
        "timeframe_logic": timeframe_logic,
        "execution_mode": "guarded",
        "market_regime": market_regime,
        "reversal_detection": reversal_detection,
        "account_tier": account_tier,
        "risk_profile": str(account_profile.get("risk_profile", "capital_preservation_standard")),
        "size_tier": {
            "current_size_tier": str(size_tier.get("current_size_tier", "")),
            "next_size_tier_requirement": str(size_tier.get("next_size_tier_requirement", "")),
            "size_tier_reason": str(size_tier.get("size_tier_reason", "")),
        },
    }
    pre_trade_memo = {
        "bias": final_trade_bias,
        "signal_bias": signal_bias,
        "chart_structure_bias": chart_structure_bias,
        "higher_timeframe_bias": higher_timeframe_bias,
        "bias_breakdown": bias_breakdown,
        "setup_type": setup_type if setup_type != "no_trade" else "no_clear_setup",
        "confluence_score": confluence_score,
        "confluence_threshold": confluence_threshold,
        "reward_vs_risk": dict(trading_response["reward_vs_risk"]),
        "supporting_factors": supporting_factors[:6],
        "contradictory_factors": contradictory_factors[:6],
        "reasons_not_to_trade": hard_blockers[:6],
        "top_blockers": top_blockers,
        "final_decision": recommended_action,
        "size_band_rationale": size_band_rationale,
        "timeframe_logic": trading_response["timeframe_logic"],
        "market_regime": market_regime,
        "reversal_detection": reversal_detection,
        "account_tier": account_tier,
        "risk_profile": str(account_profile.get("risk_profile", "capital_preservation_standard")),
        "size_tier": dict(trading_response.get("size_tier", {}) or {}),
        "setup_playbook": dict(setup_playbook.get("top_candidate", {}) or {}),
    }
    critique_pool: list[str] = []
    critique_pool.extend([_humanize_reason(x) for x in hard_blockers])
    critique_pool.extend([_humanize_reason(x) for x in warnings])
    critique_pool.extend([_humanize_reason(x) for x in contradictory_factors])
    top_failure_reasons: list[str] = []
    for item in critique_pool:
        normalized = item.lower().strip()
        if not normalized:
            continue
        if normalized in {x.lower() for x in top_failure_reasons}:
            continue
        top_failure_reasons.append(item)
        if len(top_failure_reasons) >= 3:
            break
    if not top_failure_reasons:
        top_failure_reasons = [
            "structure may weaken before confirmation",
            "momentum can fade at entry zone",
            "execution friction can degrade edge",
        ]

    # purpose: surface top blockers with numeric deltas so operators can
    # quickly see what must improve before execution eligibility.
    blocker_candidates: list[dict[str, Any]] = []

    def _add_blocker_metric(name: str, actual: float, target: float, unit: str, detail: str) -> None:
        delta_to_pass = round(max(0.0, float(target) - float(actual)), 3)
        blocker_candidates.append(
            {
                "name": name,
                "actual": round(float(actual), 3),
                "target": round(float(target), 3),
                "delta_to_pass": delta_to_pass,
                "unit": unit,
                "detail": detail,
                "delta_text": f"{delta_to_pass:.3f} {unit} to pass",
            }
        )

    if confluence_score < confluence_threshold:
        _add_blocker_metric(
            "confluence_gap",
            float(confluence_score),
            float(confluence_threshold),
            "score",
            "overall confluence remains below execution threshold",
        )
    if "structure_too_weak" in hard_blockers:
        _add_blocker_metric(
            "structure_quality",
            float(xau_structure_score),
            40.0,
            "score",
            "structure score below minimum quality gate",
        )
    if "spread_liquidity_too_poor" in hard_blockers:
        _add_blocker_metric(
            "spread_liquidity",
            float(spread_safety_score),
            35.0,
            "score",
            "spread/liquidity safety score below minimum",
        )
    if "event_risk_too_high" in hard_blockers:
        _add_blocker_metric(
            "event_risk",
            float(event_risk_score),
            float(event_risk_block_threshold),
            "score",
            "event/news risk score below allowed threshold",
        )
    if "macro_context_conflict_too_strong" in hard_blockers:
        _add_blocker_metric(
            "macro_alignment",
            float(min(dxy_alignment_score, yield_alignment_score)),
            float(macro_conflict_threshold),
            "score",
            "macro alignment score is below conflict tolerance floor",
        )
    if xau_momentum_score < 65:
        _add_blocker_metric(
            "momentum_quality",
            float(xau_momentum_score),
            65.0,
            "score",
            "momentum remains mixed/weak for execution",
        )
    if location_bonus < 0:
        _add_blocker_metric(
            "entry_location",
            float(location_bonus),
            0.0,
            "points",
            "entry location remains suboptimal",
        )
    if "no_clear_setup" in hard_blockers:
        _add_blocker_metric(
            "setup_clarity",
            float(max(buy_score, sell_score)),
            float(decision_threshold),
            "score",
            "setup clarity score below directional decision threshold",
        )

    top_blockers[:] = sorted(
        blocker_candidates,
        key=lambda item: float(item.get("delta_to_pass", 0.0)),
        reverse=True,
    )[:3]
    top_blocker_summary[:] = [
        f"{str(item.get('name', 'blocker'))}: {float(item.get('actual', 0.0)):.3f}/{float(item.get('target', 0.0)):.3f} ({str(item.get('delta_text', '0 to pass'))})"
        for item in top_blockers
    ]

    execution_gate_pass = bool(not hard_blockers and conviction_score >= confluence_threshold)
    setup_radar = build_setup_radar(
        playbook=setup_playbook,
        context={
            **context,
            "spread_state": spread_state,
            "spread_points": spread_points,
            "session": session_state,
            "momentum": momentum_state,
            "location": location_state,
            "trend_strength": trend_strength,
        },
        hard_blockers=hard_blockers,
    )
    timing_engine = {
        "state": str(setup_radar.get("timing_state", "wait") or "wait"),
        "top_watch": dict(setup_radar.get("top_watch", {}) or {}),
        "waiting_for": list(dict(setup_radar.get("top_watch", {}) or {}).get("missing_to_entry", []) or [])[:4],
        "trigger_state": str(dict(setup_radar.get("top_watch", {}) or {}).get("trigger_state", "") or ""),
        "trigger_checklist": list(dict(setup_radar.get("top_watch", {}) or {}).get("trigger_checklist", []) or [])[:10],
        "next_confirmation": str(dict(setup_radar.get("top_watch", {}) or {}).get("next_confirmation", "") or ""),
        "conditions_slowing_setups": list(setup_radar.get("conditions_slowing_setups", []) or [])[:6],
        "summary": str(setup_radar.get("operator_summary", "") or ""),
        "policy_note": "Timing engine is observability only; execution gates remain authoritative.",
    }
    strategy_coverage = _build_strategy_coverage(st, setup_radar)
    decision_protocol = {
        "observe": {
            "xauusd_structure": {
                "trend_state": trend_state or "unclear",
                "trend_structure": trend_structure or "mixed",
                "momentum_state": momentum_state or "mixed",
                "location_state": location_state or "unclear",
            },
            "dxy_direction": dxy_trend or "unknown",
            "us10y_direction": us10y_trend or "unknown",
            "spread_liquidity": {
                "spread_state": spread_state or "unknown",
                "liquidity_state": liquidity_state or "unknown",
                "spread_points": round(spread_points, 2) if spread_points > 0 else 0.0,
                "slippage_state": slippage_state or "unknown",
            },
            "session": session_state or "unknown",
            "event_news_risk": {
                "event_risk_state": event_risk_state or "unknown",
                "news_context_state": news_context_raw or "unknown",
                "high_impact_news_window": in_high_impact_news_window,
            },
            "summary": [
                f"XAU structure={trend_structure or trend_state or 'unclear'} momentum={momentum_state or 'mixed'} location={location_state or 'unclear'}",
                f"DXY={dxy_trend or 'unknown'} US10Y={us10y_trend or 'unknown'}",
                f"spread/liquidity={spread_state or 'unknown'}/{liquidity_state or 'unknown'}",
                f"session={session_state or 'unknown'} event_risk={event_risk_state or 'unknown'}",
            ],
        },
        "orient": {
            "market_regime": str(market_regime.get("regime", "unknown") or "unknown"),
            "regime_confidence": int(market_regime.get("regime_confidence", 0) or 0),
            "setup_type": setup_type if setup_type != "no_trade" else "no_clear_setup",
            "macro_alignment": {
                "dxy_state": _alignment_state(dxy_alignment_score),
                "us10y_state": _alignment_state(yield_alignment_score),
                "conflict": bool(dxy_alignment_score < macro_conflict_threshold and yield_alignment_score < macro_conflict_threshold),
            },
            "reaction_zone_context": str(
                context.get("reaction_zones_state", context.get("order_blocks_state", "unknown")) or "unknown"
            ),
            "breakout_rejection_behavior": str(
                context.get("breakout_rejection_behavior", context.get("breakout_state", "unknown")) or "unknown"
            ),
            "summary": [
                f"regime={str(market_regime.get('regime', 'unknown') or 'unknown')}",
                f"setup={setup_type if setup_type != 'no_trade' else 'no_clear_setup'}",
                f"playbook={dict(setup_playbook.get('top_candidate', {}) or {}).get('name', 'unknown')}",
                f"macro DXY/US10Y={_alignment_state(dxy_alignment_score)}/{_alignment_state(yield_alignment_score)}",
            ],
        },
        "critique": {
            "top_failure_reasons": top_failure_reasons[:3],
            "ranked_blockers": top_blockers,
            "summary": top_failure_reasons[:3],
        },
        "decide": {
            "confluence_score": confluence_score,
            "confluence_threshold": confluence_threshold,
            "score_passed": bool(confluence_score >= confluence_threshold),
            "blockers": hard_blockers[:8],
            "blockers_override_score": bool(hard_blockers),
            "recommended_action": recommended_action,
            "summary": [
                f"confluence={confluence_score}/{confluence_threshold}",
                f"blockers={'none' if not hard_blockers else ', '.join(hard_blockers[:3])}",
                f"action={recommended_action}",
                f"top_blockers={'none' if not top_blocker_summary else '; '.join(top_blocker_summary[:2])}",
            ],
        },
        "act": {
            "execution_request_eligible": execution_gate_pass,
            "requires_execution_gates": True,
            "gate_conditions": {
                "score_passed": bool(conviction_score >= confluence_threshold),
                "no_hard_blockers": bool(not hard_blockers),
                "existing_execution_gates_approve": bool(execution_gate_pass),
            },
            "summary": (
                "eligible_for_guarded_execution_request"
                if execution_gate_pass
                else "not_eligible_for_execution_request"
            ),
        },
    }

    return {
        "market_bias": market_bias,
        "signal_bias": signal_bias,
        "chart_structure_bias": chart_structure_bias,
        "higher_timeframe_bias": higher_timeframe_bias,
        "final_trade_bias": final_trade_bias,
        "bias_breakdown": bias_breakdown,
        "confidence_score": conviction_score,
        "confidence_level": conviction_level,
        "confidence_basis": "atlas_conviction",
        "signal_confidence_seed_score": base_score,
        "conviction_adjustments": conviction_adjustments,
        "trend": {"state": trend_state or "unclear", "structure": trend_structure},
        "structure_context": {
            "trend_direction": "bullish" if trend_state == "uptrend" else ("bearish" if trend_state == "downtrend" else "unclear"),
            "reaction_zones": str(context.get("reaction_zones_state", context.get("order_blocks_state", "unknown")) or "unknown"),
            "breakout_or_rejection_behavior": str(context.get("breakout_rejection_behavior", context.get("breakout_state", "unknown")) or "unknown"),
        },
        "momentum": {"state": momentum_state or "mixed"},
        "location": {"state": location_state or "unclear"},
        "setup_type": setup_type if setup_type in {"pullback", "breakout", "consolidation", "continuation", "mean_reversion"} else "no_trade",
        "entry_quality": entry_quality,
        "reasoning": reasoning[:6] or ["conditions are mixed and no high-quality setup is confirmed"],
        "warnings": list(dict.fromkeys(warnings))[:8],
        "blockers": hard_blockers[:8],
        "hard_blockers": hard_blockers[:8],
        "top_blockers": top_blockers,
        "confluence_score": confluence_score,
        "confluence_threshold": confluence_threshold,
        "confluence": {
            "score": confluence_score,
            "threshold": confluence_threshold,
            "qualified": bool(not hard_blockers and confluence_score >= confluence_threshold),
        },
        "feature_scores": {
            key: {"score": value, "level": _feature_level(value)}
            for key, value in features.items()
        },
        "macro_context": macro_context,
        "market_regime": market_regime,
        "reversal_detection": reversal_detection,
        "context_alignment": context_alignment,
        "recommended_action": recommended_action,
        "safety_note": trading_response["safety_note"],
        "timeframe_logic": trading_response["timeframe_logic"],
        "trading_response": trading_response,
        "setup_playbook": setup_playbook,
        "setup_radar": setup_radar,
        "timing_engine": timing_engine,
        "strategy_coverage": strategy_coverage,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "no_trade_score": no_trade_score,
        "trade_decision": trade_decision,
        "bias_strength": trading_response.get("bias_strength", "weak"),
        "pre_trade_memo": pre_trade_memo,
        "decision_protocol": decision_protocol,
        "explanation": explanation,
        "account_profile": {
            "account_tier": account_tier,
            "tier_reason": str(account_profile.get("tier_reason", "")),
            "risk_profile": str(account_profile.get("risk_profile", "capital_preservation_standard")),
            "current_size_tier": str(size_tier.get("current_size_tier", "")),
            "next_size_tier_requirement": str(size_tier.get("next_size_tier_requirement", "")),
            "size_tier_reason": str(size_tier.get("size_tier_reason", "")),
        },
    }
