from __future__ import annotations

from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis


def _status(**kwargs):
    base = {
        "action": "hold",
        "chosen": {"confidence": 0.5, "side": "flat", "strategy": "none"},
        "context_snapshot": {},
        "setup_family": "no_trade",
    }
    base.update(kwargs)
    return base


def test_uptrend_structure_gives_buy_bias_candidate() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.82, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback"}, setup_family="pullback"),
        runtime={},
    )
    assert out["market_bias"] == "buy"
    assert out["confidence_score"] >= 70


def test_downtrend_structure_gives_sell_bias_candidate() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="sell", chosen={"confidence": 0.8, "side": "sell"}, context_snapshot={"trend_structure": "LH/LL", "momentum": "strong", "location": "pullback"}, setup_family="pullback"),
        runtime={},
    )
    assert out["market_bias"] == "sell"
    assert out["confidence_score"] >= 70


def test_timeframe_logic_uses_live_status_not_hardcoded_m15() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            timeframe="M1",
            higher_timeframe={"confirm_tf": "M5", "bias": "buy"},
            action="buy",
            chosen={"confidence": 0.82, "side": "buy"},
            context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback"},
            setup_family="pullback",
        ),
        runtime={},
    )
    logic = out["trading_response"]["timeframe_logic"]
    assert logic["structure_trend"] == "M1"
    assert logic["entry_confirmation"] == "M1"
    assert logic["higher_timeframe_confirmation"] == "M5"
    assert logic["visible_terminal_chart_required"] is False
    assert any("M1 structure is clear" == reason for reason in out["reasoning"])
    assert not any("M15" in reason for reason in out["reasoning"])


def test_structure_against_signal_is_contradictory_not_supportive() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            timeframe="M1",
            action="buy",
            chosen={"confidence": 0.82, "side": "buy"},
            context_snapshot={"trend_structure": "LH/LL", "momentum": "mixed", "location": "mid_range"},
            setup_family="continuation",
        ),
        runtime={},
    )
    assert out["feature_scores"]["xau_structure_score"]["score"] < 50
    assert out["signal_bias"] == "buy"
    assert out["chart_structure_bias"] == "sell"
    assert out["final_trade_bias"] == "neutral"
    assert out["bias_breakdown"]["conflict"] is True
    assert "structure_against_signal" in out["warnings"]
    assert any("bearish while the signal is buy" in item for item in out["pre_trade_memo"]["contradictory_factors"])
    assert not any("lower highs and lower lows on M1" == item for item in out["pre_trade_memo"]["supporting_factors"])


def test_consolidation_range_is_neutral_or_avoid() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="hold", chosen={"confidence": 0.35, "side": "flat"}, context_snapshot={"trend_state": "range", "momentum": "weak", "location": "consolidation"}, setup_family="consolidation"),
        runtime={},
    )
    assert out["market_bias"] == "neutral"
    assert out["entry_quality"] == "avoid"


def test_pullback_classification_works() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.7, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback"}, setup_family="pullback"),
        runtime={},
    )
    assert out["location"]["state"] == "pullback"
    assert out["setup_type"] == "pullback"


def test_breakout_classification_works() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.72, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "breakout"}, setup_family="breakout"),
        runtime={},
    )
    assert out["location"]["state"] == "breakout"
    assert out["setup_type"] == "breakout"


def test_weak_momentum_lowers_entry_quality() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.8, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "weak", "location": "pullback"}, setup_family="pullback"),
        runtime={},
    )
    assert out["entry_quality"] in {"risky", "avoid"}
    assert "weak_momentum" in out["warnings"]


def test_conflicting_conditions_produce_neutral_or_avoid() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.62, "side": "buy"}, context_snapshot={"trend_state": "range", "momentum": "mixed", "location": "mid_range"}, setup_family="breakout"),
        runtime={},
    )
    assert out["market_bias"] == "neutral" or out["entry_quality"] == "avoid"


def test_confluence_engine_returns_feature_scores_and_memo() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.91, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "low",
                "news_sentiment": "supportive",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["confluence"]["score"] >= 85
    assert "xau_structure_score" in out["feature_scores"]
    assert out["pre_trade_memo"]["final_decision"] in {"guarded_buy", "wait_for_m1_confirmation_buy"}
    assert "reward_vs_risk" in out["trading_response"]


def test_hard_blockers_override_high_confluence() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.96, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "wide",
                "liquidity_state": "poor",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85, "max_spread_points": 30},
    )
    assert out["recommended_action"] == "hold"
    assert "spread_liquidity_too_poor" in out["hard_blockers"]
    assert out["setup_radar"]["mode"] == "observability_only"
    assert out["timing_engine"]["state"] in {"setup_forming", "conditions_unfavorable"}
    assert "trigger_state" in out["timing_engine"]
    assert isinstance(out["timing_engine"]["trigger_checklist"], list)
    assert out["timing_engine"]["next_confirmation"]
    assert "Timing engine is observability only" in out["timing_engine"]["policy_note"]


def test_strategy_coverage_flags_ready_playbook_without_matching_signal() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="hold",
            chosen={"confidence": 0.45, "side": "flat", "strategy": "momentum_continuation"},
            setup_family="mean_reversion",
            context_snapshot={
                "trend_state": "range",
                "trend_structure": "mixed",
                "momentum": "mixed",
                "location": "pullback",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "new_york",
            },
            signals=[
                {"strategy": "momentum_continuation", "side": "flat", "confidence": 0.0},
                {"strategy": "ema_crossover", "side": "flat", "confidence": 0.0},
            ],
        ),
        runtime={},
    )

    coverage = out["strategy_coverage"]
    assert coverage["mode"] == "observability_only"
    assert coverage["top_playbook"] == "range_edge_reversal"
    assert coverage["coverage_gap"] is True
    assert "does not enable strategies" in coverage["policy_note"]


def test_macro_conflict_and_event_risk_reduce_confidence_and_add_blockers() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.88, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "breakout",
                "dxy_alignment": "conflict",
                "us10y_alignment": "conflict",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "session": "new_york",
                "event_risk_state": "high",
                "news_sentiment": "mixed",
            },
            setup_family="breakout",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["confluence"]["score"] < 85
    assert "macro_context_conflict_too_strong" in out["hard_blockers"]
    assert "event_risk_too_high" in out["hard_blockers"]


def test_vix_and_oil_context_are_exposed_without_becoming_standalone_triggers() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.86, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "vix_regime": "elevated",
                "vix_alignment": "conflict",
                "oil_backdrop": "inflationary",
                "oil_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "watch",
                "news_sentiment": "mixed",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert "vix_alignment_score" in out["feature_scores"]
    assert "oil_context_score" in out["feature_scores"]
    assert out["context_alignment"]["vix"]["regime"] == "elevated"
    assert out["macro_context"]["oil"]["backdrop"] == "inflationary"
    assert out["trading_response"]["size_band_rationale"]["band"] in {"guarded", "probe", "blocked", "full_guarded"}


def test_high_slippage_adds_execution_friction_blocker() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.94, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "slippage_state": "high",
                "slippage_points": 24.0,
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85, "max_slippage_points": 20},
    )
    assert "slippage_risk_too_high" in out["hard_blockers"]
    assert out["context_alignment"]["spread_liquidity"]["slippage_state"] == "high"
    assert out["trading_response"]["recommended_action"] == "hold"


def test_size_band_rationale_stays_bounded_under_macro_caution() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.88, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "vix_regime": "stress",
                "vix_alignment": "conflict",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "slippage_state": "elevated",
                "session": "new_york",
                "event_risk_state": "watch",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85, "max_slippage_points": 20},
    )
    size_band = out["trading_response"]["size_band_rationale"]
    assert size_band["cap_multiplier"] <= 0.65
    assert size_band["risk_fraction_of_max"] <= 0.65
    assert size_band["rationale"]


def test_volatile_regime_blocks_entries_defensively() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.95, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "breakout",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "wide",
                "liquidity_state": "poor",
                "volatility_state": "high",
                "session": "late_us",
            },
            setup_family="breakout",
            chart_features={
                "candle_expansion": 0.84,
                "wickiness": 0.74,
                "atr_ratio": 0.09,
            },
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["market_regime"]["regime"] == "volatile"
    assert "volatile_regime_defensive_block" in out["hard_blockers"]
    assert out["recommended_action"] == "hold"


def test_small_balance_enters_micro_account_mode() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.9, "side": "buy", "take_profit_points": 120.0},
            risk={"balance": 1200.0, "daily_loss_pct": 0.001, "trades_today": 0},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "spread_state": "tight",
                "spread_points": 18.0,
                "session": "london",
            },
            setup_family="continuation",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["account_profile"]["account_tier"] == "micro_account"
    assert out["account_profile"]["risk_profile"] == "capital_preservation_strict"


def test_micro_account_mode_raises_trade_threshold() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.9, "side": "buy"},
            risk={"balance": 1000.0},
            context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback"},
            setup_family="continuation",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["confluence_threshold"] >= 95


def test_poor_spread_blocks_micro_account_trade() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.94, "side": "buy", "take_profit_points": 60.0},
            risk={"balance": 900.0},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "spread_state": "wide",
                "spread_points": 28.0,
            },
            setup_family="continuation",
        ),
        runtime={"max_spread_points": 30},
    )
    assert "spread_liquidity_too_poor" in out["hard_blockers"] or "spread_cost_too_high_for_expected_move" in out["hard_blockers"]


def test_reversal_entries_are_restricted_for_micro_account() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.96, "side": "buy"},
            risk={"balance": 1100.0},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "conflict",
                "us10y_alignment": "conflict",
                "new_local_high": True,
                "exhaustion_wick": True,
                "reversal_structure_break": False,
                "reversal_momentum_confirmation": False,
            },
            setup_family="mean_reversion",
            chart_features={
                "new_local_high": True,
                "exhaustion_wick": True,
                "reversal_structure_break": False,
                "reversal_momentum_confirmation": False,
            },
        ),
        runtime={},
    )
    assert "reversal_restricted_for_micro_account" in out["hard_blockers"] or "reversal_confirmation_not_strong_enough_for_micro_account" in out["hard_blockers"]


def test_xauusd_only_scope_blocks_other_symbols() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            symbol="EURUSD",
            action="buy",
            chosen={"confidence": 0.95, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "event_risk_state": "low",
                "session": "london",
            },
            setup_family="pullback",
        ),
        runtime={},
    )
    assert "xauusd_only_scope" in out["hard_blockers"]
    assert out["recommended_action"] == "hold"


def test_high_impact_news_window_blocks_trade() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.97, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "event_risk_state": "low",
                "in_high_impact_news_window": True,
                "session": "london",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert "high_impact_news_window" in out["hard_blockers"]
    assert out["recommended_action"] == "hold"


def test_decision_protocol_generated_for_valid_setup() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            symbol="XAUUSD",
            action="buy",
            chosen={"confidence": 0.94, "side": "buy", "take_profit_points": 120.0, "stop_loss_points": 60.0},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    protocol = dict(out.get("decision_protocol", {}) or {})
    assert {"observe", "orient", "critique", "decide", "act"}.issubset(protocol.keys())
    assert isinstance(protocol["observe"].get("summary"), list)
    assert isinstance(protocol["orient"].get("summary"), list)
    assert isinstance(protocol["critique"].get("top_failure_reasons"), list)
    assert "reward_vs_risk" in out["trading_response"]


def test_critique_step_includes_top_failure_reasons() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            symbol="XAUUSD",
            action="buy",
            chosen={"confidence": 0.92, "side": "buy"},
            context_snapshot={
                "trend_state": "range",
                "momentum": "weak",
                "location": "mid_range",
                "dxy_alignment": "conflict",
                "us10y_alignment": "conflict",
                "spread_state": "wide",
                "liquidity_state": "poor",
                "event_risk_state": "high",
            },
            setup_family="breakout",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    reasons = list((((out.get("decision_protocol", {}) or {}).get("critique", {}) or {}).get("top_failure_reasons", []) or []))
    assert reasons
    assert len(reasons) <= 3


def test_score_above_threshold_still_blocked_by_hard_blocker_in_protocol() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            symbol="XAUUSD",
            action="buy",
            chosen={"confidence": 0.99, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "wide",
                "liquidity_state": "poor",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 70},
    )
    decide = dict(((out.get("decision_protocol", {}) or {}).get("decide", {}) or {}))
    act = dict(((out.get("decision_protocol", {}) or {}).get("act", {}) or {}))
    assert decide.get("confluence_score", 0) < decide.get("confluence_threshold", 999)
    assert decide.get("blockers_override_score") is True
    assert act.get("execution_request_eligible") is False
    assert out["recommended_action"] == "hold"


def test_confidence_is_reported_as_atlas_conviction() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="buy",
            chosen={"confidence": 0.82, "side": "buy", "signal_confidence_seed": 0.82},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
            same_side_support={"side": "buy", "count": 2},
        ),
        runtime={"confluence_trade_threshold": 80},
    )
    assert out["confidence_basis"] == "atlas_conviction"
    assert out["signal_confidence_seed_score"] == 82
    assert any(str(item.get("label", "")) == "same_side_support" for item in out["conviction_adjustments"])


def test_decision_protocol_does_not_require_internal_monologue_mode() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="hold",
            chosen={"confidence": 0.3, "side": "flat"},
            context_snapshot={"trend_state": "unclear", "momentum": "mixed"},
            setup_family="no_trade",
        ),
        runtime={},
    )
    protocol_text = str(out.get("decision_protocol", "")).lower()
    assert "internal monologue" not in protocol_text
    assert "chain-of-thought" not in protocol_text


def test_learning_mode_derives_analysis_threshold_without_runtime_override() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="sell",
            learning_mode=True,
            mode_label="LEARNING_AGGRESSIVE_DEMO",
            thresholds={"confidence": 0.44},
            chosen={"confidence": 0.56, "side": "sell"},
            context_snapshot={
                "trend_structure": "LH/LL",
                "momentum": "mixed",
                "location": "pullback",
                "session": "new_york",
            },
            setup_family="continuation",
        ),
        runtime={},
    )
    assert 50 <= int(out["confluence_threshold"]) <= 75


def test_size_band_rationale_for_hold_without_blockers_is_quality_threshold() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="hold",
            chosen={"confidence": 0.42, "side": "buy"},
            context_snapshot={
                "trend_structure": "HH/HL",
                "momentum": "mixed",
                "location": "mid_range",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "new_york",
                "event_risk_state": "low",
            },
            setup_family="pullback",
        ),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["hard_blockers"] == []
    assert out["recommended_action"] == "hold"
    rationale = str(out["trading_response"]["size_band_rationale"]["rationale"][0]).lower()
    assert "confluence is below the active execution-quality threshold" in rationale


def test_recommended_action_and_trade_decision_stay_coherent() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="sell",
            chosen={"confidence": 0.67, "side": "sell"},
            context_snapshot={
                "trend_structure": "LH/LL",
                "momentum": "mixed",
                "location": "pullback",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "new_york",
                "event_risk_state": "low",
            },
            setup_family="continuation",
            learning_mode=True,
            mode_label="LEARNING_AGGRESSIVE_DEMO",
            thresholds={"confidence": 0.44},
        ),
        runtime={},
    )
    decision = str(out["trading_response"]["trade_decision"])
    action = str(out["recommended_action"])
    if decision == "BUY":
        assert action in {"guarded_buy", "wait_for_m1_confirmation_buy"}
    elif decision == "SELL":
        assert action in {"guarded_sell", "wait_for_m1_confirmation_sell"}
    else:
        assert action == "hold"


def test_top_blockers_include_numeric_deltas() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="hold",
            chosen={"confidence": 0.31, "side": "flat"},
            context_snapshot={
                "trend_state": "range",
                "momentum": "weak",
                "location": "mid_range",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "new_york",
            },
            setup_family="no_trade",
            learning_mode=True,
            mode_label="LEARNING_AGGRESSIVE_DEMO",
            thresholds={"confidence": 0.44},
        ),
        runtime={},
    )
    top = list(out.get("top_blockers", []) or [])
    assert top
    first = dict(top[0])
    assert "delta_to_pass" in first
    assert "delta_text" in first


def test_range_edge_reversal_is_not_blocked_by_trend_structure_rule() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="sell",
            chosen={"confidence": 0.62, "side": "sell"},
            context_snapshot={
                "trend_state": "range",
                "trend_structure": "mixed",
                "momentum": "mixed",
                "location": "consolidation",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="consolidation",
            learning_mode=True,
            mode_label="LEARNING_AGGRESSIVE_DEMO",
            thresholds={"confidence": 0.44},
        ),
        runtime={},
    )

    assert out["setup_playbook"]["top_candidate"]["name"] == "range_edge_reversal"
    assert "structure_too_weak" not in out["hard_blockers"]
    assert "no_clear_setup" not in out["hard_blockers"]
    assert "range_structure_context" in out["warnings"]


def test_weak_structure_still_blocks_pullback_continuation() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="sell",
            chosen={"confidence": 0.62, "side": "sell"},
            context_snapshot={
                "trend_state": "range",
                "trend_structure": "mixed",
                "momentum": "mixed",
                "location": "consolidation",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "session": "london",
                "event_risk_state": "low",
            },
            setup_family="pullback",
            learning_mode=True,
            mode_label="LEARNING_AGGRESSIVE_DEMO",
            thresholds={"confidence": 0.44},
        ),
        runtime={},
    )

    assert "structure_too_weak" in out["hard_blockers"]
