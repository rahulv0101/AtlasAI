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


def test_clear_bullish_structure_returns_buy() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.9, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback", "dxy_alignment": "aligned", "us10y_alignment": "aligned", "spread_state": "tight", "liquidity_state": "normal", "event_risk_state": "low", "session": "london"}, setup_family="pullback"),
        runtime={"confluence_trade_threshold": 80},
    )
    assert out["trading_response"]["trade_decision"] == "BUY"


def test_clear_bearish_structure_returns_sell() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="sell", chosen={"confidence": 0.9, "side": "sell"}, context_snapshot={"trend_structure": "LH/LL", "momentum": "strong", "location": "pullback", "dxy_alignment": "aligned", "us10y_alignment": "aligned", "spread_state": "tight", "liquidity_state": "normal", "event_risk_state": "low", "session": "london"}, setup_family="pullback"),
        runtime={"confluence_trade_threshold": 80},
    )
    assert out["trading_response"]["trade_decision"] == "SELL"


def test_unclear_structure_returns_no_trade() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="hold", chosen={"confidence": 0.45, "side": "flat"}, context_snapshot={"trend_state": "range", "momentum": "mixed", "location": "mid_range"}, setup_family="consolidation"),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["trading_response"]["trade_decision"] == "NO_TRADE"


def test_weak_bearish_bias_does_not_auto_sell() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="sell", chosen={"confidence": 0.52, "side": "sell"}, context_snapshot={"trend_structure": "LH/LL", "momentum": "weak", "location": "mid_range"}, setup_family="no_trade"),
        runtime={"confluence_trade_threshold": 85},
    )
    assert out["trading_response"]["trade_decision"] == "NO_TRADE"


def test_macro_conflict_lowers_confidence() -> None:
    aligned = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.86, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "breakout", "dxy_alignment": "aligned", "us10y_alignment": "aligned"}, setup_family="breakout"),
        runtime={"confluence_trade_threshold": 80},
    )
    conflict = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.86, "side": "buy"}, context_snapshot={"trend_structure": "HH/HL", "momentum": "strong", "location": "breakout", "dxy_alignment": "conflict", "us10y_alignment": "conflict"}, setup_family="breakout"),
        runtime={"confluence_trade_threshold": 80},
    )
    assert conflict["confidence_score"] < aligned["confidence_score"]


def test_weak_confluence_prefers_no_trade() -> None:
    out = build_structured_decision_analysis(
        status=_status(action="buy", chosen={"confidence": 0.55, "side": "buy"}, context_snapshot={"trend_state": "unclear", "momentum": "mixed", "location": "unclear"}, setup_family="no_trade"),
        runtime={"confluence_trade_threshold": 85},
    )
    tr = out["trading_response"]
    assert tr["trade_decision"] == "NO_TRADE"
    assert tr["no_trade_score"] >= tr["buy_score"] or tr["no_trade_score"] >= tr["sell_score"]


def test_ranging_regime_prefers_range_aware_setup_hint() -> None:
    out = build_structured_decision_analysis(
        status=_status(
            action="hold",
            chosen={"confidence": 0.58, "side": "buy"},
            context_snapshot={"trend_state": "range", "momentum": "mixed", "location": "pullback", "session": "new_york"},
            setup_family="continuation",
            chart_features={"setup_type_hint": "mean_reversion"},
        ),
        runtime={"confluence_trade_threshold": 70},
    )
    assert out["setup_type"] in {"mean_reversion", "consolidation", "continuation"}
