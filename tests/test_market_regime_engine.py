from __future__ import annotations

from atlas_ai.analysis.market_regime import build_market_regime_and_reversal


def _status(**kwargs):
    base = {
        "setup_family": "no_trade",
        "context_snapshot": {},
    }
    base.update(kwargs)
    return base


def test_clean_trending_regime_classification() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            setup_family="pullback",
            context_snapshot={
                "trend_structure": "HH/HL",
                "trend_state": "uptrend",
                "momentum": "strong",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "volatility_state": "normal",
                "session": "london",
            },
        ),
        runtime={},
        chart_input={"candle_expansion": 0.35, "wickiness": 0.22, "chop_score": 0.18},
    )
    regime = out["market_regime"]
    assert regime["regime"] == "trending"
    assert regime["regime_confidence"] >= 45


def test_clear_structure_with_mixed_momentum_is_not_unknown_regime() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            setup_family="continuation",
            context_snapshot={
                "trend_structure": "HH/HL",
                "trend_state": "uptrend",
                "momentum": "mixed",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "volatility_state": "normal",
                "session": "new_york",
            },
        ),
        runtime={},
        chart_input={"candle_expansion": 0.12, "wickiness": 0.42, "chop_score": 0.44},
    )
    regime = out["market_regime"]
    assert regime["regime"] == "trending"
    assert regime["regime_confidence"] >= 45
    assert any("mixed" in reason for reason in regime["regime_reasons"])


def test_clean_ranging_regime_classification() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            setup_family="mean_reversion",
            context_snapshot={
                "trend_state": "range",
                "momentum": "weak",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "volatility_state": "low",
            },
        ),
        runtime={},
        chart_input={
            "compression_score": 0.75,
            "wickiness": 0.68,
            "chop_score": 0.72,
            "repeated_rejection_at_extremes": True,
        },
    )
    regime = out["market_regime"]
    assert regime["regime"] == "ranging"
    assert regime["regime_confidence"] >= 45


def test_volatile_regime_classification() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            setup_family="breakout",
            context_snapshot={
                "momentum": "mixed",
                "spread_state": "wide",
                "liquidity_state": "poor",
                "volatility_state": "high",
                "session": "late_us",
            },
        ),
        runtime={},
        chart_input={"candle_expansion": 0.82, "wickiness": 0.74, "atr_ratio": 0.09},
    )
    regime = out["market_regime"]
    assert regime["regime"] == "volatile"
    assert regime["regime_confidence"] >= 55


def test_divergence_without_confirmation_does_not_confirm_reversal() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            context_snapshot={
                "dxy_alignment": "conflict",
                "us10y_alignment": "aligned",
            }
        ),
        runtime={},
        chart_input={
            "new_local_high": True,
            "exhaustion_wick": True,
            "structure_break_confirmed": False,
            "momentum_confirmation": False,
        },
    )
    reversal = out["reversal_detection"]
    assert reversal["reversal_candidate"] is True
    assert reversal["confirmed_reversal"] is False


def test_divergence_plus_structure_and_momentum_can_confirm_reversal() -> None:
    out = build_market_regime_and_reversal(
        status=_status(
            context_snapshot={
                "dxy_alignment": "conflict",
                "us10y_alignment": "conflict",
            }
        ),
        runtime={},
        chart_input={
            "new_local_low": True,
            "exhaustion_wick": True,
            "structure_break_confirmed": True,
            "momentum_confirmation": True,
        },
    )
    reversal = out["reversal_detection"]
    assert reversal["reversal_candidate"] is True
    assert reversal["confirmed_reversal"] is True
    assert reversal["reversal_bias"] == "buy"
