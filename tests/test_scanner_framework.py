from __future__ import annotations

from datetime import datetime, timezone

from atlas_ai.decision_framework import DecisionFramework, DecisionInput


def make_bullish_input(**kwargs) -> DecisionInput:
    base = dict(
        higher_highs=True,
        higher_lows=True,
        bullish_candle_strength=0.75,
        near_support=True,
        dxy_direction="falling",
        us10y_direction="falling",
        session="london",
        session_tradeable=True,
        spread_pips=1.5,
        adx=30.0,
        in_news_window=False,
        open_trades_count=0,
        daily_trades_count=0,
        daily_drawdown_pct=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    base.update(kwargs)
    return DecisionInput(**base)


def make_bearish_input(**kwargs) -> DecisionInput:
    base = dict(
        lower_highs=True,
        lower_lows=True,
        bearish_candle_strength=0.75,
        near_resistance=True,
        dxy_direction="rising",
        us10y_direction="rising",
        session="london",
        session_tradeable=True,
        spread_pips=1.5,
        adx=30.0,
        in_news_window=False,
        open_trades_count=0,
        daily_trades_count=0,
        daily_drawdown_pct=0.0,
        timestamp=datetime.now(timezone.utc),
    )
    base.update(kwargs)
    return DecisionInput(**base)


fw = DecisionFramework()


def test_clear_bullish_setup():
    out = fw.evaluate(make_bullish_input())
    assert out.trade_decision == "BUY_CANDIDATE"
    assert out.buy_score > out.sell_score


def test_clear_bearish_setup():
    out = fw.evaluate(make_bearish_input())
    assert out.trade_decision == "SELL_CANDIDATE"
    assert out.sell_score > out.buy_score


def test_mixed_setup_no_trade():
    inp = DecisionInput(
        higher_highs=True,
        lower_lows=True,
        bullish_candle_strength=0.40,
        bearish_candle_strength=0.40,
        in_middle_of_range=True,
        dxy_direction="flat",
        session="london",
        session_tradeable=True,
        spread_pips=1.5,
        adx=18.0,
        timestamp=datetime.now(timezone.utc),
    )
    out = fw.evaluate(inp)
    assert out.trade_decision == "NO_TRADE"


def test_blocked_session_no_execution_request():
    out = fw.evaluate(make_bullish_input(session="asian", session_tradeable=False))
    assert out.can_create_execution_request is False
    assert out.trade_decision == "NO_TRADE"


def test_news_window_blocks():
    out = fw.evaluate(make_bullish_input(in_news_window=True))
    assert out.trade_decision == "NO_TRADE"


def test_spread_too_wide_blocks():
    out = fw.evaluate(make_bullish_input(spread_pips=9.0))
    assert out.trade_decision == "NO_TRADE"


def test_max_trades_blocks():
    out = fw.evaluate(make_bullish_input(open_trades_count=3))
    assert out.trade_decision == "NO_TRADE"


def test_decision_always_has_reason():
    for inp in [make_bullish_input(), make_bearish_input(), make_bullish_input(session="asian", session_tradeable=False)]:
        out = fw.evaluate(inp)
        assert out.reason
        assert out.trade_decision in ["BUY_CANDIDATE", "SELL_CANDIDATE", "NO_TRADE"]


def test_dxy_conflict_reduces_confidence():
    conflict = fw.evaluate(make_bullish_input(dxy_direction="rising"))
    normal = fw.evaluate(make_bullish_input(dxy_direction="falling"))
    assert conflict.confidence < normal.confidence

