from types import SimpleNamespace

import pytest

from atlas_ai.models import TradeSignal
from atlas_ai.trader import AtlasTrader


@pytest.mark.parametrize("location,passed", [("breakout", False), ("retest", True), ("pullback", True)])
def test_current_timing_policy_distinguishes_breakout_from_retest(location, passed):
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(timing_allow_mid_range_strong_momentum=False)
    result = trader._timing_discipline_snapshot(
        TradeSignal("gold_breakout_retest", "buy", 0.90, 100, 145),
        {"entry_location_state": location, "entry_momentum_state": "strong",
         "trend_state": "uptrend", "trend_strength": 0.3, "session": "asia", "setup_type_hint": "breakout"},
        {},
    )
    assert result["passed"] is passed
    if not passed:
        assert result["missing"] == ["fresh_entry_or_retest"]
        assert result["reason_code"] == "AWAITING_RETEST_CONFIRMATION"


@pytest.mark.parametrize("confidence,approved", [(0.44, False), (0.54, False), (0.55, True), (0.59, True)])
def test_auto_approval_has_independent_confidence_floor(confidence, approved):
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(symbol="XAUUSD")
    result = trader._guarded_demo_auto_approval_decision(
        environment="demo", setup_family="continuation", volume=0.01, runtime_flags={},
        status={"chosen": {"side": "buy", "confidence": confidence}, "min_confidence_threshold": 0.44},
    )
    assert result["approved"] is approved
    assert result["min_confidence"] == 0.55
    assert result["reasons"] == ([] if approved else ["confidence_below_auto_approval_floor"])
