from __future__ import annotations

from atlas_ai.trader import AtlasTrader


def test_size_tier_scales_stepwise_without_aggressive_jump() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    low = trader._size_tier_profile(700.0)
    mid = trader._size_tier_profile(2600.0)
    high = trader._size_tier_profile(8000.0)
    assert low["current_size_tier"] == "tier_0_0p01"
    assert mid["current_size_tier"] in {"tier_1_0p02", "tier_2_0p03"}
    assert high["current_size_tier"] in {"tier_2_0p03", "tier_3_0p05"}
    assert float(high["max_lot"]) <= 0.05


def test_daily_profit_target_no_longer_forces_pause_message() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    source = AtlasTrader.step.__code__.co_consts
    assert all("Trading paused for the day." not in str(item) for item in source)
