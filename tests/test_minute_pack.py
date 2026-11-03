import pandas as pd

from atlas_ai.strategies.minute_pack import (
    GoldLiquiditySweepStrategy,
    GoldMicroImpulseStrategy,
    GoldMicroStructureReversionStrategy,
)


def _base_df(rows: int = 40) -> pd.DataFrame:
    data = []
    price = 3300.0
    for i in range(rows):
        o = price + (i * 0.08)
        c = o + 0.05
        h = c + 0.04
        l = o - 0.04
        data.append(
            {
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "tick_volume": 100 + i,
                "real_volume": 100 + i,
            }
        )
    return pd.DataFrame(data)


def test_gold_micro_impulse_detects_breakout_buy():
    df = _base_df()
    df.loc[df.index[-1], ["open", "low", "high", "close", "tick_volume"]] = [3304.0, 3303.95, 3305.0, 3304.95, 220]
    signal = GoldMicroImpulseStrategy().generate_signal(df)
    assert signal.side == "buy"
    assert signal.confidence > 0


def test_gold_liquidity_sweep_detects_sell_rejection():
    df = _base_df()
    prior_high = float(df["high"].iloc[-9:-1].max())
    df.loc[df.index[-1], ["open", "low", "high", "close"]] = [prior_high - 0.05, prior_high - 0.06, prior_high + 0.35, prior_high - 0.02]
    signal = GoldLiquiditySweepStrategy().generate_signal(df)
    assert signal.side == "sell"
    assert signal.confidence > 0


def test_gold_micro_structure_reversion_detects_buy_reclaim():
    df = _base_df()
    for i in range(len(df)):
        if i < 32:
            base = 3305.0 + ((i % 5) - 2) * 0.15
            df.loc[df.index[i], ["open", "high", "low", "close", "tick_volume"]] = [base, base + 0.25, base - 0.25, base + 0.03, 120]
        else:
            base = 3305.0 + ((i % 3) - 1) * 0.01
            df.loc[df.index[i], ["open", "high", "low", "close", "tick_volume"]] = [base, base + 0.04, base - 0.04, base + 0.005, 105]
    df.loc[df.index[-2], ["open", "high", "low", "close"]] = [3304.96, 3304.98, 3304.50, 3304.55]
    df.loc[df.index[-1], ["open", "high", "low", "close"]] = [3304.58, 3305.28, 3304.56, 3305.20]
    signal = GoldMicroStructureReversionStrategy().generate_signal(df)
    assert signal.side == "buy"
    assert signal.confidence > 0
