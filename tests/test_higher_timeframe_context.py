from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from atlas_ai.config.settings import load_settings
from atlas_ai.trader import AtlasTrader


class _RecordingFeed:
    def __init__(self) -> None:
        self.requested: list[str] = []

    def get_rates_for_timeframe(self, timeframe: str, bars: int = 220):
        self.requested.append(timeframe)
        closes = [2000.0 + idx for idx in range(90)]
        return pd.DataFrame({"close": closes})


def _trader_for_timeframe(timeframe: str) -> AtlasTrader:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe=timeframe,
        m1_htf_alignment_timeframe="M5",
        higher_timeframe_confirm="H1",
        higher_timeframe_context=("H1", "H4"),
    )
    trader.feed = _RecordingFeed()
    trader._timeframe_bias = AtlasTrader._timeframe_bias.__get__(trader, AtlasTrader)
    trader._higher_tf_context = AtlasTrader._higher_tf_context.__get__(trader, AtlasTrader)
    return trader


def test_m1_higher_tf_context_uses_dedicated_m5_alignment_setting() -> None:
    trader = _trader_for_timeframe("M1")

    context = trader._higher_tf_context()

    assert context["confirm_tf"] == "M5"
    assert context["context_tfs"] == ["M5"]
    assert trader.feed.requested == ["M5"]


def test_m15_higher_tf_context_keeps_shared_h1_h4_context_settings() -> None:
    trader = _trader_for_timeframe("M15")

    context = trader._higher_tf_context()

    assert context["confirm_tf"] == "H1"
    assert context["context_tfs"] == ["H1", "H4"]
    assert trader.feed.requested == ["H1", "H4"]


def test_m1_htf_alignment_timeframe_defaults_to_pre_migration_m5(monkeypatch) -> None:
    monkeypatch.delenv("M1_HTF_ALIGNMENT_TIMEFRAME", raising=False)
    monkeypatch.delenv("HIGHER_TIMEFRAME_CONFIRM", raising=False)
    monkeypatch.delenv("HIGHER_TIMEFRAME_CONTEXT", raising=False)

    settings = load_settings()

    assert settings.m1_htf_alignment_timeframe == "M5"
    assert settings.higher_timeframe_confirm == "H1"
    assert settings.higher_timeframe_context == ("H1", "H4")
