from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from atlas_ai.analysis.regime_tagging import classify_regime_label
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.trader import AtlasTrader


def test_regime_classifier_tags_trending_sample() -> None:
    closes = [2000.0 + idx * 1.5 for idx in range(60)]
    out = classify_regime_label(closes=closes)
    assert out["label"] == "trending"
    assert out["metrics"]["slope_pct"] > 0.12


def test_regime_classifier_tags_high_and_low_volatility_samples() -> None:
    high_closes = [2000.0 + (8 if idx % 2 else -8) for idx in range(60)]
    high = classify_regime_label(closes=high_closes)
    assert high["label"] == "high_volatility"
    assert high["metrics"]["atr_pct"] >= 0.06

    low_closes = [2000.0 + (0.02 if idx % 2 else -0.02) for idx in range(60)]
    low = classify_regime_label(closes=low_closes)
    assert low["label"] == "low_volatility"
    assert low["metrics"]["atr_pct"] <= 0.02


def test_trader_signal_generation_context_exposes_regime_label() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        m1_allowed_sessions=("asia", "london", "new_york"),
        signal_regime_gate_low_volatility_max_pct=0.03,
    )
    trader._market_regime = AtlasTrader._market_regime.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._signal_volatility_state = AtlasTrader._signal_volatility_state.__get__(trader, AtlasTrader)
    trader._entry_timing_context = AtlasTrader._entry_timing_context.__get__(trader, AtlasTrader)
    trader._signal_generation_context = AtlasTrader._signal_generation_context.__get__(trader, AtlasTrader)
    df = pd.DataFrame(
        {
            "open": [2000.0 + idx * 1.5 for idx in range(80)],
            "high": [2001.0 + idx * 1.5 for idx in range(80)],
            "low": [1999.0 + idx * 1.5 for idx in range(80)],
            "close": [2000.5 + idx * 1.5 for idx in range(80)],
            "tick_volume": [100 + idx for idx in range(80)],
        }
    )

    context = trader._signal_generation_context(df)

    assert context["regime_label"] == "trending"
    assert context["regime_label_detail"]["metrics"]["samples"] == 60


def test_compact_event_meta_persists_regime_label() -> None:
    status = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "action": "hold",
        "message": "No trade.",
        "chosen": {"strategy": "none", "side": "flat", "confidence": 0.0},
        "regime_label": "ranging",
        "regime_label_detail": {"label": "ranging", "metrics": {"samples": 60}},
        "context_snapshot": {"session": "london", "regime_label": "ranging"},
    }
    meta = AtlasTrader._compact_runtime_event_meta(AtlasTrader.__new__(AtlasTrader), status, "risk")

    assert meta["regime_label"] == "ranging"
    assert meta["context_snapshot"]["regime_label"] == "ranging"


def test_log_status_event_persists_regime_label_from_context(tmp_path) -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = KnowledgeStore(str(tmp_path))
    trader._augment_status_layers = lambda status: status

    status = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "action": "hold",
        "message": "No trade.",
        "chosen": {"strategy": "none", "side": "flat", "confidence": 0.0},
        "context_snapshot": {"session": "london", "regime": "trend"},
    }

    trader._log_status_event("risk", status)

    events = trader.store.load_events(limit=1, event_type="risk")
    meta = events[0]["meta"]
    assert meta["regime_label"] == "trending"
    assert meta["context_snapshot"]["regime_label"] == "trending"
