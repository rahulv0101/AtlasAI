from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.learning.loss_to_win_playbook import build_loss_to_win_playbook
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.trader import AtlasTrader
from atlas_ai.ui_server import _build_compact_status_payload


def test_build_loss_to_win_playbook_from_historical_losses(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "historical_loss_analysis": {
                "source": "historical_backtest",
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "bars": 3000,
                "total_trades": 12,
                "losses": 3,
                "win_rate": 0.42,
                "loss_samples": [
                    {
                        "ticket": "bt_1",
                        "strategy": "ema_crossover",
                        "side": "buy",
                        "session": "london",
                        "regime": "trend",
                        "loss_classification": "weak_pullback",
                        "net_pnl_zar": -0.55,
                    },
                    {
                        "ticket": "bt_2",
                        "strategy": "ema_crossover",
                        "side": "sell",
                        "session": "london",
                        "regime": "trend",
                        "loss_classification": "weak_pullback",
                        "net_pnl_zar": -0.45,
                    },
                    {
                        "ticket": "bt_3",
                        "strategy": "gold_breakout_retest",
                        "side": "buy",
                        "session": "new_york",
                        "regime": "range",
                        "loss_classification": "stop_hunt",
                        "net_pnl_zar": -0.55,
                    },
                ],
            }
        }
    )

    playbook = build_loss_to_win_playbook(store)
    state = store.load_learning_state()["state"]

    assert state["loss_to_win_playbook"]["status"] == "active"
    assert playbook["losses"] == 3
    assert playbook["lessons"]["weak_pullback"]["samples"] == 2
    assert playbook["lessons"]["weak_pullback"]["can_boost_confidence"] is False
    assert "strong_or_turning_m1_momentum" in playbook["lessons"]["weak_pullback"]["confirmation_required"]
    assert playbook["lessons"]["stop_hunt"]["failure_axis"] == "timing"


def test_loss_to_win_snapshot_recognizes_corrected_pattern(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "historical_loss_analysis": {
                "patterns": {
                    "classification": [{"value": "weak_pullback", "count": 72}],
                },
                "loss_samples": [
                    {
                        "ticket": "bt_1",
                        "strategy": "ema_crossover",
                        "side": "buy",
                        "session": "london",
                        "regime": "trend",
                        "loss_classification": "weak_pullback",
                        "net_pnl_zar": -0.55,
                    }
                ],
            }
        }
    )
    build_loss_to_win_playbook(store)

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._historical_loss_pattern_counts = AtlasTrader._historical_loss_pattern_counts.__get__(trader, AtlasTrader)
    trader._load_loss_to_win_playbook = AtlasTrader._load_loss_to_win_playbook.__get__(trader, AtlasTrader)
    trader._loss_to_win_snapshot = AtlasTrader._loss_to_win_snapshot.__get__(trader, AtlasTrader)

    signal = TradeSignal("ema_crossover", "buy", 0.62, 100, 160)
    snapshot = trader._loss_to_win_snapshot(
        signal,
        {
            "entry_location_state": "pullback",
            "entry_momentum_state": "strong",
            "entry_wickiness": 0.22,
            "entry_expansion": 0.1,
            "entry_position": 0.35,
            "entry_trend_strength": 0.22,
        },
    )

    assert snapshot["status"] == "corrected_loss_pattern_present"
    assert snapshot["top_lesson"]["classification"] == "weak_pullback"
    assert snapshot["top_lesson"]["confirmed"] is True


def test_loss_to_win_snapshot_marks_missing_corrections(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "historical_loss_analysis": {
                "patterns": {"classification": [{"value": "late_entry", "count": 47}]},
                "loss_samples": [
                    {
                        "ticket": "bt_9",
                        "strategy": "gold_breakout_retest",
                        "side": "sell",
                        "session": "new_york",
                        "regime": "trend",
                        "loss_classification": "late_entry",
                        "net_pnl_zar": -0.55,
                    }
                ],
            }
        }
    )
    build_loss_to_win_playbook(store)

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._historical_loss_pattern_counts = AtlasTrader._historical_loss_pattern_counts.__get__(trader, AtlasTrader)
    trader._load_loss_to_win_playbook = AtlasTrader._load_loss_to_win_playbook.__get__(trader, AtlasTrader)
    trader._loss_to_win_snapshot = AtlasTrader._loss_to_win_snapshot.__get__(trader, AtlasTrader)

    snapshot = trader._loss_to_win_snapshot(
        TradeSignal("gold_breakout_retest", "sell", 0.66, 100, 160),
        {
            "entry_location_state": "breakout",
            "entry_momentum_state": "strong",
            "entry_wickiness": 0.2,
            "entry_expansion": 0.7,
            "entry_position": 0.96,
        },
    )

    assert snapshot["status"] == "loss_pattern_needs_correction"
    assert snapshot["top_lesson"]["classification"] == "late_entry"
    assert "no_overextension" in snapshot["top_lesson"]["missing"]


def test_compact_status_exposes_loss_to_win_playbook(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    status = {
        "action": "hold",
        "loss_to_win_playbook": {
            "status": "corrected_loss_pattern_present",
            "top_lesson": {"classification": "weak_pullback", "confirmed": True},
        },
    }

    payload = _build_compact_status_payload(status, {}, store)

    assert payload["loss_to_win_playbook"]["status"] == "corrected_loss_pattern_present"
    assert payload["loss_to_win_playbook"]["top_lesson"]["classification"] == "weak_pullback"
