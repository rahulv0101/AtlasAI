from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.learning.winner_playbook import build_winner_playbook
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.trader import AtlasTrader
from atlas_ai.ui_server import _build_compact_status_payload


def test_build_winner_playbook_from_historical_wins(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "historical_winner_analysis": {
                "source": "historical_backtest",
                "symbol": "XAUUSD",
                "timeframe": "M1",
                "bars": 3000,
                "total_trades": 20,
                "wins": 3,
                "losses": 2,
                "win_rate": 0.60,
                "win_samples": [
                    {
                        "ticket": "bt_1",
                        "strategy": "gold_breakout_retest",
                        "setup_family": "breakout",
                        "side": "buy",
                        "session": "new_york",
                        "regime": "trend",
                        "net_pnl_zar": 0.62,
                        "duration_minutes": 2.0,
                    },
                    {
                        "ticket": "bt_2",
                        "strategy": "gold_breakout_retest",
                        "setup_family": "breakout",
                        "side": "buy",
                        "session": "new_york",
                        "regime": "trend",
                        "net_pnl_zar": 0.58,
                        "duration_minutes": 3.0,
                    },
                    {
                        "ticket": "bt_3",
                        "strategy": "ema_crossover",
                        "setup_family": "continuation",
                        "side": "sell",
                        "session": "london",
                        "regime": "trend",
                        "net_pnl_zar": 0.3,
                        "duration_minutes": 1.0,
                    },
                ],
            }
        }
    )

    playbook = build_winner_playbook(store)

    assert playbook["status"] == "active"
    assert playbook["wins"] == 3
    assert playbook["lessons"]["gold_breakout_retest|breakout"]["samples"] == 2
    assert playbook["lessons"]["gold_breakout_retest|breakout"]["can_force_trade"] is False
    assert "no_overextension" in playbook["lessons"]["gold_breakout_retest|breakout"]["confirmation_required"]


def test_winner_playbook_snapshot_confirms_matching_winner_pattern(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "historical_winner_analysis": {
                "win_samples": [
                    {
                        "ticket": "bt_1",
                        "strategy": "momentum_continuation",
                        "setup_family": "continuation",
                        "side": "buy",
                        "session": "london",
                        "regime": "trend",
                        "net_pnl_zar": 0.4,
                    }
                ]
            }
        }
    )
    build_winner_playbook(store)

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_to_setup_family = AtlasTrader._strategy_to_setup_family.__get__(trader, AtlasTrader)
    trader._load_winner_playbook = AtlasTrader._load_winner_playbook.__get__(trader, AtlasTrader)
    trader._winner_playbook_snapshot = AtlasTrader._winner_playbook_snapshot.__get__(trader, AtlasTrader)

    snapshot = trader._winner_playbook_snapshot(
        TradeSignal("momentum_continuation", "buy", 0.7, 100, 160),
        {
            "entry_location_state": "pullback",
            "entry_momentum_state": "strong",
            "entry_trend_strength": 0.22,
        },
    )

    assert snapshot["status"] == "winner_pattern_confirmed"
    assert snapshot["top_lesson"]["strategy"] == "momentum_continuation"
    assert snapshot["top_lesson"]["confirmed"] is True


def test_compact_status_exposes_winner_playbook(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    payload = _build_compact_status_payload(
        {
            "action": "hold",
            "winner_playbook": {
                "status": "winner_pattern_confirmed",
                "top_lesson": {"strategy": "momentum_continuation"},
            },
        },
        {},
        store,
    )

    assert payload["winner_playbook"]["status"] == "winner_pattern_confirmed"
    assert payload["winner_playbook"]["top_lesson"]["strategy"] == "momentum_continuation"
