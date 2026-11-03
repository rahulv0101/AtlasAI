import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from atlas_ai.brain.why_engine import build_why_trace
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.trader import AtlasTrader


def test_trader_step_writes_backend_heartbeat_before_cycle_checks(tmp_path, monkeypatch):
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = KnowledgeStore(str(tmp_path))
    trader.settings = SimpleNamespace(
        storage_dir=str(tmp_path),
        gemini_influence_enabled=False,
    )
    trader._execute_trades_enabled = lambda *_args, **_kwargs: False

    def stop_after_heartbeat(*_args, **_kwargs):
        raise RuntimeError("stop after heartbeat")

    monkeypatch.setattr("atlas_ai.trader.evaluate_heartbeat_state", stop_after_heartbeat)

    with pytest.raises(RuntimeError, match="stop after heartbeat"):
        trader.step()

    heartbeat = trader.store.load_heartbeat()
    assert heartbeat["component"] == "backend_trader"
    assert heartbeat["mode"] == "paper"
    assert heartbeat["cycle_count"] == 1
    assert heartbeat["pid"] > 0


def test_decision_sample_payload_captures_rich_setup_context():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._session_label = lambda *_args, **_kwargs: "london"
    trader._volatility_bucket = AtlasTrader._volatility_bucket.__get__(trader, AtlasTrader)
    snapshot = SimpleNamespace(spread_points=34.0)
    signal = TradeSignal("momentum_continuation", "buy", 0.58, 100, 160)
    status = {
        "chart_features": {
            "trend_state": "uptrend",
            "trend_structure": "higher_high_higher_low",
            "momentum_state": "strong",
            "location_state": "pullback",
            "setup_type_hint": "pullback_continuation",
            "reliability": "medium",
        },
        "visual_chart_read": {
            "available": True,
            "pattern_seen": "pullback_continuation",
            "visual_confirmation": "supports_setup",
            "setup_quality": 0.72,
            "visual_bias": "buy",
        },
        "decision_analysis": {
            "setup_radar": {
                "top_watch": {
                    "name": "pullback_continuation",
                    "phase": "forming_confirmations_needed",
                    "trigger_state": "forming_waiting_for_confirmations",
                    "missing_to_entry": ["momentum_confirmation"],
                }
            },
            "top_blockers": [{"name": "momentum_quality", "delta_to_pass": 7}],
        },
        "m1_quality": {"factor": 1.0},
    }
    contextual_detail = {
        "context": {"body_ratio": 0.71},
        "setup_quality": {"score": 68.0, "reasons": ["pullback location improving"]},
    }
    trader._build_decision_sample_payload = AtlasTrader._build_decision_sample_payload.__get__(trader, AtlasTrader)

    payload = trader._build_decision_sample_payload(
        decision_id="sample123",
        chosen=signal,
        setup_family="continuation",
        status=status,
        regime={"name": "trend", "volatility_pct": 0.033, "trend_strength": 0.21},
        higher_tf_bias="buy",
        snapshot=snapshot,
        support_count=3,
        signal_count=5,
        contextual_detail=contextual_detail,
        meta_model={"probability": 0.61},
        execution={"execution_success": True},
    )

    assert payload["schema"] == "atlas_decision_sample_v2"
    assert payload["decision_id"] == "sample123"
    assert payload["entry_location_state"] == "pullback"
    assert payload["entry_momentum_state"] == "strong"
    assert payload["entry_trend_state"] == "uptrend"
    assert payload["volatility_state"] == "medium"
    assert payload["setup_radar"]["phase"] == "forming_confirmations_needed"
    assert payload["visual_chart_read"]["pattern_seen"] == "pullback_continuation"
    assert payload["top_blockers"][0]["name"] == "momentum_quality"
    assert payload["bounded_learning_only"] is True
    assert payload["execution_success"] is True


def test_load_loss_memory_preserves_backfill_observability(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [],
                "pattern_lessons": {},
                "lesson_backfill": {
                    "recent_outcomes": 200,
                    "pattern_lessons": 20,
                    "decision_context": {"outcomes_linked": 16},
                },
                "quarantine_summary": {"latest_quarantined_count": 1},
            }
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)

    memory = trader._load_loss_memory()

    assert memory["lesson_backfill"]["pattern_lessons"] == 20
    assert memory["lesson_backfill"]["decision_context"]["outcomes_linked"] == 16
    assert memory["quarantine_summary"]["latest_quarantined_count"] == 1


def test_closed_trade_learning_inherits_decision_id_from_open_position(monkeypatch, tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.log_event(
        "decision_sample",
        "Captured live decision sample.",
        {
            "decision_id": "abc12345",
            "strategy": "momentum_continuation",
            "side": "buy",
            "execution_success": True,
        },
    )

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(symbol="XAUUSD", magic_number=4242)
    trader.allocator = SimpleNamespace(update=lambda *args, **kwargs: None)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._load_win_memory = AtlasTrader._load_win_memory.__get__(trader, AtlasTrader)
    trader._save_loss_memory = AtlasTrader._save_loss_memory.__get__(trader, AtlasTrader)
    trader._save_win_memory = AtlasTrader._save_win_memory.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._sync_closed_trade_learning = AtlasTrader._sync_closed_trade_learning.__get__(trader, AtlasTrader)

    open_deal = SimpleNamespace(
        ticket="1001",
        symbol="XAUUSD",
        magic=4242,
        comment="AAI_abc123",
        position_id="90001",
        entry=0,
        time=1712060000,
        type=0,
        profit=0.0,
        swap=0.0,
        commission=0.0,
    )
    close_deal = SimpleNamespace(
        ticket="1002",
        symbol="XAUUSD",
        magic=4242,
        comment="",
        position_id="90001",
        entry=1,
        time=1712060300,
        type=1,
        profit=125.0,
        swap=0.0,
        commission=0.0,
    )

    monkeypatch.setattr("atlas_ai.trader.mt5.history_deals_get", lambda start, end: [open_deal, close_deal])
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT", 1)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT_BY", 3)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_BUY", 0)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_SELL", 1)

    memory = trader._sync_closed_trade_learning()

    assert len(memory["recent_outcomes"]) == 1
    outcome = memory["recent_outcomes"][0]
    assert outcome["decision_id"] == "abc12345"
    assert outcome["strategy"] == "momentum_continuation"


def test_closed_trade_learning_auto_refreshes_loss_lesson_backfill(monkeypatch, tmp_path):
    store = KnowledgeStore(str(tmp_path))
    now_ts = int(time.time())
    store.log_event(
        "decision_sample",
        "Captured live decision sample.",
        {
            "schema": "atlas_decision_sample_v2",
            "decision_id": "feedface",
            "strategy": "momentum_continuation",
            "setup_family": "continuation",
            "side": "buy",
            "session": "london",
            "regime": "trend",
            "higher_tf_bias": "buy",
            "spread_points": 34.0,
            "confidence": 0.58,
            "trend_strength": 0.2,
            "volatility_pct": 0.033,
            "entry_location_state": "pullback",
            "entry_momentum_state": "strong",
            "entry_trend_state": "uptrend",
            "execution_success": True,
        },
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(symbol="XAUUSD", magic_number=4242)
    trader.allocator = SimpleNamespace(update=lambda *args, **kwargs: None)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._load_win_memory = AtlasTrader._load_win_memory.__get__(trader, AtlasTrader)
    trader._save_loss_memory = AtlasTrader._save_loss_memory.__get__(trader, AtlasTrader)
    trader._save_win_memory = AtlasTrader._save_win_memory.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_to_setup_family = AtlasTrader._strategy_to_setup_family.__get__(trader, AtlasTrader)
    trader._build_trade_pattern_fingerprint = AtlasTrader._build_trade_pattern_fingerprint.__get__(trader, AtlasTrader)
    trader._classify_loss_cause = AtlasTrader._classify_loss_cause.__get__(trader, AtlasTrader)
    trader._update_pattern_lesson_memory = AtlasTrader._update_pattern_lesson_memory.__get__(trader, AtlasTrader)
    trader._sync_closed_trade_learning = AtlasTrader._sync_closed_trade_learning.__get__(trader, AtlasTrader)

    open_deal = SimpleNamespace(
        ticket="2001",
        symbol="XAUUSD",
        magic=4242,
        comment="AAI_feedface",
        position_id="90002",
        entry=0,
        time=now_ts,
        type=0,
        profit=0.0,
        swap=0.0,
        commission=0.0,
    )
    close_deal = SimpleNamespace(
        ticket="2002",
        symbol="XAUUSD",
        magic=4242,
        comment="",
        position_id="90002",
        entry=1,
        time=now_ts + 300,
        type=1,
        profit=-50.0,
        swap=0.0,
        commission=0.0,
    )
    monkeypatch.setattr("atlas_ai.trader.mt5.history_deals_get", lambda start, end: [open_deal, close_deal])
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT", 1)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT_BY", 3)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_BUY", 0)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_SELL", 1)

    memory = trader._sync_closed_trade_learning()
    events = store.load_events(event_type="learning")
    lessons = store.load_learning_state()["state"]["loss_memory"]["pattern_lessons"]
    lesson = next(iter(lessons.values()))

    assert len(memory["recent_outcomes"]) == 1
    assert any(row["message"] == "Loss-pattern lesson memory refreshed after closed trade." for row in events)
    assert lesson["context_quality"]["decision_context_linked"] is True
    assert "momentum=strong" in lesson["fingerprint"]
    assert "volatility=normal" in lesson["fingerprint"]


def test_closed_trade_learning_recovers_position_from_decision_sample_deal_id(monkeypatch, tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.log_event(
        "decision_sample",
        "Captured live decision sample.",
        {
            "decision_id": "fedcba98",
            "strategy": "gold_liquidity_sweep",
            "side": "sell",
            "execution_success": True,
            "order_id": 12345001,
            "deal_id": 54321001,
        },
    )

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(symbol="XAUUSD", magic_number=4242)
    trader.allocator = SimpleNamespace(update=lambda *args, **kwargs: None)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._load_win_memory = AtlasTrader._load_win_memory.__get__(trader, AtlasTrader)
    trader._save_loss_memory = AtlasTrader._save_loss_memory.__get__(trader, AtlasTrader)
    trader._save_win_memory = AtlasTrader._save_win_memory.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._sync_closed_trade_learning = AtlasTrader._sync_closed_trade_learning.__get__(trader, AtlasTrader)

    open_deal = SimpleNamespace(
        ticket="54321001",
        order="12345001",
        symbol="XAUUSD",
        magic=4242,
        comment="",
        position_id="91001",
        entry=0,
        time=1712060000,
        type=1,
        profit=0.0,
        swap=0.0,
        commission=0.0,
    )
    close_deal = SimpleNamespace(
        ticket="54321002",
        order="12345001",
        symbol="XAUUSD",
        magic=4242,
        comment="",
        position_id="91001",
        entry=1,
        time=1712060300,
        type=0,
        profit=-75.0,
        swap=0.0,
        commission=0.0,
    )

    monkeypatch.setattr("atlas_ai.trader.mt5.history_deals_get", lambda start, end: [open_deal, close_deal])
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT", 1)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT_BY", 3)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_BUY", 0)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_SELL", 1)

    memory = trader._sync_closed_trade_learning()

    assert len(memory["recent_outcomes"]) == 1
    outcome = memory["recent_outcomes"][0]
    assert outcome["decision_id"] == "fedcba98"
    assert outcome["strategy"] == "gold_liquidity_sweep"
    lesson = dict(outcome.get("post_trade_lesson", {}) or {})
    assert lesson["schema"] == "atlas_post_trade_lesson_v1"
    assert lesson["outcome"] == "loss"
    assert lesson["bounded_learning_only"] is True
    assert outcome["post_trade_review"]["post_trade_lesson"] == lesson
    pattern_lesson = dict(memory["pattern_lessons"][outcome["pattern_fingerprint"]] or {})
    assert pattern_lesson["post_trade_lesson"] == lesson


def test_closed_trade_learning_quarantines_untagged_broker_deals(monkeypatch, tmp_path):
    store = KnowledgeStore(str(tmp_path))
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(symbol="XAUUSD", magic_number=4242)
    trader.allocator = SimpleNamespace(update=lambda *args, **kwargs: None)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._load_win_memory = AtlasTrader._load_win_memory.__get__(trader, AtlasTrader)
    trader._save_loss_memory = AtlasTrader._save_loss_memory.__get__(trader, AtlasTrader)
    trader._save_win_memory = AtlasTrader._save_win_memory.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._sync_closed_trade_learning = AtlasTrader._sync_closed_trade_learning.__get__(trader, AtlasTrader)

    untagged_close = SimpleNamespace(
        ticket="3002",
        order="order3002",
        symbol="XAUUSD",
        magic=0,
        comment="",
        position_id="97001",
        entry=1,
        time=1712060300,
        type=1,
        profit=-500.0,
        swap=0.0,
        commission=0.0,
    )

    monkeypatch.setattr("atlas_ai.trader.mt5.history_deals_get", lambda start, end: [untagged_close])
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT", 1)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT_BY", 3)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_BUY", 0)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_SELL", 1)

    memory = trader._sync_closed_trade_learning()

    assert memory["recent_outcomes"] == []
    assert memory["quarantine_summary"]["latest_quarantined_count"] == 1
    assert memory["quarantined_non_atlas_deals"][0]["ticket"] == "3002"


def test_trader_strategy_key_canonicalizes_truncated_aliases():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)

    assert trader._strategy_key("macd_tren") == "macd_trend"
    assert trader._strategy_key("vwap_prox") == "vwap_proxy_reversion"
    assert trader._strategy_key("candlesti") == "candlestick_patterns"


def test_m1_loss_pattern_block_reason_triggers_on_repeated_losses():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe="M1",
        m1_loss_block_enabled=True,
        m1_loss_block_min_samples=6,
        m1_loss_block_max_win_rate=0.34,
        m1_loss_block_min_streak=3,
    )
    trader._loss_pattern_block_reason = AtlasTrader._loss_pattern_block_reason.__get__(trader, AtlasTrader)

    memory = {
        "recent_outcomes": [
            {"side": "buy", "outcome": "win"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
        ],
        "side_hour_stats": {},
    }

    reason = trader._loss_pattern_block_reason("buy", memory, __import__("datetime").datetime(2026, 4, 3, 10, 0, 0))
    assert "M1 loss-pattern block" in reason


def test_m1_loss_pattern_block_detail_exposes_clearance_evidence():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe="M1",
        m1_loss_block_enabled=True,
        m1_loss_block_min_samples=6,
        m1_loss_block_max_win_rate=0.34,
        m1_loss_block_min_streak=3,
    )

    memory = {
        "recent_outcomes": [
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
            {"side": "buy", "outcome": "loss"},
        ],
        "side_hour_stats": {"buy:10": {"wins": 0, "losses": 6}},
    }

    detail = trader._loss_pattern_block_detail("buy", memory, __import__("datetime").datetime(2026, 4, 3, 10, 0, 0))

    assert detail["blocked"] is True
    assert detail["trigger"] == "repeated_side_losses"
    assert detail["samples"] == 6
    assert detail["hour_context"]["key"] == "buy:10"
    assert "loss streak reset" in detail["evidence_required"]
    assert any(item["code"] == "loss_streak_breaks" and item["status"] == "blocked" for item in detail["recovery_checklist"])
    assert any(item["code"] == "same_side_win_rate_improves" for item in detail["recovery_checklist"])


def test_entry_timing_learning_haircuts_weak_pullback_without_flattening(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.update_learning_state(
        lambda state: {
            **state,
            "historical_loss_analysis": {
                "patterns": {
                    "classification": [
                        {"value": "weak_pullback", "count": 133},
                        {"value": "stop_hunt", "count": 86},
                    ]
                }
            },
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._historical_loss_pattern_counts = AtlasTrader._historical_loss_pattern_counts.__get__(trader, AtlasTrader)
    trader._apply_entry_timing_learning = AtlasTrader._apply_entry_timing_learning.__get__(trader, AtlasTrader)

    signal = TradeSignal("ema_crossover", "buy", 0.70, 100, 150)
    adjusted = trader._apply_entry_timing_learning(
        signal,
        {
            "entry_location_state": "pullback",
            "entry_momentum_state": "mixed",
            "entry_wickiness": 0.62,
            "entry_expansion": 0.1,
            "entry_position": 0.35,
        },
    )

    assert adjusted.side == "buy"
    assert 0.0 < adjusted.confidence < 0.70


def test_entry_timing_learning_leaves_clean_strong_pullback_alone(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.update_learning_state(
        lambda state: {
            **state,
            "historical_loss_analysis": {
                "patterns": {
                    "classification": [
                        {"value": "weak_pullback", "count": 133},
                        {"value": "stop_hunt", "count": 86},
                    ]
                }
            },
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._historical_loss_pattern_counts = AtlasTrader._historical_loss_pattern_counts.__get__(trader, AtlasTrader)
    trader._apply_entry_timing_learning = AtlasTrader._apply_entry_timing_learning.__get__(trader, AtlasTrader)

    signal = TradeSignal("ema_crossover", "buy", 0.70, 100, 150)
    adjusted = trader._apply_entry_timing_learning(
        signal,
        {
            "entry_location_state": "pullback",
            "entry_momentum_state": "strong",
            "entry_wickiness": 0.25,
            "entry_expansion": 0.1,
            "entry_position": 0.35,
        },
    )

    assert adjusted.confidence == 0.70


def test_online_learning_applies_bounded_confidence_haircut_on_matching_pattern(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "online_learning": {
                "latest": {
                    "allowed_influence": "bounded_confidence_haircut",
                    "confidence_haircut_cap": 0.06,
                    "evidence_strength": 0.9,
                    "overreaction_risk": 0.5,
                    "samples": 20,
                    "top_fingerprint": "late_us_mid_range",
                    "current_blockers": ["spread_liquidity_too_poor", "no_clear_setup"],
                    "context": {
                        "session": "late_us",
                        "momentum": "mixed",
                        "location": "mid_range",
                        "spread_points": 45.0,
                    },
                }
            }
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader._online_learning_adjustment = AtlasTrader._online_learning_adjustment.__get__(trader, AtlasTrader)

    signal = TradeSignal("evolved_rule", "sell", 0.60, 100, 160)
    new_conf, meta = trader._online_learning_adjustment(
        signal,
        {
            "risk": {"spread_points": 45.0},
            "context_snapshot": {
                "session": "late_us",
                "momentum": "mixed",
                "location": "mid_range",
                "spread_points": 45.0,
            },
            "decision_analysis": {"hard_blockers": ["spread_liquidity_too_poor"]},
        },
    )

    assert meta["applied"] is True
    assert meta["allowed_influence"] == "bounded_confidence_haircut"
    assert 0.0 < meta["haircut"] <= 0.06
    assert new_conf < signal.confidence
    assert "cannot boost" in meta["policy_note"]


def test_online_learning_does_not_adjust_when_context_mismatch(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "online_learning": {
                "latest": {
                    "allowed_influence": "bounded_confidence_haircut",
                    "confidence_haircut_cap": 0.06,
                    "evidence_strength": 0.9,
                    "samples": 20,
                    "current_blockers": ["spread_liquidity_too_poor"],
                    "context": {
                        "session": "late_us",
                        "momentum": "mixed",
                        "location": "mid_range",
                        "spread_points": 45.0,
                    },
                }
            }
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader._online_learning_adjustment = AtlasTrader._online_learning_adjustment.__get__(trader, AtlasTrader)

    signal = TradeSignal("gold_breakout_retest", "buy", 0.60, 100, 160)
    new_conf, meta = trader._online_learning_adjustment(
        signal,
        {
            "risk": {"spread_points": 20.0},
            "context_snapshot": {
                "session": "new_york",
                "momentum": "strong",
                "location": "breakout",
                "spread_points": 20.0,
            },
            "decision_analysis": {"hard_blockers": []},
        },
    )

    assert new_conf == signal.confidence
    assert meta["applied"] is False
    assert meta["reason"] == "context_mismatch"


def test_online_learning_report_only_never_changes_confidence(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "online_learning": {
                "latest": {
                    "allowed_influence": "report_only",
                    "confidence_haircut_cap": 0.08,
                    "evidence_strength": 1.0,
                    "samples": 100,
                    "current_blockers": ["no_clear_setup"],
                    "context": {
                        "session": "london",
                        "momentum": "mixed",
                        "location": "mid_range",
                        "spread_points": 35.0,
                    },
                }
            }
        }
    )
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader._online_learning_adjustment = AtlasTrader._online_learning_adjustment.__get__(trader, AtlasTrader)

    signal = TradeSignal("momentum_continuation", "buy", 0.60, 100, 160)
    new_conf, meta = trader._online_learning_adjustment(
        signal,
        {
            "risk": {"spread_points": 35.0},
            "context_snapshot": {
                "session": "london",
                "momentum": "mixed",
                "location": "mid_range",
                "spread_points": 35.0,
            },
            "decision_analysis": {"hard_blockers": ["no_clear_setup"]},
        },
    )

    assert new_conf == signal.confidence
    assert meta["applied"] is False
    assert meta["allowed_influence"] == "report_only"
    assert meta["reason"] == "report_only_no_confidence_change"


def test_closed_trade_learning_persists_post_trade_loss_classification(monkeypatch, tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.log_event(
        "decision_sample",
        "Captured live decision sample.",
        {
            "decision_id": "loss12345",
            "strategy": "momentum_continuation",
            "side": "buy",
            "execution_success": True,
        },
    )

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(symbol="XAUUSD", magic_number=4242, max_spread_points=80.0)
    trader.allocator = SimpleNamespace(update=lambda *args, **kwargs: None)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._load_win_memory = AtlasTrader._load_win_memory.__get__(trader, AtlasTrader)
    trader._save_loss_memory = AtlasTrader._save_loss_memory.__get__(trader, AtlasTrader)
    trader._save_win_memory = AtlasTrader._save_win_memory.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._sync_closed_trade_learning = AtlasTrader._sync_closed_trade_learning.__get__(trader, AtlasTrader)
    trader._classify_loss_cause = AtlasTrader._classify_loss_cause.__get__(trader, AtlasTrader)

    open_deal = SimpleNamespace(
        ticket="2001",
        order="order2001",
        symbol="XAUUSD",
        magic=4242,
        comment="AAI_loss123 [sl]",
        position_id="95001",
        entry=0,
        time=1712060000,
        type=0,
        profit=0.0,
        swap=0.0,
        commission=0.0,
    )
    close_deal = SimpleNamespace(
        ticket="2002",
        order="order2001",
        symbol="XAUUSD",
        magic=4242,
        comment="AAI_loss123 [sl]",
        position_id="95001",
        entry=1,
        time=1712060180,
        type=1,
        profit=-52.0,
        swap=0.0,
        commission=0.0,
    )

    monkeypatch.setattr("atlas_ai.trader.mt5.history_deals_get", lambda start, end: [open_deal, close_deal])
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT", 1)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_ENTRY_OUT_BY", 3)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_BUY", 0)
    monkeypatch.setattr("atlas_ai.trader.mt5.DEAL_TYPE_SELL", 1)

    memory = trader._sync_closed_trade_learning()

    outcome = memory["recent_outcomes"][0]
    assert outcome["loss_classification"] == "false_continuation"
    assert outcome["post_trade_review"]["classification"] == "false_continuation"
    assert outcome["post_trade_review"]["validation_required"] is True
    assert outcome["setup_family"] == "continuation"
    assert outcome["regime"] == "unknown"
    assert outcome["pattern_fingerprint"]
    assert outcome["execution_lesson"]["classification"] == "false_continuation"
    assert "momentum" in outcome["execution_lesson"]["correction"]
    assert memory["pattern_lessons"][outcome["pattern_fingerprint"]]["losses"] == 1


def test_loss_pattern_lesson_haircuts_repeat_until_correction_present(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_to_setup_family = AtlasTrader._strategy_to_setup_family.__get__(trader, AtlasTrader)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._build_trade_pattern_fingerprint = AtlasTrader._build_trade_pattern_fingerprint.__get__(trader, AtlasTrader)
    trader._loss_lesson_match_score = AtlasTrader._loss_lesson_match_score.__get__(trader, AtlasTrader)
    trader._find_loss_pattern_lesson = AtlasTrader._find_loss_pattern_lesson.__get__(trader, AtlasTrader)
    trader._loss_pattern_lesson_adjustment = AtlasTrader._loss_pattern_lesson_adjustment.__get__(trader, AtlasTrader)

    weak_context = {
        "session": "london",
        "regime": "trend",
        "entry_trend_state": "trend",
        "entry_location_state": "pullback",
        "entry_momentum_state": "mixed",
        "entry_wickiness": 0.62,
        "entry_expansion": 0.1,
        "entry_position": 0.35,
        "entry_trend_strength": 0.16,
    }
    fingerprint = trader._build_trade_pattern_fingerprint(
        strategy="momentum_continuation",
        side="buy",
        setup_family="continuation",
        session="london",
        regime="trend",
        context=weak_context,
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [],
                "pattern_lessons": {
                    fingerprint: {
                        "fingerprint": fingerprint,
                        "samples": 3,
                        "wins": 0,
                        "losses": 3,
                        "loss_rate": 1.0,
                        "execution_lesson": {
                            "classification": "weak_pullback",
                            "correction": "Wait for pullback location and M1 momentum turn.",
                            "confirmation_required": ["valid_pullback_location", "m1_momentum_turn"],
                        },
                        "post_trade_lesson": {
                            "schema": "atlas_post_trade_lesson_v1",
                            "failure_axis": "location_or_structure",
                            "idea_quality": "unproven_or_weak",
                            "timing_quality": "poor_location_timing",
                            "next_time_action": "treat_similar_setup_as_developing_until_confirmation_improves",
                            "future_recognition": ["setup=continuation", "location=pullback", "momentum=mixed"],
                        },
                    }
                },
            }
        }
    )

    signal = TradeSignal("momentum_continuation", "buy", 0.70, 100, 160)
    adjusted, meta = trader._loss_pattern_lesson_adjustment(signal, weak_context)

    assert meta["applied"] is True
    assert adjusted < signal.confidence
    assert meta["execution_lesson"]["classification"] == "weak_pullback"
    assert meta["post_trade_lesson"]["failure_axis"] == "location_or_structure"
    assert meta["decision_critic"]["status"] == "similar_weak_fingerprint_seen"
    assert meta["match_strength"] == "exact_match"
    assert meta["decision_critic"]["next_time_action"] == "treat_similar_setup_as_developing_until_confirmation_improves"
    assert meta["haircut"] > 0.09

    corrected_context = {**weak_context, "entry_momentum_state": "strong", "entry_wickiness": 0.25}
    corrected, corrected_meta = trader._loss_pattern_lesson_adjustment(signal, corrected_context)

    assert corrected == signal.confidence
    assert corrected_meta["applied"] is False
    assert corrected_meta["reason"] == "corrective_confirmation_present"
    assert corrected_meta["match_strength"] == "strong_partial_match"


def test_loss_pattern_lesson_uses_strong_partial_match_without_requiring_unknown_fields(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(timeframe="M1")
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_to_setup_family = AtlasTrader._strategy_to_setup_family.__get__(trader, AtlasTrader)
    trader._load_loss_memory = AtlasTrader._load_loss_memory.__get__(trader, AtlasTrader)
    trader._build_trade_pattern_fingerprint = AtlasTrader._build_trade_pattern_fingerprint.__get__(trader, AtlasTrader)
    trader._loss_lesson_match_score = AtlasTrader._loss_lesson_match_score.__get__(trader, AtlasTrader)
    trader._find_loss_pattern_lesson = AtlasTrader._find_loss_pattern_lesson.__get__(trader, AtlasTrader)
    trader._loss_pattern_lesson_adjustment = AtlasTrader._loss_pattern_lesson_adjustment.__get__(trader, AtlasTrader)

    lesson_fingerprint = (
        "strategy=momentum_continuation|side=sell|setup=continuation|session=london|"
        "regime=unknown|trend=unknown|location=unknown|momentum=unknown|volatility=unknown"
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [],
                "pattern_lessons": {
                    lesson_fingerprint: {
                        "fingerprint": lesson_fingerprint,
                        "samples": 4,
                        "wins": 1,
                        "losses": 3,
                        "loss_rate": 0.75,
                        "execution_lesson": {
                            "classification": "false_continuation",
                            "confirmation_required": ["strong_m1_momentum", "structure_hold"],
                        },
                        "post_trade_lesson": {
                            "failure_axis": "confirmation_quality",
                            "idea_quality": "unproven_or_weak",
                            "timing_quality": "needs_review",
                            "next_time_action": "treat_similar_setup_as_developing_until_confirmation_improves",
                            "future_recognition": ["setup=continuation", "side=sell", "session=london"],
                        },
                    }
                },
            }
        }
    )

    context = {
        "session": "london",
        "regime": "trend",
        "entry_trend_state": "trend",
        "entry_location_state": "pullback",
        "entry_momentum_state": "mixed",
        "entry_trend_strength": 0.14,
    }
    signal = TradeSignal("momentum_continuation", "sell", 0.70, 100, 160)
    adjusted, meta = trader._loss_pattern_lesson_adjustment(signal, context)

    assert meta["applied"] is True
    assert adjusted < signal.confidence
    assert meta["match_strength"] == "strong_partial_match"
    assert meta["matched_fingerprint"] == lesson_fingerprint
    assert meta["match_score"] >= 0.8
    assert "strategy" in meta["matched_fields"]
    assert meta["haircut"] < 0.12


def test_timing_readiness_marks_timing_lesson_not_ready_without_momentum():
    trader = AtlasTrader.__new__(AtlasTrader)
    critic = {
        "previous_failure_axis": "timing",
        "timing_quality": "early_or_late",
        "next_time_action": "wait_for_fresher_entry_or_retest_before_repeating",
    }
    context = {
        "entry_location_state": "pullback",
        "entry_momentum_state": "mixed",
        "setup_radar": {"phase": "forming", "trigger_state": "waiting"},
    }

    payload = trader._timing_readiness_snapshot(critic, context)

    assert payload["status"] == "timing_not_ready"
    assert payload["fresh_location_or_retest"] is True
    assert payload["momentum_turning_or_strong"] is False
    assert payload["missing"] == ["strong_or_turning_m1_momentum"]
    assert payload["recommendation"] == "wait_for_fresher_entry_or_retest"
    assert payload["policy"]["does_not_execute"] is True
    assert payload["policy"]["does_not_change_strategy_logic"] is True
    assert payload["policy"]["does_not_change_execution_policy"] is True


def test_timing_readiness_clears_to_existing_gates_when_fresh_and_momentum_strong():
    trader = AtlasTrader.__new__(AtlasTrader)
    critic = {
        "previous_failure_axis": "timing",
        "timing_quality": "early_or_late",
    }
    context = {
        "entry_location_state": "retest",
        "entry_momentum_state": "strong",
    }

    payload = trader._timing_readiness_snapshot(critic, context)

    assert payload["status"] == "timing_ready_for_normal_gates"
    assert payload["missing"] == []
    assert payload["recommendation"] == "timing_clear_for_existing_gates"
    assert payload["policy"]["observability_only"] is True
    assert payload["policy"]["does_not_change_risk_policy"] is True


def test_timing_readiness_uses_context_snapshot_location_and_momentum_fields():
    trader = AtlasTrader.__new__(AtlasTrader)
    critic = {"previous_failure_axis": "timing", "timing_quality": "early_or_late"}
    context = {
        "location": "retest",
        "momentum": "turning",
    }

    payload = trader._timing_readiness_snapshot(critic, context)

    assert payload["status"] == "timing_ready_for_normal_gates"
    assert payload["fresh_location_or_retest"] is True
    assert payload["momentum_turning_or_strong"] is True
    assert payload["missing"] == []


def test_timing_discipline_blocks_mid_range_mixed_momentum_directional_candidate():
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    context = {
        "location": "mid_range",
        "momentum": "mixed",
        "trend_state": "trend",
        "trend_strength": 0.26,
        "session": "new_york",
        "setup_type_hint": "pullback_continuation",
    }
    status = {
        "learning_summary": {
            "loss_reason_summary": {
                "top_reason": "stop_hunt",
            }
        }
    }

    payload = trader._timing_discipline_snapshot(signal, context, status)

    assert payload["status"] == "timing_block"
    assert payload["hard_block"] is True
    assert payload["passed"] is False
    assert payload["check"] == "timing_discipline"
    assert payload["reason_code"] == "AWAITING_TRADABLE_LOCATION"
    assert "tradable_location" in payload["missing"]
    assert "strong_or_turning_m1_momentum" in payload["missing"]
    assert "sweep_rejection_or_structure_reclaim" in payload["missing"]
    assert payload["condition_values"]["observed"]["location"] == "mid_range"
    assert payload["condition_values"]["observed"]["momentum"] == "mixed"
    assert payload["condition_values"]["observed"]["session"] == "new_york"
    assert payload["structured_result"]["detail"] == payload["detail"]
    message = trader._operator_message_from_timing_discipline(payload)
    assert message == f"Timing discipline blocked trade: {payload['detail']}"
    assert "location=mid_range" in message
    assert "momentum=mixed" in message
    assert payload["recommendation"] == "wait_for_clean_timing_confirmation"
    assert payload["policy"]["does_not_force_entries"] is True
    assert payload["policy"]["does_not_lower_thresholds"] is True


def test_timing_discipline_preset_allows_mid_range_only_with_strong_momentum():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(timing_allow_mid_range_strong_momentum=True)
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    status = {"learning_summary": {"loss_reason_summary": {"top_reason": "none"}}}

    strong_payload = trader._timing_discipline_snapshot(
        signal,
        {
            "location": "mid_range",
            "momentum": "strong",
            "trend_state": "trend",
            "trend_strength": 0.26,
            "session": "london",
            "setup_type_hint": "pullback_continuation",
        },
        status,
    )

    assert strong_payload["hard_block"] is False
    assert strong_payload["passed"] is True
    assert "tradable_location" not in strong_payload["missing"]
    assert "strong_or_turning_m1_momentum" not in strong_payload["missing"]
    assert "mid_range_accepted_by_strong_momentum" in strong_payload["reasons"]

    mixed_payload = trader._timing_discipline_snapshot(
        signal,
        {
            "location": "mid_range",
            "momentum": "mixed",
            "trend_state": "trend",
            "trend_strength": 0.26,
            "session": "london",
            "setup_type_hint": "pullback_continuation",
        },
        status,
    )

    assert mixed_payload["hard_block"] is True
    assert mixed_payload["passed"] is False
    assert "tradable_location" in mixed_payload["missing"]
    assert "strong_or_turning_m1_momentum" in mixed_payload["missing"]


def test_timing_discipline_allows_clean_retest_with_strong_momentum_to_existing_gates():
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "sell", 0.72, 100, 160)
    context = {
        "location": "retest",
        "momentum": "strong",
        "trend_state": "trend",
        "trend_strength": 0.3,
        "session": "london",
        "setup_type_hint": "pullback_continuation",
    }
    status = {
        "learning_summary": {
            "loss_reason_summary": {
                "top_reason": "stop_hunt",
            }
        }
    }

    payload = trader._timing_discipline_snapshot(signal, context, status)

    assert payload["status"] == "timing_clear_for_existing_gates"
    assert payload["hard_block"] is False
    assert payload["passed"] is True
    assert payload["reason_code"] is None
    assert payload["missing"] == []
    assert payload["recommendation"] == "allow_existing_gates_to_decide"


def test_timing_discipline_allows_clean_retest_with_turning_momentum_to_existing_gates():
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "sell", 0.72, 100, 160)
    context = {
        "location": "retest",
        "momentum": "turning",
        "trend_state": "trend",
        "trend_strength": 0.3,
        "session": "london",
        "setup_type_hint": "pullback_continuation",
    }
    status = {
        "learning_summary": {
            "loss_reason_summary": {
                "top_reason": "stop_hunt",
            }
        }
    }

    payload = trader._timing_discipline_snapshot(signal, context, status)

    assert payload["status"] == "timing_clear_for_existing_gates"
    assert payload["hard_block"] is False
    assert payload["passed"] is True
    assert payload["reason_code"] is None
    assert payload["missing"] == []
    assert payload["observed"]["momentum"] == "turning"


def test_timing_discipline_uses_continuous_visual_watch_as_execution_quality_evidence():
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    context = {
        "location": "pullback",
        "momentum": "strong",
        "trend_state": "uptrend",
        "trend_strength": 0.3,
        "session": "london",
        "setup_type_hint": "pullback_continuation",
    }
    status = {
        "visual_chart_read": {
            "visual_watch_used_for_decision": True,
            "visual_confirmation": "does_not_confirm",
            "visual_features": {
                "visual_location": "mid_range",
                "visual_momentum": "weak",
                "visual_direction": "sideways",
            },
        }
    }

    payload = trader._timing_discipline_snapshot(signal, context, status)

    assert payload["hard_block"] is True
    assert payload["reason_code"] == "AWAITING_VISUAL_CONFIRMATION"
    assert "visual_confirmation" in payload["missing"]
    assert "visual_tradable_location" in payload["missing"]
    assert "visual_momentum_confirmation" in payload["missing"]
    assert "visual_directional_impulse" in payload["missing"]
    assert payload["observed"]["visual_watch_used"] is True


def test_timing_discipline_block_is_persisted_in_status_event_meta(tmp_path):
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = KnowledgeStore(str(tmp_path))
    trader.settings = SimpleNamespace(storage_dir=str(tmp_path))
    trader._augment_status_layers = lambda status: status
    trader._attach_regime_label = lambda status, regime=None: None
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    timing = trader._timing_discipline_snapshot(
        signal,
        {
            "location": "mid_range",
            "momentum": "mixed",
            "trend_state": "trend",
            "trend_strength": 0.26,
            "session": "new_york",
            "setup_type_hint": "pullback_continuation",
        },
        {},
    )
    status = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "action": "blocked",
        "message": trader._operator_message_from_timing_discipline(timing),
        "chosen": {"strategy": signal.strategy, "side": signal.side, "confidence": signal.confidence},
        "timing_discipline": timing,
    }

    trader._log_status_event("risk", status)

    events = trader.store.load_events(limit=1, event_type="risk")
    persisted = dict(dict(events[-1].get("meta", {}) or {}).get("timing_discipline", {}) or {})
    assert persisted["check"] == "timing_discipline"
    assert persisted["reason_code"] == "AWAITING_TRADABLE_LOCATION"
    assert persisted["condition_values"]["observed"]["location"] == "mid_range"
    assert events[-1]["message"] == f"Timing discipline blocked trade: {persisted['detail']}"


def test_why_trace_uses_structured_timing_discipline_as_conservative_filter_detail():
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    timing = trader._timing_discipline_snapshot(
        signal,
        {"location": "mid_range", "momentum": "mixed", "session": "new_york"},
        {},
    )
    status = {
        "action": "blocked",
        "message": trader._operator_message_from_timing_discipline(timing),
        "chosen": {"strategy": signal.strategy, "side": signal.side, "confidence": signal.confidence},
        "timing_discipline": timing,
    }

    trace = build_why_trace(status, {}, {})
    conservative = [row for row in trace["stages"] if row["stage"] == "conservative_filter"][0]

    assert conservative["status"] == "blocked"
    assert conservative["reason"] == timing["detail"]
    assert conservative["data"]["timing_discipline"]["reason_code"] == "AWAITING_TRADABLE_LOCATION"


def test_attach_visual_chart_read_prefers_fresh_visual_watch_payload(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    visual_dir = tmp_path / "chart_vision"
    visual_dir.mkdir()
    image_path = visual_dir / "latest_desktop_chart.png"
    image_path.write_bytes(b"fake-image")
    store.save_status({})
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(storage_dir=str(tmp_path), symbol="XAUUSD", timeframe="M1")
    trader._last_visual_chart_capture_at = 0.0
    read = {
        "source": "screenshot_plus_chart_features",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "live_chart_window_confirmed": True,
        "image": {"path": str(image_path)},
        "visual_confirmation": "does_not_confirm",
        "visual_features": {"status": "parsed"},
        "capture": {"capture_source": "mt5_window"},
    }
    (visual_dir / "latest_visual_read.json").write_text(__import__("json").dumps(read), encoding="utf-8")

    status = {"symbol": "XAUUSD", "timeframe": "M1", "chart_features": {}}

    trader._attach_visual_chart_read(status)

    assert status["visual_chart_read"]["visual_watch_used_for_decision"] is True
    assert status["visual_chart_read"]["visual_features"]["status"] == "parsed"


def test_live_loss_classifier_understands_pullback_and_late_entry(tmp_path):
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(max_spread_points=80.0)
    trader._classify_loss_cause = AtlasTrader._classify_loss_cause.__get__(trader, AtlasTrader)

    weak_pullback = trader._classify_loss_cause(
        outcome="loss",
        exit_reason="stop_loss",
        duration_minutes=9.0,
        session="london",
        comment="AAI_test [sl]",
        spread_points=0.0,
        setup_family="pullback",
        regime="trend",
    )
    late_entry = trader._classify_loss_cause(
        outcome="loss",
        exit_reason="stop_loss",
        duration_minutes=18.0,
        session="london",
        comment="AAI_test [sl]",
        spread_points=0.0,
        setup_family="continuation",
        regime="trend",
    )

    assert weak_pullback == "weak_pullback"
    assert late_entry == "late_entry"
