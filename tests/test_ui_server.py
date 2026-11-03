from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.paper_compare import ComparisonTracker
from atlas_ai.self_improvement import append_or_update_proposal, load_self_improvement_bundle
from atlas_ai.ui_server import create_ui_app, start_visual_watch_loop
from atlas_ai.vision.chart_vision import save_visual_read


def test_ui_execution_gate_uses_mt5_adapter(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    app = create_ui_app(store)
    svc = app.view_functions["ask_atlas"].__closure__[0].cell_contents
    assert svc.execution.snapshot()["adapter_type"] == "MT5BrokerAdapter"


def test_create_ui_app_triggers_startup_greeting_without_ui_ready_call(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    calls = {"n": 0}

    def _fake_announce(*_args, **_kwargs) -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", _fake_announce)
    _ = create_ui_app(store)
    assert calls["n"] == 1


def test_health_api_is_lightweight_and_reachable(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    app = create_ui_app(store)

    response = app.test_client().get("/api/health")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["service"] == "atlas-ui"
    assert "ts" in payload


def test_compact_status_exposes_backend_staleness_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKEND_HEARTBEAT_STALE_AFTER_SECONDS", "10")
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status({"action": "hold", "message": "Last known hold."})
    store.heartbeat_file.write_text(
        json.dumps(
            {
                "schema": "atlas_backend_heartbeat_v1",
                "component": "backend_trader",
                "last_heartbeat": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_ts": time.time() - 30,
                "mode": "paper",
                "pid": 1234,
                "cycle_count": 7,
                "cycle_status": "running",
            }
        ),
        encoding="utf-8",
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    backend_health = payload["system_health"]["backend_health"]

    assert backend_health["status"] == "stale"
    assert backend_health["cycle_count"] == 7
    assert "Backend heartbeat is stale" in payload["system_health"]["backend_staleness_warning"]


def test_compact_status_reconstructs_mode_truth_from_startup_flags(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "startup_verification": {
                "startup_ok": True,
                "execute_trades": True,
                "effective_mode": "live",
                "gate_execution_environment": "demo",
                "startup_message": "ATLAS STARTED IN GUARDED DEMO MODE",
            },
        }
    )
    store.save_status(
        {
            "action": "connected",
            "message": "Broker connected. Live loop started.",
            "execute_trades": True,
            "system_health": {"warnings": []},
        }
    )
    store.write_backend_heartbeat(mode="live", pid=1234)

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()

    assert payload["execute_trades"] is True
    assert payload["effective_mode"] == "live"
    assert payload["gate_execution_environment"] == "demo"
    assert payload["system_health"]["effective_mode"] == "live"
    assert payload["system_health"]["mode_truth"]["gate_execution_environment"] == "demo"
    assert payload["system_health"]["backend_health"]["status"] == "live"
    assert payload["system_health"]["backend_staleness_warning"] == ""


def test_compact_status_api_returns_operator_critical_fields(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "message": "No high-confidence trade found.",
            "gemini_enabled": True,
            "chosen": {"confidence": 0.45, "side": "flat"},
            "thresholds": {"confidence": 0.44},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "market": {"recent_close": list(range(100))},
            "signals": [{"strategy": "ema", "side": "buy", "confidence": 0.1}],
            "learning_critic": {
                "enabled": True,
                "applied": True,
                "status": "similar_weak_fingerprint_seen",
                "loss_rate": 0.75,
                "previous_failure_axis": "timing",
                "next_time_action": "wait for cleaner confirmation",
            },
            "blocker_classification": {"class": "signal_quality", "primary": "No high-confidence trade found."},
            "system_health": {
                "effective_mode_label": "LEARNING_AGGRESSIVE_DEMO",
                "broker_connectivity_status": {"connected": True},
            },
            "decision_analysis": {
                "blockers": ["no_clear_setup"],
                "setup_quality_observation": {"top_setup": "pullback_continuation"},
            },
            "visual_chart_read": {
                "advisory_only": True,
                "pattern_seen": "pullback_continuation",
                "setup_quality": 0.74,
                "visual_confirmation": "supports_setup",
            },
        }
    )
    store.save_runtime_flags(
        {
            "ui_desktop_window": {"ready": True, "wrapper_pid": 0},
            "execution": {
                "auto_trade_reason": "outside_active_session",
                "automation_live_allowed": True,
            },
        }
    )
    app = create_ui_app(store)
    app = create_ui_app(store)
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "session_control": {
                        "session_active": False,
                        "current_session": "none",
                        "next_session_local": "london@2026-05-21T08:00:00+01:00",
                        "session_block_reason": "outside_active_session",
                    },
                    "automation_live_allowed": True,
                    "mismatch_detected": True,
                    "reconciliation_status": "mismatch_detected",
                    "orphan_orders": {
                        "T1": {
                            "broker_order_id": "T1",
                            "instrument": "XAUUSD",
                            "side": "sell",
                            "size": 0.5,
                            "reason": "unknown_to_atlas",
                            "reviewed": False,
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    payload = app.test_client().get("/api/status/compact").get_json()

    assert payload["compact"] is True
    assert payload["visual_chart_read"]["pattern_seen"] == "pullback_continuation"
    assert payload["action"] == "hold"
    assert payload["symbol"] == "XAUUSD"
    assert payload["blocker_classification"]["class"] == "signal_quality"
    assert payload["system_health"]["effective_mode_label"] == "LEARNING_AGGRESSIVE_DEMO"
    assert payload["system_health"]["session_state"] == "none"
    assert payload["system_health"]["session_state_source"] == "session_control"
    assert payload["system_health"]["broker_reconciliation_mismatch"] is True
    assert "broker_reconciliation_mismatch" in payload["system_health"]["warnings"]
    assert payload["market"]["recent_close"] == list(range(20, 100))
    assert payload["learning_critic"]["status"] == "similar_weak_fingerprint_seen"
    assert payload["learning_critic"]["previous_failure_axis"] == "timing"
    assert payload["learning_loop"]["status"] == "waiting"
    assert payload["learning_loop"]["pattern_lessons"] == 0
    assert payload["execution"]["session_control"]["next_session_local"] == "london@2026-05-21T08:00:00+01:00"
    assert payload["execution"]["auto_trade_reason"] == "outside_active_session"
    assert payload["execution"]["reconciliation_status"] == "mismatch_detected"
    assert payload["execution"]["orphan_order_count"] == 1
    assert payload["structured_blocker"]["code"] == "no_high_confidence_setup"
    assert payload["structured_blocker"]["reconciliation"]["orphan_order_count"] == 1
    assert payload["structured_blocker"]["reconciliation"]["top_orphan_orders"][0]["broker_order_id"] == "T1"
    assert payload["operator_readiness"]["lanes"]["market_quality"]["ready"] is False
    assert payload["operator_readiness"]["primary_lane"] == "market_quality"
    assert "learning_summary" not in payload


def test_compact_status_builds_learning_critic_from_loss_memory(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.0, "side": "flat"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
        }
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "pattern_lessons": {
                    "ema|buy|london|trend": {
                        "fingerprint": "ema|buy|london|trend",
                        "samples": 4,
                        "losses": 3,
                        "wins": 1,
                        "loss_rate": 0.75,
                        "post_trade_lesson": {
                            "failure_axis": "confirmation_quality",
                            "idea_quality": "reasonable",
                            "timing_quality": "early",
                            "next_time_action": "require momentum follow-through before entry",
                            "future_recognition": ["same fingerprint with weak close confirmation"],
                        },
                    }
                }
            }
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()

    assert payload["learning_critic"]["status"] == "watching_for_repeated_patterns"
    assert payload["learning_critic"]["fingerprint"] == "ema|buy|london|trend"
    assert payload["learning_critic"]["matched_fingerprint"] == "ema|buy|london|trend"
    assert payload["learning_critic"]["match_strength"] == "stored_loss_memory_watch"
    assert payload["learning_critic"]["previous_failure_axis"] == "confirmation_quality"
    assert payload["learning_critic"]["loss_rate"] == 0.75
    assert payload["loss_learning_review"]["is_pattern_ban"] is False
    assert payload["loss_learning_review"]["same_pattern_can_trade_again"] is True
    assert "does not ban patterns" in payload["loss_learning_review"]["policy_note"]


def test_loss_learning_review_explains_correction_not_pattern_ban(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "loss_to_win_playbook": {
                "status": "loss_pattern_needs_correction",
                "top_lesson": {
                    "classification": "late_entry",
                    "corrected_entry": "Skip the chase and wait for a retest, pullback, or fresh structure entry before executing.",
                    "confirmation_required": ["no_overextension", "retest_or_pullback", "fresh_entry_location"],
                    "missing": ["fresh_entry_location"],
                    "confirmed": False,
                },
            },
            "winner_playbook": {
                "top_lesson": {
                    "missing": ["strong_m1_momentum"],
                }
            },
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold",
                }
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/loss-learning-review").get_json()

    assert payload["status"] == "same_pattern_allowed_after_corrections"
    assert payload["is_pattern_ban"] is False
    assert payload["same_pattern_can_trade_again"] is True
    assert payload["bounded_caution_active"] is True
    assert payload["previous_mistake"] == "late_entry"
    assert payload["current_missing_confirmations"] == ["fresh_entry_location"]
    assert payload["winner_confirmation_missing"] == ["strong_m1_momentum"]
    assert payload["safety_lock_separate"]["active"] is True
    assert "not a permanent pattern ban" in payload["safety_lock_separate"]["note"]
    teaching_ids = [item["id"] for item in payload["durable_teachings"]["all"]]
    assert "loss_pattern_is_not_pattern_ban" in teaching_ids
    assert "safety_lock_is_separate_from_learning" in teaching_ids
    assert "do not rewrite strategies" in payload["durable_teachings"]["policy_note"].lower()
    assert payload["historical_loss_teachings"]["sample_summary"]["losses_reviewed"] >= 100
    cluster_ids = [item["id"] for item in payload["historical_loss_teachings"]["clusters"]]
    assert "broad_buy_timing_stop_hunt_cluster" in cluster_ids
    assert "false_continuation_confirmation_cluster" in cluster_ids
    assert "bounded knowledge" in payload["historical_loss_teachings"]["policy_note"].lower()


def test_learning_lanes_endpoint_summarizes_observe_classify_remember_explain_caution_and_proposals(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    knowledge_dir = tmp_path.parent / "knowledge"
    knowledge_dir.mkdir(exist_ok=True)
    (knowledge_dir / "proposal_queue.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-06-09T10:00:00+00:00",
                "proposals": [
                    {"proposal_id": "proposal_pending", "status": "pending_review"},
                    {"proposal_id": "proposal_shadow", "status": "shadow_active"},
                ],
            }
        ),
        encoding="utf-8",
    )
    store.save_status(
        {
            "updated_at": "2026-06-09T10:01:00+00:00",
            "action": "blocked",
            "message": "Loss-learning guard reduced confidence.",
            "market": {"recent_close": [1.0, 2.0, 3.0]},
            "visual_chart_read": {"available": True, "stale": False, "pattern_seen": "pullback_continuation"},
            "structured_blocker": {"code": "timing_not_ready", "class": "timing", "detail": "entry timing needs confirmation"},
            "decision_explanation": {"summary": "Timing is not clean enough yet."},
            "learning_critic": {"enabled": True, "applied": True, "status": "similar_weak_fingerprint_seen"},
            "loss_learning": {"applied": True, "penalty": 0.12},
            "loss_to_win_playbook": {
                "status": "loss_pattern_needs_correction",
                "top_lesson": {
                    "classification": "false_continuation",
                    "missing": ["strong_m1_momentum"],
                    "confirmation_required": ["strong_m1_momentum"],
                },
            },
            "timing_readiness": {"hard_block": True},
            "self_improvement_state": {"proposal_count": 2},
        }
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [{"outcome": "loss"}],
                "pattern_lessons": {"a": {"losses": 1, "samples": 1}},
                "lesson_backfill": {"rebuilt_at": "2026-06-09T09:59:00+00:00", "pattern_lessons": 1},
            }
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/learning-lanes").get_json()

    assert payload["enabled"] is True
    assert set(payload["lanes"]) == {
        "observe",
        "classify",
        "remember",
        "explain",
        "bounded_caution",
        "propose_improvements",
    }
    assert payload["lanes"]["observe"]["active"] is True
    assert payload["lanes"]["observe"]["evidence"]["visual_pattern_seen"] == "pullback_continuation"
    assert payload["lanes"]["classify"]["evidence"]["blocker_code"] == "timing_not_ready"
    assert payload["lanes"]["remember"]["evidence"]["pattern_lessons"] == 1
    assert payload["lanes"]["explain"]["evidence"]["summary"] == "Timing is not clean enough yet."
    assert payload["lanes"]["bounded_caution"]["active"] is True
    assert payload["lanes"]["bounded_caution"]["evidence"]["learning_critic_applied"] is True
    assert payload["lanes"]["propose_improvements"]["evidence"]["pending_review"] == 1
    assert payload["lanes"]["propose_improvements"]["evidence"]["active_or_rollout"] == 1
    assert "does not change strategy" in payload["policy_note"]


def test_compact_status_and_audit_include_learning_lanes_without_policy_authority(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.0, "side": "flat"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "learning_critic": {"enabled": True, "applied": False, "status": "watching"},
        }
    )
    app = create_ui_app(store)

    compact = app.test_client().get("/api/status/compact").get_json()
    audit = app.test_client().get("/api/system-audit").get_json()

    assert compact["learning_lanes"]["lanes"]["bounded_caution"]["status"] == "watching"
    assert "cannot lower thresholds" in compact["learning_lanes"]["lanes"]["bounded_caution"]["policy_note"]
    assert "review items only" in compact["learning_lanes"]["lanes"]["propose_improvements"]["policy_note"]
    assert audit["learning_lanes"]["enabled"] is True
    assert "does not change strategy" in audit["learning_lanes"]["policy_note"]


def test_cognitive_state_endpoint_presents_one_coordinated_mind_without_policy_bypass(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "updated_at": "2026-06-09T10:10:00+00:00",
            "action": "blocked",
            "message": "Safety lock active.",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "chosen": {"confidence": 0.51, "side": "buy"},
            "decision_analysis": {
                "final_trade_bias": "neutral",
                "higher_timeframe_bias": "flat",
                "setup_readiness": "forming_waiting_for_confirmations",
            },
            "visual_chart_read": {"available": True, "stale": False, "pattern_seen": "messy_range"},
            "learning_critic": {"enabled": True, "applied": False, "status": "watching_for_repeated_patterns"},
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold",
                }
            },
        }
    )
    store.save_runtime_flags({"execute_trades": True})
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [{"outcome": "loss"}, {"outcome": "win"}],
                "pattern_lessons": {"a": {"samples": 2}},
            }
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/cognitive-state").get_json()
    compact = app.test_client().get("/api/status/compact").get_json()
    audit = app.test_client().get("/api/system-audit").get_json()

    assert payload["schema"] == "atlas_cognitive_state_v1"
    assert payload["mind_model"] == "one_coordinated_mind_with_modular_specialists"
    assert payload["coordinator"]["sees"]["symbol"] == "XAUUSD"
    assert payload["learning_lanes"]["enabled"] is True
    assert payload["execution_readiness"]["ready"] is False
    assert payload["cognitive_core"]["safety_contract"]["can_bypass_execution_gate"] is False
    assert "does not merge or bypass" in payload["policy_note"]
    assert compact["cognitive_state"]["mind_model"] == "one_coordinated_mind_with_modular_specialists"
    assert audit["cognitive_state"]["schema"] == "atlas_cognitive_state_v1"


def test_safety_lock_review_explains_reasons_recovery_and_learning_separation(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Safety lock active: consecutive_loss_threshold, supporting_win_rate_weak. No trade executed.",
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold, supporting_win_rate_weak",
                    "safety_lock_started_at": "2026-06-09T10:00:00+00:00",
                    "safety_lock_scope": "new_entries",
                    "loss_streak": 4,
                    "supporting_win_rate": 0.25,
                    "rolling_drawdown_pct": 0.004,
                    "rolling_expectancy_zar": -22.5,
                    "dominant_loss_class": "timing",
                    "volatility_state": "normal",
                    "recovery_mode": "lockdown",
                    "recovery_ready": False,
                    "recovery_blockers": ["safety_lock_active"],
                    "recovery_requirements": [
                        "manual_operator_unlock_required",
                        "cooldown_expired",
                        "no_active_hard_blocker",
                        "volatility_back_to_acceptable",
                        "next_setup_meets_high_confluence",
                    ],
                }
            },
        }
    )
    app = create_ui_app(store)

    direct = app.test_client().get("/api/safety-lock-review").get_json()
    compact = app.test_client().get("/api/status/compact").get_json()
    audit = app.test_client().get("/api/system-audit").get_json()

    assert direct["active"] is True
    assert direct["status"] == "locked"
    assert direct["scope"] == "new_entries"
    assert direct["metrics"]["loss_streak"] == 4
    assert direct["metrics"]["supporting_win_rate"] == 0.25
    assert [item["code"] for item in direct["evidence"][:2]] == [
        "consecutive_loss_threshold",
        "supporting_win_rate_weak",
    ]
    assert any(item["code"] == "manual_operator_unlock_required" for item in direct["recovery_checklist"])
    assert direct["separate_from_learning"]["pattern_ban"] is False
    assert "does not permanently ban" in direct["separate_from_learning"]["note"]
    assert "does not unlock safety" in direct["policy_note"]
    assert compact["safety_lock_review"]["active"] is True
    assert audit["safety_lock_review"]["status"] == "locked"


def test_safety_lock_review_marks_missing_metrics_without_hiding_recovery_path(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Safety lock active: consecutive_loss_threshold. No trade executed.",
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold, supporting_win_rate_weak",
                    "recovery_blockers": ["safety_lock_active"],
                    "recovery_ready": False,
                }
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/safety-lock-review").get_json()

    assert payload["evidence"][0]["value"] == "not_reported"
    assert payload["evidence"][1]["value"] == "not_reported"
    assert payload["metrics"]["metric_availability"]["loss_streak"] is False
    assert any(item["code"] == "manual_operator_unlock_required" for item in payload["recovery_checklist"])
    assert any(item["code"] == "next_setup_meets_high_confluence" for item in payload["recovery_checklist"])


def test_safety_lock_review_merges_health_summary_with_raw_safety_metrics(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Safety lock active: consecutive_loss_threshold. No trade executed.",
            "safety_layer": {
                "safety_lock_active": True,
                "safety_lock_reason": "consecutive_loss_threshold, supporting_win_rate_weak",
                "loss_streak": 4,
                "supporting_win_rate": 0.25,
                "rolling_drawdown_pct": 0.006,
                "rolling_expectancy_zar": -35.5,
                "dominant_loss_class": "timing",
                "recovery_requirements": ["manual_operator_unlock_required", "next_setup_meets_high_confluence"],
            },
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold, supporting_win_rate_weak",
                    "recovery_ready": False,
                    "recovery_blockers": ["safety_lock_active"],
                }
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/safety-lock-review").get_json()

    assert payload["metrics"]["loss_streak"] == 4
    assert payload["metrics"]["supporting_win_rate"] == 0.25
    assert payload["metrics"]["rolling_expectancy_zar"] == -35.5
    assert payload["metrics"]["metric_availability"]["loss_streak"] is True
    assert payload["evidence"][0]["value"] == 4
    assert payload["evidence"][1]["value"] == 0.25
    assert any(item["code"] == "next_setup_meets_high_confluence" for item in payload["recovery_checklist"])


def test_volatility_protection_review_explains_extreme_volatility_hard_block(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True, "frontend_mode": "desktop"},
            "visual_watch": {"enabled": True, "last_capture_ok": True},
        }
    )
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Volatility protection active: extreme volatility regime; new entries paused.",
            "execute_trades": True,
            "system_health": {
                "broker_connectivity_status": {"connected": True, "symbol_selected": True},
                "safety_layer": {
                    "safety_lock_active": False,
                    "volatility_state": "extreme",
                    "volatility_action": "block_new_entries",
                    "volatility_reason": "extreme volatility regime; new entries paused",
                    "recovery_blockers": ["extreme_volatility"],
                },
                "spread_liquidity": {"spread_status": "normal", "spread_points": 34.0, "liquidity_state": "normal"},
            },
            "decision_analysis": {
                "market_regime": {
                    "regime": "volatile",
                    "regime_confidence": 52,
                    "regime_reasons": ["candle expansion / volatility is elevated"],
                }
            },
        }
    )
    app = create_ui_app(store)

    review = app.test_client().get("/api/volatility-protection-review").get_json()
    compact = app.test_client().get("/api/status/compact").get_json()
    audit = app.test_client().get("/api/system-audit").get_json()

    assert review["status"] == "hard_block"
    assert review["hard_block"] is True
    assert review["state"] == "extreme"
    assert review["recovery_requirements"][0]["code"] == "volatility_back_to_acceptable"
    assert compact["volatility_protection_review"]["hard_block"] is True
    assert audit["summary"]["volatility_hard_block"] is True
    assert any(item["code"] == "volatility_hard_block" for item in audit["issues"])
    assert "read-only observability" in review["policy_note"]


def test_compact_status_exposes_learning_loop_status(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.0, "side": "flat"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
        }
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [{"outcome": "loss"}],
                "pattern_lessons": {"a": {"losses": 1, "samples": 1}},
                "lesson_backfill": {
                    "rebuilt_at": "2026-05-26T07:17:42+00:00",
                    "recent_outcomes": 200,
                    "pattern_lessons": 20,
                    "loss_dominant_lessons": 17,
                    "decision_context": {
                        "decision_samples_scanned": 128,
                        "outcomes_scanned": 200,
                        "outcomes_linked": 16,
                        "link_rate": 0.08,
                    },
                    "context_quality": {
                        "location_known": 0,
                        "momentum_known": 16,
                        "volatility_known": 16,
                    },
                },
            }
        }
    )
    store.log_event("learning", "Closed-trade learning summary updated.", {"processed_now": 1})
    store.log_event("learning", "Loss-pattern lesson memory refreshed after closed trade.", {"pattern_lessons": 20})
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    loop = payload["learning_loop"]

    assert loop["status"] == "active"
    assert loop["last_closed_trade_learning_at"]
    assert loop["last_lesson_refresh_at"] == "2026-05-26T07:17:42+00:00"
    assert loop["recent_outcomes"] == 200
    assert loop["pattern_lessons"] == 20
    assert loop["loss_dominant_lessons"] == 17
    assert loop["decision_context"]["outcomes_linked"] == 16
    assert loop["context_quality"]["momentum_known"] == 16


def test_learning_loop_derives_context_quality_when_backfill_summary_missing(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.0, "side": "flat"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
        }
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [{"outcome": "loss"}, {"outcome": "win"}],
                "pattern_lessons": {
                    "a": {
                        "losses": 1,
                        "samples": 1,
                        "context_quality": {
                            "decision_context_linked": True,
                            "momentum_known": True,
                            "volatility_known": True,
                            "location_known": False,
                        },
                    }
                },
            }
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    loop = payload["learning_loop"]

    assert loop["pattern_lessons"] == 1
    assert loop["decision_context"]["outcomes_scanned"] == 2
    assert loop["decision_context"]["outcomes_linked"] == 1
    assert loop["context_quality"]["momentum_known"] == 1
    assert loop["context_quality"]["volatility_known"] == 1


def test_learning_loop_event_scan_uses_bounded_tail(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.0, "side": "flat"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
        }
    )
    store.save_learning_state({"loss_memory": {"recent_outcomes": [], "pattern_lessons": {}}})
    with store.events_file.open("w", encoding="utf-8") as fh:
        fh.write("x" * 900_000)
        fh.write("\n")
        fh.write(json.dumps({"time": "2026-05-26T07:00:00+00:00", "type": "learning", "message": "Loss-pattern lesson memory refreshed after closed trade.", "meta": {}}))
        fh.write("\n")
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()

    assert payload["learning_loop"]["status"] == "active"


def test_compact_status_exposes_loss_pattern_recovery(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "M1 loss-pattern block: repeated buy losses (wr=0.33, streak=3).",
            "chosen": {"confidence": 0.45, "side": "buy"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "loss_learning": {
                "hard_blocked": True,
                "hard_block": {
                    "blocked": True,
                    "code": "m1_loss_pattern_block",
                    "trigger": "repeated_side_losses",
                    "reason": "M1 loss-pattern block: repeated buy losses (wr=0.33, streak=3).",
                    "side": "buy",
                    "scope": "m1_side_memory",
                    "samples": 6,
                    "wins": 2,
                    "losses": 4,
                    "win_rate": 0.3333,
                    "loss_streak": 3,
                    "hour_utc": 10,
                    "hour_context": {"key": "buy:10", "samples": 6, "wins": 1, "losses": 5, "win_rate": 0.1667},
                    "thresholds": {"min_samples": 6, "max_win_rate": 0.34, "min_loss_streak": 3},
                    "clears_when": "same-side win rate improves, the loss streak breaks, or hour context recovers",
                    "evidence_required": ["fresh M1 outcomes", "loss streak reset"],
                    "recovery_checklist": [
                        {
                            "code": "loss_streak_breaks",
                            "label": "A same-side or opposite-side win breaks the immediate loss streak.",
                            "status": "blocked",
                            "detail": "current_streak=3, required_below=3",
                        }
                    ],
                },
            },
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    block = payload["loss_pattern_block"]

    assert block["blocked"] is True
    assert block["trigger"] == "repeated_side_losses"
    assert block["side"] == "buy"
    assert block["loss_streak"] == 3
    assert block["hour_context"]["key"] == "buy:10"
    assert block["recovery_checklist"][0]["code"] == "loss_streak_breaks"


def test_compact_status_exposes_soft_loss_learning_recovery(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Loss-learning guard reduced confidence.",
            "chosen": {"confidence": 0.42, "side": "buy"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "loss_learning": {
                "applied": True,
                "penalty": 0.13,
                "reason": "Recent buy penalty 0.130 (live, wr=0.25, loss_streak=3)",
                "clears_when": "loss streak breaks",
                "evidence_required": ["fresh win", "cleaner setup"],
                "details": {
                    "recent_side": {
                        "applied": True,
                        "source": "live",
                        "side": "buy",
                        "scope": "recent_side_memory",
                        "samples": 8,
                        "wins": 2,
                        "losses": 6,
                        "win_rate": 0.25,
                        "loss_streak": 3,
                    }
                },
            },
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    block = payload["loss_pattern_block"]

    assert block["blocked"] is False
    assert block["soft_penalty_active"] is True
    assert block["code"] == "loss_learning_penalty"
    assert block["side"] == "buy"
    assert block["penalty"] == 0.13
    assert block["loss_streak"] == 3
    assert block["recovery_checklist"][0]["code"] == "loss_streak_breaks"
    assert payload["recovery_status"]["category"] == "loss_learning"


def test_compact_status_exposes_consensus_recovery(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "chosen": {"confidence": 0.45, "side": "buy"},
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "same_side_support": {"side": "buy", "count": 1, "active_count": 1, "raw_count": 1, "source": "active_signals"},
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()
    recovery = payload["recovery_status"]

    assert payload["same_side_support"]["count"] == 1
    assert recovery["category"] == "consensus"
    assert recovery["blocked"] is True
    assert recovery["primary_metric"]["value"] == "1/2"
    assert recovery["checklist"][0]["status"] == "blocked"


def test_automation_truth_hides_stale_block_reason_when_resumed(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "chosen": {"confidence": 0.0, "side": "flat"},
        }
    )
    app = create_ui_app(store)
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_profiles": {"p1": {"enabled": True}},
                    "active_automation_policy_id": "p1",
                    "automation_paused": False,
                    "automation_explicit_resume_active": True,
                    "last_automation_block_reason": "simulation_market_state_cannot_execute",
                    "last_automation_event": {
                        "ok": False,
                        "reason": "simulation_market_state_cannot_execute",
                        "ts": time.time() - 7200.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    payload = app.test_client().get("/api/status/compact").get_json()

    assert payload["automation_truth"]["automation_enabled"] is True
    assert payload["automation_truth"]["last_auto_execution_block_reason"] == ""
    assert payload["automation_truth"]["last_auto_execution_block_stale"] is True
    assert payload["execution"]["auto_trade_allowed"] is None
    assert payload["execution"]["auto_trade_status_known"] is False
    assert payload["execution_readiness"]["state"]["auto_trade_status_known"] is False


def test_compact_status_distinguishes_flat_direction_from_low_confidence(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "chosen": {"confidence": 0.45, "side": "flat"},
            "min_confidence_threshold": 0.44,
            "risk": {"equity": 100.0, "balance": 100.0, "open_positions": 0, "daily_loss_pct": 0.0},
            "blocker_classification": {
                "class": "signal_quality",
                "primary": "No directional trade signal selected.",
                "recoverability": "wait_for_buy_or_sell_signal_confirmation",
            },
        }
    )
    app = create_ui_app(store)

    payload = app.test_client().get("/api/status/compact").get_json()

    assert payload["structured_blocker"]["code"] == "no_directional_trade_signal"
    assert payload["structured_blocker"]["recoverability"] == "wait_for_buy_or_sell_signal_confirmation"


def test_status_reconciles_desktop_started_after_transient_failure(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    monkeypatch.setattr("atlas_ai.ui_server._process_exists", lambda pid: pid == 1234)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "ui_startup_diagnostics.log").write_text(
        "\n".join(
            [
                "2026-05-20T17:42:10+00:00 | ok=False | stage=start_loop | type= | class=generic_desktop_init_failure | error=",
                "2026-05-20T17:42:13+00:00 | ok=True | stage=started | type= | class= | error=",
            ]
        ),
        encoding="utf-8",
    )
    store.save_runtime_flags(
        {
            "ui_desktop_window": {
                "wrapper_pid": 1234,
                "ok": False,
                "ready": False,
                "desktop_window_ready": False,
                "frontend_mode": "desktop_unavailable",
                "failure_reason": "generic_desktop_init_failure",
                "stage": "start_loop",
                "updated_at": "2026-05-20T17:42:10+00:00",
            }
        }
    )

    payload = create_ui_app(store).test_client().get("/api/status").get_json()

    assert payload["ui_desktop_window_ready"] is True
    assert payload["ui_desktop_window"]["frontend_mode"] == "desktop"
    assert payload["ui_desktop_window"]["failure_reason"] == ""
    assert payload["operator_readiness"]["lanes"]["operator_ui"]["ready"] is True
    assert payload["operator_readiness"]["lanes"]["operator_ui"]["desktop"]["failure_reason"] == ""
    assert payload["operator_health"]["desktop_window_ready"] is True
    assert payload["operator_health"]["degraded_reason"] == ""
    assert payload["ui_desktop_window"]["failure_evidence"]["desktop_started_reconciled_from_diagnostics"].startswith("2026-05-20T17:42:13")


def test_blocker_summary_api_returns_compact_observability_payload(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "blocker_class": "signal_quality",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.41},
            "thresholds": {"confidence": 0.44},
            "blocker_classification": {
                "class": "signal_quality",
                "primary": "No high-confidence trade found.",
                "recoverability": "wait_for_setup_or_review_signal_fit",
            },
            "decision_analysis": {
                "hard_blockers": ["no_clear_setup"],
                "top_blockers": [{"name": "setup_clarity", "actual": 40, "target": 60}],
                "setup_radar": {
                    "timing_state": "wait",
                    "top_watch": {"name": "pullback_continuation", "missing_to_entry": ["pullback_location"]},
                    "conditions_slowing_setups": ["momentum_mixed"],
                    "operator_summary": "Closest setup is still forming.",
                },
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/blocker-summary").get_json()

    assert payload["action"] == "hold"
    assert payload["blocker_class"] == "signal_quality"
    assert payload["primary"] == "No high-confidence trade found."
    assert payload["confidence_gap"] == -0.03
    assert payload["hard_blockers"] == ["no_clear_setup"]
    assert payload["setup_radar"]["waiting_for"] == ["pullback_location"]
    assert payload["structured_blocker"]["code"] == "no_high_confidence_setup"
    assert "observability only" in payload["policy_note"]


def test_compact_status_includes_setup_hunter(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "thresholds": {"confidence": 0.44},
            "signals": [{"strategy": "ema_crossover", "side": "buy", "confidence": 0.18}],
            "decision_analysis": {
                "setup_radar": {
                    "watchlist": [
                        {
                            "name": "pullback_continuation",
                            "side": "buy",
                            "score": 64,
                            "phase": "forming_confirmations_needed",
                            "missing_to_entry": ["pullback_entry_location"],
                        }
                    ]
                }
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()

    assert payload["setup_hunter"]["mode"] == "proactive_scan"
    assert payload["setup_hunter"]["state"] == "forming"
    assert payload["setup_hunter"]["closest_setup"]["name"] == "pullback_continuation"


def test_compact_status_explains_setup_hunter_candidate_not_execution_signal(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True, "frontend_mode": "desktop"},
        }
    )
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "chosen": {"side": "flat", "confidence": 0.0},
            "thresholds": {"confidence": 0.44},
            "system_health": {"broker_connectivity_status": {"connected": True, "symbol_selected": True}},
            "setup_hunter": {
                "mode": "proactive_scan",
                "state": "candidate_context_blocked",
                "candidate_count": 1,
                "closest_setup": {
                    "name": "range_edge_reversal",
                    "side": "sell",
                    "confidence": 0.82,
                    "score": 82,
                    "readiness": "ready_for_existing_gates",
                    "phase": "candidate_context_blocked",
                },
                "top_candidates": [],
                "hard_blockers": ["no_clear_setup"],
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()

    handoff = payload["setup_execution_handoff"]
    assert handoff["status"] == "watched_candidate_not_selected_signal"
    assert handoff["candidate_exists"] is True
    assert handoff["selected_directional_signal"] is False
    assert handoff["candidate_is_execution_signal"] is False
    assert payload["execution_readiness"]["ready"] is False
    assert "does not create signals" in handoff["policy_note"]


def test_signal_selection_review_explains_raw_candidate_not_selected(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True, "frontend_mode": "desktop"},
        }
    )
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "chosen": {"side": "flat", "confidence": 0.45, "strategy": "momentum_continuation"},
            "thresholds": {"confidence": 0.44},
            "signals": [{"strategy": "ema_crossover", "side": "sell", "confidence": 0.78}],
            "system_health": {"broker_connectivity_status": {"connected": True, "symbol_selected": True}},
            "decision_analysis": {
                "hard_blockers": ["no_clear_setup"],
                "confluence": {"score": 45, "threshold": 52, "qualified": False},
                "top_blockers": [{"name": "momentum_quality", "actual": 50, "target": 65}],
                "setup_radar": {
                    "top_watch": {
                        "name": "pullback_continuation",
                        "side": "sell",
                        "missing_to_entry": ["pullback_entry_location", "momentum_confirmation"],
                    }
                },
            },
        }
    )
    app = create_ui_app(store)

    compact = app.test_client().get("/api/status/compact").get_json()
    review = app.test_client().get("/api/signal-selection-review").get_json()

    assert compact["signal_selection_review"]["status"] == "raw_candidate_not_selected"
    assert review["status"] == "raw_candidate_not_selected"
    assert review["selected_signal"]["side"] == "flat"
    assert review["candidate"]["name"] == "ema_crossover"
    assert "chosen signal is flat" in review["selection_reasons"]
    assert "no_clear_setup" in review["selection_reasons"]
    assert "pullback_entry_location" in review["missing_confirmations"]
    assert "read-only observability" in review["policy_note"]


def test_compact_status_includes_decision_panels_payload(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "decision_explanation": {
                "candidate_id": "candidate-1",
                "final_decision": "hold",
                "explanation_text": "No valid setup is confirmed yet.",
                "reason_codes": ["no_clear_setup"],
            },
            "decision_trace": {
                "schema": "atlas_decision_trace_v1",
                "final_action": "hold",
                "final_decision": "block",
                "message": "No directional trade signal selected.",
                "selected_candidate": {"strategy": "momentum_continuation", "side": "flat", "confidence": 0.0},
                "context": {"session": "london", "regime": "trend", "spread_state": "normal"},
                "filters": {"hard_blockers": ["no_clear_setup"]},
                "risk": {"spread_points": 12.0},
                "execution_gate": {"execute_trades": False},
                "signal": {
                    "active": [{"strategy": "ema_crossover", "side": "buy", "confidence": 0.12}],
                    "active_count": 1,
                    "raw_count": 1,
                },
                "blocker_summary": {"class": "signal_quality", "primary": "No clear setup."},
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()

    assert payload["decision_explanation"]["candidate_id"] == "candidate-1"
    assert payload["decision_trace"]["schema"] == "atlas_decision_trace_v1"
    assert payload["decision_trace"]["selected_candidate"]["strategy"] == "momentum_continuation"
    assert payload["decision_trace"]["signal"]["active"][0]["strategy"] == "ema_crossover"


def test_compact_status_includes_decision_analysis_panel_fields(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "message": "No clear setup.",
            "decision_analysis": {
                "confidence_score": 57,
                "confidence_level": "medium",
                "confluence": {"score": 57, "threshold": 52},
                "trend": {"state": "range", "structure": "mixed"},
                "momentum": {"state": "mixed"},
                "location": {"state": "consolidation"},
                "setup_type": "consolidation",
                "market_regime": {"regime": "ranging"},
                "reversal_detection": {"reversal_candidate": True, "confirmed_reversal": False},
                "context_alignment": {"session": {"state_class": "acceptable"}},
                "account_profile": {
                    "account_tier": "standard_account",
                    "risk_profile": "capital_preservation_standard",
                    "current_size_tier": "tier_5_0p10",
                },
                "entry_quality": "risky",
                "reasoning": ["range-edge candidate is forming"],
                "warnings": ["momentum_mixed"],
                "decision_protocol": {
                    "observe": {"summary": ["observe summary"]},
                    "orient": {"summary": ["orient summary"]},
                    "critique": {"top_failure_reasons": ["momentum mixed"]},
                    "decide": {"confluence_score": 57, "confluence_threshold": 52, "recommended_action": "hold"},
                    "act": {"execution_request_eligible": False},
                },
                "macro_context": {"operator_terminal": {"reads": ["DXY=mixed"]}},
                "trading_response": {"size_band_rationale": {"band": "blocked", "cap_multiplier": 0.0}},
                "timing_engine": {"state": "setup_forming", "waiting_for": ["momentum_confirmation"]},
                "setup_radar": {"watchlist": [{"name": "range_edge_reversal", "score": 78}]},
                "explanation": "Setup is developing.",
            },
            "trader_cognition": {
                "schema": "atlas_trader_cognition_v1",
                "mode": "bounded_trader_like_reasoning",
                "observation": "new_york session, range.",
                "thesis": "No directional thesis is strong enough yet.",
                "counter_thesis": "momentum is mixed",
                "discipline_check": {"status": "disciplined", "checks": ["wait for cleaner momentum"]},
                "service_intent": {"statement": "Serve the operator by pursuing profitable XAUUSD trading decisions.", "real_feelings": False},
                "decision": "wait",
                "confidence": 0.57,
                "threshold": 0.52,
                "required_next_evidence": ["momentum_confirmation"],
                "autonomy_boundary": {"execution_authority": "ExecutionGate", "does_not_change_risk_policy": True},
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()
    analysis = payload["decision_analysis"]

    assert analysis["trend"]["state"] == "range"
    assert analysis["momentum"]["state"] == "mixed"
    assert analysis["decision_protocol"]["observe"]["summary"] == ["observe summary"]
    assert analysis["macro_context"]["operator_terminal"]["reads"] == ["DXY=mixed"]
    assert analysis["timing_engine"]["state"] == "setup_forming"
    cognition = payload["trader_cognition"]
    assert cognition["schema"] == "atlas_trader_cognition_v1"
    assert cognition["service_intent"]["real_feelings"] is False
    assert cognition["autonomy_boundary"]["execution_authority"] == "ExecutionGate"


def test_operator_readiness_separates_market_from_system_blockers(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
            "blocker_classification": {
                "class": "quality_gate",
                "primary": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
            },
            "system_health": {"broker_connectivity_status": {"connected": True, "symbol_selected": True}},
        }
    )
    store.save_runtime_flags({"ui_desktop_window": {"ok": True, "ready": True}, "execute_trades": True})
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_profiles": {"p1": {"id": "p1", "enabled": True, "environment": "live"}},
                    "active_automation_policy_id": "p1",
                    "automation_paused": False,
                    "session_control": {"session_active": True, "current_session": "london"},
                }
            }
        ),
        encoding="utf-8",
    )

    app = create_ui_app(store)
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_profiles": {"p1": {"id": "p1", "enabled": True, "environment": "live"}},
                    "active_automation_policy_id": "p1",
                    "automation_paused": False,
                    "session_control": {"session_active": True, "current_session": "london"},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = app.test_client().get("/api/operator-readiness").get_json()

    assert payload["ready_for_execution_attempt"] is False
    assert payload["primary_lane"] == "market_quality"
    assert payload["lanes"]["automation"]["ready"] is True
    assert payload["lanes"]["session"]["ready"] is True
    assert payload["lanes"]["broker"]["ready"] is True
    assert payload["lanes"]["market_quality"]["blocker"]["code"] == "htf_alignment_conflict"


def test_current_teaching_lesson_structures_htf_conflict(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
            "chosen": {"side": "buy"},
            "blocker_classification": {
                "class": "quality_gate",
                "primary": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
            },
            "decision_analysis": {
                "bias_breakdown": {"higher_timeframe_bias": "sell", "final_trade_bias": "neutral"},
                "setup_radar": {"top_watch": {"name": "pullback_continuation", "trigger_state": "forming_waiting_for_confirmations"}},
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/teaching/current-lesson").get_json()

    assert "htf_bias=sell" in payload["condition"]
    assert payload["confidence_effect"] == "reduce_or_hold_confidence_until_alignment"
    assert "HTF bias" in payload["evidence_required"][0]


def test_operator_readiness_does_not_treat_startup_connected_as_market_blocker(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status({"action": "connected", "message": "Broker connected. Live loop started. Historical training disabled in live runtime."})
    store.save_runtime_flags({"ui_desktop_window": {"ok": True, "ready": True}, "execute_trades": True})
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_profiles": {"p1": {"id": "p1", "enabled": True, "environment": "live"}},
                    "active_automation_policy_id": "p1",
                    "automation_paused": False,
                    "session_control": {"session_active": False, "session_block_reason": "outside_active_session"},
                }
            }
        ),
        encoding="utf-8",
    )

    app = create_ui_app(store)
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_profiles": {"p1": {"id": "p1", "enabled": True, "environment": "live"}},
                    "active_automation_policy_id": "p1",
                    "automation_paused": False,
                    "session_control": {"session_active": False, "session_block_reason": "outside_active_session"},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = app.test_client().get("/api/operator-readiness").get_json()

    assert payload["primary_lane"] == "session"
    assert payload["lanes"]["broker"]["ready"] is True
    assert all(row["lane"] != "market_quality" for row in payload["blockers"])


def test_structured_blocker_normalizes_loss_learning_guard(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "blocked",
            "message": "Loss-learning guard reduced confidence. Loss memory penalty 0.150.",
            "loss_learning": {
                "applied": True,
                "penalty": 0.15,
                "evidence_required": ["fresh live outcomes"],
            },
            "blocker_classification": {
                "class": "quality_gate",
                "primary": "Loss-learning guard reduced confidence. Loss memory penalty 0.150.",
            },
        }
    )
    store.save_runtime_flags({"ui_desktop_window": {"ok": True, "ready": True}})

    payload = create_ui_app(store).test_client().get("/api/status/compact").get_json()

    assert payload["structured_blocker"]["code"] == "loss_pattern_memory"
    assert payload["loss_learning"]["penalty"] == 0.15


def test_visual_latest_endpoint_requires_live_terminal_capture(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "chart_features": {
                "trend_state": "downtrend",
                "trend_structure": "LH/LL",
                "momentum_state": "strong",
                "location_state": "pullback",
                "setup_type_hint": "pullback",
            },
        }
    )

    payload = create_ui_app(store).test_client().get("/api/vision/latest").get_json()

    assert payload["advisory_only"] is True
    assert payload["available"] is False
    assert payload["source"] == "live_terminal_required"
    assert payload["status"] == "waiting_for_live_terminal_capture"


def test_setup_radar_history_api_exposes_near_setup_learning(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    monitoring = tmp_path / "monitoring"
    monitoring.mkdir()
    (monitoring / "atlas_near_setup_learning.json").write_text(
        json.dumps(
            {
                "mode": "observability_only",
                "updated_at": "2026-05-20T10:00:00+00:00",
                "total_samples": 12,
                "top_patterns": [
                    {
                        "fingerprint": "momentum=mixed|location=mid_range",
                        "count": 7,
                        "last_seen": "2026-05-20T10:00:00+00:00",
                        "teaching_focus": ["wait for cleaner entry", "require confirmation"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = create_ui_app(store).test_client().get("/api/setup-radar/history").get_json()

    assert payload["mode"] == "observability_only"
    assert payload["total_samples"] == 12
    assert payload["top_patterns"][0]["count"] == 7
    assert payload["top_patterns"][0]["teaching_focus"] == ["wait for cleaner entry", "require confirmation"]


def test_status_api_includes_decision_explanation_when_present(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "decision_explanation": {
                "candidate_id": "cand-123",
                "trade_id": "trade-123",
                "final_decision": "block",
                "explanation_text": "Blocked due to poor liquidity.",
                "reason_codes": ["poor_liquidity"],
                "setup_quality_reasons": ["weak_consensus"],
                "active_guards": ["conservative_pretrade_filter"],
                "penalties": ["low_volume"],
                "symbol": "XAUUSD",
                "regime": "trend",
                "session": "late_us",
                "trend_strength": 0.21,
                "signal_source": "momentum_continuation",
                "timestamp": "2026-04-04T12:00:00+00:00",
                "lane": "control",
                "mode": "paper",
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["decision_explanation"]["candidate_id"] == "cand-123"
    assert payload["decision_explanation"]["trade_id"] == "trade-123"
    assert payload["decision_explanation"]["final_decision"] == "block"
    assert payload["decision_explanation"]["signal_source"] == "momentum_continuation"


def test_learning_insights_exposes_last_confirmed_trade(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "ticket": "T1",
                        "decision_id": "D1",
                        "strategy": "evolved_rule",
                        "side": "buy",
                        "session": "new_york",
                        "regime": "trend",
                        "higher_tf_bias": "buy",
                        "htf_aligned": True,
                        "confidence": 0.56,
                        "trend_strength": 0.21,
                        "net_pnl_zar": -16.87,
                        "outcome": "loss",
                    }
                ]
            }
        }
    )
    app = create_ui_app(store)
    payload = app.test_client().get("/api/learning-insights").get_json()
    last = payload["last_confirmed_trade"]
    assert last["available"] is True
    assert last["ticket"] == "T1"
    assert last["outcome"] == "loss"
    assert "similar setups should require stronger trend" in last["lesson"]


def test_teaching_review_endpoint_records_bounded_operator_label(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(
        {
            "action": "hold",
            "primary_blocker": "No high-confidence trade found.",
            "setup_family": "no_trade",
            "decision_explanation": {"candidate_id": "cand-teach-1"},
            "decision_analysis": {
                "trade_decision": "NO_TRADE",
                "recommended_action": "hold",
                "confluence_score": 42,
                "confluence_threshold": 60,
                "trend": {"state": "downtrend", "structure": "LH/LL"},
                "momentum": {"state": "mixed"},
                "location": {"state": "mid_range"},
                "higher_timeframe_bias": "sell",
                "final_trade_bias": "neutral",
                "hard_blockers": ["no_clear_setup"],
            },
            "setup_playbook": {
                "top_candidate": {
                    "name": "range_edge_reversal",
                    "score": 74,
                    "readiness": "developing",
                    "missing": ["range_edge_location"],
                }
            },
            "visual_chart_read": {
                "source": "screenshot_plus_chart_features",
                "visual_confirmation": "does_not_confirm",
                "pattern_seen": "range_reaction",
                "setup_quality": 0.42,
                "capture": {"ok": True, "capture_source": "mt5_window"},
            },
            "safety_layer": {"volatility_state": "normal"},
            "risk": {"spread_points": 45.0, "open_positions": 0},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/teaching/reviews", json={"label": "correct_hold", "note": "Good patience."})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    review = payload["review"]
    assert review["label"] == "correct_hold"
    assert review["decision_id"] == "cand-teach-1"
    assert review["playbook_name"] == "range_edge_reversal"
    assert review["decision_context_snapshot"]["location_state"] == "mid_range"
    assert review["decision_context_snapshot"]["momentum_state"] == "mixed"
    assert review["decision_context_snapshot"]["visual_capture_source"] == "mt5_window"
    assert review["context_quality"]["decision_context_linked"] is True
    assert review["context_quality"]["location_known"] is True
    assert review["context_quality"]["momentum_known"] is True
    assert review["context_quality"]["volatility_known"] is True
    assert "No automatic strategy" in review["policy_note"]

    listed = client.get("/api/teaching/reviews").get_json()
    assert listed["count"] == 1
    assert listed["reviews"][0]["label"] == "correct_hold"
    assert listed["digest"]["label_counts"]["correct_hold"] == 1

    digest = client.get("/api/teaching/digest").get_json()
    assert digest["sample_count"] == 1
    assert digest["context_quality"]["decision_context_linked"] == 1
    assert digest["context_quality"]["visual_context_known"] == 1
    assert digest["current_decision_prompt"]["playbook"] == "range_edge_reversal"


def test_teaching_review_endpoint_rejects_unknown_label(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    app = create_ui_app(store)
    response = app.test_client().post("/api/teaching/reviews", json={"label": "change_risk_rules"})
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_label"


def test_diagnostics_api_handles_missing_decision_explanation_safely(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "idle", "message": "Waiting for first cycle."})
    app = create_ui_app(store)
    client = app.test_client()

    response = client.get("/api/diagnostics")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["decision_explanation"] == {}
    assert payload["decision_trace"] == {}
    assert payload["context_snapshot"] == {}
    assert payload["system_health"] == {}
    assert payload["shadow_advice"] == {}


def test_diagnostics_api_includes_context_snapshot_when_present(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "context_snapshot": {
                "symbol": "XAUUSD",
                "timestamp": "2026-04-04T11:59:00+00:00",
                "regime": "trend",
                "session": "london",
                "trend_strength": 0.24,
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "normal",
                "repeated_entry_risk": False,
                "overstretched_entry_risk": True,
            }
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.get("/api/diagnostics")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["context_snapshot"]["regime"] == "trend"
    assert payload["context_snapshot"]["overstretched_entry_risk"] is True


def test_diagnostics_api_includes_shadow_advice_when_present(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "shadow_advice": {
                "trade_id": "cand-123",
                "setup_score": -0.18,
                "confidence": 0.31,
                "anomaly_flags": ["poor_liquidity", "blocked_trade"],
                "alternative_ranking": [
                    {"strategy": "momentum_continuation", "score": 0.22, "confidence": 0.31}
                ],
                "recommendation": "observe",
            }
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.get("/api/diagnostics")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["shadow_advice"]["trade_id"] == "cand-123"
    assert payload["shadow_advice"]["recommendation"] == "observe"


def test_status_api_includes_system_health_when_present(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "paper",
            "message": "Trade candidate found, but EXECUTE_TRADES is disabled.",
            "system_health": {
                "effective_mode": "paper",
                "paper_mode_confirmed": True,
                "filter_state": {"conservative_filter_enabled": True},
                "ml_state": {"meta_model_enabled": False, "decision_authority": "disabled"},
                "gemini_state": {"enabled": True, "influence_enabled": False},
                "stale_data_detected": False,
                "config_mismatch_detected": False,
                "routing_risk_detected": False,
                "broker_connectivity_status": {"connected": True, "symbol_selected": True},
                "warnings": ["stale_data_detected"],
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["system_health"]["effective_mode"] == "paper"
    assert payload["system_health"]["warnings"] == ["stale_data_detected"]


def test_ui_assets_expose_decision_support_fields(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    index_html = client.get("/").data.decode("utf-8")
    app_js = client.get("/app.js").data.decode("utf-8")

    assert "decision-candidate-id" in index_html
    assert "decision-trade-id" in index_html
    assert "decision-final-decision" in index_html
    assert "decision-symbol" in index_html
    assert "decision-explanation-text" in index_html
    assert "decision-reason-codes" in index_html
    assert "decision-setup-quality" in index_html
    assert "decision-active-guards" in index_html
    assert "decision-penalties" in index_html
    assert "ask-atlas-form" in index_html
    assert "ask-atlas-input" in index_html
    assert "ask-atlas-response" in index_html
    assert "ask-atlas-history" in index_html
    assert "ask-primary-label" in index_html
    assert "ask-primary-copy" in index_html
    assert "health-effective-mode" in index_html
    assert "stat-freshness" in index_html
    assert "health-warnings" in index_html
    assert "macro-terminal-panel" in index_html
    assert "macro-dxy" in index_html
    assert "macro-us10y" in index_html
    assert "macro-vix" in index_html
    assert "macro-size-band" in index_html
    assert "da-confluence-score" in index_html
    assert "da-session-quality" in index_html
    assert "da-account-tier" in index_html
    assert "da-risk-profile" in index_html
    assert "da-size-tier" in index_html
    assert "da-recommended-action" in index_html
    assert "da-blockers" in index_html
    assert "health-safety-lock" in index_html
    assert "health-cooldown" in index_html
    assert "health-volatility" in index_html
    assert "health-recovery-mode" in index_html
    assert "context-vix-regime" in index_html
    assert "context-slippage-state" in index_html


def test_runtime_api_exposes_canonical_freshness_contract(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "idle", "message": "ok"})
    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    assert "freshness_runtime" in payload
    assert "status_age_ms" in payload["freshness_runtime"]
    assert "state" in payload["freshness_runtime"]
    assert "aging_threshold_ms" in payload["freshness_runtime"]
    assert "primary_age_ms" in payload["freshness_runtime"]
    assert "is_stale" in payload["freshness_runtime"]
    assert "max_freshness_age_ms" in payload["freshness_runtime"]
    assert "max_freshness_age_ms" in payload


def test_runtime_api_exposes_enriched_macro_terminal_in_decision_analysis(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "symbol": "XAUUSD",
            "action": "buy",
            "chosen": {"confidence": 0.91, "side": "buy"},
            "setup_family": "pullback",
            "context_snapshot": {
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "dxy_alignment": "aligned",
                "us10y_alignment": "aligned",
                "vix_regime": "elevated",
                "vix_alignment": "conflict",
                "oil_backdrop": "inflationary",
                "oil_alignment": "aligned",
                "event_risk_state": "watch",
                "news_risk_classification": "watch",
                "news_sentiment": "mixed",
                "spread_state": "normal",
                "liquidity_state": "normal",
                "slippage_state": "elevated",
                "session": "london",
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    analysis = dict(payload.get("decision_analysis", {}) or {})
    assert analysis["context_alignment"]["vix"]["regime"] == "elevated"
    assert analysis["context_alignment"]["oil"]["backdrop"] == "inflationary"
    assert "size_band_rationale" in analysis["trading_response"]
    assert "confluence" in analysis
    assert "recommended_action" in analysis
    assert "hard_blockers" in analysis
    assert "session" in analysis["context_alignment"]
    assert analysis["macro_context"]["operator_terminal"]["reads"]
    assert "market_regime" in analysis
    assert "reversal_detection" in analysis
    assert "decision_protocol" in analysis
    protocol = dict(analysis.get("decision_protocol", {}) or {})
    assert {"observe", "orient", "critique", "decide", "act"}.issubset(protocol.keys())


def test_runtime_api_exposes_ui_desktop_operator_status(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "ui_desktop_window": {
                "ok": False,
                "stage": "webview2_backend_init",
                "fallback_active": True,
                "fallback_url": "http://127.0.0.1:8765",
                "failure_count": 3,
                "disabled_due_to_repeated_failures": True,
                "ts": "2026-04-22T12:00:00+00:00",
                "classification": {
                    "code": "corewebview2_init_failure",
                    "operator_cause": "CoreWebView2 failed to initialize.",
                    "operator_action": "Repair WebView2 Runtime.",
                },
            }
        }
    )
    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    status = dict(payload.get("ui_desktop_operator_status", {}) or {})
    assert status["desktop_available"] is False
    assert status["stage"] == "webview2_backend_init"
    assert status["failure_code"] == "corewebview2_init_failure"
    assert status["fallback_active"] is True
    assert status["fallback_url"] == "http://127.0.0.1:8765"
    assert status["disabled_due_to_repeated_failures"] is True
    assert status["failure_count"] == 3
    assert "Use browser fallback" in status["action"]


def test_visual_watch_loop_refreshes_latest_visual_read_without_execution(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"chart_features": {"trend_state": "uptrend", "trend_structure": "HH/HL"}})

    monkeypatch.setattr(
        "atlas_ai.ui_server.capture_desktop_screenshot",
        lambda output_dir: {
            "ok": True,
            "path": "",
            "capture_source": "mt5_window",
            "capture_method": {"ok": True, "method": "test_capture"},
        },
    )
    monkeypatch.setattr(
        "atlas_ai.ui_server.analyze_chart_visual_context",
        lambda **_kwargs: {
            "enabled": True,
            "advisory_only": True,
            "source": "screenshot_plus_chart_features",
            "captured_at": "2026-06-08T12:00:00+00:00",
            "visual_features": {"status": "parsed"},
        },
    )

    stop = start_visual_watch_loop(store, interval_seconds=1.0)
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline:
            flags = store.load_runtime_flags()
            if dict(flags.get("visual_watch", {}) or {}).get("last_visual_status") == "parsed":
                break
            time.sleep(0.1)
    finally:
        stop.set()

    visual_watch = dict(store.load_runtime_flags().get("visual_watch", {}) or {})
    latest = json.loads((tmp_path / "chart_vision" / "latest_visual_read.json").read_text(encoding="utf-8"))
    assert visual_watch["enabled"] is True
    assert visual_watch["last_capture_method"] == "test_capture"
    assert visual_watch["last_visual_status"] == "parsed"
    assert "cannot execute trades" in visual_watch["policy_note"]
    assert latest["visual_watch"]["does_not_execute"] is True


def test_runtime_api_exposes_unified_operator_health(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "startup_verification": {
                "startup_ok": True,
                "execute_trades": True,
                "gate_execution_environment": "demo",
                "startup_message": "ATLAS STARTED IN GUARDED DEMO MODE",
            },
            "ui_desktop_window": {
                "ok": False,
                "desktop_window_ready": False,
                "fallback_active": True,
                "frontend_mode": "browser_fallback",
                "failure_reason": "desktop_host_accessibility_recursion",
                "degraded_since": "2026-04-22T16:31:05+00:00",
            },
        }
    )
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "execute_trades": True,
            "system_health": {
                "config_mismatch_detected": False,
                "warnings": [],
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()
    health = dict(payload.get("operator_health", {}) or {})
    assert health["effective_mode"] == "live"
    assert health["effective_mode_source"] == "startup_verification"
    assert health["startup_ok"] is True
    assert health["frontend_mode"] == "browser_fallback"
    assert health["fallback_active"] is True
    assert health["degraded_reason"] == "desktop_host_accessibility_recursion"


def test_execution_readiness_explains_safety_lock_and_stale_execution_truth(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "execution": {
                "execution_truth": {
                    "current": False,
                    "stale": True,
                    "display_reason": "stale_previous_execution_truth",
                    "request_id": "exec_old",
                }
            },
        }
    )
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Safety lock active: consecutive_loss_threshold. No trade executed.",
            "execute_trades": True,
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold",
                },
                "broker_connectivity_status": {
                    "connected": True,
                    "symbol_selected": True,
                },
            },
        }
    )
    automation_path = tmp_path / "automation_state.json"
    automation_path.write_text(
        json.dumps(
            {
                "payload": {
                    "automation_enabled": True,
                    "automation_paused": False,
                    "auto_trade_allowed": False,
                    "auto_trade_reason": "runtime_context_required",
                    "execution_truth": {
                        "current": False,
                        "stale": True,
                        "display_reason": "stale_previous_execution_truth",
                        "request_id": "exec_old",
                    },
                    "session_control": {
                        "session_active": True,
                        "session_pause": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    client = create_ui_app(store).test_client()

    status_payload = client.get("/api/status").get_json()
    runtime_payload = client.get("/api/runtime").get_json()

    readiness = status_payload["execution_readiness"]
    assert readiness["ready"] is False
    assert readiness["primary_blocker"]["lane"] == "safety"
    assert readiness["state"]["execute_trades"] is True
    assert readiness["state"]["safety_lock_active"] is True
    assert readiness["stale_execution_truth"]["present"] is True
    assert readiness["stale_execution_truth"]["request_id"] == "exec_old"
    assert "not the current blocker" in readiness["stale_execution_truth"]["policy_note"]
    assert runtime_payload["execution_readiness"]["primary_blocker"]["lane"] == "safety"


def test_execution_readiness_treats_runtime_context_required_as_market_wait(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "execution": {
                "automation_enabled": True,
                "automation_paused": False,
                "auto_trade_allowed": False,
                "auto_trade_reason": "runtime_context_required",
                "session_control": {"session_active": True, "session_pause": False},
            },
            "ui_desktop_window": {"ok": True, "ready": True},
        }
    )
    store.save_status(
        {
            "action": "blocked",
            "message": "No directional trade signal selected.",
            "execute_trades": True,
            "system_health": {
                "broker_connectivity_status": {
                    "connected": True,
                    "symbol_selected": True,
                },
            },
        }
    )
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_enabled": True,
                    "automation_paused": False,
                    "auto_trade_allowed": False,
                    "auto_trade_reason": "runtime_context_required",
                    "session_control": {"session_active": True, "session_pause": False},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = create_ui_app(store).test_client().get("/api/status").get_json()

    readiness = payload["execution_readiness"]
    assert readiness["ready"] is False
    assert readiness["primary_blocker"]["lane"] == "market"
    assert readiness["primary_blocker"]["code"] == "no_directional_trade_signal"
    assert readiness["state"]["auto_trade_waiting_for_context"] is True
    assert not any(item["code"] == "auto_trade_not_allowed" for item in readiness["blocker_chain"])


def test_execution_readiness_missing_auto_status_does_not_fake_automation_block(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True},
        }
    )
    store.save_status(
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "execute_trades": True,
            "system_health": {
                "broker_connectivity_status": {
                    "connected": True,
                    "symbol_selected": True,
                },
            },
        }
    )
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_enabled": True,
                    "automation_paused": False,
                    "session_control": {"session_active": True, "session_pause": False},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = create_ui_app(store).test_client().get("/api/status").get_json()

    readiness = payload["execution_readiness"]
    assert readiness["primary_blocker"]["lane"] == "market"
    assert readiness["state"]["auto_trade_status_known"] is False
    assert readiness["state"]["auto_trade_waiting_for_context"] is True
    assert not any(item["code"] == "auto_trade_not_allowed" for item in readiness["blocker_chain"])


def test_system_audit_endpoint_summarizes_wiring_without_policy_changes(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True, "frontend_mode": "desktop"},
            "visual_watch": {"enabled": True, "last_capture_ok": True},
        }
    )
    store.save_status(
        {
            "action": "blocked",
            "message": "Risk guard active: Safety lock active: consecutive_loss_threshold. No trade executed.",
            "execute_trades": True,
            "system_health": {
                "safety_layer": {
                    "safety_lock_active": True,
                    "safety_lock_reason": "consecutive_loss_threshold",
                },
                "broker_connectivity_status": {"connected": True, "symbol_selected": True},
                "warnings": ["llm_openai_quota_exhausted"],
            },
        }
    )
    (tmp_path / "automation_state.json").write_text(
        json.dumps(
            {
                "payload": {
                    "automation_enabled": True,
                    "automation_paused": False,
                    "session_control": {"session_active": True, "session_pause": False},
                }
            }
        ),
        encoding="utf-8",
    )

    payload = create_ui_app(store).test_client().get("/api/system-audit").get_json()

    assert payload["summary"]["desktop_ready"] is True
    assert payload["summary"]["execute_trades"] is True
    assert payload["summary"]["primary_blocker"]["code"] == "safety_lock_active"
    assert any(item["code"] == "safety_lock_active" for item in payload["issues"])
    assert "read-only observability" in payload["policy_note"]


def test_visual_evidence_review_exposes_proxy_only_capture_degradation(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_runtime_flags(
        {
            "execute_trades": True,
            "ui_desktop_window": {"ok": True, "ready": True, "frontend_mode": "desktop"},
            "visual_watch": {
                "enabled": True,
                "last_capture_ok": False,
                "last_visual_status": "no_screenshot_supplied",
            },
        }
    )
    store.save_status(
        {
            "action": "blocked",
            "message": "Timing discipline blocked trade: wait for clean entry timing confirmation.",
            "execute_trades": True,
            "system_health": {"broker_connectivity_status": {"connected": True, "symbol_selected": True}},
            "timing_discipline": {"hard_block": True, "missing": ["tradable_location"]},
        }
    )
    save_visual_read(
        {
            "available": True,
            "advisory_only": True,
            "source": "chart_features_visual_proxy",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "pattern_seen": "messy_range",
            "setup_quality": 0.31,
            "visual_confirmation": "does_not_confirm",
            "image": {"available": False, "status": "no_screenshot_supplied"},
            "capture": {
                "ok": False,
                "target_window": "mt5",
                "error": "mt5_or_hfm_window_not_found",
                "window": {"found": False, "reason": "mt5_or_hfm_window_not_found"},
            },
        },
        tmp_path / "chart_vision",
    )
    app = create_ui_app(store)

    status = app.test_client().get("/api/status/compact").get_json()
    review = app.test_client().get("/api/visual-evidence-review").get_json()
    audit = app.test_client().get("/api/system-audit").get_json()

    assert status["visual_evidence_review"]["status"] == "proxy_only"
    assert status["visual_evidence_review"]["degraded"] is True
    assert status["execution_readiness"]["state"]["visual_capture_degraded"] is True
    assert status["execution_readiness"]["state"]["visual_capture_error"] == "mt5_or_hfm_window_not_found"
    assert review["capture"]["window_found"] is False
    assert review["proxy_only"] is True
    assert audit["summary"]["visual_capture_degraded"] is True
    assert any(item["code"] == "visual_capture_degraded" for item in audit["issues"])


def test_runtime_and_ask_share_execution_state_source(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()
    index_html = client.get("/").data.decode("utf-8")
    app_js = client.get("/app.js").data.decode("utf-8")
    ask = client.post("/api/ask-atlas", json={"query": "set execution environment demo"}).get_json()
    assert ask["execution_state"]["execution_environment"] == "demo"
    runtime = client.get("/api/runtime").get_json()
    assert runtime["execution"]["execution_environment"] == "demo"
    assert "op-snapshot-facts" in index_html
    assert "op-reco-recommendation" in index_html
    assert "op-feedback-facts" in index_html
    assert "op-tuning-facts" in index_html
    assert "op-live-facts" in index_html
    assert "op-live-inference" in index_html
    assert "op-live-recommendation" in index_html
    assert "op-live-details" in index_html
    assert "op-pending-items" in index_html
    assert "op-approve-item" in index_html
    assert "op-approve-all" in index_html
    assert "op-reject-item" in index_html
    assert "op-defer-item" in index_html
    assert "op-show-pending" in index_html
    assert "op-explain-item" in index_html
    assert "op-why-item" in index_html
    assert "const OPERATOR_QUERIES" in app_js
    assert "function renderOperatorSnapshot(status, runtime)" in app_js
    assert "function renderOperatorRecommendation(payload)" in app_js
    assert "function renderOperatorFeedback(payload)" in app_js
    assert "function renderOperatorTuning(payload)" in app_js
    assert "function renderOperatorQueue(payload)" in app_js
    assert "function renderOperatorLivePilotReview(status, runtime, queuePayload, reviewPayload)" in app_js
    assert "function bindOperatorActions()" in app_js
    assert "approve item" in app_js
    assert "approve all pending items" in app_js
    assert "reject item" in app_js
    assert "defer item" in app_js
    assert "show pending items" in app_js
    assert "explain item" in app_js
    assert "why did you suggest item" in app_js
    assert "No automatic execution" in app_js


def test_paper_compare_report_still_builds_with_decision_explanations(tmp_path):
    tracker = ComparisonTracker(tmp_path)
    settings = SimpleNamespace(symbol="XAUUSD", timeframe="M1")
    control_status = {
        "action": "paper",
        "message": "control paper",
        "updated_at": "2026-04-04T11:46:23+00:00",
        "chosen": {
            "strategy": "momentum_continuation",
            "side": "buy",
            "confidence": 0.31,
            "stop_loss_points": 115.0,
            "take_profit_points": 175.0,
        },
        "market": {"recent_time": ["2026-04-02 23:59:00"]},
        "regime": {"name": "trend", "trend_strength": 0.24},
        "research_regime_overlay": {"session": "late_us"},
        "decision_explanation": {
            "final_decision": "allow",
            "explanation_text": "Allowed after passing the conservative pre-trade checks.",
            "reason_codes": ["allowed_after_filters"],
        },
        "context_snapshot": {"regime": "trend", "session": "late_us"},
        "shadow_advice": {"recommendation": "caution", "anomaly_flags": ["poor_liquidity"]},
        "system_health": {"warnings": ["stale_data_detected"]},
    }
    experiment_status = {
        **control_status,
        "action": "blocked",
        "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
        "decision_explanation": {
            "final_decision": "block",
            "explanation_text": "Blocked due to poor liquidity.",
            "reason_codes": ["poor_liquidity"],
        },
        "context_snapshot": {"regime": "trend", "session": "late_us", "liquidity_state": "poor"},
        "shadow_advice": {"recommendation": "observe", "anomaly_flags": ["blocked_trade"]},
        "system_health": {"warnings": ["routing_risk_detected"]},
    }

    tracker.record_cycle(control_status, experiment_status, settings)
    report = tracker.build_report(
        SimpleNamespace(closed_trades=[], open_trades=[], filtered_trades=0, metrics={}),
        SimpleNamespace(closed_trades=[], open_trades=[], filtered_trades=1, metrics={}),
    )

    removed = next(iter(report["trades_removed_breakdown"]), {})
    assert removed["decision_explanation"]["final_decision"] == "block"
    assert removed["control_decision_explanation"]["final_decision"] == "allow"
    assert removed["experiment_decision_explanation"]["final_decision"] == "block"
    assert removed["experiment_context_snapshot"]["liquidity_state"] == "poor"
    assert removed["experiment_shadow_advice"]["recommendation"] == "observe"
    assert removed["experiment_system_health"]["warnings"] == ["routing_risk_detected"]
    assert report["lane_diagnostics"]["latest_control"]["shadow_advice"]["recommendation"] == "caution"
    assert report["lane_diagnostics"]["latest_experiment"]["system_health"]["warnings"] == ["routing_risk_detected"]
    assert report["decision_checklist"]["classification"] == "CONTINUE_TESTING"


def test_status_api_uses_comparison_experiment_store_when_run_is_bound(tmp_path):
    default_store = KnowledgeStore(str(tmp_path / "storage"))
    default_store.save_status({"action": "connected", "message": "default store"})

    run_root = tmp_path / "storage_compare_runs" / "run123"
    (run_root / "control").mkdir(parents=True)
    (run_root / "experiment").mkdir(parents=True)

    KnowledgeStore(str(run_root / "experiment")).save_status(
        {
            "action": "hold",
            "message": "experiment store",
            "lane": "experiment",
            "decision_explanation": {"final_decision": "block"},
        }
    )
    KnowledgeStore(str(run_root / "control")).save_status(
        {
            "action": "blocked",
            "message": "control store",
            "lane": "control",
        }
    )
    (run_root / "comparison_summary.json").write_text(
        json.dumps(
            {
                "updated_at": "2026-04-06T08:30:00+00:00",
                "run_id": "run123",
                "control_status": {"action": "blocked", "message": "control store"},
                "experiment_status": {"action": "hold", "message": "experiment store"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "comparison_report.json").write_text(
        json.dumps(
            {
                "control": {"trades_closed": 1},
                "experiment": {"trades_closed": 2},
                "cross_lane": {"filtered_by_experiment": 1},
                "filter_effectiveness": {"assessment": "neutral"},
                "decision_checklist": {"classification": "CONTINUE_TESTING"},
            }
        ),
        encoding="utf-8",
    )
    (run_root / "preflight.json").write_text(
        json.dumps({"run_id": "run123", "preset": "setup_quality_rollout_v1"}),
        encoding="utf-8",
    )

    app = create_ui_app(default_store, comparison_root=str(tmp_path / "storage_compare_runs"), comparison_run_id="run123")
    client = app.test_client()

    response = client.get("/api/status")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["message"] == "experiment store"
    assert payload["lane"] == "experiment"
    assert payload["comparison_active"] is True
    assert payload["comparison"]["run_id"] == "run123"
    assert payload["comparison"]["control_status"]["message"] == "control store"


def test_runtime_mutation_is_blocked_when_viewing_comparison_run(tmp_path):
    default_store = KnowledgeStore(str(tmp_path / "storage"))
    run_root = tmp_path / "storage_compare_runs" / "run123"
    (run_root / "control").mkdir(parents=True)
    (run_root / "experiment").mkdir(parents=True)
    (run_root / "comparison_summary.json").write_text(json.dumps({"run_id": "run123"}), encoding="utf-8")
    (run_root / "comparison_report.json").write_text(json.dumps({}), encoding="utf-8")
    (run_root / "preflight.json").write_text(json.dumps({"run_id": "run123", "preset": "setup_quality_rollout_v1"}), encoding="utf-8")

    app = create_ui_app(default_store, comparison_root=str(tmp_path / "storage_compare_runs"), comparison_run_id="run123")
    client = app.test_client()

    runtime_payload = client.get("/api/runtime").get_json()
    response = client.post("/api/runtime/execute-trades", json={"enabled": True})

    assert runtime_payload["comparison_active"] is True
    assert runtime_payload["read_only_ui"] is True
    assert response.status_code == 409
    assert response.get_json()["ok"] is False


def test_runtime_execute_trades_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/runtime/execute-trades", json={"enabled": True})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == "set execute_trades true"


def test_runtime_execute_trades_accepts_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post(
        "/api/runtime/execute-trades",
        json={"enabled": True, "confirmation": "set execute_trades true"},
    )

    assert response.status_code == 200
    assert response.get_json()["execute_trades"] is True


def test_runtime_gemini_influence_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/runtime/gemini-influence", json={"enabled": False})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == "set gemini_influence false"


def test_safety_unlock_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/safety/unlock", json={"confirm": True})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == "unlock safety"


def test_ask_atlas_endpoint_returns_structured_grounded_response(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "context_snapshot": {"session": "late_us"},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "why are you not trading?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["mode"] in {"standard", "quick", "voice", "deep"}
    assert payload["reasoning_mode"] == "INTENT_SYNTHESIS"
    assert isinstance(payload["data_sources"], list)
    assert payload["result"]
    assert "approval" in payload


def test_ask_atlas_endpoint_refuses_runtime_mutation_requests(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "enable live and place a trade"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["intent"] == "refusal_mutating_request"
    assert "will not place trades" in payload["result"].lower()
    assert "approval" in payload["approval"].lower()


def test_ask_atlas_endpoint_routes_chart_question_to_chart_understanding(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.21},
            "context_snapshot": {
                "session": "late_us",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "what is your current chart read?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["intent"] in {"unknown_operator_query", "comparison", "summary"}
    assert payload["result"]


def test_ask_atlas_endpoint_routes_task_command_to_task_result(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.21},
            "context_snapshot": {
                "session": "late_us",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "higher_tf_bias": "buy",
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "min_confidence_threshold": 0.50,
        }
    )
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "go study the charts"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["response_kind"] in {"question_answer", "task_result"}
    assert payload["result"]


def test_ask_atlas_endpoint_returns_prioritized_blocker_analysis(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Spread too high (42.0 pts).",
            "context_snapshot": {"spread_state": "wide", "liquidity_state": "poor"},
        }
    )
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Spread too high (42.0 pts).")
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "what is the main blocker"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["reasoning_mode"] == "INTENT_SYNTHESIS"
    assert payload["result"]


def test_ask_atlas_endpoint_routes_self_assessment_query(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "regime": {"name": "range", "trend_strength": 0.06},
            "context_snapshot": {
                "volatility_state": "low",
                "spread_state": "wide",
                "liquidity_state": "unknown",
            },
            "win_rate_focus": {"samples": 5, "recent_win_rate": 0.2},
            "drift": {"active": True},
            "system_health": {"paper_mode_confirmed": True},
            "decision_explanation": {"final_decision": "block", "reason_codes": ["no_high_confidence_trade_found"]},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "how confident are you in your analysis"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["reasoning_mode"] == "INTENT_SYNTHESIS"
    assert payload["intent"] in {"confidence", "help", "unknown_operator_query"}
    assert payload["result"]


def test_ask_atlas_endpoint_routes_capability_assessment_query(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "win_rate_focus": {"samples": 30, "recent_win_rate": 0.267, "target": 0.55},
            "drift": {"active": True, "recent_win_rate_20": 0.15, "baseline_win_rate": 0.34},
            "system_health": {"effective_mode": "paper", "paper_mode_confirmed": True},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "what do you need to become a day trader"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["reasoning_mode"] == "INTENT_SYNTHESIS"
    assert payload["result"]


def test_ask_atlas_endpoint_uses_conversation_layer_for_informal_query(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "what's going on"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["reasoning_mode"] == "INTENT_SYNTHESIS"
    assert payload["result"]


def test_ask_atlas_routes_general_question_without_trading_output(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "How long does it take to become your best self"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["handled_as"] == "general"
    assert payload["intent"] == "general_question"
    assert "hold" not in payload["result"].lower()
    assert "no high-confidence trade found" not in payload["result"].lower()


def test_ask_atlas_routes_trading_question_as_trading(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "decision_explanation": {"final_decision": "block", "reason_codes": ["no_high_confidence_trade_found"]},
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "Should I buy EURUSD?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["handled_as"] == "trading"
    assert payload["result"]
    trading_response = dict(payload.get("trading_response", {}) or {})
    assert {"bias", "confidence", "blockers", "context_alignment", "recommended_action"}.issubset(trading_response.keys())


def test_ask_atlas_routes_explanation_query_to_explanation(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "decision_explanation": {
                "final_decision": "block",
                "reason_codes": ["no_high_confidence_trade_found"],
                "setup_quality_reasons": ["weak_consensus"],
            },
            "context_snapshot": {
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "dxy_alignment": "conflict",
                "us10y_alignment": "aligned",
            },
            "chart_features": {
                "new_local_high": True,
                "exhaustion_wick": True,
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "Why no trade?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["handled_as"] == "explanation"
    assert payload["intent"] == "explanation_request"
    assert "blocker" in payload["result"].lower() or "decision" in payload["result"].lower()
    assert "regime" in payload["result"].lower()


def test_ask_atlas_routes_system_query_to_system_status(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "updated_at": "2026-04-23T10:00:00+00:00",
            "operator_health": {
                "frontend_mode": "desktop",
                "desktop_window_ready": True,
                "effective_mode": "live",
            },
            "system_health": {
                "config_mismatch_detected": False,
                "routing_risk_detected": False,
                "stale_data_detected": False,
            },
        }
    )
    app = create_ui_app(store)
    client = app.test_client()

    response = client.post("/api/ask-atlas", json={"query": "Is backend running?"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["handled_as"] == "system"
    assert payload["intent"] == "system_status_query"
    assert "backend" in payload["result"].lower()


def test_ask_atlas_endpoint_conversation_memory_supports_follow_up(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "context_snapshot": {"session": "new_york", "volatility_state": "normal", "spread_state": "normal", "liquidity_state": "unknown"},
            "regime": {"name": "range", "trend_strength": 0.08},
            "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")
    app = create_ui_app(store)
    client = app.test_client()

    first = client.post("/api/ask-atlas", json={"query": "what should I do about consensus"}).get_json()
    second = client.post("/api/ask-atlas", json={"query": "what about that"}).get_json()

    assert first["result"]
    assert second["result"]
    assert second["intent"] in {"recommendation", "blocker", "safe_next_step", "blocker_resolution"}
    assert second["understanding"].endswith("(follow-up resolved)")


def test_ask_atlas_control_actions_require_explicit_item_commands(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    pending = client.post("/api/ask-atlas", json={"query": "show pending items"}).get_json()
    approve = client.post("/api/ask-atlas", json={"query": "approve item 999"}).get_json()
    why = client.post("/api/ask-atlas", json={"query": "why did you suggest item 999"}).get_json()

    assert pending["response_kind"] == "control"
    assert pending["intent"] == "control_list"
    assert approve["response_kind"] == "control"
    assert approve["intent"] == "control_action"
    assert "not found" in approve["result"].lower()
    assert why["response_kind"] == "control"
    assert why["intent"] == "control_explain"


def test_status_api_auto_detects_freshest_comparison_run(tmp_path):
    default_store = KnowledgeStore(str(tmp_path / "storage"))
    base = tmp_path / "storage_compare_runs"
    old_root = base / "old_run"
    new_root = base / "new_run"
    for root, message in [(old_root, "old experiment"), (new_root, "new experiment")]:
        (root / "control").mkdir(parents=True)
        (root / "experiment").mkdir(parents=True)
        KnowledgeStore(str(root / "experiment")).save_status({"action": "hold", "message": message, "lane": "experiment"})
        (root / "comparison_summary.json").write_text(json.dumps({"run_id": root.name}), encoding="utf-8")
        (root / "comparison_report.json").write_text(json.dumps({}), encoding="utf-8")
        (root / "preflight.json").write_text(json.dumps({"run_id": root.name, "preset": "setup_quality_rollout_v1"}), encoding="utf-8")
    old_time = 1700000000
    new_time = 1800000000
    import os
    os.utime(old_root / "comparison_summary.json", (old_time, old_time))
    os.utime(new_root / "comparison_summary.json", (new_time, new_time))

    app = create_ui_app(default_store, comparison_root=str(base))
    client = app.test_client()
    payload = client.get("/api/status").get_json()

    assert payload["message"] == "new experiment"
    assert payload["comparison"]["run_id"] == "new_run"


def test_runtime_api_exposes_self_improvement_summary(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    app = create_ui_app(store)
    client = app.test_client()

    payload = client.get("/api/runtime").get_json()

    assert payload["self_improvement_state"]["proposal_count"] == 0
    assert payload["self_improvement_state"]["active_rule_version"] == "atlas_rules_v1"
    assert payload["self_improvement"]["self_improvement_policy.json"]["version"] == "1.0"


def test_self_improvement_api_endpoints_are_read_only_and_grounded(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.append_self_improvement_audit({"event": "proposal_created", "proposal_id": "sip_1"})
    app = create_ui_app(store)
    client = app.test_client()

    policy = client.get("/api/self-improvement/policy").get_json()
    proposals = client.get("/api/self-improvement/proposals").get_json()
    insights = client.get("/api/self-improvement/insights").get_json()
    rollouts = client.get("/api/self-improvement/rollouts").get_json()
    audit = client.get("/api/self-improvement/audit").get_json()

    assert policy["version"] == "1.0"
    assert proposals["summary"]["proposal_count"] == 0
    assert insights["summary"]["insight_count"] == 0
    assert rollouts["summary"]["rollout_count"] == 0
    assert audit[-1]["proposal_id"] == "sip_1"


def test_self_improvement_operator_review_endpoint_is_read_only(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    append_or_update_proposal(
        store,
        {
            "proposal_id": "proposal_consensus_selectivity_context_review",
            "status": "awaiting_approval",
            "category": "filters",
            "scope": {"target_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2."},
            "current_behavior": {"summary": "Consensus guard requires same-side support."},
            "proposed_change": {"type": "context_confirmation_candidate", "does_not_allow_trade": True},
            "validation": {
                "required": ["richer_context_replay", "shadow_compare", "regression_tests", "operator_review"],
                "completed": ["richer_context_replay", "shadow_compare", "regression_tests"],
                "missing": ["operator_review"],
            },
            "approval": {"required": True, "approved": False, "approved_by": "", "approved_at": ""},
        },
    )
    client = create_ui_app(store).test_client()

    packet = client.get("/api/self-improvement/proposals/proposal_consensus_selectivity_context_review/operator-review").get_json()
    proposal = load_self_improvement_bundle(store)["proposal_queue.json"]["proposals"][0]
    saved_path = tmp_path / "proposal_reports" / "proposal_consensus_selectivity_context_review_operator_review_packet.json"
    saved_packet = json.loads(saved_path.read_text(encoding="utf-8"))

    assert packet["schema"] == "atlas_consensus_selectivity_operator_review_packet_v1"
    assert saved_packet["schema"] == packet["schema"]
    assert saved_packet["created_at"] == packet["created_at"]
    assert saved_packet["evidence"] == packet["evidence"]
    assert packet["approval"]["approved"] is False
    assert packet["operator_posture"]["inputs"]["regression_ok"] is True
    assert packet["evidence"]["regression_tests"]["source"] == "proposal_validation_completed_marker"
    assert packet["safety"]["does_not_approve"] is True
    assert packet["safety"]["does_not_activate_rollout"] is True
    assert proposal["approval"]["approved"] is False


def test_ai_observability_api_endpoints_are_grounded_and_read_only(tmp_path):
    from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
    from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool

    store = KnowledgeStore(str(tmp_path))
    journal = AIDecisionJournal(tmp_path)
    journal.append(
        {
            "schema": "atlas_ai_decision_trace_v1",
            "recommendation": {"action": "wait_for_confirmation", "confidence": 0.62, "reason": "confirmation weak"},
            "blocker_analysis": {"primary_blocker": "confidence below threshold"},
            "strategy_invention": {"candidate_count": 1, "candidate_archetypes": ["trend_pullback_continuation"]},
        },
        source="test",
        query="why did you wait",
    )
    store.save_learning_state({"loss_memory": {"recent_outcomes": [{"outcome": "loss"}]}})
    store.log_event("blocked", "confidence below threshold")
    chromosome = StrategyChromosome(
        uid="ai_candidate",
        generation=0,
        genes=make_standard_gene_pool(),
        origin="ai_inventor:trend_pullback_continuation",
        promotion_stage="untested",
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"chromosomes": [chromosome.to_dict()]}),
        encoding="utf-8",
    )

    app = create_ui_app(store)
    client = app.test_client()

    latest = client.get("/api/ai/decision/latest").get_json()
    journal_payload = client.get("/api/ai/decision/journal?limit=5").get_json()
    review = client.post("/api/ai/decision/review", json={"limit": 5}).get_json()
    reviews = client.get("/api/ai/decision/reviews").get_json()
    prefs = client.get("/api/ai/strategy/preferences").get_json()
    recompute = client.post("/api/ai/promotion/recompute").get_json()
    promo = client.get("/api/ai/promotion/recommendations").get_json()
    queue = client.get("/api/ai/research/queue").get_json()
    item_id = queue["queue"]["items"][0]["id"]
    paper_test = client.post(
        f"/api/ai/research/items/{item_id}/paper-test",
        json={"note": "paper only", "confirmation": f"paper test {item_id}"},
    ).get_json()
    demo_review = client.post(
        f"/api/ai/research/items/{item_id}/demo-review",
        json={"note": "demo review only", "confirmation": f"demo review {item_id}"},
    ).get_json()
    demo_packet = client.post(f"/api/ai/research/items/{item_id}/demo-review-packet", json={}).get_json()
    demo_approve = client.post(
        f"/api/ai/research/items/{item_id}/demo-review-approve",
        json={"note": "approval only", "confirmation": f"approve demo review {item_id}"},
    ).get_json()
    demo_proposal = client.post(f"/api/ai/research/items/{item_id}/demo-activation-proposal", json={}).get_json()

    assert latest["available"] is True
    assert journal_payload["records"][0]["source"] == "test"
    assert review["ok"] is True
    assert review["executed"] is False
    assert review["review"]["reviewed"] >= 1
    assert reviews["reviews"]
    assert prefs["preferences"]["policy"]["does_not_execute"] is True
    assert recompute["executed"] is False
    assert recompute["recommendations"]["safety"]["does_not_execute"] is True
    assert promo["available"] is True
    assert queue["summary"]["pending_review"] >= 1
    assert queue["queue"]["policy"]["shadow_drawdown_policy"] == "raw_peak"
    assert queue["queue"]["policy"]["shadow_drawdown_limit_pct"] == 0.05
    assert paper_test["executed"] is False
    assert paper_test["item"]["status"] == "paper_test_requested"
    assert demo_review["executed"] is False
    assert demo_review["item"]["status"] == "demo_review_requested"
    assert demo_packet["executed"] is False
    assert demo_packet["safety"]["does_not_promote_demo"] is True
    assert demo_packet["queue_update"]["executed"] is False
    assert demo_packet["queue_update"]["item"]["demo_review_packet"]["does_not_promote_live"] is True
    assert demo_approve["executed"] is False
    assert demo_approve["item"]["status"] == "demo_review_approved"
    assert demo_proposal["executed"] is False
    assert demo_proposal["ok"] is False
    assert demo_proposal["reason"] == "demo_review_not_approved"


def test_ai_demo_review_approval_requires_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": "demo_review_requested",
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )

    client = create_ui_app(store).test_client()
    response = client.post(
        "/api/ai/research/items/demo_candidate:eligible_for_demo_review/demo-review-approve",
        json={"note": "approval only"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == "approve demo review demo_candidate:eligible_for_demo_review"


def test_ai_research_queue_actions_require_explicit_confirmation_phrase(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": "pending_review",
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )

    client = create_ui_app(store).test_client()
    response = client.post(
        "/api/ai/research/items/demo_candidate:eligible_for_demo_review/paper-test",
        json={"note": "paper only"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"] == "confirmation_required"
    assert payload["confirmation_required"] == "paper test demo_candidate:eligible_for_demo_review"


def test_ai_demo_activation_proposal_endpoint_is_record_only(tmp_path):
    from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool

    store = KnowledgeStore(str(tmp_path))
    candidate = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        backtest_trades=260,
        backtest_win_rate=0.61,
        shadow_passed=True,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"chromosomes": [candidate.to_dict()]}),
        encoding="utf-8",
    )
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": "demo_review_approved",
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_review_packet_demo_candidate.json").write_text(
        json.dumps({"safety": {"does_not_promote_demo": True, "does_not_promote_live": True}}),
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_reviews.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "recommendation": "eligible_for_demo_review",
            "summary": {"trades": 260, "win_rate": 0.61},
            "promotion_governor": {"can_promote_to_demo": True},
        })
        + "\n",
        encoding="utf-8",
    )

    client = create_ui_app(store).test_client()
    payload = client.post("/api/ai/research/items/demo_candidate:eligible_for_demo_review/demo-activation-proposal").get_json()
    preflight = client.post("/api/ai/research/items/demo_candidate:eligible_for_demo_review/demo-activation-preflight").get_json()

    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["safety"]["does_not_promote_demo"] is True
    assert payload["operator_boundary"]["does_not_call_promote_to_demo"] is True
    assert payload["queue_update"]["item"]["status"] == "demo_activation_proposed"
    assert preflight["ok"] is True
    assert preflight["executed"] is False
    assert preflight["safety"]["does_not_promote_demo"] is True


def test_ai_demo_activation_endpoint_changes_demo_state_only(tmp_path):
    from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool

    store = KnowledgeStore(str(tmp_path))
    candidate = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        backtest_trades=260,
        backtest_win_rate=0.61,
        shadow_passed=True,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"chromosomes": [candidate.to_dict()]}),
        encoding="utf-8",
    )
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": "demo_activation_proposed",
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_activation_preflight_demo_candidate.json").write_text(
        json.dumps({"ok": True, "blockers": []}),
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_reviews.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "recommendation": "eligible_for_demo_review",
            "summary": {"trades": 260, "win_rate": 0.61},
            "promotion_governor": {"can_promote_to_demo": True},
        })
        + "\n",
        encoding="utf-8",
    )

    client = create_ui_app(store).test_client()
    payload = client.post(
        "/api/ai/research/items/demo_candidate:eligible_for_demo_review/activate-demo",
        json={"explicit_command": "Activate demo for demo_candidate"},
    ).get_json()
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))

    assert payload["ok"] is True
    assert payload["trade_executed"] is False
    assert payload["live_promoted"] is False
    assert payload["candidate"]["promotion_stage"] == "demo"
    assert population["chromosomes"][0]["promotion_stage"] == "demo"
    assert population["chromosomes"][0]["demo_active"] is True


def test_ai_demo_activation_endpoint_rejects_loose_command_with_uid(tmp_path):
    from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool

    store = KnowledgeStore(str(tmp_path))
    candidate = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        backtest_trades=260,
        backtest_win_rate=0.61,
        shadow_passed=True,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"chromosomes": [candidate.to_dict()]}),
        encoding="utf-8",
    )
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": "demo_activation_proposed",
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_activation_preflight_demo_candidate.json").write_text(
        json.dumps({"ok": True, "blockers": []}),
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_reviews.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "recommendation": "eligible_for_demo_review",
            "summary": {"trades": 260, "win_rate": 0.61},
            "promotion_governor": {"can_promote_to_demo": True},
        })
        + "\n",
        encoding="utf-8",
    )

    client = create_ui_app(store).test_client()
    response = client.post(
        "/api/ai/research/items/demo_candidate:eligible_for_demo_review/activate-demo",
        json={"explicit_command": "Activate demo immediately for demo_candidate"},
    )
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))

    assert response.status_code == 409
    payload = response.get_json()
    assert payload["reason"] == "explicit_demo_activation_command_required"
    assert payload["state_changed"] is False
    assert population["chromosomes"][0]["promotion_stage"] == "shadow"
    assert population["chromosomes"][0].get("demo_active") is not True


def test_runtime_reconciles_stale_desktop_wrapper_pid(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "ui_desktop_window": {
                "ok": True,
                "ready": True,
                "stage": "started",
                "frontend_mode": "desktop",
                "desktop_window_ready": True,
                "wrapper_pid": 99999,
            }
        }
    )
    monkeypatch.setattr("atlas_ai.ui_server._process_exists", lambda pid: False)
    monkeypatch.setattr("atlas_ai.ui_server._discover_ui_wrapper_pid", lambda: 54321)

    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()

    assert payload["ui_desktop_window"]["wrapper_pid"] == 54321
    assert payload["ui_desktop_window"]["ready"] is True
    assert payload["ui_desktop_window"]["failure_evidence"]["wrapper_pid_reconciled_from_process_scan"] == 54321


def test_runtime_clears_stale_wrapper_pid_when_no_ui_wrapper_found(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "ui_desktop_window": {
                "ok": True,
                "ready": True,
                "stage": "started",
                "frontend_mode": "desktop",
                "desktop_window_ready": True,
                "wrapper_pid": 99999,
            }
        }
    )
    monkeypatch.setattr("atlas_ai.ui_server._process_exists", lambda pid: False)
    monkeypatch.setattr("atlas_ai.ui_server._discover_ui_wrapper_pid", lambda: 0)

    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get("/api/runtime").get_json()

    assert payload["ui_desktop_window"]["wrapper_pid"] == 0
    assert payload["ui_desktop_window"]["ready"] is False
    assert payload["ui_desktop_window"]["desktop_window_ready"] is False


def test_learning_summary_is_exposed_across_status_runtime_and_diagnostics(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "outcome": "loss",
                        "loss_classification": "false_continuation",
                        "setup_family": "continuation",
                        "net_pnl_zar": -25.0,
                    },
                    {
                        "outcome": "loss",
                        "loss_classification": "late_entry",
                        "setup_family": "pullback",
                        "net_pnl_zar": -18.0,
                    },
                    {
                        "outcome": "win",
                        "setup_family": "pullback",
                        "net_pnl_zar": 31.0,
                    },
                ]
            },
            "best_evolved_strategy": {
                "signature": "f37_s98_rb48.8_rs49.2",
                "source": "mutation_tournament",
            },
            "evolved_strategy_pool": {
                "champion_signature": "f37_s98_rb48.8_rs49.2",
            },
        }
    )
    store.log_event(
        "training",
        "Historical training completed",
        {
            "bars": 3600,
            "total_return": 0.2145,
            "trades": 242,
            "wins": 115,
            "losses": 127,
            "best_strategy": {"signature": "f37_s98_rb48.8_rs49.2"},
            "weak_point_summary": {
                "strengths": ["trend conditions are no longer loss-dominant"],
                "weaknesses": [],
            },
            "setup_family_summary": {
                "best_family": "pullback",
                "families": [
                    {"family": "pullback", "count": 3, "avg_score": 0.08, "avg_win_rate": 0.53, "avg_trend_edge": 0.02}
                ],
            },
        },
    )
    app = create_ui_app(store)
    client = app.test_client()

    status_payload = client.get("/api/status").get_json()
    runtime_payload = client.get("/api/runtime").get_json()
    diagnostics_payload = client.get("/api/diagnostics").get_json()

    for payload in (status_payload, runtime_payload, diagnostics_payload):
        summary = payload["learning_summary"]
        assert summary["champion_signature"] == "f37_s98_rb48.8_rs49.2"
        assert summary["top_setup_family"] == "pullback"
        assert summary["loss_reason_summary"]["top_reason"] == "false_continuation"
        assert summary["loss_reason_summary"]["total_losses"] == 2
        assert summary["latest_training"]["return_pct"] == 21.45


def test_learning_insights_returns_structured_learning_summary(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {"outcome": "loss", "loss_classification": "false_continuation", "setup_family": "continuation", "net_pnl_zar": -12.0},
                    {"outcome": "loss", "loss_classification": "false_continuation", "setup_family": "continuation", "net_pnl_zar": -11.0},
                    {"outcome": "loss", "loss_classification": "late_entry", "setup_family": "pullback", "net_pnl_zar": -9.0},
                    {"outcome": "win", "setup_family": "pullback", "net_pnl_zar": 15.0},
                ]
            },
            "best_evolved_strategy": {"signature": "f30_s92_rb50.0_rs45.3", "source": "mutation_tournament"},
        }
    )
    store.log_event(
        "training",
        "Historical training completed",
        {
            "bars": 1500,
            "total_return": 0.1423,
            "trades": 193,
            "wins": 90,
            "losses": 103,
            "setup_family_summary": {
                "best_family": "continuation",
                "families": [{"family": "continuation", "count": 4, "avg_score": 0.07, "avg_win_rate": 0.51, "avg_trend_edge": 0.019}],
            },
            "weak_point_summary": {"strengths": ["trend edge is measurable"], "weaknesses": []},
        },
    )
    app = create_ui_app(store)
    client = app.test_client()

    payload = client.get("/api/learning-insights").get_json()

    assert payload["learning_summary"]["champion_signature"] == "f30_s92_rb50.0_rs45.3"
    assert payload["learning_summary"]["top_setup_family"] == "continuation"
    assert payload["learning_summary"]["loss_reason_summary"]["reasons"][0]["reason"] == "false_continuation"
    assert payload["learning_summary"]["loss_reason_summary"]["reasons"][0]["count"] == 2
