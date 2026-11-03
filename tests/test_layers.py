from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from atlas_ai.decision_support import build_decision_explanation, build_decision_trace
from atlas_ai.learning.self_improvement_scoring import score_evidence_strength
from atlas_ai.market_context import CONTEXT_SNAPSHOT_FIELDS, build_context_snapshot
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.shadow_intelligence import SHADOW_ADVICE_FIELDS, build_shadow_advice
from atlas_ai.self_improvement import append_or_update_proposal, build_self_improvement_summary, load_self_improvement_bundle
from atlas_ai.system_health import build_freshness_runtime, build_system_health
from atlas_ai.trader import AtlasTrader


def _base_status() -> dict:
    return {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "updated_at": "2026-04-04T12:00:00+00:00",
        "action": "blocked",
        "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
        "execute_trades": False,
        "gemini_enabled": True,
        "gemini_influence_enabled": False,
        "chosen": {
            "strategy": "momentum_continuation",
            "side": "buy",
            "confidence": 0.31,
        },
        "risk": {
            "spread_points": 18.0,
        },
        "regime": {
            "name": "trend",
            "trend_strength": 0.24,
            "volatility_pct": 0.028,
        },
        "market": {
            "recent_time": ["2026-04-04T11:59:00+00:00"],
        },
        "health": {
            "broker": {"connected": True, "symbol_selected": True},
            "llm": {"enabled": True, "providers": []},
        },
        "research_regime_overlay": {
            "session": "late_us",
            "volume_ratio": 0.74,
            "spike_ratio": 1.72,
        },
        "conservative_pretrade_filter": {
            "enabled": True,
            "reason": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "inputs": {
                "session": "late_us",
                "regime_name": "trend",
                "volume_ratio": 0.74,
                "spike_ratio": 1.72,
            },
            "hard_blocked": True,
        },
        "best_actionable_candidate": {
            "strategy": "momentum_continuation",
            "score": 0.22,
            "setup_quality_score": -0.18,
            "reasons": ["weak_consensus", "post_spike_entry"],
        },
        "contextual_ranking": {
            "top": [
                {"strategy": "momentum_continuation", "score": 0.22, "confidence": 0.31},
                {"strategy": "gold_micro_impulse", "score": 0.11, "confidence": 0.18},
            ]
        },
    }


def test_context_snapshot_uses_pre_entry_fields_only():
    snapshot = build_context_snapshot(_base_status())

    assert snapshot["symbol"] == "XAUUSD"
    assert snapshot["session"] == "late_us"
    assert snapshot["regime"] == "trend"
    assert snapshot["liquidity_state"] == "poor"
    assert snapshot["overstretched_entry_risk"] is True


def test_context_snapshot_schema_is_stable():
    snapshot = build_context_snapshot(_base_status())

    assert tuple(snapshot.keys()) == CONTEXT_SNAPSHOT_FIELDS


def test_context_snapshot_handles_missing_fields_safely():
    snapshot = build_context_snapshot({"updated_at": "2026-04-04T12:00:00+00:00"})

    assert tuple(snapshot.keys()) == CONTEXT_SNAPSHOT_FIELDS
    assert snapshot["symbol"] == ""
    assert snapshot["timestamp"] == "2026-04-04T12:00:00+00:00"
    assert snapshot["regime"] == "unknown"
    assert snapshot["session"] == "unknown"
    assert snapshot["trend_strength"] == 0.0
    assert snapshot["volatility_state"] == "unknown"
    assert snapshot["spread_state"] == "unknown"
    assert snapshot["liquidity_state"] == "unknown"
    assert snapshot["repeated_entry_risk"] is False
    assert snapshot["overstretched_entry_risk"] is False


def test_context_snapshot_derives_session_and_liquidity_when_runtime_context_exists():
    snapshot = build_context_snapshot(
        {
            "symbol": "XAUUSD",
            "updated_at": "2026-04-04T12:00:00+00:00",
            "risk": {"spread_points": 35.0},
            "regime": {"name": "trend", "trend_strength": 0.2, "volatility_pct": 0.03},
        }
    )

    assert snapshot["session"] != "unknown"
    assert snapshot["liquidity_state"] == "normal"


def test_context_snapshot_prefers_market_bar_timestamp_when_available():
    snapshot = build_context_snapshot(
        {
            **_base_status(),
            "updated_at": "2026-04-04T12:05:00+00:00",
            "market": {"recent_time": ["2026-04-04T11:59:00+00:00"]},
        }
    )

    assert snapshot["timestamp"] == "2026-04-04T11:59:00+00:00"


def test_decision_explanation_is_structured_and_read_only():
    status = _base_status()
    context_snapshot = build_context_snapshot(status)

    explanation = build_decision_explanation(status, context_snapshot, {"paper_rollout": {"lane": "control"}})

    assert explanation["final_decision"] == "block"
    assert explanation["lane"] == "control"
    assert any(code.startswith("consensus_guard_blocked_trade") for code in explanation["reason_codes"])
    assert status["action"] == "blocked"
    assert status["chosen"]["side"] == "buy"


def test_decision_trace_records_pipeline_without_mutating_status():
    status = _base_status()
    context_snapshot = build_context_snapshot(status)
    explanation = build_decision_explanation(status, context_snapshot, {"paper_rollout": {"lane": "control"}})
    status["decision_explanation"] = explanation
    status["same_side_support"] = {"side": "buy", "count": 1}

    trace = build_decision_trace(status, context_snapshot, {"execute_trades": False})

    assert trace["schema"] == "atlas_decision_trace_v1"
    assert trace["final_action"] == "blocked"
    assert trace["final_decision"] == "block"
    assert trace["selected_candidate"]["strategy"] == "momentum_continuation"
    assert trace["selected_candidate"]["side"] == "buy"
    assert trace["signal"]["summary"] == dict(status.get("signal_summary", {}) or {})
    assert trace["filters"]["reason_codes"]
    assert trace["execution_gate"]["execute_trades"] is False
    assert trace["blocker_class"] == "quality_gate"
    assert "read_only_observability" in trace["trace_notes"]
    assert status["action"] == "blocked"
    assert status["chosen"]["confidence"] == 0.31


def test_decision_trace_classifies_no_high_confidence_as_signal_quality():
    status = {
        **_base_status(),
        "action": "hold",
        "message": "No high-confidence trade found.",
        "chosen": {"strategy": "momentum_continuation", "side": "flat", "confidence": 0.44},
        "min_confidence_threshold": 0.44,
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "signal_quality"
    assert trace["blocker_summary"]["recoverability"] == "wait_for_setup_or_review_signal_fit"


def test_decision_trace_classifies_drawdown_as_safety_gate():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Risk guard active: Weekly drawdown limit reached. No trade executed.",
        "chosen": {"strategy": "gold_liquidity_sweep", "side": "sell", "confidence": 0.62},
        "risk": {"weekly_loss_pct": 0.061, "daily_loss_pct": 0.0, "open_positions": 0, "spread_points": 30.0},
    }
    context_snapshot = build_context_snapshot(status)
    explanation = build_decision_explanation(status, context_snapshot, {"execute_trades": True})
    trace = build_decision_trace({**status, "decision_explanation": explanation}, context_snapshot, {"execute_trades": True})
    assert explanation["blocker_class"] == "safety_gate"
    assert explanation["active_guards"][0]["name"] == "risk_guard"
    assert explanation["active_guards"][0]["hard_blocked"] is True
    assert trace["blocker_class"] == "safety_gate"
    assert trace["filters"]["hard_blockers"][0]["name"] == "risk_guard"
    assert trace["blocker_summary"]["risk"]["weekly_loss_pct"] == 0.061


def test_decision_trace_exposes_bias_breakdown_for_signal_structure_conflict():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Directional-structure guard blocked trade: chosen=buy, structure=LH/LL, location=mid_range, momentum=mixed.",
        "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.7},
        "chart_features": {"trend_structure": "LH/LL", "trend_state": "downtrend"},
        "higher_timeframe": {"confirm_tf": "M5", "bias": "buy"},
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["bias_breakdown"]["signal_bias"] == "buy"
    assert trace["bias_breakdown"]["chart_structure_bias"] == "sell"
    assert trace["bias_breakdown"]["higher_timeframe_bias"] == "buy"
    assert trace["bias_breakdown"]["final_trade_bias"] == "neutral"
    assert trace["bias_breakdown"]["conflict"] is True
    assert trace["filters"]["hard_blockers"][0]["name"] == "directional_structure_guard"


def test_decision_trace_classifies_safety_lock_as_safety_gate_even_when_flat():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Risk guard active: Safety lock active: consecutive_loss_threshold. No trade executed.",
        "chosen": {"strategy": "momentum_continuation", "side": "flat", "confidence": 0.0},
        "decision_explanation": {"reason_codes": ["flat_signal"]},
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "safety_gate"
    assert trace["blocker_summary"]["recoverability"] == "wait_or_operator_review"


def test_decision_trace_classifies_directional_structure_guard_as_quality_gate():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Directional-structure guard blocked trade: chosen=sell, structure=HH/HL, location=mid_range, momentum=mixed.",
        "chosen": {"strategy": "momentum_continuation", "side": "sell", "confidence": 0.7},
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "quality_gate"


def test_decision_trace_classifies_higher_timeframe_alignment_as_quality_gate():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
        "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.7},
        "higher_timeframe": {"confirm_tf": "M5", "bias": "sell"},
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "quality_gate"
    assert trace["blocker_summary"]["recoverability"] == "review_gate_or_wait_for_cleaner_setup"


def test_decision_trace_classifies_heartbeat_as_runtime_or_data():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Trading blocked: worker heartbeat is stale (heartbeat_stale).",
        "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.7},
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "runtime_or_data"
    assert trace["blocker_recoverability"] == "fix_runtime"


def test_decision_trace_classifies_adaptive_guard_as_safety_gate():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Adaptive guard reduced confidence below execution threshold.",
        "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.44},
        "min_confidence_threshold": 0.44,
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "safety_gate"
    assert trace["blocker_summary"]["recoverability"] == "wait_or_review_recent_loss_adaptation"


def test_decision_trace_classifies_online_learning_as_quality_gate():
    status = {
        **_base_status(),
        "action": "blocked",
        "message": "Online learning reduced confidence below execution threshold.",
        "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.43},
        "min_confidence_threshold": 0.44,
        "online_learning_influence": {
            "applied": True,
            "allowed_influence": "bounded_confidence_haircut",
            "haircut": 0.05,
        },
    }
    context_snapshot = build_context_snapshot(status)
    trace = build_decision_trace(status, context_snapshot, {"execute_trades": True})
    assert trace["blocker_class"] == "quality_gate"
    assert trace["blocker_summary"]["recoverability"] == "wait_for_cleaner_setup_or_review_learning_evidence"


def test_system_health_reports_paper_and_ml_state():
    status = _base_status()
    health = build_system_health(
        status,
        SimpleNamespace(
            timeframe_minutes=1,
            execute_trades=False,
            meta_model_enabled=False,
            gemini_influence_enabled=False,
            research_regime_filter_enabled=False,
        ),
        runtime_flags={"execute_trades": False, "gemini_influence_enabled": False},
    )

    assert health["effective_mode"] == "paper"
    assert health["paper_mode_confirmed"] is True
    assert health["ml_state"]["meta_model_enabled"] is False
    assert health["broker_connectivity_status"]["connected"] is True
    assert health["self_improvement"]["proposal_count"] == 0


def test_system_health_warns_when_llm_provider_is_in_quota_cooldown():
    status = _base_status()
    status["health"]["llm"] = {
        "enabled": True,
        "providers": [
            {
                "name": "openai",
                "in_cooldown": True,
                "cooldown_seconds_left": 1200,
                "last_error_kind": "quota_exhausted",
            }
        ],
    }
    health = build_system_health(
        status,
        SimpleNamespace(
            timeframe_minutes=1,
            execute_trades=False,
            meta_model_enabled=True,
            gemini_influence_enabled=True,
            research_regime_filter_enabled=False,
        ),
        runtime_flags={"execute_trades": False, "gemini_influence_enabled": True},
    )

    assert "llm_openai_quota_exhausted" in health["warnings"]
    assert "llm_openai_quota_exhausted" in health["classification"]["WARNING"]


def test_freshness_runtime_and_system_health_share_same_stale_contract():
    status = _base_status()
    settings = SimpleNamespace(
        timeframe_minutes=1,
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
    )

    freshness = build_freshness_runtime(status, settings)
    health = build_system_health(
        status,
        settings,
        runtime_flags={"execute_trades": False, "gemini_influence_enabled": False},
    )

    assert freshness["state"] == "fresh"
    assert freshness["is_stale"] is False
    assert freshness["stale_threshold_ms"] == 480000.0
    assert health["stale_data_detected"] is False
    assert health["freshness_runtime"]["state"] == freshness["state"]
    assert health["freshness_runtime"]["stale_threshold_ms"] == freshness["stale_threshold_ms"]


def test_system_health_explains_settings_env_mismatch_when_runtime_and_startup_align():
    status = _base_status()
    status["execute_trades"] = True
    health = build_system_health(
        status,
        SimpleNamespace(
            timeframe_minutes=1,
            execute_trades=False,
            meta_model_enabled=False,
            gemini_influence_enabled=False,
            research_regime_filter_enabled=False,
        ),
        runtime_flags={
            "execute_trades": True,
            "gemini_influence_enabled": False,
            "startup_verification": {
                "execute_trades": True,
                "startup_message": "ATLAS STARTED IN GUARDED DEMO MODE",
            },
        },
    )

    assert health["config_mismatch_detected"] is True
    assert health["mode_truth"]["status_execute_trades"] is True
    assert health["mode_truth"]["settings_execute_trades"] is False
    assert health["mode_truth"]["startup_execute_trades"] is True
    assert "persisted settings/env value for EXECUTE_TRADES does not match" in health["mode_truth"]["explanation"]


def test_shadow_advice_is_observer_only():
    status = _base_status()
    context_snapshot = build_context_snapshot(status)
    explanation = build_decision_explanation(status, context_snapshot, {})

    advice = build_shadow_advice(status, context_snapshot, explanation)

    assert advice["recommendation"] == "observe"
    assert "blocked_trade" in advice["anomaly_flags"]


def test_shadow_advice_schema_is_stable():
    status = _base_status()
    context_snapshot = build_context_snapshot(status)
    explanation = build_decision_explanation(status, context_snapshot, {})

    advice = build_shadow_advice(status, context_snapshot, explanation)

    assert tuple(advice.keys()) == SHADOW_ADVICE_FIELDS


def test_shadow_advice_handles_missing_inputs_safely():
    advice = build_shadow_advice({}, {}, {})

    assert tuple(advice.keys()) == SHADOW_ADVICE_FIELDS
    assert advice["trade_id"] == ""
    assert advice["setup_score"] == 0.0
    assert advice["confidence"] == 0.0
    assert advice["anomaly_flags"] == []
    assert advice["alternative_ranking"] == []
    assert advice["recommendation"] == "observe"


def test_shadow_advice_prefers_trade_id_then_candidate_id():
    status = _base_status()
    context_snapshot = build_context_snapshot(status)
    explanation = build_decision_explanation(status, context_snapshot, {})
    explanation["trade_id"] = "dec-123"
    explanation["candidate_id"] = "cand-123"

    advice = build_shadow_advice(status, context_snapshot, explanation)

    assert advice["trade_id"] == "dec-123"


def test_trader_status_augmentation_preserves_decision_fields():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe_minutes=1,
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
    )
    trader._last_runtime_flags = {"paper_rollout": {"lane": "control"}}

    status = _base_status()
    chosen_before = dict(status["chosen"])
    action_before = status["action"]

    augmented = AtlasTrader._augment_status_layers(trader, status)

    assert augmented["action"] == action_before
    assert augmented["chosen"] == chosen_before
    assert augmented["lane"] == "control"
    assert "decision_explanation" in augmented
    assert "decision_trace" in augmented
    assert augmented["decision_trace"]["schema"] == "atlas_decision_trace_v1"
    assert augmented["decision_trace"]["selected_candidate"]["strategy"] == "momentum_continuation"
    assert "context_snapshot" in augmented
    assert "setup_hunter" in augmented
    assert augmented["setup_hunter"]["mode"] == "proactive_scan"
    assert "system_health" in augmented
    assert "shadow_advice" in augmented
    assert "self_improvement_state" in augmented


def test_trader_status_augmentation_surfaces_mode_truth_top_level():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe_minutes=1,
        execute_trades=True,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
    )
    trader._last_runtime_flags = {
        "execute_trades": True,
        "startup_verification": {
            "execute_trades": True,
            "gate_execution_environment": "demo",
            "startup_message": "ATLAS STARTED IN GUARDED DEMO MODE",
        },
    }
    trader.execution_gate = SimpleNamespace(
        snapshot=lambda: {
            "execution_environment": "demo",
            "execution_environment_label": "Live Demo",
            "execution_gate_enabled": True,
            "rollout_stage": "limited_live_pilot",
            "adapter_type": "MT5BrokerAdapter",
        }
    )

    status = {
        **_base_status(),
        "execute_trades": True,
        "system_health": {"warnings": []},
    }

    augmented = AtlasTrader._augment_status_layers(trader, status)

    assert augmented["effective_mode"] == "live"
    assert augmented["effective_mode_label"] in {"LIVE", "GUARDED_DEMO"}
    assert augmented["paper_mode_confirmed"] is False
    assert augmented["gate_execution_environment"] == "demo"
    assert augmented["system_health"]["mode_truth"]["gate_execution_environment"] == "demo"
    assert augmented["system_health"]["mode_truth"]["execution_gate_enabled"] is True


def test_knowledge_store_loads_layer_sections(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "context_snapshot": {"session": "london"},
            "decision_explanation": {"final_decision": "allow"},
            "decision_trace": {"schema": "atlas_decision_trace_v1"},
            "system_health": {"effective_mode": "paper"},
            "shadow_advice": {"recommendation": "observe"},
        }
    )

    assert store.load_context_snapshot()["session"] == "london"
    assert store.load_decision_explanation()["final_decision"] == "allow"
    assert store.load_decision_trace()["schema"] == "atlas_decision_trace_v1"
    assert store.load_system_health()["effective_mode"] == "paper"
    assert store.load_shadow_advice()["recommendation"] == "observe"


def test_self_improvement_bundle_and_summary_load_defaults(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    bundle = load_self_improvement_bundle(store)
    summary = build_self_improvement_summary(bundle)

    assert "self_improvement_policy.json" in bundle
    assert summary["proposal_count"] == 0
    assert summary["approval_pending_count"] == 0
    assert summary["active_rule_version"] == "atlas_rules_v1"


def test_self_improvement_bundle_uses_isolated_store_knowledge(tmp_path):
    store = KnowledgeStore(str(tmp_path / "isolated_storage"))

    bundle = load_self_improvement_bundle(store)
    summary = build_self_improvement_summary(bundle)

    assert summary["proposal_count"] == 0
    assert bundle["proposal_queue.json"]["proposals"] == []
    assert (tmp_path / "isolated_storage" / "knowledge" / "proposal_queue.json").exists()


def test_self_improvement_bundle_without_store_uses_global_knowledge() -> None:
    global_queue = Path("knowledge") / "proposal_queue.json"
    expected = json.loads(global_queue.read_text(encoding="utf-8"))

    bundle = load_self_improvement_bundle()

    assert bundle["proposal_queue.json"]["proposals"] == expected.get("proposals", [])
    assert build_self_improvement_summary(bundle)["proposal_count"] == len(expected.get("proposals", []))


def test_self_improvement_summary_exposes_pending_proposal_details(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    append_or_update_proposal(
        store,
        {
            "proposal_id": "proposal_pending",
            "status": "awaiting_approval",
            "category": "filters",
            "scope": {"target_blocker": "Consensus guard blocked trade."},
        },
    )

    summary = build_self_improvement_summary(load_self_improvement_bundle(store))

    assert summary["approval_pending_count"] == 1
    assert summary["pending_proposal_id"] == "proposal_pending"
    assert summary["pending_proposal_category"] == "filters"
    assert summary["pending_proposal_target_blocker"] == "Consensus guard blocked trade."


def test_knowledge_store_self_improvement_audit_and_snapshot_helpers(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    store.append_self_improvement_audit({"event": "proposal_created", "proposal_id": "sip_1"})
    rows = store.tail_self_improvement_audit(limit=10)
    snapshot = store.save_rule_snapshot("snapshot_a", {"rule": "demo"})

    assert rows[-1]["event"] == "proposal_created"
    assert rows[-1]["proposal_id"] == "sip_1"
    assert snapshot.exists()
    assert store.load_rule_snapshot("snapshot_a")["rule"] == "demo"


def test_self_improvement_evidence_scoring_is_bounded():
    result = score_evidence_strength(
        sample_size_score=0.8,
        persistence_score=0.7,
        cross_window_score=0.75,
        regime_alignment_score=0.8,
        risk_impact_score=0.6,
        overreaction_risk_score=0.2,
    )

    assert 0.0 <= float(result["strength"]) <= 1.0
    assert result["grade"] in {"weak", "moderate", "strong"}
