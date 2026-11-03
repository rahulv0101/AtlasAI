from __future__ import annotations

import json
from types import SimpleNamespace

from atlas_ai.learning.missed_entry_audit import (
    backfill_missed_entry_collection_context,
    backfill_missed_entry_audits_from_events,
    build_consensus_selectivity_operator_review_packet,
    build_blocker_selectivity_collection_plan,
    build_blocker_selectivity_collection_readiness,
    build_blocker_selectivity_offline_collector,
    build_blocker_selectivity_replay,
    build_blocker_selectivity_review,
    build_missed_entry_audit,
    record_missed_entry_audit,
    review_missed_entry_outcomes,
    summarize_missed_entry_learning,
    sync_blocker_selectivity_proposal,
    validate_blocker_selectivity_richer_context_replay,
    validate_blocker_selectivity_shadow_compare,
)
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import append_or_update_proposal
from atlas_ai.self_improvement import load_self_improvement_bundle
from atlas_ai.trader import AtlasTrader


def _missed_status() -> dict:
    return {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "action": "blocked",
        "message": "M1 loss-pattern block: repeated sell losses.",
        "chosen": {"strategy": "gold_breakout_retest", "side": "sell", "confidence": 0.64},
        "market": {"last_close": 4500.0},
        "best_actionable_candidate": {
            "strategy": "gold_breakout_retest",
            "side": "sell",
            "confidence": 0.63,
            "score": 0.58,
            "advisory_setup_quality_score": 75.0,
            "reasons": ["htf_aligned"],
        },
        "decision_explanation": {
            "final_decision": "block",
            "primary_blocker": "loss_learning",
            "blocker_class": "safety_gate",
            "reason_codes": ["loss_pattern_block"],
        },
        "decision_analysis": {
            "confluence_score": 57,
            "confluence_threshold": 52,
            "decision_protocol": {"act": {"execution_request_eligible": False}},
        },
        "context_snapshot": {
            "regime": "trend",
            "trend_state": "downtrend",
            "trend_structure": "LH/LL",
            "momentum": "mixed",
            "location": "mid_range",
            "session": "new_york",
            "spread_points": 35,
            "spread_state": "normal",
            "loss_pattern_state": "active",
        },
        "higher_timeframe": {"confirm_tf": "M5", "bias": "sell"},
        "signal_summary": {"raw_count": 8, "active_count": 6, "evolved_active_count": 1},
        "learning_status": {
            "demo_chromosome_active_count": 1,
            "demo_chromosome_active_ids": ["demo_1"],
        },
    }


def test_missed_entry_audit_records_strong_blocked_setup(tmp_path) -> None:
    result = record_missed_entry_audit(tmp_path, _missed_status())

    assert result["recorded"] is True
    assert result["audit"]["fingerprint"]
    assert result["audit"]["policy"]["audit_only"] is True
    assert result["audit"]["signal_visibility"]["demo_chromosome_active_count"] == 1
    assert result["audit"]["collection_context"]["schema"] == "atlas_blocker_collection_context_v1"
    assert result["audit"]["collection_context"]["higher_timeframe_alignment"] is True
    assert result["audit"]["collection_context"]["directional_structure_aligned"] is True
    assert result["audit"]["collection_context"]["spread_state"] == "normal"
    assert result["audit"]["collection_context"]["loss_pattern_state"] == "active"
    assert "validate_loss_pattern_block_against_next_price_move" in result["audit"]["learning_focus"]
    assert KnowledgeStore(str(tmp_path)).tail_missed_entry_audits(limit=5)[0]["schema"] == "atlas_missed_entry_audit_v1"


def test_missed_entry_audit_dedupes_recent_same_pattern(tmp_path) -> None:
    first = record_missed_entry_audit(tmp_path, _missed_status())
    second = record_missed_entry_audit(tmp_path, _missed_status())

    assert first["recorded"] is True
    assert second["recorded"] is False
    assert second["deduped"] is True
    assert len(KnowledgeStore(str(tmp_path)).tail_missed_entry_audits(limit=10)) == 1


def test_missed_entry_audit_records_changed_pattern(tmp_path) -> None:
    first = record_missed_entry_audit(tmp_path, _missed_status())
    changed = _missed_status()
    changed["message"] = "No directional trade signal selected."
    changed["decision_explanation"]["primary_blocker"] = "No directional trade signal selected."
    second = record_missed_entry_audit(tmp_path, changed)

    assert first["recorded"] is True
    assert second["recorded"] is True
    assert len(KnowledgeStore(str(tmp_path)).tail_missed_entry_audits(limit=10)) == 2


def test_missed_entry_audit_ignores_weak_flat_hold(tmp_path) -> None:
    result = record_missed_entry_audit(
        tmp_path,
        {
            "action": "hold",
            "message": "No directional trade signal selected.",
            "chosen": {"side": "flat", "confidence": 0.1},
            "decision_analysis": {"confluence_score": 12, "confluence_threshold": 52},
        },
    )

    assert result["recorded"] is False
    assert not (tmp_path / "missed_entry_audits.jsonl").exists()


def test_missed_entry_audit_records_consensus_blocker_from_blocker_side(tmp_path) -> None:
    result = record_missed_entry_audit(
        tmp_path,
        {
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=sell, support=1, required=2.",
            "chosen": {"side": "flat", "confidence": 0.0},
            "best_actionable_candidate": {"side": "flat", "confidence": 0.0, "score": 0.0},
            "market": {"last_close": 4500.0},
            "decision_explanation": {
                "primary_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2.",
                "blocker_class": "consensus_guard",
            },
            "context_snapshot": {
                "trend_state": "downtrend",
                "trend_structure": "LH/LL",
                "momentum": "strong",
                "spread_state": "normal",
                "loss_pattern_state": "inactive",
            },
            "higher_timeframe": {"bias": "sell"},
        },
    )

    assert result["recorded"] is True
    audit = result["audit"]
    assert audit["best_actionable_candidate"]["side"] == "flat"
    assert audit["collection_context"]["side"] == "sell"
    assert audit["collection_context"]["higher_timeframe_alignment"] is True
    assert audit["policy"]["does_not_execute"] is True


def test_trader_publish_status_attaches_missed_entry_audit(tmp_path) -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        storage_dir=str(tmp_path),
        timeframe_minutes=1,
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
    )
    trader.store = KnowledgeStore(str(tmp_path))
    trader._last_runtime_flags = {}
    trader._current_cycle_timing = {}

    status = _missed_status()
    AtlasTrader._publish_status(trader, status)

    saved = trader.store.load_status()
    assert saved["missed_entry_audit"]["recorded"] is True
    assert saved["missed_entry_outcome_review"]["reviewed"] is True
    assert saved["blocker_selectivity_review"]["schema"] == "atlas_blocker_selectivity_review_v1"
    assert saved["blocker_selectivity_replay"]["schema"] == "atlas_blocker_selectivity_replay_v1"
    assert saved["blocker_selectivity_collection_plan"]["schema"] == "atlas_blocker_selectivity_collection_plan_v1"
    assert saved["blocker_selectivity_offline_collector"]["schema"] == "atlas_blocker_selectivity_offline_collector_v1"
    assert saved["blocker_selectivity_collection_readiness"]["schema"] == "atlas_blocker_selectivity_collection_readiness_v1"
    assert saved["blocker_selectivity_shadow_compare_retry"]["attempted"] is False
    assert saved["blocker_selectivity_shadow_compare_retry"]["reason"] == "collection_readiness_not_met"
    assert trader.store.tail_missed_entry_audits(limit=5)


def test_trader_publish_status_retries_shadow_compare_when_offline_collector_ready(monkeypatch, tmp_path) -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        storage_dir=str(tmp_path),
        timeframe_minutes=1,
        execute_trades=False,
        meta_model_enabled=False,
        gemini_influence_enabled=False,
        research_regime_filter_enabled=False,
    )
    trader.store = KnowledgeStore(str(tmp_path))
    trader._last_runtime_flags = {}
    trader._current_cycle_timing = {}

    monkeypatch.setattr(
        "atlas_ai.trader.build_blocker_selectivity_offline_collector",
        lambda _storage_dir: {
            "schema": "atlas_blocker_selectivity_offline_collector_v1",
            "ready_to_retry_shadow_compare": True,
            "remaining_enriched_samples_needed": 0,
            "policy": {
                "does_not_execute": True,
                "does_not_change_strategy_logic": True,
                "does_not_change_risk_policy": True,
                "does_not_change_execution_policy": True,
            },
        },
    )
    monkeypatch.setattr(
        "atlas_ai.trader.validate_blocker_selectivity_shadow_compare",
        lambda _store: {
            "ok": True,
            "missing": [],
            "report": {
                "comparison": {"verdict": "shadow_compare_passed"},
                "policy": {"shadow_review_only": True, "does_not_execute": True},
            },
        },
    )

    AtlasTrader._publish_status(trader, _missed_status())

    saved = trader.store.load_status()
    retry = saved["blocker_selectivity_shadow_compare_retry"]
    assert saved["blocker_selectivity_collection_readiness"]["ready_to_retry_shadow_compare"] is False
    assert retry["attempted"] is True
    assert retry["ok"] is True
    assert retry["source"] == "offline_collector"
    assert retry["policy"]["does_not_execute"] is True


def test_missed_entry_outcome_review_labels_missed_opportunity(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status())

    review = review_missed_entry_outcomes(
        tmp_path,
        {"market": {"last_close": 4498.5}},
        min_move_points=80.0,
    )

    assert review["reviewed"] is True
    assert review["summary"]["missed_opportunity"] == 1
    assert review["reviews"][0]["side"] == "sell"


def test_missed_entry_outcome_review_labels_correct_block(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status())

    review = review_missed_entry_outcomes(
        tmp_path,
        {"market": {"last_close": 4501.2}},
        min_move_points=80.0,
    )

    assert review["summary"]["correct_hold_or_block"] == 1


def test_missed_entry_learning_summary_prioritizes_missed_blocker(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    changed = _missed_status()
    changed["message"] = "No directional trade signal selected."
    changed["decision_explanation"]["primary_blocker"] = "No directional trade signal selected."
    record_missed_entry_audit(tmp_path, changed, dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)

    summary = summarize_missed_entry_learning(tmp_path)

    assert summary["available"] is True
    assert summary["policy"]["summary_only"] is True
    assert summary["top_repair_candidate"]["primary_blocker"] == "loss_learning"
    assert summary["top_repair_candidate"]["repair_priority"] in {"medium", "high"}


def test_store_loads_missed_entry_learning_summary(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status())
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)

    summary = KnowledgeStore(str(tmp_path)).load_missed_entry_learning_summary()

    assert summary["schema"] == "atlas_missed_entry_learning_summary_v1"
    assert summary["missed_opportunity"] >= 1


def test_blocker_selectivity_review_builds_non_executing_repair_plan(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.0}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)

    review = build_blocker_selectivity_review(tmp_path, primary_blocker="loss_learning")

    assert review["schema"] == "atlas_blocker_selectivity_review_v1"
    assert review["available"] is True
    assert review["target_blocker"] == "loss_learning"
    assert review["outcome_counts"]["missed_opportunity"] >= 1
    assert review["proposal"]["type"] == "offline_selectivity_review"
    assert review["policy"]["does_not_execute"] is True
    assert review["policy"]["does_not_change_strategy_logic"] is True
    assert (tmp_path / "blocker_selectivity_review.json").exists()


def test_store_loads_blocker_selectivity_review(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status())
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)

    review = KnowledgeStore(str(tmp_path)).load_blocker_selectivity_review(primary_blocker="loss_learning")

    assert review["schema"] == "atlas_blocker_selectivity_review_v1"
    assert review["policy"]["review_only"] is True


def test_blocker_selectivity_replay_builds_shadow_only_candidate(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4497.9}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)

    replay = build_blocker_selectivity_replay(tmp_path, primary_blocker="loss_learning")

    assert replay["schema"] == "atlas_blocker_selectivity_replay_v1"
    assert replay["available"] is True
    assert replay["mode"] == "shadow_review_only"
    assert replay["candidate_rule"]["does_not_allow_trade"] is True
    assert replay["candidate_rule"]["does_not_lower_min_same_side_signals"] is True
    assert replay["metrics"]["missed_opportunity_capture_count"] >= 1
    assert replay["policy"]["does_not_lower_thresholds"] is True
    assert (tmp_path / "blocker_selectivity_replay.json").exists()


def test_store_loads_blocker_selectivity_replay(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status())
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)

    replay = KnowledgeStore(str(tmp_path)).load_blocker_selectivity_replay(primary_blocker="loss_learning")

    assert replay["schema"] == "atlas_blocker_selectivity_replay_v1"
    assert replay["policy"]["shadow_review_only"] is True


def test_sync_blocker_selectivity_proposal_creates_draft_without_active_rule_change(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    before_registry = load_self_improvement_bundle(store)["active_rule_registry.json"]
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4497.9}}, min_move_points=80.0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)

    result = sync_blocker_selectivity_proposal(store, primary_blocker="loss_learning")

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    after_registry = bundle["active_rule_registry.json"]
    audit = store.tail_self_improvement_audit(limit=20)

    assert result["ok"] is True
    assert proposal["proposal_id"] == "proposal_consensus_selectivity_context_review"
    assert proposal["status"] == "draft"
    assert proposal["category"] == "filters"
    assert proposal["proposed_change"]["does_not_allow_trade"] is True
    assert proposal["policy"]["proposal_only"] is True
    assert after_registry == before_registry
    assert any(row.get("proposal_id") == proposal["proposal_id"] for row in audit)


def test_validate_blocker_selectivity_richer_context_replay_only_completes_one_step(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    for _ in range(30):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    for _ in range(11):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)
    sync_blocker_selectivity_proposal(store, primary_blocker="loss_learning")

    result = validate_blocker_selectivity_richer_context_replay(store)

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    report = result["report"]

    assert result["ok"] is True
    assert proposal["status"] == "draft"
    assert "richer_context_replay" in proposal["validation"]["completed"]
    assert "shadow_compare" in proposal["validation"]["missing"]
    assert "regression_tests" in proposal["validation"]["missing"]
    assert "operator_review" in proposal["validation"]["missing"]
    assert report["promotion_readiness"]["ready_for_rollout_review"] is False
    assert report["policy"]["does_not_lower_thresholds"] is True


def test_validate_blocker_selectivity_shadow_compare_keeps_proposal_draft(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    for _ in range(30):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    for _ in range(11):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)
    sync_blocker_selectivity_proposal(store, primary_blocker="loss_learning")
    validate_blocker_selectivity_richer_context_replay(store)

    result = validate_blocker_selectivity_shadow_compare(store)

    bundle = load_self_improvement_bundle(store)
    proposal = bundle["proposal_queue.json"]["proposals"][-1]
    report = result["report"]

    assert result["ok"] is True
    assert proposal["status"] == "draft"
    assert "richer_context_replay" in proposal["validation"]["completed"]
    assert "shadow_compare" in proposal["validation"]["completed"]
    assert "regression_tests" in proposal["validation"]["missing"]
    assert "operator_review" in proposal["validation"]["missing"]
    assert report["comparison"]["decision"] == "pass"
    assert report["promotion_readiness"]["ready_for_rollout_review"] is False
    assert report["policy"]["does_not_allow_trade"] is True


def test_blocker_selectivity_collection_plan_after_failed_shadow_compare(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    for _ in range(10):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    for _ in range(24):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4501.2}}, min_move_points=80.0)
    for _ in range(40):
        review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4499.8}}, min_move_points=80.0)
    sync_blocker_selectivity_proposal(store, primary_blocker="loss_learning")
    validate_blocker_selectivity_richer_context_replay(store)
    shadow = validate_blocker_selectivity_shadow_compare(store)

    plan = build_blocker_selectivity_collection_plan(store)
    store_plan = store.load_blocker_selectivity_collection_plan()

    assert shadow["ok"] is False
    assert plan["schema"] == "atlas_blocker_selectivity_collection_plan_v1"
    assert "false_unlock_risk_exceeds_capture" in plan["failure_reasons"]
    assert plan["retry_requirements"]["capture_must_exceed_false_unlock"] is True
    assert plan["policy"]["does_not_lower_thresholds"] is True
    assert store_plan["schema"] == "atlas_blocker_selectivity_collection_plan_v1"
    assert (tmp_path / "blocker_selectivity_collection_plan.json").exists()


def test_blocker_selectivity_collection_readiness_counts_enriched_rows(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    changed = _missed_status()
    changed["message"] = "No directional trade signal selected."
    changed["decision_explanation"]["primary_blocker"] = "No directional trade signal selected."
    record_missed_entry_audit(tmp_path, changed, dedupe_window_seconds=0)
    build_blocker_selectivity_collection_plan(KnowledgeStore(str(tmp_path)))

    readiness = build_blocker_selectivity_collection_readiness(tmp_path)
    store_readiness = KnowledgeStore(str(tmp_path)).load_blocker_selectivity_collection_readiness()

    assert readiness["schema"] == "atlas_blocker_selectivity_collection_readiness_v1"
    assert readiness["enriched_samples"] >= 2
    assert readiness["ready_to_retry_shadow_compare"] is False
    assert readiness["remaining_enriched_samples_needed"] > 0
    assert readiness["policy"]["readiness_only"] is True
    assert store_readiness["schema"] == "atlas_blocker_selectivity_collection_readiness_v1"
    assert (tmp_path / "blocker_selectivity_collection_readiness.json").exists()


def test_backfill_missed_entry_collection_context_updates_old_rows(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    path = tmp_path / "missed_entry_audits.jsonl"
    row = __import__("json").loads(path.read_text(encoding="utf-8").splitlines()[0])
    row.pop("collection_context", None)
    path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")

    result = backfill_missed_entry_collection_context(tmp_path)
    readiness = build_blocker_selectivity_collection_readiness(tmp_path)
    store_result = KnowledgeStore(str(tmp_path)).backfill_missed_entry_collection_context()

    updated = __import__("json").loads(path.read_text(encoding="utf-8").splitlines()[0])
    ctx = updated["collection_context"]

    assert result["ok"] is True
    assert result["updated"] >= 1
    assert ctx["schema"] == "atlas_blocker_collection_context_v1"
    assert ctx["backfilled"] is True
    assert ctx["directional_structure"] == "lh/ll"
    assert ctx["momentum_quality"] == "mixed"
    assert ctx["spread_state"] == "normal"
    assert readiness["enriched_samples"] >= 1
    assert store_result["schema"] == "atlas_missed_entry_collection_context_backfill_v1"


def test_backfill_marks_non_loss_blocker_loss_pattern_inactive(tmp_path) -> None:
    status = _missed_status()
    status["message"] = "No directional trade signal selected."
    status["decision_explanation"]["primary_blocker"] = "No directional trade signal selected."
    record_missed_entry_audit(tmp_path, status, dedupe_window_seconds=0)
    path = tmp_path / "missed_entry_audits.jsonl"
    row = __import__("json").loads(path.read_text(encoding="utf-8").splitlines()[0])
    row.pop("collection_context", None)
    path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")

    backfill_missed_entry_collection_context(tmp_path)

    updated = __import__("json").loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert updated["collection_context"]["loss_pattern_state"] == "inactive"


def test_blocker_selectivity_offline_collector_mines_real_stored_samples_only(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)

    collector = build_blocker_selectivity_offline_collector(tmp_path, min_samples=3)
    store_collector = KnowledgeStore(str(tmp_path)).load_blocker_selectivity_offline_collector()

    assert collector["schema"] == "atlas_blocker_selectivity_offline_collector_v1"
    assert collector["mode"] == "offline_collection_only"
    assert collector["all_time_audit_samples"] == 1
    assert collector["enriched_samples"] == 1
    assert collector["reviewed_outcome_samples"] == 1
    assert collector["outcome_counts"]["missed_opportunity"] == 1
    assert collector["ready_to_retry_shadow_compare"] is False
    assert collector["remaining_enriched_samples_needed"] == 2
    assert collector["policy"]["synthetic_samples_created"] is False
    assert collector["policy"]["does_not_execute"] is True
    assert store_collector["schema"] == "atlas_blocker_selectivity_offline_collector_v1"
    assert (tmp_path / "blocker_selectivity_offline_collector.json").exists()


def test_blocker_selectivity_offline_collector_can_mark_ready_without_policy_change(tmp_path) -> None:
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4497.0}}, min_move_points=80.0)

    collector = build_blocker_selectivity_offline_collector(tmp_path, min_samples=1)

    assert collector["ready_to_retry_shadow_compare"] is True
    assert collector["policy"]["does_not_change_strategy_logic"] is True
    assert collector["policy"]["does_not_lower_thresholds"] is True


def test_event_backfill_records_real_consensus_blocker_metadata(tmp_path) -> None:
    event = {
        "time": "2026-06-03T08:00:00+00:00",
        "type": "risk",
        "message": "Consensus guard blocked trade: side=sell, support=1, required=2.",
        "meta": {
            **_missed_status(),
            "message": "Consensus guard blocked trade: side=sell, support=1, required=2.",
            "decision_explanation": {
                "primary_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2.",
                "blocker_class": "consensus_guard",
            },
        },
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = backfill_missed_entry_audits_from_events(tmp_path)
    store_result = KnowledgeStore(str(tmp_path)).backfill_missed_entry_audits_from_events()
    audits = KnowledgeStore(str(tmp_path)).tail_missed_entry_audits(limit=5)

    assert result["schema"] == "atlas_missed_entry_event_backfill_v1"
    assert result["appended"] == 1
    assert store_result["skipped_duplicate"] == 1
    assert audits[0]["source_event"]["backfilled"] is True
    assert audits[0]["primary_blocker"] == "Consensus guard blocked trade: side=sell, support=1, required=2."
    assert audits[0]["policy"]["does_not_execute"] is True
    assert result["policy"]["synthetic_samples_created"] is False


def test_event_backfill_ignores_non_candidate_events(tmp_path) -> None:
    event = {
        "time": "2026-06-03T08:00:00+00:00",
        "type": "system",
        "message": "Broker connected. Live loop started.",
        "meta": {"action": "started"},
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    result = backfill_missed_entry_audits_from_events(tmp_path)

    assert result["appended"] == 0
    assert (tmp_path / "missed_entry_audits.jsonl").read_text(encoding="utf-8") == ""


def test_operator_review_packet_does_not_approve_or_activate(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    record_missed_entry_audit(tmp_path, _missed_status(), dedupe_window_seconds=0)
    review_missed_entry_outcomes(tmp_path, {"market": {"last_close": 4498.5}}, min_move_points=80.0)
    append_or_update_proposal(
        store,
        {
            "proposal_id": "proposal_consensus_selectivity_context_review",
            "status": "awaiting_approval",
            "category": "filters",
            "scope": {"target_blocker": "loss_learning"},
            "current_behavior": {"summary": "Consensus guard requires same-side support."},
            "proposed_change": {"type": "context_confirmation_candidate", "does_not_allow_trade": True},
            "reason": "review validated blocker selectivity evidence",
            "validation": {
                "required": ["richer_context_replay", "shadow_compare", "regression_tests", "operator_review"],
                "completed": ["richer_context_replay", "shadow_compare", "regression_tests"],
                "missing": ["operator_review"],
            },
            "approval": {"required": True, "approved": False, "approved_by": "", "approved_at": ""},
        },
    )

    packet = build_consensus_selectivity_operator_review_packet(store)

    assert packet["schema"] == "atlas_consensus_selectivity_operator_review_packet_v1"
    assert packet["approval"]["approved"] is False
    assert packet["operator_posture"]["does_not_approve"] is True
    assert packet["operator_posture"]["approval_still_requires_explicit_operator_command"] is True
    assert packet["operator_posture"]["suggested_operator_action"] in {
        "defer",
        "consider_guarded_rollout_approval",
        "review_packet_then_consider_guarded_rollout_approval",
        "reject_or_defer",
    }
    assert packet["operator_posture"]["posture"] in {
        "defer_until_validation_complete",
        "guarded_rollout_candidate",
        "guarded_rollout_candidate_with_replay_caution",
        "reject_or_defer_candidate",
        "defer_for_more_evidence",
    }
    assert packet["safety"]["does_not_approve"] is True
    assert packet["safety"]["does_not_activate_rollout"] is True
    assert "approve_guarded_rollout" in [row["decision"] for row in packet["operator_options"]]
    assert (tmp_path / "proposal_reports" / "proposal_consensus_selectivity_context_review_operator_review_packet.json").exists()
