from __future__ import annotations

import json

from atlas_ai.brain.ai_research_queue import AIResearchQueue


def test_ai_research_queue_syncs_promotion_recommendations(tmp_path) -> None:
    queue = AIResearchQueue(tmp_path)
    payload = queue.sync_from_promotion_report(
        {
            "recommendations": [
                {
                    "uid": "ai_1",
                    "origin": "ai_inventor:trend_pullback_continuation",
                    "stage": "untested",
                    "recommendation": "run_offline_backtest",
                    "confidence": 0.7,
                    "reasons": ["backtest_pending"],
                    "blockers": [],
                    "preference_score": 0.2,
                }
            ]
        }
    )

    assert payload["items"][0]["id"] == "ai_1:run_offline_backtest"
    assert payload["items"][0]["status"] == "pending_review"
    assert payload["items"][0]["safety"]["does_not_execute"] is True
    assert (tmp_path / "ai_research_queue.json").exists()


def test_ai_research_queue_updates_item_without_execution(tmp_path) -> None:
    queue = AIResearchQueue(tmp_path)
    queue.sync_from_promotion_report({"recommendations": [{"uid": "ai_1", "recommendation": "run_offline_backtest"}]})

    result = queue.update_item("ai_1:run_offline_backtest", "paper_test_requested", note="paper only")

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["item"]["status"] == "paper_test_requested"
    assert result["item"]["operator_note"] == "paper only"


def test_ai_research_queue_requests_demo_review_without_execution(tmp_path) -> None:
    queue = AIResearchQueue(tmp_path)
    queue.sync_from_promotion_report({"recommendations": [{"uid": "ai_1", "recommendation": "eligible_for_demo_review"}]})

    result = queue.update_item("ai_1:eligible_for_demo_review", "demo_review_requested", note="operator review")

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["item"]["status"] == "demo_review_requested"
    assert result["item"]["operator_note"] == "operator review"
    assert AIResearchQueue.summary(result["queue"])["demo_review_requested"] == 1


def test_ai_research_queue_approves_demo_review_without_execution(tmp_path) -> None:
    queue = AIResearchQueue(tmp_path)
    queue.sync_from_promotion_report({"recommendations": [{"uid": "ai_1", "recommendation": "eligible_for_demo_review"}]})

    result = queue.update_item("ai_1:eligible_for_demo_review", "demo_review_approved", note="review approved")

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["item"]["status"] == "demo_review_approved"
    assert result["item"]["operator_note"] == "review approved"
    assert AIResearchQueue.summary(result["queue"])["demo_review_approved"] == 1


def test_ai_research_queue_attaches_demo_review_packet_without_execution(tmp_path) -> None:
    queue = AIResearchQueue(tmp_path)
    queue.sync_from_promotion_report({"recommendations": [{"uid": "ai_1", "recommendation": "eligible_for_demo_review"}]})

    result = queue.attach_demo_review_packet(
        "ai_1:eligible_for_demo_review",
        {
            "schema": "atlas_ai_demo_review_packet_v1",
            "created_at": "2026-06-02T00:00:00+00:00",
            "path": str(tmp_path / "packet.json"),
            "executed": False,
            "safety": {"does_not_promote_demo": True, "does_not_promote_live": True},
        },
    )

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["item"]["demo_review_packet"]["path"].endswith("packet.json")
    assert result["item"]["demo_review_packet"]["does_not_promote_demo"] is True


def test_ai_research_queue_surfaces_archived_rejects_without_execution(tmp_path) -> None:
    archive = tmp_path / "evolution_archive.jsonl"
    archive.write_text(
        json.dumps({
            "uid": "ai_reject_1",
            "origin": "ai_inventor:balanced_confirmation",
            "reject_reason": "insufficient_trades:0<200",
        })
        + "\n",
        encoding="utf-8",
    )
    queue = AIResearchQueue(tmp_path)

    payload = queue.sync_from_archive()

    assert payload["items"][0]["id"] == "ai_reject_1:review_archived_candidate"
    assert payload["items"][0]["recommendation"] == "review_archived_candidate"
    assert payload["items"][0]["safety"]["does_not_execute"] is True
    assert payload["items"][0]["blockers"] == ["insufficient_trades:0<200"]


def test_ai_research_queue_flags_promising_under_sampled_archive(tmp_path) -> None:
    archive = tmp_path / "evolution_archive.jsonl"
    archive.write_text(
        json.dumps({
            "uid": "ai_promising_1",
            "origin": "ai_inventor:range_reversion_filter",
            "reject_reason": "insufficient_trades:31<200",
            "backtest_trades": 31,
            "backtest_win_rate": 0.58,
            "backtest_profit_factor": 3.06,
            "backtest_max_drawdown": 0.01,
        })
        + "\n",
        encoding="utf-8",
    )
    queue = AIResearchQueue(tmp_path)

    payload = queue.sync_from_archive()

    item = payload["items"][0]
    assert item["id"] == "ai_promising_1:extend_backtest_sample"
    assert item["recommendation"] == "extend_backtest_sample"
    assert "promising_under_sampled" in item["reasons"][1]
    assert item["safety"]["does_not_execute"] is True


def test_ai_research_queue_syncs_extended_backtest_results(tmp_path) -> None:
    results = tmp_path / "ai_extended_backtests.jsonl"
    results.write_text(
        json.dumps({
            "uid": "candidate_pass",
            "origin": "ai_inventor:range_reversion_filter",
            "recommendation": "eligible_for_shadow_review",
            "passed_extended_sample": True,
            "bars": 5000,
            "score": 62.4,
            "result": {
                "trades": 255,
                "win_rate": 0.545,
                "profit_factor": 2.65,
                "max_drawdown": 0.029,
            },
            "safety": {"does_not_execute": True},
        })
        + "\n",
        encoding="utf-8",
    )
    queue = AIResearchQueue(tmp_path)

    payload = queue.sync_from_extended_backtests()

    item = payload["items"][0]
    assert item["id"] == "candidate_pass:eligible_for_shadow_review"
    assert item["recommendation"] == "eligible_for_shadow_review"
    assert item["confidence"] == 0.78
    assert item["safety"]["does_not_execute"] is True


def test_ai_research_queue_syncs_shadow_review_results(tmp_path) -> None:
    results = tmp_path / "ai_shadow_reviews.jsonl"
    results.write_text(
        json.dumps({
            "uid": "candidate_shadow",
            "recommendation": "continue_shadow_review",
            "bars": 1200,
            "summary": {"trades": 75, "wins": 40, "losses": 35, "win_rate": 0.533},
            "promotion_governor": {
                "can_promote_to_demo": False,
                "reason": "insufficient_shadow_trades:75<100",
                "shadow_drawdown_policy": "gross_normalized",
                "max_shadow_drawdown_pct": 0.05,
            },
        })
        + "\n",
        encoding="utf-8",
    )
    queue = AIResearchQueue(tmp_path)

    payload = queue.sync_from_shadow_reviews()

    item = payload["items"][0]
    assert item["id"] == "candidate_shadow:continue_shadow_review"
    assert item["recommendation"] == "continue_shadow_review"
    assert item["blockers"] == ["insufficient_shadow_trades:75<100"]
    assert "drawdown_policy=gross_normalized,limit=5.00%" in item["reasons"]
    assert item["safety"]["does_not_execute"] is True


def test_ai_research_queue_syncs_shadow_risk_audits(tmp_path) -> None:
    audits = tmp_path / "ai_shadow_risk_audits.jsonl"
    audits.write_text(
        json.dumps({
            "uid": "candidate_risk",
            "origin": "ai_hybrid:test",
            "metrics": {
                "trades": 165,
                "win_rate": 0.624,
                "net_pnl": 17.27,
                "gross_normalized_drawdown_pct": 0.084,
                "worst_loss_streak": 5,
            },
            "diagnosis": {
                "primary": "drawdown_exceeds_gross_normalized_limit",
                "suggested_repair_focus": "reduce_loss_clusters",
            },
        })
        + "\n",
        encoding="utf-8",
    )
    queue = AIResearchQueue(tmp_path)

    payload = queue.sync_from_shadow_risk_audits()

    item = payload["items"][0]
    assert item["id"] == "candidate_risk:repair_reduce_loss_clusters"
    assert item["recommendation"] == "repair_reduce_loss_clusters"
    assert item["blockers"] == ["drawdown_exceeds_gross_normalized_limit"]
    assert item["safety"]["does_not_execute"] is True
