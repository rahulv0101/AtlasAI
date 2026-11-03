from __future__ import annotations

from atlas_ai.learning.experiment_runner import build_self_improvement_validation_report
from atlas_ai.learning.outcome_intelligence import sync_self_improvement_from_outcome_intelligence
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import load_self_improvement_bundle
from atlas_ai.ui_server import create_ui_app


def _sample_outcome_intelligence() -> dict:
    return {
        "sample_count": 48,
        "loser_patterns": {
            "top_patterns": {
                "trend_bucket": [
                    {"value": "weak", "samples": 28, "win_rate": 0.25, "avg_pnl_zar": -320.0}
                ],
                "strategy": [
                    {"value": "momentum_continuation", "samples": 30, "win_rate": 0.32, "avg_pnl_zar": -140.0}
                ],
            }
        },
        "pattern_comparison": {
            "winner_vs_loser_deltas": {
                "confidence": 0.11,
                "trend_strength": 0.09,
                "duration_minutes": 3.5,
            }
        },
    }


def test_outcome_intelligence_sync_creates_insight_and_proposal(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    before_registry = load_self_improvement_bundle(store)["active_rule_registry.json"]

    result = sync_self_improvement_from_outcome_intelligence(store, _sample_outcome_intelligence())
    bundle = load_self_improvement_bundle(store)

    assert result["created_insights"]
    assert result["created_proposals"]
    assert bundle["learned_insights.json"]["insights"][-1]["recommendation_type"] == "proposal_candidate"
    assert bundle["proposal_queue.json"]["proposals"][-1]["category"] == "strategy_logic"
    assert bundle["active_rule_registry.json"] == before_registry
    audit = store.tail_self_improvement_audit(10)
    assert any(str(row.get("event", "")) in {"proposal_created", "proposal_updated"} for row in audit)


def test_validation_report_scaffold_is_generated_for_proposal(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    sync_self_improvement_from_outcome_intelligence(store, _sample_outcome_intelligence())
    proposal_id = load_self_improvement_bundle(store)["proposal_queue.json"]["proposals"][-1]["proposal_id"]

    report = build_self_improvement_validation_report(store, proposal_id)

    assert report["proposal_id"] == proposal_id
    assert report["promotion_readiness"]["operator_approval_required"] is True
    assert "required" in report["validation"]


def test_ui_exposes_self_improvement_validation_report(tmp_path):
    store = KnowledgeStore(str(tmp_path / "storage"))
    sync_self_improvement_from_outcome_intelligence(store, _sample_outcome_intelligence())
    proposal_id = load_self_improvement_bundle(store)["proposal_queue.json"]["proposals"][-1]["proposal_id"]

    app = create_ui_app(store)
    client = app.test_client()
    payload = client.get(f"/api/self-improvement/validation/{proposal_id}").get_json()

    assert payload["proposal_id"] == proposal_id
    assert payload["promotion_readiness"]["meets_strength_threshold"] is True
