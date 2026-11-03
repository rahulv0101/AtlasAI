from __future__ import annotations

from atlas_ai.brain.ai_strategy_preferences import AIStrategyPreferences
from atlas_ai.strategies.strategy_inventor import StrategyInventor


def test_strategy_preferences_update_from_reviews(tmp_path) -> None:
    prefs = AIStrategyPreferences(tmp_path)
    payload = prefs.update_from_reviews(
        [
            {
                "schema": "atlas_ai_decision_review_v2",
                "decision_id": "one",
                "outcome": "linked_prepare_win",
                "outcome_evidence": {"status": "linked_outcome", "ticket": "1", "time": "later", "decision_id": "execution-1"},
                "strategy_archetype_review": {"status": "matched_executed_archetype", "archetypes": ["trend_pullback_continuation"]},
            },
            {
                "schema": "atlas_ai_decision_review_v2",
                "decision_id": "two",
                "outcome": "linked_prepare_loss",
                "outcome_evidence": {"status": "linked_outcome", "ticket": "2", "time": "later", "decision_id": "execution-2"},
                "strategy_archetype_review": {"status": "matched_executed_archetype", "archetypes": ["range_reversion_filter"]},
            },
        ]
    )

    assert payload["scores"]["trend_pullback_continuation"] > 0
    assert payload["scores"]["range_reversion_filter"] < 0
    assert payload["policy"]["does_not_execute"] is True
    assert (tmp_path / "ai_review_evidence.sqlite3").exists()


def test_legacy_scores_and_wait_rewards_are_excluded_without_erasing_history(tmp_path):
    import json
    prefs = AIStrategyPreferences(tmp_path)
    original = json.dumps({"scores": {"trend_pullback_continuation": 1.0}})
    prefs.path.write_text(original)
    review = {"outcome": "validated_wait",
              "strategy_archetype_review": {"archetypes": ["trend_pullback_continuation"]}}
    payload = prefs.update_from_reviews([review] * 100)
    assert payload["scores"]["trend_pullback_continuation"] == 0
    assert payload["legacy_scores_excluded"]
    assert prefs.path.read_text() == original


def test_same_execution_cannot_be_credited_twice(tmp_path):
    prefs = AIStrategyPreferences(tmp_path)
    review = {
        "schema": "atlas_ai_decision_review_v2", "decision_id": "one",
        "outcome": "linked_prepare_win",
        "outcome_evidence": {"status": "linked_outcome", "ticket": "1",
                             "time": "later", "decision_id": "execution-1"},
        "strategy_archetype_review": {"status": "matched_executed_archetype",
                                     "archetypes": ["trend_pullback_continuation"]},
    }
    prefs.update_from_reviews([review] * 3)
    payload = AIStrategyPreferences(tmp_path).update_from_reviews([{**review, "decision_id": "another"}])
    assert payload["scores"]["trend_pullback_continuation"] == 0.18
    assert payload["evidence"]["trend_pullback_continuation"]["samples"] == 1


def test_strategy_inventor_uses_preference_scores_for_ordering() -> None:
    plans = StrategyInventor().invent(
        status={
            "primary_blocker": "confidence below threshold",
            "context_snapshot": {"regime": "range"},
        },
        cognitive_core={
            "decision_understanding": {"final_trade_bias": "buy"},
            "learning_state": {"stability_score": 0.8, "overreaction_score": 0.1},
        },
        strategy_preferences={
            "scores": {
                "trend_pullback_continuation": -0.5,
                "range_reversion_filter": 0.7,
            }
        },
        limit=2,
    )

    assert plans[0].archetype == "range_reversion_filter"
    assert all(not hasattr(plan.chromosome, "execute") for plan in plans)
