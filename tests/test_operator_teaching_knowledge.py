from __future__ import annotations

from atlas_ai.brain.answer_engine import AnswerEngine
from atlas_ai.brain.knowledge import BrainKnowledge
from atlas_ai.learning.operator_teaching import build_operator_teaching_digest
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_operator_teaching_examples_load_from_structured_knowledge() -> None:
    payload = BrainKnowledge().load("operator_teaching_examples")
    assert payload["examples"]
    labels = {str(row.get("label", "")) for row in payload["examples"]}
    assert {"correct_hold", "missed_trade", "execution_issue"}.issubset(labels)
    scenarios = " ".join(str(row.get("scenario", "")) for row in payload["examples"]).lower()
    assert "lh/ll" in scenarios
    assert "mid-range" in scenarios
    assert "one supporting signal" in scenarios


def test_lessons_include_current_mid_range_consensus_teaching() -> None:
    payload = BrainKnowledge().load("lessons_learned")
    lessons = " ".join(str(x) for x in payload["lessons"]).lower()
    assert "mid-range" in lessons
    assert "consensus support of 1/2" in lessons
    assert "structure is lh/ll" in lessons
    assert "trend label is not execution permission" in lessons
    assert "weak or mixed momentum" in lessons
    assert "not permission to rewrite strategy, risk, or execution policy" in lessons
    assert "2026-05-25" in lessons
    assert "visual m1 sell structure" in lessons
    assert "final trade bias is neutral" in lessons
    assert "losing or blocked buy/sell evidence with setup family" in lessons


def test_loss_pattern_teachings_include_no_ban_and_safety_separation() -> None:
    payload = BrainKnowledge().load("loss_pattern_teachings")
    teachings = {str(item.get("id", "")): item for item in list(payload.get("teachings", []) or [])}

    assert "loss_pattern_is_not_pattern_ban" in teachings
    assert "safety_lock_is_separate_from_learning" in teachings
    assert teachings["loss_pattern_is_not_pattern_ban"]["teaching_action"].startswith("Keep the pattern tradable")
    assert "do not rewrite strategies" in str(payload.get("policy_note", "")).lower()


def test_loss_history_teachings_include_all_loss_clusters_and_grounding_limits() -> None:
    payload = BrainKnowledge().load("loss_history_teachings")
    clusters = {str(item.get("id", "")): item for item in list(payload.get("loss_clusters", []) or [])}
    summary = dict(payload.get("sample_summary", {}) or {})
    text = " ".join(str(item.get("lesson", "")) for item in clusters.values()).lower()

    assert summary["losses_reviewed"] >= 100
    assert summary["pattern_lessons"] >= 20
    assert "broad_buy_timing_stop_hunt_cluster" in clusters
    assert "false_continuation_confirmation_cluster" in clusters
    assert "not precise proof" in str(payload.get("grounding_note", "")).lower()
    assert "not teaching atlas to stop trading continuation" in text
    assert "do not rewrite strategies" in str(payload.get("policy_note", "")).lower()


def test_strategy_answer_includes_taught_good_and_bad_examples() -> None:
    result = AnswerEngine().answer(
        "Explain breakout retest and when to avoid it",
        {"status": {}, "runtime": {}, "events": [], "comparison": {}},
    )
    text = result.text.lower()
    assert result.selected_intent == "strategy_knowledge"
    assert "good example" in text
    assert "bad example" in text
    assert "teaching note" in text


def test_market_knowledge_explains_execution_friction() -> None:
    result = AnswerEngine().answer(
        "What is execution friction?",
        {"status": {}, "runtime": {}, "events": [], "comparison": {}},
    )
    assert result.selected_domain == "market_knowledge"
    assert "execution friction" in result.text.lower()


def test_operator_teaching_digest_collects_before_review_threshold(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_structured_knowledge_file(
        "operator_teaching_reviews.json",
        {
            "schema_version": 1,
            "reviews": [
                {"label": "correct_hold", "playbook_name": "pullback_continuation"}
                for _ in range(3)
            ],
        },
    )

    digest = build_operator_teaching_digest(store)

    assert digest["training_readiness"] == "collect_more_operator_labels"
    assert digest["samples_needed"] == 9
    assert digest["review_plan"]["status"] == "collecting"
    assert digest["review_plan"]["focus"] == "collect_more_operator_labels"
    assert "do_not_change_strategy_logic_automatically" in digest["review_plan"]["blocked_actions"]


def test_operator_teaching_digest_builds_bounded_review_plan_when_ready(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    reviews = []
    for idx in range(12):
        reviews.append(
            {
                "label": "correct_hold" if idx < 8 else "execution_issue",
                "playbook_name": "pullback_continuation",
                "context_quality": {
                    "decision_context_linked": idx < 7,
                    "location_known": idx < 7,
                    "momentum_known": idx < 7,
                    "volatility_known": idx < 7,
                    "visual_context_known": idx < 7,
                    "selected_signal_known": idx < 7,
                    "setup_context_known": idx < 7,
                },
            }
        )
    store.save_structured_knowledge_file(
        "operator_teaching_reviews.json",
        {"schema_version": 1, "reviews": reviews},
    )

    digest = build_operator_teaching_digest(store)
    plan = digest["review_plan"]

    assert digest["training_readiness"] == "usable_for_review"
    assert digest["samples_needed"] == 0
    assert plan["status"] == "ready"
    assert plan["focus"] == "review_teaching_clusters"
    assert plan["rich_context_samples"] == 7
    assert plan["review_items"][0]["target"] == "pullback_continuation"
    assert "propose_bounded_review_items" in plan["allowed_actions"]
    assert "do_not_change_execution_policy_automatically" in plan["blocked_actions"]
    assert "proposal-only" in plan["policy_note"]
