from __future__ import annotations

from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
from atlas_ai.brain.ai_decision_reviewer import AIDecisionReviewer


def test_ai_decision_reviewer_does_not_validate_wait_from_blocked_event(tmp_path) -> None:
    journal = AIDecisionJournal(tmp_path)
    journal.append(
        {
            "schema": "atlas_ai_decision_trace_v1",
            "recommendation": {
                "action": "wait_for_confirmation",
                "confidence": 0.62,
                "trade_bias": "sell",
                "reason": "AI core recommends waiting because confidence below threshold.",
            },
            "blocker_analysis": {
                "primary_blocker": "confidence below threshold",
                "blockers": ["confidence_below_threshold"],
            },
            "strategy_invention": {
                "candidate_count": 1,
                "candidate_archetypes": ["trend_pullback_continuation"],
            },
        },
        source="test",
        query="why did you wait",
    )

    reviewer = AIDecisionReviewer(tmp_path)
    batch = reviewer.review(
        learning_state={"state": {"loss_memory": {"recent_outcomes": []}}},
        events=[{"type": "blocked", "message": "confidence below threshold"}],
    )

    assert batch["reviewed"] == 1
    latest = batch["latest_review"]
    assert latest["outcome"] == "pending"
    assert latest["blocker_quality"] == "blocker_text_observed_not_outcome_validation"
    assert latest["confidence_quality"] == "not_calibrated"
    assert latest["safety"]["does_not_execute"] is True
    assert batch["strategy_preferences"]["top_archetype"] == "trend_pullback_continuation"
    assert (tmp_path / "ai_decision_reviews.jsonl").exists()
    assert reviewer.preferences.load()["scores"]["trend_pullback_continuation"] == 0


def test_ai_decision_reviewer_ignores_unrelated_win(tmp_path) -> None:
    journal = AIDecisionJournal(tmp_path)
    journal.append(
        {
            "schema": "atlas_ai_decision_trace_v1",
            "recommendation": {"action": "hold_and_observe", "confidence": 0.74, "trade_bias": "buy"},
            "blocker_analysis": {"primary_blocker": "none", "blockers": []},
            "strategy_invention": {"candidate_count": 0, "candidate_archetypes": []},
        },
        source="test",
        query="latest decision",
    )

    batch = AIDecisionReviewer(tmp_path).review(
        learning_state={"state": {"loss_memory": {"recent_outcomes": [{"outcome": "win"}]}}},
        events=[],
    )

    assert batch["latest_review"]["outcome"] == "pending"
    assert batch["latest_review"]["latest_trade_outcome"] == "none"
    assert batch["latest_review"]["confidence_quality"] == "not_calibrated"


def _record(tmp_path, action="hold_and_observe"):
    return AIDecisionJournal(tmp_path).append({
        "recommendation": {"action": action, "confidence": 0.761},
        "blocker_analysis": {"primary_blocker": "timing blocked"},
        "strategy_invention": {"candidate_archetypes": ["trend_pullback_continuation"]},
    }, source="test", query="why wait")


def test_unrelated_risk_event_cannot_validate_wait(tmp_path):
    _record(tmp_path)
    review = AIDecisionReviewer(tmp_path).review(events=[
        {"type": "risk", "message": "spread too wide"}
    ])["latest_review"]
    assert review["outcome"] == "pending"
    assert review["blocker_quality"] == "unconfirmed"
    assert "justified" not in review["lesson"]


def test_review_is_idempotent_across_instances_and_concurrent_callers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    _record(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(lambda _: AIDecisionReviewer(tmp_path).review(), range(8)))
    assert sum(b["new_or_changed"] for b in batches) == 1
    reviewer = AIDecisionReviewer(tmp_path)
    assert len(reviewer.recent_reviews(20)) == 1
    assert len(reviewer.path.read_text().splitlines()) == 1
    assert reviewer.preferences.load()["evidence"] == {}


def test_prepare_requires_exact_later_outcome_and_does_not_credit_invented_archetypes(tmp_path):
    from datetime import datetime, timedelta, timezone
    record = _record(tmp_path, "prepare_trade_candidate")
    record["trace"]["audit"] = {"execution_decision_id": "execution-1"}
    reviewer = AIDecisionReviewer(tmp_path)
    later = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    outcome = {"decision_id": "execution-1", "ticket": "123", "time": later, "outcome": "win"}
    for bad in [
        {**outcome, "decision_id": "other"},
        {**outcome, "ticket": ""},
        {**outcome, "time": "2020-01-01T00:00:00Z"},
        {**outcome, "time": "invalid"},
        {**outcome, "time": "2026-10-20T00:00:00"},
    ]:
        assert reviewer._review_record(record, [bad], [])["outcome"] == "pending"
    review = reviewer._review_record(record, [outcome], [])
    assert review["outcome"] == "linked_prepare_win"
    assert review["confidence_quality"] == "not_calibrated"
    assert review["strategy_archetype_review"]["status"] == "pending_outcome"
    assert reviewer.preferences.update_from_reviews([review])["evidence"] == {}
    assert reviewer._review_record(record, [outcome, outcome], [])["outcome"] == "pending"


def test_legacy_records_are_preserved_but_not_trusted(tmp_path):
    import json
    reviewer = AIDecisionReviewer(tmp_path)
    legacy = {"outcome": "validated_wait", "lesson": "Waiting was justified"}
    original = json.dumps(legacy) + "\n"
    reviewer.path.write_text(original)
    shown = reviewer.recent_reviews()[0]
    assert shown["legacy_unverified"]
    assert shown["outcome"] == "unverified_legacy"
    assert reviewer.path.read_text() == original
