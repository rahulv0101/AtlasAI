from __future__ import annotations

from atlas_ai.brain.ai_decision_journal import AIDecisionJournal, summarize_decision_record


def test_ai_decision_journal_appends_and_reads_recent(tmp_path) -> None:
    journal = AIDecisionJournal(tmp_path)
    trace = {
        "schema": "atlas_ai_decision_trace_v1",
        "recommendation": {"action": "wait_for_confirmation", "confidence": 0.61, "reason": "confirmation weak"},
        "blocker_analysis": {"primary_blocker": "confidence_below_threshold"},
        "strategy_invention": {"candidate_count": 2, "candidate_archetypes": ["trend_pullback_continuation"]},
    }

    journal.append(trace, source="test", query="why did you wait")
    rows = journal.recent(limit=3)

    assert len(rows) == 1
    assert rows[0]["source"] == "test"
    assert rows[0]["trace"]["recommendation"]["action"] == "wait_for_confirmation"
    summary = summarize_decision_record(rows[0])
    assert summary["action"] == "wait_for_confirmation"
    assert summary["strategy_candidate_count"] == 2
