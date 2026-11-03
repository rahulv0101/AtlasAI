from __future__ import annotations

from atlas_ai.brain.relevance_guard import check_relevance


def test_self_question_rejects_status_dump_without_self_shape() -> None:
    result = check_relevance(
        query="What can you do?",
        domain="self_model",
        answer="Status: HOLD | Reason: No high-confidence trade found. | Mode: PAPER | Run: atlas_wake_latest",
    )

    assert result.passed is False
    assert result.reason == "irrelevant_state_leakage"


def test_limitation_question_rejects_trade_blocker_answer_without_self_shape() -> None:
    result = check_relevance(
        query="What are your limitations?",
        domain="self_model",
        answer="No high-confidence trade found. Conservative pre-trade filter blocked trade.",
    )

    assert result.passed is False
    assert result.reason == "irrelevant_trade_state_answer"


def test_self_question_allows_grounded_self_model_even_when_role_mentions_paper() -> None:
    result = check_relevance(
        query="What can you do?",
        domain="self_model",
        answer="Capabilities:\n- explain status\n\nCurrent role:\n- Paper-mode operator assistant with read-only Ask Atlas answers.",
    )

    assert result.passed is True


def test_liquidity_question_rejects_runtime_state_leakage() -> None:
    result = check_relevance(
        query="What is liquidity?",
        domain="market_knowledge",
        answer="Status: HOLD\nReason: No high-confidence trade found.\nMode: Paper",
    )

    assert result.passed is False
    assert result.reason == "knowledge_query_state_leakage"
