from __future__ import annotations

from atlas_ai.learning.operator_teaching import build_current_decision_teaching_prompt, build_operator_teaching_digest
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_operator_teaching_digest_summarizes_labels_by_playbook(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_structured_knowledge_file(
        "operator_teaching_reviews.json",
        {
            "schema_version": 1,
            "reviews": [
                {"label": "correct_hold", "playbook_name": "range_edge_reversal"},
                {"label": "correct_hold", "playbook_name": "range_edge_reversal"},
                {"label": "missed_trade", "playbook_name": "pullback_continuation"},
            ],
        },
    )
    digest = build_operator_teaching_digest(store)
    assert digest["sample_count"] == 3
    assert digest["label_counts"]["correct_hold"] == 2
    assert digest["playbook_summary"][0]["playbook"] == "range_edge_reversal"
    assert digest["playbook_summary"][0]["teaching_effect"] == "reinforce_patience"
    assert digest["training_readiness"] == "collect_more_operator_labels"


def test_current_decision_teaching_prompt_prefers_execution_issue_for_friction() -> None:
    prompt = build_current_decision_teaching_prompt(
        {
            "action": "hold",
            "setup_playbook": {
                "top_candidate": {
                    "name": "range_edge_reversal",
                    "readiness": "developing",
                    "missing": [],
                    "risk_notes": ["execution_friction_elevated"],
                }
            },
        }
    )
    assert prompt["suggested_label"] == "execution_issue"
    assert prompt["playbook"] == "range_edge_reversal"


def test_current_decision_teaching_prompt_marks_clean_hold_as_too_cautious() -> None:
    prompt = build_current_decision_teaching_prompt(
        {
            "action": "hold",
            "setup_playbook": {
                "top_candidate": {
                    "name": "pullback_continuation",
                    "readiness": "ready_for_existing_gates",
                    "missing": [],
                    "risk_notes": [],
                }
            },
        }
    )
    assert prompt["suggested_label"] == "too_cautious"
