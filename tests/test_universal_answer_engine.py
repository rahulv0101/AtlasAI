from __future__ import annotations

import json

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self._status = {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.21, "side": "flat", "strategy": "momentum_continuation"},
            "min_confidence_threshold": 0.58,
            "comparison_active": True,
            "comparison_run_id": "atlas_wake_latest",
            "comparison_preset": "conservative_filter_rollout_v1",
            "comparison": {
                "active": True,
                "run_id": "atlas_wake_latest",
                "preset": "conservative_filter_rollout_v1",
                "decision_checklist": {"classification": "CONTINUE_TESTING"},
            },
            "context_snapshot": {"regime": "trend", "session": "london", "liquidity_state": "normal", "spread_state": "tight"},
            "system_health": {"paper_mode_confirmed": True},
        }
        self._runtime = {"execute_trades": False}
        self._events = [{"type": "risk", "message": "No high-confidence trade found."}]

    def load_status(self) -> dict:
        return dict(self._status)

    def load_events(self, limit: int = 160) -> list[dict]:
        return list(self._events)[-limit:]

    def load_runtime_flags(self) -> dict:
        return dict(self._runtime)

    def load_comparison(self) -> dict:
        return dict(self._status.get("comparison", {}) or {})

    def load_self_improvement_bundle(self) -> dict:
        si = dict(dict(self._status.get("system_health", {}) or {}).get("self_improvement", {}) or {})
        if int(si.get("approval_pending_count", 0) or 0) <= 0:
            return {"proposal_queue.json": {"proposals": []}}
        return {
            "proposal_queue.json": {
                "proposals": [
                    {
                        "proposal_id": "proposal_consensus_selectivity_context_review",
                        "status": "awaiting_approval",
                        "category": "filters",
                        "scope": {
                            "target_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2."
                        },
                    }
                ]
            }
        }


def _ask(q: str) -> dict:
    return AskAtlasService(_Source()).answer(q)


def test_status_query_relevant() -> None:
    r = _ask("What state are you in?")
    assert r["intent"] == "status"
    assert "Status:" in r["result"]


def test_blocker_query_relevant() -> None:
    r = _ask("What is holding you back?")
    assert r["intent"] == "blocker"
    assert "Main blocker:" in r["result"]


def test_why_not_trading_query_relevant() -> None:
    r = _ask("Why are you not taking trades?")
    assert r["intent"] == "blocker"
    assert "Main blocker:" in r["result"]


def test_readiness_query_relevant() -> None:
    r = _ask("Are you ready for live?")
    assert r["intent"] == "readiness"
    assert r["result"].startswith("Readiness:")


def test_comparison_query_relevant() -> None:
    r = _ask("What changed in the latest comparison?")
    assert r["intent"] == "comparison"
    assert "Comparison:" in r["result"]


def test_next_safe_experiment_query_relevant() -> None:
    r = _ask("What is the next safe experiment?")
    assert r["intent"] == "next_safe_experiment"
    assert "Next safe step:" in r["result"]


def test_self_knowledge_does_not_dump_state() -> None:
    r = _ask("How much do you know about the markets?")
    assert r["intent"] in {"self_knowledge", "market_knowledge"}
    assert "Knowledge level:" in r["result"] or "Market knowledge summary:" in r["result"]
    assert "BLOCKED" not in r["result"]
    assert "Mode: PAPER" not in r["result"]
    assert "Run: atlas_wake_latest" not in r["result"]


def test_capabilities_query_relevant() -> None:
    r = _ask("What can you do?")
    assert r["intent"] == "capabilities"
    assert "Capabilities:" in r["result"]


def test_limitations_query_relevant() -> None:
    r = _ask("What are your limitations?")
    assert r["intent"] == "limitations"
    assert "Limitations:" in r["result"]


def test_volatility_query_uses_market_domain() -> None:
    r = _ask("Do you understand volatility?")
    assert r["selected_domain"] in {"market_knowledge", "self_model"}
    assert "volatility" in r["result"].lower() or "market knowledge summary" in r["result"].lower()


def test_help_query_relevant() -> None:
    r = _ask("Help me understand what you can answer.")
    assert r["intent"] == "help"
    assert "I can answer:" in r["result"]


def test_intelligence_query_uses_self_model() -> None:
    r = _ask("Are you intelligent?")
    assert r["intent"] == "self_knowledge"
    assert "Knowledge level:" in r["result"]


def test_abstract_smarter_question_routes_to_self_assessment() -> None:
    r = _ask("What is holding you back from becoming smarter?")
    assert r["intent"] == "self_assessment"
    text = r["result"].lower()
    assert "current blocker" in text
    assert "system limitations" in text
    assert "improvement path" in text


def test_abstract_improvement_question_routes_to_self_assessment() -> None:
    r = _ask("How can you improve?")
    assert r["intent"] == "self_assessment"
    assert "learning constraints" in r["result"].lower()


def test_abstract_intelligence_limits_question_routes_to_self_assessment() -> None:
    r = _ask("What limits your intelligence?")
    assert r["intent"] == "self_assessment"
    assert "no autonomous internet learning" in r["result"].lower()


def test_trading_question_still_routes_to_recommendation() -> None:
    r = _ask("Should I buy or sell?")
    assert r["intent"] in {"recommendation", "safe_next_step"}


def test_conversational_trade_question_does_not_fall_to_unknown() -> None:
    r = _ask("What do you think we should do right now?")
    assert r["intent"] in {"recommendation", "safe_next_step"}
    assert r["intent"] != "unknown_operator_query"


def test_short_whats_next_routes_to_safe_next_step() -> None:
    r = _ask("whats next")
    assert r["intent"] in {"recommendation", "safe_next_step"}
    assert r["intent"] != "unknown_operator_query"
    assert "Recommended next step:" in r["result"] or "Recommendation:" in r["result"]


def test_short_whats_next_prioritizes_pending_operator_review() -> None:
    source = _Source()
    source._status["system_health"]["self_improvement"] = {
        "approval_pending_count": 1,
        "latest_proposal_id": "proposal_consensus_selectivity_context_review",
        "rollout_active": True,
    }

    r = AskAtlasService(source).answer("whats next")

    assert r["intent"] in {"recommendation", "safe_next_step"}
    text = r["result"].lower()
    assert "operator approval gate" in text
    assert "proposal_consensus_selectivity_context_review" in text
    assert "consensus guard blocked trade" in text
    assert "approval_action=approve proposal_consensus_selectivity_context_review" in text
    assert "close_actions=defer or reject proposal_consensus_selectivity_context_review" in text
    assert "do not auto-approve" in text


def test_short_whats_next_includes_operator_review_posture(tmp_path) -> None:
    source = _Source()
    source.store = type("Store", (), {"base": tmp_path})()
    source._status["system_health"]["self_improvement"] = {
        "approval_pending_count": 1,
        "latest_proposal_id": "proposal_consensus_selectivity_context_review",
        "rollout_active": True,
    }
    report_dir = tmp_path / "proposal_reports"
    report_dir.mkdir(parents=True)
    (report_dir / "proposal_consensus_selectivity_context_review_operator_review_packet.json").write_text(
        json.dumps({
            "operator_posture": {
                "posture": "guarded_rollout_candidate_with_replay_caution",
                "reason": "Shadow compare and collector evidence are positive, but replay evidence still needs caution.",
                "suggested_operator_action": "review_packet_then_consider_guarded_rollout_approval",
            }
        }),
        encoding="utf-8",
    )

    r = AskAtlasService(source).answer("whats next")

    text = r["result"].lower()
    assert "operator_posture=guarded_rollout_candidate_with_replay_caution" in text
    assert "suggested_operator_action=review_packet_then_consider_guarded_rollout_approval" in text
    assert "replay evidence still needs caution" in text


def test_component_guidance_uses_missed_entry_repair_summary() -> None:
    source = _Source()
    source._status["missed_entry_learning_summary"] = {
        "available": True,
        "top_repair_candidate": {
            "primary_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2.",
            "repair_priority": "high",
            "suggested_action": "review_blocker_selectivity",
            "samples": 150,
            "missed_opportunity": 55,
            "correct_hold_or_block": 25,
        },
    }

    r = AskAtlasService(source).answer("what should we do about consensus right now")

    assert r["intent"] in {"recommendation", "safe_next_step"}
    assert r["intent"] != "unknown_operator_query"
    assert "offline blocker-selectivity review" in r["result"].lower()
    assert "missed opportunities" in r["result"].lower()


def test_abstract_not_getting_smarter_routes_to_self_assessment() -> None:
    r = _ask("Why are you not getting smarter?")
    assert r["intent"] == "self_assessment"
    txt = r["result"].lower()
    assert "system limitations" in txt
    assert "improvement path" in txt


def test_no_high_confidence_answer_includes_drag_diagnostics_when_available() -> None:
    source = _Source()
    source._status["confidence_diagnostics"] = {"dominant_drag_reasons": ["weak_consensus", "spread_pressure"]}
    out = AskAtlasService(source).answer("Should I buy or sell?")
    assert "Main drag factors" in out["result"]
    assert "Recommendation: hold." in out["result"]
