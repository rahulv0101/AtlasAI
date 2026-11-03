from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.22, "side": "flat"},
            "min_confidence_threshold": 0.58,
            "comparison_active": True,
            "comparison_run_id": "atlas_wake_latest",
            "comparison": {
                "active": True,
                "run_id": "atlas_wake_latest",
                "decision_checklist": {"classification": "CONTINUE_TESTING"},
            },
            "context_snapshot": {"regime": "trend", "session": "london"},
            "system_health": {"paper_mode_confirmed": True},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "No high-confidence trade found."}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest", "decision_checklist": {"classification": "CONTINUE_TESTING"}}


class _LearningLaneSource(_Source):
    def load_status(self) -> dict:
        status = super().load_status()
        status["learning_lanes"] = {
            "active_lanes": 5,
            "lanes": {
                "observe": {"active": True, "status": "active", "evidence": {"visual_available": True}},
                "classify": {"active": True, "status": "active", "evidence": {"blocker_class": "safety_gate"}},
                "remember": {"active": True, "status": "active", "evidence": {"pattern_lessons": 20, "recent_outcomes": 200}},
                "explain": {"active": True, "status": "active", "evidence": {"summary": "No-trade explanation available."}},
                "bounded_caution": {
                    "active": False,
                    "status": "watching",
                    "evidence": {"missing_confirmations": ["strong_m1_momentum", "structure_hold"]},
                },
                "propose_improvements": {"active": True, "status": "active", "evidence": {"total": 2}},
            },
        }
        return status

    def load_learning_state(self) -> dict:
        return {
            "state": {
                "loss_memory": {
                    "recent_outcomes": [{"outcome": "loss"}] * 200,
                    "pattern_lessons": {str(i): {"samples": 1} for i in range(20)},
                }
            }
        }


class _SafetyLockSource(_Source):
    def load_status(self) -> dict:
        status = super().load_status()
        status["action"] = "blocked"
        status["message"] = "Risk guard active: Safety lock active: consecutive_loss_threshold, supporting_win_rate_weak. No trade executed."
        status["safety_lock_review"] = {
            "active": True,
            "status": "locked",
            "reason": "consecutive_loss_threshold, supporting_win_rate_weak",
            "evidence": [
                {
                    "code": "consecutive_loss_threshold",
                    "label": "Loss streak",
                    "value": "not_reported",
                    "detail": "Recent or session loss streak reached the protective threshold.",
                },
                {
                    "code": "supporting_win_rate_weak",
                    "label": "Supporting win rate",
                    "value": "not_reported",
                    "detail": "Recent supporting evidence is too weak for normal live entry flow.",
                },
            ],
            "recovery_checklist": [
                {
                    "code": "manual_operator_unlock_required",
                    "label": "Manual operator review is required before unlock.",
                    "status": "blocked",
                },
                {
                    "code": "next_setup_meets_high_confluence",
                    "label": "The next setup must meet high-confluence normal gates.",
                    "status": "review",
                },
            ],
            "separate_from_learning": {
                "pattern_ban": False,
                "note": "Safety lock protects the account. It is separate from loss-pattern learning and does not permanently ban a setup family.",
            },
        }
        return status


def test_safety_lock_question_uses_review_surface_not_generic_execution_status() -> None:
    svc = AskAtlasService(_SafetyLockSource())
    reply = svc.answer("What clears the safety lock?")

    assert reply["intent"] == "safety_lock_review"
    assert reply["selected_domain"] == "risk_policy"
    assert "Safety lock: ACTIVE" in reply["result"]
    assert "Manual operator review is required before unlock" in reply["result"]
    assert "not a pattern ban" in reply["result"]
    assert "does not unlock safety" in reply["result"]


def test_voice_safety_lock_question_stays_short_and_safe() -> None:
    svc = AskAtlasService(_SafetyLockSource())
    reply = svc.answer("why are you locked", mode="voice")

    assert reply["intent"] == "safety_lock_review"
    assert reply["selected_domain"] == "risk_policy"
    assert len(reply["result"].splitlines()) <= 2
    assert "Safety lock is active" in reply["result"]
    assert "not a pattern ban" in reply["result"]


def test_learning_question_reports_learning_lanes_not_trade_status() -> None:
    svc = AskAtlasService(_LearningLaneSource())
    reply = svc.answer("Does Atlas study all the time?")

    assert reply["intent"] == "learning_status"
    assert reply["selected_domain"] == "learning_state"
    assert "5/6 learning lanes are active" in reply["result"]
    assert "Observe:" in reply["result"]
    assert "Bounded caution: watching" in reply["result"]
    assert "force trades" in reply["result"]
    assert "STATUS:" not in reply["result"]


def test_voice_learning_question_uses_same_learning_path_and_stays_short() -> None:
    svc = AskAtlasService(_LearningLaneSource())
    reply = svc.answer("what is atlas learning right now", mode="voice")

    assert reply["intent"] == "learning_status"
    assert reply["selected_domain"] == "learning_state"
    assert len(reply["result"].splitlines()) <= 2
    assert "5/6 learning lanes are active" in reply["result"]
    assert "cannot change strategy" in reply["result"]


def test_followup_why_stays_on_blocker_context() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("What is holding you back?")
    r2 = svc.answer("Why?")
    assert r1["intent"] == "blocker"
    assert r2["intent"] in {"blocker", "explanation"}
    assert "Main blocker" in r2["result"] or "blocker" in r2["result"].lower()


def test_followup_what_changed_after_readiness_stays_relevant() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("Are you ready for live?")
    r2 = svc.answer("What changed?")
    assert r1["intent"] == "readiness"
    assert r2["selected_domain"] in {"readiness_assessment", "comparison_results"}
    assert "Readiness" in r2["result"] or "Comparison" in r2["result"] or "changed" in r2["result"].lower()


def test_followup_what_do_you_mean_after_self_knowledge_stays_self_domain() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("How much do you know about the markets?")
    r2 = svc.answer("What do you mean?")
    assert r1["selected_domain"] in {"self_model", "market_knowledge"}
    assert r2["selected_domain"] in {"self_model", "market_knowledge", "general_operator_help"}
    assert "Knowledge" in r2["result"] or "market" in r2["result"].lower() or "I can answer" in r2["result"]


def test_followup_why_after_limitations_stays_self_domain() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("What are your limitations?")
    r2 = svc.answer("Why does that matter?")
    assert r1["intent"] == "limitations"
    assert r2["selected_domain"] == "self_model"
    assert "Knowledge" in r2["result"] or "Limitations" in r2["result"]


def test_followup_what_about_that_after_recommendation_stays_recommendation() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("What should I do next?")
    r2 = svc.answer("What about that?")
    assert r1["selected_domain"] == "current_system_state"
    assert r2["selected_domain"] in {"recommendation", "blockers"}
    assert "recommend" in r2["result"].lower() or "safe" in r2["result"].lower()


def test_followup_why_not_trading_after_state_links_to_state_or_blocker() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("What state are you in?")
    r2 = svc.answer("Why not trading?")
    assert r1["intent"] == "status"
    assert r2["intent"] in {"blocker", "status", "explanation"}
    assert "blocker" in r2["result"].lower() or "no high-confidence trade" in r2["result"].lower()


def test_followup_compared_to_what_after_comparison_stays_comparison() -> None:
    svc = AskAtlasService(_Source())
    r1 = svc.answer("What changed in the latest comparison?")
    r2 = svc.answer("Compared to what?")
    assert r1["intent"] == "comparison"
    assert r2["selected_domain"] == "comparison_results"


def test_voice_mode_is_short_and_relevant_for_status() -> None:
    svc = AskAtlasService(_Source())
    r = svc.answer("What state are you in?", mode="voice")
    assert len(r["result"].splitlines()) <= 2
    assert "hold" in r["result"].lower() or "no high-confidence trade" in r["result"].lower()


def test_voice_mode_short_for_blocker() -> None:
    svc = AskAtlasService(_Source())
    r = svc.answer("What is holding you back?", mode="voice")
    assert len(r["result"].splitlines()) <= 2
    assert "blocker" in r["result"].lower()
