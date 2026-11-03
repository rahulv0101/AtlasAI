from __future__ import annotations

from atlas_ai.brain.cognitive_core import build_cognitive_core
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "blocked",
            "message": "Higher-timeframe alignment required: chosen=sell, htf_bias=buy.",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "chosen": {"confidence": 0.66, "side": "sell", "strategy": "gold_micro_impulse"},
            "min_confidence_threshold": 0.44,
            "context_snapshot": {
                "session": "new_york",
                "regime": "range",
                "trend_structure": "LH/LL",
                "momentum": "strong",
                "location": "breakout",
            },
            "risk": {"spread_points": 36.0},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [
            {"type": "learning", "message": "Operator teaching review recorded: execution_issue"},
            {"type": "risk", "message": "Breakout follow-through confirmed with aligned momentum."},
        ]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": True}

    def load_comparison(self) -> dict:
        return {}

    def load_learning_state(self) -> dict:
        return {
            "state": {
                "loss_memory": {
                    "recent_outcomes": [
                        {"outcome": "loss", "setup_family": "breakout", "session": "new_york", "side": "sell"},
                        {"outcome": "win", "setup_family": "breakout", "session": "new_york", "side": "sell"},
                    ]
                }
            }
        }


def test_cognitive_core_builds_read_only_world_and_learning_model() -> None:
    src = _Source()
    core = build_cognitive_core(
        status=src.load_status(),
        runtime={"execution": {"session_control": {"current_session": "new_york"}}},
        events=src.load_events(),
        learning_state=src.load_learning_state(),
        knowledge_payload={"lessons_learned": {"lessons": ["Teach timing separately from idea quality."]}},
        adaptation_summary={"active_learning_mode": "medium_path", "evidence_strength_score": 0.72, "stability_score": 0.8},
        evidence_separation={
            "market_bias": "sell",
            "signal_direction": "sell",
            "higher_timeframe_bias": "buy",
            "final_trade_bias": "neutral",
            "setup_readiness": "armed_but_context_blocked",
            "execution_permission": "not_allowed",
        },
    )
    assert core["schema"] == "atlas_cognitive_core_v1"
    assert core["mode"] == "read_only_understanding"
    assert core["world_model"]["session"] == "new_york"
    assert core["decision_understanding"]["market_bias"] == "sell"
    assert core["trade_memory"]["closed_sample_size"] == 2
    assert core["trade_memory"]["win_rate"] == 0.5
    assert core["safety_contract"]["can_bypass_execution_gate"] is False
    assert "does not change strategy" in core["safety_contract"]["policy_note"]


def test_brain_routes_system_wide_brain_questions_to_cognitive_core() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("Does Atlas have its own brain for understanding and learning?")
    assert out["intent"] == "cognitive_core"
    assert out["selected_domain"] == "cognitive_core"
    assert out["answer_trace"]["selected_plan"] == "cognitive_core_plan"
    assert out["answer_trace"]["cognitive_core"]["mode"] == "read_only_understanding"
    assert out["answer_trace"]["ai_decision_trace"]["schema"] == "atlas_ai_decision_trace_v1"
    assert "cognitive core" in out["result"].lower()
    assert "AI decision core" in out["result"]
    assert "Can bypass ExecutionGate: False" in out["result"]


def test_one_big_brain_question_routes_to_coordinated_cognitive_core() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("Is Atlas one big brain now?")
    assert out["intent"] == "cognitive_core"
    assert out["selected_domain"] == "cognitive_core"
    assert "one coordinated mind" in out["result"]
    assert "Specialist modules underneath" in out["result"]
    assert "Risk and execution gates remain separate authority" in out["result"]


def test_voice_giant_brain_question_stays_short_and_safe() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("Would a giant brain be better?", mode="voice")
    assert out["intent"] == "cognitive_core"
    assert out["selected_domain"] == "cognitive_core"
    assert len(out["result"].splitlines()) <= 2
    assert "one coordinated cognitive core" in out["result"]
    assert "gate-controlled" in out["result"]


def test_lesson_questions_route_to_cognitive_core_not_status_fallback() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("What did you learn from the new brain and post trade lesson upgrade?")
    assert out["intent"] == "cognitive_core"
    assert out["selected_domain"] == "cognitive_core"
    assert "Current lessons it should remember" in out["result"]
    assert "status:" not in out["result"].lower()
