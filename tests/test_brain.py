from __future__ import annotations

from atlas_ai.brain import AtlasBrain, BrainMemory, BrainSnapshot, BrainTurn


def _snapshot() -> BrainSnapshot:
    return BrainSnapshot(
        status={
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.22},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
            "decision_explanation": {
                "final_decision": "block",
                "reason_codes": ["poor_liquidity"],
                "setup_quality_reasons": ["weak_consensus"],
            },
            "shadow_advice": {"anomaly_flags": ["blocked_trade"]},
            "system_health": {"paper_mode_confirmed": True},
            "win_rate_focus": {"samples": 8, "recent_win_rate": 0.25},
            "drift": {"active": True},
        },
        events=[
            {"type": "risk", "message": "Conservative pre-trade filter blocked trade: poor liquidity."},
            {"type": "risk", "message": "Conservative pre-trade filter blocked trade: poor liquidity."},
            {"type": "risk", "message": "Spread too high (42.0 pts)."},
        ],
        runtime={"execute_trades": False},
        learning_state={"state": {"evolved_strategy_pool": {"count": 3}}},
        comparison={
            "run_id": "run123",
            "control_metrics": {"trades_closed": 2},
            "experiment_metrics": {"trades_closed": 1},
            "decision_checklist": {"classification": "CONTINUE_TESTING"},
            "control_status": {"message": "No high-confidence trade found."},
            "experiment_status": {"message": "Conservative pre-trade filter blocked trade: poor liquidity."},
        },
    )


def test_brain_produces_blocker_rank_output():
    brain = AtlasBrain(_snapshot(), BrainMemory())

    output = brain.blocker_rank()

    assert output.kind == "blocker_rank"
    assert output.details["ranking"][0][0].lower().startswith("conservative pre-trade filter blocked trade")


def test_brain_produces_setup_assessment_output():
    brain = AtlasBrain(_snapshot(), BrainMemory())

    output = brain.setup_assessment()

    assert output.kind == "setup_assessment"
    assert output.details["quality"] == "weak"
    assert "liquidity" in output.details["issue_types"]
    assert "confidence" in output.details["issue_types"]


def test_brain_produces_research_plan_output():
    brain = AtlasBrain(_snapshot(), BrainMemory())

    output = brain.research_plan()

    assert output.kind == "research_plan"
    assert output.details["steps"]
    assert any("Primary research target" in item for item in output.details["priorities"])


def test_brain_produces_self_assessment_output():
    brain = AtlasBrain(_snapshot(), BrainMemory())

    output = brain.self_assessment()

    assert output.kind == "self_assessment"
    assert 0 < output.details["score"] <= 100
    assert output.details["limits"]
    assert output.details["readiness"]


def test_brain_produces_self_model_output():
    brain = AtlasBrain(_snapshot(), BrainMemory())

    output = brain.self_model()

    assert output.kind == "self_model"
    assert "Atlas AI" in output.details["identity"]
    assert output.details["limitations"]
    assert output.details["readiness"]


def test_brain_memory_keeps_recent_turns_for_follow_up_resolution():
    memory = BrainMemory()
    memory.remember_turn(BrainTurn(query="what is the main blocker", intent="prioritization", reasoning_mode="PRIORITIZATION", result="Primary blocker: poor liquidity.", subject="blockers"))
    memory.remember_turn(BrainTurn(query="why does that matter", intent="follow_up", reasoning_mode="EXPLANATION", result="It blocks trades.", subject="blockers"))

    turns = memory.recent_relevant_turns(2)

    assert len(turns) == 2
    assert turns[-1].subject == "blockers"


def test_brain_memory_keeps_up_to_ten_recent_turns():
    memory = BrainMemory()
    for idx in range(12):
        memory.remember_turn(BrainTurn(query=f"q{idx}", intent="market_context", reasoning_mode="STATE", result=f"r{idx}", subject="market"))

    turns = memory.recent_relevant_turns(10)

    assert len(turns) == 10
    assert turns[0].query == "q2"


def test_brain_action_guidance_for_consensus_recommends_upstream_improvement():
    brain = AtlasBrain(
        BrainSnapshot(
            status={
                "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
                "action": "blocked",
                "context_snapshot": {"regime": "range", "spread_state": "normal", "liquidity_state": "unknown", "volatility_state": "normal"},
                "regime": {"name": "range", "trend_strength": 0.08},
                "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
            },
            events=[{"type": "risk", "message": "Consensus guard blocked trade: side=buy, support=1, required=2."}],
            runtime={},
            learning_state={},
            comparison={},
        )
    )

    output = brain.action_guidance("consensus")

    assert output.kind == "action_guidance"
    assert output.details["component"] == "consensus"
    assert output.details["recommendation"] == "improve upstream"
    assert "primary blocker" in output.details["reasoning"].lower() or "current primary blocker" in output.details["reasoning"].lower()


def test_brain_action_guidance_uses_missed_entry_repair_candidate():
    brain = AtlasBrain(
        BrainSnapshot(
            status={
                "message": "Consensus guard blocked trade: side=sell, support=1, required=2.",
                "action": "blocked",
                "context_snapshot": {"regime": "range", "spread_state": "normal", "liquidity_state": "unknown", "volatility_state": "normal"},
                "regime": {"name": "range", "trend_strength": 0.08},
                "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
                "missed_entry_learning_summary": {
                    "available": True,
                    "top_repair_candidate": {
                        "primary_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2.",
                        "repair_priority": "high",
                        "suggested_action": "review_blocker_selectivity",
                        "samples": 150,
                        "missed_opportunity": 55,
                        "correct_hold_or_block": 25,
                    },
                },
            },
            events=[{"type": "risk", "message": "Consensus guard blocked trade: side=sell, support=1, required=2."}],
            runtime={},
            learning_state={},
            comparison={},
        )
    )

    output = brain.action_guidance("consensus")

    assert output.details["recommendation"] == "run experiment"
    assert output.details["missed_entry_repair_candidate"]["repair_priority"] == "high"
    assert "missed opportunities" in output.details["reasoning"].lower()
    assert "offline" in output.details["next_step"].lower()
