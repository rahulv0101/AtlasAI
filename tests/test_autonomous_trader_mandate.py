from __future__ import annotations

from atlas_ai.analysis.trader_cognition import build_trader_cognition
from atlas_ai.brain.intent import classify_intent_candidates
from atlas_ai.brain.self_model import build_self_model
from atlas_ai.execution.execution_gate import ExecutionGate


def test_trader_cognition_exposes_bounded_service_intent_without_claiming_feelings() -> None:
    status = {
        "action": "blocked",
        "message": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
        "chosen": {"side": "buy", "confidence": 0.51, "strategy": "momentum_continuation"},
        "higher_timeframe": {"bias": "sell"},
        "same_side_support": {"count": 1},
        "thresholds": {"confidence": 0.55},
        "context_snapshot": {
            "session": "new_york",
            "trend_structure": "HH/HL",
            "momentum": "mixed",
            "location": "pullback",
            "spread_state": "normal",
        },
        "decision_analysis": {"hard_blockers": ["higher_timeframe_alignment"]},
    }
    mandate = {
        "operator_intent": {
            "primary_service_goal": "Serve the operator by pursuing profitable XAUUSD trading decisions."
        },
        "autonomy_model": {"target_role": "bounded_autonomous_trader"},
    }

    out = build_trader_cognition(status=status, runtime={"min_same_side_signals": 2}, mandate=mandate)

    assert out["schema"] == "atlas_trader_cognition_v1"
    assert out["decision"] == "block"
    assert out["service_intent"]["real_feelings"] is False
    assert "Serve the operator" in out["service_intent"]["statement"]
    assert "higher-timeframe" in " ".join(out["required_next_evidence"])
    assert out["autonomy_boundary"]["execution_authority"] == "ExecutionGate"
    assert out["autonomy_boundary"]["does_not_change_risk_policy"] is True


def test_guarded_autonomous_demo_mode_is_supported_and_demo_only() -> None:
    gate = ExecutionGate()
    snap = gate.set_autonomy_mode("guarded_autonomous_demo")

    assert snap["autonomy_mode"] == "guarded_autonomous_demo"
    semantics = snap["autonomy_mode_semantics"]["guarded_autonomous_demo"]
    assert semantics["can_create_execution_request"] is True
    assert semantics["requires_existing_gates"] is True
    assert semantics["demo_only"] is True
    assert semantics["real_live_allowed"] is False


def test_execution_snapshot_exposes_autonomous_trader_mandate_summary() -> None:
    gate = ExecutionGate()
    snap = gate.snapshot()
    mandate = dict(snap.get("autonomous_trader_mandate", {}) or {})

    assert mandate["schema"] == "atlas_autonomous_trader_mandate_v1"
    assert mandate["target_role"] == "bounded_autonomous_trader"
    assert mandate["execution_authority"] == "ExecutionGate"


def test_self_model_describes_bounded_autonomous_trader_without_real_feelings() -> None:
    runtime = {"execution": {"autonomy_mode": "guarded_autonomous_demo", "autonomous_trader_mandate": {"target_role": "bounded_autonomous_trader"}}}
    model = build_self_model({"chosen": {"confidence": 0.6}, "trader_cognition": {"autonomy_boundary": {"target_role": "bounded_autonomous_trader"}}}, runtime, {})

    assert model["identity"] == "Atlas bounded trading operator"
    assert model["target_role"] == "bounded_autonomous_trader"
    assert model["autonomy_mode"] == "guarded_autonomous_demo"
    assert "no real feelings" in model["feeling_model"].lower()
    assert any("Trader-like cognition" in item for item in model["capabilities"])
    assert any("No real emotions" in item for item in model["limitations"])


def test_desire_and_autonomous_trader_questions_route_to_self_knowledge() -> None:
    for query in ("Do you have desire?", "Do you have feelings?", "Are you an autonomous trader?"):
        top = classify_intent_candidates(query)[0]
        assert top.name == "self_knowledge"
