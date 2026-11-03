from __future__ import annotations

from atlas_ai.brain.knowledge import BrainKnowledge
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.2},
            "min_confidence_threshold": 0.58,
            "comparison": {"run_id": "atlas_wake_latest"},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "Conservative pre-trade filter blocked trade."}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest"}


def _ask(q: str) -> dict:
    return AskAtlasService(_Source()).answer(q)


def test_knowledge_loader_api() -> None:
    k = BrainKnowledge()
    vol = k.get_market_concept("volatility")
    assert vol is not None
    assert "definition" in vol
    strat = k.get_strategy("continuation")
    assert strat is not None
    assert "conditions" in strat
    rules = k.get_risk_rules()
    assert len(rules) >= 1
    hold = k.get_glossary_term("hold")
    assert hold is not None
    blocker = k.get_blocker_info("No high-confidence setup")
    assert blocker is not None


def test_volatility_question_uses_market_knowledge() -> None:
    r = _ask("What is volatility?")
    assert r["selected_domain"] == "market_knowledge"
    assert "volatility" in r["result"].lower()
    assert "status:" not in r["result"].lower()


def test_trend_question_uses_market_knowledge() -> None:
    r = _ask("What is a trend?")
    assert r["selected_domain"] == "market_knowledge"
    assert "trend" in r["result"].lower()


def test_liquidity_question_selects_liquidity_concept_not_first_concept() -> None:
    r = _ask("What is liquidity?")
    assert r["selected_domain"] == "market_knowledge"
    assert "liquidity" in r["result"].lower()
    assert "volatility is" not in r["result"].lower()


def test_momentum_question_selects_momentum_concept_not_first_concept() -> None:
    r = _ask("Explain momentum quality")
    assert r["selected_domain"] == "market_knowledge"
    assert "momentum_quality" in r["result"].lower()
    assert "volatility is" not in r["result"].lower()


def test_strategy_question_uses_strategy_knowledge() -> None:
    r = _ask("What strategies do you use?")
    assert r["selected_domain"] == "strategy_knowledge"
    text = r["result"].lower()
    assert "active executable strategies:" in text
    assert "momentum_continuation" in text
    assert "gold_micro_impulse" in text
    assert "strategy concepts atlas can explain:" in text
    assert "evolved strategy family:" in text


def test_only_strategies_question_does_not_collapse_to_continuation() -> None:
    r = _ask("Are these the only strategies Atlas knows?")
    assert r["selected_domain"] == "strategy_knowledge"
    text = r["result"].lower()
    assert text.startswith("no.")
    assert "multiple strategy layers" in text
    assert "continuation\n" not in text[:80]


def test_worker_neuron_question_uses_brain_coordination_knowledge() -> None:
    r = _ask("Why not give workers to neurons so it works with the brain?")
    assert r["selected_domain"] == "worker_neurons"
    text = r["result"].lower()
    assert "workers as evidence neurons" in text
    assert "brain contract" in text
    assert "executiongate" in text
    assert "status:" not in text[:80]


def test_specific_strategy_question_selects_matching_strategy() -> None:
    r = _ask("How does breakout retest work?")
    assert r["selected_domain"] == "strategy_knowledge"
    text = r["result"].lower()
    assert "breakout_retest" in text
    assert "breakout confirmed with follow-through" in text
    assert "fakeout behavior" in text


def test_risk_rules_question_uses_risk_policy() -> None:
    r = _ask("What are your risk rules?")
    assert r["selected_domain"] == "risk_policy"
    assert "risk rules:" in r["result"].lower()
    assert "mode:" not in r["result"].lower()


def test_hold_meaning_uses_glossary() -> None:
    r = _ask("What does hold mean?")
    assert r["selected_domain"] == "glossary"
    assert "hold:" in r["result"].lower()


def test_blocker_question_includes_blocker_catalog() -> None:
    r = _ask("What blocks you from trading?")
    assert r["selected_domain"] == "blockers"
    assert "main blocker:" in r["result"].lower()
    assert "safe response:" in r["result"].lower()


def test_debug_exposes_knowledge_selection() -> None:
    r = _ask("What is volatility?")
    knowledge_selected = list(r.get("debug", {}).get("knowledge_selected", []) or [])
    assert len(knowledge_selected) >= 1
    assert any("knowledge.market_knowledge" in str(x.get("source", "")) for x in knowledge_selected)
