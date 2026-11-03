from __future__ import annotations

from atlas_ai.conversation_layer import ConversationalAtlasInterface
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.operator_assistant import AtlasAssistant


class _Source:
    def __init__(self, store: KnowledgeStore):
        self.store = store

    def load_status(self) -> dict:
        return dict(self.store.load_status() or {})

    def load_events(self, limit: int = 120) -> list[dict]:
        return list(self.store.tail_events(limit=limit))

    def load_learning_state(self) -> dict:
        return dict(self.store.load_learning_state() or {})

    def load_runtime_flags(self) -> dict:
        return dict(self.store.load_runtime_flags() or {})

    def load_comparison(self) -> dict:
        return {}


def test_conversation_layer_maps_informal_state_query(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    convo = ConversationalAtlasInterface(_Source(store), assistant=AtlasAssistant(_Source(store)))

    reply = convo.respond("what's going on")

    assert reply["conversation_query"] == "what is happening now"
    assert reply["reasoning_mode"] == "STATE"


def test_conversation_layer_defaults_to_canonical_ask_atlas_service(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    convo = ConversationalAtlasInterface(_Source(store))

    reply = convo.respond("what can you do")

    assert reply["intent"] == "capabilities"
    assert reply["handled_as"] == "general"
    assert "answer_trace" in reply
    assert reply["answer_trace"]["selected_plan"] == "capabilities_plan"


def test_conversation_layer_maps_market_good_or_bad_to_setup_evaluation(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "context_snapshot": {"spread_state": "tight", "liquidity_state": "poor", "volatility_state": "normal"},
            "regime": {"name": "trend", "trend_strength": 0.21},
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.5,
        }
    )
    convo = ConversationalAtlasInterface(_Source(store), assistant=AtlasAssistant(_Source(store)))

    reply = convo.respond("is the market good or bad")

    assert reply["conversation_query"] == "tell me whether the market is favorable or unfavorable right now"
    assert reply["intent"] == "setup_evaluation"


def test_conversation_layer_uses_recent_memory_for_action_follow_up(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "context_snapshot": {"session": "new_york", "volatility_state": "normal", "spread_state": "normal", "liquidity_state": "unknown"},
            "regime": {"name": "range", "trend_strength": 0.08},
            "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")
    convo = ConversationalAtlasInterface(_Source(store), assistant=AtlasAssistant(_Source(store)))

    first = convo.respond("what should I do about consensus")
    second = convo.respond("what about that")

    assert first["intent"] == "action_guidance"
    assert second["conversation_query"] == "what should I do about consensus"
    assert second["intent"] == "action_guidance"
    assert second["conversation_memory_size"] >= 1
