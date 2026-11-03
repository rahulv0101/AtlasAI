from __future__ import annotations

from pathlib import Path

import atlas_ai.operator_assistant as operator_assistant
import atlas_ai.operator_core as operator_core
import atlas_ai.conversation_layer as conversation_layer
import atlas_ai.ui_server as ui_server
import atlas_ai.voice.wake_loop as wake_loop
from atlas_ai.conversation_layer import ConversationalAtlasInterface
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui.ask_atlas import AskAtlasService


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


def test_routing_contract_documents_canonical_paths():
    text = Path("docs/ROUTING_CONTRACT.md").read_text(encoding="utf-8")

    assert "/api/ask-atlas" in text
    assert "AskAtlasService.answer()" in text
    assert "voice.wake_loop.run_voice_loop()" in text
    assert "answer_trace" in text
    assert "decision_trace" in text
    assert "compatibility-only" in text


def test_legacy_operator_modules_are_marked_compatibility_only():
    assert operator_assistant.LEGACY_COMPATIBILITY_ONLY is True
    assert "AskAtlasService" in operator_assistant.LEGACY_ROUTE_NOTE
    assert operator_core.LEGACY_COMPATIBILITY_ONLY is True


def test_conversation_wrapper_defaults_to_canonical_service(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    convo = ConversationalAtlasInterface(_Source(store))

    assert isinstance(convo.core, AskAtlasService)


def test_active_answer_surfaces_do_not_default_to_legacy_assistant():
    assert "AskAtlasService" in conversation_layer.__dict__
    assert "AtlasAssistant" not in conversation_layer.__dict__
    assert wake_loop.AskAtlasService is AskAtlasService
    assert ui_server.AskAtlasService is AskAtlasService
