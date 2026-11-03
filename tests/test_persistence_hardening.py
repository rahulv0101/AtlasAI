from __future__ import annotations

from atlas_ai.execution.automation_state_store import AutomationStateStore
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_status_save_uses_atomic_locked_update(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "message": "ok"})
    payload = store.load_status()
    assert payload["action"] == "hold"
    assert "updated_at" in payload


def test_automation_state_store_tracks_revision_and_previous_saved_at(tmp_path) -> None:
    state = AutomationStateStore(tmp_path / "automation_state.json")
    first = state.save({"gate_enabled": True}, event="initial")
    second = state.save({"gate_enabled": False}, event="update")
    payload, meta = state.load()
    assert first["ok"] is True
    assert second["ok"] is True
    assert int(second["revision"]) == 2
    assert payload["gate_enabled"] is False
    assert meta["revision"] == 2
    assert meta["saved_at"] is not None
