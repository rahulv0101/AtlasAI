from __future__ import annotations

from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_load_events_uses_cached_snapshot_and_filters_recent_rows(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    for idx in range(6):
        store.log_event("risk" if idx % 2 else "system", f"event {idx}", {"idx": idx})

    first = store.load_events(limit=3)
    second = store.load_events(limit=3)
    assert [x["message"] for x in first] == ["event 3", "event 4", "event 5"]
    assert second == first

    risk = store.load_events(limit=2, event_type="risk")
    assert [x["message"] for x in risk] == ["event 3", "event 5"]


def test_event_cache_ingests_appended_rows_without_losing_history(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.log_event("system", "first")
    assert [x["message"] for x in store.load_events()] == ["first"]

    store.log_event("risk", "second")
    assert [x["message"] for x in store.load_events()] == ["first", "second"]
    assert [x["message"] for x in store.tail_events(limit=1)] == ["second"]


def test_new_store_does_not_drop_existing_events_when_logging_before_read(tmp_path) -> None:
    first_store = KnowledgeStore(str(tmp_path))
    first_store.log_event("system", "old")

    second_store = KnowledgeStore(str(tmp_path))
    second_store.log_event("system", "new")

    assert [x["message"] for x in second_store.load_events()] == ["old", "new"]


def test_status_and_runtime_cache_return_defensive_copies(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_status({"action": "hold", "nested": {"value": 1}})
    store.save_runtime_flags({"execute_trades": True, "nested": {"value": 2}})

    status = store.load_status()
    runtime = store.load_runtime_flags()
    status["nested"]["value"] = 99
    runtime["nested"]["value"] = 99

    assert store.load_status()["nested"]["value"] == 1
    assert store.load_runtime_flags()["nested"]["value"] == 2
