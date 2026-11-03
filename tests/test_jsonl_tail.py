from __future__ import annotations

import json
import tracemalloc
from pathlib import Path

from atlas_ai.memory.jsonl_tail import read_jsonl_tail
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _append(path: Path, rows: list[object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if isinstance(row, str):
                handle.write(row + "\n")
            else:
                handle.write(json.dumps(row) + "\n")


def test_read_jsonl_tail_returns_last_matches_in_source_order(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _append(
        path,
        [
            {"type": "risk", "id": 1},
            {"type": "decision", "id": 2, "side": "buy"},
            {"type": "decision", "id": 3, "side": "sell"},
            {"type": "decision", "id": 4, "side": "buy"},
            {"type": "decision", "id": 5, "side": "buy"},
        ],
    )

    rows = read_jsonl_tail(
        path,
        2,
        event_type_filter=" DECISION ",
        matcher=lambda row: row.get("side") == "buy",
    )

    assert [row["id"] for row in rows] == [4, 5]


def test_read_jsonl_tail_skips_blank_malformed_and_non_object_lines(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n"
        + json.dumps({"id": 1})
        + "\nnot-json\n[]\n   \n"
        + json.dumps({"id": 2})
        + "\n{broken\n"
        + json.dumps({"id": 3}),
        encoding="utf-8",
    )

    assert [row["id"] for row in read_jsonl_tail(path, 3)] == [1, 2, 3]
    assert read_jsonl_tail(tmp_path / "missing.jsonl", 5) == []


def test_read_jsonl_tail_never_uses_full_file_text_read(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "events.jsonl"
    _append(path, [{"id": index} for index in range(20)])

    def fail_read_text(*args, **kwargs):
        raise AssertionError("bounded JSONL tail must not call Path.read_text")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    assert [row["id"] for row in read_jsonl_tail(path, 2)] == [18, 19]


def test_read_jsonl_tail_memory_is_bounded_for_large_sparse_match_scan(tmp_path: Path) -> None:
    path = tmp_path / "large.jsonl"
    payload = "x" * 1024
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "target", "id": 0, "payload": payload}) + "\n")
        for index in range(25_000):
            handle.write(json.dumps({"type": "noise", "id": index, "payload": payload}) + "\n")

    tracemalloc.start()
    try:
        rows = read_jsonl_tail(path, 5, event_type_filter="target")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert [row["id"] for row in rows] == [0]
    assert path.stat().st_size > 20 * 1024 * 1024
    assert peak < 3 * 1024 * 1024


def test_knowledge_store_tail_events_uses_shared_reader(tmp_path: Path, monkeypatch) -> None:
    store = KnowledgeStore(tmp_path)
    store.log_event("risk", "one")
    store.log_event("decision", "two")

    def fail_load_events(*args, **kwargs):
        raise AssertionError("tail_events must not call the full-history loader")

    monkeypatch.setattr(store, "load_events", fail_load_events)
    rows = store.tail_events(1, event_type="decision")
    assert [row["message"] for row in rows] == ["two"]
