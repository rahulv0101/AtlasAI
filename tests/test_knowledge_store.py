import gc
import json
import os
import tracemalloc

import pytest

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.runtime_heartbeat import start_runtime_heartbeat


def test_backend_heartbeat_file_records_cycle_identity_and_count(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    first = store.write_backend_heartbeat(mode="paper", pid=1234)
    second = store.write_backend_heartbeat(mode="paper", pid=1234)
    loaded = store.load_heartbeat()

    assert first["schema"] == "atlas_backend_heartbeat_v1"
    assert first["component"] == "backend_trader"
    assert first["cycle_count"] == 1
    assert second["cycle_count"] == 2
    assert loaded["cycle_count"] == 2
    assert loaded["mode"] == "paper"
    assert loaded["pid"] == 1234
    assert loaded["last_heartbeat"]
    assert (tmp_path / "heartbeat.json").exists()


def test_backend_heartbeat_mirrors_to_runtime_flags(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    heartbeat = store.write_backend_heartbeat(mode="live", pid=4321)
    flags = store.load_runtime_flags()
    row = dict(dict(flags.get("runtime_heartbeat", {}) or {}).get("backend_trader", {}) or {})

    assert row["cycle_count"] == heartbeat["cycle_count"]
    assert row["mode"] == "live"
    assert row["pid"] == 4321
    assert row["last_heartbeat_ts"] == heartbeat["last_heartbeat_ts"]
    assert row["heartbeat_file"].endswith("heartbeat.json")


def test_ui_heartbeat_file_records_independent_ui_state(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    first = store.write_ui_heartbeat(pid=8765, cycle_status="starting")
    second = store.write_ui_heartbeat(pid=8765, cycle_status="running")
    loaded = store.load_ui_heartbeat()

    assert first["schema"] == "atlas_ui_heartbeat_v1"
    assert first["component"] == "ui_server"
    assert first["cycle_count"] == 1
    assert second["cycle_count"] == 2
    assert loaded["cycle_count"] == 2
    assert loaded["pid"] == 8765
    assert loaded["cycle_status"] == "running"
    assert (tmp_path / "ui_heartbeat.json").exists()
    assert not (tmp_path / "heartbeat.json").exists()


def test_ui_runtime_heartbeat_writes_separate_ui_heartbeat_file(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    stop = start_runtime_heartbeat(store, component="ui_server", interval_seconds=0.15, stale_after_seconds=0.45)
    try:
        heartbeat = store.load_ui_heartbeat()
    finally:
        stop.set()

    assert heartbeat["schema"] == "atlas_ui_heartbeat_v1"
    assert heartbeat["component"] == "ui_server"
    assert heartbeat["cycle_count"] >= 1
    assert (tmp_path / "ui_heartbeat.json").exists()
    assert not (tmp_path / "heartbeat.json").exists()


def test_update_learning_state_preserves_existing_keys(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "allocator": {"momentum_continuation": {"wins": 2, "losses": 1, "pnl": 10.0}},
            "loss_memory": {"recent_outcomes": [{"ticket": "1", "outcome": "loss"}]},
        }
    )

    store.update_learning_state(lambda state: {**state, "win_memory": {"recent_wins": [{"ticket": "2"}]}})

    state = store.load_learning_state().get("state", {})
    assert "allocator" in state
    assert "loss_memory" in state
    assert state["loss_memory"]["recent_outcomes"][0]["ticket"] == "1"
    assert state["win_memory"]["recent_wins"][0]["ticket"] == "2"


def test_write_json_atomic_falls_back_to_in_place_overwrite(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    target = tmp_path / "learning_state.json"
    calls = {"count": 0}
    real_replace = __import__("os").replace

    def flaky_replace(src, dst):
        calls["count"] += 1
        if calls["count"] <= 12:
            raise PermissionError("simulated windows sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr("atlas_ai.memory.knowledge_store.os.replace", flaky_replace)

    store.save_learning_state({"allocator": {"x": {"wins": 1, "losses": 0, "pnl": 1.0}}})

    state = store.load_learning_state().get("state", {})
    assert state["allocator"]["x"]["wins"] == 1
    assert target.exists()


def test_learning_state_normalizes_truncated_strategy_aliases(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "allocator": {
                "macd_tren": {"wins": 2, "losses": 3, "pnl": -5.0},
                "macd_trend": {"wins": 4, "losses": 1, "pnl": 8.0},
                "vwap_prox": {"wins": 1, "losses": 2, "pnl": -3.0},
            },
            "loss_memory": {
                "recent_outcomes": [
                    {"ticket": "1", "strategy": "macd_tren"},
                    {"ticket": "2", "strategy": "vwap_prox"},
                ]
            },
            "win_memory": {
                "context_stats": {
                    "macd_tren|sell|london|11": {"wins": 2, "pnl": 7.0},
                    "macd_trend|sell|london|11": {"wins": 1, "pnl": 1.0},
                },
                "recent_wins": [{"ticket": "3", "strategy": "vwap_prox"}],
            },
        }
    )

    state = store.load_learning_state().get("state", {})

    assert "macd_tren" not in state["allocator"]
    assert state["allocator"]["macd_trend"]["wins"] == 6
    assert state["allocator"]["macd_trend"]["losses"] == 4
    assert state["allocator"]["vwap_proxy_reversion"]["wins"] == 1
    assert state["loss_memory"]["recent_outcomes"][0]["strategy"] == "macd_trend"
    assert state["loss_memory"]["recent_outcomes"][1]["strategy"] == "vwap_proxy_reversion"
    assert state["win_memory"]["context_stats"]["macd_trend|sell|london|11"]["wins"] == 3
    assert state["win_memory"]["recent_wins"][0]["strategy"] == "vwap_proxy_reversion"


def test_load_events_limit_applies_after_type_filter(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.log_event("risk", "first")
    store.log_event("decision_sample", "a", {"decision_id": "1"})
    store.log_event("risk", "second")
    store.log_event("decision_sample", "b", {"decision_id": "2"})
    store.log_event("decision_sample", "c", {"decision_id": "3"})

    rows = store.load_events(limit=2, event_type="decision_sample")

    assert [row["message"] for row in rows] == ["b", "c"]


def test_tail_events_skips_malformed_json_lines(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.log_event("risk", "first")
    with store.events_file.open("a", encoding="utf-8") as fh:
        fh.write('{"type":"broken","message":"oops"\n')
    store.log_event("risk", "second")

    rows = store.tail_events(limit=10)

    assert [row["message"] for row in rows] == ["first", "second"]


def test_tail_events_reads_only_last_valid_records_without_load_events(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    expected = []
    with store.events_file.open("w", encoding="utf-8") as fh:
        for idx in range(60):
            row = {
                "time": f"time-{idx}",
                "type": "test",
                "message": f"event-{idx}",
                "meta": {"idx": idx, "payload": "x" * 4096},
            }
            expected.append(row)
            fh.write(json.dumps(row) + "\n")
            if idx in {42, 53}:
                fh.write('{"malformed"\n')
        fh.write("\n")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("tail_events called load_events")

    monkeypatch.setattr(store, "load_events", fail_if_called)

    assert store.tail_events(20) == expected[-20:]


def test_tail_events_peak_memory_does_not_scale_with_event_file_size(tmp_path):
    payload = "x" * 512

    def build_and_measure(name: str, count: int) -> tuple[list[dict], int]:
        store = KnowledgeStore(str(tmp_path / name))
        with store.events_file.open("w", encoding="utf-8") as fh:
            for idx in range(count):
                record = {"type": "test", "message": str(idx), "meta": {"payload": payload}}
                fh.write(json.dumps(record) + "\n")

        gc.collect()
        tracemalloc.start()
        try:
            rows = store.tail_events(10)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return rows, peak

    small_rows, small_peak = build_and_measure("small", 1_000)
    large_rows, large_peak = build_and_measure("large", 30_000)

    assert [row["message"] for row in small_rows] == [str(idx) for idx in range(990, 1_000)]
    assert [row["message"] for row in large_rows] == [str(idx) for idx in range(29_990, 30_000)]
    assert large_peak < 4 * 1024 * 1024
    assert large_peak <= small_peak + 2 * 1024 * 1024


def test_tail_events_filters_by_type_across_the_journal(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    expected = []
    with store.events_file.open("w", encoding="utf-8") as fh:
        for idx in range(80):
            event_type = "decision_sample" if idx % 11 == 0 else "risk"
            row = {
                "type": event_type,
                "message": f"event-{idx}",
                "meta": {"idx": idx, "payload": "x" * 2048},
            }
            fh.write(json.dumps(row) + "\n")
            if event_type == "decision_sample":
                expected.append(row)

    rows = store.tail_events(limit=4, event_type=" DECISION_SAMPLE ")

    assert rows == expected[-4:]


def test_tail_events_filtered_scan_memory_stays_bounded_when_matches_are_rare(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    payload = "x" * 512
    expected = []
    with store.events_file.open("w", encoding="utf-8") as fh:
        for idx in range(30_000):
            event_type = "decision_sample" if idx in {3, 101, 997} else "risk"
            row = {"type": event_type, "message": str(idx), "meta": {"payload": payload}}
            fh.write(json.dumps(row) + "\n")
            if event_type == "decision_sample":
                expected.append(row)

    gc.collect()
    tracemalloc.start()
    try:
        rows = store.tail_events(limit=5, event_type="decision_sample")
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert rows == expected
    assert peak < 4 * 1024 * 1024
    assert store._events_cache == []
    assert store._events_cache_offset == 0


def test_maintain_events_streams_keeps_latest_and_archives(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    for idx in range(6):
        store.log_event("risk", f"row-{idx}")
    with store.events_file.open("a", encoding="utf-8") as fh:
        fh.write('{"broken"\n')

    result = store.maintain_events(max_lines=3)
    rows = store.load_events(limit=10)

    assert result["kept"] == 3
    assert result["archived"] == 3
    assert result["dropped_malformed"] == 1
    assert [row["message"] for row in rows] == ["row-3", "row-4", "row-5"]
    assert list((tmp_path / "archive").glob("events_*.jsonl"))


@pytest.mark.parametrize("limit", [0, 1, 3, 100])
def test_maintenance_preserves_exact_retention_and_deduplication(tmp_path, limit):
    store = KnowledgeStore(str(tmp_path))
    rows = [' {"b": 2, "a": 1} ', '{"a":1,"b":2}', '', '{broken',
            '{"text":"caf\u00e9"}', '[1, 2]', 'null', '42', '{"b":2,"a":1}']
    store.events_file.write_text("\n".join(rows), encoding="utf-8")
    expected = [rows[0].strip(), rows[4], rows[5], rows[6], rows[7]]
    retained = min(len(expected), max(1, limit))
    store.load_events()  # Maintenance must also invalidate any existing cache.
    result = store.maintain_events(max_lines=limit)
    assert result == {"kept": retained, "archived": len(expected) - retained,
                      "dropped_malformed": 1, "dropped_duplicates": 2}
    assert store.events_file.read_bytes() == (os.linesep.join(expected[-retained:]) + os.linesep).encode("utf-8")
    archives = list((tmp_path / "archive").glob("events_*.jsonl"))
    if len(expected) > retained:
        assert len(archives) == 1
        assert archives[0].read_bytes() == (os.linesep.join(expected[:-retained]) + os.linesep).encode("utf-8")
    else:
        assert archives == []
    assert store._events_cache == []
    assert store._events_cache_offset == 0


def test_maintenance_hash_collision_does_not_discard_distinct_records(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("atlas_ai.memory.knowledge_store.hashlib.sha256",
                        lambda *_: SimpleNamespace(digest=lambda: b"same-digest"))
    store = KnowledgeStore(str(tmp_path))
    store.events_file.write_text('{"id":1}\n{"id":2}\n{"id":1}\n{"id":3}\n', encoding="utf-8")
    result = store.maintain_events(max_lines=2)
    assert result == {"kept": 2, "archived": 1, "dropped_duplicates": 1, "dropped_malformed": 0}
    assert store.tail_events(10) == [{"id": 2}, {"id": 3}]


@pytest.mark.parametrize("content", [None, "", "\n{broken\n"])
def test_maintenance_empty_or_missing_journal(tmp_path, content):
    store = KnowledgeStore(str(tmp_path))
    if content is not None:
        store.events_file.write_text(content, encoding="utf-8")
    result = store.maintain_events(10)
    assert result["kept"] == result["archived"] == 0
    assert not list((tmp_path / "archive").glob("events_*.jsonl"))


def test_maintenance_keeps_payloads_on_disk_not_in_memory(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    with store.events_file.open("w", encoding="utf-8") as stream:
        for i in range(4000):
            stream.write(json.dumps({"id": i, "payload": "x" * 8192}) + "\n")
    assert store.events_file.stat().st_size > 30 * 1024 * 1024
    monkeypatch.setattr(store, "load_events", lambda *a, **k: pytest.fail("Must not load full history"))
    gc.collect()
    tracemalloc.start()
    try:
        result = store.maintain_events(max_lines=3000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["kept"] == 3000
    assert result["archived"] == 1000
    assert peak < 4 * 1024 * 1024
    assert store.tail_events(1)[0]["id"] == 3999


def test_maintenance_failed_replace_keeps_original_and_cleans_scratch(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    original = '{"id":1}\n{"id":2}\n'
    store.events_file.write_text(original, encoding="utf-8")
    before = store.events_file.read_bytes()
    def fail_replace(*args):
        raise PermissionError("reader still holds file")
    monkeypatch.setattr("atlas_ai.memory.knowledge_store.os.replace", fail_replace)
    with pytest.raises(PermissionError):
        store.maintain_events(max_lines=1)
    assert store.events_file.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
