from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.trader import AtlasTrader


class TailOnlyStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple[int, str | None]] = []

    def tail_events(self, limit: int, event_type: str | None = None) -> list[dict]:
        self.calls.append((limit, event_type))
        matches = self.rows
        if event_type:
            wanted = event_type.strip().lower()
            matches = [row for row in matches if str(row.get("type", "")).strip().lower() == wanted]
        return matches[-limit:]

    def load_events(self, *args, **kwargs):
        raise AssertionError("trader must not load the full event journal")


def _trader(store: TailOnlyStore) -> AtlasTrader:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader._step_event_cache = {}
    return trader


def test_tail_events_cached_uses_bounded_filtered_tail() -> None:
    rows = [
        {"type": "risk", "message": "ignored"},
        {"type": "decision_sample", "message": "first"},
        {"type": "decision_sample", "message": "second"},
    ]
    store = TailOnlyStore(rows)
    trader = _trader(store)

    result = trader._tail_events_cached(1, "decision_sample")

    assert [row["message"] for row in result] == ["second"]
    assert store.calls == [(1, "decision_sample")]


def test_decision_sample_maps_uses_bounded_filtered_tail() -> None:
    rows = [
        {"type": "risk", "message": "ignored", "meta": {}},
        {
            "type": "decision_sample",
            "message": "sample",
            "meta": {
                "decision_id": "ABCDEF123",
                "strategy": "ema_crossover",
                "order_id": "42",
                "confidence": 0.6,
            },
        },
    ]
    store = TailOnlyStore(rows)
    trader = _trader(store)
    trader._decision_sample_cache = {"loaded_at": 0.0, "limit": 0, "maps": ({}, {}, {})}
    trader._strategy_key = lambda value: str(value).strip().lower()

    strategy_map, aliases, meta_by_ref = trader._decision_sample_maps(limit=50, ttl_seconds=15.0)

    assert strategy_map["abcdef123"] == "ema_crossover"
    assert aliases["abcdef"] == "abcdef123"
    assert meta_by_ref["42"]["decision_id"] == "abcdef123"
    assert store.calls == [(50, "decision_sample")]
