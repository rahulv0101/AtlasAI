from __future__ import annotations

import logging
import time
from types import SimpleNamespace

from atlas_ai import macro_context_updater as updater
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.trader import AtlasTrader


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_macro_context_builder_maps_price_proxy_fields(monkeypatch) -> None:
    data = {
        "DX-Y.NYB": [100.0, 99.0],
        "^TNX": [42.0, 43.0],
        "^VIX": [17.0, 18.0],
        "CL=F": [70.0, 72.0],
    }
    monkeypatch.setattr(updater, "_fetch_chart", lambda symbol: data[symbol])

    payload = updater.build_macro_context()
    items = {row["key"]: row["value"] for row in payload["items"]}

    assert items["dxy_trend"] == "falling"
    assert items["dxy_alignment"] == "supportive"
    assert items["us10y_trend"] == "rising"
    assert items["us10y_alignment"] == "conflict"
    assert items["vix_regime"] == "normal"
    assert items["oil_backdrop"] == "inflationary"
    assert items["macro_classification"] == "price_proxy_only"


def test_write_external_context_uses_repo_visible_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        updater,
        "build_macro_context",
        lambda: {"generated_at": "2026-05-12T00:00:00+00:00", "source": "test", "items": []},
    )

    target = tmp_path / "external_context.json"
    payload = updater.write_external_context(target)

    assert payload["source"] == "test"
    assert target.exists()
    assert "generated_at" in target.read_text(encoding="utf-8")


def test_backend_live_cycle_macro_refresh_runs_once_per_interval(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_write_external_context():
        calls.append("refresh")
        return {"generated_at": "2026-05-12T00:00:00+00:00", "source": "test", "errors": {}}

    monkeypatch.setenv("ATLAS_MACRO_CONTEXT_REFRESH_SECONDS", "3600")
    monkeypatch.setattr(updater, "write_external_context", fake_write_external_context)

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = KnowledgeStore(str(tmp_path))
    trader.settings = SimpleNamespace(storage_dir=str(tmp_path))

    trader._maybe_refresh_external_macro_context()
    assert _wait_until(lambda: not bool(getattr(trader, "_macro_context_refresh_in_flight", False)))

    trader._maybe_refresh_external_macro_context()
    time.sleep(0.05)

    assert calls == ["refresh"]
    events = trader.store.load_events(event_type="system")
    assert any(row["message"] == "External macro context refreshed by backend live cycle." for row in events)


def test_backend_macro_refresh_failure_is_visible(monkeypatch, tmp_path, caplog) -> None:
    def fail_write_external_context():
        raise RuntimeError("macro source unavailable")

    monkeypatch.setenv("ATLAS_MACRO_CONTEXT_REFRESH_SECONDS", "3600")
    monkeypatch.setattr(updater, "write_external_context", fail_write_external_context)

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = KnowledgeStore(str(tmp_path))
    trader.settings = SimpleNamespace(storage_dir=str(tmp_path))

    with caplog.at_level(logging.WARNING):
        trader._maybe_refresh_external_macro_context()
        assert _wait_until(lambda: not bool(getattr(trader, "_macro_context_refresh_in_flight", False)))

    assert "External macro context refresh failed in backend live cycle" in caplog.text
    events = trader.store.load_events(event_type="warning")
    assert any(row["message"] == "External macro context refresh failed in backend live cycle." for row in events)
