from __future__ import annotations

import json
import time

from atlas_ai.brain.why_engine import WHY_TRACE_STAGES, build_why_trace
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.trader import AtlasTrader, TradeSignal
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.ui_server import create_ui_app


class _Source:
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "Approval is missing.",
            "symbol": "XAUUSD",
            "timeframe": "M1",
            "chosen": {"strategy": "continuation", "side": "buy", "confidence": 0.72},
            "contextual_ranking": {"chosen_score": 0.72, "top": [{"strategy": "continuation", "score": 0.72}]},
            "conservative_pretrade_filter": {"enabled": True, "blocked": False, "reason": "Conservative filter passed."},
            "risk": {"daily_loss_pct": 0.0, "spread_points": 12, "open_positions": 0},
        }

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_events(self, limit: int = 160) -> list[dict]:
        return []

    def load_comparison(self) -> dict:
        return {}


class _StoreSource(_Source):
    def __init__(self, store: KnowledgeStore) -> None:
        self.store = store

    def load_status(self) -> dict:
        return self.store.load_status()

    def load_runtime_flags(self) -> dict:
        return self.store.load_runtime_flags()

    def load_events(self, limit: int = 160) -> list[dict]:
        return self.store.load_events(limit=limit)


def _write_heartbeat(store: KnowledgeStore, *, last_ts: float, cycle_count: int = 1) -> None:
    store.heartbeat_file.write_text(
        json.dumps(
            {
                "schema": "atlas_backend_heartbeat_v1",
                "component": "backend_trader",
                "last_heartbeat": "2026-01-01T00:00:00+00:00",
                "last_heartbeat_ts": last_ts,
                "mode": "paper",
                "pid": 1234,
                "cycle_count": cycle_count,
                "cycle_status": "running",
            }
        ),
        encoding="utf-8",
    )


def test_why_trace_returns_ordered_pipeline_with_execution_gate_block() -> None:
    trace = build_why_trace(
        _Source().load_status(),
        {"execute_trades": False},
        {
            "gate_decisions_last_cycle": {
                "allowed": False,
                "reason": "approval_missing",
                "reason_code": "APPROVAL_MISSING",
                "checked_at": "2026-01-01T00:00:00+00:00",
            }
        },
    )
    assert [row["stage"] for row in trace["stages"]] == list(WHY_TRACE_STAGES)
    assert trace["stages"][-1]["stage"] == "execution"
    assert trace["stages"][-1]["status"] == "blocked"
    assert trace["stages"][-1]["data"]["gate_decision"]["reason_code"] == "APPROVAL_MISSING"
    assert trace["read_only"] is True


def test_ask_atlas_exposes_read_only_why_trace() -> None:
    svc = AskAtlasService(_Source())
    out = svc.why_trace()
    assert out["schema"] == "atlas_why_trace_v1"
    assert [row["stage"] for row in out["stages"]] == list(WHY_TRACE_STAGES)


def test_ask_atlas_why_trading_uses_execution_gate_reason() -> None:
    svc = AskAtlasService(_Source())
    svc.execution.set_gate_enabled(True)
    req = svc.execution.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    svc.execution.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})

    out = svc.answer("why isn't it trading")
    result = str(out.get("result", ""))
    assert out["intent"] == "blocker"
    assert "Blocked at execution" in result
    assert "APPROVAL_MISSING" in result
    assert "approval_missing" in result


def test_ask_atlas_why_trading_prefers_structured_timing_discipline_block(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    trader = AtlasTrader.__new__(AtlasTrader)
    signal = TradeSignal("momentum_continuation", "buy", 0.72, 100, 160)
    timing = trader._timing_discipline_snapshot(
        signal,
        {
            "location": "mid_range",
            "momentum": "mixed",
            "trend_state": "trend",
            "trend_strength": 0.26,
            "session": "new_york",
            "setup_type_hint": "pullback_continuation",
        },
        {"learning_summary": {"loss_reason_summary": {"top_reason": "stop_hunt"}}},
    )
    store.save_status(
        {
            "action": "blocked",
            "message": trader._operator_message_from_timing_discipline(timing),
            "execute_trades": False,
            "chosen": {"strategy": signal.strategy, "side": signal.side, "confidence": signal.confidence},
            "timing_discipline": timing,
        }
    )
    store.save_runtime_flags({"execute_trades": False})
    store.write_backend_heartbeat(mode="paper", pid=1234)

    for query in ("why isn't it trading", "why is it not trading", "why no trade"):
        out = AskAtlasService(_StoreSource(store)).answer(query)
        result = str(out.get("result", ""))
        assert out["intent"] in {"blocker", "explanation_request"}
        assert "Blocked at conservative filter" in result or "Blocked by timing discipline" in result
        assert "AWAITING_TRADABLE_LOCATION" in result
        assert "location=mid_range" in result
        assert "momentum=mixed" in result
        assert "session=new_york" in result
        assert "Main blocker: No high-confidence setup" not in result


def test_ask_atlas_warns_when_backend_heartbeat_is_stale_for_blocker_question(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_HEARTBEAT_STALE_AFTER_SECONDS", "10")
    store = KnowledgeStore(str(tmp_path))
    store.save_status(_Source().load_status())
    store.save_runtime_flags({"execute_trades": False})
    now = time.time()
    _write_heartbeat(store, last_ts=now - 30)

    out = AskAtlasService(_StoreSource(store)).answer("why isn't it trading")

    result = str(out.get("result", ""))
    assert out["backend_health"]["status"] == "stale"
    assert "Warning: Backend has not reported in" in result
    assert "Last-known answer:" in result


def test_ask_atlas_does_not_warn_when_backend_heartbeat_is_fresh_for_blocker_question(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BACKEND_HEARTBEAT_STALE_AFTER_SECONDS", "10")
    store = KnowledgeStore(str(tmp_path))
    store.save_status(_Source().load_status())
    store.save_runtime_flags({"execute_trades": False})
    _write_heartbeat(store, last_ts=time.time())

    out = AskAtlasService(_StoreSource(store)).answer("why isn't it trading")

    result = str(out.get("result", ""))
    assert "backend_health" not in out
    assert "Warning: Backend" not in result
    assert "Last-known answer:" not in result


def test_why_api_returns_trace_from_existing_store_state(tmp_path, monkeypatch) -> None:
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(_Source().load_status())
    store.save_runtime_flags({"execute_trades": False})
    app = create_ui_app(store)
    svc = app.view_functions["ask_atlas"].__closure__[0].cell_contents
    svc.execution.set_gate_enabled(True)
    req = svc.execution.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    svc.execution.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})

    payload = app.test_client().get("/api/why").get_json()
    assert payload["ok"] is True
    assert [row["stage"] for row in payload["why"]["stages"]] == list(WHY_TRACE_STAGES)
    assert payload["why"]["stages"][-1]["data"]["gate_decision"]["reason_code"] == "APPROVAL_MISSING"
