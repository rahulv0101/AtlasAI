from __future__ import annotations

import time

from atlas_ai.execution.broker_adapter import BrokerOrderResponse, InMemoryBrokerAdapter
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.runtime_heartbeat import evaluate_heartbeat_state, get_backend_health, start_runtime_heartbeat
from atlas_ai.voice.wake_loop import _sleep_phrase, _wake_phrase


class _SafeAdapter:
    def submit_order(self, *, idempotency_key: str, request: dict[str, object]) -> object:
        class _R:
            ok = True
            status = "acknowledged"
            broker_order_id = f"BRK_{idempotency_key}"
            message = "ok"
            payload = {}
            fill_qty = 0.0
            avg_fill_price = 0.0

        return _R()

    def cancel_order(self, *, broker_order_id: str) -> object:
        return type(
            "CancelResponse",
            (),
            {
                "ok": True,
                "status": "cancelled",
                "message": "ok",
                "broker_order_id": broker_order_id,
            },
        )()

    def query_order_status(self, *, broker_order_id: str) -> object:
        return type(
            "StatusResponse",
            (),
            {
                "ok": True,
                "status": "acknowledged",
                "message": "ok",
                "broker_order_id": broker_order_id,
                "fill_qty": 0.0,
                "avg_fill_price": 0.0,
            },
        )()

    def list_open_orders(self) -> list[dict[str, object]]:
        return []


class _MismatchAdapter(_SafeAdapter):
    def list_open_orders(self) -> list[dict[str, object]]:
        return [
            {
                "broker_order_id": "BRK_ORPHAN_1",
                "instrument": "XAUUSD",
                "side": "buy",
                "size": 0.01,
                "status": "acknowledged",
            }
        ]


class _FailingScanAdapter(_SafeAdapter):
    def list_open_orders(self) -> list[dict[str, object]]:
        raise RuntimeError("mt5_positions_get_failed")


class _EmptyScanAdapter(_SafeAdapter):
    def list_open_orders(self) -> list[dict[str, object]]:
        return []


def _live_runtime() -> dict[str, object]:
    return {
        "execute_trades": True,
        "freshness_runtime": {"status_age_ms": 50, "is_stale": False},
        "day_trade_count": 0,
        "daily_loss_pct": 0.0,
        "consecutive_losses": 0,
    }


def test_live_like_execution_blocked_for_inmemory_adapter() -> None:
    gate = ExecutionGate()
    gate.set_broker_adapter(InMemoryBrokerAdapter())
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="live",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01 data_source=broker_runtime environment_mode=demo",
        requested_by="operator",
        dry_run=False,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags=_live_runtime(), risk_policy={"daily_loss_cap_pct": 2.0})
    assert out["ok"] is False
    assert out["reason"] == "in_memory_adapter_blocked_for_live"


def test_preflight_blocks_when_safety_lock_active_even_for_dry_run() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    gate.approve_request(req["id"], approved_by="operator")
    out = gate.execute(
        req["id"],
        runtime_flags={"execute_trades": False, "safety_lock_active": True, "freshness_runtime": {"status_age_ms": 50, "is_stale": False}},
        risk_policy={"daily_loss_cap_pct": 2.0},
    )
    assert out["ok"] is False
    assert out["reason"] == "safety_lock_active"


def test_mandatory_reconciliation_blocks_execution_on_mismatch() -> None:
    gate = ExecutionGate()
    gate.set_broker_adapter(_MismatchAdapter())
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="live",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01 data_source=broker_runtime environment_mode=demo",
        requested_by="operator",
        dry_run=False,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags=_live_runtime(), risk_policy={"daily_loss_cap_pct": 2.0})
    assert out["ok"] is False
    assert out["reason"] == "reconciliation_required"
    snap = gate.snapshot()
    assert snap["mismatch_detected"] is True
    assert snap["reconciliation_status"] == "mismatch_detected"
    assert float(snap["last_reconciled_at"]) > 0


def test_mandatory_reconciliation_blocks_execution_when_broker_scan_fails() -> None:
    gate = ExecutionGate()
    gate.set_broker_adapter(_FailingScanAdapter())
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="live",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01 data_source=broker_runtime environment_mode=demo",
        requested_by="operator",
        dry_run=False,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags=_live_runtime(), risk_policy={"daily_loss_cap_pct": 2.0})
    assert out["ok"] is False
    assert out["reason"] == "reconciliation_required"
    snap = gate.snapshot()
    assert snap["mismatch_detected"] is True
    assert snap["reconciliation_status"] == "mismatch_detected"


def test_broker_scan_clears_stale_orphan_orders_when_positions_are_closed() -> None:
    gate = ExecutionGate()
    gate.set_broker_adapter(_EmptyScanAdapter())
    gate._orphan_orders["BRK_OLD"] = {  # type: ignore[attr-defined]
        "broker_order_id": "BRK_OLD",
        "instrument": "XAUUSD",
        "broker_status": "open",
        "reason": "unknown_to_atlas",
        "reviewed": False,
    }

    out = gate.scan_broker_open_orders()

    assert out["ok"] is True
    assert out["stale_orphans_cleared"] == 1
    assert gate.list_orphan_orders(include_reviewed=False, limit=10) == []


def test_simulation_data_source_blocked_for_live_like_execution() -> None:
    gate = ExecutionGate()
    gate.set_broker_adapter(_SafeAdapter())
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="live",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01 data_source=simulation environment_mode=demo",
        requested_by="operator",
        dry_run=False,
    )
    gate.approve_request(req["id"], approved_by="operator", explicit_live_confirmation=True)
    out = gate.execute(req["id"], runtime_flags=_live_runtime(), risk_policy={"daily_loss_cap_pct": 2.0})
    assert out["ok"] is False
    assert out["reason"] == "simulation_data_source_blocked_for_live_like_execution"


def test_voice_phrase_requires_confidence_and_boundaries() -> None:
    assert _wake_phrase("atlas wake up now", confidence=0.93)
    assert not _wake_phrase("atlas wake up now", confidence=0.4)
    assert _sleep_phrase("atlas sleep", confidence=0.93)
    assert not _sleep_phrase("atlas sleep", confidence=0.3)
    assert not _wake_phrase("metatlas wakeful", confidence=0.99)
    assert not _sleep_phrase("sleepatlas", confidence=0.99)


def test_runtime_heartbeat_updates_and_stale_detection(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    stop = start_runtime_heartbeat(store, component="test_component", interval_seconds=0.15, stale_after_seconds=0.45)
    try:
        time.sleep(0.35)
        flags = store.load_runtime_flags()
        row = dict(dict(flags.get("runtime_heartbeat", {}) or {}).get("test_component", {}) or {})
        state = evaluate_heartbeat_state(
            flags,
            component="test_component",
            now_ts=float(row.get("last_heartbeat_ts", 0.0) or 0.0) + 0.1,
            stale_after_seconds=0.45,
        )
        assert state["cycle_status"] == "running"
        assert state["is_stale"] is False
    finally:
        stop.set()
    time.sleep(0.05)
    stale = evaluate_heartbeat_state({"runtime_heartbeat": {"test_component": {"last_heartbeat_ts": 0.0}}}, component="test_component", stale_after_seconds=0.45)
    assert stale["is_stale"] is True


def test_backend_health_reports_unknown_without_heartbeat_file(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))

    health = get_backend_health(store, threshold_seconds=10)

    assert health == {
        "is_fresh": False,
        "last_heartbeat": "",
        "seconds_since_heartbeat": 0,
        "staleness_threshold_seconds": 10,
        "cycle_count": 0,
        "status": "unknown",
    }


def test_backend_health_reports_live_and_stale_from_heartbeat_file(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    heartbeat = store.write_backend_heartbeat(mode="paper", pid=1234)

    live = get_backend_health(
        store,
        threshold_seconds=10,
        now_ts=float(heartbeat["last_heartbeat_ts"]) + 3,
    )
    stale = get_backend_health(
        store,
        threshold_seconds=10,
        now_ts=float(heartbeat["last_heartbeat_ts"]) + 11,
    )

    assert live["is_fresh"] is True
    assert live["status"] == "live"
    assert live["last_heartbeat"] == heartbeat["last_heartbeat"]
    assert live["seconds_since_heartbeat"] == 3
    assert live["staleness_threshold_seconds"] == 10
    assert live["cycle_count"] == 1

    assert stale["is_fresh"] is False
    assert stale["status"] == "stale"
    assert stale["seconds_since_heartbeat"] == 11
    assert stale["cycle_count"] == 1
