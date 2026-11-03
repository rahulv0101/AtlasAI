from __future__ import annotations

from pathlib import Path

from atlas_ai.execution.automation_state_store import AutomationStateStore
from atlas_ai.execution.execution_gate import ExecutionGate


def _runtime_flags(*, stale: bool = False) -> dict:
    return {
        "execute_trades": True,
        "freshness_runtime": {"status_age_ms": (240000 if stale else 50), "is_stale": bool(stale)},
        "max_freshness_age_ms": 120000,
    }


def _prepare_demo_request(gate: ExecutionGate) -> str:
    gate.set_gate_enabled(True)
    gate.set_execution_environment("demo")
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=True,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    rid = str(req.get("id", ""))
    gate.approve_request(rid, approved_by="trader_engine", explicit_live_confirmation=True)
    return rid


def test_ui_kill_switch_blocks_stale_trader_gate_before_execution(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    req_id = _prepare_demo_request(trader_gate)
    assert trader_gate.snapshot().get("kill_switch_active") is False
    ui_gate.activate_kill_switch("operator_requested")
    # stale in-memory state before execute
    assert trader_gate.snapshot().get("kill_switch_active") is False
    out = trader_gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is False
    assert out.get("reason") == "kill_switch_active"


def test_persisted_kill_switch_overrides_stale_in_memory_state(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    req_id = _prepare_demo_request(trader_gate)
    ui_gate.activate_kill_switch("operator_requested")
    out = trader_gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is False
    assert out.get("reason") == "kill_switch_active"
    # trader gate should now reflect shared kill-switch state after refresh
    assert trader_gate.snapshot().get("kill_switch_active") is True


def test_shared_state_reload_failure_is_fail_closed(tmp_path) -> None:
    state_file = Path(tmp_path) / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_file))
    req_id = _prepare_demo_request(gate)
    state_file.write_text("{invalid_json", encoding="utf-8")
    out = gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is False
    assert str(out.get("reason", "")).startswith("shared_state_sync_failed:")


def test_normal_guarded_execution_still_works_when_shared_state_is_healthy(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    ui_gate.set_gate_enabled(True)
    ui_gate.set_execution_environment("demo")
    ui_gate.set_rollout_stage("limited_live_pilot", approved=True)
    req_id = _prepare_demo_request(trader_gate)
    out = trader_gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is True
    assert bool(out.get("dry_run", False)) is True


def test_sleep_mode_persisted_state_also_blocks_after_refresh(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    req_id = _prepare_demo_request(trader_gate)
    ui_gate.activate_sleep_mode(reason="operator_requested")
    out = trader_gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is False
    assert out.get("reason") == "sleep_mode_active_execution_blocked"


def test_create_request_cannot_overwrite_newer_persisted_kill_switch(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    ui_gate.activate_kill_switch("operator_requested")
    req = trader_gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=True,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    assert str(req.get("id", "")) != ""
    payload, info = AutomationStateStore(state_path).load()
    assert info.get("ok") is True
    ks = dict(payload.get("kill_switch", {}) or {})
    assert bool(ks.get("active", False)) is True


def test_live_like_create_request_fails_closed_when_shared_state_load_fails(tmp_path) -> None:
    state_file = Path(tmp_path) / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_file))
    gate.set_execution_environment("demo")
    state_file.write_text("{invalid_json", encoding="utf-8")
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=True,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    assert req.get("id", "") == ""
    assert str(req.get("last_block_reason", "")).startswith("shared_state_sync_failed:")


def test_create_request_then_execute_cannot_nullify_newer_kill_switch_state(tmp_path) -> None:
    state_path = str(Path(tmp_path) / "automation_state.json")
    ui_gate = ExecutionGate(state_store_path=state_path)
    trader_gate = ExecutionGate(state_store_path=state_path)
    ui_gate.set_gate_enabled(True)
    ui_gate.set_execution_environment("demo")
    ui_gate.set_rollout_stage("limited_live_pilot", approved=True)
    ui_gate.activate_kill_switch("operator_requested")
    req = trader_gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=True,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    req_id = str(req.get("id", ""))
    assert req_id
    trader_gate.approve_request(req_id, approved_by="trader_engine", explicit_live_confirmation=True)
    out = trader_gate.execute(req_id, runtime_flags=_runtime_flags(), risk_policy={"daily_loss_cap_pct": 5.0})
    assert out.get("ok") is False
    assert out.get("reason") == "kill_switch_active"
