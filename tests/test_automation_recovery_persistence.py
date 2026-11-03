from __future__ import annotations

import json
from pathlib import Path

from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.automation_state_store import AutomationStateStore
from atlas_ai.execution.broker_adapter import InMemoryBrokerAdapter
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {"execute_trades": True, "read_only_ui": False}

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.64},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold under weak evidence"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _create_policy(gate: ExecutionGate, *, policy_id: str = "ap1", environment: str = "paper", stage: str = "approved_paper_execution") -> None:
    gate.create_automation_policy(
        policy_id=policy_id,
        environment=environment,
        allowed_setup_families=["continuation"],
        allowed_instruments=["XAUUSD"],
        allowed_sessions=["london"],
        size_cap=0.02,
        daily_loss_cap=1.0,
        max_trades_per_session=2,
        active_rollout_stage=stage,
    )
    gate.approve_automation_policy(policy_id)
    gate.activate_automation_policy(policy_id)


def test_approved_policy_persists_and_reloads(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    _create_policy(gate1, environment="paper", stage="approved_paper_execution")
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_active_policy_id"] == "ap1"
    assert snap["restored_policy_id"] == "ap1"
    assert snap["recovery_mode"] in {"restored_but_paused", "restored_non_live_only"}


def test_live_automation_does_not_auto_resume_after_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    gate1.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate1, environment="live", stage="limited_live_pilot")
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_paused"] is True
    assert snap["recovery_requires_explicit_resume"] is True
    assert "live_auto_resume_blocked_after_restart" in snap["recovery_reason"]


def test_explicit_resume_persists_live_automation_after_clean_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    gate1.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate1, environment="live", stage="limited_live_pilot")
    assert gate1.resume_automation() is True
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_paused"] is False
    assert snap["automation_enabled"] is True
    assert snap["automation_explicit_resume_active"] is True
    assert snap["automation_explicit_resume_policy_id"] == "ap1"
    assert snap["recovery_requires_explicit_resume"] is False
    assert snap["recovery_reason"] == "explicit_operator_resume_persisted"


def test_explicit_resume_survives_process_restart_without_clean_marker(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    gate1.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate1, environment="live", stage="limited_live_pilot")
    assert gate1.resume_automation() is True
    # Intentionally no clean-shutdown mark.

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_paused"] is False
    assert snap["automation_explicit_resume_active"] is True
    assert snap["recovery_requires_explicit_resume"] is False
    assert snap["recovery_reason"] == "explicit_operator_resume_persisted_after_process_restart"


def test_sticky_resume_survives_unclean_restart_without_reasserting_pause(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(json.dumps({"automation_resume_sticky": True}), encoding="utf-8")

    gate1 = ExecutionGate(state_store_path=str(state_path))
    gate1.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate1, policy_id="p1", environment="live", stage="limited_live_pilot")
    gate1.configure_tiny_live_pilot(
        allowed_instrument="XAUUSD",
        allowed_setup_family="continuation",
        size_cap=0.01,
        max_trades_per_day=2,
        daily_loss_cap=0.6,
        max_consecutive_losses=1,
    )
    gate1.approve_tiny_live_pilot(note="test")
    gate1.activate_tiny_live_pilot()
    gate1._automation_explicit_resume_active = False  # type: ignore[attr-defined]
    gate1._automation_explicit_resume_policy_id = ""  # type: ignore[attr-defined]
    gate1._last_shutdown_clean = False  # type: ignore[attr-defined]
    gate1._tiny_live_pilot.paused = True  # type: ignore[attr-defined]
    gate1._tiny_live_pilot.pause_reason = "weekend_learning_only_operator_requested"  # type: ignore[attr-defined]
    gate1._persist_state(event="simulated_unclean_restart_state")  # type: ignore[attr-defined]

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_paused"] is False
    assert snap["automation_explicit_resume_active"] is True
    assert snap["automation_explicit_resume_policy_id"] == "p1"
    assert snap["tiny_live_pilot"]["paused"] is False
    assert snap["tiny_live_pilot"]["pause_reason"] == ""
    assert snap["recovery_reason"] in {
        "sticky_operator_resume_persisted_after_process_restart",
        "explicit_operator_resume_persisted_after_process_restart",
    }


def test_sticky_resume_blocks_in_memory_gate_from_reasserting_unclean_pause(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(json.dumps({"automation_resume_sticky": True}), encoding="utf-8")

    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, policy_id="p1", environment="live", stage="limited_live_pilot")
    gate.configure_tiny_live_pilot(
        allowed_instrument="XAUUSD",
        allowed_setup_family="continuation",
        size_cap=0.01,
        max_trades_per_day=2,
        daily_loss_cap=0.6,
        max_consecutive_losses=1,
    )
    gate.approve_tiny_live_pilot(note="test")
    gate.activate_tiny_live_pilot()
    gate._automation_explicit_resume_active = False  # type: ignore[attr-defined]
    gate._automation_explicit_resume_policy_id = ""  # type: ignore[attr-defined]
    gate._automation_paused = True  # type: ignore[attr-defined]
    gate._automation_pause_reason = "unclean_shutdown_forced_pause"  # type: ignore[attr-defined]
    gate._tiny_live_pilot.paused = True  # type: ignore[attr-defined]
    gate._tiny_live_pilot.pause_reason = "weekend_learning_only_operator_requested"  # type: ignore[attr-defined]

    gate._persist_state(event="mandatory_reconciliation_completed")  # type: ignore[attr-defined]

    payload, meta = AutomationStateStore(state_path).load()
    assert meta["ok"] is True
    assert payload["adapter_type"] == "TestBrokerAdapter"
    assert payload["automation_paused"] is False
    assert payload["automation_pause_reason"] == ""
    assert payload["automation_explicit_resume_active"] is True
    assert payload["automation_explicit_resume_policy_id"] == "p1"
    assert payload["tiny_live_pilot"]["paused"] is False
    assert payload["tiny_live_pilot"]["pause_reason"] == ""


def test_in_memory_gate_does_not_run_mandatory_reconciliation_against_shared_state(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, policy_id="p1", environment="live", stage="limited_live_pilot")
    gate.set_broker_adapter(InMemoryBrokerAdapter())
    gate._automation_paused = True  # type: ignore[attr-defined]
    gate._automation_pause_reason = "unclean_shutdown_forced_pause"  # type: ignore[attr-defined]

    result = gate._run_mandatory_reconciliation(limit=1)  # type: ignore[attr-defined]

    assert result["status"] == "skipped"
    payload, _ = AutomationStateStore(state_path).load()
    assert payload["automation_paused"] is False
    assert payload["automation_pause_reason"] == ""


def test_in_memory_gate_never_overwrites_authoritative_shared_state(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, policy_id="p1", environment="live", stage="limited_live_pilot")
    assert gate.resume_automation() is True
    payload_before, meta_before = AutomationStateStore(state_path).load()

    gate.set_broker_adapter(InMemoryBrokerAdapter())
    gate._automation_paused = True  # type: ignore[attr-defined]
    gate._automation_pause_reason = "unclean_shutdown_forced_pause"  # type: ignore[attr-defined]
    gate._tiny_live_pilot.paused = True  # type: ignore[attr-defined]
    gate._tiny_live_pilot.pause_reason = "weekend_learning_only_operator_requested"  # type: ignore[attr-defined]
    gate._persist_state(event="broker_adapter_set")  # type: ignore[attr-defined]
    gate._persist_state(event="automation_paused")  # type: ignore[attr-defined]
    gate._persist_state(event="mandatory_reconciliation_completed")  # type: ignore[attr-defined]

    payload_after, meta_after = AutomationStateStore(state_path).load()
    assert meta_after["revision"] == meta_before["revision"]
    assert payload_after == payload_before


def test_in_memory_gate_does_not_persist_orphan_scan_to_shared_state(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, policy_id="p1", environment="live", stage="limited_live_pilot")
    gate.set_broker_adapter(InMemoryBrokerAdapter())
    payload_before, _ = AutomationStateStore(state_path).load()
    revision_before = AutomationStateStore(state_path).load()[1]["revision"]

    result = gate.scan_broker_open_orders()

    payload_after, meta_after = AutomationStateStore(state_path).load()
    assert result["status"] == "skipped"
    assert meta_after["revision"] == revision_before
    assert payload_after == payload_before


def test_state_store_blocks_stale_unclean_pause_when_sticky_resume_enabled(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(json.dumps({"automation_resume_sticky": True}), encoding="utf-8")
    store = AutomationStateStore(state_path)

    result = store.save(
        {
            "active_automation_policy_id": "p1",
            "automation_paused": True,
            "automation_pause_reason": "unclean_shutdown_forced_pause",
            "automation_explicit_resume_active": False,
            "automation_explicit_resume_policy_id": "",
            "kill_switch": {"active": False},
            "sleep_mode_active": False,
            "tiny_live_pilot": {
                "enabled": True,
                "approved": True,
                "paused": True,
                "pause_reason": "weekend_learning_only_operator_requested",
            },
        },
        event="mandatory_reconciliation_completed",
    )

    assert result["ok"] is True
    payload, meta = store.load()
    assert meta["ok"] is True
    assert payload["automation_paused"] is False
    assert payload["automation_pause_reason"] == ""
    assert payload["automation_explicit_resume_active"] is True
    assert payload["automation_explicit_resume_policy_id"] == "p1"
    assert payload["tiny_live_pilot"]["paused"] is False
    assert payload["tiny_live_pilot"]["pause_reason"] == ""


def test_state_store_blocks_stale_pilot_pause_when_explicit_resume_is_active(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    store = AutomationStateStore(state_path)

    result = store.save(
        {
            "active_automation_policy_id": "p1",
            "automation_paused": False,
            "automation_pause_reason": "",
            "automation_explicit_resume_active": True,
            "automation_explicit_resume_policy_id": "p1",
            "kill_switch": {"active": False},
            "sleep_mode_active": False,
            "tiny_live_pilot": {
                "enabled": True,
                "approved": True,
                "paused": True,
                "pause_reason": "weekend_learning_only_operator_requested",
            },
        },
        event="mandatory_reconciliation_completed",
    )

    assert result["ok"] is True
    payload, meta = store.load()
    assert meta["ok"] is True
    assert payload["automation_paused"] is False
    assert payload["automation_explicit_resume_active"] is True
    assert payload["tiny_live_pilot"]["paused"] is False
    assert payload["tiny_live_pilot"]["pause_reason"] == ""


def test_state_store_sticky_resume_does_not_override_kill_switch(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    runtime_path = tmp_path / "runtime_flags.json"
    runtime_path.write_text(json.dumps({"automation_resume_sticky": True}), encoding="utf-8")
    store = AutomationStateStore(state_path)

    store.save(
        {
            "active_automation_policy_id": "p1",
            "automation_paused": True,
            "automation_pause_reason": "unclean_shutdown_forced_pause",
            "automation_explicit_resume_active": False,
            "kill_switch": {"active": True},
            "sleep_mode_active": False,
        },
        event="mandatory_reconciliation_completed",
    )

    payload, _ = store.load()
    assert payload["automation_paused"] is True
    assert payload["automation_pause_reason"] == "unclean_shutdown_forced_pause"
    assert payload["automation_explicit_resume_active"] is False


def test_explicit_resume_keeps_soft_blockers_from_turning_automation_off(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, environment="live", stage="limited_live_pilot")
    assert gate.resume_automation() is True

    gate._maybe_pause_automation(reason="regime_noise_too_high", runtime_flags={"regime_noise": 0.8})

    snap = gate.snapshot()
    assert snap["automation_paused"] is False
    assert snap["automation_explicit_resume_active"] is True


def test_execution_gate_clears_stale_pilot_pause_during_explicit_resume(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, environment="live", stage="limited_live_pilot")
    gate.configure_tiny_live_pilot(
        allowed_instrument="XAUUSD",
        allowed_setup_family="continuation",
        size_cap=0.01,
        max_trades_per_day=2,
        daily_loss_cap=0.6,
        max_consecutive_losses=1,
    )
    gate.approve_tiny_live_pilot(note="test")
    gate.activate_tiny_live_pilot()
    assert gate.resume_automation() is True

    gate._tiny_live_pilot.paused = True  # type: ignore[attr-defined]
    gate._tiny_live_pilot.pause_reason = "weekend_learning_only_operator_requested"  # type: ignore[attr-defined]
    snap = gate.snapshot()

    assert snap["automation_paused"] is False
    assert snap["automation_explicit_resume_active"] is True
    assert snap["tiny_live_pilot"]["paused"] is False
    assert snap["tiny_live_pilot"]["pause_reason"] == ""


def test_explicit_resume_does_not_override_hard_pause(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    _create_policy(gate, environment="live", stage="limited_live_pilot")
    assert gate.resume_automation() is True

    gate._maybe_pause_automation(reason="daily_loss_cap_hit", runtime_flags={"daily_loss_used": 1.0})

    snap = gate.snapshot()
    assert snap["automation_paused"] is True
    assert snap["automation_pause_reason"] == "daily_loss_cap_hit"
    assert snap["automation_explicit_resume_active"] is False


def test_paper_automation_restores_conservatively(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    _create_policy(gate1, environment="paper", stage="approved_paper_execution")
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["automation_paused"] is True
    assert snap["recovery_requires_explicit_resume"] is True
    assert snap["recovery_reason"] == "paper_auto_resume_requires_review"


def test_kill_switch_persists_across_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    gate1.activate_kill_switch("test_kill")
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["kill_switch_active"] is True
    assert snap["kill_switch_reason"] == "test_kill"


def test_corrupt_state_file_falls_back_safely(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    state_path.write_text("{not-json", encoding="utf-8")

    gate = ExecutionGate(state_store_path=str(state_path))
    snap = gate.snapshot()
    assert snap["recovery_mode"] == "restored_pending_review"
    assert snap["automation_paused"] is True
    assert snap["recovery_requires_explicit_resume"] is True


def test_schema_mismatch_falls_back_safely(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    state_path.write_text(
        json.dumps({"schema_version": 999, "payload": {"gate_enabled": True}}),
        encoding="utf-8",
    )
    gate = ExecutionGate(state_store_path=str(state_path))
    snap = gate.snapshot()
    assert snap["recovery_mode"] == "restored_pending_review"
    assert snap["automation_paused"] is True


def test_unclean_shutdown_forces_paused_recovery(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    _create_policy(gate1, environment="paper", stage="approved_paper_execution")
    # Intentionally no clean-shutdown mark.

    gate2 = ExecutionGate(state_store_path=str(state_path))
    snap = gate2.snapshot()
    assert snap["last_shutdown_clean"] is False
    assert snap["automation_paused"] is True
    assert snap["recovery_requires_explicit_resume"] is True
    assert snap["recovery_reason"] in {
        "unclean_shutdown_forced_pause",
        "paper_auto_resume_requires_review",
    }


def test_ask_atlas_reports_recovery_status(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate1 = ExecutionGate(state_store_path=str(state_path))
    _create_policy(gate1, environment="live", stage="limited_live_pilot")
    gate1.mark_clean_shutdown()

    service = AskAtlasService(_Source())
    service.execution = ExecutionGate(state_store_path=str(state_path))

    out = service.answer("What recovery mode are you in?")
    assert out["intent"] == "automation_recovery_status"
    assert "recovery mode:" in out["result"].lower()
    assert "execution_state" in out


def test_runtime_snapshot_includes_recovery_safe_fields(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    snap = gate.snapshot()
    for field in (
        "recovery_mode",
        "recovery_reason",
        "recovery_requires_explicit_resume",
        "restored_policy_id",
        "last_shutdown_clean",
    ):
        assert field in snap


def test_recovery_load_has_no_execution_side_effects(tmp_path: Path) -> None:
    state_path = tmp_path / "automation_state.json"
    source = _Source()
    gate1 = ExecutionGate(state_store_path=str(state_path))
    _create_policy(gate1, environment="paper", stage="approved_paper_execution")
    gate1.mark_clean_shutdown()

    service = AskAtlasService(source)
    service.execution = ExecutionGate(state_store_path=str(state_path))
    service.answer("Was automation restored after restart?")
    assert source.execute_called is False
