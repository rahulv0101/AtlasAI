from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.54},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold under weak evidence"}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False, "read_only_ui": False}

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def test_execution_request_requires_approval() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="XAUUSD buy 0.01",
        requested_by="operator",
        dry_run=True,
    )
    result = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert result["ok"] is False
    assert result["reason"] == "approval_missing"


def test_trader_has_no_direct_trade_executor_send_path() -> None:
    trader_py = Path("atlas_ai/trader.py").read_text(encoding="utf-8")
    assert "self.executor.execute(" not in trader_py
    assert "close_stale_positions" in trader_py
    assert "def _housekeeping_close_allowed" in trader_py
    assert "refresh_critical_controls_from_persisted_state()" in trader_py
    assert "close_stale_positions(allow_close=True)" in trader_py
    assert "Skipped overnight housekeeping close due to guarded control state." in trader_py
    assert "guarded_boundary" in trader_py


def test_trader_guarded_requests_are_not_marked_automation_origin() -> None:
    trader_py = Path("atlas_ai/trader.py").read_text(encoding="utf-8")
    assert 'request_origin="trader_guarded"' in trader_py
    assert "auto_generated=False" in trader_py


def test_trader_reports_missing_approval_as_blocked_not_broker_failure() -> None:
    trader_py = Path("atlas_ai/trader.py").read_text(encoding="utf-8")
    assert 'gate_block_reasons = {"approval_missing", "live_requires_explicit_confirmation"}' in trader_py
    assert 'gate_out["reason"] = "broker_confirmation_missing"' in trader_py
    assert '"broker_order_id": str(result.broker_order_id)' in trader_py
    assert "Execution blocked for" in trader_py


def test_trader_maps_strategy_to_setup_family_for_gate_payload() -> None:
    trader_py = Path("atlas_ai/trader.py").read_text(encoding="utf-8")
    assert "def _strategy_to_setup_family" in trader_py
    assert "setup_family=" in trader_py
    assert 'gate_env = "demo" if bool(execute_trades) else "paper"' in trader_py


def test_strict_runtime_checks_apply_without_auto_generated_flag() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="trader",
        dry_run=False,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    gate.approve_request(req["id"], approved_by="trader", note="ok")
    out = gate.execute(
        req["id"],
        runtime_flags={"freshness_runtime": {"status_age_ms": 120000, "is_stale": True}, "execute_trades": False},
        risk_policy={"daily_loss_cap_pct": 5.0},
        execute_fn=lambda _req: {"ok": True, "note": "should_not_run"},
    )
    assert out["ok"] is False
    assert out["reason"] == "freshness_stale"


def test_runtime_freshness_accepts_string_false_without_marking_stale() -> None:
    gate = ExecutionGate()
    out = gate._check_runtime_freshness(
        {"freshness_runtime": {"status_age_ms": "50", "is_stale": "false"}},
        strict=True,
    )
    assert out["ok"] is True


def test_gate_block_reasons_include_structured_code_and_last_cycle() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    snap = gate.snapshot()
    assert out["reason"] == "approval_missing"
    assert out["code"] == "approval_missing"
    assert out["reason_code"] == "APPROVAL_MISSING"
    assert out["structured_result"]["allowed"] is False
    assert out["structured_result"]["reason_code"] == "APPROVAL_MISSING"
    assert out["structured_result"]["checked_at"]
    assert snap["gate_decisions_last_cycle"]["blocked_by"] == "approval"
    assert snap["gate_decisions_last_cycle"]["code"] == "approval_missing"
    assert snap["gate_decisions_last_cycle"]["reason_code"] == "APPROVAL_MISSING"
    last = gate.get_last_rejection_reason()
    assert last["reason_code"] == "APPROVAL_MISSING"
    assert last["legacy_reason"] == "approval_missing"


def test_execution_gate_writes_structured_decision_journal_for_block_and_approval() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    blocked_req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    blocked = gate.execute(blocked_req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert blocked["reason_code"] == "APPROVAL_MISSING"

    approved_req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )
    gate.approve_request(approved_req["id"], approved_by="operator", note="ok")
    approved = gate.execute(approved_req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert approved["ok"] is True
    assert approved["structured_result"]["allowed"] is True
    assert "passed_checks" in approved["structured_result"]

    rows = [json.loads(line) for line in gate._decision_journal_path.read_text(encoding="utf-8").splitlines()]  # type: ignore[attr-defined]
    traces = [dict(row.get("trace", {}) or {}) for row in rows]
    assert any(trace.get("allowed") is False and trace.get("reason_code") == "APPROVAL_MISSING" for trace in traces)
    assert any(trace.get("allowed") is True and trace.get("reason_code") is None for trace in traces)


def test_last_rejection_reason_survives_gate_reload(tmp_path) -> None:
    state_path = tmp_path / "automation_state.json"
    gate = ExecutionGate(state_store_path=str(state_path))
    gate.set_gate_enabled(True)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
    )

    blocked = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})

    assert blocked["reason_code"] == "APPROVAL_MISSING"
    restored = ExecutionGate(state_store_path=str(state_path))
    last = restored.get_last_rejection_reason()
    assert last["allowed"] is False
    assert last["reason_code"] == "APPROVAL_MISSING"
    assert last["legacy_reason"] == "approval_missing"
    assert last["checked_at"]


def test_trim_preserves_request_dictionary_order() -> None:
    gate = ExecutionGate()
    for i in range(805):
        request_id = f"req_{i}"
        gate._order.append(request_id)
        gate._requests[request_id] = {"id": request_id}
    gate._trim()
    assert list(gate._requests.keys()) == gate._order
    assert gate._order[0] == "req_5"


def test_snapshot_exposes_environment_model_and_autonomy_semantics() -> None:
    gate = ExecutionGate()
    gate.set_execution_environment("demo")
    snap = gate.snapshot()
    assert snap["environment_model"]["canonical"] == "demo"
    assert snap["environment_model"]["label"] == "Live Demo"
    assert snap["autonomy_mode_semantics"]["guarded_execution"]["requires_existing_gates"] is True


def test_non_dry_execution_without_executor_does_not_report_success() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate)
    req = gate.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="trader",
        dry_run=False,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=False,
    )
    gate.approve_request(req["id"], approved_by="trader", note="ok")
    out = gate.execute(
        req["id"],
        runtime_flags={"execute_trades": False},
        risk_policy={"daily_loss_cap_pct": 5.0},
    )
    assert out["ok"] is False
    assert out["reason"] == "executor_missing"


def test_missing_approval_blocks_execution() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    created = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")
    req_id = created["execution_request"]["id"]
    blocked = svc.answer(f"run execution {req_id}")
    assert "blocked" in blocked["result"].lower()
    assert "approval_missing" in blocked["result"].lower()


def test_kill_switch_blocks_execution() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    created = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")
    req_id = created["execution_request"]["id"]
    svc.answer(f"approve execution {req_id}")
    svc.answer("activate kill switch")
    blocked = svc.answer(f"run execution {req_id}")
    assert "kill_switch_active" in blocked["result"]


def test_dry_run_logs_but_does_not_execute() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    created = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")
    req_id = created["execution_request"]["id"]
    svc.answer(f"approve execution {req_id}")
    result = svc.answer(f"run execution {req_id}")
    assert "dry-run completed" in result["result"].lower()
    assert source.execute_called is False


def test_paper_live_separation_enforced() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    created = svc.answer("create execution request live place_trade real XAUUSD buy 0.01")
    req_id = created["execution_request"]["id"]
    # Live approval needs explicit live confirmation.
    not_ready = svc.answer(f"approve execution {req_id}")
    assert "still pending" in not_ready["result"].lower()
    svc.answer(f"approve execution {req_id} live-confirm")
    blocked = svc.answer(f"run execution {req_id}")
    assert (
        ("runtime_not_live_enabled" in blocked["result"])
        or ("live_blocked_in_stage:paper_only" in blocked["result"])
        or ("in_memory_broker_adapter_blocked_for_live_like_execution" in blocked["result"])
    )


def test_invalid_request_blocked_safely() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    created = svc.answer("create execution request paper unknown_action dry bad payload")
    req_id = created["execution_request"]["id"]
    svc.answer(f"approve execution {req_id}")
    blocked = svc.answer(f"run execution {req_id}")
    assert "unsupported_action_type" in blocked["result"]


def test_audit_log_entry_created() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    created = svc.answer("create execution request paper place_trade dry XAUUSD buy 0.01")
    req_id = created["execution_request"]["id"]
    svc.answer(f"approve execution {req_id}")
    svc.answer(f"run execution {req_id}")
    audit = svc.answer("show execution audit")
    assert "last execution audit event" in audit["result"].lower()
    assert isinstance(audit.get("execution_audit"), list)
    assert len(audit["execution_audit"]) >= 1


def test_no_execution_from_recommendation_or_voice_query_alone() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("What should you do next?")
    svc.answer("Is execution enabled?")
    svc.answer("atlas wake")
    assert source.execute_called is False
    assert svc.execution.last_request() is None


def test_status_queries_do_not_mutate_control_or_execution_state() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    before_requests = svc.execution.pending_requests(limit=50)
    before_control = svc.control.to_debug()
    svc.answer("is execution enabled?")
    svc.answer("is the kill switch active?")
    after_requests = svc.execution.pending_requests(limit=50)
    after_control = svc.control.to_debug()
    assert before_requests == after_requests
    assert before_control == after_control


def test_explicit_auto_execution_command_is_required_for_action_path() -> None:
    source = _Source()
    source.load_runtime_flags = lambda: {  # type: ignore[method-assign]
        "execute_trades": True,
        "read_only_ui": False,
        "freshness_runtime": {"status_age_ms": 50, "is_stale": False},
    }
    svc = AskAtlasService(source)
    svc.answer("set execution environment demo")
    svc.answer("set autonomy mode guarded_execution")
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    svc.answer("create automation policy p1 demo continuation XAUUSD test 0.01 0.6 2 limited_live_pilot")
    svc.answer("approve automation policy p1")
    svc.answer("activate automation policy p1")
    svc.answer("activate operator present mode")
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "automation_result" in out


def test_live_blocked_in_paper_stages() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    # default stage is paper_only
    req_id = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    out = svc.answer(f"run execution {req_id}")
    assert ("live_blocked_in_stage:paper_only" in out["result"]) or ("in_memory_broker_adapter_blocked_for_live_like_execution" in out["result"])


def test_live_requires_explicit_approval_and_live_confirmation() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    req_id = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    pending = svc.answer(f"approve execution {req_id}")
    assert "still pending" in pending["result"].lower()


def test_pilot_restrictions_enforced() -> None:
    source = _Source()
    source.load_runtime_flags = lambda: {"execute_trades": True, "read_only_ui": False}  # type: ignore[method-assign]
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")

    req_bad_inst = svc.answer("create execution request live place_trade real instrument=EURUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_bad_inst} live-confirm")
    blocked1 = svc.answer(f"run execution {req_bad_inst}")
    assert ("instrument_restricted_in_stage" in blocked1["result"]) or ("in_memory_broker_adapter_blocked_for_live_like_execution" in blocked1["result"])

    req_bad_setup = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=breakout size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_bad_setup} live-confirm")
    blocked2 = svc.answer(f"run execution {req_bad_setup}")
    assert ("setup_family_restricted_in_stage" in blocked2["result"]) or ("in_memory_broker_adapter_blocked_for_live_like_execution" in blocked2["result"])

    req_bad_size = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.2")["execution_request"]["id"]
    svc.answer(f"approve execution {req_bad_size} live-confirm")
    blocked3 = svc.answer(f"run execution {req_bad_size}")
    assert ("size_cap_exceeded_in_stage" in blocked3["result"]) or ("in_memory_broker_adapter_blocked_for_live_like_execution" in blocked3["result"])


def test_regression_conditions_reduce_rollout_stage() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    svc.answer("set rollout stage restricted_live_operations approved")
    # force daily loss gate failure
    source.load_runtime_flags = lambda: {"execute_trades": True, "read_only_ui": False, "daily_loss_pct": 2.0}  # type: ignore[method-assign]
    req_id = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    svc.answer(f"run execution {req_id}")
    snap = svc.execution_snapshot()
    assert snap["regression_triggered"] is True
    assert snap["rollout_stage"] in {"limited_live_pilot", "approved_paper_execution", "paper_only"}


def test_kill_switch_blocks_all_live_execution() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    source.load_runtime_flags = lambda: {"execute_trades": True, "read_only_ui": False}  # type: ignore[method-assign]
    svc.answer("set rollout stage limited_live_pilot approved")
    req_id = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
    svc.answer(f"approve execution {req_id} live-confirm")
    svc.answer("activate kill switch")
    blocked = svc.answer(f"run execution {req_id}")
    assert "kill_switch_active" in blocked["result"]


def test_dry_run_available_in_all_stages() -> None:
    source = _Source()
    source.load_runtime_flags = lambda: {"execute_trades": True, "read_only_ui": False}  # type: ignore[method-assign]
    svc = AskAtlasService(source)
    svc.answer("enable execution gate")
    for stage in ["paper_only", "approved_paper_execution", "limited_live_pilot", "restricted_live_operations", "scaled_live_reviewed"]:
        svc.answer(f"set rollout stage {stage} approved")
        req_id = svc.answer("create execution request paper place_trade dry instrument=XAUUSD setup=continuation size=0.01")["execution_request"]["id"]
        svc.answer(f"approve execution {req_id}")
        out = svc.answer(f"run execution {req_id}")
        assert "dry-run completed" in out["result"].lower()


def _set_active_session_for_now(gate: ExecutionGate, *, minutes_to_close: int = 120, buffer: int = 10) -> None:
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=30)).strftime("%H:%M")
    end = (now + timedelta(minutes=minutes_to_close)).strftime("%H:%M")
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="test", start=start, end=end, no_trade_buffer_minutes=buffer, weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]


def test_execution_blocked_outside_session() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate._session_control._sessions = [SessionWindow(name="closed", start="00:00", end="00:01", no_trade_buffer_minutes=1)]  # type: ignore[attr-defined]
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "outside_active_session"


def test_execution_blocked_near_session_close() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate, minutes_to_close=3, buffer=10)
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "near_session_close_no_new_trades"


def test_forced_flat_required_before_session_end() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate, minutes_to_close=2, buffer=10)
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False, "open_positions": 1}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "near_session_close_no_new_trades"
    audit = gate.recent_audit(limit=10)
    assert any(str(x.get("event", "")) == "forced_flat_required" for x in audit)


def test_daily_loss_limit_stops_trading() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate)
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False, "daily_loss_pct": 2.0}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "daily_loss_limit_reached"


def test_consecutive_loss_limit_pauses_trading() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate)
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False, "consecutive_losses": 3}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "max_consecutive_losses_reached"


def test_max_trades_per_day_enforced() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    _set_active_session_for_now(gate)
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False, "day_trade_count": 99}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "max_trades_per_day_reached"


def test_session_reset_restores_eligibility_safely() -> None:
    gate = ExecutionGate()
    _set_active_session_for_now(gate)
    gate._session_control.daily_stop_triggered = True  # type: ignore[attr-defined]
    gate._session_control.session_pause = True  # type: ignore[attr-defined]
    gate._session_control._day_key = "1999-01-01"  # type: ignore[attr-defined]
    changed = gate._session_control.maybe_reset_new_day()  # type: ignore[attr-defined]
    assert changed is True
    assert gate._session_control.daily_stop_triggered is False  # type: ignore[attr-defined]
    assert gate._session_control.session_pause is False  # type: ignore[attr-defined]


def test_ask_atlas_returns_session_state() -> None:
    svc = AskAtlasService(_Source())
    snap = svc.execution.snapshot()
    assert "session_control" in snap
    out = svc.answer("Are we in a trading session?")
    assert out["intent"] in {"runtime_status", "session_status"}
    assert "session active" in out["result"].lower()


def test_timezone_aware_session_open_close_calculation() -> None:
    gate = ExecutionGate()
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="ny", start="09:30", end="16:00", timezone_name="America/New_York", no_trade_buffer_minutes=10)
    ]
    state = gate.snapshot()["session_control"]
    assert "session_timezone" in state
    assert "current_local_time" in state
    assert state["session_timezone"] == "America/New_York"


def test_holiday_blocks_execution() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    now_local = datetime.now(timezone.utc).date().isoformat()
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="holiday_test", start="00:00", end="23:59", timezone_name="UTC", holidays=(now_local,), weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "market_closed_holiday"


def test_early_close_blocks_new_trades() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    now = datetime.now(timezone.utc)
    date_key = now.date().isoformat()
    start = (now - timedelta(minutes=60)).strftime("%H:%M")
    end = (now + timedelta(minutes=180)).strftime("%H:%M")
    early_close = (now + timedelta(minutes=3)).strftime("%H:%M")
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="early_close_test",
            start=start,
            end=end,
            timezone_name="UTC",
            no_trade_buffer_minutes=10,
            early_close_overrides={date_key: early_close},
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "early_close_no_new_trades"


def test_invalid_timezone_config_blocks_safely() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="bad_tz", start="00:00", end="23:59", timezone_name="Mars/Phobos")
    ]
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "invalid_timezone_config"


def test_next_session_calculation_present() -> None:
    gate = ExecutionGate()
    state = gate.snapshot()["session_control"]
    assert "next_session_local" in state
    assert isinstance(state["next_session_local"], str)


def test_ask_atlas_timezone_calendar_answers() -> None:
    svc = AskAtlasService(_Source())
    out_tz = svc.answer("What timezone are your session controls using?")
    out_cal = svc.answer("What market calendar is active?")
    assert out_tz["intent"] == "timezone_status"
    assert out_cal["intent"] in {"session_calendar_status", "session_status"}


def test_restart_recovery_still_respects_session_block() -> None:
    gate = ExecutionGate()
    gate.set_gate_enabled(True)
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="closed", start="00:00", end="00:01", timezone_name="UTC")
    ]
    req = gate.create_request(environment="paper", action_type="place_trade", payload_summary="instrument=XAUUSD setup=continuation size=0.01", requested_by="operator", dry_run=True)
    gate.approve_request(req["id"], approved_by="operator", note="ok")
    out = gate.execute(req["id"], runtime_flags={"execute_trades": False}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "outside_active_session"
