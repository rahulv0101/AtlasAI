from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.broker_adapter import BrokerOrderResponse, TestBrokerAdapter
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.mt5_broker_adapter import MT5BrokerAdapter
from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService
import atlas_ai.execution.mt5_broker_adapter as mt5_adapter_module
from atlas_ai.execution.trade_executor import ExecutionResult


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": True, "read_only_ui": False}
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.61},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "hold"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}

    def execute_trade(self) -> None:
        self.execute_called = True


def _open_session(gate: ExecutionGate) -> None:
    now = datetime.now(timezone.utc)
    gate._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=30)).strftime("%H:%M"),
            end=(now + timedelta(minutes=120)).strftime("%H:%M"),
            timezone_name="UTC",
            no_trade_buffer_minutes=0,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def _ready_live_gate(gate: ExecutionGate) -> None:
    gate.set_gate_enabled(True)
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    gate.set_broker_adapter(TestBrokerAdapter())
    gate.configure_tiny_live_pilot(
        allowed_instrument="XAUUSD",
        allowed_setup_family="continuation",
        size_cap=0.01,
        max_trades_per_day=3,
        daily_loss_cap=0.6,
        max_consecutive_losses=2,
    )
    gate.approve_tiny_live_pilot(note="test")
    gate.activate_tiny_live_pilot()
    _open_session(gate)


def _live_req(gate: ExecutionGate) -> str:
    req = gate.create_request(
        environment="live",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup=continuation size=0.01",
        requested_by="tester",
        dry_run=False,
    )
    gate.approve_request(req["id"], approved_by="tester", explicit_live_confirmation=True)
    return str(req["id"])


def test_approved_live_request_submits_once() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is True
    snap = gate.snapshot()
    assert snap["last_live_order_status"] in {"acknowledged", "submitted", "filled", "partially_filled"}


def test_duplicate_submit_does_not_create_duplicate_order() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    audit = gate.recent_audit(limit=50)
    submit_events = [x for x in audit if x.get("event") == "order_submit_attempt" and x.get("request_id") == rid]
    assert len(submit_events) == 1


def test_broker_reject_is_captured_and_exposed() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=False, status="rejected", broker_order_id="BRK_REJ_1", message="insufficient_margin"),
    )
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert "broker_rejected" in out["reason"]
    assert "insufficient_margin" in gate.snapshot()["last_broker_reject_reason"]
    truth = gate.snapshot()["execution_truth"]
    assert truth["ok"] is False
    assert truth["failure_class"] == "broker_reject"
    assert truth["request_id"] == rid


def test_partial_fill_updates_normalized_state() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(
            ok=True,
            status="partially_filled",
            broker_order_id="BRK_PF_1",
            message="partial",
            fill_qty=0.005,
            avg_fill_price=2321.2,
        ),
    )
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is True
    assert gate.snapshot()["last_live_order_status"] == "partially_filled"


def test_cancel_request_updates_state_safely() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    cancel = gate.cancel_live_order(rid)
    assert cancel["ok"] is True
    assert gate.snapshot()["last_live_order_status"] == "cancelled"


def test_uncertain_submit_enters_reconciliation_required() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=False, status="unknown", broker_order_id="BRK_UNK_1", message="timeout"),
    )
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "order_reconciliation_required"
    assert gate.snapshot()["reconciliation_required"] is True
    assert gate.snapshot()["execution_truth"]["failure_class"] == "reconciliation_required"


def test_successful_live_execution_truth_exposes_broker_order() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=True, status="filled", broker_order_id="BRK_FILLED_1", message="filled"),
    )
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is True
    truth = gate.snapshot()["execution_truth"]
    assert truth["ok"] is True
    assert truth["broker_order_id"] == "BRK_FILLED_1"
    assert truth["broker_status"] == "filled"


def test_restart_pending_order_requires_reconciliation_not_blind_resubmit(tmp_path) -> None:
    state = tmp_path / "exec_state.json"
    gate1 = ExecutionGate(state_store_path=str(state))
    _ready_live_gate(gate1)
    rid = _live_req(gate1)
    gate1._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=True, status="submitted", broker_order_id="BRK_RES_1", message="submitted"),
    )
    gate1.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    gate1.mark_clean_shutdown()

    gate2 = ExecutionGate(state_store_path=str(state))
    # No blind resubmit on restart; unresolved order remains visible for reconciliation.
    snap = gate2.snapshot()
    assert snap["pending_live_orders_count"] >= 1


def test_malformed_broker_response_fails_safely() -> None:
    gate = ExecutionGate()
    _ready_live_gate(gate)
    rid = _live_req(gate)
    gate._broker_adapter.set_submit_override(  # type: ignore[attr-defined]
        rid,
        BrokerOrderResponse(ok=True, status="acknowledged", broker_order_id="", message="missing_id"),
    )
    out = gate.execute(rid, runtime_flags={"execute_trades": True}, risk_policy={"rules": ["x"]})
    assert out["ok"] is False
    assert out["reason"] == "broker_response_invalid"


def test_mt5_adapter_rejects_success_without_broker_confirmation() -> None:
    class _Executor:
        def execute(self, signal, volume: float, decision_id: str = "") -> ExecutionResult:
            return ExecutionResult(True, 0, "filled", order_id=0, deal_id=0)

    adapter = MT5BrokerAdapter(_Executor())  # type: ignore[arg-type]
    out = adapter.submit_order(
        idempotency_key="REQ_1",
        request={
            "side": "buy",
            "strategy": "continuation",
            "confidence": 0.7,
            "stop_loss_points": 100,
            "take_profit_points": 150,
            "volume": 0.01,
            "decision_id": "DEC_1",
        },
    )
    assert out.ok is False
    assert out.status == "unknown"
    assert out.broker_order_id == ""
    assert "missing_broker_confirmation" in out.message


def test_ask_atlas_reports_order_and_reconciliation_state() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    req = svc.answer("create execution request live place_trade real instrument=XAUUSD setup=continuation size=0.01")
    rid = req["execution_request"]["id"]
    svc.answer(f"approve execution {rid} live-confirm")
    svc.answer(f"run execution {rid}")
    r1 = svc.answer("What happened to the last live order?")
    r2 = svc.answer("Is reconciliation required?")
    assert r1["intent"] == "order_status"
    assert r2["intent"] == "order_reconciliation"


def test_snapshot_includes_order_health_info() -> None:
    gate = ExecutionGate()
    snap = gate.snapshot()
    assert "last_live_order_status" in snap
    assert "pending_live_orders_count" in snap
    assert "partially_filled_count" in snap
    assert "reconciliation_required" in snap


def test_request_payload_extracts_trade_fields_without_setup_collision() -> None:
    gate = ExecutionGate()
    gate.set_rollout_stage("limited_live_pilot", approved=True)
    req = gate.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary=(
            "instrument=XAUUSD setup_family=continuation setup=breakout "
            "strategy=gold_liquidity_sweep side=sell size=0.01 confidence=0.62 "
            "stop_loss_points=95 take_profit_points=135 data_source=broker_runtime environment_mode=demo"
        ),
        requested_by="tester",
        dry_run=False,
    )
    assert req["setup_family"] == "breakout"
    assert req["strategy"] == "gold_liquidity_sweep"
    assert req["side"] == "sell"
    assert req["confidence"] == 0.62
    assert req["stop_loss_points"] == 95.0
    assert req["take_profit_points"] == 135.0


def test_mt5_adapter_lists_open_positions_for_reconciliation(monkeypatch) -> None:
    class _Settings:
        symbol = "XAUUSD"

    class _Executor:
        settings = _Settings()

    class _Position:
        ticket = 123
        type = 1
        volume = 1.0
        symbol = "XAUUSD"
        profit = -50.0
        sl = 0.0
        tp = 0.0
        magic = 0
        comment = ""

    monkeypatch.setattr(mt5_adapter_module.mt5, "POSITION_TYPE_BUY", 0, raising=False)
    monkeypatch.setattr(mt5_adapter_module.mt5, "positions_get", lambda symbol: [_Position()])

    rows = MT5BrokerAdapter(_Executor()).list_open_orders()  # type: ignore[arg-type]

    assert rows == [
        {
            "broker_order_id": "123",
            "instrument": "XAUUSD",
            "side": "sell",
            "size": 1.0,
            "status": "open",
            "profit": -50.0,
            "sl": 0.0,
            "tp": 0.0,
            "magic": 0,
            "comment": "",
        }
    ]


def test_mt5_adapter_preserves_buy_position_type_zero_for_reconciliation(monkeypatch) -> None:
    class _Settings:
        symbol = "XAUUSD"

    class _Executor:
        settings = _Settings()

    class _Position:
        ticket = 124
        type = 0
        volume = 0.01
        symbol = "XAUUSD"
        profit = -1.25
        sl = 4400.0
        tp = 4450.0
        magic = 20260330
        comment = "AAI_manual_test"

    monkeypatch.setattr(mt5_adapter_module.mt5, "POSITION_TYPE_BUY", 0, raising=False)
    monkeypatch.setattr(mt5_adapter_module.mt5, "positions_get", lambda symbol: [_Position()])

    rows = MT5BrokerAdapter(_Executor()).list_open_orders()  # type: ignore[arg-type]

    assert rows[0]["side"] == "buy"
