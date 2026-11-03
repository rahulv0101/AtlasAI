from __future__ import annotations

from datetime import datetime, timedelta, timezone
import time

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {
            "execute_trades": True,
            "read_only_ui": False,
            "open_positions": 0,
            "day_trade_count": 0,
            "daily_loss_pct": 0.0,
            "consecutive_losses": 0,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False},
        }
        self.status = {
            "action": "buy",
            "message": "High-confidence continuation setup.",
            "symbol": "XAUUSD",
            "setup_family": "continuation",
            "size": 0.01,
            "evidence_strength": 0.85,
            "min_confidence_threshold": 0.58,
            "chosen": {"confidence": 0.86},
            "cycle_id": "cycle_001",
        }

    def load_status(self) -> dict:
        return dict(self.status)

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "market", "message": "continuation alignment"}]

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}


def _set_active_session_for_now(svc: AskAtlasService, *, minutes_to_close: int = 90) -> None:
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=20)).strftime("%H:%M"),
            end=(now + timedelta(minutes=minutes_to_close)).strftime("%H:%M"),
            timezone_name="UTC",
            no_trade_buffer_minutes=0,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]


def _setup_demo_auto(svc: AskAtlasService) -> None:
    svc.answer("set execution environment demo")
    svc.answer("set autonomy mode guarded_execution")
    svc.answer("enable execution gate")
    svc.answer("set rollout stage limited_live_pilot approved")
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")
    svc.answer("create automation policy p1 live continuation XAUUSD test 0.01 0.6 2 limited_live_pilot")
    svc.answer("approve automation policy p1")
    svc.answer("activate automation policy p1")
    svc.answer("activate operator present mode")
    _set_active_session_for_now(svc)


def test_generic_recommendation_query_is_read_only() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    out = svc.answer("Should I buy or sell?")
    assert "auto_execution_result" not in out
    assert svc.execution.last_request() is None


def test_voice_like_recommendation_query_is_read_only() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    _ = svc.answer("atlas should we take this one now")
    assert svc.execution.last_request() is None


def test_explicit_auto_execution_command_required_for_action_path() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    out = svc.answer("run auto execution demo place_trade XAUUSD continuation 0.01 test 0.8 0.8 real")
    assert "automation_result" in out
    req = svc.execution.last_request()
    assert req is not None
    assert req.get("requested_by") == "automation"
    assert req.get("auto_generated") is True
    assert req.get("request_origin") == "automation"
    assert req.get("side") == "buy"
    assert float(req.get("confidence", 0.0) or 0.0) > 0.0


def test_duplicate_signal_blocked_only_within_window() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    first = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    second = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert first.get("ok") is True
    assert second.get("reason") == "duplicate_auto_signal"


def test_duplicate_signal_allowed_after_expiry() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    _ = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    svc.execution._auto_signal_history["cycle_001"] = time.time() - 1200  # type: ignore[attr-defined]
    out = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert out.get("reason") != "duplicate_auto_signal"


def test_temporary_block_is_not_deduped_and_can_retry() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    svc.answer("deactivate operator present mode")
    blocked = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert blocked.get("reason") == "operator_not_present"
    svc.answer("activate operator present mode")
    retried = svc.execution.attempt_auto_execution_from_signal(
        signal_id="cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=src.runtime,
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert retried.get("reason") != "duplicate_auto_signal"


def test_trader_guarded_origin_semantics_distinct_from_automation() -> None:
    svc = AskAtlasService(_Source())
    req = svc.execution.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=True,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    assert req.get("auto_generated") is False
    assert req.get("request_origin") == "trader_guarded"
    assert req.get("strict_runtime_checks") is True


def test_strict_runtime_freshness_still_blocks_guarded_execution() -> None:
    svc = AskAtlasService(_Source())
    svc.execution.set_gate_enabled(True)
    req = svc.execution.create_request(
        environment="paper",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="trader",
        dry_run=False,
        auto_generated=False,
        request_origin="trader_guarded",
        strict_runtime_checks=True,
    )
    rid = str(req.get("id", ""))
    svc.execution.approve_request(rid, approved_by="trader_engine")
    out = svc.execution.execute(
        rid,
        runtime_flags={"execute_trades": False, "freshness_runtime": {"status_age_ms": 120000, "is_stale": True}},
        risk_policy={"daily_loss_cap_pct": 5.0},
        execute_fn=lambda _req: {"ok": True, "note": "should_not_run"},
    )
    assert out.get("ok") is False
    assert out.get("reason") == "freshness_stale"


def test_snapshot_auto_trade_status_not_missing_freshness_by_default() -> None:
    gate = AskAtlasService(_Source()).execution
    snap = gate.snapshot()
    assert snap.get("auto_trade_allowed") is False
    assert snap.get("auto_trade_reason") != "missing_runtime_freshness"


def test_live_like_approval_count_includes_demo_requests() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    req = svc.execution.create_request(
        environment="demo",
        action_type="place_trade",
        payload_summary="instrument=XAUUSD setup_family=continuation size=0.01",
        requested_by="operator",
        dry_run=True,
        auto_generated=False,
    )
    rid = str(req.get("id", ""))
    svc.execution.approve_request(rid, approved_by="operator", explicit_live_confirmation=True)
    snap = svc.execution.snapshot()
    assert int(snap.get("live_approvals_used", 0) or 0) >= 1


def test_invalid_runtime_flag_reasons_are_specific() -> None:
    src = _Source()
    svc = AskAtlasService(src)
    _setup_demo_auto(svc)
    out_day = svc.execution.attempt_auto_execution_from_signal(
        signal_id="sig_bad_day_trade_count",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags={**src.runtime, "day_trade_count": "bad"},
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert out_day.get("reason") == "invalid_day_trade_count"
    out_loss = svc.execution.attempt_auto_execution_from_signal(
        signal_id="sig_bad_daily_loss",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags={**src.runtime, "daily_loss_pct": "bad"},
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert out_loss.get("reason") == "invalid_daily_loss_pct"
    out_con = svc.execution.attempt_auto_execution_from_signal(
        signal_id="sig_bad_consecutive_losses",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="test",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags={**src.runtime, "consecutive_losses": "bad"},
        risk_policy={"daily_loss_cap_pct": 5.0},
        dry_run=False,
    )
    assert out_con.get("reason") == "invalid_consecutive_losses"


def test_runtime_and_ask_share_coherent_execution_state() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("is execution enabled?")
    ask_state = dict(out.get("execution_state", {}) or {})
    direct_state = svc.execution.snapshot()
    assert ask_state.get("execution_gate_enabled") == direct_state.get("execution_gate_enabled")
    assert ask_state.get("execution_environment") == direct_state.get("execution_environment")
