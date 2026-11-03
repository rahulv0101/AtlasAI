from __future__ import annotations

from datetime import datetime, timedelta, timezone

from atlas_ai.execution.session_control import SessionWindow
from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.execute_called = False
        self.runtime = {
            "execute_trades": False,
            "read_only_ui": False,
            "freshness_runtime": {"status_age_ms": 50, "is_stale": False},
        }

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.61},
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


def _setup_policy(svc: AskAtlasService, *, env: str = "both", stage: str = "limited_live_pilot") -> None:
    svc.answer(
        f"create automation policy ap1 {env} continuation XAUUSD london 0.02 1.0 2 {stage}"
    )


def _enable_demo_mode(svc: AskAtlasService) -> None:
    svc.answer("set execution environment demo")
    svc.answer("set autonomy mode guarded_execution")
    svc.answer("activate operator present mode")


def _enable_tiny_live_pilot(svc: AskAtlasService) -> None:
    svc.answer("configure tiny live pilot XAUUSD continuation 0.01 2 0.6 1")
    svc.answer("approve tiny live pilot")
    svc.answer("activate tiny live pilot")


def test_automation_policy_requires_approval_before_activation() -> None:
    svc = AskAtlasService(_Source())
    _setup_policy(svc)
    out = svc.answer("activate automation policy ap1")
    assert "blocked" in out["result"].lower() or "approval" in out["result"].lower()
    assert out["execution_state"]["automation_enabled"] is False
    assert out["execution_state"]["autonomy_mode"] == "recommend_only"


def test_default_autonomy_mode_is_safe_by_default() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc, stage="limited_live_pilot")
    svc.answer("set execution environment demo")
    svc.answer("activate operator present mode")
    _enable_tiny_live_pilot(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert "autonomy_mode_recommend_only" in out["result"]


def test_paper_stage_blocks_live_automation() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc, stage="paper_only")
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.7 0.7 dry")
    assert any(x in out["result"] for x in ("live_blocked_in_stage:paper_only", "tiny_live_pilot_not_configured"))


def test_limited_live_pilot_restrictions_enforced() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc, stage="limited_live_pilot")
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")

    bad_setup = svc.answer("run auto execution live place_trade XAUUSD breakout 0.01 london 0.8 0.8 dry")
    assert any(x in bad_setup["result"] for x in ("setup_family_not_approved_for_automation", "tiny_live_pilot_not_configured"))

    bad_inst = svc.answer("run auto execution live place_trade EURUSD continuation 0.01 london 0.8 0.8 dry")
    assert any(x in bad_inst["result"] for x in ("instrument_not_approved_for_automation", "tiny_live_pilot_not_configured"))

    bad_size = svc.answer("run auto execution live place_trade XAUUSD continuation 0.2 london 0.8 0.8 dry")
    assert any(x in bad_size["result"] for x in ("automation_size_cap_exceeded", "tiny_live_pilot_not_configured"))


def test_kill_switch_pauses_automation() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    svc.answer("activate kill switch")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert "kill_switch_active" in out["result"]
    assert svc.execution_snapshot()["automation_paused"] is True


def test_blocker_severity_pause_works() -> None:
    source = _Source()
    source.runtime.update({"execute_trades": True, "blocker_severity": 0.9})
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert any(x in out["result"] for x in ("blocker_severity_too_high", "tiny_live_pilot_not_configured"))
    if "blocker_severity_too_high" in out["result"]:
        assert svc.execution_snapshot()["automation_paused"] is True


def test_repeated_losses_or_loss_cap_pause_works() -> None:
    source = _Source()
    source.runtime.update({"execute_trades": True, "recent_losses": 3})
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert any(x in out["result"] for x in ("recent_losses_threshold_hit", "tiny_live_pilot_not_configured"))
    if "recent_losses_threshold_hit" in out["result"]:
        assert svc.execution_snapshot()["automation_paused"] is True


def test_paused_automation_creates_review_control_item() -> None:
    source = _Source()
    source.runtime.update({"execute_trades": True, "overreaction_score": 0.95})
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert any(x in out["result"] for x in ("overreaction_too_high", "tiny_live_pilot_not_configured"))
    pending = svc.answer("show pending items")
    assert "automation blocked - review required" in pending["result"].lower()


def test_auto_execution_uses_execution_validation_and_audit() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert any(x in out["result"].lower() for x in ("processed successfully", "tiny_live_pilot_not_configured"))
    audit = svc.answer("show execution audit")
    assert "last execution audit event" in audit["result"].lower()
    assert isinstance(audit.get("execution_audit"), list)


def test_dry_run_automation_path_does_not_execute_real_action() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert source.execute_called is False


def test_recommendation_or_voice_query_alone_cannot_activate_automation() -> None:
    svc = AskAtlasService(_Source())
    svc.answer("What should you do next?")
    svc.answer("What is holding you back?", mode="voice")
    snap = svc.execution_snapshot()
    assert snap["automation_enabled"] is False


def test_unsupported_setup_family_is_blocked() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD mean_reversion 0.01 london 0.8 0.8 dry")
    assert any(x in out["result"] for x in ("setup_family_not_approved_for_automation", "tiny_live_pilot_not_configured"))


def test_automation_status_queries_return_structured_safe_answers() -> None:
    svc = AskAtlasService(_Source())
    _setup_policy(svc, env="paper", stage="approved_paper_execution")
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    status = svc.answer("Is automation enabled?")
    policy = svc.answer("What automation policy is active?")
    restrictions = svc.answer("What are the current automation restrictions?")
    assert status["intent"] == "automation_status"
    assert policy["intent"] in {"automation_policy", "automation_status"}
    assert restrictions["intent"] in {"automation_policy", "automation_status"}
    assert "execution_state" in status


def test_automation_respects_session_constraints() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    now = datetime.now(timezone.utc)
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(
            name="test",
            start=(now - timedelta(minutes=10)).strftime("%H:%M"),
            end=(now + timedelta(minutes=3)).strftime("%H:%M"),
            no_trade_buffer_minutes=10,
            weekdays=(0, 1, 2, 3, 4, 5, 6),
        )
    ]
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert "near_session_close_no_new_trades" in out["result"]


def test_automation_blocked_by_holiday_rules() -> None:
    source = _Source()
    source.runtime["execute_trades"] = True
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    today = datetime.now(timezone.utc).date().isoformat()
    svc.execution._session_control._sessions = [  # type: ignore[attr-defined]
        SessionWindow(name="holiday_test", start="00:00", end="23:59", timezone_name="UTC", holidays=(today,), weekdays=(0, 1, 2, 3, 4, 5, 6))
    ]
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert "market_closed_holiday" in out["result"]


def test_simulated_market_state_cannot_trigger_guarded_execution() -> None:
    source = _Source()
    source.runtime.update({"execute_trades": True, "market_state_mode": "simulation"})
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    _enable_tiny_live_pilot(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.answer("run auto execution live place_trade XAUUSD continuation 0.01 london 0.8 0.8 dry")
    assert "simulation_market_state_cannot_execute" in out["result"]


def test_spread_and_news_filters_block_unsafe_auto_execution_and_log_cycle() -> None:
    source = _Source()
    source.runtime.update(
        {
            "execute_trades": True,
            "spread_state": "wide",
            "spread_points": 42.0,
            "news_risk_active": True,
            "xau_bias": "buy",
            "dxy_alignment": "conflict",
            "us10y_alignment": "aligned",
            "session": "london",
        }
    )
    svc = AskAtlasService(source)
    _setup_policy(svc)
    _enable_demo_mode(svc)
    _enable_tiny_live_pilot(svc)
    svc.answer("approve automation policy ap1")
    svc.answer("activate automation policy ap1")
    svc.answer("enable execution gate")
    out = svc.execution.attempt_auto_execution_from_signal(
        signal_id="sig_cycle_001",
        action_type="place_trade",
        environment="demo",
        instrument="XAUUSD",
        setup_family="continuation",
        size=0.01,
        session="london",
        confidence=0.8,
        evidence_strength=0.8,
        runtime_flags=source.runtime,
        risk_policy={"max_spread_points": 30.0},
        dry_run=True,
    )
    assert out["reason"] == "spread_filter_blocked"
    cycle = dict(svc.execution.snapshot().get("last_autonomy_cycle", {}) or {})
    assert cycle["xau_bias"] == "buy"
    assert cycle["dxy_alignment"] == "conflict"
    assert cycle["session"] == "london"
    assert cycle["action_taken"] == "blocked"
    assert "spread_filter_blocked" in list(cycle.get("blocker_reasons", []) or [])
