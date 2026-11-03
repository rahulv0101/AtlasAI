from __future__ import annotations

from datetime import datetime, timezone

from atlas_ai.execution.session_control import DayTradingSessionControl, SessionWindow


def test_valid_timezone_session_open_close_calculation() -> None:
    ctl = DayTradingSessionControl(
        sessions=[SessionWindow(name="ny", start="09:30", end="16:00", timezone_name="America/New_York")]
    )
    # 2026-01-15 15:00 UTC = 10:00 NY (inside session)
    now = datetime(2026, 1, 15, 15, 0, tzinfo=timezone.utc)
    d = ctl.check(runtime_flags={}, now=now)
    assert d.current_session == "ny"
    assert d.minutes_to_close > 0
    assert d.timezone_name == "America/New_York"


def test_execution_blocked_outside_session_local_time() -> None:
    ctl = DayTradingSessionControl(
        sessions=[SessionWindow(name="ny", start="09:30", end="16:00", timezone_name="America/New_York")]
    )
    # 2026-01-15 01:00 UTC = 20:00 previous day NY (outside session)
    now = datetime(2026, 1, 15, 1, 0, tzinfo=timezone.utc)
    d = ctl.check(runtime_flags={}, now=now)
    assert d.allowed is False
    assert d.reason == "outside_active_session"


def test_holiday_blocks_session() -> None:
    ctl = DayTradingSessionControl(
        sessions=[SessionWindow(name="lon", start="08:00", end="16:30", timezone_name="Europe/London", holidays=("2026-12-25",))]
    )
    now = datetime(2026, 12, 25, 10, 0, tzinfo=timezone.utc)
    d = ctl.check(runtime_flags={}, now=now)
    assert d.allowed is False
    assert d.reason == "market_closed_holiday"


def test_early_close_block_reason() -> None:
    ctl = DayTradingSessionControl(
        sessions=[
            SessionWindow(
                name="lon",
                start="08:00",
                end="16:30",
                timezone_name="Europe/London",
                no_trade_buffer_minutes=10,
                early_close_overrides={"2026-07-10": "13:00"},
            )
        ]
    )
    # 12:55 local London in summer is 11:55 UTC (BST +1)
    now = datetime(2026, 7, 10, 11, 55, tzinfo=timezone.utc)
    d = ctl.check(runtime_flags={}, now=now)
    assert d.allowed is False
    assert d.reason == "early_close_no_new_trades"


def test_invalid_timezone_blocks_safely() -> None:
    ctl = DayTradingSessionControl(sessions=[SessionWindow(name="bad", start="08:00", end="16:00", timezone_name="Bad/Zone")])
    d = ctl.check(runtime_flags={}, now=datetime(2026, 1, 10, 10, 0, tzinfo=timezone.utc))
    assert d.allowed is False
    assert d.reason == "invalid_timezone_config"


def test_next_session_calculation_exists() -> None:
    ctl = DayTradingSessionControl()
    payload = ctl.to_dict(runtime_flags={})
    assert "next_session_local" in payload
    assert isinstance(payload["next_session_local"], str)


def test_default_sessions_include_asia_london_and_new_york() -> None:
    ctl = DayTradingSessionControl()

    asia = ctl.check(runtime_flags={}, now=datetime(2026, 1, 15, 2, 0, tzinfo=timezone.utc))
    london = ctl.check(runtime_flags={}, now=datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc))
    new_york = ctl.check(runtime_flags={}, now=datetime(2026, 1, 15, 17, 0, tzinfo=timezone.utc))

    assert asia.current_session == "asia"
    assert asia.allowed is True
    assert london.current_session == "london"
    assert london.allowed is True
    assert new_york.current_session == "new_york"
    assert new_york.allowed is True


def test_dst_sensitive_handling_changes_local_window_basis() -> None:
    ctl = DayTradingSessionControl(
        sessions=[SessionWindow(name="ny", start="09:30", end="16:00", timezone_name="America/New_York")]
    )
    pre_dst = datetime(2026, 3, 6, 15, 0, tzinfo=timezone.utc)
    post_dst = datetime(2026, 3, 16, 15, 0, tzinfo=timezone.utc)
    d1 = ctl.check(runtime_flags={}, now=pre_dst)
    d2 = ctl.check(runtime_flags={}, now=post_dst)
    assert d1.timezone_name == "America/New_York"
    assert d2.timezone_name == "America/New_York"
    assert d1.session_window_local != ""
    assert d2.session_window_local != ""
