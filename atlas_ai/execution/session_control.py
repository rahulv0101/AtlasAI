from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: str
    end: str
    timezone_name: str = "UTC"
    no_trade_buffer_minutes: int = 10
    no_trade_open_buffer_minutes: int = 0
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: tuple[str, ...] = ()
    early_close_overrides: dict[str, str] | None = None


@dataclass
class SessionDecision:
    allowed: bool
    reason: str
    current_session: str
    session_active: bool
    minutes_to_close: int
    forced_flat_required: bool
    timezone_name: str
    local_time_iso: str
    session_window_local: str
    calendar_status: str
    next_session_local: str


class DayTradingSessionControl:
    def __init__(
        self,
        *,
        sessions: list[SessionWindow] | None = None,
        daily_loss_limit: float = 1.5,
        max_trades_per_day: int = 6,
        max_consecutive_losses: int = 2,
        cooldown_minutes_after_loss_cluster: int = 30,
    ) -> None:
        self._sessions = sessions or [
            SessionWindow(name="asia", start="00:00", end="07:00", timezone_name="UTC", no_trade_buffer_minutes=10),
            SessionWindow(name="london", start="08:00", end="16:30", timezone_name="Europe/London", no_trade_buffer_minutes=10),
            SessionWindow(name="new_york", start="09:30", end="16:00", timezone_name="America/New_York", no_trade_buffer_minutes=10),
        ]
        self.daily_loss_limit = float(max(0.0, daily_loss_limit))
        self.max_trades_per_day = int(max(1, max_trades_per_day))
        self.max_consecutive_losses = int(max(1, max_consecutive_losses))
        self.cooldown_minutes_after_loss_cluster = int(max(0, cooldown_minutes_after_loss_cluster))
        self._day_key = self._today_key(datetime.now(timezone.utc), self._sessions[0].timezone_name if self._sessions else "UTC")
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.daily_stop_triggered = False
        self.session_pause = False
        self.pause_reason = ""
        self.block_reason = ""
        self.last_reset_at = datetime.now(timezone.utc).timestamp()
        self.last_pause_at = 0.0
        self.session_source = "in_memory_default"
        self._last_timezone_error = ""
        self._last_calendar_note = ""

    @staticmethod
    def _parse_hhmm(value: str) -> tuple[int, int]:
        hh, mm = str(value).split(":")
        return int(hh), int(mm)

    @staticmethod
    def _to_local(now_utc: datetime, tz_name: str) -> datetime:
        return now_utc.astimezone(ZoneInfo(tz_name))

    @staticmethod
    def _minutes_between(a: datetime, b: datetime) -> int:
        return int((b - a).total_seconds() // 60)

    @staticmethod
    def _today_key(now_utc: datetime, tz_name: str) -> str:
        try:
            return now_utc.astimezone(ZoneInfo(tz_name)).date().isoformat()
        except Exception:
            return now_utc.date().isoformat()

    def _resolve_window_bounds(self, local_now: datetime, session: SessionWindow) -> tuple[datetime, datetime]:
        sh, sm = self._parse_hhmm(session.start)
        eh, em = self._parse_hhmm(session.end)
        start_dt = local_now.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end_dt = local_now.replace(hour=eh, minute=em, second=0, microsecond=0)
        if end_dt <= start_dt:
            end_dt = end_dt + timedelta(days=1)
        return start_dt, end_dt

    def _effective_close_dt(self, local_now: datetime, session: SessionWindow, default_end: datetime) -> tuple[datetime, str]:
        overrides = dict(session.early_close_overrides or {})
        day_key = local_now.date().isoformat()
        if day_key not in overrides:
            return default_end, "regular"
        try:
            hh, mm = self._parse_hhmm(str(overrides[day_key]))
            close_dt = local_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if close_dt <= local_now.replace(hour=0, minute=0, second=0, microsecond=0):
                close_dt = default_end
            return close_dt, "early_close"
        except Exception:
            return default_end, "invalid_early_close_override"

    def _find_next_session(self, now_utc: datetime, days_ahead: int = 7) -> str:
        soonest: datetime | None = None
        soonest_name = ""
        for offset in range(0, max(1, days_ahead + 1)):
            check_utc = now_utc + timedelta(days=offset)
            for s in self._sessions:
                try:
                    local = self._to_local(check_utc, s.timezone_name)
                except Exception:
                    continue
                if local.weekday() not in set(s.weekdays):
                    continue
                if local.date().isoformat() in set(s.holidays):
                    continue
                sh, sm = self._parse_hhmm(s.start)
                start_dt = local.replace(hour=sh, minute=sm, second=0, microsecond=0)
                if offset == 0 and start_dt <= local:
                    continue
                if soonest is None or start_dt < soonest:
                    soonest = start_dt
                    soonest_name = s.name
        if soonest is None:
            return "unknown"
        return f"{soonest_name}@{soonest.isoformat()}"

    def _closed_decision(self, *, reason: str, now_utc: datetime, tz_name: str = "UTC", calendar_status: str = "unknown") -> SessionDecision:
        next_s = self._find_next_session(now_utc)
        local_iso = now_utc.isoformat()
        try:
            local_iso = self._to_local(now_utc, tz_name).isoformat()
        except Exception:
            pass
        return SessionDecision(
            allowed=False,
            reason=reason,
            current_session="none",
            session_active=False,
            minutes_to_close=-1,
            forced_flat_required=False,
            timezone_name=tz_name,
            local_time_iso=local_iso,
            session_window_local="none",
            calendar_status=calendar_status,
            next_session_local=next_s,
        )

    def _detect_session(self, now_utc: datetime) -> SessionDecision:
        if not self._sessions:
            return self._closed_decision(reason="session_calendar_unknown", now_utc=now_utc, calendar_status="missing_session_config")

        saw_invalid_tz = False
        saw_holiday = False
        last_holiday_tz = "UTC"
        for s in self._sessions:
            try:
                local_now = self._to_local(now_utc, s.timezone_name)
            except Exception:
                saw_invalid_tz = True
                self._last_timezone_error = s.timezone_name
                continue

            if local_now.weekday() not in set(s.weekdays):
                continue
            if local_now.date().isoformat() in set(s.holidays):
                saw_holiday = True
                last_holiday_tz = s.timezone_name
                continue

            start_dt, end_dt = self._resolve_window_bounds(local_now, s)
            close_dt, close_mode = self._effective_close_dt(local_now, s, end_dt)
            if close_mode == "invalid_early_close_override":
                return self._closed_decision(
                    reason="session_calendar_unknown",
                    now_utc=now_utc,
                    tz_name=s.timezone_name,
                    calendar_status="invalid_early_close_override",
                )

            if not (start_dt <= local_now <= close_dt):
                continue

            mins_to_close = max(0, self._minutes_between(local_now, close_dt))
            open_buffer = int(max(0, s.no_trade_open_buffer_minutes))
            close_buffer = int(max(0, s.no_trade_buffer_minutes))
            mins_from_open = max(0, self._minutes_between(start_dt, local_now))
            window = f"{start_dt.isoformat()}->{close_dt.isoformat()}"
            if mins_from_open <= open_buffer:
                return SessionDecision(
                    allowed=False,
                    reason="near_session_open_no_new_trades",
                    current_session=s.name,
                    session_active=True,
                    minutes_to_close=mins_to_close,
                    forced_flat_required=False,
                    timezone_name=s.timezone_name,
                    local_time_iso=local_now.isoformat(),
                    session_window_local=window,
                    calendar_status=close_mode,
                    next_session_local=self._find_next_session(now_utc),
                )
            if mins_to_close <= close_buffer:
                reason = "early_close_no_new_trades" if close_mode == "early_close" else "near_session_close_no_new_trades"
                return SessionDecision(
                    allowed=False,
                    reason=reason,
                    current_session=s.name,
                    session_active=True,
                    minutes_to_close=mins_to_close,
                    forced_flat_required=False,
                    timezone_name=s.timezone_name,
                    local_time_iso=local_now.isoformat(),
                    session_window_local=window,
                    calendar_status=close_mode,
                    next_session_local=self._find_next_session(now_utc),
                )
            return SessionDecision(
                allowed=True,
                reason="",
                current_session=s.name,
                session_active=True,
                minutes_to_close=mins_to_close,
                forced_flat_required=False,
                timezone_name=s.timezone_name,
                local_time_iso=local_now.isoformat(),
                session_window_local=window,
                calendar_status=close_mode,
                next_session_local=self._find_next_session(now_utc),
            )

        if saw_invalid_tz:
            return self._closed_decision(reason="invalid_timezone_config", now_utc=now_utc, calendar_status="invalid_timezone")
        if saw_holiday:
            return self._closed_decision(reason="market_closed_holiday", now_utc=now_utc, tz_name=last_holiday_tz, calendar_status="holiday")
        fallback_tz = self._sessions[0].timezone_name if self._sessions else "UTC"
        return self._closed_decision(reason="outside_active_session", now_utc=now_utc, tz_name=fallback_tz, calendar_status="closed")

    def maybe_reset_new_day(self, now: datetime | None = None) -> bool:
        ts = now or datetime.now(timezone.utc)
        reset_tz = self._sessions[0].timezone_name if self._sessions else "UTC"
        key = self._today_key(ts, reset_tz)
        if key == self._day_key:
            return False
        self._day_key = key
        self.daily_trade_count = 0
        self.consecutive_losses = 0
        self.daily_stop_triggered = False
        self.session_pause = False
        self.pause_reason = ""
        self.block_reason = ""
        self.last_reset_at = ts.timestamp()
        return True

    def check(
        self,
        *,
        runtime_flags: dict[str, Any],
        now: datetime | None = None,
        opening_new_trade: bool = True,
    ) -> SessionDecision:
        ts = now or datetime.now(timezone.utc)
        self.maybe_reset_new_day(now=ts)
        base = self._detect_session(ts)

        # Runtime overrides (authoritative when available).
        daily_loss = float(runtime_flags.get("daily_loss_pct", 0.0) or 0.0)
        trades_today = int(runtime_flags.get("day_trade_count", self.daily_trade_count) or self.daily_trade_count)
        consecutive_losses = int(runtime_flags.get("consecutive_losses", self.consecutive_losses) or self.consecutive_losses)
        open_positions = int(runtime_flags.get("open_positions", 0) or 0)

        if self.cooldown_minutes_after_loss_cluster > 0 and self.session_pause and self.last_pause_at > 0:
            elapsed = int((ts.timestamp() - self.last_pause_at) // 60)
            if elapsed >= self.cooldown_minutes_after_loss_cluster and not self.daily_stop_triggered:
                self.session_pause = False
                self.pause_reason = ""

        if daily_loss >= self.daily_loss_limit > 0:
            self.daily_stop_triggered = True
            self.block_reason = "daily_loss_limit_reached"
            base.allowed = False
            base.reason = self.block_reason
            base.forced_flat_required = open_positions > 0
            return base
        if trades_today >= self.max_trades_per_day:
            self.block_reason = "max_trades_per_day_reached"
            base.allowed = False
            base.reason = self.block_reason
            return base
        if consecutive_losses >= self.max_consecutive_losses:
            self.session_pause = True
            self.pause_reason = "max_consecutive_losses_reached"
            self.last_pause_at = ts.timestamp()
            self.block_reason = "max_consecutive_losses_reached"
            base.allowed = False
            base.reason = self.block_reason
            return base
        if self.session_pause:
            self.block_reason = f"session_paused:{self.pause_reason or 'risk_pause'}"
            base.allowed = False
            base.reason = self.block_reason
            return base
        if not base.allowed:
            self.block_reason = base.reason or "outside_active_session"
            base.forced_flat_required = open_positions > 0 and base.reason in {"near_session_close_no_new_trades", "early_close_no_new_trades"}
            return base
        if opening_new_trade and base.minutes_to_close <= 1:
            self.block_reason = "insufficient_time_to_close"
            base.allowed = False
            base.reason = self.block_reason
            base.forced_flat_required = open_positions > 0
            return base
        self.block_reason = ""
        return base

    def record_execution(self, *, executed: bool, dry_run: bool, outcome: str | None = None) -> None:
        if dry_run or not executed:
            return
        self.daily_trade_count += 1
        if str(outcome or "").lower() == "loss":
            self.consecutive_losses += 1
        elif str(outcome or "").lower() == "win":
            self.consecutive_losses = 0

    def to_dict(self, *, runtime_flags: dict[str, Any] | None = None) -> dict[str, Any]:
        flags = dict(runtime_flags or {})
        decision = self.check(runtime_flags=flags, opening_new_trade=False)
        return {
            "current_session": decision.current_session,
            "session_active": decision.session_active,
            "minutes_to_session_close": decision.minutes_to_close,
            "session_timezone": decision.timezone_name,
            "current_local_time": decision.local_time_iso,
            "current_session_window_local": decision.session_window_local,
            "session_calendar_status": decision.calendar_status,
            "next_session_local": decision.next_session_local,
            "session_source": self.session_source,
            "daily_trade_count": int(flags.get("day_trade_count", self.daily_trade_count) or self.daily_trade_count),
            "daily_trade_limit": self.max_trades_per_day,
            "consecutive_losses": int(flags.get("consecutive_losses", self.consecutive_losses) or self.consecutive_losses),
            "consecutive_loss_limit": self.max_consecutive_losses,
            "daily_stop_triggered": self.daily_stop_triggered,
            "session_pause": self.session_pause,
            "session_pause_reason": self.pause_reason,
            "session_block_reason": self.block_reason or decision.reason,
            "forced_flat_required": decision.forced_flat_required,
            "last_reset_at": self.last_reset_at,
        }
