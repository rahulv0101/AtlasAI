from __future__ import annotations

from datetime import datetime, timezone
import time

import MetaTrader5 as mt5
import pandas as pd

from atlas_ai.config.settings import Settings

_TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
}


class MarketDataFeed:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_fetch_meta: dict[str, object] = {}

    def _ensure_initialized(self) -> None:
        terminal = mt5.terminal_info()
        if terminal is not None and bool(getattr(terminal, "connected", False)):
            return

        init_ok = False
        last_error = None
        for _ in range(3):
            if self.settings.hfm_terminal_path:
                init_ok = mt5.initialize(path=self.settings.hfm_terminal_path)
            else:
                init_ok = mt5.initialize()
            if init_ok:
                return
            last_error = mt5.last_error()
            time.sleep(1.0)

        if self.settings.hfm_terminal_path:
            for _ in range(2):
                init_ok = mt5.initialize()
                if init_ok:
                    return
                last_error = mt5.last_error()
                time.sleep(1.0)

        raise RuntimeError(f"MT5 initialize failed: {last_error or mt5.last_error()}")

    def get_rates(self, bars: int = 1000) -> pd.DataFrame:
        started = datetime.now(timezone.utc)
        timeframe = _TIMEFRAME_MAP.get(self.settings.timeframe.upper(), mt5.TIMEFRAME_M15)
        rates = None
        last_error = None
        retries = 0
        retry_sleep_seconds = 0.0
        for _ in range(3):
            try:
                self._ensure_initialized()
            except Exception as exc:
                last_error = str(exc)
                retries += 1
                retry_sleep_seconds += 1.0
                time.sleep(1.0)
                continue
            mt5.symbol_select(self.settings.symbol, True)
            rates = mt5.copy_rates_from_pos(self.settings.symbol, timeframe, 0, bars)
            if rates is not None and len(rates):
                break
            last_error = mt5.last_error()
            retries += 1
            retry_sleep_seconds += 1.0
            time.sleep(1.0)
        if rates is None:
            ended = datetime.now(timezone.utc)
            self.last_fetch_meta = {
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_ms": round((ended - started).total_seconds() * 1000.0, 2),
                "retries": int(retries),
                "retry_sleep_ms": round(retry_sleep_seconds * 1000.0, 2),
                "bars_requested": int(bars),
                "bars_received": 0,
                "ok": False,
                "error": str(last_error or mt5.last_error()),
            }
            raise RuntimeError(f"Failed to fetch MT5 rates: {last_error or mt5.last_error()}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        ended = datetime.now(timezone.utc)
        self.last_fetch_meta = {
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_ms": round((ended - started).total_seconds() * 1000.0, 2),
            "retries": int(retries),
            "retry_sleep_ms": round(retry_sleep_seconds * 1000.0, 2),
            "bars_requested": int(bars),
            "bars_received": int(len(df)),
            "ok": True,
            "error": "",
        }
        return df

    def get_rates_for_timeframe(self, timeframe_name: str, bars: int = 400) -> pd.DataFrame:
        self._ensure_initialized()
        timeframe = _TIMEFRAME_MAP.get(str(timeframe_name).upper(), mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(self.settings.symbol, timeframe, 0, bars)
        if rates is None:
            raise RuntimeError(f"Failed to fetch MT5 rates for timeframe {timeframe_name}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def get_rates_range(self, start: datetime, end: datetime) -> pd.DataFrame:
        self._ensure_initialized()
        timeframe = _TIMEFRAME_MAP.get(self.settings.timeframe.upper(), mt5.TIMEFRAME_M15)
        rates = mt5.copy_rates_range(self.settings.symbol, timeframe, start, end)
        if rates is None:
            raise RuntimeError("Failed to fetch MT5 rates range")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def diagnostics(self, bars: int = 50) -> dict[str, object]:
        self._ensure_initialized()
        timeframe_name = self.settings.timeframe.upper()
        timeframe = _TIMEFRAME_MAP.get(timeframe_name, mt5.TIMEFRAME_M15)
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        mt5.symbol_select(self.settings.symbol, True)
        symbol = mt5.symbol_info(self.settings.symbol)
        tick = mt5.symbol_info_tick(self.settings.symbol)
        rates = mt5.copy_rates_from_pos(self.settings.symbol, timeframe, 0, bars)
        latest_bar_time = None
        bars_received = 0
        timestamp_source = "unavailable"
        if rates is not None and len(rates):
            bars_received = int(len(rates))
            latest_bar_time = datetime.fromtimestamp(int(rates[-1]["time"]), tz=timezone.utc).isoformat()
            timestamp_source = "live_feed"
        tick_time = None
        if tick is not None and getattr(tick, "time", None):
            tick_time = datetime.fromtimestamp(int(tick.time), tz=timezone.utc).isoformat()
        return {
            "terminal_connected": bool(getattr(terminal, "connected", False)) if terminal is not None else False,
            "trade_allowed": bool(getattr(terminal, "trade_allowed", False)) if terminal is not None else False,
            "account_login": int(getattr(account, "login", 0) or 0) if account is not None else 0,
            "account_server": str(getattr(account, "server", "") or "") if account is not None else "",
            "symbol": str(self.settings.symbol),
            "symbol_selected": bool(getattr(symbol, "select", False)) if symbol is not None else False,
            "symbol_visible": bool(getattr(symbol, "visible", False)) if symbol is not None else False,
            "timeframe_requested": timeframe_name,
            "latest_tick_utc": tick_time or "",
            "latest_bar_utc": latest_bar_time or "",
            "bars_received": bars_received,
            "timestamp_source": timestamp_source,
            "mt5_last_error": mt5.last_error(),
        }
