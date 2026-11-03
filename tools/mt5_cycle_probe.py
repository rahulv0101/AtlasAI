from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.config.settings import load_settings


def main() -> None:
    settings = load_settings()
    rows: list[dict[str, Any]] = []

    def mark(name: str, fn: Callable[[], Any]) -> None:
        started = time.perf_counter()
        ok = True
        err = ""
        result: Any = None
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - diagnostic output needs the raw failure.
            ok = False
            err = repr(exc)
        rows.append(
            {
                "call": name,
                "ok": ok,
                "seconds": round(time.perf_counter() - started, 3),
                "result_type": type(result).__name__,
                "result_len": len(result) if hasattr(result, "__len__") else None,
                "error": err,
                "last_error": str(mt5.last_error()),
            }
        )

    mark(
        "initialize",
        lambda: mt5.initialize(path=settings.hfm_terminal_path)
        if settings.hfm_terminal_path
        else mt5.initialize(),
    )
    mark(
        "login",
        lambda: mt5.login(
            login=settings.hfm_login,
            password=settings.hfm_password,
            server=settings.hfm_server,
        ),
    )
    mark("account_info", mt5.account_info)
    mark("positions_get", lambda: mt5.positions_get(symbol=settings.symbol))
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=14)
    mark("history_deals_get_14d", lambda: mt5.history_deals_get(start, end))
    mark("symbol_info_tick", lambda: mt5.symbol_info_tick(settings.symbol))
    mark("symbol_info", lambda: mt5.symbol_info(settings.symbol))
    timeframe = getattr(mt5, f"TIMEFRAME_{settings.timeframe.upper()}")
    mark("copy_rates_from_pos_1200", lambda: mt5.copy_rates_from_pos(settings.symbol, timeframe, 0, 1200))
    print(json.dumps(rows, indent=2, default=str))
    mt5.shutdown()


if __name__ == "__main__":
    main()
