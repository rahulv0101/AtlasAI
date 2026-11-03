from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import load_settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.memory.knowledge_store import KnowledgeStore


BASELINE_DIR = PROJECT_ROOT / "docs" / "baselines"
DEFAULT_START = "2026-08-03T00:00:00+00:00"
DEFAULT_END = "2026-08-07T23:59:59+00:00"


def _parse_dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "baseline"


def _result_payload(result: Any) -> dict[str, Any]:
    trade_list = list(getattr(result, "trade_list", None) or [])
    per_strategy: dict[str, dict[str, Any]] = {}
    for trade in trade_list:
        strategy = str(trade.get("strategy", "unknown") or "unknown")
        row = per_strategy.setdefault(
            strategy,
            {
                "trade_count": 0,
                "wins": 0,
                "losses": 0,
                "total_return_fraction": 0.0,
                "average_return_fraction": 0.0,
                "win_rate": 0.0,
                "average_hold_bars": 0.0,
            },
        )
        row["trade_count"] += 1
        ret = float(trade.get("return_fraction", 0.0) or 0.0)
        row["total_return_fraction"] += ret
        if ret > 0.0:
            row["wins"] += 1
        elif ret < 0.0:
            row["losses"] += 1
        row["average_hold_bars"] += int(trade.get("hold_bars", 0) or 0)
    for row in per_strategy.values():
        count = max(1, int(row["trade_count"]))
        classified = int(row["wins"]) + int(row["losses"])
        row["average_return_fraction"] = float(row["total_return_fraction"]) / count
        row["win_rate"] = (float(row["wins"]) / classified) if classified else 0.0
        row["average_hold_bars"] = float(row["average_hold_bars"]) / count
    return {
        "total_return": float(getattr(result, "total_return", 0.0)),
        "trade_count": int(getattr(result, "trades", 0)),
        "wins": int(getattr(result, "wins", 0)),
        "losses": int(getattr(result, "losses", 0)),
        "win_rate": float(getattr(result, "win_rate", 0.0)),
        "expectancy": float(getattr(result, "expectancy", 0.0)),
        "profit_factor": float(getattr(result, "profit_factor", 0.0)),
        "max_drawdown": float(getattr(result, "max_drawdown", 0.0)),
        "sharpe": float(getattr(result, "sharpe", 0.0)),
        "sortino": float(getattr(result, "sortino", 0.0)),
        "average_win": float(getattr(result, "average_win", 0.0)),
        "average_loss": float(getattr(result, "average_loss", 0.0)),
        "risk_of_ruin": float(getattr(result, "risk_of_ruin", 1.0)),
        "average_hold_bars": float(getattr(result, "average_hold_bars", 0.0)),
        "per_strategy_contribution": per_strategy,
        "trade_list": trade_list,
    }


def capture_baseline(label: str, start: datetime, end: datetime, timeframe: str | None) -> Path:
    settings = load_settings()
    if timeframe:
        settings = replace(settings, timeframe=timeframe.upper())

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = _safe_label(label)
    output_path = BASELINE_DIR / f"baseline_{timestamp}_{safe_label}.json"

    with tempfile.TemporaryDirectory(prefix="atlas_baseline_store_") as tmp:
        baseline_settings = replace(
            settings,
            execute_trades=False,
            storage_dir=tmp,
            backtest_persist_learning_state=False,
        )
        feed = MarketDataFeed(baseline_settings)
        df = feed.get_rates_range(start, end)
        result = Backtester(store=KnowledgeStore(tmp), settings=baseline_settings).run(df)

    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": safe_label,
        "window": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": baseline_settings.timeframe,
            "symbol": baseline_settings.symbol,
            "bars": int(len(df)),
        },
        "configuration": {
            "enabled_strategies": list(getattr(baseline_settings, "enabled_strategies", ()) or ()),
            "disabled_strategies": list(getattr(baseline_settings, "disabled_strategies", ()) or ()),
            "backtest_include_evolved": bool(getattr(baseline_settings, "backtest_include_evolved", False)),
        },
        "safety": {
            "execute_trades": bool(baseline_settings.execute_trades),
            "storage_dir": "isolated_temp_store",
            "note": "Backtest capture only; no live execution path is invoked.",
        },
        "data_fetch": dict(getattr(feed, "last_fetch_meta", {}) or {}),
        "metrics": _result_payload(result),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a fixed-window Atlas backtest baseline.")
    parser.add_argument("--label", required=True, help="Short task label, e.g. before_m15_recalibration.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"ISO start timestamp. Default: {DEFAULT_START}")
    parser.add_argument("--end", default=DEFAULT_END, help=f"ISO end timestamp. Default: {DEFAULT_END}")
    parser.add_argument("--timeframe", default=os.getenv("BASELINE_TIMEFRAME", ""), help="Optional timeframe override.")
    args = parser.parse_args()

    path = capture_baseline(
        label=args.label,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        timeframe=args.timeframe.strip() or None,
    )
    print(str(path))


if __name__ == "__main__":
    main()
