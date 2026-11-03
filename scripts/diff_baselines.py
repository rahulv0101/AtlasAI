from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "total_return",
    "trade_count",
    "wins",
    "losses",
    "win_rate",
    "expectancy",
    "profit_factor",
    "max_drawdown",
    "sharpe",
    "sortino",
    "average_win",
    "average_loss",
    "risk_of_ruin",
    "average_hold_bars",
]


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    return value


def diff_baselines(before_path: str, after_path: str) -> dict[str, Any]:
    before = _load(before_path)
    after = _load(after_path)
    before_metrics = dict((before.get("metrics") or {}))
    after_metrics = dict((after.get("metrics") or {}))

    metric_diffs = {}
    for key in METRIC_KEYS:
        before_value = _round_float(before_metrics.get(key))
        after_value = _round_float(after_metrics.get(key))
        if before_value != after_value:
            delta = None
            if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
                delta = _round_float(after_value - before_value)
            metric_diffs[key] = {
                "before": before_value,
                "after": after_value,
                "delta": delta,
            }

    before_trades = before_metrics.get("trade_list") or []
    after_trades = after_metrics.get("trade_list") or []
    trade_list_changed = before_trades != after_trades
    trade_diff = {
        "changed": bool(trade_list_changed),
        "before_count": len(before_trades),
        "after_count": len(after_trades),
    }
    if trade_list_changed:
        first_changed = None
        for idx, (before_trade, after_trade) in enumerate(zip(before_trades, after_trades)):
            if before_trade != after_trade:
                first_changed = {
                    "index": idx,
                    "before": before_trade,
                    "after": after_trade,
                }
                break
        if first_changed is None and len(before_trades) != len(after_trades):
            first_changed = {
                "index": min(len(before_trades), len(after_trades)),
                "before": before_trades[min(len(before_trades), len(after_trades)) :] or None,
                "after": after_trades[min(len(before_trades), len(after_trades)) :] or None,
            }
        trade_diff["first_changed"] = first_changed

    return {
        "before": str(Path(before_path)),
        "after": str(Path(after_path)),
        "window_changed": before.get("window") != after.get("window"),
        "metrics_changed": bool(metric_diffs),
        "metric_diffs": metric_diffs,
        "trade_list_diff": trade_diff,
        "behavior_changed": bool(metric_diffs or trade_list_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two Atlas baseline JSON files.")
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    result = diff_baselines(args.before, args.after)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1 if result["behavior_changed"] else 0)


if __name__ == "__main__":
    main()
