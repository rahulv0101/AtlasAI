from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_ai.backtest import Backtester
from atlas_ai.trader import AtlasTrader


BASELINE_SCHEMA = "atlas_behavior_baseline_v1"


def _fixed_window_df(rows: int = 900) -> pd.DataFrame:
    base = 2000.0
    data: list[dict[str, Any]] = []
    start = pd.Timestamp("2026-01-05T00:00:00+00:00")
    for i in range(rows):
        drift = 0.08 * i
        wave = ((i % 20) - 10) * 1.7
        open_px = base + drift + wave
        close_px = open_px + (1.8 if i % 3 else -1.2)
        high_px = max(open_px, close_px) + 1.5
        low_px = min(open_px, close_px) - 1.5
        data.append(
            {
                "time": start + pd.Timedelta(minutes=i),
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tick_volume": 120 + (i % 9) * 8,
            }
        )
    return pd.DataFrame(data)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
        backtest_max_hold_bars=4,
        backtest_point_size=0.01,
        backtest_evolved_refresh_bars=250,
        backtest_evolved_lookback_bars=500,
        backtest_include_evolved=False,
        backtest_record_loss_analysis=False,
        backtest_persist_learning_state=False,
        research_confidence_threshold=0.1,
        max_risk_per_trade=0.005,
        enabled_strategies=[],
        timeframe_minutes=1,
    )


def run_fixed_backtest(*, label: str, task: str, patch_timing_legacy: bool = False) -> dict[str, Any]:
    timing_calls = 0
    original_timing = AtlasTrader._timing_discipline_snapshot

    def legacy_timing(self: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal timing_calls
        timing_calls += 1
        payload = dict(original_timing(self, *args, **kwargs))
        for key in ("passed", "check", "reason_code", "detail", "condition_values", "checked_at", "structured_result"):
            payload.pop(key, None)
        return payload

    if patch_timing_legacy:
        AtlasTrader._timing_discipline_snapshot = legacy_timing  # type: ignore[method-assign]

    trades: list[dict[str, Any]] = []
    try:
        backtester = Backtester(settings=_settings())
        original_simulate: Callable[..., Any] = backtester._simulate_trade

        def traced_simulate(df: pd.DataFrame, signal: Any, entry_idx: int) -> Any:
            out = backtester._normalize_simulated_trade(original_simulate(df, signal, entry_idx))
            r_multiple, exit_idx, best_mfe_r = out
            trades.append(
                {
                    "entry_index": int(entry_idx),
                    "exit_index": int(exit_idx),
                    "entry_time": str(df.iloc[int(entry_idx)].get("time", "")),
                    "exit_time": str(df.iloc[int(exit_idx)].get("time", "")),
                    "strategy": str(getattr(signal, "strategy", "")),
                    "side": str(getattr(signal, "side", "")),
                    "confidence": round(float(getattr(signal, "confidence", 0.0) or 0.0), 6),
                    "r_multiple": round(float(r_multiple), 8),
                    "best_mfe_r": round(float(best_mfe_r), 8),
                }
            )
            return out

        backtester._simulate_trade = traced_simulate  # type: ignore[method-assign]
        result = backtester.run(_fixed_window_df())
    finally:
        if patch_timing_legacy:
            AtlasTrader._timing_discipline_snapshot = original_timing  # type: ignore[method-assign]

    return {
        "schema": BASELINE_SCHEMA,
        "task": task,
        "label": label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "source": "deterministic_synthetic_m1_ohlcv",
            "start": "2026-01-05T00:00:00+00:00",
            "rows": 900,
            "timeframe": "M1",
        },
        "settings": dict(vars(_settings())),
        "result": asdict(result),
        "trades": trades,
        "instrumentation": {
            "timing_discipline_snapshot_calls": timing_calls,
            "patch_timing_legacy": bool(patch_timing_legacy),
        },
    }


def _behavior_signature(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload.get("result", {}) or {})
    return {
        "result": {
            key: result.get(key)
            for key in (
                "total_return",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "expectancy",
                "profit_factor",
                "max_drawdown",
                "average_hold_bars",
            )
        },
        "trades": list(payload.get("trades", []) or []),
    }


def diff_baselines(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_sig = _behavior_signature(before)
    after_sig = _behavior_signature(after)
    return {
        "schema": "atlas_behavior_baseline_diff_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before_label": before.get("label"),
        "after_label": after.get("label"),
        "equal": before_sig == after_sig,
        "differences": {} if before_sig == after_sig else {"before": before_sig, "after": after_sig},
        "instrumentation": {
            "before": dict(before.get("instrumentation", {}) or {}),
            "after": dict(after.get("instrumentation", {}) or {}),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture/diff fixed-window Atlas behavior baselines.")
    parser.add_argument("--task", default="timing_discipline_transparency", help="Task label for output filenames")
    parser.add_argument("--output-dir", default="docs/baselines", help="Directory for baseline artifacts")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir)
    task = str(args.task).strip() or "behavior_change"

    before = run_fixed_backtest(label="reconstructed_before_structured_timing_output", task=task, patch_timing_legacy=True)
    after = run_fixed_backtest(label="current_after", task=task, patch_timing_legacy=False)
    diff = diff_baselines(before, after)

    before_path = out_dir / f"baseline_{stamp}_{task}_before.json"
    after_path = out_dir / f"baseline_{stamp}_{task}_after.json"
    diff_path = out_dir / f"baseline_{stamp}_{task}_diff.json"
    _write_json(before_path, before)
    _write_json(after_path, after)
    _write_json(diff_path, diff)

    print(json.dumps({"before": str(before_path), "after": str(after_path), "diff": str(diff_path), **diff}, indent=2))


if __name__ == "__main__":
    main()
