from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas_ai.config.settings import load_settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.learning.trade_labeling import _simulate_trade_path
from atlas_ai.models import TradeSignal


BASELINE_DIR = PROJECT_ROOT / "docs" / "baselines"
EVENT_PATHS = [PROJECT_ROOT / "storage" / "events.jsonl", *sorted((PROJECT_ROOT / "storage" / "archive").glob("events_*.jsonl"))]
TIMING_TEXT = "timing discipline blocked trade"


@dataclass(frozen=True)
class Candidate:
    group: str
    timestamp: str
    strategy: str
    side: str
    confidence: float
    stop_loss_points: float
    take_profit_points: float
    message: str
    session: str
    regime: str
    location: str
    momentum: str
    source_file: str


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "timing_discipline_selectivity"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _context_value(meta: dict[str, Any], key: str) -> str:
    context = _as_dict(meta.get("context_snapshot")) or _as_dict(meta.get("context"))
    chart = _as_dict(meta.get("chart_features"))
    if key == "momentum":
        return str(context.get("momentum") or context.get("momentum_state") or chart.get("momentum_state") or "")
    if key == "location":
        return str(context.get("location") or context.get("location_state") or chart.get("location_state") or "")
    return str(context.get(key) or chart.get(key) or "")


def _extract_candidate(payload: dict[str, Any], source_file: Path) -> Candidate | None:
    meta = _as_dict(payload.get("meta"))
    message = str(payload.get("message") or meta.get("message") or "")
    timing = _as_dict(meta.get("timing_discipline"))
    timing_result = _as_dict(timing.get("structured_result")) or timing
    chosen = _as_dict(meta.get("chosen")) or _as_dict(meta.get("selected_candidate"))
    trace = _as_dict(meta.get("decision_trace"))
    if not chosen:
        chosen = _as_dict(trace.get("selected_candidate"))

    side = str(chosen.get("side") or meta.get("action") or "").lower()
    if side not in {"buy", "sell"}:
        return None

    timeframe = str(meta.get("timeframe") or trace.get("timeframe") or "").upper()
    if timeframe and timeframe != "M1":
        return None

    strategy = str(chosen.get("strategy") or "unknown")
    ts = _parse_dt(str(payload.get("time") or meta.get("timestamp") or ""))
    if ts is None:
        return None

    message_lower = message.lower()
    hard_timing_block = bool(timing.get("hard_block") or timing_result.get("passed") is False)
    timing_block = TIMING_TEXT in message_lower or (hard_timing_block and str(timing_result.get("check") or "") == "timing_discipline")
    executed = bool(
        meta.get("execution_request_created")
        or meta.get("trade_executed")
        or str(meta.get("action") or "").lower() in {"buy", "sell"}
        or message_lower.startswith("executed buy")
        or message_lower.startswith("executed sell")
    )

    if timing_block:
        group = "timing_blocked"
    elif executed:
        group = "timing_passed_executed"
    else:
        return None

    return Candidate(
        group=group,
        timestamp=ts.isoformat(),
        strategy=strategy,
        side=side,
        confidence=float(chosen.get("confidence") or 0.0),
        stop_loss_points=float(chosen.get("stop_loss_points") or 100.0),
        take_profit_points=float(chosen.get("take_profit_points") or 160.0),
        message=message,
        session=_context_value(meta, "session"),
        regime=_context_value(meta, "regime"),
        location=_context_value(meta, "location"),
        momentum=_context_value(meta, "momentum"),
        source_file=str(source_file.relative_to(PROJECT_ROOT)),
    )


def _load_candidates(start: datetime | None, end: datetime | None) -> list[Candidate]:
    seen: set[tuple[str, str, str, str, str]] = set()
    candidates: list[Candidate] = []
    for path in EVENT_PATHS:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "buy" not in line and "sell" not in line:
                    continue
                if TIMING_TEXT not in line.lower() and "execut" not in line.lower():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                candidate = _extract_candidate(payload, path)
                if candidate is None:
                    continue
                ts = _parse_dt(candidate.timestamp)
                if ts is None:
                    continue
                if start and ts < start:
                    continue
                if end and ts > end:
                    continue
                key = (
                    candidate.timestamp,
                    candidate.strategy,
                    candidate.side,
                    f"{candidate.confidence:.5f}",
                    candidate.message,
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
    return sorted(candidates, key=lambda item: item.timestamp)


def _nearest_signal_index(df: pd.DataFrame, timestamp: datetime) -> int | None:
    times = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True)).asi8
    target = int(pd.Timestamp(timestamp).tz_convert("UTC").timestamp())
    position = int(times.searchsorted(target, side="right")) - 1
    if position < 0 or position >= len(df) - 1:
        return None
    return position


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return float(ordered[int(k)])
    return float(ordered[lower] * (upper - k) + ordered[upper] * (k - lower))


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_r": 0.0,
            "median_r": 0.0,
            "exit_reasons": {},
            "hold_bars_distribution": {},
            "by_strategy": {},
        }
    wins = [float(row["r_multiple"]) for row in rows if float(row["r_multiple"]) > 0.0]
    losses = [abs(float(row["r_multiple"])) for row in rows if float(row["r_multiple"]) < 0.0]
    classified = len(wins) + len(losses)
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)
    r_values = [float(row["r_multiple"]) for row in rows]
    return {
        "count": len(rows),
        "wins": len(wins),
        "losses": len(losses),
        "flat_or_zero": len(rows) - classified,
        "win_rate": (len(wins) / classified) if classified else 0.0,
        "profit_factor": (sum(wins) / sum(losses)) if losses else (999.0 if wins else 0.0),
        "average_r": sum(r_values) / len(r_values),
        "median_r": _percentile(r_values, 0.5),
        "exit_reasons": dict(Counter(str(row["exit_type"]) for row in rows)),
        "hold_bars_distribution": dict(Counter(str(int(row["hold_bars"])) for row in rows)),
        "by_strategy": {
            strategy: {
                "count": len(strategy_rows),
                "win_rate": _summarize(strategy_rows)["win_rate"] if len(strategy_rows) != len(rows) else None,
                "average_r": sum(float(row["r_multiple"]) for row in strategy_rows) / len(strategy_rows),
                "exit_reasons": dict(Counter(str(row["exit_type"]) for row in strategy_rows)),
            }
            for strategy, strategy_rows in sorted(by_strategy.items())
        },
    }


def _simulate(candidates: list[Candidate], df: pd.DataFrame, settings: Any) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for candidate in candidates:
        ts = _parse_dt(candidate.timestamp)
        idx = _nearest_signal_index(df, ts) if ts else None
        if idx is None:
            skipped += 1
            continue
        signal = TradeSignal(
            strategy=candidate.strategy,
            side=candidate.side,  # type: ignore[arg-type]
            confidence=candidate.confidence,
            stop_loss_points=candidate.stop_loss_points,
            take_profit_points=candidate.take_profit_points,
        )
        outcome = _simulate_trade_path(df, signal, idx, settings)
        exit_idx = int(outcome.get("exit_idx", idx) or idx)
        entry_idx = min(idx + 1, len(df) - 1)
        rows.append(
            {
                **asdict(candidate),
                "signal_idx": idx,
                "signal_bar_time": pd.Timestamp(df.iloc[idx]["time"]).isoformat(),
                "entry_bar_time": pd.Timestamp(df.iloc[entry_idx]["time"]).isoformat(),
                "exit_bar_time": pd.Timestamp(df.iloc[min(exit_idx, len(df) - 1)]["time"]).isoformat(),
                "hold_bars": max(1, exit_idx - entry_idx + 1),
                "r_multiple": float(outcome.get("r_multiple") or 0.0),
                "mfe_r": float(outcome.get("mfe_r") or 0.0),
                "mae_r": float(outcome.get("mae_r") or 0.0),
                "exit_type": str(outcome.get("exit_type") or "unknown"),
            }
        )
    return rows, skipped


def run_report(start: datetime | None, end: datetime | None, label: str) -> Path:
    settings = replace(load_settings(), timeframe="M1", execute_trades=False)

    candidates = _load_candidates(start, end)
    if not candidates:
        raise SystemExit("No timing-discipline or executed directional candidates found in retained events.")

    first = _parse_dt(candidates[0].timestamp)
    last = _parse_dt(candidates[-1].timestamp)
    assert first is not None and last is not None
    fetch_start = start or first
    fetch_end = end or last
    feed = MarketDataFeed(settings)
    df = feed.get_rates_range(fetch_start, fetch_end)
    if df.empty:
        raise SystemExit("No M1 market data returned for candidate window.")

    blocked = [candidate for candidate in candidates if candidate.group == "timing_blocked"]
    controls = [candidate for candidate in candidates if candidate.group == "timing_passed_executed"]
    blocked_rows, blocked_skipped = _simulate(blocked, df, settings)
    control_rows, control_skipped = _simulate(controls, df, settings)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"baseline_{timestamp}_{_safe_label(label)}.json"
    payload = {
        "schema": "atlas_timing_discipline_selectivity_report_v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "window": {
            "start": fetch_start.isoformat(),
            "end": fetch_end.isoformat(),
            "timeframe": "M1",
            "symbol": settings.symbol,
            "bars": int(len(df)),
        },
        "method": {
            "source": "retained storage/events.jsonl and storage/archive/events_*.jsonl",
            "simulation": "atlas_ai.learning.trade_labeling._simulate_trade_path",
            "entry_assumption": "candidate signal bar maps to nearest retained M1 bar at or before event timestamp; hypothetical entry uses next bar open, matching backtest simulator",
            "control_group": "directional candidates in retained events that created/executed an execution request and were not timing-discipline blocked",
            "safety": "offline analysis only; execute_trades forced false; no strategy, risk, or execution policy changed",
        },
        "candidate_counts": {
            "timing_blocked_found": len(blocked),
            "timing_blocked_simulated": len(blocked_rows),
            "timing_blocked_skipped_no_price_bar": blocked_skipped,
            "timing_passed_executed_found": len(controls),
            "timing_passed_executed_simulated": len(control_rows),
            "timing_passed_executed_skipped_no_price_bar": control_skipped,
        },
        "summaries": {
            "timing_blocked_hypothetical": _summarize(blocked_rows),
            "timing_passed_executed_control": _summarize(control_rows),
        },
        "samples": {
            "timing_blocked_hypothetical": blocked_rows[:250],
            "timing_passed_executed_control": control_rows[:250],
        },
        "data_fetch": dict(getattr(feed, "last_fetch_meta", {}) or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay retained timing-discipline blocks as hypothetical trades.")
    parser.add_argument("--start", default="", help="Optional ISO UTC start timestamp.")
    parser.add_argument("--end", default="", help="Optional ISO UTC end timestamp.")
    parser.add_argument("--label", default="timing_discipline_selectivity", help="Artifact label.")
    args = parser.parse_args()
    start = _parse_dt(args.start) if args.start else None
    end = _parse_dt(args.end) if args.end else None
    path = run_report(start=start, end=end, label=args.label)
    print(path)


if __name__ == "__main__":
    main()
