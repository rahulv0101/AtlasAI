from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SIDE_VALUE = {
    "buy": 1.0,
    "long": 1.0,
    "sell": -1.0,
    "short": -1.0,
    "flat": 0.0,
    "hold": 0.0,
    "none": 0.0,
    "": 0.0,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strategy_name(signal: dict[str, Any]) -> str:
    return str(signal.get("strategy") or signal.get("strategy_name") or signal.get("family") or "").strip()


def _side_value(signal: dict[str, Any]) -> float:
    raw = str(signal.get("side") or signal.get("direction") or signal.get("signal") or "").strip().lower()
    if raw in SIDE_VALUE:
        return SIDE_VALUE[raw]
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if value > 0:
        return 1.0
    if value < 0:
        return -1.0
    return 0.0


def _timestamp(row: dict[str, Any], fallback_idx: int) -> str:
    for key in ("timestamp", "time", "ts", "updated_at", "recorded_at"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return f"row_{fallback_idx:06d}"


def _signals_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = [
        row.get("signals"),
        row.get("signals_active"),
        dict(row.get("meta", {}) or {}).get("signals") if isinstance(row.get("meta"), dict) else None,
        dict(row.get("status", {}) or {}).get("signals") if isinstance(row.get("status"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [dict(item or {}) for item in candidate if isinstance(item, dict)]
    return []


def normalize_signal_rows(rows: list[dict[str, Any]], *, rolling_window: int = 500) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    source_rows = list(rows or [])[-max(1, int(rolling_window)) :]
    for idx, row in enumerate(source_rows):
        signals = _signals_from_row(dict(row or {}))
        values: dict[str, float] = {}
        for signal in signals:
            name = _strategy_name(signal)
            if name:
                values[name] = _side_value(signal)
        if values:
            normalized.append({"timestamp": _timestamp(dict(row or {}), idx), "signals": values})
    return normalized


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den_x = math.sqrt(sum(x * x for x in dx))
    den_y = math.sqrt(sum(y * y for y in dy))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (den_x * den_y)


def _agreement(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(1 for x, y in zip(xs, ys) if x == y) / len(xs)


def build_signal_correlation_report(
    rows: list[dict[str, Any]],
    *,
    rolling_window: int = 500,
    threshold: float = 0.8,
) -> dict[str, Any]:
    series_rows = normalize_signal_rows(rows, rolling_window=rolling_window)
    strategies = sorted({name for row in series_rows for name in dict(row.get("signals", {}) or {})})
    by_strategy: dict[str, list[float]] = {name: [] for name in strategies}
    for row in series_rows:
        values = dict(row.get("signals", {}) or {})
        for name in strategies:
            by_strategy[name].append(float(values.get(name, 0.0)))

    matrix: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in strategies}
    flagged: list[dict[str, Any]] = []
    for left in strategies:
        for right in strategies:
            xs = by_strategy[left]
            ys = by_strategy[right]
            pearson = _pearson(xs, ys)
            agreement = _agreement(xs, ys)
            cell = {
                "pearson": round(float(pearson), 6) if pearson is not None else None,
                "agreement_rate": round(float(agreement), 6),
                "samples": len(xs),
            }
            matrix[left][right] = cell
            if left >= right:
                continue
            redundant = (pearson is not None and abs(float(pearson)) >= threshold) or agreement >= threshold
            if redundant:
                flagged.append(
                    {
                        "strategy_a": left,
                        "strategy_b": right,
                        "pearson": cell["pearson"],
                        "agreement_rate": cell["agreement_rate"],
                        "samples": cell["samples"],
                        "threshold": float(threshold),
                    }
                )
    flagged.sort(key=lambda row: (-(abs(float(row.get("pearson") or 0.0))), -float(row.get("agreement_rate", 0.0)), str(row.get("strategy_a"))))
    return {
        "schema": "atlas_signal_correlation_report_v1",
        "generated_at": _now_iso(),
        "read_only": True,
        "rolling_window": int(rolling_window),
        "threshold": float(threshold),
        "source_rows": len(rows or []),
        "usable_rows": len(series_rows),
        "strategies": strategies,
        "matrix": matrix,
        "flagged_pairs": flagged,
        "summary": summarize_signal_correlation_report(flagged, strategy_count=len(strategies), usable_rows=len(series_rows)),
        "policy_note": "Read-only redundancy measurement; ranking weights are not changed automatically.",
    }


def summarize_signal_correlation_report(flagged_pairs: list[dict[str, Any]], *, strategy_count: int, usable_rows: int) -> str:
    if strategy_count < 2:
        return f"Not enough strategy families to measure redundancy ({strategy_count} found across {usable_rows} usable rows)."
    if not flagged_pairs:
        return f"No strategy pairs exceeded the redundancy threshold across {strategy_count} strategies and {usable_rows} usable rows."
    top = flagged_pairs[0]
    return (
        f"{len(flagged_pairs)} potentially redundant strategy pairs found. "
        f"Top pair: {top['strategy_a']} vs {top['strategy_b']} "
        f"(pearson={top.get('pearson')}, agreement={top.get('agreement_rate')})."
    )


def signal_correlation_matrix_csv(report: dict[str, Any]) -> str:
    strategies = list(report.get("strategies", []) or [])
    matrix = dict(report.get("matrix", {}) or {})
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["strategy", *strategies])
    for left in strategies:
        row = [left]
        for right in strategies:
            cell = dict(dict(matrix.get(left, {}) or {}).get(right, {}) or {})
            value = cell.get("pearson")
            row.append("" if value is None else value)
        writer.writerow(row)
    return buf.getvalue()


def load_signal_rows_from_store(store: Any, *, rolling_window: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        status = dict(store.load_status() or {})
        if status:
            rows.append(status)
    except Exception:
        pass
    try:
        events = list(store.tail_events(limit=max(1, int(rolling_window))) or [])
        rows.extend(dict(row or {}) for row in events if isinstance(row, dict))
    except Exception:
        path = Path(getattr(store, "events_file", Path(getattr(store, "base", "storage")) / "events.jsonl"))
        if path.exists():
            from atlas_ai.memory.jsonl_tail import read_jsonl_tail

            rows.extend(read_jsonl_tail(path, rolling_window))
    return rows[-max(1, int(rolling_window)) :]


def build_signal_correlation_report_from_store(
    store: Any,
    *,
    rolling_window: int = 500,
    threshold: float = 0.8,
    write_files: bool = True,
) -> dict[str, Any]:
    report = build_signal_correlation_report(
        load_signal_rows_from_store(store, rolling_window=rolling_window),
        rolling_window=rolling_window,
        threshold=threshold,
    )
    if write_files:
        base = Path(getattr(store, "base", "storage"))
        base.mkdir(parents=True, exist_ok=True)
        (base / "signal_correlation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        (base / "signal_correlation_matrix.csv").write_text(signal_correlation_matrix_csv(report), encoding="utf-8")
    return report
