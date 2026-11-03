from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


FORWARD_SAFE_FEATURE_WEIGHTS: dict[str, float] = {
    "late_us_trend": -20.0,
    "london_drag": -16.0,
    "low_vol_trend": -14.0,
    "post_spike": -16.0,
    "repeated_reentry": -10.0,
    "poor_liquidity": -18.0,
    "high_volatility_range": -5.0,
    "favorable_session": 3.0,
    "range_regime": 4.0,
    "mid_trend_strength": 2.0,
    "strong_trend_overstretch": -4.0,
}

LOW_SCORE_CUTOFF = 74.0


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _trend_strength_bucket(value: float) -> str:
    if value < 2.0:
        return "very_weak"
    if value < 4.0:
        return "weak"
    if value < 8.0:
        return "medium"
    return "strong"


def _score_bucket(score: float) -> str:
    if score >= 92.0:
        return "high"
    if score >= LOW_SCORE_CUTOFF:
        return "mid"
    return "low"


def score_forward_safe_setup(record: dict[str, Any]) -> dict[str, Any]:
    session = str(record.get("session", "unknown") or "unknown")
    regime = str(record.get("regime", "unknown") or "unknown")
    volatility_state = str(record.get("volatility_bucket", "unknown") or "unknown")
    trend_strength = _safe_float(record.get("trend_strength_ratio", 0.0))
    volume_ratio = _safe_float(record.get("volume_ratio", 1.0), 1.0)
    reason_codes = {str(item) for item in list(record.get("reason_codes", []) or [])}

    score = 100.0
    contributions: dict[str, float] = {}
    reasons: list[str] = []

    def apply(name: str) -> None:
        nonlocal score
        if name in contributions:
            return
        weight = float(FORWARD_SAFE_FEATURE_WEIGHTS[name])
        contributions[name] = round(weight, 4)
        reasons.append(name)
        score += weight

    if session == "late_us" and regime == "trend":
        apply("late_us_trend")
    if "london_drag" in reason_codes or bool(record.get("london_extreme_volatility", False)):
        apply("london_drag")
    if "low_vol_trend" in reason_codes or (regime == "trend" and volatility_state == "low_volatility"):
        apply("low_vol_trend")
    if bool(record.get("post_spike_context", False)) or "post_spike" in reason_codes:
        apply("post_spike")
    if bool(record.get("repeated_reentry", False)) or bool(record.get("repeat_entry", False)):
        apply("repeated_reentry")
    if (
        bool(record.get("poor_liquidity", False))
        or "poor_liquidity" in reason_codes
        or volume_ratio < 0.8
    ):
        apply("poor_liquidity")
    if regime == "high_volatility_range":
        apply("high_volatility_range")

    if session in {"asia", "new_york"}:
        apply("favorable_session")
    if regime == "range":
        apply("range_regime")
    if 4.0 <= trend_strength < 8.0:
        apply("mid_trend_strength")
    if trend_strength >= 8.0 and bool(record.get("post_spike_context", False)):
        apply("strong_trend_overstretch")

    score = _clamp(score, 0.0, 100.0)
    return {
        "setup_quality_score": round(score, 4),
        "setup_quality_bucket": _score_bucket(score),
        "reason_contributions": contributions,
        "reasons": reasons,
    }


def score_live_forward_safe_setup(
    *,
    session: str,
    regime: str,
    trend_strength: float,
    volatility_pct: float,
    volume_ratio: float,
    spike_ratio: float,
    same_side_support: int,
    setup_quality_reasons: list[str] | None = None,
) -> dict[str, Any]:
    score = 100.0
    contributions: dict[str, float] = {}
    reasons: list[str] = []
    setup_reason_set = {str(item) for item in list(setup_quality_reasons or [])}

    def apply(name: str) -> None:
        nonlocal score
        if name in contributions:
            return
        weight = float(FORWARD_SAFE_FEATURE_WEIGHTS[name])
        contributions[name] = round(weight, 4)
        reasons.append(name)
        score += weight

    session_key = str(session or "unknown").lower()
    regime_key = str(regime or "unknown").lower()

    if session_key == "late_us" and regime_key == "trend":
        apply("late_us_trend")
    if session_key == "london" and (volatility_pct < 0.03 or volatility_pct >= 0.07):
        apply("london_drag")
    if "london_extreme_volatility" in setup_reason_set:
        apply("london_drag")
    if regime_key == "trend" and volatility_pct < 0.03:
        apply("low_vol_trend")
    if "trend_low_volatility" in setup_reason_set:
        apply("low_vol_trend")
    if spike_ratio >= 1.6 or "post_spike_entry" in setup_reason_set:
        apply("post_spike")
    if volume_ratio < 0.8 or "thin_liquidity" in setup_reason_set:
        apply("poor_liquidity")
    if same_side_support >= 3 and spike_ratio >= 1.2:
        apply("repeated_reentry")
    if regime_key == "high_volatility_range":
        apply("high_volatility_range")

    if session_key in {"asia", "new_york"}:
        apply("favorable_session")
    if regime_key == "range":
        apply("range_regime")
    if 4.0 <= float(trend_strength) < 8.0:
        apply("mid_trend_strength")
    if float(trend_strength) >= 8.0 and spike_ratio >= 1.6:
        apply("strong_trend_overstretch")

    score = _clamp(score, 0.0, 100.0)
    return {
        "setup_quality_score": round(score, 4),
        "setup_quality_bucket": _score_bucket(score),
        "reason_contributions": contributions,
        "reasons": reasons,
    }


def _max_drawdown(records: list[dict[str, Any]]) -> float:
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for record in records:
        equity *= 1.0 + _safe_float(record.get("trade_return", 0.0))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / max(peak, 1e-9))
    return round(max_dd, 6)


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "max_drawdown": 0.0,
        }
    wins = [_safe_float(r.get("trade_return", 0.0)) for r in records if _safe_float(r.get("trade_return", 0.0)) > 0.0]
    losses = [_safe_float(r.get("trade_return", 0.0)) for r in records if _safe_float(r.get("trade_return", 0.0)) < 0.0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(records),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / max(1, len(records)), 6),
        "expectancy": round(sum(_safe_float(r.get("trade_return", 0.0)) for r in records) / max(1, len(records)), 6),
        "profit_factor": round(gross_profit / max(gross_loss, 1e-9), 6),
        "average_win": round(sum(wins) / max(1, len(wins)), 6),
        "average_loss": round(sum(losses) / max(1, len(losses)), 6),
        "max_drawdown": _max_drawdown(records),
    }


def _bucket_breakdown(records: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get(field, "unknown") or "unknown")
        grouped.setdefault(key, []).append(record)
    out: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        out[key] = _metrics(rows)
    return out


def _flag_stats(records: list[dict[str, Any]], name: str, predicate) -> dict[str, Any]:
    rows = [record for record in records if predicate(record)]
    stats = _metrics(rows)
    stats["name"] = name
    return stats


def _top_loss_buckets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        _flag_stats(records, "poor_liquidity", lambda r: bool(r.get("poor_liquidity", False)) or "poor_liquidity" in set(r.get("reason_codes", []) or []) or _safe_float(r.get("volume_ratio", 1.0), 1.0) < 0.8),
        _flag_stats(records, "london_drag", lambda r: "london_drag" in set(r.get("reason_codes", []) or []) or bool(r.get("london_extreme_volatility", False))),
        _flag_stats(records, "low_vol_trend", lambda r: "low_vol_trend" in set(r.get("reason_codes", []) or []) or (str(r.get("regime", "")) == "trend" and str(r.get("volatility_bucket", "")) == "low_volatility")),
        _flag_stats(records, "late_us_trend", lambda r: str(r.get("session", "")) == "late_us" and str(r.get("regime", "")) == "trend"),
        _flag_stats(records, "post_spike", lambda r: bool(r.get("post_spike_context", False)) or "post_spike" in set(r.get("reason_codes", []) or [])),
        _flag_stats(records, "repeated_reentry", lambda r: bool(r.get("repeated_reentry", False)) or bool(r.get("repeat_entry", False))),
    ]
    ranked = [
        item for item in candidates
        if int(item.get("trades", 0)) >= 15
    ]
    ranked.sort(key=lambda item: (float(item.get("expectancy", 0.0)), float(item.get("win_rate", 0.0))))
    return ranked[:5]


def build_edge_building_report(
    label_report_path: str | Path,
    output_path: str | Path | None = None,
    cutoff_score: float = LOW_SCORE_CUTOFF,
) -> dict[str, Any]:
    label_report = _load_json(label_report_path)
    if not bool(label_report.get("ok", False)):
        return {"ok": False, "reason": "Trade label report is missing or invalid."}

    records = [dict(item) for item in list(label_report.get("records", []) or [])]
    if not records:
        return {"ok": False, "reason": "No labeled trade records found."}

    for record in records:
        record.update(score_forward_safe_setup(record))
        raw_entry = str(record.get("entry_time", "") or "")
        record["_entry_dt"] = datetime.fromisoformat(raw_entry) if raw_entry else datetime.min
        record["trend_strength_bucket"] = _trend_strength_bucket(_safe_float(record.get("trend_strength_ratio", 0.0)))

    records.sort(key=lambda item: item["_entry_dt"])

    bands = {
        "high": [row for row in records if str(row.get("setup_quality_bucket", "")) == "high"],
        "mid": [row for row in records if str(row.get("setup_quality_bucket", "")) == "mid"],
        "low": [row for row in records if str(row.get("setup_quality_bucket", "")) == "low"],
    }

    filtered = [row for row in records if _safe_float(row.get("setup_quality_score", 0.0)) >= cutoff_score]
    removed = [row for row in records if _safe_float(row.get("setup_quality_score", 0.0)) < cutoff_score]

    winners_removed = sum(1 for row in removed if _safe_float(row.get("trade_return", 0.0)) > 0.0)
    losers_removed = sum(1 for row in removed if _safe_float(row.get("trade_return", 0.0)) < 0.0)

    mid = len(records) // 2
    first_half = records[:mid]
    second_half = records[mid:]

    removed_reason_counts: Counter[str] = Counter()
    for row in removed:
        removed_reason_counts.update(list(row.get("reasons", []) or []))

    result = {
        "ok": True,
        "source_label_report": str(label_report_path),
        "scoring_mode": "forward_safe_setup_quality_v1",
        "feature_weights": {k: round(float(v), 4) for k, v in FORWARD_SAFE_FEATURE_WEIGHTS.items()},
        "control_metrics": _metrics(records),
        "loss_breakdown": {
            "by_session": _bucket_breakdown(records, "session"),
            "by_regime": _bucket_breakdown(records, "regime"),
            "by_trend_strength": _bucket_breakdown(records, "trend_strength_bucket"),
            "by_volatility_state": _bucket_breakdown(records, "volatility_bucket"),
            "repeated_entry": {
                "repeat": _metrics([row for row in records if bool(row.get("repeat_entry", False))]),
                "first": _metrics([row for row in records if not bool(row.get("repeat_entry", False))]),
            },
            "overstretched_entry": {
                "post_spike": _metrics([row for row in records if bool(row.get("post_spike_context", False))]),
                "not_post_spike": _metrics([row for row in records if not bool(row.get("post_spike_context", False))]),
            },
            "top_loss_buckets": _top_loss_buckets(records),
        },
        "score_validation": {
            "bands": {name: _metrics(rows) for name, rows in bands.items()},
            "cutoff_score": round(float(cutoff_score), 4),
        },
        "filter_experiment": {
            "cutoff_score": round(float(cutoff_score), 4),
            "control": _metrics(records),
            "experiment": _metrics(filtered),
            "removed": _metrics(removed),
            "trades_removed": len(removed),
            "winners_removed": winners_removed,
            "losers_removed": losers_removed,
            "filter_precision": round(losers_removed / max(1, losers_removed + winners_removed), 6),
            "removed_reason_counts": dict(removed_reason_counts.most_common()),
        },
        "stability": {
            "first_half": {
                "control": _metrics(first_half),
                "experiment": _metrics([row for row in first_half if _safe_float(row.get("setup_quality_score", 0.0)) >= cutoff_score]),
            },
            "second_half": {
                "control": _metrics(second_half),
                "experiment": _metrics([row for row in second_half if _safe_float(row.get("setup_quality_score", 0.0)) >= cutoff_score]),
            },
        },
        "readiness": {
            "ready_for_research_filtering": True,
            "ready_for_live_trading_integration": False,
            "notes": [
                "Score uses only pre-entry/context fields and ignores realized outcome when assigning per-trade quality.",
                "This remains a research-only ranking/filtering layer until it survives broader out-of-sample testing.",
            ],
        },
    }

    output_file = Path(output_path or Path(label_report_path).with_name("edge_building_report.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
