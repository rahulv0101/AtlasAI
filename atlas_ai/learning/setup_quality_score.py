from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABEL_BASE_SCORES: dict[str, float] = {
    "high-quality winner": 0.65,
    "acceptable trade": 0.0,
    "avoidable loser": -0.55,
    "low-quality setup": -0.75,
}

REASON_WEIGHTS: dict[str, float] = {
    "first_entry_quality": 0.14,
    "strong_trend_clean_entry": 0.18,
    "favorable_session": 0.10,
    "high_quality_confirmation": 0.16,
    "late_us_trend": -0.22,
    "post_spike": -0.28,
    "london_drag": -0.18,
    "low_vol_trend": -0.16,
    "poor_liquidity": -0.22,
    "weak_confirmation": -0.12,
    "repeated_reentry": -0.17,
    "mixed_context": -0.04,
    "neutral_context": 0.0,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_bucket(score: float) -> str:
    if score >= 0.45:
        return "high_quality"
    if score >= 0.10:
        return "good"
    if score > -0.15:
        return "neutral"
    if score > -0.45:
        return "weak"
    return "avoid"


def _trade_ref(row: dict[str, Any], idx: int) -> str:
    strategy = str(row.get("strategy", "unknown")).split("(", 1)[0].strip()
    side = str(row.get("side", "flat"))
    entry = str(row.get("entry_time", ""))
    return f"T{idx:04d}:{strategy}:{side}:{entry}"


def score_trade_record(row: dict[str, Any]) -> dict[str, Any]:
    label = str(row.get("label", "acceptable trade"))
    reasons = list(row.get("reason_codes", []) or [])
    base_score = float(LABEL_BASE_SCORES.get(label, 0.0))
    reason_contribs: dict[str, float] = {}
    for reason in reasons:
        weight = float(REASON_WEIGHTS.get(str(reason), 0.0))
        if weight:
            reason_contribs[str(reason)] = weight

    score = base_score + sum(reason_contribs.values())

    # Keep the neutral bucket deliberately compressed so it doesn't dominate learning.
    if label == "acceptable trade":
        score *= 0.35

    # Preserve outcome-independent downgrades for low-quality context.
    if label == "low-quality setup":
        score = min(score, -0.35)
    elif label == "avoidable loser":
        score = min(score, -0.25)
    elif label == "high-quality winner":
        score = max(score, 0.35)

    score = _clamp(score, -1.0, 1.0)
    bucket = _score_bucket(score)
    return {
        "setup_quality_score": round(float(score), 4),
        "setup_quality_bucket": bucket,
        "base_label_score": round(float(base_score), 4),
        "reason_contributions": {k: round(float(v), 4) for k, v in reason_contribs.items()},
    }


def _summary_by_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["setup_quality_bucket"]), []).append(row)
    for bucket, rows in grouped.items():
        wins = [float(r["trade_return"]) for r in rows if float(r["trade_return"]) > 0]
        losses = [float(r["trade_return"]) for r in rows if float(r["trade_return"]) < 0]
        out[bucket] = {
            "count": len(rows),
            "win_rate": (len(wins) / max(1, len(rows))),
            "expectancy": (sum(float(r["trade_return"]) for r in rows) / max(1, len(rows))),
            "average_win": (sum(wins) / len(wins)) if wins else 0.0,
            "average_loss": (sum(losses) / len(losses)) if losses else 0.0,
        }
    return out


def build_offline_setup_quality_score(
    label_report_path: str | Path,
    review_report_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    label_report = _load_json(label_report_path)
    if not bool(label_report.get("ok", False)):
        return {"ok": False, "reason": "Trade label report is missing or invalid."}

    review = _load_json(review_report_path) if review_report_path and Path(review_report_path).exists() else {}
    records = list(label_report.get("records", []) or [])
    if not records:
        return {"ok": False, "reason": "No labeled trade records found."}

    scored_records: list[dict[str, Any]] = []
    for idx, row in enumerate(records, start=1):
        enriched = dict(row)
        enriched["trade_ref"] = _trade_ref(row, idx)
        enriched.update(score_trade_record(row))
        scored_records.append(enriched)

    summary_by_label: dict[str, Any] = {}
    grouped_by_label: dict[str, list[dict[str, Any]]] = {}
    for row in scored_records:
        grouped_by_label.setdefault(str(row["label"]), []).append(row)
    for label, rows in grouped_by_label.items():
        scores = [float(r["setup_quality_score"]) for r in rows]
        summary_by_label[label] = {
            "count": len(rows),
            "average_score": sum(scores) / max(1, len(scores)),
            "score_bucket_distribution": {
                bucket: sum(1 for r in rows if str(r["setup_quality_bucket"]) == bucket)
                for bucket in ["high_quality", "good", "neutral", "weak", "avoid"]
            },
        }

    ranked_signals: list[dict[str, Any]] = []
    for reason, weight in sorted(REASON_WEIGHTS.items(), key=lambda item: abs(item[1]), reverse=True):
        if reason in {"mixed_context", "neutral_context"}:
            continue
        ranked_signals.append(
            {
                "reason_code": reason,
                "weight": round(float(weight), 4),
                "direction": "positive" if weight > 0 else "negative",
            }
        )

    result = {
        "ok": True,
        "source_label_report": str(label_report_path),
        "source_review_report": str(review_report_path) if review_report_path else "",
        "scoring_mode": "offline_hybrid_v1",
        "label_weight_policy": {
            "high-quality winner": LABEL_BASE_SCORES["high-quality winner"],
            "acceptable trade": "near-neutral and compressed",
            "avoidable loser": LABEL_BASE_SCORES["avoidable loser"],
            "low-quality setup": LABEL_BASE_SCORES["low-quality setup"],
        },
        "reason_weights": {k: round(float(v), 4) for k, v in REASON_WEIGHTS.items()},
        "summary_by_label": summary_by_label,
        "summary_by_score_bucket": _summary_by_bucket(scored_records),
        "top_quality_signals": ranked_signals,
        "readiness": {
            "ready_for_offline_research_scoring": True,
            "ready_for_trading_integration": False,
            "notes": [
                "Acceptable trade remains too broad to weight heavily.",
                "Low-quality setup should remain an outcome-independent downgrade.",
                "Use this as a research ranking target before any trading integration.",
            ],
            "review_scoring_readiness": str(
                ((review.get("recommendations", {}) or {}).get("scoring_readiness", ""))
            ),
        },
        "records": scored_records,
    }

    output_file = Path(output_path or Path(label_report_path).with_name("trade_setup_quality_report.json"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
