from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABEL_ORDER = [
    "high-quality winner",
    "acceptable trade",
    "avoidable loser",
    "low-quality setup",
]


def _load_label_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _trade_ref(row: dict[str, Any], idx: int) -> str:
    strategy = str(row.get("strategy", "unknown")).split("(", 1)[0].strip()
    side = str(row.get("side", "flat"))
    entry = str(row.get("entry_time", ""))
    return f"T{idx:04d}:{strategy}:{side}:{entry}"


def _entry_type(row: dict[str, Any]) -> str:
    if bool(row.get("repeated_reentry", False)):
        return "repeated_reentry"
    if bool(row.get("first_entry_quality", False)):
        return "first_entry_quality"
    if bool(row.get("post_spike_context", False)):
        return "late_post_spike_entry"
    return "standard_entry"


def _example_view(row: dict[str, Any], idx: int) -> dict[str, Any]:
    explanation = (
        f"Labeled {row.get('label')} because reasons={list(row.get('reason_codes', [])) or ['none']} "
        f"with outcome={row.get('outcome')} in {row.get('session')} / {row.get('regime')}."
    )
    return {
        "trade_ref": _trade_ref(row, idx),
        "symbol": row.get("symbol"),
        "timestamp": row.get("entry_time"),
        "session": row.get("session"),
        "regime": row.get("regime"),
        "trend_strength_ratio": row.get("trend_strength_ratio"),
        "volatility_state": row.get("volatility_bucket"),
        "confidence": row.get("confidence"),
        "entry_type": _entry_type(row),
        "direction": row.get("side"),
        "holding_time_bars": row.get("hold_bars"),
        "outcome": row.get("outcome"),
        "trade_return": row.get("trade_return"),
        "exit_type": row.get("exit_type"),
        "reason_codes": row.get("reason_codes", []),
        "label_confidence": row.get("label_confidence"),
        "explanation": explanation,
    }


def _typical_examples(rows: list[tuple[int, dict[str, Any]]], label: str) -> list[dict[str, Any]]:
    ordered = rows
    if label == "high-quality winner":
        ordered = sorted(rows, key=lambda item: (-len(item[1].get("reason_codes", [])), -float(item[1].get("trade_return", 0.0))))
    elif label == "acceptable trade":
        ordered = sorted(rows, key=lambda item: (abs(float(item[1].get("trade_return", 0.0))), len(item[1].get("reason_codes", []))))
    elif label == "avoidable loser":
        ordered = sorted(rows, key=lambda item: (-len(item[1].get("reason_codes", [])), float(item[1].get("trade_return", 0.0))))
    elif label == "low-quality setup":
        ordered = sorted(rows, key=lambda item: (-len(item[1].get("reason_codes", [])), float(item[1].get("trade_return", 0.0))))
    return [_example_view(row, idx) for idx, row in ordered[:3]]


def _borderline_examples(rows: list[tuple[int, dict[str, Any]]], label: str) -> list[dict[str, Any]]:
    ordered = rows
    if label == "high-quality winner":
        ordered = sorted(rows, key=lambda item: (len(item[1].get("reason_codes", [])), float(item[1].get("trade_return", 0.0))))
    elif label == "acceptable trade":
        ordered = [item for item in rows if "mixed_context" in list(item[1].get("reason_codes", []))]
        ordered = sorted(ordered or rows, key=lambda item: abs(float(item[1].get("trade_return", 0.0))), reverse=True)
    elif label == "avoidable loser":
        ordered = sorted(rows, key=lambda item: (len(item[1].get("reason_codes", [])), float(item[1].get("trade_return", 0.0)), -float(item[1].get("confidence", 0.0))))
    elif label == "low-quality setup":
        ordered = sorted(rows, key=lambda item: (-1 if str(item[1].get("outcome")) == "win" else 0, len(item[1].get("reason_codes", []))))
    return [_example_view(row, idx) for idx, row in ordered[:2]]


def _suspicious_examples(rows: list[tuple[int, dict[str, Any]]], label: str) -> list[dict[str, Any]]:
    suspicious: list[tuple[int, dict[str, Any]]] = []
    if label == "high-quality winner":
        suspicious = [item for item in rows if any(code in {"poor_liquidity", "london_drag", "low_vol_trend", "post_spike"} for code in item[1].get("reason_codes", []))]
    elif label == "acceptable trade":
        suspicious = [item for item in rows if float(item[1].get("trade_return", 0.0)) < -0.005 or float(item[1].get("trade_return", 0.0)) > 0.007]
    elif label == "avoidable loser":
        suspicious = [item for item in rows if float(item[1].get("confidence", 0.0)) >= 0.75]
    elif label == "low-quality setup":
        suspicious = [item for item in rows if str(item[1].get("outcome", "")) == "win"]
    suspicious = suspicious[:3]
    return [_example_view(row, idx) for idx, row in suspicious]


def _reason_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in list(row.get("reason_codes", []) or []):
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def build_label_review(label_report_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    report = _load_label_report(label_report_path)
    records = list(report.get("records", []) or [])
    indexed = list(enumerate(records, start=1))

    label_examples: dict[str, Any] = {}
    suspicious_summary: dict[str, Any] = {}
    for label in LABEL_ORDER:
        rows = [(idx, row) for idx, row in indexed if str(row.get("label", "")) == label]
        label_examples[label] = {
            "count": len(rows),
            "typical_examples": _typical_examples(rows, label),
            "borderline_examples": _borderline_examples(rows, label),
            "suspicious_examples": _suspicious_examples(rows, label),
            "reason_code_counts": _reason_summary([row for _, row in rows]),
        }
        suspicious_summary[label] = len(label_examples[label]["suspicious_examples"])

    reason_code_review = {
        "high_quality_winner_wrong_bucket_candidates": [
            code for code in label_examples["high-quality winner"]["reason_code_counts"].keys()
            if code in {"poor_liquidity", "london_drag", "low_vol_trend", "post_spike", "late_us_trend"}
        ],
        "avoidable_loser_missing_negative_reason_candidates": [
            _trade_ref(row, idx)
            for idx, row in indexed
            if str(row.get("label", "")) == "avoidable loser" and not list(row.get("reason_codes", []))
        ],
        "acceptable_trade_mixed_context_count": sum(
            1
            for _, row in indexed
            if str(row.get("label", "")) == "acceptable trade" and "mixed_context" in list(row.get("reason_codes", []))
        ),
    }

    result = {
        "label_bucket_overview": {
            label: {
                "count": int(label_examples[label]["count"]),
                "suspicious_example_count": int(suspicious_summary[label]),
            }
            for label in LABEL_ORDER
        },
        "representative_examples": label_examples,
        "consistency_assessment": {
            "high-quality winner": "Mostly clean; positive reasons dominate. A few winners can still arise from poor-context setups, so outcome alone should not define quality.",
            "acceptable trade": "Broadest and noisiest bucket; useful as a neutral class, but still too mixed for direct scoring weight without sub-ranking.",
            "avoidable loser": "Clean enough for negative teaching signal; mostly driven by poor liquidity, London drag, and low-volatility trend contexts.",
            "low-quality setup": "Strong negative teaching bucket; post-spike and repeated re-entry dominate and remain poor even when some trades win.",
        },
        "reason_code_review": reason_code_review,
        "recommendations": {
            "relabels": [
                "Keep acceptable trade as a neutral bucket, but avoid using it as a strong positive or negative label in the first score version.",
                "Treat low-quality setup as an outcome-independent downgrade signal.",
                "Treat avoidable loser as the clearest negative training target.",
            ],
            "definition_fixes": [
                "Keep positive and negative reason codes separate; do not mix favorable codes into avoidable-loser labels.",
                "Preserve low-confidence ambiguous cases in acceptable trade instead of forcing them into avoidable loser.",
            ],
            "scoring_readiness": "Labels are usable for a first offline hybrid score, but acceptable trade is still too broad to weight heavily.",
            "safest_scoring_approach": "Hybrid first: rule-based negative downgrades from low-quality/avoidable labels plus soft positive boosts from high-quality-winner patterns.",
        },
    }

    output_file = Path(output_path or Path(label_report_path).with_name("trade_label_review.json"))
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
