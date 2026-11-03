from __future__ import annotations

from typing import Any


def analyze_chart_features(chart_input: dict[str, Any] | None) -> dict[str, Any]:
    raw = dict(chart_input or {})
    if not raw:
        return {
            "trend_state": "unclear",
            "trend_structure": "mixed",
            "momentum_state": "mixed",
            "location_state": "unclear",
            "setup_type_hint": "no_trade",
            "reliability": "low",
            "warnings": ["no_chart_input_available"],
        }

    trend_state = str(raw.get("trend_state", "unclear") or "unclear").lower()
    trend_structure = str(raw.get("trend_structure", "mixed") or "mixed")
    momentum_state = str(raw.get("momentum_state", "mixed") or "mixed").lower()
    location_state = str(raw.get("location_state", "unclear") or "unclear").lower()
    setup_hint = str(raw.get("setup_type_hint", "no_trade") or "no_trade").lower()
    reliability = str(raw.get("reliability", "medium") or "medium").lower()
    warnings = [str(x) for x in list(raw.get("warnings", []) or []) if str(x).strip()]

    if trend_state not in {"uptrend", "downtrend", "range", "unclear"}:
        trend_state = "unclear"
    if momentum_state not in {"strong", "weak", "mixed"}:
        momentum_state = "mixed"
    if location_state not in {"pullback", "breakout", "consolidation", "mid_range", "unclear"}:
        location_state = "unclear"
    # purpose: preserve richer setup hints so range-aware analysis can reason
    # about mean-reversion vs continuation without forcing no_trade.
    if setup_hint not in {"pullback", "breakout", "consolidation", "continuation", "mean_reversion", "no_trade"}:
        setup_hint = "no_trade"
    if reliability not in {"high", "medium", "low"}:
        reliability = "low"

    if reliability == "low" and "chart_low_reliability" not in warnings:
        warnings.append("chart_low_reliability")
    return {
        "trend_state": trend_state,
        "trend_structure": trend_structure,
        "momentum_state": momentum_state,
        "location_state": location_state,
        "setup_type_hint": setup_hint,
        "reliability": reliability,
        "warnings": warnings,
    }
