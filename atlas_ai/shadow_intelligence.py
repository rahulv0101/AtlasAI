from __future__ import annotations

import re
from typing import Any


SHADOW_ADVICE_FIELDS = (
    "trade_id",
    "setup_score",
    "confidence",
    "anomaly_flags",
    "alternative_ranking",
    "recommendation",
)
ALLOWED_RECOMMENDATIONS = {"observe", "caution", "strong_context"}


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _normalize_recommendation(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in ALLOWED_RECOMMENDATIONS:
        return text
    return "observe"


def _safe_trade_id(status: dict[str, Any], decision_explanation: dict[str, Any]) -> str:
    trade_id = str(decision_explanation.get("trade_id", "") or "").strip()
    if trade_id:
        return trade_id
    candidate_id = str(decision_explanation.get("candidate_id", "") or "").strip()
    if candidate_id:
        return candidate_id
    return str(status.get("updated_at", "") or "").strip()


def build_shadow_advice(
    status: dict[str, Any],
    context_snapshot: dict[str, Any],
    decision_explanation: dict[str, Any],
) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    best_actionable = dict(status.get("best_actionable_candidate", {}) or {})
    ranking = dict(status.get("contextual_ranking", {}) or {})
    top = list(ranking.get("top", []) or [])
    setup_score = _safe_float(best_actionable.get("setup_quality_score", 0.0))
    chosen_conf = _safe_float(chosen.get("confidence", 0.0))
    anomaly_flags: list[str] = []

    if str(context_snapshot.get("spread_state", "")) == "wide":
        anomaly_flags.append("wide_spread")
    if str(context_snapshot.get("liquidity_state", "")) == "poor":
        anomaly_flags.append("poor_liquidity")
    if bool(context_snapshot.get("overstretched_entry_risk", False)):
        anomaly_flags.append("overstretched_entry_risk")
    if bool(context_snapshot.get("repeated_entry_risk", False)):
        anomaly_flags.append("repeated_entry_risk")
    if str(decision_explanation.get("final_decision", "")) == "block":
        anomaly_flags.append("blocked_trade")

    if str(decision_explanation.get("final_decision", "")) == "block":
        recommendation = "observe"
    elif anomaly_flags:
        recommendation = "caution"
    elif setup_score >= 0.10 and chosen_conf >= 0.30:
        recommendation = "strong_context"
    else:
        recommendation = "observe"

    alternative_ranking = [
        {
            "strategy": str(row.get("strategy", "")),
            "score": round(_safe_float(row.get("score", 0.0)), 4),
            "confidence": round(_safe_float(row.get("confidence", 0.0)), 4),
        }
        for row in top[:3]
    ]

    advice = {
        "trade_id": _safe_trade_id(status, decision_explanation),
        "setup_score": round(setup_score, 4),
        "confidence": round(chosen_conf, 4),
        "anomaly_flags": anomaly_flags,
        "alternative_ranking": alternative_ranking,
        "recommendation": _normalize_recommendation(recommendation),
    }
    return {field: advice.get(field) for field in SHADOW_ADVICE_FIELDS}
