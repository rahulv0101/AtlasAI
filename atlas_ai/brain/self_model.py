from __future__ import annotations

from typing import Any


def build_self_model(status: dict[str, Any], runtime: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    trader_cognition = dict(status.get("trader_cognition", {}) or {})
    execution = dict(runtime.get("execution", {}) or {})
    mandate = dict(execution.get("autonomous_trader_mandate", {}) or {})
    confidence = float(chosen.get("confidence", 0.0) or 0.0)
    threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
    mode = "LIVE" if bool(runtime.get("execute_trades", False)) else "PAPER"
    comparison_active = bool(comparison.get("active", status.get("comparison_active", False)))
    target_role = str(
        mandate.get("target_role")
        or dict(trader_cognition.get("autonomy_boundary", {}) or {}).get("target_role")
        or "bounded_autonomous_trader"
    )
    autonomy_mode = str(execution.get("autonomy_mode", "recommend_only") or "recommend_only")
    current_role = (
        f"Bounded autonomous trader candidate in {autonomy_mode} mode; "
        "ExecutionGate remains the only execution authority."
    )
    return {
        "identity": "Atlas bounded trading operator",
        "purpose": "Grounded trading operation using configured Atlas state, structured knowledge, and approved execution boundaries.",
        "current_role": current_role,
        "target_role": target_role,
        "autonomy_mode": autonomy_mode,
        "feeling_model": "Atlas can simulate disciplined trader-like service intent, but it has no real feelings, consciousness, or private desires.",
        "capabilities": [
            "Status/blocker/readiness/comparison reasoning",
            "Confidence and safety interpretation",
            "Self-knowledge and capability clarification",
            "Market and strategy concept explanation from structured knowledge",
            "Cross-component coordination guidance for voice, Ask Atlas, scanner, trader, and ExecutionGate handoffs",
            "Trader-like cognition fields for observation, thesis, counter-thesis, discipline, and required next evidence",
            "Bounded autonomous demo posture when explicitly placed in guarded autonomous mode",
        ],
        "limitations": [
            "No hallucinated facts beyond available state/knowledge",
            "No live trading actions from Ask Atlas replies",
            "No guarantee of future market outcomes",
            "No real emotions, desire, consciousness, or human-like private thoughts",
            "No autonomous execution outside the approved mandate and ExecutionGate controls",
        ],
        "knowledge_domains": [
            "Market structure, trend, volatility, liquidity",
            "Strategy/risk policy and readiness constraints",
            "Current runtime and comparison diagnostics",
            "Full-stack coordination contracts and safe escalation rules",
        ],
        "readiness": "Not yet" if mode == "PAPER" else "Operator-gated",
        "confidence_in_self_assessment": max(20, min(92, int(round((confidence + 0.25) * 100)))),
        "confidence": confidence,
        "confidence_gap": confidence - threshold,
        "comparison_active": comparison_active,
    }
