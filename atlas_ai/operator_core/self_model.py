from __future__ import annotations

from typing import Any


def build_self_model(status: dict[str, Any], runtime: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    context = dict(status.get("context_snapshot", {}) or {})
    confidence = float(dict(status.get("chosen", {}) or {}).get("confidence", 0.0) or 0.0)
    threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
    mode = "LIVE" if bool(runtime.get("execute_trades", False)) else "PAPER"
    comparison_active = bool(comparison.get("active", status.get("comparison_active", False)))

    return {
        "identity": "Atlas operator assistant",
        "purpose": "Grounded trading-operations reasoning using configured Atlas state and knowledge.",
        "capabilities": [
            "Answer state, blocker, readiness, confidence, and comparison questions",
            "Explain reasoning in concise operator format",
            "Recommend bounded next-safe experiments in paper mode",
            "Interpret market concepts and strategy/risk rules from structured knowledge",
        ],
        "limitations": [
            "Cannot invent facts outside available Atlas state and knowledge files",
            "Ask Atlas remains read-only for trading behavior and execution",
            "Cannot guarantee future market outcomes",
        ],
        "knowledge_domains": [
            "Market structure, trend, volatility, liquidity, momentum",
            "Configured strategy and risk-policy constraints",
            "Runtime/comparison diagnostics and confidence gating",
        ],
        "readiness": "Not yet" if mode == "PAPER" else "Operator-gated",
        "readiness_reason": "Paper validation remains primary safety mode." if mode == "PAPER" else "Live mode still requires operator governance.",
        "confidence_in_self_assessment": max(25, min(90, int(round((confidence + 0.25) * 100)))),
        "confidence": confidence,
        "threshold": threshold,
        "confidence_gap": confidence - threshold,
        "current_role": (
            f"Focused on {str(context.get('regime', 'current'))} context, "
            f"session {str(context.get('session', 'unknown'))}, "
            f"with {'active' if comparison_active else 'no active'} comparison tracking."
        ),
    }

