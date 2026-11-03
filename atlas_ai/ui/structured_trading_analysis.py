from __future__ import annotations

from typing import Any

from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.decision_support import build_evidence_separation


STRUCTURED_TRADING_ANALYSIS_MARKERS = (
    "market bias",
    "trade setup",
    "good entry",
    "is this a good entry",
    "what setup is this",
    "pullback or breakout",
    "timeframe logic",
    "entry quality",
    "why is this no-trade",
    "why is this no trade",
    "buy or sell",
    "what should i do",
)

STRUCTURED_TRADING_ANALYSIS_EXCLUSIONS = (
    "should i buy or sell",
    "what do you think i should do",
    "what do you think we should do",
)


def is_structured_trading_analysis_query(query: str) -> bool:
    q = str(query or "").strip().lower()
    if any(marker in q for marker in STRUCTURED_TRADING_ANALYSIS_EXCLUSIONS):
        return False
    return any(marker in q for marker in STRUCTURED_TRADING_ANALYSIS_MARKERS)


def format_trading_response_text(trading_response: dict[str, Any], fallback: str = "") -> str:
    payload = dict(trading_response or {})
    if not payload:
        return str(fallback or "").strip()
    confidence = dict(payload.get("confidence", {}) or {})
    blockers = list(payload.get("blockers", []) or [])
    context_alignment = dict(payload.get("context_alignment", {}) or {})
    dxy = dict(context_alignment.get("dxy", {}) or {})
    us10y = dict(context_alignment.get("us10y", {}) or {})
    lines = [
        f"Bias: {str(payload.get('bias', 'neutral') or 'neutral')}",
        f"Confidence: {confidence.get('score', 0)}/100 ({str(confidence.get('level', 'low') or 'low')})",
        f"Blockers: {', '.join(str(x).replace('_', ' ') for x in blockers) if blockers else 'none'}",
        f"Context alignment: DXY={str(dxy.get('state', 'unknown') or 'unknown')}, US10Y={str(us10y.get('state', 'unknown') or 'unknown')}",
        f"Recommended action: {str(payload.get('recommended_action', 'hold') or 'hold')}",
    ]
    market_regime = dict(payload.get("market_regime", {}) or {})
    reversal_detection = dict(payload.get("reversal_detection", {}) or {})
    if market_regime:
        lines.append(
            f"Regime: {str(market_regime.get('regime', 'unknown') or 'unknown')} "
            f"({int(market_regime.get('regime_confidence', 0) or 0)}/100)"
        )
    if reversal_detection:
        lines.append(
            "Reversal: "
            + (
                "confirmed"
                if bool(reversal_detection.get("confirmed_reversal", False))
                else ("candidate" if bool(reversal_detection.get("reversal_candidate", False)) else "none")
            )
        )
    safety_note = str(payload.get("safety_note", "") or "").strip()
    if safety_note:
        lines.append(f"Safety note: {safety_note}")
    account_tier = str(payload.get("account_tier", "") or "").strip()
    risk_profile = str(payload.get("risk_profile", "") or "").strip()
    size_tier = dict(payload.get("size_tier", {}) or {})
    if account_tier:
        lines.append(f"Account tier: {account_tier}")
    if risk_profile:
        lines.append(f"Risk profile: {risk_profile}")
    if size_tier:
        current_tier = str(size_tier.get("current_size_tier", "") or "-")
        next_req = str(size_tier.get("next_size_tier_requirement", "") or "-")
        lines.append("Size tier: " + f"{current_tier} " + f"(next: {next_req})")
    return "\n".join(lines)


def build_structured_trading_analysis_reply(
    *,
    status: dict[str, Any],
    runtime: dict[str, Any],
    execution_state: dict[str, Any],
) -> dict[str, Any]:
    evidence_separation = build_evidence_separation(status, {**runtime, "execution": execution_state})
    chart_input = status.get("chart_features") if isinstance(status.get("chart_features"), dict) else (
        status.get("chart_analysis_input") if isinstance(status.get("chart_analysis_input"), dict) else None
    )
    decision = build_structured_decision_analysis(status=status, runtime=runtime, chart_input=chart_input)
    analysis = {
        "market_bias": decision.get("market_bias", "neutral"),
        "signal_bias": decision.get("signal_bias", "neutral"),
        "chart_structure_bias": decision.get("chart_structure_bias", "neutral"),
        "higher_timeframe_bias": decision.get("higher_timeframe_bias", "unknown"),
        "final_trade_bias": decision.get("final_trade_bias", "neutral"),
        "bias_breakdown": dict(decision.get("bias_breakdown", {}) or {}),
        "bias_strength": str(decision.get("bias_strength", "weak")),
        "trade_decision": str(decision.get("trade_decision", "NO_TRADE")),
        "confidence_score": int(decision.get("confidence_score", 0) or 0),
        "confidence_level": str(decision.get("confidence_level", "low")),
        "buy_score": int(decision.get("buy_score", 0) or 0),
        "sell_score": int(decision.get("sell_score", 0) or 0),
        "no_trade_score": int(decision.get("no_trade_score", 0) or 0),
        "trend": str(((decision.get("trend", {}) or {}).get("structure", "mixed"))),
        "momentum": str(((decision.get("momentum", {}) or {}).get("state", "mixed"))),
        "location": str(((decision.get("location", {}) or {}).get("state", "unclear"))),
        "structure_read": dict(((decision.get("trading_response", {}) or {}).get("structure_read", {}) or {})),
        "momentum_read": dict(((decision.get("trading_response", {}) or {}).get("momentum_read", {}) or {})),
        "location_read": dict(((decision.get("trading_response", {}) or {}).get("location_read", {}) or {})),
        "confirmation_read": dict(((decision.get("trading_response", {}) or {}).get("confirmation_read", {}) or {})),
        "setup_type": str(decision.get("setup_type", "no_trade")),
        "entry_quality": str(decision.get("entry_quality", "avoid")),
        "reasoning": list(decision.get("reasoning", []) or []),
        "blockers": list(decision.get("blockers", []) or []),
        "context_alignment": dict(decision.get("context_alignment", {}) or {}),
        "recommended_action": str(decision.get("recommended_action", "hold")),
        "safety_note": str(decision.get("safety_note", "") or ""),
        "timeframe_logic": dict(decision.get("timeframe_logic", {}) or {}),
        "warnings": list(decision.get("warnings", []) or []),
        "explanation": str(decision.get("explanation", "No-trade: insufficient setup quality.")),
        "response_schema": dict(decision.get("trading_response", {}) or {}),
        "evidence_separation": evidence_separation,
    }
    trading_response = dict(analysis.get("response_schema", {}) or {})
    return {
        "mode": "standard",
        "intent": "trading_analysis",
        "reasoning_mode": "STRUCTURED_ANALYSIS",
        "acknowledge": "Structured analysis ready.",
        "understanding": "Interpreted as a trading-analysis question; read-only analysis only.",
        "data_sources": ["status snapshot", "runtime flags", "execution snapshot", "evidence separation"],
        "result": format_trading_response_text(trading_response, fallback=str(analysis.get("explanation", ""))),
        "trading_analysis": analysis,
        "trading_response": trading_response,
        "evidence_separation": evidence_separation,
        "approval": "Read-only analysis. No execution action was taken.",
        "response_kind": "question_answer",
        "execution_state": execution_state,
    }
