from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# FIX: standalone expert knowledge for XAUUSD so signal confidence is informed by gold-specific behavior.
GOLD_SESSION_BEHAVIOUR = {
    "asian": {"trade": False},
    "london": {"trade": True},
    "overlap": {"trade": True},
    "new_york": {"trade": True},
}

GOLD_MACRO_RULES = [
    {"rule": "DXY inverse correlation", "confidence_boost": 0.08},
    {"rule": "US10Y yield inverse correlation", "confidence_boost": 0.06},
    {"rule": "Risk-off = gold up", "confidence_boost": 0.05},
]

GOLD_SETUP_LIBRARY = {
    "london_open_drive": {"win_rate_historical": 0.64},
    "order_block_retest": {"win_rate_historical": 0.61},
    "liquidity_sweep": {"win_rate_historical": 0.58},
    "structure_break_retest": {"win_rate_historical": 0.62},
    "fair_value_gap": {"win_rate_historical": 0.59},
}


def check_macro_alignment(rule: dict[str, Any], signal: dict[str, Any], market_state: dict[str, Any]) -> bool:
    direction = str(signal.get("direction", "")).lower()
    dxy = str(market_state.get("dxy_direction", "flat")).lower()
    us10y = str(market_state.get("us10y_direction", "flat")).lower()
    vix = str(market_state.get("vix_state", "normal")).lower()
    if rule["rule"] == "DXY inverse correlation":
        return (direction == "buy" and dxy == "falling") or (direction == "sell" and dxy == "rising")
    if rule["rule"] == "US10Y yield inverse correlation":
        return (direction == "buy" and us10y == "falling") or (direction == "sell" and us10y == "rising")
    if rule["rule"] == "Risk-off = gold up":
        return direction == "buy" and vix in {"high", "elevated"}
    return False


def apply_gold_knowledge_to_signal(signal: dict[str, Any], market_state: dict[str, Any]) -> dict[str, Any]:
    confidence = float(signal.get("confidence", 0.5) or 0.5)
    base_conf = confidence
    boosts: list[str] = []
    penalties: list[str] = []
    for rule in GOLD_MACRO_RULES:
        if check_macro_alignment(rule, signal, market_state):
            confidence += float(rule["confidence_boost"])
            boosts.append(f"{rule['rule']}:+{rule['confidence_boost']}")
    session = str(market_state.get("session", "unknown")).lower()
    if not bool(GOLD_SESSION_BEHAVIOUR.get(session, {}).get("trade", False)):
        confidence *= 0.5
        penalties.append(f"session_{session}:x0.5")
    setup_type = str(signal.get("setup_type", "") or "")
    if setup_type in GOLD_SETUP_LIBRARY and float(GOLD_SETUP_LIBRARY[setup_type].get("win_rate_historical", 0.5)) >= 0.60:
        confidence += 0.04
        boosts.append(f"setup_{setup_type}:+0.04")
    confidence = min(0.95, max(0.0, confidence))
    logger.info(
        "Gold knowledge applied | base=%.3f boosts=%s penalties=%s final=%.3f",
        base_conf,
        boosts,
        penalties,
        confidence,
    )
    out = dict(signal)
    out["confidence"] = confidence
    out["knowledge_applied"] = True
    out["boosts"] = boosts
    out["penalties"] = penalties
    return out

