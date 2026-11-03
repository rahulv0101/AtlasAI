from __future__ import annotations


_EXACT_ALIASES = {
    "candlesti": "candlestick_patterns",
    "candlestick": "candlestick_patterns",
    "macd_tren": "macd_trend",
    "vwap_prox": "vwap_proxy_reversion",
    "rsi_trend": "rsi_trend_hybrid",
    "evolved_r": "evolved_rule",
    "mom": "momentum_continuation",
}


def canonical_strategy_name(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "unknown"
    if raw.startswith("evolved_rule("):
        return raw

    lowered = raw.lower()
    if lowered in _EXACT_ALIASES:
        return _EXACT_ALIASES[lowered]
    if lowered.startswith("candlesti"):
        return "candlestick_patterns"
    if lowered.startswith("macd_tren"):
        return "macd_trend"
    if lowered.startswith("vwap_prox"):
        return "vwap_proxy_reversion"
    if lowered.startswith("rsi_trend"):
        return "rsi_trend_hybrid"
    if lowered.startswith("evolved_r"):
        return "evolved_rule"
    return raw


def canonical_context_key(key: str) -> str:
    raw = str(key or "")
    if "|" not in raw:
        return canonical_strategy_name(raw)
    strategy, rest = raw.split("|", 1)
    return f"{canonical_strategy_name(strategy)}|{rest}"
