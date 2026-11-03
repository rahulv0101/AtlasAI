from __future__ import annotations

import re
from typing import Any

from atlas_ai.strategy_aliases import canonical_strategy_name

MAX_BROKER_COMMENT_LEN = 31
_COMMENT_PREFIX = "AAI"
_COMMENT_PATTERN = re.compile(
    r"(?i)\baai(?:_d(?P<decision>[a-z0-9]+))?(?:_s(?P<strategy>[a-z0-9]+))?"
    r"(?:_l(?P<lane>[a-z0-9]+))?(?:_o(?P<overlay>[a-z0-9]+))?(?:_r(?P<regime>[a-z0-9]+))?\b"
)
_LEGACY_STRATEGY_PATTERN = re.compile(r"aai(?:_d_[a-z0-9]+)?_st_([a-z0-9_]+)", re.IGNORECASE)
_LEGACY_DECISION_PATTERN = re.compile(r"aai(?:_d)?_([a-z0-9]{4,8})(?:_st_)?", re.IGNORECASE)
_BROKER_STRATEGY_TAGS = {
    "ema_crossover": "emax",
    "rsi_mean_reversion": "rsim",
    "breakout": "brkt",
    "candlestick_patterns": "cand",
    "macd_trend": "macd",
    "bollinger_mean_reversion": "boll",
    "stochastic_reversal": "stoc",
    "vwap_proxy_reversion": "vwap",
    "momentum_continuation": "momc",
    "pullback_trend": "pull",
    "rsi_trend_hybrid": "rsih",
    "gold_micro_impulse": "gmi",
    "gold_breakout_retest": "gbr",
    "gold_liquidity_sweep": "gls",
    "gold_micro_structure_reversion": "gmsr",
    "evolved_rule": "evr",
}
_BROKER_TAG_TO_STRATEGY = {tag: name for name, tag in _BROKER_STRATEGY_TAGS.items()}


def _safe_token(raw: str, max_len: int) -> str:
    token = re.sub(r"[^a-z0-9]", "", str(raw or "").lower())
    return token[: max(0, int(max_len))]


def _fallback_strategy_tag(strategy_name: str) -> str:
    base = canonical_strategy_name(strategy_name).split("(", 1)[0].strip().lower()
    parts = [part[:2] for part in re.split(r"[^a-z0-9]+", base) if part]
    tag = "".join(parts)[:4]
    return tag or "strt"


def broker_strategy_tag(strategy_name: str) -> str:
    canonical = canonical_strategy_name(strategy_name)
    if canonical.startswith("evolved_rule("):
        return _BROKER_STRATEGY_TAGS["evolved_rule"]
    base = canonical.split("(", 1)[0].strip().lower()
    return _BROKER_STRATEGY_TAGS.get(base, _fallback_strategy_tag(base))


def strategy_name_from_broker_tag(tag: str) -> str | None:
    clean = _safe_token(tag, max_len=12)
    if not clean:
        return None
    mapped = _BROKER_TAG_TO_STRATEGY.get(clean)
    if mapped:
        return mapped
    return canonical_strategy_name(clean)


def build_broker_comment(
    strategy_name: str,
    decision_id: str = "",
    lane_tag: str = "",
    overlay_tag: str = "",
    regime_tag: str = "",
    max_len: int = MAX_BROKER_COMMENT_LEN,
) -> str:
    max_len = max(8, int(max_len))
    parts = [_COMMENT_PREFIX]
    decision = _safe_token(decision_id, max_len=6)
    strategy_tag = broker_strategy_tag(strategy_name)

    if decision:
        parts.append(f"D{decision}")
    if strategy_tag:
        parts.append(f"S{strategy_tag}")

    optional_tokens = [
        ("L", _safe_token(lane_tag, max_len=4)),
        ("O", _safe_token(overlay_tag, max_len=4)),
        ("R", _safe_token(regime_tag, max_len=4)),
    ]
    for prefix, token in optional_tokens:
        if not token:
            continue
        candidate = "_".join(parts + [f"{prefix}{token}"])
        if len(candidate) <= max_len:
            parts.append(f"{prefix}{token}")
    comment = "_".join(parts)
    return comment[:max_len]


def parse_broker_comment(raw: Any) -> dict[str, str]:
    text = str(raw or "").strip()
    normalized = re.sub(r"[^a-z0-9_]", "_", text.lower())
    match = _COMMENT_PATTERN.search(normalized)
    if match:
        strategy_tag = str(match.group("strategy") or "")
        return {
            "decision_id": _safe_token(match.group("decision") or "", max_len=8),
            "strategy_tag": strategy_tag,
            "strategy": strategy_name_from_broker_tag(strategy_tag) or "",
            "lane_tag": _safe_token(match.group("lane") or "", max_len=4),
            "overlay_tag": _safe_token(match.group("overlay") or "", max_len=4),
            "regime_tag": _safe_token(match.group("regime") or "", max_len=4),
        }

    strategy_match = _LEGACY_STRATEGY_PATTERN.search(normalized)
    decision_match = _LEGACY_DECISION_PATTERN.search(normalized)
    strategy_raw = str(strategy_match.group(1) if strategy_match else "")
    return {
        "decision_id": _safe_token(decision_match.group(1) if decision_match else "", max_len=8),
        "strategy_tag": strategy_raw,
        "strategy": canonical_strategy_name(strategy_raw) if strategy_raw else "",
        "lane_tag": "",
        "overlay_tag": "",
        "regime_tag": "",
    }
