from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTEXT_SNAPSHOT_FIELDS = (
    "symbol",
    "timestamp",
    "regime",
    "session",
    "trend_strength",
    "trend_state",
    "trend_structure",
    "momentum",
    "location",
    "setup_type_hint",
    "chart_reliability",
    "volatility_state",
    "spread_state",
    "liquidity_state",
    "spread_points",
    "dxy_alignment",
    "dxy_trend",
    "us10y_alignment",
    "us10y_trend",
    "vix_regime",
    "vix_alignment",
    "oil_backdrop",
    "oil_alignment",
    "event_risk_state",
    "news_risk_classification",
    "news_sentiment",
    "macro_classification",
    "slippage_state",
    "slippage_points",
    "repeated_entry_risk",
    "overstretched_entry_risk",
    "context_completeness",
)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_time(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).isoformat()
    except Exception:
        return raw


def _status_timestamp(status: dict[str, Any]) -> str:
    market = dict(status.get("market", {}) or {})
    recent_times = list(market.get("recent_time", []) or [])
    if recent_times:
        return _safe_time(str(recent_times[-1]))
    return _safe_time(str(status.get("updated_at", "")))


def _volatility_state(volatility_pct: float) -> str:
    if volatility_pct <= 0.0:
        return "unknown"
    if volatility_pct < 0.03:
        return "low"
    if volatility_pct >= 0.07:
        return "high"
    return "normal"


def _spread_state(spread_points: float) -> str:
    if spread_points <= 0.0:
        return "unknown"
    if spread_points >= 40.0:
        return "wide"
    if spread_points <= 20.0:
        return "tight"
    return "normal"


def _slippage_state(slippage_points: float) -> str:
    if slippage_points <= 0.0:
        return "unknown"
    if slippage_points >= 18.0:
        return "high"
    if slippage_points >= 10.0:
        return "elevated"
    return "normal"


def _liquidity_state(volume_ratio: float) -> str:
    if volume_ratio <= 0.0:
        return "unknown"
    if volume_ratio < 0.80:
        return "poor"
    if volume_ratio < 1.0:
        return "below_normal"
    return "normal"


def _derive_session_from_utc() -> str:
    # FIX: derive session deterministically when runtime has market/risk context but no explicit session.
    now = datetime.now(timezone.utc)
    hour = int(now.hour)
    if 13 <= hour < 17:
        return "overlap"
    if 7 <= hour < 12:
        return "london"
    if 12 <= hour < 20:
        return "new_york"
    if 0 <= hour < 7:
        return "asia"
    return "off_hours"


def _liquidity_state_from_spread(spread_points: float) -> str:
    # FIX: infer coarse liquidity state from spread when volume ratio is unavailable.
    if spread_points <= 0.0:
        return "unknown"
    if spread_points >= 40.0:
        return "poor"
    return "normal"


def _fresh_external_context_fields() -> dict[str, str]:
    paths = [
        Path(__file__).resolve().parent / "knowledge" / "external_context.json",
        Path(__file__).resolve().parent.parent / "knowledge" / "external_context.json",
    ]
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    generated_at_raw = str(payload.get("generated_at", "") or "").strip()
    if generated_at_raw:
        try:
            generated_at = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            max_age_seconds = 6 * 60 * 60 if str(payload.get("source", "") or "").startswith("yahoo_chart_price_proxy") else 90 * 60
            if (datetime.now(timezone.utc) - generated_at.astimezone(timezone.utc)).total_seconds() > max_age_seconds:
                return {}
        except Exception:
            return {}
    fields: dict[str, str] = {}
    allowed = {
        "dxy_alignment",
        "dxy_trend",
        "us10y_alignment",
        "us10y_trend",
        "vix_regime",
        "vix_alignment",
        "oil_backdrop",
        "oil_alignment",
        "event_risk_state",
        "news_risk_classification",
        "news_sentiment",
        "macro_classification",
    }
    raw_items = payload.get("items", [])
    if isinstance(raw_items, dict):
        raw_items = [{"key": key, "value": value} for key, value in raw_items.items()]
    for item in list(raw_items or []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", item.get("name", item.get("field", ""))) or "").strip()
        value = str(item.get("value", item.get("state", item.get("classification", ""))) or "").strip()
        if key in allowed and value:
            fields[key] = value
    for key in allowed:
        value = str(payload.get(key, "") or "").strip()
        if value:
            fields[key] = value
    return fields


def build_context_snapshot(status: dict[str, Any]) -> dict[str, Any]:
    regime = dict(status.get("regime", {}) or {})
    chart = dict(status.get("chart_features", {}) or {})
    risk = dict(status.get("risk", {}) or {})
    overlay = dict(status.get("research_regime_overlay", {}) or {})
    conservative = dict(status.get("conservative_pretrade_filter", {}) or {})
    overlay_inputs = dict(conservative.get("inputs", {}) or {})
    best_actionable = dict(status.get("best_actionable_candidate", {}) or {})
    reasons = {str(x) for x in list(best_actionable.get("reasons", []) or [])}
    volatility_pct = _safe_float(regime.get("volatility_pct", 0.0))
    spread_points = _safe_float(risk.get("spread_points", 0.0))
    slippage_points = _safe_float(
        risk.get(
            "slippage_points",
            status.get("slippage_points", overlay.get("slippage_points", overlay_inputs.get("slippage_points", 0.0))),
        )
    )
    volume_ratio = _safe_float(overlay.get("volume_ratio", overlay_inputs.get("volume_ratio", 0.0)))
    spike_ratio = _safe_float(overlay.get("spike_ratio", overlay_inputs.get("spike_ratio", 0.0)))
    raw_session = str(overlay.get("session", status.get("session", "")) or "").strip().lower()
    has_runtime_context = bool(status.get("symbol")) or bool(status.get("risk")) or bool(status.get("regime"))
    if raw_session in {"", "unknown", "-", "unavailable"}:
        session = _derive_session_from_utc() if has_runtime_context else "unknown"
    else:
        session = raw_session
    liquidity_state = _liquidity_state(volume_ratio) if volume_ratio > 0 else _liquidity_state_from_spread(spread_points)
    external = _fresh_external_context_fields()
    trend_state = str(chart.get("trend_state", regime.get("name", "unknown")) or "unknown").strip().lower()
    if trend_state == "trend":
        trend_state = "uptrend" if str(chart.get("trend_structure", "")).upper() == "HH/HL" else "downtrend" if str(chart.get("trend_structure", "")).upper() == "LH/LL" else "trend"
    macro_fields = {
        "dxy_alignment": str(status.get("dxy_alignment", overlay.get("dxy_alignment", external.get("dxy_alignment", "unknown"))) or "unknown"),
        "dxy_trend": str(status.get("dxy_trend", overlay.get("dxy_trend", external.get("dxy_trend", "unknown"))) or "unknown"),
        "us10y_alignment": str(status.get("us10y_alignment", overlay.get("us10y_alignment", external.get("us10y_alignment", "unknown"))) or "unknown"),
        "us10y_trend": str(status.get("us10y_trend", overlay.get("us10y_trend", external.get("us10y_trend", "unknown"))) or "unknown"),
        "vix_regime": str(status.get("vix_regime", overlay.get("vix_regime", external.get("vix_regime", "unknown"))) or "unknown"),
        "vix_alignment": str(status.get("vix_alignment", overlay.get("vix_alignment", external.get("vix_alignment", "unknown"))) or "unknown"),
        "oil_backdrop": str(status.get("oil_backdrop", overlay.get("oil_backdrop", external.get("oil_backdrop", "unknown"))) or "unknown"),
        "oil_alignment": str(status.get("oil_alignment", overlay.get("oil_alignment", external.get("oil_alignment", "unknown"))) or "unknown"),
        "event_risk_state": str(status.get("event_risk_state", overlay.get("event_risk_state", external.get("event_risk_state", "unknown"))) or "unknown"),
        "news_risk_classification": str(
            status.get("news_risk_classification", overlay.get("news_risk_classification", external.get("news_risk_classification", "unknown"))) or "unknown"
        ),
        "news_sentiment": str(status.get("news_sentiment", overlay.get("news_sentiment", external.get("news_sentiment", "unknown"))) or "unknown"),
        "macro_classification": str(status.get("macro_classification", overlay.get("macro_classification", external.get("macro_classification", "unknown"))) or "unknown"),
    }
    unknown_tokens = {"", "unknown", "unavailable", "none", "n/a"}
    missing_context = sorted(k for k, v in macro_fields.items() if str(v or "").strip().lower() in unknown_tokens)
    known_count = len(macro_fields) - len(missing_context)

    snapshot = {
        "symbol": str(status.get("symbol", "")),
        "timestamp": _status_timestamp(status),
        "regime": str(regime.get("name", "unknown") or "unknown"),
        "session": session,
        "trend_strength": round(_safe_float(regime.get("trend_strength", 0.0)), 6),
        "trend_state": trend_state,
        "trend_structure": str(chart.get("trend_structure", "mixed") or "mixed"),
        "momentum": str(chart.get("momentum_state", "mixed") or "mixed"),
        "location": str(chart.get("location_state", "unclear") or "unclear"),
        "setup_type_hint": str(chart.get("setup_type_hint", "no_trade") or "no_trade"),
        "chart_reliability": str(chart.get("reliability", "unknown") or "unknown"),
        "volatility_state": _volatility_state(volatility_pct),
        "spread_state": _spread_state(spread_points),
        "liquidity_state": liquidity_state,
        "spread_points": round(spread_points, 2) if spread_points > 0 else 0.0,
        **macro_fields,
        "slippage_state": str(status.get("slippage_state", overlay.get("slippage_state", _slippage_state(slippage_points))) or "unknown"),
        "slippage_points": round(slippage_points, 2) if slippage_points > 0 else 0.0,
        "repeated_entry_risk": bool("repeated_reentry" in reasons),
        "overstretched_entry_risk": bool("post_spike_entry" in reasons or spike_ratio >= 1.6),
        "context_completeness": {
            "known_macro_fields": known_count,
            "total_macro_fields": len(macro_fields),
            "missing_macro_fields": missing_context,
            "macro_available": bool(known_count > 0),
            "external_context_loaded": bool(external),
        },
    }
    return {field: snapshot.get(field) for field in CONTEXT_SNAPSHOT_FIELDS}
