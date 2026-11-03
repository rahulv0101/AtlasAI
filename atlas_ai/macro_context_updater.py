from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen


SYMBOLS = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "oil": "CL=F",
}


def _fetch_chart(symbol: str) -> list[float]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol)}?range=5d&interval=1d"
    req = Request(url, headers={"User-Agent": "AtlasAI/1.0"})
    with urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    result = ((payload.get("chart") or {}).get("result") or [{}])[0]
    close = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or [])
    return [float(x) for x in close if x is not None and math.isfinite(float(x))]


def _direction(values: list[float]) -> str:
    if len(values) < 2:
        return "unknown"
    first = float(values[-2])
    last = float(values[-1])
    if first <= 0:
        return "unknown"
    pct = ((last - first) / first) * 100.0
    if pct >= 0.15:
        return "rising"
    if pct <= -0.15:
        return "falling"
    return "flat"


def _gold_alignment_from_inverse(direction: str, *, supports_buy_when_falling: bool = True) -> str:
    if direction == "falling" and supports_buy_when_falling:
        return "supportive"
    if direction == "rising" and supports_buy_when_falling:
        return "conflict"
    return "mixed"


def build_macro_context() -> dict[str, Any]:
    fetched: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    for key, symbol in SYMBOLS.items():
        try:
            fetched[key] = _fetch_chart(symbol)
        except Exception as exc:
            errors[key] = str(exc)
            fetched[key] = []

    dxy_trend = _direction(fetched["dxy"])
    us10y_trend = _direction(fetched["us10y"])
    vix_trend = _direction(fetched["vix"])
    oil_trend = _direction(fetched["oil"])
    vix_last = fetched["vix"][-1] if fetched["vix"] else 0.0

    vix_regime = "unknown"
    if vix_last > 0:
        if vix_last >= 25:
            vix_regime = "elevated"
        elif vix_last <= 16:
            vix_regime = "calm"
        else:
            vix_regime = "normal"

    oil_backdrop = "unknown"
    if oil_trend == "rising":
        oil_backdrop = "inflationary"
    elif oil_trend == "falling":
        oil_backdrop = "softening"
    elif oil_trend == "flat":
        oil_backdrop = "neutral"

    items = {
        "dxy_trend": dxy_trend,
        "dxy_alignment": _gold_alignment_from_inverse(dxy_trend),
        "us10y_trend": us10y_trend,
        "us10y_alignment": _gold_alignment_from_inverse(us10y_trend),
        "vix_regime": vix_regime,
        "vix_alignment": "supportive" if vix_regime in {"normal", "calm"} else ("conflict" if vix_regime == "elevated" else "mixed"),
        "oil_backdrop": oil_backdrop,
        "oil_alignment": "supportive" if oil_backdrop == "inflationary" else ("mixed" if oil_backdrop in {"neutral", "softening"} else "unknown"),
        "event_risk_state": "unknown",
        "news_risk_classification": "unknown",
        "news_sentiment": "unknown",
        "macro_classification": "price_proxy_only",
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "yahoo_chart_price_proxy",
        "items": [{"key": key, "value": value} for key, value in items.items()],
        "raw": {
            key: {"symbol": SYMBOLS[key], "last": values[-1] if values else None, "direction": _direction(values)}
            for key, values in fetched.items()
        },
        "errors": errors,
        "notes": [
            "Price-proxy macro context only; event/news risk remains unknown until a calendar/news feed is supplied.",
            "Gold alignment uses broad inverse DXY/US10Y heuristics and should influence confidence, not override safety gates.",
        ],
    }


def write_external_context(path: Path | None = None) -> dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "knowledge" / "external_context.json")
    payload = build_macro_context()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    payload = write_external_context()
    print(json.dumps({"generated_at": payload.get("generated_at"), "source": payload.get("source"), "errors": payload.get("errors", {})}, indent=2))


if __name__ == "__main__":
    main()
