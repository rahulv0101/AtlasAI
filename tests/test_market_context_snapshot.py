from __future__ import annotations

from atlas_ai.market_context import build_context_snapshot
from atlas_ai import market_context


def test_context_snapshot_carries_chart_features_and_missing_macro_inventory(monkeypatch) -> None:
    monkeypatch.setattr(market_context, "_fresh_external_context_fields", lambda: {})
    snapshot = build_context_snapshot(
        {
            "symbol": "XAUUSD",
            "session": "new_york",
            "regime": {"name": "trend", "trend_strength": 0.22, "volatility_pct": 0.034},
            "risk": {"spread_points": 34.0},
            "chart_features": {
                "trend_state": "uptrend",
                "trend_structure": "HH/HL",
                "momentum_state": "mixed",
                "location_state": "mid_range",
                "setup_type_hint": "continuation",
                "reliability": "medium",
            },
        }
    )

    assert snapshot["trend_state"] == "uptrend"
    assert snapshot["trend_structure"] == "HH/HL"
    assert snapshot["momentum"] == "mixed"
    assert snapshot["location"] == "mid_range"
    assert snapshot["setup_type_hint"] == "continuation"
    completeness = snapshot["context_completeness"]
    assert completeness["macro_available"] is False
    assert "dxy_alignment" in completeness["missing_macro_fields"]


def test_context_snapshot_reports_partial_macro_availability() -> None:
    snapshot = build_context_snapshot(
        {
            "symbol": "XAUUSD",
            "session": "london",
            "regime": {"name": "trend", "trend_strength": 0.25, "volatility_pct": 0.04},
            "risk": {"spread_points": 18.0},
            "chart_features": {"trend_structure": "LH/LL", "momentum_state": "strong", "location_state": "pullback"},
            "dxy_alignment": "aligned",
            "us10y_alignment": "conflict",
            "event_risk_state": "low",
        }
    )

    completeness = snapshot["context_completeness"]
    assert completeness["macro_available"] is True
    assert completeness["known_macro_fields"] >= 3
    assert "dxy_alignment" not in completeness["missing_macro_fields"]


def test_context_snapshot_can_use_fresh_structured_external_context(monkeypatch) -> None:
    monkeypatch.setattr(
        market_context,
        "_fresh_external_context_fields",
        lambda: {
            "dxy_alignment": "aligned",
            "us10y_alignment": "conflict",
            "vix_regime": "normal",
            "event_risk_state": "low",
        },
    )

    snapshot = build_context_snapshot(
        {
            "symbol": "XAUUSD",
            "session": "new_york",
            "regime": {"name": "trend", "trend_strength": 0.2, "volatility_pct": 0.04},
            "risk": {"spread_points": 24.0},
            "chart_features": {"trend_structure": "HH/HL", "momentum_state": "strong", "location_state": "pullback"},
        }
    )

    assert snapshot["dxy_alignment"] == "aligned"
    assert snapshot["us10y_alignment"] == "conflict"
    assert snapshot["vix_regime"] == "normal"
    assert snapshot["event_risk_state"] == "low"
    assert snapshot["context_completeness"]["external_context_loaded"] is True
