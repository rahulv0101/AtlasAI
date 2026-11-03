from __future__ import annotations

from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui_server import create_ui_app


class _ChartSource:
    def __init__(self, chart_features: dict | None) -> None:
        self._chart = chart_features

    def load_status(self) -> dict:
        payload = {
            "action": "hold",
            "message": "Chart-assisted analysis",
            "chosen": {"confidence": 0.55, "side": "flat", "strategy": "none"},
            "setup_family": "no_trade",
            "comparison": {"active": False},
        }
        if self._chart is not None:
            payload["chart_features"] = dict(self._chart)
        return payload

    def load_events(self, limit: int = 160) -> list[dict]:
        return []

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False, "read_only_ui": False}

    def load_comparison(self) -> dict:
        return {}


def test_chart_analysis_returns_structured_read_only_output() -> None:
    svc = AskAtlasService(
        _ChartSource(
            {
                "trend_state": "uptrend",
                "trend_structure": "HH/HL",
                "momentum_state": "strong",
                "location_state": "pullback",
                "setup_type_hint": "pullback",
                "reliability": "high",
            }
        )
    )
    before = svc.execution.pending_requests(limit=50)
    out = svc.answer("is this a good entry")
    after = svc.execution.pending_requests(limit=50)
    assert out["intent"] == "trading_analysis"
    assert out["trading_analysis"]["market_bias"] in {"buy", "sell", "neutral"}
    assert out["trading_analysis"]["entry_quality"] in {"good", "risky", "avoid"}
    assert before == after


def test_uncertain_chart_analysis_fails_safely_with_neutral_avoid() -> None:
    svc = AskAtlasService(
        _ChartSource(
            {
                "trend_state": "unclear",
                "trend_structure": "mixed",
                "momentum_state": "mixed",
                "location_state": "unclear",
                "setup_type_hint": "no_trade",
                "reliability": "low",
            }
        )
    )
    out = svc.answer("market bias")
    analysis = out["trading_analysis"]
    assert analysis["market_bias"] == "neutral"
    assert analysis["entry_quality"] == "avoid"


def test_no_chart_input_still_returns_safe_no_trade_without_side_effects() -> None:
    svc = AskAtlasService(_ChartSource(None))
    before = svc.execution.pending_requests(limit=50)
    out = svc.answer("buy or sell")
    after = svc.execution.pending_requests(limit=50)
    assert out["trading_analysis"]["entry_quality"] in {"avoid", "risky"}
    assert before == after
    assert svc.execution.last_request() is None


def test_runtime_api_exposes_structured_decision_analysis(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_status(
        {
            "action": "buy",
            "chosen": {"confidence": 0.81, "side": "buy"},
            "context_snapshot": {"trend_structure": "HH/HL", "momentum": "strong", "location": "pullback"},
            "setup_family": "pullback",
        }
    )
    app = create_ui_app(store)
    payload = app.test_client().get("/api/runtime").get_json()
    assert "decision_analysis" in payload
    da = dict(payload["decision_analysis"] or {})
    assert "market_bias" in da
    assert "confidence_score" in da
    assert "entry_quality" in da
