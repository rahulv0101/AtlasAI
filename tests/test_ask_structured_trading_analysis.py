from __future__ import annotations

import pytest

from atlas_ai.ui.ask_atlas import AskAtlasService


class _Source:
    def __init__(self) -> None:
        self.runtime = {"execute_trades": False, "read_only_ui": False}

    def load_status(self) -> dict:
        return {
            "action": "buy",
            "message": "High-confidence continuation setup with breakout confirmation.",
            "setup_family": "continuation",
            "chosen": {"confidence": 0.82, "side": "buy", "strategy": "momentum_continuation"},
            "context_snapshot": {"trend_structure": "HH/HL", "momentum": "strong", "location": "breakout"},
            "comparison": {"active": False},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return []

    def load_runtime_flags(self) -> dict:
        return dict(self.runtime)

    def load_comparison(self) -> dict:
        return {}


class _NoSetupSource(_Source):
    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "Range market, no momentum confirmation.",
            "setup_family": "none",
            "chosen": {"confidence": 0.32, "side": "flat", "strategy": "none"},
            "context_snapshot": {"location": "consolidation", "momentum": "weak"},
            "comparison": {"active": False},
        }


class _TraceSource(_Source):
    def load_status(self) -> dict:
        status = super().load_status()
        status["action"] = "blocked"
        status["message"] = "Setup-quality filter blocked trade: score=62.0 < 76.0."
        status["decision_explanation"] = {
            "final_decision": "block",
            "reason_codes": ["setup_quality_filter"],
            "setup_quality_reasons": ["weak_consensus"],
        }
        status["decision_trace"] = {
            "schema": "atlas_decision_trace_v1",
            "selected_candidate": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.62},
            "bias_breakdown": {
                "signal_bias": "buy",
                "chart_structure_bias": "sell",
                "higher_timeframe_bias": "buy",
                "final_trade_bias": "neutral",
                "conflict": True,
            },
            "filters": {
                "hard_blockers": [
                    {"name": "setup_quality_filter", "reason": "score=62.0 < 76.0", "hard_blocked": True}
                ],
                "reason_codes": ["setup_quality_filter"],
            },
        }
        return status


@pytest.mark.parametrize(
    "query",
    [
        "buy or sell",
        "market bias",
        "what should I do",
        "what setup is this?",
    ],
)
def test_structured_trading_analysis_known_overlap_phrases(query: str) -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer(query)

    assert out["intent"] == "trading_analysis"
    assert out["intent"] != "recommendation"
    analysis = dict(out.get("trading_analysis", {}) or {})
    assert analysis
    assert {
        "market_bias",
        "signal_bias",
        "chart_structure_bias",
        "higher_timeframe_bias",
        "final_trade_bias",
        "entry_quality",
        "evidence_separation",
    }.issubset(analysis.keys())


def test_structured_trading_analysis_response_contains_required_fields() -> None:
    svc = AskAtlasService(_Source())
    out = svc.answer("buy or sell")
    assert out["intent"] == "trading_analysis"
    analysis = dict(out.get("trading_analysis", {}) or {})
    required = {
        "market_bias",
        "signal_bias",
        "chart_structure_bias",
        "higher_timeframe_bias",
        "final_trade_bias",
        "bias_breakdown",
        "confidence_score",
        "confidence_level",
        "trend",
        "momentum",
        "location",
        "setup_type",
        "entry_quality",
        "explanation",
    }
    assert required.issubset(set(analysis.keys()))
    assert analysis["market_bias"] in {"buy", "sell", "neutral"}
    assert analysis["final_trade_bias"] in {"buy", "sell", "neutral"}
    assert 0 <= int(analysis["confidence_score"]) <= 100
    assert analysis["confidence_level"] in {"low", "medium", "high"}
    separation = dict(analysis.get("evidence_separation", {}) or {})
    assert separation["signal_direction"] == "buy"
    assert separation["market_bias"] == "buy"
    assert separation["execution_permission"] in {"allowed", "not_allowed"}
    assert "do not change strategy" in separation["policy_note"]
    assert out["evidence_separation"] == separation


def test_no_valid_setup_explains_what_is_missing() -> None:
    svc = AskAtlasService(_NoSetupSource())
    out = svc.answer("why is this no-trade?")
    analysis = dict(out.get("trading_analysis", {}) or {})
    assert analysis["entry_quality"] == "avoid"
    assert "missing" in str(analysis["explanation"]).lower() or "no valid setup" in str(analysis["explanation"]).lower()


def test_structured_analysis_queries_create_no_execution_requests_or_control_mutation() -> None:
    svc = AskAtlasService(_Source())
    before_requests = svc.execution.pending_requests(limit=50)
    before_control = svc.control.to_debug()
    _ = svc.answer("market bias")
    _ = svc.answer("what should I do")
    _ = svc.answer("buy or sell")
    _ = svc.answer("what setup is this?")
    after_requests = svc.execution.pending_requests(limit=50)
    after_control = svc.control.to_debug()
    assert before_requests == after_requests
    assert before_control == after_control
    assert svc.execution.last_request() is None


def test_explanation_prefers_decision_trace_when_available() -> None:
    svc = AskAtlasService(_TraceSource())
    out = svc.answer("Why no trade?")
    text = out["result"].lower()
    assert out["handled_as"] == "explanation"
    assert "setup quality filter" in text
    assert "selected candidate: buy via momentum_continuation" in text
    assert "bias read: signal=buy, chart=sell" in text
    assert "evidence lanes:" in text
    assert "decision trace" in out["data_sources"]
    separation = dict(out.get("evidence_separation", {}) or {})
    assert separation["signal_direction"] == "buy"
    assert separation["market_bias"] == "sell"
    assert separation["execution_permission"] == "not_allowed"
