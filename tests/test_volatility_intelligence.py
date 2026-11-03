from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from atlas_ai.decision_framework import DecisionFramework, DecisionInput
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.market_scanner import MarketScanner
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.volatility_intelligence import (
    VolatilityFeatures,
    VolatilityOutcomeRecord,
    classify_volatility,
    summarize_performance_by_volatility,
)


def _base_features(**overrides) -> VolatilityFeatures:
    base = VolatilityFeatures(
        atr_current=10.0,
        atr_14=10.0,
        atr_baseline=10.0,
        atr_ratio=1.0,
        candle_body_ratio=0.55,
        wick_ratio=0.35,
        average_range=9.0,
        range_expansion_ratio=1.0,
        spread_points=25.0,
        spread_pips=2.5,
        spread_status="normal",
        session="london",
        session_transition_flag=False,
        news_window_flag=False,
        high_impact_news_hard_block=False,
        dxy_alignment_state="aligned",
        us10y_alignment_state="aligned",
        vix_state="normal",
        price_structure_state="trend_up",
        momentum_state="bullish",
        macro_available=True,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_compressed_volatility_classification():
    out = classify_volatility(_base_features(atr_ratio=0.65, candle_body_ratio=0.2, wick_ratio=0.2))
    assert out.volatility_class == "compressed_volatility"


def test_normal_volatility_classification():
    out = classify_volatility(_base_features(atr_ratio=1.0, spread_status="normal"))
    assert out.volatility_class == "normal_volatility"


def test_elevated_volatility_reduces_risk_response():
    out = classify_volatility(_base_features(atr_ratio=1.35, price_structure_state="range_or_mixed", momentum_state="mixed"))
    assert out.volatility_class == "elevated_volatility"
    assert out.response_rule == "reduce_risk_require_higher_confluence"


def test_extreme_volatility_blocks_entries():
    out = classify_volatility(_base_features(atr_ratio=2.4, spread_status="high"))
    assert out.volatility_class == "extreme_volatility"
    assert "extreme_volatility" in out.blockers


def test_news_spike_volatility_hard_blocks():
    out = classify_volatility(_base_features(news_window_flag=True, high_impact_news_hard_block=True))
    assert out.volatility_class == "news_spike_volatility"
    assert "high_impact_news_window" in out.blockers


def test_liquidity_gap_blocks_on_spread_risk():
    out = classify_volatility(_base_features(atr_ratio=1.5, spread_status="blocked"))
    assert out.volatility_class == "liquidity_gap_volatility"
    assert "spread_liquidity_too_poor" in out.blockers


def test_chop_volatility_lowers_confidence_in_framework():
    fw = DecisionFramework()
    inp = DecisionInput(
        higher_highs=False,
        higher_lows=False,
        lower_highs=False,
        lower_lows=False,
        in_middle_of_range=True,
        bullish_candle_strength=0.2,
        bearish_candle_strength=0.2,
        session="london",
        session_tradeable=True,
        spread_pips=1.5,
        volatility_class="chop_volatility",
        volatility_confidence=0.8,
    )
    out = fw.evaluate(inp)
    assert out.trade_decision == "NO_TRADE"
    assert out.no_trade_score >= 0.3


def test_trend_expansion_can_allow_candidate():
    fw = DecisionFramework()
    inp = DecisionInput(
        higher_highs=True,
        higher_lows=True,
        bullish_candle_strength=0.8,
        near_support=True,
        session="london",
        session_tradeable=True,
        spread_pips=1.2,
        volatility_class="trend_expansion_volatility",
        volatility_confidence=0.8,
        momentum_direction="bullish",
        dxy_direction="falling",
    )
    out = fw.evaluate(inp)
    assert out.trade_decision in {"BUY_CANDIDATE", "NO_TRADE"}


def test_reversal_exhaustion_requires_confirmation():
    fw = DecisionFramework()
    inp = DecisionInput(
        lower_highs=True,
        lower_lows=True,
        bearish_candle_strength=0.72,
        near_resistance=True,
        session="london",
        session_tradeable=True,
        spread_pips=1.2,
        volatility_class="reversal_exhaustion_volatility",
        volatility_confidence=0.6,
    )
    out = fw.evaluate(inp)
    assert out.confidence <= 0.9


def test_missing_macro_data_falls_back_to_price_features():
    out = classify_volatility(_base_features(macro_available=False))
    assert "macro_unavailable_price_inferred" in out.tags
    assert out.volatility_confidence < 0.74


def test_scanner_logs_volatility_class_every_cycle(tmp_path):
    class _Gate:
        def create_request(self, **kwargs):
            return {"ok": True, **kwargs}

    class _Scanner(MarketScanner):
        def _build_decision_input(self, now):
            self._latest_volatility = classify_volatility(_base_features())
            self._latest_volatility_features = _base_features()
            return DecisionInput(
                session="asian",
                session_tradeable=False,
                spread_pips=1.5,
                timestamp=now,
                volatility_class=self._latest_volatility.volatility_class,
                volatility_confidence=self._latest_volatility.volatility_confidence,
            )

    settings = type("S", (), {"scan_interval_seconds": 1, "execute_trades": False, "learning_mode": False, "duplicate_signal_window_seconds": 60, "scanner_cooldown_seconds": 300})()
    s = _Scanner(_Gate(), broker_adapter=None, settings=settings)
    s.SCAN_LOG_PATH = str(tmp_path / "scan.jsonl")
    s._run_single_scan()
    row = json.loads(Path(s.SCAN_LOG_PATH).read_text(encoding="utf-8").strip())
    assert "volatility_class" in row


def test_completed_trades_update_volatility_stats():
    rows = [
        VolatilityOutcomeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            volatility_class_at_entry="normal_volatility",
            atr_ratio_at_entry=1.0,
            spread_status_at_entry="normal",
            session="london",
            outcome="win",
            r_multiple=1.2,
            mfe=1.8,
            mae=-0.4,
            response_was_correct=True,
            blocker_prevented_loss=False,
            blocker_missed_profit=False,
        ),
        VolatilityOutcomeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            volatility_class_at_entry="normal_volatility",
            atr_ratio_at_entry=1.1,
            spread_status_at_entry="normal",
            session="new_york",
            outcome="loss",
            r_multiple=-0.8,
            mfe=0.3,
            mae=-1.1,
            response_was_correct=False,
            blocker_prevented_loss=False,
            blocker_missed_profit=False,
        ),
    ]
    perf = summarize_performance_by_volatility(rows)
    assert perf["sample_size"] == 2
    assert "normal_volatility" in perf["classes"]


def test_ask_atlas_explains_current_volatility_state():
    class _Source:
        def load_status(self):
            return {
                "volatility_intelligence": {
                    "class": "elevated_volatility",
                    "confidence": 0.82,
                    "response_rule": "reduce_risk_require_higher_confluence",
                    "blockers": [],
                    "tags": ["elevated_risk"],
                    "explanation": "ATR above baseline",
                }
            }

        def load_events(self, limit: int = 160):
            return []

        def load_runtime_flags(self):
            return {}

        def load_comparison(self):
            return {}

    out = AskAtlasService(_Source()).answer("What volatility are we in?")
    assert "volatility class" in str(out.get("result", "")).lower()


def test_execution_gate_enforces_volatility_blockers():
    gate = ExecutionGate()
    reason = gate._safety_gate_reason(flags={"volatility_state": "extreme", "volatility_action": "block_new_entries"})
    assert reason == "volatility_block_active"


def test_volatility_recommendations_do_not_auto_change_live_logic():
    rows = [
        VolatilityOutcomeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            volatility_class_at_entry="chop_volatility",
            atr_ratio_at_entry=1.1,
            spread_status_at_entry="elevated",
            session="london",
            outcome="loss",
            r_multiple=-0.5,
            mfe=0.2,
            mae=-0.7,
            response_was_correct=False,
            blocker_prevented_loss=False,
            blocker_missed_profit=False,
        )
    ]
    perf = summarize_performance_by_volatility(rows)
    assert isinstance(perf.get("recommendations", []), list)
    assert "set learning_mode" not in " ".join(perf.get("recommendations", [])).lower()
