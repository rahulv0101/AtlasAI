from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool
from atlas_ai.strategies.minute_pack import GoldBreakoutRetestStrategy, MomentumContinuationStrategy
from atlas_ai.config.settings import Settings
from atlas_ai.trader import AtlasTrader


def _make_trader(timeframe: str = "M1") -> AtlasTrader:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        timeframe=timeframe,
        m1_allowed_sessions=("asia", "london", "new_york"),
        signal_regime_gate_min_trend_strength_pct=0.18,
        signal_regime_gate_low_volatility_max_pct=0.03,
        signal_trend_strength_weak_pct=0.12,
        signal_trend_strength_strong_pct=0.28,
        signal_trend_strength_weak_confidence_multiplier=0.55,
        signal_trend_strength_strong_confidence_boost=1.12,
        suppress_directional_signals_outside_sessions=True,
        signal_outside_session_confidence_multiplier=0.5,
        momentum_signal_min_trend_strength_pct=0.18,
        momentum_signal_min_volatility_pct=0.03,
    )
    trader._sanitize_signal = AtlasTrader._sanitize_signal.__get__(trader, AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._market_regime = AtlasTrader._market_regime.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._signal_volatility_state = AtlasTrader._signal_volatility_state.__get__(trader, AtlasTrader)
    trader._signal_generation_context = AtlasTrader._signal_generation_context.__get__(trader, AtlasTrader)
    trader._suppress_signal = AtlasTrader._suppress_signal.__get__(trader, AtlasTrader)
    trader._apply_signal_generation_gating = AtlasTrader._apply_signal_generation_gating.__get__(trader, AtlasTrader)
    return trader


def test_current_cycle_age_uses_cycle_start_before_status_publish() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._current_cycle_timing = {"cycle_started_at": datetime.now(timezone.utc).isoformat()}

    age_ms = AtlasTrader._current_cycle_age_ms(trader)
    payload = AtlasTrader._freshness_runtime_payload(status_age_ms=age_ms, max_age_ms=120000)

    assert age_ms is not None
    assert age_ms >= 0
    assert payload["status_age_ms"] is not None
    assert payload["is_stale"] is False


def test_execution_reject_does_not_count_as_trade_outcome() -> None:
    rejected = SimpleNamespace(success=False, retcode=-5, comment="missing_runtime_freshness", order_id=0, deal_id=0)
    accepted = SimpleNamespace(success=True, retcode=10009, comment="Request executed", order_id=123, deal_id=456)
    ambiguous = SimpleNamespace(success=True, retcode=10009, comment="Request executed", order_id=0, deal_id=0)

    assert AtlasTrader._execution_result_counts_as_trade(rejected) is False
    assert AtlasTrader._execution_result_counts_as_trade(accepted) is True
    assert AtlasTrader._execution_result_counts_as_trade(ambiguous) is False


def test_risk_snapshot_uses_daily_anchor_for_realized_drawdown(monkeypatch) -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(symbol="XAUUSD")
    trader._load_guard_state = AtlasTrader._load_guard_state.__get__(trader, AtlasTrader)

    monkeypatch.setattr(
        "atlas_ai.trader.mt5.account_info",
        lambda: SimpleNamespace(balance=900.0, equity=900.0),
    )
    monkeypatch.setattr("atlas_ai.trader.mt5.positions_get", lambda symbol=None: [])
    monkeypatch.setattr("atlas_ai.trader.mt5.symbol_info_tick", lambda symbol: SimpleNamespace(ask=1.2, bid=1.0))
    monkeypatch.setattr("atlas_ai.trader.mt5.symbol_info", lambda symbol: SimpleNamespace(point=0.1))

    snapshot = trader._risk_snapshot(
        {
            "risk_guard_state": {
                "start_balance": 1000.0,
                "week_start_balance": 1200.0,
            }
        }
    )

    assert round(snapshot.daily_loss_pct, 4) == 0.1
    assert round(snapshot.weekly_loss_pct, 4) == 0.25


def test_recent_side_penalty_does_not_punish_positive_win_rate_two_loss_streak() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    memory = {
        "recent_outcomes": [
            {"side": "sell", "outcome": "win"},
            {"side": "sell", "outcome": "win"},
            {"side": "sell", "outcome": "win"},
            {"side": "sell", "outcome": "win"},
            {"side": "sell", "outcome": "loss"},
            {"side": "sell", "outcome": "loss"},
        ]
    }

    penalty, reason = trader._recent_side_penalty("sell", memory)

    assert penalty == 0.0
    assert reason == ""


def test_recent_side_penalty_prefers_live_session_scope_over_old_backtest_rows() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    memory = {
        "recent_outcomes": [
            {
                "ticket": f"bt_{idx}",
                "side": "sell",
                "outcome": "loss",
                "session": "other",
                "post_trade_review": {"source": "historical_backtest"},
            }
            for idx in range(10)
        ]
        + [
            {"ticket": "live_1", "side": "sell", "outcome": "loss", "session": "london"},
            {"ticket": "live_2", "side": "sell", "outcome": "loss", "session": "london"},
            {"ticket": "live_3", "side": "sell", "outcome": "win", "session": "new_york"},
            {"ticket": "live_4", "side": "sell", "outcome": "win", "session": "new_york"},
            {"ticket": "live_5", "side": "sell", "outcome": "loss", "session": "new_york"},
        ]
    }

    penalty, reason = trader._recent_side_penalty("sell", memory, current_session="new_york")

    assert penalty == 0.0
    assert reason == ""


def test_recent_side_penalty_detail_explains_live_session_scope() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    memory = {
        "recent_outcomes": [
            {
                "ticket": f"bt_{idx}",
                "side": "sell",
                "outcome": "win",
                "session": "new_york",
                "post_trade_review": {"source": "historical_backtest"},
            }
            for idx in range(8)
        ]
        + [
            {"ticket": f"live_{idx}", "side": "sell", "outcome": "loss", "session": "new_york"}
            for idx in range(6)
        ]
    }

    detail = trader._recent_side_penalty_detail("sell", memory, current_session="new_york")

    assert detail["applied"] is True
    assert detail["source"] == "live"
    assert detail["scope"] == "side_session"
    assert detail["session"] == "new_york"
    assert detail["samples"] == 6
    assert detail["loss_streak"] == 6
    assert detail["penalty"] > 0


def test_recent_side_penalty_ignores_stale_live_losses() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._fresh_live_outcomes = AtlasTrader._fresh_live_outcomes.__get__(trader, AtlasTrader)
    memory = {
        "recent_outcomes": [
            {
                "ticket": f"live_old_{idx}",
                "side": "buy",
                "outcome": "loss",
                "session": "new_york",
                "entry_session": "new_york",
                "time": "2026-05-14T22:00:00+00:00",
            }
            for idx in range(6)
        ]
    }

    detail = trader._recent_side_penalty_detail("buy", memory, current_session="new_york")

    assert detail["applied"] is False
    assert detail["source"] == "stale_live_memory"
    assert detail["stale_live_excluded"] == 6


def test_hour_loss_penalty_ignores_stale_live_losses() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._fresh_live_outcomes = AtlasTrader._fresh_live_outcomes.__get__(trader, AtlasTrader)
    memory = {
        "recent_outcomes": [
            {
                "ticket": f"live_old_{idx}",
                "side": "buy",
                "outcome": "loss",
                "hour_utc": 16,
                "time": "2026-05-14T16:00:00+00:00",
            }
            for idx in range(6)
        ]
    }

    detail = trader._loss_pattern_penalty_detail("buy", datetime(2026, 5, 28, 16, tzinfo=timezone.utc), memory)

    assert detail["applied"] is False
    assert detail["source"] == "stale_live_memory"
    assert detail["stale_live_excluded"] == 6


def test_loss_learning_recovery_evidence_clears_corrected_setup() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    status = {
        "chosen": {"side": "sell", "confidence": 0.62},
        "thresholds": {"confidence": 0.44},
        "higher_timeframe": {"bias": "sell"},
        "same_side_support": {"count": 2},
        "context_snapshot": {
            "trend_structure": "LH/LL",
            "trend_state": "downtrend",
            "momentum": "strong",
            "location": "pullback",
            "setup_type_hint": "pullback",
        },
    }

    evidence = AtlasTrader._loss_learning_recovery_evidence(
        trader,
        side="sell",
        status=status,
        hour_detail={"applied": False},
        side_detail={"applied": True, "penalty": 0.15},
    )

    assert evidence["cleared"] is True
    assert evidence["reduction_factor"] == 0.0
    assert "strong_momentum" in evidence["confirmations"]
    assert "clean_entry_location" not in evidence["missing"]


def test_loss_learning_recovery_evidence_keeps_penalty_for_same_weak_pattern() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    status = {
        "chosen": {"side": "sell", "confidence": 0.49},
        "thresholds": {"confidence": 0.44},
        "higher_timeframe": {"bias": "sell"},
        "same_side_support": {"count": 1},
        "context_snapshot": {
            "trend_structure": "HH/HL",
            "trend_state": "uptrend",
            "momentum": "weak",
            "location": "mid_range",
            "setup_type_hint": "no_trade",
        },
    }

    evidence = AtlasTrader._loss_learning_recovery_evidence(
        trader,
        side="sell",
        status=status,
        hour_detail={"applied": False},
        side_detail={"applied": True, "penalty": 0.15},
    )

    assert evidence["cleared"] is False
    assert evidence["reduction_factor"] == 1.0
    assert "clean_entry_location" in evidence["missing"]
    assert "strong_momentum" in evidence["missing"]


def test_hour_loss_penalty_does_not_apply_aggregate_when_live_hour_samples_are_sparse() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    memory = {
        "side_hour_stats": {"buy:17": {"wins": 1, "losses": 3}},
        "recent_outcomes": [
            {
                "ticket": "live_1",
                "side": "buy",
                "hour_utc": 17,
                "outcome": "loss",
                "session": "new_york",
            }
        ],
    }

    detail = trader._loss_pattern_penalty_detail("buy", datetime(2026, 5, 25, 17, tzinfo=timezone.utc), memory)

    assert detail["applied"] is False
    assert detail["source"] == "live_recent"
    assert detail["samples"] == 1
    assert detail["insufficient_live_samples"] is True


def test_hour_loss_penalty_applies_with_enough_live_hour_samples() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    memory = {
        "recent_outcomes": [
            {"ticket": "live_1", "side": "buy", "hour_utc": 17, "outcome": "loss"},
            {"ticket": "live_2", "side": "buy", "hour_utc": 17, "outcome": "loss"},
            {"ticket": "live_3", "side": "buy", "hour_utc": 17, "outcome": "loss"},
            {"ticket": "live_4", "side": "buy", "hour_utc": 17, "outcome": "win"},
        ],
    }

    detail = trader._loss_pattern_penalty_detail("buy", datetime(2026, 5, 25, 17, tzinfo=timezone.utc), memory)

    assert detail["applied"] is True
    assert detail["source"] == "live_recent"
    assert detail["loss_rate"] == 0.75
    assert detail["penalty"] > 0


def test_directional_structure_guard_blocks_wrong_side_mid_range_sell() -> None:
    reason = AtlasTrader._directional_structure_block_reason(
        side="sell",
        chart_features={"trend_structure": "HH/HL", "trend_state": "uptrend", "location_state": "mid_range", "momentum_state": "mixed"},
        context_snapshot={},
        reversal_detection={"confirmed_reversal": False, "reversal_bias": "neutral"},
    )

    assert "Directional-structure guard blocked trade" in reason
    assert "structure=HH/HL" in reason


def test_directional_structure_guard_allows_confirmed_reversal() -> None:
    reason = AtlasTrader._directional_structure_block_reason(
        side="sell",
        chart_features={"trend_structure": "HH/HL", "trend_state": "uptrend", "location_state": "range_high", "momentum_state": "strong"},
        context_snapshot={},
        reversal_detection={"confirmed_reversal": True, "reversal_bias": "sell"},
    )

    assert reason == ""


def test_directional_structure_guard_allows_aligned_side() -> None:
    reason = AtlasTrader._directional_structure_block_reason(
        side="buy",
        chart_features={"trend_structure": "HH/HL", "trend_state": "uptrend", "location_state": "pullback", "momentum_state": "strong"},
        context_snapshot={},
        reversal_detection={},
    )

    assert reason == ""


def test_guarded_demo_auto_approval_requires_tiny_clean_demo_setup() -> None:
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(symbol="XAUUSD")
    status = {
        "chosen": {"side": "buy", "confidence": 0.7},
        "min_confidence_threshold": 0.44,
        "safety_layer": {"safety_lock_active": False, "cooldown_active": False},
        "directional_structure_guard": {"hard_blocked": False},
    }

    out = AtlasTrader._guarded_demo_auto_approval_decision(
        trader,
        environment="demo",
        status=status,
        setup_family="continuation",
        volume=0.01,
        runtime_flags={},
    )
    too_big = AtlasTrader._guarded_demo_auto_approval_decision(
        trader,
        environment="demo",
        status=status,
        setup_family="continuation",
        volume=0.02,
        runtime_flags={},
    )

    assert out["approved"] is True
    assert too_big["approved"] is False
    assert "size_above_guarded_demo_cap" in too_big["reasons"]


def _low_energy_df(rows: int = 80) -> pd.DataFrame:
    data = []
    base = 2000.0
    for i in range(rows):
        price = base + ((i % 4) - 2) * 0.04
        data.append(
            {
                "open": price,
                "high": price + 0.06,
                "low": price - 0.06,
                "close": price + (0.01 if i % 2 == 0 else -0.01),
                "tick_volume": 100 + (i % 5),
            }
        )
    return pd.DataFrame(data)


def _strong_trend_df(rows: int = 120) -> pd.DataFrame:
    data = []
    base = 2000.0
    for i in range(rows):
        open_px = base + (i * 1.9)
        close_px = open_px + 1.4 + ((i % 3) * 0.2)
        data.append(
            {
                "open": open_px,
                "high": close_px + 0.8,
                "low": open_px - 0.6,
                "close": close_px,
                "tick_volume": 120 + ((i % 7) * 8),
            }
        )
    return pd.DataFrame(data)


def _breakout_retest_df(side: str = "sell") -> pd.DataFrame:
    data = []
    base = 2000.0
    for i in range(44):
        center = base - (i * 0.18 if side == "sell" else -i * 0.18)
        data.append(
            {
                "open": center + 0.15,
                "high": center + 1.0,
                "low": center - 1.0,
                "close": center - 0.15,
                "tick_volume": 110 + (i % 6) * 4,
            }
        )
    if side == "sell":
        level = min(row["low"] for row in data[-28:])
        sequence = [
            (level - 0.4, level + 0.2, level - 1.8, level - 1.2, 150),
            (level - 1.1, level - 0.1, level - 1.5, level - 0.3, 145),
            (level - 0.2, level + 0.8, level - 0.6, level - 0.4, 142),
            (level - 0.5, level + 0.4, level - 0.9, level - 0.6, 144),
            (level - 0.7, level - 0.1, level - 1.1, level - 0.9, 146),
            (level - 0.8, level - 0.4, level - 2.2, level - 1.9, 170),
        ]
    else:
        level = max(row["high"] for row in data[-28:])
        sequence = [
            (level + 0.4, level + 1.8, level - 0.2, level + 1.2, 150),
            (level + 1.1, level + 1.5, level + 0.1, level + 0.3, 145),
            (level + 0.2, level + 0.6, level - 0.8, level + 0.4, 142),
            (level + 0.5, level + 0.9, level - 0.4, level + 0.6, 144),
            (level + 0.7, level + 1.1, level + 0.1, level + 0.9, 146),
            (level + 0.8, level + 2.2, level + 0.4, level + 1.9, 170),
        ]
    for open_px, high_px, low_px, close_px, volume in sequence:
        data.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tick_volume": volume,
            }
        )
    return pd.DataFrame(data)


def _raw_breakout_without_retest_df() -> pd.DataFrame:
    df = _breakout_retest_df("sell")
    low_level = float(df.iloc[:28]["low"].min())
    for idx in range(len(df) - 6, len(df)):
        df.loc[idx, "open"] = low_level - 1.2 - (idx % 3) * 0.2
        df.loc[idx, "high"] = low_level - 0.8
        df.loc[idx, "low"] = low_level - 2.3
        df.loc[idx, "close"] = low_level - 1.9
    return df


def test_session_label_keeps_london_and_new_york_windows_clear():
    trader = _make_trader()

    assert trader._session_label(datetime(2026, 4, 16, 8, tzinfo=timezone.utc)) == "london"
    assert trader._session_label(datetime(2026, 4, 16, 15, tzinfo=timezone.utc)) == "new_york"
    assert trader._session_label(datetime(2026, 4, 16, 21, tzinfo=timezone.utc)) == "late_us"


def test_signal_generation_context_marks_low_energy_ranges():
    trader = _make_trader()

    context = trader._signal_generation_context(
        _low_energy_df(),
        when_utc=datetime(2026, 4, 16, 9, tzinfo=timezone.utc),
    )

    assert context["session"] == "london"
    assert context["regime"] in {"range", "unknown"}
    assert context["volatility_state"] == "low"
    assert context["trend_strength"] < 0.18


def test_low_energy_context_suppresses_directional_signal_generation():
    trader = _make_trader()
    signal = TradeSignal("gold_micro_impulse", "buy", 0.44, 100.0, 150.0)

    conditioned = trader._apply_signal_generation_gating(
        signal,
        {
            "session": "london",
            "allowed_session": True,
            "regime": "range",
            "trend_strength": 0.08,
            "volatility_pct": 0.01,
            "volatility_state": "low",
        },
    )

    assert conditioned.side == "flat"
    assert conditioned.confidence == 0.0


def test_low_energy_context_keeps_directional_candidate_in_learning_mode():
    trader = _make_trader()
    trader.settings.learning_mode = True
    signal = TradeSignal("gold_micro_impulse", "buy", 0.50, 100.0, 150.0)

    conditioned = trader._apply_signal_generation_gating(
        signal,
        {
            "session": "london",
            "allowed_session": True,
            "regime": "range",
            "trend_strength": 0.08,
            "volatility_pct": 0.01,
            "volatility_state": "low",
        },
    )

    assert conditioned.side == "buy"
    assert 0.0 < conditioned.confidence < 0.50


def test_demo_active_chromosome_signals_load_when_evolution_scheduler_disabled(tmp_path, monkeypatch):
    class Store:
        base = str(tmp_path)

        def load_learning_state(self):
            return {"state": {}}

        def log_event(self, *_args, **_kwargs):
            return None

    class DemoStrategy:
        name = "demo_chromosome_strategy"

        def generate_signal(self, _df):
            return TradeSignal(self.name, "buy", 0.62, 100.0, 160.0)

    chromosome = StrategyChromosome(
        "demo_candidate",
        1,
        make_standard_gene_pool(),
        promotion_stage="demo",
        demo_active=True,
        backtest_completed=True,
        backtest_score=70.0,
        shadow_passed=True,
    )
    (tmp_path / "evolution_population.json").write_text(
        __import__("json").dumps({"chromosomes": [chromosome.to_dict()]}),
        encoding="utf-8",
    )

    trader = _make_trader()
    trader.settings.storage_dir = str(tmp_path)
    trader.settings.max_demo_strategies = 3
    trader.settings.evolution_max_demo_drawdown_pct = 0.05
    trader.settings.evolution_max_shadow_drawdown_pct = 0.05
    trader.settings.evolution_enabled = False
    trader.settings.learning_mode = True
    trader.base_strategies = []
    trader.store = Store()
    trader._logged_evolved_fallback = True
    trader._load_evolved_strategy_pool = lambda: []
    trader._format_evolved_strategy_name = AtlasTrader._format_evolved_strategy_name.__get__(trader, AtlasTrader)
    trader._load_chromosome_demo_pool = AtlasTrader._load_chromosome_demo_pool.__get__(trader, AtlasTrader)
    trader._build_signals = AtlasTrader._build_signals.__get__(trader, AtlasTrader)
    monkeypatch.setattr("atlas_ai.trader.StrategyFactory.build", lambda _chromosome: DemoStrategy())

    df = pd.DataFrame({
        "open": [2000.0 + i * 0.2 for i in range(140)],
        "high": [2001.0 + i * 0.2 for i in range(140)],
        "low": [1999.0 + i * 0.2 for i in range(140)],
        "close": [2000.5 + i * 0.2 for i in range(140)],
        "tick_volume": [120 + (i % 15) for i in range(140)],
        "time": pd.date_range("2026-06-02 13:00:00", periods=140, freq="1min"),
    })

    signals = trader._build_signals(df)

    assert any(signal.strategy == "demo_chromosome_strategy" for signal in signals)
    assert trader._evolved_strategy_status["active_count"] == 1


def test_outside_allowed_sessions_suppress_directional_signals():
    trader = _make_trader()
    signal = TradeSignal("gold_micro_impulse", "sell", 0.52, 100.0, 150.0)

    conditioned = trader._apply_signal_generation_gating(
        signal,
        {
            "session": "asia",
            "allowed_session": False,
            "regime": "trend",
            "trend_strength": 0.34,
            "volatility_pct": 0.04,
            "volatility_state": "normal",
        },
    )

    assert conditioned.side == "flat"
    assert conditioned.confidence == 0.0


def test_strong_trend_context_boosts_directional_signal_confidence():
    trader = _make_trader()
    signal = TradeSignal("gold_micro_impulse", "buy", 0.50, 100.0, 150.0)

    conditioned = trader._apply_signal_generation_gating(
        signal,
        {
            "session": "new_york",
            "allowed_session": True,
            "regime": "trend",
            "trend_strength": 0.36,
            "volatility_pct": 0.045,
            "volatility_state": "normal",
        },
    )

    assert conditioned.side == "buy"
    assert conditioned.confidence > 0.50


def test_momentum_continuation_stays_flat_in_low_energy_conditions():
    signal = MomentumContinuationStrategy().generate_signal(_low_energy_df())

    assert signal.side == "flat"
    assert signal.confidence == 0.0


def test_momentum_continuation_only_fires_in_strong_trend_conditions():
    signal = MomentumContinuationStrategy().generate_signal(_strong_trend_df())

    assert signal.side in {"buy", "sell"}
    assert signal.confidence > 0.0


def test_gold_breakout_retest_requires_retest_confirmation():
    signal = GoldBreakoutRetestStrategy().generate_signal(_raw_breakout_without_retest_df())

    assert signal.side == "flat"
    assert signal.confidence == 0.0


def test_gold_breakout_retest_detects_confirmed_sell_retest():
    signal = GoldBreakoutRetestStrategy().generate_signal(_breakout_retest_df("sell"))

    assert signal.side == "sell"
    assert signal.confidence > 0.0


def test_default_m1_enabled_strategies_include_continuation_coverage():
    enabled = set(Settings(hfm_login=1, hfm_password="x", hfm_server="demo").enabled_strategies)

    assert "momentum_continuation" in enabled
    assert "gold_micro_impulse" in enabled
    assert "gold_breakout_retest" in enabled
    assert "gold_liquidity_sweep" in enabled
    assert "ema_crossover" in enabled
    assert "pullback_trend" in enabled
