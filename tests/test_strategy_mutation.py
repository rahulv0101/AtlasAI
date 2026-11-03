from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.learning.online_allocator import StrategyState
from atlas_ai.regime_filters import research_regime_filter_reason
from atlas_ai.setup_quality import score_setup_quality
from atlas_ai.strategies.strategy_lab import EvolvedRuleStrategy, EvolvedStrategyCandidate, StrategyLab
from atlas_ai.strategies.built_in import EMACrossoverStrategy
from atlas_ai.trader import AtlasTrader


def _sample_df(rows: int = 520) -> pd.DataFrame:
    base = 2000.0
    data = []
    for i in range(rows):
        drift = 0.18 * i
        wave_fast = ((i % 18) - 9) * 2.4
        wave_slow = ((i % 42) - 21) * 1.1
        wobble = wave_fast - wave_slow
        open_px = base + drift + wobble
        close_px = open_px + (2.2 if i % 4 in {1, 2} else -2.0)
        high_px = max(open_px, close_px) + 2.0
        low_px = min(open_px, close_px) - 2.0
        data.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "tick_volume": 100 + (i % 11) * 10,
            }
        )
    return pd.DataFrame(data)


def test_strategy_lab_discovers_ranked_unique_candidates():
    df = _sample_df()
    lab = StrategyLab(seed=11)

    candidates = lab.discover_candidates(df, generations=4, population=10, survivors=3)

    assert len(candidates) == 3
    assert [c.score for c in candidates] == sorted([c.score for c in candidates], reverse=True)
    assert len({c.signature() for c in candidates}) == len(candidates)
    assert all(c.trades > 0 for c in candidates)
    assert all(0.0 <= c.win_rate <= 1.0 for c in candidates)


def test_strategy_lab_discover_candidates_honors_score_adjuster(monkeypatch):
    df = _sample_df()
    lab = StrategyLab(seed=11)

    def fake_evaluate(strategy, frame):
        if int(strategy.fast) <= 20:
            return {
                "score": 0.50,
                "trades": 210,
                "wins": 110,
                "losses": 100,
                "win_rate": 110 / 210,
                "profit_component": 0.50,
                "confidence_floor": 0.6,
                "trend_trades": 80,
                "trend_wins": 30,
                "trend_losses": 50,
                "trend_win_rate": 0.375,
                "trend_profit_component": -0.05,
                "range_trades": 130,
                "range_wins": 80,
                "range_losses": 50,
                "range_win_rate": 80 / 130,
                "range_profit_component": 0.12,
            }
        return {
            "score": 0.45,
            "trades": 210,
            "wins": 118,
            "losses": 92,
            "win_rate": 118 / 210,
            "profit_component": 0.45,
            "confidence_floor": 0.6,
            "trend_trades": 90,
            "trend_wins": 55,
            "trend_losses": 35,
            "trend_win_rate": 55 / 90,
            "trend_profit_component": 0.09,
            "range_trades": 120,
            "range_wins": 63,
            "range_losses": 57,
            "range_win_rate": 63 / 120,
            "range_profit_component": 0.03,
        }

    monkeypatch.setattr(lab, "evaluate_strategy", fake_evaluate)

    def adjuster(metrics):
        return float(metrics["score"]) + (0.5 * float(metrics["trend_profit_component"])) - (0.3 * max(0.0, float(metrics["range_profit_component"]) - float(metrics["trend_profit_component"])))

    candidates = lab.discover_candidates(df, generations=2, population=6, survivors=2, score_adjuster=adjuster)

    assert len(candidates) == 2
    assert candidates[0].fast > 20
    assert candidates[0].trend_edge_score > 0.0
    assert candidates[0].range_dependency_score == 0.0
    assert "trend edge is usable" in candidates[0].explainability_summary


def test_evolved_candidate_explainability_marks_trend_weakness():
    metrics = {
        "trades": 200,
        "trend_trades": 120,
        "trend_wins": 45,
        "trend_losses": 75,
        "trend_win_rate": 45 / 120,
        "trend_profit_component": -0.04,
        "range_win_rate": 0.62,
        "range_profit_component": 0.09,
    }

    summary = EvolvedStrategyCandidate.explainability_from_metrics(metrics)

    assert summary["trend_edge_score"] < 0.0
    assert summary["range_dependency_score"] > 0.0
    assert summary["stop_hunt_vulnerability"] > 0.1
    assert "trend edge is weak" in summary["explainability_summary"]


def test_trader_build_signals_uses_promoted_evolved_pool(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "evolved_strategy_pool": {
                "count": 2,
                "candidates": [
                    {"fast": 8, "slow": 26, "rsi_buy": 44.0, "rsi_sell": 56.0},
                    {"fast": 12, "slow": 34, "rsi_buy": 41.5, "rsi_sell": 58.5},
                ],
            }
        }
    )

    trader = AtlasTrader.__new__(AtlasTrader)
    trader.store = store
    trader.settings = SimpleNamespace(evolved_strategy_survivors=4)
    trader.base_strategies = []
    trader._logged_evolved_fallback = False
    trader._format_evolved_strategy_name = AtlasTrader._format_evolved_strategy_name.__get__(trader, AtlasTrader)
    trader._sanitize_signal = AtlasTrader._sanitize_signal.__get__(trader, AtlasTrader)
    trader._load_evolved_strategy_pool = AtlasTrader._load_evolved_strategy_pool.__get__(trader, AtlasTrader)
    trader._build_signals = AtlasTrader._build_signals.__get__(trader, AtlasTrader)

    df = _sample_df()
    signals = trader._build_signals(df)

    evolved = [s for s in signals if str(s.strategy).startswith("evolved_rule(")]
    assert len(evolved) == 2
    assert all(isinstance(s, TradeSignal) for s in evolved)
    assert any("fast=8,slow=26" in s.strategy for s in evolved)
    assert any("fast=12,slow=34" in s.strategy for s in evolved)


def test_evolved_rule_flat_signal_has_zero_confidence():
    df = _sample_df()
    strategy = EvolvedRuleStrategy(fast=8, slow=26, rsi_buy=10.0, rsi_sell=90.0)

    signal = strategy.generate_signal(df)

    assert signal.side == "flat"
    assert signal.confidence == 0.0


def test_trader_sanitizes_non_actionable_signal_confidence():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._sanitize_signal = AtlasTrader._sanitize_signal.__get__(trader, AtlasTrader)

    signal = TradeSignal("rogue_flat", "flat", 1.0, 100, 150)

    sanitized = trader._sanitize_signal(signal)

    assert sanitized.side == "flat"
    assert sanitized.confidence == 0.0


def test_trader_sanitizes_evolved_signal_confidence_ceiling():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._sanitize_signal = AtlasTrader._sanitize_signal.__get__(trader, AtlasTrader)

    signal = TradeSignal("evolved_rule(fast=13,slow=82,rsi_buy=47.7,rsi_sell=40.0)", "buy", 1.0, 100, 150)

    sanitized = trader._sanitize_signal(signal)

    assert sanitized.side == "buy"
    assert 0.79 <= sanitized.confidence <= 0.82


def test_trader_compact_status_views_are_trimmed():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._top_signal_rows = AtlasTrader._top_signal_rows.__get__(trader, AtlasTrader)
    trader._market_status_view = AtlasTrader._market_status_view.__get__(trader, AtlasTrader)
    trader._contextual_ranking_view = AtlasTrader._contextual_ranking_view.__get__(trader, AtlasTrader)

    signals = [
        TradeSignal(f"s{i}", "buy" if i % 3 == 0 else "flat", 0.1 * i, 100, 150)
        for i in range(10)
    ]
    scores = {f"s{i}": float(i) / 10.0 for i in range(10)}
    top_rows = trader._top_signal_rows(signals, contextual_scores=scores, limit=5)

    assert len(top_rows) == 5
    assert all("contextual_score" in row for row in top_rows)

    market = trader._market_status_view(list(range(30)), [str(i) for i in range(30)], limit=20)
    assert len(market["recent_close"]) == 20
    assert len(market["recent_time"]) == 20
    assert market["last_close"] == 29

    details = {
        f"s{i}": {
            "strategy": f"s{i}",
            "final_score": float(i),
            "factor": 1.0,
            "reasons": ["a", "b", "c", "d", "e", "f"],
        }
        for i in range(6)
    }
    ranking = trader._contextual_ranking_view(details, chosen_strategy="s5", contextual_scores=scores)
    assert len(ranking["top"]) == 3
    assert ranking["chosen_score"] == scores["s5"]
    assert all(len(row["reasons"]) <= 5 for row in ranking["top"])


def test_trader_compact_runtime_event_meta_keeps_best_actionable_summary():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._compact_runtime_event_meta = AtlasTrader._compact_runtime_event_meta.__get__(trader, AtlasTrader)

    status = {
        "symbol": "XAUUSD",
        "timeframe": "M1",
        "action": "hold",
        "message": "No high-confidence trade found.",
        "chosen": {"strategy": "ema_crossover", "side": "flat", "confidence": 0.0},
        "risk": {
            "equity": 1000.0,
            "balance": 1000.0,
            "daily_loss_pct": 0.0,
            "open_positions": 0,
            "spread_points": 20.0,
            "trades_today": 1,
            "consecutive_losses": 0,
            "daily_profit_zar": 0.0,
        },
        "signal_summary": {"raw_count": 10, "active_count": 4},
        "higher_timeframe": {"bias": "buy"},
        "drift": {"active": True},
        "contextual_ranking": {"top": [{"strategy": "s1"}]},
        "best_actionable_candidate": {"strategy": "gold_liquidity_sweep", "side": "buy", "score": 0.42},
        "signals": [{"strategy": "heavy"}],
        "signals_active": [{"strategy": "heavy2"}],
        "market": {"recent_close": list(range(20))},
    }

    compact = trader._compact_runtime_event_meta(status, "decision")

    assert compact["best_actionable_candidate"]["strategy"] == "gold_liquidity_sweep"
    assert "signals" not in compact
    assert "signals_active" not in compact
    assert "market" not in compact


def test_trader_research_regime_overlay_meta_is_disabled_by_default():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = SimpleNamespace(
        research_regime_filter_enabled=False,
        research_regime_filter_low_liquidity=True,
        research_regime_filter_abnormal_spike=True,
        research_regime_filter_post_spike=False,
        research_regime_filter_london_extreme_volatility=False,
        research_regime_filter_weak_trend=False,
        research_regime_filter_trend_low_volatility=False,
        research_regime_filter_trend_late_us=False,
    )
    trader._research_regime_overlay_meta = AtlasTrader._research_regime_overlay_meta.__get__(trader, AtlasTrader)
    df = _sample_df()
    snapshot = SimpleNamespace(spread_points=18.0)

    meta = trader._research_regime_overlay_meta(df, snapshot, pd.Timestamp("2026-04-03T10:00:00Z").to_pydatetime())

    assert meta["enabled"] is False
    assert research_regime_filter_reason(trader.settings, meta) == ""


def test_research_regime_filter_reason_blocks_weak_trend_when_enabled():
    settings = SimpleNamespace(
        research_regime_filter_enabled=True,
        research_regime_filter_low_liquidity=False,
        research_regime_filter_abnormal_spike=False,
        research_regime_filter_post_spike=False,
        research_regime_filter_london_extreme_volatility=False,
        research_regime_low_liquidity_volume_ratio=0.75,
        research_regime_abnormal_spike_atr_multiple=2.2,
        research_regime_filter_weak_trend=True,
        research_regime_trend_min_strength_pct=0.20,
        research_regime_filter_trend_low_volatility=False,
        research_regime_trend_low_volatility_max_pct=0.03,
        research_regime_filter_trend_late_us=False,
    )
    reason = research_regime_filter_reason(
        settings,
        {
            "regime_name": "trend",
            "trend_strength_pct": 0.12,
            "volatility_pct": 0.025,
            "session": "london",
            "volume_ratio": 1.0,
            "spike_ratio": 1.0,
        },
    )
    assert "weak trend" in reason


def test_research_regime_filter_reason_blocks_post_spike_when_enabled():
    settings = SimpleNamespace(
        research_regime_filter_enabled=True,
        research_regime_filter_low_liquidity=False,
        research_regime_filter_abnormal_spike=False,
        research_regime_filter_post_spike=True,
        research_regime_post_spike_atr_multiple=1.6,
        research_regime_filter_london_extreme_volatility=False,
        research_regime_filter_weak_trend=False,
        research_regime_filter_trend_low_volatility=False,
        research_regime_filter_trend_late_us=False,
    )
    reason = research_regime_filter_reason(
        settings,
        {
            "regime_name": "range",
            "trend_strength_pct": 0.0,
            "volatility_pct": 0.05,
            "session": "london",
            "volume_ratio": 1.0,
            "spike_ratio": 1.8,
        },
    )
    assert "post-spike" in reason


def test_research_regime_filter_reason_blocks_london_extreme_volatility_when_enabled():
    settings = SimpleNamespace(
        research_regime_filter_enabled=True,
        research_regime_filter_low_liquidity=False,
        research_regime_filter_abnormal_spike=False,
        research_regime_filter_post_spike=False,
        research_regime_filter_london_extreme_volatility=True,
        research_regime_london_low_volatility_max_pct=0.03,
        research_regime_london_high_volatility_min_pct=0.07,
        research_regime_filter_weak_trend=False,
        research_regime_filter_trend_low_volatility=False,
        research_regime_filter_trend_late_us=False,
    )
    reason = research_regime_filter_reason(
        settings,
        {
            "regime_name": "trend",
            "trend_strength_pct": 0.4,
            "volatility_pct": 0.075,
            "session": "london",
            "volume_ratio": 1.0,
            "spike_ratio": 1.0,
        },
    )
    assert "london extreme volatility" in reason


def test_setup_quality_scores_post_spike_as_lower_quality():
    settings = SimpleNamespace(
        setup_quality_enabled=True,
        setup_quality_penalty_cap=0.18,
        setup_quality_boost_cap=0.10,
        setup_quality_min_bucket_samples=6,
        setup_quality_strong_trend_min_pct=0.20,
        setup_quality_trend_low_volatility_max_pct=0.03,
        setup_quality_london_high_volatility_min_pct=0.07,
        setup_quality_post_spike_atr_multiple=1.6,
        max_spread_points=80.0,
    )
    result = score_setup_quality(
        settings=settings,
        context={
            "strategy": "momentum_continuation",
            "side": "buy",
            "session": "new_york",
            "hour_utc": 15,
            "regime": "trend",
            "higher_tf_bias": "buy",
            "spread_points": 12.0,
            "volatility_pct": 0.05,
            "trend_strength": 0.24,
            "volume_ratio": 1.0,
            "body_ratio": 0.42,
            "same_side_ratio": 0.5,
            "spike_ratio": 1.9,
            "htf_aligned": True,
        },
        recent_outcomes=[],
    )
    assert result["enabled"] is True
    assert result["factor"] < 1.0
    assert "post_spike_entry" in result["reasons"]


def test_setup_quality_uses_recent_outcomes_as_evidence():
    settings = SimpleNamespace(
        setup_quality_enabled=True,
        setup_quality_penalty_cap=0.18,
        setup_quality_boost_cap=0.10,
        setup_quality_min_bucket_samples=4,
        setup_quality_strong_trend_min_pct=0.20,
        setup_quality_trend_low_volatility_max_pct=0.03,
        setup_quality_london_high_volatility_min_pct=0.07,
        setup_quality_post_spike_atr_multiple=1.6,
        max_spread_points=80.0,
    )
    recent_outcomes = [
        {"strategy": "momentum_continuation", "side": "buy", "session": "asia", "hour_utc": 2, "outcome": "win"},
        {"strategy": "momentum_continuation", "side": "buy", "session": "asia", "hour_utc": 2, "outcome": "win"},
        {"strategy": "momentum_continuation", "side": "buy", "session": "asia", "hour_utc": 2, "outcome": "win"},
        {"strategy": "momentum_continuation", "side": "buy", "session": "asia", "hour_utc": 2, "outcome": "loss"},
    ]
    result = score_setup_quality(
        settings=settings,
        context={
            "strategy": "momentum_continuation",
            "side": "buy",
            "session": "asia",
            "hour_utc": 2,
            "regime": "trend",
            "higher_tf_bias": "buy",
            "spread_points": 10.0,
            "volatility_pct": 0.05,
            "trend_strength": 0.35,
            "volume_ratio": 1.1,
            "body_ratio": 0.5,
            "same_side_ratio": 0.6,
            "spike_ratio": 1.0,
            "htf_aligned": True,
        },
        recent_outcomes=recent_outcomes,
    )
    assert result["factor"] >= 1.0
    assert result["evidence"]["strategy_side"]["samples"] == 4


def test_trader_filter_weak_strategies_hard_prunes_chronic_losers():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._filter_weak_strategies = AtlasTrader._filter_weak_strategies.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        win_rate_focus_enabled=True,
        min_strategy_samples=8,
        min_strategy_win_rate=0.2,
        hard_prune_weak_strategies_enabled=True,
        hard_prune_min_samples=120,
        hard_prune_max_win_rate=0.43,
        hard_prune_require_negative_pnl=True,
        severe_weak_strategy_min_samples=5000,
        severe_weak_strategy_max_win_rate=0.28,
    )
    trader.allocator = SimpleNamespace(
        stats=lambda: {
            "macd_trend": StrategyState(wins=40, losses=160, pnl=-2500.0),
            "gold_liquidity_sweep": StrategyState(wins=3, losses=2, pnl=50.0),
        }
    )

    signals = [
        TradeSignal("macd_trend", "sell", 0.42, 100, 160),
        TradeSignal("gold_liquidity_sweep", "buy", 0.39, 100, 160),
    ]

    kept, blocked = trader._filter_weak_strategies(signals)

    assert [signal.strategy for signal in kept] == ["gold_liquidity_sweep"]
    assert any("macd_trend hard_pruned" in item for item in blocked)


def test_trader_filter_weak_strategies_keeps_new_evolved_candidates():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._filter_weak_strategies = AtlasTrader._filter_weak_strategies.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        win_rate_focus_enabled=True,
        min_strategy_samples=8,
        min_strategy_win_rate=0.2,
        hard_prune_weak_strategies_enabled=True,
        hard_prune_min_samples=120,
        hard_prune_max_win_rate=0.43,
        hard_prune_require_negative_pnl=True,
        severe_weak_strategy_min_samples=5000,
        severe_weak_strategy_max_win_rate=0.28,
    )
    trader.allocator = SimpleNamespace(
        stats=lambda: {
            "evolved_rule(fast=23,slow=86,rsi_buy=40.7,rsi_sell=56.3)": StrategyState(wins=2, losses=3, pnl=-10.0),
        }
    )

    signal = TradeSignal("evolved_rule(fast=23,slow=86,rsi_buy=40.7,rsi_sell=56.3)", "buy", 0.41, 100, 160)

    kept, blocked = trader._filter_weak_strategies([signal])

    assert [item.strategy for item in kept] == [signal.strategy]
    assert blocked == []


def test_ema_crossover_stays_directional_after_fresh_cross():
    df = _sample_df()
    strategy = EMACrossoverStrategy(fast=8, slow=21)

    signal = strategy.generate_signal(df)

    assert signal.side in {"buy", "sell"}
    assert signal.confidence > 0.0


def test_trader_filter_weak_strategies_keeps_momentum_continuation_out_of_hard_prune():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._filter_weak_strategies = AtlasTrader._filter_weak_strategies.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        win_rate_focus_enabled=True,
        min_strategy_samples=8,
        min_strategy_win_rate=0.35,
        hard_prune_weak_strategies_enabled=True,
        hard_prune_min_samples=120,
        hard_prune_max_win_rate=0.43,
        hard_prune_require_negative_pnl=True,
    )
    trader.allocator = SimpleNamespace(
        stats=lambda: {
            "momentum_continuation": StrategyState(wins=725, losses=1290, pnl=-1805.86),
        }
    )

    signal = TradeSignal("momentum_continuation", "sell", 0.42, 100, 160)

    kept, blocked = trader._filter_weak_strategies([signal])

    assert [item.strategy for item in kept] == ["momentum_continuation"]
    assert not any("momentum_continuation hard_pruned" in item for item in blocked)


def test_trader_filter_weak_strategies_keeps_ema_crossover_out_of_hard_prune():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._filter_weak_strategies = AtlasTrader._filter_weak_strategies.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        win_rate_focus_enabled=True,
        min_strategy_samples=8,
        min_strategy_win_rate=0.2,
        hard_prune_weak_strategies_enabled=True,
        hard_prune_min_samples=120,
        hard_prune_max_win_rate=0.43,
        hard_prune_require_negative_pnl=True,
        severe_weak_strategy_min_samples=5000,
        severe_weak_strategy_max_win_rate=0.28,
    )
    trader.allocator = SimpleNamespace(
        stats=lambda: {
            "ema_crossover": StrategyState(wins=300, losses=1100, pnl=-1284.14),
        }
    )

    signal = TradeSignal("ema_crossover", "sell", 0.42, 100, 160)

    kept, blocked = trader._filter_weak_strategies([signal])

    assert [item.strategy for item in kept] == ["ema_crossover"]
    assert not any("ema_crossover hard_pruned" in item for item in blocked)


def test_trader_filter_weak_strategies_severely_blocks_chronic_loser_even_if_exempt():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._filter_weak_strategies = AtlasTrader._filter_weak_strategies.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        win_rate_focus_enabled=True,
        min_strategy_samples=8,
        min_strategy_win_rate=0.35,
        hard_prune_weak_strategies_enabled=True,
        hard_prune_min_samples=120,
        hard_prune_max_win_rate=0.43,
        hard_prune_require_negative_pnl=True,
        severe_weak_strategy_min_samples=500,
        severe_weak_strategy_max_win_rate=0.28,
    )
    trader.allocator = SimpleNamespace(
        stats=lambda: {
            "ema_crossover": StrategyState(wins=309, losses=1098, pnl=-12897.81),
        }
    )

    kept, blocked = trader._filter_weak_strategies([TradeSignal("ema_crossover", "buy", 0.4, 100, 160)])

    assert kept == []
    assert any("ema_crossover severe_blocked" in item for item in blocked)


def test_consensus_support_can_count_raw_directional_corrobation_after_weak_filter():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._same_side_support_count = AtlasTrader._same_side_support_count.__get__(trader, AtlasTrader)
    raw_signals = [
        TradeSignal("evolved_rule(fast=21,slow=86,rsi_buy=45.9,rsi_sell=58.2)", "sell", 1.0, 100, 160),
        TradeSignal("ema_crossover", "sell", 0.35, 100, 160),
        TradeSignal("pullback_trend", "sell", 0.06, 100, 160),
    ]
    active_signals = [raw_signals[0]]

    active_support = trader._same_side_support_count(active_signals, "sell")
    raw_support = trader._same_side_support_count(raw_signals, "sell")

    assert active_support == 1
    assert raw_support == 3
    assert max(active_support, raw_support) >= 2


def test_contextual_signal_scores_penalize_htf_opposed_spread_combo():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._session_label = AtlasTrader._session_label.__get__(trader, AtlasTrader)
    trader._signal_micro_context = AtlasTrader._signal_micro_context.__get__(trader, AtlasTrader)
    trader._build_signal_context = AtlasTrader._build_signal_context.__get__(trader, AtlasTrader)
    trader._same_side_support_count = AtlasTrader._same_side_support_count.__get__(trader, AtlasTrader)
    trader._contextual_signal_scores = AtlasTrader._contextual_signal_scores.__get__(trader, AtlasTrader)
    trader._meta_model_probability = lambda context: (0.69, "ok")
    trader._meta_model_effect_scale = lambda: 0.2
    trader.settings = SimpleNamespace(
        min_same_side_signals=2,
        max_spread_points=80.0,
        htf_spread_combo_penalty=0.78,
        timeframe="M1",
    )
    trader.allocator = SimpleNamespace(score_signal=lambda signal: float(signal.confidence))

    df = _sample_df()
    sell_signal = TradeSignal("rsi_mean_reversion", "sell", 0.41, 100, 160)
    buy_signal = TradeSignal("momentum_continuation", "buy", 0.30, 100, 160)
    scores, details = trader._contextual_signal_scores(
        signals=[sell_signal, buy_signal],
        df=df,
        snapshot=SimpleNamespace(spread_points=45.0),
        regime={"name": "range", "trend_strength": 0.1, "volatility_pct": 0.02},
        higher_tf_bias="buy",
        drift={"active": False, "delta": 0.0},
        memory={},
        now_utc=pd.Timestamp("2026-04-02T21:00:00Z").to_pydatetime(),
    )

    assert "htf_opposed_spread_combo" in details["rsi_mean_reversion"]["reasons"]
    assert scores["momentum_continuation"] > scores["rsi_mean_reversion"]


def test_execution_cluster_guard_blocks_repeated_same_side_and_family():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_family = AtlasTrader._strategy_family.__get__(trader, AtlasTrader)
    trader._recent_execution_cluster = AtlasTrader._recent_execution_cluster.__get__(trader, AtlasTrader)
    trader._execution_cluster_guard = AtlasTrader._execution_cluster_guard.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        cluster_guard_enabled=True,
        cluster_guard_lookback=12,
        cluster_guard_max_same_side=8,
        cluster_guard_max_same_family=5,
    )
    recent_exec = []
    for _ in range(9):
        recent_exec.append(
            {
                "type": "execution",
                "meta": {
                    "chosen": {
                        "strategy": "evolved_rule(fast=13,slow=82,rsi_buy=47.7,rsi_sell=40.0)",
                        "side": "sell",
                    }
                },
            }
        )
    trader.store = SimpleNamespace(tail_events=lambda limit=0: recent_exec)

    meta = trader._execution_cluster_guard(
        TradeSignal("evolved_rule(fast=13,slow=82,rsi_buy=47.7,rsi_sell=40.0)", "sell", 0.4, 100, 160)
    )

    assert meta["applied"] is True
    assert meta["same_side_count"] == 9
    assert meta["same_family_count"] == 9


def test_execution_cluster_guard_allows_varied_recent_mix():
    trader = AtlasTrader.__new__(AtlasTrader)
    trader._strategy_key = AtlasTrader._strategy_key.__get__(trader, AtlasTrader)
    trader._strategy_family = AtlasTrader._strategy_family.__get__(trader, AtlasTrader)
    trader._recent_execution_cluster = AtlasTrader._recent_execution_cluster.__get__(trader, AtlasTrader)
    trader._execution_cluster_guard = AtlasTrader._execution_cluster_guard.__get__(trader, AtlasTrader)
    trader.settings = SimpleNamespace(
        cluster_guard_enabled=True,
        cluster_guard_lookback=12,
        cluster_guard_max_same_side=8,
        cluster_guard_max_same_family=5,
    )
    trader.store = SimpleNamespace(
        tail_events=lambda limit=0: [
            {"type": "execution", "meta": {"chosen": {"strategy": "ema_crossover", "side": "sell"}}},
            {"type": "execution", "meta": {"chosen": {"strategy": "momentum_continuation", "side": "buy"}}},
            {"type": "execution", "meta": {"chosen": {"strategy": "gold_liquidity_sweep", "side": "sell"}}},
            {"type": "execution", "meta": {"chosen": {"strategy": "evolved_rule(fast=13,slow=82,rsi_buy=47.7,rsi_sell=40.0)", "side": "sell"}}},
            {"type": "execution", "meta": {"chosen": {"strategy": "evolved_rule(fast=12,slow=84,rsi_buy=48.6,rsi_sell=40.0)", "side": "sell"}}},
            {"type": "execution", "meta": {"chosen": {"strategy": "rsi_mean_reversion", "side": "buy"}}},
        ]
    )

    meta = trader._execution_cluster_guard(TradeSignal("ema_crossover", "sell", 0.4, 100, 160))

    assert meta["applied"] is False
