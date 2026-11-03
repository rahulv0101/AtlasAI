from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Keep .env as a default source, but never let it override explicit
# per-process environment variables. This matters for isolated training
# runs that inject temporary STORAGE_DIR / budget settings.
load_dotenv(override=False)


_TIMEFRAMES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}


@dataclass(frozen=True)
class Settings:
    hfm_login: int
    hfm_password: str
    hfm_server: str
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    max_risk_per_trade: float = 0.005
    max_daily_drawdown: float = 0.03
    max_weekly_drawdown: float = 0.06
    max_open_trades: int = 2
    base_lot: float = 0.01
    magic_number: int = 20260330
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    gemini_influence_enabled: bool = False
    gemini_max_confidence_boost: float = 0.08
    gemini_max_confidence_cut: float = 0.35
    storage_dir: str = "storage"
    execute_trades: bool = False
    hfm_terminal_path: str = ""
    enforce_day_trading: bool = True
    enable_historical_training: bool = True
    historical_train_months: int = 24
    historical_train_max_bars: int = 30000
    historical_backtest_max_bars: int = 800
    historical_training_stage_timeout_seconds: float = 45.0
    backtest_initial_balance: float = 10000.0
    backtest_persist_learning_state: bool = False
    backtest_record_loss_analysis: bool = True
    backtest_loss_analysis_max_samples: int = 200
    historical_retrain_hours: float = 6.0
    event_max_lines: int = 20000
    m1_htf_alignment_timeframe: str = "M5"
    higher_timeframe_confirm: str = "H1"
    higher_timeframe_context: tuple[str, ...] = ("H1", "H4")
    min_confidence_trend: float = 0.11
    min_confidence_range: float = 0.09
    min_confidence_high_volatility: float = 0.13
    research_confidence_threshold: float = 0.50
    confluence_trade_threshold: float = 85.0
    research_regime_filter_enabled: bool = False
    research_regime_filter_low_liquidity: bool = True
    research_regime_filter_abnormal_spike: bool = True
    research_regime_low_liquidity_volume_ratio: float = 0.75
    research_regime_abnormal_spike_atr_multiple: float = 2.2
    research_regime_filter_weak_trend: bool = False
    research_regime_trend_min_strength_pct: float = 0.20
    research_regime_filter_trend_low_volatility: bool = False
    research_regime_trend_low_volatility_max_pct: float = 0.03
    research_regime_filter_trend_late_us: bool = False
    research_regime_filter_post_spike: bool = False
    research_regime_post_spike_atr_multiple: float = 1.6
    research_regime_filter_london_extreme_volatility: bool = False
    research_regime_london_low_volatility_max_pct: float = 0.03
    research_regime_london_high_volatility_min_pct: float = 0.07
    conservative_filter_enabled: bool = False
    conservative_filter_block_late_us_trend: bool = True
    conservative_filter_block_post_spike: bool = True
    conservative_filter_block_poor_liquidity: bool = True
    conservative_filter_poor_liquidity_volume_ratio: float = 0.80
    conservative_filter_post_spike_atr_multiple: float = 1.6
    macro_conflict_block_threshold: float = 30.0
    event_risk_block_threshold: float = 25.0
    setup_quality_enabled: bool = False
    setup_quality_filter_enabled: bool = False
    setup_quality_filter_min_score: float = 76.0
    setup_quality_penalty_cap: float = 0.18
    setup_quality_boost_cap: float = 0.10
    setup_quality_min_bucket_samples: int = 6
    setup_quality_strong_trend_min_pct: float = 0.20
    setup_quality_trend_low_volatility_max_pct: float = 0.03
    setup_quality_london_high_volatility_min_pct: float = 0.07
    setup_quality_post_spike_atr_multiple: float = 1.6
    max_trades_per_day: int = 20
    max_consecutive_losses: int = 4
    cooldown_minutes_after_loss_streak: int = 45
    max_spread_points: float = 80.0
    max_slippage_points: int = 20
    daily_profit_target_zar: float = 0.0
    require_ema_fresh_cross: bool = True
    adaptive_ema_side_switch: bool = True
    win_rate_focus_enabled: bool = True
    target_win_rate: float = 0.55
    win_rate_lookback: int = 30
    low_win_rate_confidence_bump: float = 0.08
    min_strategy_samples: int = 8
    min_strategy_win_rate: float = 0.35
    hard_prune_weak_strategies_enabled: bool = True
    hard_prune_min_samples: int = 120
    hard_prune_max_win_rate: float = 0.43
    hard_prune_require_negative_pnl: bool = True
    cluster_guard_enabled: bool = True
    cluster_guard_lookback: int = 12
    cluster_guard_max_same_side: int = 8
    cluster_guard_max_same_family: int = 5
    side_balance_enabled: bool = True
    side_balance_lookback: int = 50
    side_balance_max_skew: float = 0.25
    side_balance_confidence_penalty: float = 0.12
    smart_context_enabled: bool = True
    smart_context_lookback: int = 120
    smart_context_penalty_cap: float = 0.18
    smart_context_boost_cap: float = 0.08
    require_higher_tf_alignment: bool = True
    min_same_side_signals: int = 2
    timing_allow_mid_range_strong_momentum: bool = False
    severe_drift_hard_mode: bool = True
    fast_learning_mode: bool = True
    drift_throttle_seconds: int = 45
    drift_throttle_seconds_severe: int = 75
    open_feedback_enabled: bool = True
    open_feedback_pnl_threshold: float = 250.0
    open_feedback_reward: float = 0.01
    win_memory_enabled: bool = True
    win_memory_boost_cap: float = 0.08
    exploration_epsilon: float = 0.04
    meta_model_enabled: bool = True
    meta_model_confidence_cap: float = 0.08
    meta_model_candidate_scale: float = 0.0
    severe_weak_strategy_min_samples: int = 500
    severe_weak_strategy_max_win_rate: float = 0.28
    htf_spread_combo_penalty: float = 0.78
    m1_loss_block_enabled: bool = True
    m1_loss_block_min_samples: int = 6
    m1_loss_block_max_win_rate: float = 0.34
    m1_loss_block_min_streak: int = 3
    promotion_min_samples: int = 40
    promotion_min_accuracy_lift: float = 0.03
    promotion_min_brier_improvement: float = 0.01
    model_update_cooldown_hours: float = 24.0
    model_min_walkforward_folds: int = 3
    model_recent_window: int = 40
    model_max_drift_delta: float = 0.18
    backtest_spread_cost_bps: float = 3.0
    backtest_slippage_cost_bps: float = 1.5
    backtest_max_hold_bars: int = 12
    backtest_point_size: float = 0.01
    backtest_break_even_r_trigger: float = 0.9
    backtest_break_even_lock_r: float = 0.05
    backtest_trailing_r_trigger: float = 1.4
    backtest_trailing_buffer_r: float = 0.9
    backtest_stale_exit_bars: int = 4
    backtest_stale_min_progress_r: float = 0.15
    backtest_evolved_refresh_bars: int = 500
    backtest_evolved_lookback_bars: int = 1200
    evolved_strategy_generations: int = 12
    evolved_strategy_population: int = 18
    evolved_strategy_survivors: int = 4
    evolved_strategy_train_max_bars: int = 2000
    training_trend_robustness_weight: float = 0.0
    training_range_dependency_penalty: float = 0.0
    training_trend_loss_penalty_weight: float = 0.0
    training_profit_alignment_weight: float = 0.0
    training_profit_regression_penalty_weight: float = 0.0
    training_min_win_rate_alignment: float = 0.0
    training_pullback_quality_weight: float = 0.0
    training_wick_pressure_penalty_weight: float = 0.0
    training_trend_maturity_weight: float = 0.0
    training_target_candidate_win_rate: float = 0.0
    training_overtrade_penalty_weight: float = 0.0
    training_continuation_quality_weight: float = 0.0
    training_quality_validation_slices: int = 3
    training_quality_min_slice_win_rate: float = 0.48
    training_quality_min_slice_profit_component: float = -0.002
    training_quality_validation_weight: float = 0.0
    disabled_strategies: tuple[str, ...] = (
        "vwap_proxy_reversion",
        "rsi_trend_hybrid",
        "macd_trend",
        "candlestick_patterns",
    )
    enabled_strategies: tuple[str, ...] = (
        "ema_crossover",
        "pullback_trend",
        "momentum_continuation",
        "gold_micro_impulse",
        "gold_breakout_retest",
        "gold_liquidity_sweep",
    )
    backtest_include_evolved: bool = True
    m1_allowed_sessions: tuple[str, ...] = ("asia", "london", "new_york")
    signal_regime_gate_min_trend_strength_pct: float = 0.18
    signal_regime_gate_low_volatility_max_pct: float = 0.03
    signal_trend_strength_weak_pct: float = 0.12
    signal_trend_strength_strong_pct: float = 0.28
    signal_trend_strength_weak_confidence_multiplier: float = 0.55
    signal_trend_strength_strong_confidence_boost: float = 1.12
    suppress_directional_signals_outside_sessions: bool = True
    signal_outside_session_confidence_multiplier: float = 0.50
    momentum_signal_min_trend_strength_pct: float = 0.18
    momentum_signal_min_volatility_pct: float = 0.03
    m1_hard_block_spread_points: float = 30.0
    m1_hard_block_min_body_ratio: float = 0.28
    m1_hard_block_min_volume_ratio: float = 0.72
    evolved_candidate_min_trades: int = 200
    evolved_candidate_min_win_rate: float = 0.52
    evolved_candidate_min_profit_component: float = 0.0
    evolution_enabled: bool = False
    evolution_interval_hours: float = 6.0
    max_demo_strategies: int = 3
    evolved_max_risk_pct: float = 0.005
    live_promotion_requires_approval: bool = True
    evolution_max_shadow_drawdown_pct: float = 0.05
    evolution_shadow_drawdown_policy: str = "raw_peak"
    evolution_max_demo_drawdown_pct: float = 0.05
    evolution_min_shadow_trades: int = 100
    evolution_min_demo_trades: int = 100
    evolution_min_demo_win_rate: float = 0.50
    evolution_min_demo_profit_factor: float = 1.10
    evolution_human_approval_for_live: bool = True
    # FIX: demo/live calibration flags so demo can generate learning trades safely.
    demo_mode: bool = True
    learning_mode: bool = False
    force_demo_entry: bool = True
    # FIX: aggressive demo-learning profile to increase safe trade flow.
    confidence_threshold_learning: float = 0.44
    structure_score_min_learning: float = 0.24
    spread_max_pips_learning: float = 7.0
    min_signal_score_learning: float = 0.40
    min_signal_score_live: float = 0.60
    spread_max_pips_demo: float = 4.5
    spread_max_pips_live: float = 2.5
    structure_score_min_demo: float = 0.35
    structure_score_min_live: float = 0.50
    confidence_threshold_demo: float = 0.52
    confidence_threshold_live: float = 0.65
    learning_max_lot_size: float = 0.01
    learning_max_trades_per_day: int = 6
    learning_max_concurrent: int = 2
    learning_sl_atr_multiple: float = 2.0
    learning_tp_rr_ratio: float = 1.5
    learning_force_entry_per_session: bool = True
    learning_force_min_confidence: float = 0.40
    # FIX: adaptive probe quality controls for learning mode.
    learning_probe_min_edge_gap: float = 0.07
    learning_probe_confidence_bump_elevated: float = 0.04
    learning_probe_confidence_bump_chop: float = 0.06
    learning_probe_confidence_bump_transition: float = 0.03
    learning_graduation_min_trades: int = 100
    learning_graduation_min_winrate: float = 0.52
    learning_graduation_min_rr: float = 1.2
    scan_interval_seconds: int = 30
    backend_heartbeat_stale_after_seconds: int = 180
    max_concurrent_trades: int = 3
    duplicate_signal_window_seconds: int = 45
    scanner_cooldown_seconds: int = 180
    rejected_candidate_debounce_seconds: int = 5

    @property
    def timeframe_minutes(self) -> int:
        return _TIMEFRAMES.get(self.timeframe.upper(), 15)



def load_settings() -> Settings:
    raw_storage_dir = os.getenv("STORAGE_DIR", "storage").strip()
    storage_path = Path(raw_storage_dir)
    if not storage_path.is_absolute():
        # Resolve relative storage paths against the project root so all
        # launches (different CWDs/interpreters) share the same lock/state.
        project_root = Path(__file__).resolve().parents[2]
        storage_path = project_root / storage_path

    return Settings(
        hfm_login=int(os.getenv("HFM_LOGIN", "0")),
        hfm_password=os.getenv("HFM_PASSWORD", ""),
        hfm_server=os.getenv("HFM_SERVER", ""),
        symbol=os.getenv("SYMBOL", "XAUUSD"),
        timeframe=os.getenv("TIMEFRAME", "M15"),
        max_risk_per_trade=float(os.getenv("MAX_RISK_PER_TRADE", "0.005")),
        max_daily_drawdown=float(os.getenv("MAX_DAILY_DRAWDOWN", "0.03")),
        max_weekly_drawdown=float(os.getenv("MAX_WEEKLY_DRAWDOWN", "0.06")),
        max_open_trades=int(os.getenv("MAX_OPEN_TRADES", "2")),
        base_lot=float(os.getenv("BASE_LOT", "0.01")),
        magic_number=int(os.getenv("MAGIC_NUMBER", "20260330")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        gemini_influence_enabled=os.getenv("GEMINI_INFLUENCE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        gemini_max_confidence_boost=float(os.getenv("GEMINI_MAX_CONFIDENCE_BOOST", "0.08")),
        gemini_max_confidence_cut=float(os.getenv("GEMINI_MAX_CONFIDENCE_CUT", "0.35")),
        storage_dir=str(storage_path),
        execute_trades=os.getenv("EXECUTE_TRADES", "false").strip().lower() in {"1", "true", "yes", "on"},
        hfm_terminal_path=os.getenv("HFM_TERMINAL_PATH", "").strip(),
        enforce_day_trading=os.getenv("ENFORCE_DAY_TRADING", "true").strip().lower() in {"1", "true", "yes", "on"},
        enable_historical_training=os.getenv("ENABLE_HISTORICAL_TRAINING", "true").strip().lower() in {"1", "true", "yes", "on"},
        historical_train_months=int(os.getenv("HISTORICAL_TRAIN_MONTHS", "24")),
        historical_train_max_bars=int(os.getenv("HISTORICAL_TRAIN_MAX_BARS", "30000")),
        historical_backtest_max_bars=int(os.getenv("HISTORICAL_BACKTEST_MAX_BARS", "800")),
        historical_training_stage_timeout_seconds=float(
            os.getenv("HISTORICAL_TRAINING_STAGE_TIMEOUT_SECONDS", "45")
        ),
        backtest_initial_balance=float(os.getenv("BACKTEST_INITIAL_BALANCE", "10000")),
        backtest_persist_learning_state=os.getenv(
            "BACKTEST_PERSIST_LEARNING_STATE", "false"
        ).strip().lower() in {"1", "true", "yes", "on"},
        backtest_record_loss_analysis=os.getenv(
            "BACKTEST_RECORD_LOSS_ANALYSIS", "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        backtest_loss_analysis_max_samples=int(os.getenv("BACKTEST_LOSS_ANALYSIS_MAX_SAMPLES", "200")),
        historical_retrain_hours=float(os.getenv("HISTORICAL_RETRAIN_HOURS", "6")),
        event_max_lines=int(os.getenv("EVENT_MAX_LINES", "20000")),
        m1_htf_alignment_timeframe=os.getenv("M1_HTF_ALIGNMENT_TIMEFRAME", "M5").strip().upper(),
        higher_timeframe_confirm=os.getenv("HIGHER_TIMEFRAME_CONFIRM", "H1").strip().upper(),
        higher_timeframe_context=tuple(
            tf.strip().upper()
            for tf in os.getenv("HIGHER_TIMEFRAME_CONTEXT", "H1,H4").split(",")
            if tf.strip()
        )
        or ("H1", "H4"),
        min_confidence_trend=float(os.getenv("MIN_CONFIDENCE_TREND", "0.11")),
        min_confidence_range=float(os.getenv("MIN_CONFIDENCE_RANGE", "0.09")),
        min_confidence_high_volatility=float(os.getenv("MIN_CONFIDENCE_HIGH_VOLATILITY", "0.13")),
        research_confidence_threshold=float(os.getenv("RESEARCH_CONFIDENCE_THRESHOLD", "0.50")),
        confluence_trade_threshold=float(os.getenv("CONFLUENCE_TRADE_THRESHOLD", "85")),
        research_regime_filter_enabled=os.getenv("RESEARCH_REGIME_FILTER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_filter_low_liquidity=os.getenv("RESEARCH_REGIME_FILTER_LOW_LIQUIDITY", "true").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_filter_abnormal_spike=os.getenv("RESEARCH_REGIME_FILTER_ABNORMAL_SPIKE", "true").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_low_liquidity_volume_ratio=float(os.getenv("RESEARCH_REGIME_LOW_LIQUIDITY_VOLUME_RATIO", "0.75")),
        research_regime_abnormal_spike_atr_multiple=float(os.getenv("RESEARCH_REGIME_ABNORMAL_SPIKE_ATR_MULTIPLE", "2.2")),
        research_regime_filter_weak_trend=os.getenv("RESEARCH_REGIME_FILTER_WEAK_TREND", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_trend_min_strength_pct=float(os.getenv("RESEARCH_REGIME_TREND_MIN_STRENGTH_PCT", "0.20")),
        research_regime_filter_trend_low_volatility=os.getenv("RESEARCH_REGIME_FILTER_TREND_LOW_VOLATILITY", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_trend_low_volatility_max_pct=float(os.getenv("RESEARCH_REGIME_TREND_LOW_VOLATILITY_MAX_PCT", "0.03")),
        research_regime_filter_trend_late_us=os.getenv("RESEARCH_REGIME_FILTER_TREND_LATE_US", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_filter_post_spike=os.getenv("RESEARCH_REGIME_FILTER_POST_SPIKE", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_post_spike_atr_multiple=float(os.getenv("RESEARCH_REGIME_POST_SPIKE_ATR_MULTIPLE", "1.6")),
        research_regime_filter_london_extreme_volatility=os.getenv("RESEARCH_REGIME_FILTER_LONDON_EXTREME_VOLATILITY", "false").strip().lower() in {"1", "true", "yes", "on"},
        research_regime_london_low_volatility_max_pct=float(os.getenv("RESEARCH_REGIME_LONDON_LOW_VOLATILITY_MAX_PCT", "0.03")),
        research_regime_london_high_volatility_min_pct=float(os.getenv("RESEARCH_REGIME_LONDON_HIGH_VOLATILITY_MIN_PCT", "0.07")),
        conservative_filter_enabled=os.getenv("CONSERVATIVE_FILTER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        conservative_filter_block_late_us_trend=os.getenv("CONSERVATIVE_FILTER_BLOCK_LATE_US_TREND", "true").strip().lower() in {"1", "true", "yes", "on"},
        conservative_filter_block_post_spike=os.getenv("CONSERVATIVE_FILTER_BLOCK_POST_SPIKE", "true").strip().lower() in {"1", "true", "yes", "on"},
        conservative_filter_block_poor_liquidity=os.getenv("CONSERVATIVE_FILTER_BLOCK_POOR_LIQUIDITY", "true").strip().lower() in {"1", "true", "yes", "on"},
        conservative_filter_poor_liquidity_volume_ratio=float(os.getenv("CONSERVATIVE_FILTER_POOR_LIQUIDITY_VOLUME_RATIO", "0.80")),
        conservative_filter_post_spike_atr_multiple=float(os.getenv("CONSERVATIVE_FILTER_POST_SPIKE_ATR_MULTIPLE", "1.6")),
        macro_conflict_block_threshold=float(os.getenv("MACRO_CONFLICT_BLOCK_THRESHOLD", "30")),
        event_risk_block_threshold=float(os.getenv("EVENT_RISK_BLOCK_THRESHOLD", "25")),
        setup_quality_enabled=os.getenv("SETUP_QUALITY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        setup_quality_filter_enabled=os.getenv("SETUP_QUALITY_FILTER_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        setup_quality_filter_min_score=float(os.getenv("SETUP_QUALITY_FILTER_MIN_SCORE", "76.0")),
        setup_quality_penalty_cap=float(os.getenv("SETUP_QUALITY_PENALTY_CAP", "0.18")),
        setup_quality_boost_cap=float(os.getenv("SETUP_QUALITY_BOOST_CAP", "0.10")),
        setup_quality_min_bucket_samples=int(os.getenv("SETUP_QUALITY_MIN_BUCKET_SAMPLES", "6")),
        setup_quality_strong_trend_min_pct=float(os.getenv("SETUP_QUALITY_STRONG_TREND_MIN_PCT", "0.20")),
        setup_quality_trend_low_volatility_max_pct=float(os.getenv("SETUP_QUALITY_TREND_LOW_VOLATILITY_MAX_PCT", "0.03")),
        setup_quality_london_high_volatility_min_pct=float(os.getenv("SETUP_QUALITY_LONDON_HIGH_VOLATILITY_MIN_PCT", "0.07")),
        setup_quality_post_spike_atr_multiple=float(os.getenv("SETUP_QUALITY_POST_SPIKE_ATR_MULTIPLE", "1.6")),
        max_trades_per_day=int(os.getenv("MAX_TRADES_PER_DAY", "20")),
        max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4")),
        cooldown_minutes_after_loss_streak=int(os.getenv("COOLDOWN_MINUTES_AFTER_LOSS_STREAK", "45")),
        max_spread_points=float(os.getenv("MAX_SPREAD_POINTS", "80")),
        max_slippage_points=int(os.getenv("MAX_SLIPPAGE_POINTS", "20")),
        daily_profit_target_zar=float(os.getenv("DAILY_PROFIT_TARGET_ZAR", "0")),
        require_ema_fresh_cross=os.getenv("REQUIRE_EMA_FRESH_CROSS", "true").strip().lower() in {"1", "true", "yes", "on"},
        adaptive_ema_side_switch=os.getenv("ADAPTIVE_EMA_SIDE_SWITCH", "true").strip().lower() in {"1", "true", "yes", "on"},
        win_rate_focus_enabled=os.getenv("WIN_RATE_FOCUS_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        target_win_rate=float(os.getenv("TARGET_WIN_RATE", "0.55")),
        win_rate_lookback=int(os.getenv("WIN_RATE_LOOKBACK", "30")),
        low_win_rate_confidence_bump=float(os.getenv("LOW_WIN_RATE_CONFIDENCE_BUMP", "0.08")),
        min_strategy_samples=int(os.getenv("MIN_STRATEGY_SAMPLES", "8")),
        min_strategy_win_rate=float(os.getenv("MIN_STRATEGY_WIN_RATE", "0.35")),
        hard_prune_weak_strategies_enabled=os.getenv("HARD_PRUNE_WEAK_STRATEGIES_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        hard_prune_min_samples=int(os.getenv("HARD_PRUNE_MIN_SAMPLES", "120")),
        hard_prune_max_win_rate=float(os.getenv("HARD_PRUNE_MAX_WIN_RATE", "0.43")),
        hard_prune_require_negative_pnl=os.getenv("HARD_PRUNE_REQUIRE_NEGATIVE_PNL", "true").strip().lower() in {"1", "true", "yes", "on"},
        cluster_guard_enabled=os.getenv("CLUSTER_GUARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        cluster_guard_lookback=int(os.getenv("CLUSTER_GUARD_LOOKBACK", "12")),
        cluster_guard_max_same_side=int(os.getenv("CLUSTER_GUARD_MAX_SAME_SIDE", "8")),
        cluster_guard_max_same_family=int(os.getenv("CLUSTER_GUARD_MAX_SAME_FAMILY", "5")),
        side_balance_enabled=os.getenv("SIDE_BALANCE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        side_balance_lookback=int(os.getenv("SIDE_BALANCE_LOOKBACK", "50")),
        side_balance_max_skew=float(os.getenv("SIDE_BALANCE_MAX_SKEW", "0.25")),
        side_balance_confidence_penalty=float(os.getenv("SIDE_BALANCE_CONFIDENCE_PENALTY", "0.12")),
        smart_context_enabled=os.getenv("SMART_CONTEXT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        smart_context_lookback=int(os.getenv("SMART_CONTEXT_LOOKBACK", "120")),
        smart_context_penalty_cap=float(os.getenv("SMART_CONTEXT_PENALTY_CAP", "0.18")),
        smart_context_boost_cap=float(os.getenv("SMART_CONTEXT_BOOST_CAP", "0.08")),
        require_higher_tf_alignment=os.getenv("REQUIRE_HIGHER_TF_ALIGNMENT", "true").strip().lower() in {"1", "true", "yes", "on"},
        min_same_side_signals=int(os.getenv("MIN_SAME_SIDE_SIGNALS", "2")),
        timing_allow_mid_range_strong_momentum=os.getenv(
            "TIMING_ALLOW_MID_RANGE_STRONG_MOMENTUM", "false"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        severe_drift_hard_mode=os.getenv("SEVERE_DRIFT_HARD_MODE", "true").strip().lower() in {"1", "true", "yes", "on"},
        fast_learning_mode=os.getenv("FAST_LEARNING_MODE", "true").strip().lower() in {"1", "true", "yes", "on"},
        drift_throttle_seconds=int(os.getenv("DRIFT_THROTTLE_SECONDS", "45")),
        drift_throttle_seconds_severe=int(os.getenv("DRIFT_THROTTLE_SECONDS_SEVERE", "75")),
        open_feedback_enabled=os.getenv("OPEN_FEEDBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        open_feedback_pnl_threshold=float(os.getenv("OPEN_FEEDBACK_PNL_THRESHOLD", "250")),
        open_feedback_reward=float(os.getenv("OPEN_FEEDBACK_REWARD", "0.01")),
        win_memory_enabled=os.getenv("WIN_MEMORY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        win_memory_boost_cap=float(os.getenv("WIN_MEMORY_BOOST_CAP", "0.08")),
        exploration_epsilon=float(os.getenv("EXPLORATION_EPSILON", "0.04")),
        meta_model_enabled=os.getenv("META_MODEL_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        meta_model_confidence_cap=float(os.getenv("META_MODEL_CONFIDENCE_CAP", "0.08")),
        meta_model_candidate_scale=float(os.getenv("META_MODEL_CANDIDATE_SCALE", "0.0")),
        severe_weak_strategy_min_samples=int(os.getenv("SEVERE_WEAK_STRATEGY_MIN_SAMPLES", "500")),
        severe_weak_strategy_max_win_rate=float(os.getenv("SEVERE_WEAK_STRATEGY_MAX_WIN_RATE", "0.28")),
        htf_spread_combo_penalty=float(os.getenv("HTF_SPREAD_COMBO_PENALTY", "0.78")),
        m1_loss_block_enabled=os.getenv("M1_LOSS_BLOCK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"},
        m1_loss_block_min_samples=int(os.getenv("M1_LOSS_BLOCK_MIN_SAMPLES", "6")),
        m1_loss_block_max_win_rate=float(os.getenv("M1_LOSS_BLOCK_MAX_WIN_RATE", "0.34")),
        m1_loss_block_min_streak=int(os.getenv("M1_LOSS_BLOCK_MIN_STREAK", "3")),
        promotion_min_samples=int(os.getenv("PROMOTION_MIN_SAMPLES", "40")),
        promotion_min_accuracy_lift=float(os.getenv("PROMOTION_MIN_ACCURACY_LIFT", "0.03")),
        promotion_min_brier_improvement=float(os.getenv("PROMOTION_MIN_BRIER_IMPROVEMENT", "0.01")),
        model_update_cooldown_hours=float(os.getenv("MODEL_UPDATE_COOLDOWN_HOURS", "24")),
        model_min_walkforward_folds=int(os.getenv("MODEL_MIN_WALKFORWARD_FOLDS", "3")),
        model_recent_window=int(os.getenv("MODEL_RECENT_WINDOW", "40")),
        model_max_drift_delta=float(os.getenv("MODEL_MAX_DRIFT_DELTA", "0.18")),
        backtest_spread_cost_bps=float(os.getenv("BACKTEST_SPREAD_COST_BPS", "3.0")),
        backtest_slippage_cost_bps=float(os.getenv("BACKTEST_SLIPPAGE_COST_BPS", "1.5")),
        backtest_max_hold_bars=int(os.getenv("BACKTEST_MAX_HOLD_BARS", "12")),
        backtest_point_size=float(os.getenv("BACKTEST_POINT_SIZE", "0.01")),
        backtest_break_even_r_trigger=float(os.getenv("BACKTEST_BREAK_EVEN_R_TRIGGER", "0.9")),
        backtest_break_even_lock_r=float(os.getenv("BACKTEST_BREAK_EVEN_LOCK_R", "0.05")),
        backtest_trailing_r_trigger=float(os.getenv("BACKTEST_TRAILING_R_TRIGGER", "1.4")),
        backtest_trailing_buffer_r=float(os.getenv("BACKTEST_TRAILING_BUFFER_R", "0.9")),
        backtest_stale_exit_bars=int(os.getenv("BACKTEST_STALE_EXIT_BARS", "4")),
        backtest_stale_min_progress_r=float(os.getenv("BACKTEST_STALE_MIN_PROGRESS_R", "0.15")),
        backtest_evolved_refresh_bars=int(os.getenv("BACKTEST_EVOLVED_REFRESH_BARS", "500")),
        backtest_evolved_lookback_bars=int(os.getenv("BACKTEST_EVOLVED_LOOKBACK_BARS", "1200")),
        evolved_strategy_generations=int(os.getenv("EVOLVED_STRATEGY_GENERATIONS", "12")),
        evolved_strategy_population=int(os.getenv("EVOLVED_STRATEGY_POPULATION", "18")),
        evolved_strategy_survivors=int(os.getenv("EVOLVED_STRATEGY_SURVIVORS", "4")),
        evolved_strategy_train_max_bars=int(os.getenv("EVOLVED_STRATEGY_TRAIN_MAX_BARS", "2000")),
        training_trend_robustness_weight=float(os.getenv("TRAINING_TREND_ROBUSTNESS_WEIGHT", "0.0")),
        training_range_dependency_penalty=float(os.getenv("TRAINING_RANGE_DEPENDENCY_PENALTY", "0.0")),
        training_trend_loss_penalty_weight=float(os.getenv("TRAINING_TREND_LOSS_PENALTY_WEIGHT", "0.0")),
        training_profit_alignment_weight=float(os.getenv("TRAINING_PROFIT_ALIGNMENT_WEIGHT", "0.0")),
        training_profit_regression_penalty_weight=float(os.getenv("TRAINING_PROFIT_REGRESSION_PENALTY_WEIGHT", "0.0")),
        training_min_win_rate_alignment=float(os.getenv("TRAINING_MIN_WIN_RATE_ALIGNMENT", "0.0")),
        training_pullback_quality_weight=float(os.getenv("TRAINING_PULLBACK_QUALITY_WEIGHT", "0.0")),
        training_wick_pressure_penalty_weight=float(os.getenv("TRAINING_WICK_PRESSURE_PENALTY_WEIGHT", "0.0")),
        training_trend_maturity_weight=float(os.getenv("TRAINING_TREND_MATURITY_WEIGHT", "0.0")),
        training_target_candidate_win_rate=float(os.getenv("TRAINING_TARGET_CANDIDATE_WIN_RATE", "0.0")),
        training_overtrade_penalty_weight=float(os.getenv("TRAINING_OVERTRADE_PENALTY_WEIGHT", "0.0")),
        training_continuation_quality_weight=float(os.getenv("TRAINING_CONTINUATION_QUALITY_WEIGHT", "0.0")),
        disabled_strategies=tuple(
            s.strip()
            for s in os.getenv(
                "DISABLED_STRATEGIES",
                "vwap_proxy_reversion,rsi_trend_hybrid,macd_trend,candlestick_patterns",
            ).split(",")
            if s.strip()
        ),
        enabled_strategies=tuple(
            s.strip()
            for s in os.getenv(
                "ENABLED_STRATEGIES",
                "ema_crossover,pullback_trend,momentum_continuation,gold_micro_impulse,gold_breakout_retest,gold_liquidity_sweep",
            ).split(",")
            if s.strip()
        ),
        backtest_include_evolved=os.getenv("BACKTEST_INCLUDE_EVOLVED", "true").strip().lower() in {"1", "true", "yes", "on"},
        m1_allowed_sessions=tuple(
            s.strip().lower()
            for s in os.getenv("M1_ALLOWED_SESSIONS", "asia,london,new_york").split(",")
            if s.strip()
        ),
        signal_regime_gate_min_trend_strength_pct=float(os.getenv("SIGNAL_REGIME_GATE_MIN_TREND_STRENGTH_PCT", "0.18")),
        signal_regime_gate_low_volatility_max_pct=float(os.getenv("SIGNAL_REGIME_GATE_LOW_VOLATILITY_MAX_PCT", "0.03")),
        signal_trend_strength_weak_pct=float(os.getenv("SIGNAL_TREND_STRENGTH_WEAK_PCT", "0.12")),
        signal_trend_strength_strong_pct=float(os.getenv("SIGNAL_TREND_STRENGTH_STRONG_PCT", "0.28")),
        signal_trend_strength_weak_confidence_multiplier=float(
            os.getenv("SIGNAL_TREND_STRENGTH_WEAK_CONFIDENCE_MULTIPLIER", "0.55")
        ),
        signal_trend_strength_strong_confidence_boost=float(
            os.getenv("SIGNAL_TREND_STRENGTH_STRONG_CONFIDENCE_BOOST", "1.12")
        ),
        suppress_directional_signals_outside_sessions=os.getenv(
            "SUPPRESS_DIRECTIONAL_SIGNALS_OUTSIDE_SESSIONS", "true"
        ).strip().lower() in {"1", "true", "yes", "on"},
        signal_outside_session_confidence_multiplier=float(
            os.getenv("SIGNAL_OUTSIDE_SESSION_CONFIDENCE_MULTIPLIER", "0.50")
        ),
        momentum_signal_min_trend_strength_pct=float(os.getenv("MOMENTUM_SIGNAL_MIN_TREND_STRENGTH_PCT", "0.18")),
        momentum_signal_min_volatility_pct=float(os.getenv("MOMENTUM_SIGNAL_MIN_VOLATILITY_PCT", "0.03")),
        m1_hard_block_spread_points=float(os.getenv("M1_HARD_BLOCK_SPREAD_POINTS", "30")),
        m1_hard_block_min_body_ratio=float(os.getenv("M1_HARD_BLOCK_MIN_BODY_RATIO", "0.28")),
        m1_hard_block_min_volume_ratio=float(os.getenv("M1_HARD_BLOCK_MIN_VOLUME_RATIO", "0.72")),
        evolved_candidate_min_trades=int(os.getenv("EVOLVED_CANDIDATE_MIN_TRADES", "200")),
        evolved_candidate_min_win_rate=float(os.getenv("EVOLVED_CANDIDATE_MIN_WIN_RATE", "0.52")),
        evolved_candidate_min_profit_component=float(os.getenv("EVOLVED_CANDIDATE_MIN_PROFIT_COMPONENT", "0.0")),
        evolution_enabled=os.getenv("EVOLUTION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        evolution_interval_hours=float(os.getenv("EVOLUTION_INTERVAL_HOURS", "6.0")),
        max_demo_strategies=int(os.getenv("MAX_DEMO_STRATEGIES", "3")),
        evolved_max_risk_pct=float(os.getenv("EVOLVED_MAX_RISK_PCT", "0.005")),
        live_promotion_requires_approval=os.getenv("LIVE_PROMOTION_REQUIRES_APPROVAL", "true").strip().lower() in {"1", "true", "yes", "on"},
        evolution_max_shadow_drawdown_pct=float(os.getenv("EVOLUTION_MAX_SHADOW_DRAWDOWN_PCT", "0.05")),
        evolution_shadow_drawdown_policy=os.getenv("EVOLUTION_SHADOW_DRAWDOWN_POLICY", "raw_peak").strip().lower(),
        evolution_max_demo_drawdown_pct=float(os.getenv("EVOLUTION_MAX_DEMO_DRAWDOWN_PCT", "0.05")),
        evolution_min_shadow_trades=int(os.getenv("EVOLUTION_MIN_SHADOW_TRADES", "100")),
        evolution_min_demo_trades=int(os.getenv("EVOLUTION_MIN_DEMO_TRADES", "100")),
        evolution_min_demo_win_rate=float(os.getenv("EVOLUTION_MIN_DEMO_WIN_RATE", "0.50")),
        evolution_min_demo_profit_factor=float(os.getenv("EVOLUTION_MIN_DEMO_PROFIT_FACTOR", "1.10")),
        evolution_human_approval_for_live=os.getenv("EVOLUTION_HUMAN_APPROVAL_FOR_LIVE", "true").strip().lower() in {"1", "true", "yes", "on"},
        # FIX: demo/live calibration flags so demo can generate learning trades safely.
        demo_mode=os.getenv("DEMO_MODE", "true").strip().lower() in {"1", "true", "yes", "on"},
        learning_mode=os.getenv("LEARNING_MODE", "true").strip().lower() in {"1", "true", "yes", "on"},
        force_demo_entry=os.getenv("FORCE_DEMO_ENTRY", "true").strip().lower() in {"1", "true", "yes", "on"},
        confidence_threshold_learning=float(os.getenv("CONFIDENCE_THRESHOLD_LEARNING", "0.44")),
        structure_score_min_learning=float(os.getenv("STRUCTURE_SCORE_MIN_LEARNING", "0.24")),
        spread_max_pips_learning=float(os.getenv("SPREAD_MAX_PIPS_LEARNING", "7.0")),
        min_signal_score_learning=float(os.getenv("MIN_SIGNAL_SCORE_LEARNING", "0.40")),
        min_signal_score_live=float(os.getenv("MIN_SIGNAL_SCORE_LIVE", "0.60")),
        spread_max_pips_demo=float(os.getenv("SPREAD_MAX_PIPS_DEMO", "4.5")),
        spread_max_pips_live=float(os.getenv("SPREAD_MAX_PIPS_LIVE", "2.5")),
        structure_score_min_demo=float(os.getenv("STRUCTURE_SCORE_MIN_DEMO", "0.35")),
        structure_score_min_live=float(os.getenv("STRUCTURE_SCORE_MIN_LIVE", "0.50")),
        confidence_threshold_demo=float(os.getenv("CONFIDENCE_THRESHOLD_DEMO", "0.52")),
        confidence_threshold_live=float(os.getenv("CONFIDENCE_THRESHOLD_LIVE", "0.65")),
        learning_max_lot_size=float(os.getenv("LEARNING_MAX_LOT_SIZE", "0.01")),
        learning_max_trades_per_day=int(os.getenv("LEARNING_MAX_TRADES_PER_DAY", "6")),
        learning_max_concurrent=int(os.getenv("LEARNING_MAX_CONCURRENT", "2")),
        learning_sl_atr_multiple=float(os.getenv("LEARNING_SL_ATR_MULTIPLE", "2.0")),
        learning_tp_rr_ratio=float(os.getenv("LEARNING_TP_RR_RATIO", "1.5")),
        learning_force_entry_per_session=os.getenv("LEARNING_FORCE_ENTRY_PER_SESSION", "true").strip().lower() in {"1", "true", "yes", "on"},
        learning_force_min_confidence=float(os.getenv("LEARNING_FORCE_MIN_CONFIDENCE", "0.40")),
        learning_probe_min_edge_gap=float(os.getenv("LEARNING_PROBE_MIN_EDGE_GAP", "0.07")),
        learning_probe_confidence_bump_elevated=float(os.getenv("LEARNING_PROBE_CONFIDENCE_BUMP_ELEVATED", "0.04")),
        learning_probe_confidence_bump_chop=float(os.getenv("LEARNING_PROBE_CONFIDENCE_BUMP_CHOP", "0.06")),
        learning_probe_confidence_bump_transition=float(os.getenv("LEARNING_PROBE_CONFIDENCE_BUMP_TRANSITION", "0.03")),
        learning_graduation_min_trades=int(os.getenv("LEARNING_GRADUATION_MIN_TRADES", "100")),
        learning_graduation_min_winrate=float(os.getenv("LEARNING_GRADUATION_MIN_WINRATE", "0.52")),
        learning_graduation_min_rr=float(os.getenv("LEARNING_GRADUATION_MIN_RR", "1.2")),
        scan_interval_seconds=int(os.getenv("SCAN_INTERVAL_SECONDS", "30")),
        backend_heartbeat_stale_after_seconds=int(
            os.getenv(
                "BACKEND_HEARTBEAT_STALE_AFTER_SECONDS",
                str(max(180, int(os.getenv("SCAN_INTERVAL_SECONDS", "30")) * 3)),
            )
        ),
        max_concurrent_trades=int(os.getenv("MAX_CONCURRENT_TRADES", "3")),
        duplicate_signal_window_seconds=int(os.getenv("DUPLICATE_SIGNAL_WINDOW_SECONDS", "45")),
        scanner_cooldown_seconds=int(os.getenv("SCANNER_COOLDOWN_SECONDS", "180")),
        rejected_candidate_debounce_seconds=int(os.getenv("REJECTED_CANDIDATE_DEBOUNCE_SECONDS", "5")),
    )


def get_active_thresholds(settings: Settings) -> dict[str, float | int | str]:
    """Return active threshold profile by current mode."""
    if bool(getattr(settings, "learning_mode", False)):
        return {
            "confidence": float(getattr(settings, "confidence_threshold_learning", 0.44)),
            "structure": float(getattr(settings, "structure_score_min_learning", 0.24)),
            "spread_max": float(getattr(settings, "spread_max_pips_learning", 7.0)),
            "min_score": float(getattr(settings, "min_signal_score_learning", 0.40)),
            "max_lot": float(getattr(settings, "learning_max_lot_size", 0.01)),
            "max_trades_day": int(getattr(settings, "learning_max_trades_per_day", 6)),
            "max_concurrent": int(getattr(settings, "learning_max_concurrent", 2)),
            "sl_atr_mult": float(getattr(settings, "learning_sl_atr_multiple", 2.0)),
            "tp_rr": float(getattr(settings, "learning_tp_rr_ratio", 1.5)),
            "mode_label": "LEARNING_AGGRESSIVE_DEMO",
        }
    if bool(getattr(settings, "demo_mode", False)):
        return {
            "confidence": float(getattr(settings, "confidence_threshold_demo", 0.52)),
            "structure": float(getattr(settings, "structure_score_min_demo", 0.35)),
            "spread_max": float(getattr(settings, "spread_max_pips_demo", 4.5)),
            "min_score": 0.52,
            "max_lot": 0.05,
            "max_trades_day": int(getattr(settings, "max_trades_per_day", 6)),
            "max_concurrent": int(getattr(settings, "max_concurrent_trades", 3)),
            "sl_atr_mult": 1.5,
            "tp_rr": 2.0,
            "mode_label": "DEMO",
        }
    return {
        "confidence": float(getattr(settings, "confidence_threshold_live", getattr(settings, "confidence_threshold_demo", 0.52))),
        "structure": float(getattr(settings, "structure_score_min_live", getattr(settings, "structure_score_min_demo", 0.35))),
        "spread_max": float(getattr(settings, "spread_max_pips_live", getattr(settings, "spread_max_pips_demo", 4.5))),
        "min_score": float(getattr(settings, "min_signal_score_live", 0.60)),
        "max_lot": 2.0,
        "max_trades_day": 6,
        "max_concurrent": 3,
        "sl_atr_mult": 1.0,
        "tp_rr": 2.0,
        "mode_label": "LIVE",
    }
