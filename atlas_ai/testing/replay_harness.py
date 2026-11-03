from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pandas as pd

from atlas_ai.execution.trade_executor import ExecutionResult
from atlas_ai.models import TradeSignal
from atlas_ai.risk.risk_manager import RiskSnapshot
from atlas_ai.trader import AtlasTrader
import atlas_ai.trader as trader_module


@dataclass(frozen=True)
class ReplayScenario:
    name: str
    runtime_flags: dict[str, Any]
    raw_signals: list[TradeSignal]
    active_signals: list[TradeSignal]
    chosen_signal: TradeSignal
    risk_snapshot: RiskSnapshot
    regime: dict[str, Any]
    higher_tf_bias: str
    contextual_scores: dict[str, float]
    contextual_details: dict[str, Any]
    best_actionable_candidate: dict[str, Any]
    overlay_meta: dict[str, Any]
    support_count: int
    min_confidence: float
    risk_allowed: bool = True
    risk_reason: str = ""
    guard_block_reason: str = ""
    base_volume: float = 0.32
    execution_result: ExecutionResult | None = None


class ReplayDiffError(AssertionError):
    """Raised when replay output diverges from a stored baseline."""


class _ReplayStore:
    def __init__(self, runtime_flags: dict[str, Any]):
        self._runtime_flags = dict(runtime_flags)
        self.saved_status: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def load_runtime_flags(self) -> dict[str, Any]:
        return dict(self._runtime_flags)

    def save_runtime_flags(self, flags: dict[str, Any]) -> None:
        self._runtime_flags = dict(flags)

    def log_event(self, event_type: str, message: str, meta: dict[str, Any] | None = None) -> None:
        self.events.append(
            {
                "type": str(event_type),
                "message": str(message),
                "meta": dict(meta or {}),
            }
        )

    def save_status(self, payload: dict[str, Any]) -> None:
        self.saved_status = dict(payload)

    def load_status(self) -> dict[str, Any]:
        return dict(self.saved_status)

    def load_events(self, limit: int = 0, event_type: str | None = None) -> list[dict[str, Any]]:
        rows = list(self.events)
        if event_type:
            rows = [row for row in rows if str(row.get("type", "")).lower() == str(event_type).lower()]
        if limit > 0:
            rows = rows[-limit:]
        return rows

    def tail_events(self, limit: int = 0) -> list[dict[str, Any]]:
        rows = list(self.events)
        if limit > 0:
            rows = rows[-limit:]
        return rows

    def load_learning_state(self) -> dict[str, Any]:
        return {"state": {}}

    @property
    def base(self) -> str:
        return ""


def _fixture_df() -> pd.DataFrame:
    closes = [3320.0 + (i * 0.4) for i in range(60)]
    times = pd.date_range("2026-04-04T11:00:00+00:00", periods=60, freq="min")
    rows = []
    for idx, (ts, close) in enumerate(zip(times, closes, strict=True)):
        rows.append(
            {
                "time": ts,
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "tick_volume": 100 + idx,
            }
        )
    return pd.DataFrame(rows)


def _top_rows(signals: list[TradeSignal]) -> list[dict[str, Any]]:
    return [
        {
            "strategy": signal.strategy,
            "side": signal.side,
            "confidence": round(float(signal.confidence), 4),
        }
        for signal in signals
    ]


def _base_settings(*, execute_trades: bool) -> SimpleNamespace:
    return SimpleNamespace(
        symbol="XAUUSD",
        timeframe="M1",
        timeframe_minutes=1,
        higher_timeframe_confirm="M15",
        execute_trades=bool(execute_trades),
        gemini_influence_enabled=False,
        meta_model_enabled=False,
        research_regime_filter_enabled=False,
        conservative_filter_enabled=False,
        enforce_day_trading=False,
        max_trades_per_day=8,
        max_consecutive_losses=3,
        max_daily_drawdown=0.03,
        cooldown_minutes_after_loss_streak=30,
        daily_profit_target_zar=0.0,
        fast_learning_mode=False,
        gemini_max_confidence_boost=0.0,
        gemini_max_confidence_cut=0.0,
        min_same_side_signals=2,
        severe_drift_hard_mode=False,
        adaptive_ema_side_switch=False,
        require_ema_fresh_cross=False,
        require_higher_tf_alignment=False,
        max_risk_per_trade=0.005,
        base_lot=0.01,
        max_spread_points=80.0,
        min_confidence_trend=0.55,
        min_confidence_range=0.60,
        min_confidence_high_volatility=0.65,
        magic_number=123456,
        storage_dir="",
        drift_throttle_seconds=30,
        drift_throttle_seconds_severe=60,
        win_rate_focus_enabled=False,
        min_strategy_samples=30,
        min_strategy_win_rate=0.50,
        win_rate_lookback=20,
        low_win_rate_confidence_bump=0.05,
        target_win_rate=0.55,
        side_balance_enabled=False,
        smart_context_enabled=False,
        m1_loss_block_enabled=True,
        m1_loss_block_max_win_rate=0.34,
        m1_loss_block_min_streak=3,
        m1_loss_block_min_samples=6,
        m1_hard_block_spread_points=30.0,
        m1_hard_block_min_body_ratio=0.28,
        m1_hard_block_min_volume_ratio=0.72,
        signal_regime_gate_low_volatility_max_pct=0.03,
        signal_regime_gate_min_trend_strength_pct=0.18,
        momentum_signal_min_trend_strength_pct=0.18,
        momentum_signal_min_volatility_pct=0.03,
        signal_trend_strength_weak_pct=0.12,
        signal_trend_strength_strong_pct=0.28,
        signal_trend_strength_weak_confidence_multiplier=0.55,
        signal_trend_strength_strong_confidence_boost=1.12,
        suppress_directional_signals_outside_sessions=True,
        signal_outside_session_confidence_multiplier=0.5,
        m1_allowed_sessions=("london", "new_york"),
        htf_spread_combo_penalty=0.78,
    )


def default_replay_scenarios() -> list[ReplayScenario]:
    raw_signals = [
        TradeSignal("momentum_continuation", "buy", 0.36, 115.0, 175.0),
        TradeSignal("gold_micro_impulse", "buy", 0.22, 95.0, 150.0),
    ]
    active_signals = deepcopy(raw_signals)
    chosen = TradeSignal("momentum_continuation", "buy", 0.36, 115.0, 175.0)
    risk_snapshot = RiskSnapshot(
        equity=10000.0,
        balance=10000.0,
        daily_loss_pct=0.0,
        weekly_loss_pct=0.0,
        open_positions=0,
        spread_points=18.0,
    )
    regime = {"name": "trend", "trend_strength": 0.24, "volatility_pct": 0.028}
    contextual_details = {
        "momentum_continuation": {
            "context": {
                "strategy": "momentum_continuation",
                "side": "buy",
                "session": "london",
                "hour_utc": 11,
                "regime": "trend",
            },
            "score": 0.22,
            "confidence": 0.36,
        }
    }
    best_actionable = {
        "strategy": "momentum_continuation",
        "score": 0.22,
        "confidence": 0.36,
        "setup_quality_score": 0.14,
        "reasons": ["htf_alignment", "strong_trend_structure"],
    }
    overlay_meta = {
        "session": "london",
        "regime_name": "trend",
        "volume_ratio": 1.05,
        "spike_ratio": 1.1,
    }
    return [
        ReplayScenario(
            name="paper_allow",
            runtime_flags={"execute_trades": False, "gemini_influence_enabled": False},
            raw_signals=raw_signals,
            active_signals=active_signals,
            chosen_signal=chosen,
            risk_snapshot=risk_snapshot,
            regime=regime,
            higher_tf_bias="buy",
            contextual_scores={"momentum_continuation": 0.22, "gold_micro_impulse": 0.11},
            contextual_details=contextual_details,
            best_actionable_candidate=best_actionable,
            overlay_meta=overlay_meta,
            support_count=2,
            min_confidence=0.25,
        ),
        ReplayScenario(
            name="risk_block",
            runtime_flags={"execute_trades": False, "gemini_influence_enabled": False},
            raw_signals=raw_signals,
            active_signals=active_signals,
            chosen_signal=deepcopy(chosen),
            risk_snapshot=risk_snapshot,
            regime=regime,
            higher_tf_bias="buy",
            contextual_scores={"momentum_continuation": 0.22, "gold_micro_impulse": 0.11},
            contextual_details=contextual_details,
            best_actionable_candidate=best_actionable,
            overlay_meta=overlay_meta,
            support_count=2,
            min_confidence=0.25,
            risk_allowed=False,
            risk_reason="Daily drawdown limit reached",
        ),
        ReplayScenario(
            name="live_execute",
            runtime_flags={"execute_trades": True, "gemini_influence_enabled": False},
            raw_signals=raw_signals,
            active_signals=active_signals,
            chosen_signal=deepcopy(chosen),
            risk_snapshot=risk_snapshot,
            regime=regime,
            higher_tf_bias="buy",
            contextual_scores={"momentum_continuation": 0.22, "gold_micro_impulse": 0.11},
            contextual_details=contextual_details,
            best_actionable_candidate=best_actionable,
            overlay_meta=overlay_meta,
            support_count=2,
            min_confidence=0.25,
            base_volume=0.32,
            execution_result=ExecutionResult(
                success=True,
                retcode=10009,
                comment="filled",
                order_id=321,
                deal_id=654,
            ),
        ),
    ]


def _build_trader(scenario: ReplayScenario) -> tuple[AtlasTrader, _ReplayStore]:
    store = _ReplayStore(scenario.runtime_flags)
    settings = _base_settings(execute_trades=bool(scenario.runtime_flags.get("execute_trades", False)))
    trader = AtlasTrader.__new__(AtlasTrader)
    trader.settings = settings
    trader.store = store
    trader.feed = SimpleNamespace(get_rates=lambda bars: _fixture_df().copy())
    trader.allocator = SimpleNamespace(
        choose=lambda selection_pool, score_overrides=None: deepcopy(scenario.chosen_signal),
        update=lambda strategy, pnl: None,
    )
    trader.risk = SimpleNamespace(
        evaluate=lambda snapshot: (bool(scenario.risk_allowed), str(scenario.risk_reason)),
        lot_size=lambda equity, stop_loss_points, tick_value: float(scenario.base_volume),
    )
    trader.executor = SimpleNamespace(
        execute=lambda signal, volume, decision_id="": scenario.execution_result,
        close_stale_positions=lambda: [],
    )
    trader.gemini = SimpleNamespace(
        enabled=False,
        health_summary=lambda: {"enabled": False, "providers": []},
        summarize_market_context=lambda payload: "observer disabled",
    )
    trader.connector = SimpleNamespace(health=lambda: {"connected": True})
    trader.voice = SimpleNamespace(speak=lambda text: None)
    trader._step_event_cache = {}
    trader._decision_sample_cache = {"loaded_at": 0.0, "limit": 0, "maps": ({}, {}, {})}
    trader._last_runtime_flags = {}

    trader._sync_closed_trade_learning = lambda runtime_flags=None: {}
    trader._apply_open_position_feedback = lambda runtime_flags=None: None
    trader._maybe_emit_drift_alert = lambda loss_memory, runtime_flags=None: None
    trader._drift_state = lambda loss_memory: {
        "active": False,
        "severe": False,
        "delta": 0.0,
        "recent_win_rate_20": 0.5,
        "baseline_win_rate": 0.5,
    }
    trader._build_signals = lambda df: deepcopy(scenario.raw_signals)
    trader._filter_weak_strategies = lambda raw_signals: (deepcopy(scenario.active_signals), [])
    trader._apply_drift_signal_adjustment = lambda signals, drift: {}
    trader._apply_drift_strategy_weighting = lambda signals, drift: {}
    trader._risk_snapshot = lambda runtime_flags=None: scenario.risk_snapshot
    trader._market_regime = lambda recent_closes: dict(scenario.regime)
    trader._higher_tf_bias = lambda: str(scenario.higher_tf_bias)
    trader._contextual_signal_scores = lambda **kwargs: (
        dict(scenario.contextual_scores),
        deepcopy(scenario.contextual_details),
    )
    trader._contextual_ranking_view = lambda contextual_details, chosen_strategy, contextual_scores: {
        "chosen_score": round(float(dict(scenario.contextual_scores).get(str(chosen_strategy), 0.0) or 0.0), 4),
        "top": [
            {
                "strategy": signal.strategy,
                "score": round(float(dict(scenario.contextual_scores).get(signal.strategy, 0.0) or 0.0), 4),
                "factor": 1.0,
                "setup_quality_score": round(float(scenario.best_actionable_candidate.get("setup_quality_score", 0.0) or 0.0), 4),
                "reasons": list(scenario.best_actionable_candidate.get("reasons", []) or []),
                "confidence": round(float(signal.confidence), 4),
                "side": signal.side,
            }
            for signal in list(scenario.active_signals)[:3]
        ],
        "best_actionable": {},
    }
    trader._best_actionable_candidate_view = lambda **kwargs: deepcopy(scenario.best_actionable_candidate)
    trader._top_signal_rows = lambda signals, contextual_scores=None, limit=5: _top_rows(list(signals)[:limit])
    trader._market_status_view = lambda closes, times, limit=20: {
        "last_close": float(closes[-1]),
        "recent_close": [float(x) for x in closes[-limit:]],
        "recent_time": list(times[-limit:]),
    }
    # Keep replay deterministic across wall-clock timezones/hours by pinning
    # session resolution to the scenario's synthetic overlay context.
    trader._session_label = lambda when_utc: str(scenario.overlay_meta.get("session", "london") or "london")
    trader._load_guard_state = lambda runtime_flags=None: {
        "trades_today": 0,
        "consecutive_losses": 0,
        "start_balance": 10000.0,
        "week_start_balance": 10000.0,
    }
    trader._save_guard_state = lambda guard_state, runtime_flags=None: None
    trader._emit_llm_health_alert_if_needed = lambda llm_health, flags=None: None
    trader._guard_block_reason = lambda guard_state: str(scenario.guard_block_reason)
    trader._drift_throttle_active = lambda drift, now_utc, runtime_flags=None: (False, 0)
    trader._same_side_support_count = lambda signals, side: int(scenario.support_count)
    trader._min_conf_threshold = lambda regime: float(scenario.min_confidence)
    trader._dynamic_min_confidence = lambda base_min_conf, loss_memory: (float(scenario.min_confidence), {"enabled": True})
    trader._loss_pattern_penalty = lambda side, now_utc, loss_memory: (0.0, "")
    trader._loss_pattern_penalty_detail = lambda side, now_utc, loss_memory: {"applied": False, "penalty": 0.0, "reason": ""}
    trader._recent_side_penalty = lambda side, loss_memory, current_session="": (0.0, "")
    trader._recent_side_penalty_detail = lambda side, loss_memory, current_session="": {"applied": False, "penalty": 0.0, "reason": ""}
    trader._loss_pattern_block_reason = lambda side, loss_memory, now_utc: ""
    trader._loss_pattern_block_detail = lambda side, loss_memory, now_utc: {"blocked": False, "reason": ""}
    trader._confidence_multiplier = lambda side, snapshot, guard_state, regime: (1.0, "")
    trader._apply_side_balance_penalty = lambda chosen: (float(chosen.confidence), {"applied": False})
    trader._execution_cluster_guard = lambda chosen: {"applied": False}
    trader._smart_context_adjustment = lambda **kwargs: (float(kwargs["chosen"].confidence), {"applied": False})
    trader._m1_quality_adjustment = lambda **kwargs: (float(kwargs["chosen"].confidence), {"applied": False})
    trader._m1_entry_block_reason = lambda chosen, meta: ""
    trader._timing_discipline_snapshot = lambda signal, context, status=None: {
        "hard_block": False,
        "reason": "",
        "observability_only": True,
    }
    trader._research_regime_overlay_meta = lambda df, snapshot, now_utc: dict(scenario.overlay_meta)
    trader._win_memory_boost = lambda chosen, now_utc: (float(chosen.confidence), {"applied": False})
    trader._meta_model_adjustment = lambda **kwargs: (float(kwargs["chosen"].confidence), {"enabled": False})
    trader._volume_multiplier = lambda guard_state, regime: (1.0, "neutral")
    trader._set_drift_throttle_after_execution = lambda drift, now_utc, runtime_flags=None: None
    trader._size_band_multiplier = lambda structured_analysis, guard_state, regime: (1.0, {"band": "guarded", "cap_multiplier": 1.0, "rationale": []})
    trader._size_tier_profile = lambda balance: {
        "current_size_tier": "test_tier",
        "max_lot": 1.0,
        "next_size_tier_requirement": "none",
        "size_tier_reason": "test",
    }

    class _MockExecutionGate:
        def __init__(self, scenario: ReplayScenario) -> None:
            self._scenario = scenario

        def set_execution_environment(self, env):
            pass

        def create_request(self, **kwargs):
            return {"id": "req_test_001", **kwargs}

        def execute(self, req_id, **kwargs):
            result = self._scenario.execution_result
            if result is not None:
                return {
                    "ok": True,
                    "request": {"id": req_id, "execution_status": "executed"},
                    "result": {
                        "broker_order_id": f"BRK_{req_id}",
                        "status": result.comment,
                        "order_id": result.order_id,
                        "deal_id": result.deal_id,
                        "retcode": result.retcode,
                    },
                }
            return {"ok": True, "request": {"id": req_id, "execution_status": "executed"}}

        def approve_request(self, req_id, **kwargs):
            pass

    trader.execution_gate = _MockExecutionGate(scenario)
    return trader, store


def project_behavior(status: dict[str, Any], store: _ReplayStore) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    execution = dict(status.get("execution", {}) or {})
    published = dict(store.saved_status or {})
    return {
        "action": str(status.get("action", "")),
        "message": str(status.get("message", "")),
        "execute_trades": bool(status.get("execute_trades", False)),
        "chosen": {
            "strategy": str(chosen.get("strategy", "")),
            "side": str(chosen.get("side", "")),
            "confidence": round(float(chosen.get("confidence", 0.0) or 0.0), 4),
            "stop_loss_points": round(float(chosen.get("stop_loss_points", 0.0) or 0.0), 4),
            "take_profit_points": round(float(chosen.get("take_profit_points", 0.0) or 0.0), 4),
        },
        "same_side_support": int(dict(status.get("same_side_support", {}) or {}).get("count", 0) or 0),
        "min_confidence_threshold": round(float(status.get("min_confidence_threshold", 0.0) or 0.0), 4),
        "volume": round(float(status.get("volume", 0.0) or 0.0), 4),
        "execution": {
            "success": bool(execution.get("success", False)),
            "retcode": int(execution.get("retcode", 0) or 0),
            "comment": str(execution.get("comment", "")),
            "decision_id": str(execution.get("decision_id", "")),
            "order_id": int(execution.get("order_id", 0) or 0),
            "deal_id": int(execution.get("deal_id", 0) or 0),
        },
        "logged_events": [
            {
                "type": str(row.get("type", "")),
                "message": str(row.get("message", "")),
            }
            for row in list(store.events)
        ],
        "published": {
            "action": str(published.get("action", "")),
            "message": str(published.get("message", "")),
        },
    }


def run_replay_scenario(scenario: ReplayScenario) -> dict[str, Any]:
    trader, store = _build_trader(scenario)
    fake_symbol_info = SimpleNamespace(trade_tick_value=1.0)
    with patch.object(trader_module.mt5, "account_info", return_value=SimpleNamespace(login=123456)):
        with patch.object(trader_module.mt5, "symbol_info", return_value=fake_symbol_info):
            with patch("atlas_ai.trader.uuid4", return_value=SimpleNamespace(hex="deadbeefcafebabe")):
                with patch("atlas_ai.trader.evaluate_heartbeat_state", return_value={"is_stale": False, "reason": "ok"}):
                    status = trader.step()
    return project_behavior(status, store)


def _diff_values(expected: Any, actual: Any, path: str = "") -> list[str]:
    diffs: list[str] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            next_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append(f"{next_path}: unexpected key in actual output")
                continue
            if key not in actual:
                diffs.append(f"{next_path}: missing from actual output")
                continue
            diffs.extend(_diff_values(expected[key], actual[key], next_path))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        max_len = max(len(expected), len(actual))
        for idx in range(max_len):
            next_path = f"{path}[{idx}]"
            if idx >= len(expected):
                diffs.append(f"{next_path}: unexpected item in actual output")
                continue
            if idx >= len(actual):
                diffs.append(f"{next_path}: missing from actual output")
                continue
            diffs.extend(_diff_values(expected[idx], actual[idx], next_path))
        return diffs
    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
    return diffs


def assert_behavior_matches_baseline(name: str, actual: dict[str, Any], baseline: dict[str, Any]) -> None:
    diffs = _diff_values(baseline, actual)
    if diffs:
        joined = "\n".join(f"- {line}" for line in diffs)
        raise ReplayDiffError(f"Deterministic replay drift detected for scenario={name}\n{joined}")


def load_replay_baselines(path: str | Path) -> dict[str, Any]:
    import json

    raw = Path(path).read_text(encoding="utf-8")
    return dict(json.loads(raw))
