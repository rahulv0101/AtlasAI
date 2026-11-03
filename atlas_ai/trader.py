from __future__ import annotations

import logging
import contextlib
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import MetaTrader5 as mt5

from atlas_ai.config.settings import Settings, get_active_thresholds
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.analysis.regime_tagging import classify_regime_label
from atlas_ai.analysis.setup_hunter import build_setup_hunter
from atlas_ai.analysis.trader_cognition import build_trader_cognition
from atlas_ai.decision_support import build_decision_explanation, build_decision_trace
from atlas_ai.execution.comment_codec import parse_broker_comment
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.mt5_broker_adapter import MT5BrokerAdapter
from atlas_ai.execution.trade_executor import TradeExecutor
from atlas_ai.learning.edge_building import score_live_forward_safe_setup
from atlas_ai.learning.outcome_intelligence import refresh_trade_outcome_intelligence
from atlas_ai.learning.online_allocator import OnlineAllocator
from atlas_ai.learning.meta_model import SignalQualityModel, context_to_features
from atlas_ai.learning.post_trade_lessons import build_post_trade_lesson
from atlas_ai.learning.loss_lesson_backfill import rebuild_loss_pattern_lessons
from atlas_ai.learning.loss_to_win_playbook import build_loss_to_win_playbook
from atlas_ai.learning.missed_entry_audit import (
    build_blocker_selectivity_collection_plan,
    build_blocker_selectivity_collection_readiness,
    build_blocker_selectivity_offline_collector,
    build_blocker_selectivity_replay,
    build_blocker_selectivity_review,
    record_missed_entry_audit,
    review_missed_entry_outcomes,
    summarize_missed_entry_learning,
    validate_blocker_selectivity_shadow_compare,
)
from atlas_ai.learning.winner_playbook import build_winner_playbook
from atlas_ai.llm.gemini_helper import GeminiAdvisor
from atlas_ai.runtime_heartbeat import evaluate_heartbeat_state
from atlas_ai.voice.output import SpeechOutput
from atlas_ai.market_context import build_context_snapshot
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.regime_filters import (
    build_research_regime_overlay_context,
    conservative_pretrade_filter_reason,
    research_regime_filter_reason,
)
from atlas_ai.shadow_intelligence import build_shadow_advice
from atlas_ai.self_improvement import build_self_improvement_summary, load_self_improvement_bundle
from atlas_ai.setup_quality import score_setup_quality
from atlas_ai.risk.risk_manager import RiskManager, RiskSnapshot
from atlas_ai.strategy_aliases import canonical_strategy_name
from atlas_ai.strategies.built_in import (
    BreakoutStrategy,
    CandlestickPatternStrategy,
    EMACrossoverClosedConfirmMidExitStrategy,
    EMACrossoverStrategy,
    RSIMeanReversionStrategy,
)
from atlas_ai.strategies.minute_pack import (
    BollingerMeanReversionStrategy,
    GoldBreakoutRetestStrategy,
    GoldLiquiditySweepStrategy,
    GoldMicroImpulseStrategy,
    GoldMicroStructureReversionStrategy,
    MACDTrendStrategy,
    MomentumContinuationStrategy,
    PullbackTrendM15MidExitStrategy,
    PullbackTrendStrategy,
    RSITrendHybridStrategy,
    StochasticReversalStrategy,
    VWAPProxyReversionStrategy,
)
from atlas_ai.strategies.strategy_lab import EvolvedRuleStrategy, StrategyLab
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.strategy_factory import StrategyFactory
from atlas_ai.strategies.evolution_drawdown_guard import EvolutionDrawdownGuard
from atlas_ai.strategies.evolution_integration import load_demo_chromosome_pool
from atlas_ai.system_health import build_system_health
from atlas_ai.gold_knowledge import apply_gold_knowledge_to_signal
from atlas_ai.vision.chart_vision import analyze_chart_visual_context, capture_desktop_screenshot, load_latest_visual_read, save_visual_read

logger = logging.getLogger(__name__)


class AtlasTrader:
    def __init__(self, settings: Settings, store: KnowledgeStore | None = None):
        self.settings = settings
        self.store = store or KnowledgeStore(settings.storage_dir)
        self.feed = MarketDataFeed(settings)
        self.allocator = OnlineAllocator(store=self.store)
        self.allocator.set_exploration(getattr(settings, "exploration_epsilon", 0.0))
        self.risk = RiskManager(settings)
        # Explicit exception: TradeExecutor remains for stale/open position management only.
        # New entry execution routing is guarded through ExecutionGate.
        self.executor = TradeExecutor(settings)
        # Single execution authority for demo/live-like routing.
        self.execution_gate = ExecutionGate(state_store_path=str(Path(self.settings.storage_dir) / "automation_state.json"))
        self.execution_gate.set_broker_adapter(MT5BrokerAdapter(self.executor))
        self._last_visual_chart_capture_at = 0.0
        self.lab = StrategyLab()
        self.gemini = GeminiAdvisor(
            gemini_api_key=settings.gemini_api_key,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        self.voice = SpeechOutput()
        self.base_strategies = [
            EMACrossoverStrategy(),
            RSIMeanReversionStrategy(),
            BreakoutStrategy(),
            CandlestickPatternStrategy(),
            MACDTrendStrategy(),
            BollingerMeanReversionStrategy(),
            StochasticReversalStrategy(),
            VWAPProxyReversionStrategy(),
            MomentumContinuationStrategy(),
            PullbackTrendStrategy(),
            RSITrendHybridStrategy(),
        ]
        if str(self.settings.timeframe).upper() == "M1":
            self.base_strategies.extend(
                [
                    GoldMicroImpulseStrategy(),
                    GoldBreakoutRetestStrategy(),
                    GoldLiquiditySweepStrategy(),
                    GoldMicroStructureReversionStrategy(),
                ]
            )
        if str(self.settings.timeframe).upper() == "M15" and not bool(getattr(self.settings, "execute_trades", False)):
            self.base_strategies.append(EMACrossoverClosedConfirmMidExitStrategy())
            self.base_strategies.append(PullbackTrendM15MidExitStrategy())
        self._logged_evolved_fallback = False
        self._step_event_cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._decision_sample_cache: dict[str, Any] = {
            "loaded_at": 0.0,
            "limit": 0,
            "maps": ({}, {}, {}),
        }
        self._current_cycle_timing: dict[str, Any] = {}
        self._last_runtime_flags: dict[str, Any] = {}
        self._closed_learning_synced_until: datetime | None = None
        self._closed_learning_sync_persisted_at: datetime | None = None
        self._last_macro_context_refresh_attempt_at = 0.0
        self._macro_context_refresh_in_flight = False
        self._macro_context_refresh_lock = threading.Lock()
        self._ensure_learning_playbook()

    @staticmethod
    def _freshness_runtime_payload(*, status_age_ms: float | None, max_age_ms: int = 120000) -> dict[str, Any]:
        age = float(status_age_ms) if status_age_ms is not None else None
        return {
            "status_age_ms": age,
            "is_stale": bool(age is None or age > float(max_age_ms)),
        }

    def _current_cycle_age_ms(self) -> float | None:
        timing = dict(getattr(self, "_current_cycle_timing", {}) or {})
        raw_age = timing.get("backend_cycle_age_ms")
        if raw_age is not None:
            try:
                return float(raw_age)
            except Exception:
                pass
        cycle_started_at = str(timing.get("cycle_started_at", "") or "")
        if not cycle_started_at:
            return None
        try:
            cycle_dt = datetime.fromisoformat(cycle_started_at)
            if cycle_dt.tzinfo is None:
                cycle_dt = cycle_dt.replace(tzinfo=timezone.utc)
            return round(max(0.0, (datetime.now(timezone.utc) - cycle_dt).total_seconds()) * 1000.0, 2)
        except Exception:
            return None

    def _macro_context_refresh_interval_seconds(self) -> float:
        raw = str(os.getenv("ATLAS_MACRO_CONTEXT_REFRESH_SECONDS", "") or "").strip()
        if raw:
            try:
                return max(60.0, float(raw))
            except ValueError:
                logger.warning("Invalid ATLAS_MACRO_CONTEXT_REFRESH_SECONDS=%r; using 90 minutes.", raw)
        return 90.0 * 60.0

    def _maybe_refresh_external_macro_context(self) -> None:
        interval_seconds = self._macro_context_refresh_interval_seconds()
        now_mono = time.monotonic()
        lock = getattr(self, "_macro_context_refresh_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._macro_context_refresh_lock = lock

        with lock:
            if bool(getattr(self, "_macro_context_refresh_in_flight", False)):
                return
            last_attempt = float(getattr(self, "_last_macro_context_refresh_attempt_at", 0.0) or 0.0)
            if last_attempt > 0.0 and (now_mono - last_attempt) < interval_seconds:
                return
            self._last_macro_context_refresh_attempt_at = now_mono
            self._macro_context_refresh_in_flight = True

        store = self.store

        def _refresh_worker() -> None:
            try:
                from atlas_ai.macro_context_updater import write_external_context

                payload = write_external_context()
                meta = {
                    "generated_at": payload.get("generated_at"),
                    "source": payload.get("source"),
                    "errors": payload.get("errors", {}),
                    "refresh_interval_seconds": interval_seconds,
                    "trigger": "backend_live_cycle",
                }
                logger.info("External macro context refreshed by backend live cycle: %s", meta)
                store.log_event("system", "External macro context refreshed by backend live cycle.", meta)
            except Exception as exc:
                meta = {
                    "error": str(exc),
                    "refresh_interval_seconds": interval_seconds,
                    "trigger": "backend_live_cycle",
                }
                logger.warning("External macro context refresh failed in backend live cycle: %s", exc, exc_info=True)
                with contextlib.suppress(Exception):
                    store.log_event("warning", "External macro context refresh failed in backend live cycle.", meta)
            finally:
                with lock:
                    self._macro_context_refresh_in_flight = False

        threading.Thread(
            target=_refresh_worker,
            name="atlas-macro-context-refresh",
            daemon=True,
        ).start()

    @staticmethod
    def _execution_result_counts_as_trade(result: Any) -> bool:
        if not bool(getattr(result, "success", False)):
            return False
        try:
            return int(getattr(result, "order_id", 0) or 0) > 0 or int(getattr(result, "deal_id", 0) or 0) > 0
        except Exception:
            return False

    def _guarded_demo_auto_approval_decision(
        self,
        *,
        environment: str,
        status: dict[str, Any],
        setup_family: str,
        volume: float,
        runtime_flags: dict[str, Any],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        env = str(environment or "").lower()
        symbol = str(getattr(self.settings, "symbol", "") or "")
        chosen = dict(status.get("chosen", {}) or {})
        side = str(chosen.get("side", "") or "").lower()
        confidence = float(chosen.get("confidence", 0.0) or 0.0)
        min_conf = float(status.get("min_confidence_threshold", 0.0) or 0.0)
        safety_layer = dict(status.get("safety_layer", {}) or {})
        directional_guard = dict(status.get("directional_structure_guard", {}) or {})
        allowed_setups = {"continuation", "pullback", "breakout"}

        if env != "demo":
            reasons.append("environment_not_demo")
        if symbol.upper() != "XAUUSD":
            reasons.append("instrument_not_xauusd")
        if side not in {"buy", "sell"}:
            reasons.append("side_not_directional")
        if str(setup_family or "").lower() not in allowed_setups:
            reasons.append("setup_family_not_auto_approved")
        if float(volume) > 0.01:
            reasons.append("size_above_guarded_demo_cap")
        if confidence < max(min_conf, 0.55):
            reasons.append("confidence_below_auto_approval_floor")
        if bool(safety_layer.get("safety_lock_active", False)):
            reasons.append("safety_lock_active")
        if bool(safety_layer.get("cooldown_active", False)):
            reasons.append("cooldown_active")
        if bool(directional_guard.get("hard_blocked", False)):
            reasons.append("directional_structure_blocked")
        if bool(dict(runtime_flags or {}).get("reconciliation_required", False)):
            reasons.append("reconciliation_required")
        return {
            "approved": not reasons,
            "reasons": reasons,
            "policy": "guarded_demo_auto_approval_v1",
            "max_size": 0.01,
            "min_confidence": max(min_conf, 0.55),
        }

    def _strategy_enabled(self, strategy_name: str) -> bool:
        key = self._strategy_key(strategy_name)
        disabled = {self._strategy_key(x) for x in getattr(self.settings, "disabled_strategies", ())}
        enabled = {self._strategy_key(x) for x in getattr(self.settings, "enabled_strategies", ()) if str(x).strip()}
        if enabled and key not in enabled:
            return False
        return key not in disabled

    def _strategy_to_setup_family(self, strategy_name: str) -> str:
        key = self._strategy_key(strategy_name)
        if any(x in key for x in ("breakout", "liquidity_sweep")):
            return "breakout"
        if any(x in key for x in ("reversion", "mean_reversion", "stochastic_reversal", "rsi_mean_reversion", "vwap_proxy_reversion")):
            return "mean_reversion"
        if any(x in key for x in ("pullback", "continuation", "momentum", "ema_crossover", "macd_trend", "rsi_trend_hybrid", "gold_micro_impulse")):
            return "continuation"
        return "continuation"

    def _format_evolved_strategy_name(self, evolved: EvolvedRuleStrategy) -> str:
        return (
            f"evolved_rule(fast={evolved.fast},slow={evolved.slow},"
            f"rsi_buy={evolved.rsi_buy:.1f},rsi_sell={evolved.rsi_sell:.1f})"
        )

    def _learning_status_snapshot(self) -> dict[str, Any]:
        """P2 — Surface learning pipeline status for UI/operator visibility."""
        state = self.store.load_learning_state().get("state", {})
        pool = dict(state.get("evolved_strategy_pool", {}) or {})
        demo_status = dict(getattr(self, "_evolved_strategy_status", {}) or {})
        mm = dict(state.get("meta_model", {}) or {})
        last_training = str(pool.get("generated_at", "") or "")
        pool_count = int(pool.get("count", 0) or 0)
        pool_stale = False
        if last_training:
            try:
                generated = datetime.fromisoformat(last_training.replace("Z", "+00:00"))
                age_hours = max(0.0, (datetime.now(timezone.utc) - generated).total_seconds() / 3600.0)
                threshold_hours = max(0.5, float(getattr(self.settings, "historical_retrain_hours", 24.0)))
                pool_stale = age_hours >= threshold_hours
            except Exception:
                pass
        return {
            "evolved_pool_count": pool_count,
            "demo_chromosome_active_count": int(demo_status.get("active_count", 0) or 0),
            "demo_chromosome_active_ids": list(demo_status.get("active_ids", []) or [])[:5],
            "demo_chromosome_status": demo_status,
            "evolved_pool_stale": pool_stale,
            "last_training_time": last_training,
            "meta_model_status": str(mm.get("status", "untrained") or "untrained"),
            "meta_model_accuracy": round(float(dict(mm.get("metrics", {}) or {}).get("accuracy", 0.0)), 4),
            "meta_model_brier": round(float(dict(mm.get("metrics", {}) or {}).get("brier", 1.0)), 4),
            "learning_playbook_present": bool(state.get("learning_playbook", {})),
            "allocator_strategies_tracked": len(dict(state.get("allocator", {}) or {})),
        }

    def _sanitize_signal(self, signal: TradeSignal) -> TradeSignal:
        side = str(getattr(signal, "side", "flat")).lower()
        if side not in {"buy", "sell"}:
            signal.side = "flat"
            signal.confidence = 0.0
        else:
            conf = max(0.0, min(1.0, float(getattr(signal, "confidence", 0.0) or 0.0)))
            strategy = str(getattr(signal, "strategy", "") or "")
            if strategy.startswith("evolved_rule("):
                # Prevent evolved variants from bunching at 1.0 and dominating ranking
                # before live evidence separates them.
                conf = min(0.82, (conf * 0.72) + 0.08)
            signal.confidence = conf
        return signal

    def _load_chromosome_demo_pool(self) -> list[StrategyChromosome]:
        """Load demo-stage chromosomes from the evolution engine population file.

        Uses evolution_integration.load_demo_chromosome_pool for safe,
        testable loading with drawdown guard and promotion filtering.
        """
        storage_dir = str(
            getattr(
                self.settings,
                "storage_dir",
                getattr(getattr(self, "store", None), "base", "storage"),
            )
            or "storage"
        )
        active, status = load_demo_chromosome_pool(
            storage_dir=storage_dir,
            max_demo=max(0, int(getattr(self.settings, "max_demo_strategies", 3))),
            max_demo_drawdown_pct=float(getattr(self.settings, "evolution_max_demo_drawdown_pct", 0.05)),
            max_shadow_drawdown_pct=float(getattr(self.settings, "evolution_max_shadow_drawdown_pct", 0.05)),
        )
        # Persist runtime status for API/UI visibility
        self._evolved_strategy_status = status
        if status.get("paused"):
            for p in status["paused"]:
                self.store.log_event(
                    "evolution",
                    f"Demo strategy {p['uid']} paused by drawdown guard: {p['reason']}",
                )
        return active

    def _load_evolved_strategy_pool(self) -> list[EvolvedRuleStrategy]:
        state = self.store.load_learning_state().get("state", {})
        pool = list(dict(state.get("evolved_strategy_pool", {})).get("candidates", []))
        strategies: list[EvolvedRuleStrategy] = []
        seen: set[str] = set()
        for candidate in pool:
            try:
                evolved = EvolvedRuleStrategy(
                    fast=int(candidate.get("fast", 20)),
                    slow=int(candidate.get("slow", 50)),
                    rsi_buy=float(candidate.get("rsi_buy", 45.0)),
                    rsi_sell=float(candidate.get("rsi_sell", 55.0)),
                )
            except Exception:
                continue
            signature = self._format_evolved_strategy_name(evolved)
            if signature in seen:
                continue
            seen.add(signature)
            strategies.append(evolved)
        learned = dict(state.get("best_evolved_strategy", {}))
        if learned:
            evolved = EvolvedRuleStrategy(
                fast=int(learned.get("fast", 20)),
                slow=int(learned.get("slow", 50)),
                rsi_buy=float(learned.get("rsi_buy", 45.0)),
                rsi_sell=float(learned.get("rsi_sell", 55.0)),
            )
            signature = self._format_evolved_strategy_name(evolved)
            if signature not in seen:
                strategies.insert(0, evolved)
        return strategies[: max(1, int(getattr(self.settings, "evolved_strategy_survivors", 4)))]

    def _load_meta_model(self) -> tuple[SignalQualityModel | None, dict[str, Any]]:
        raw = self.store.load_learning_state().get("state", {})
        mm = dict(raw.get("meta_model", {}))
        model_payload = dict(mm.get("model", {}))
        if not model_payload:
            return None, mm
        return SignalQualityModel.from_dict(model_payload), mm

    def _meta_model_probability(self, context: dict[str, Any]) -> tuple[float | None, str]:
        if not bool(getattr(self.settings, "meta_model_enabled", True)):
            return None, "disabled"
        model, _ = self._load_meta_model()
        if model is None:
            return None, "model_unavailable"
        return float(model.predict_proba(context_to_features(context))), "ok"

    def _meta_model_effect_scale(self) -> float:
        _, meta = self._load_meta_model()
        status = str(meta.get("status", "") or "").strip().lower()
        if status == "promoted":
            return 1.0
        return 0.0

    def _meta_model_adjustment(
        self,
        chosen: TradeSignal,
        regime: dict[str, Any],
        higher_tf_bias: str,
        snapshot: RiskSnapshot,
        when_utc: datetime,
        context: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        if not bool(getattr(self.settings, "meta_model_enabled", True)):
            return float(chosen.confidence), {"enabled": False}
        if str(getattr(chosen, "side", "flat")) not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}
        built_context = dict(context or {})
        if not built_context:
            built_context = {
                "strategy": str(getattr(chosen, "strategy", "")),
                "side": str(getattr(chosen, "side", "")),
                "session": self._session_label(when_utc),
                "hour_utc": int(when_utc.hour),
                "regime": str(regime.get("name", "unknown")),
                "higher_tf_bias": str(higher_tf_bias),
                "spread_points": round(float(getattr(snapshot, "spread_points", 0.0) or 0.0), 2),
                "confidence": round(float(getattr(chosen, "confidence", 0.0) or 0.0), 4),
            }
        p, reason = self._meta_model_probability(built_context)
        if p is None:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": reason}
        effect_scale = self._meta_model_effect_scale()
        if effect_scale <= 0.0:
            return float(chosen.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "candidate_disabled",
                "probability": round(float(p), 4),
                "effect_scale": 0.0,
            }
        cap = max(0.0, min(0.25, float(getattr(self.settings, "meta_model_confidence_cap", 0.08)) * effect_scale))
        # Center around 0.5: convert probability into bounded confidence multiplier.
        delta = (float(p) - 0.5) * 2.0 * cap
        factor = max(0.8, min(1.2, 1.0 + delta))
        new_conf = max(0.0, min(1.0, float(chosen.confidence) * factor))
        return new_conf, {
            "enabled": True,
            "applied": abs(factor - 1.0) > 1e-6,
            "probability": round(float(p), 4),
            "factor": round(factor, 4),
            "cap": round(cap, 4),
            "effect_scale": round(effect_scale, 4),
            "context": {
                "session": built_context.get("session", self._session_label(when_utc)),
                "hour_utc": int(built_context.get("hour_utc", int(when_utc.hour))),
                "side": built_context.get("side", str(getattr(chosen, "side", ""))),
                "strategy": self._strategy_key(str(built_context.get("strategy", str(getattr(chosen, "strategy", ""))))),
                "regime": str(built_context.get("regime", str(regime.get("name", "unknown")))),
                "higher_tf_bias": str(built_context.get("higher_tf_bias", str(higher_tf_bias))),
                "spread_points": round(float(built_context.get("spread_points", float(getattr(snapshot, "spread_points", 0.0) or 0.0))), 2),
            },
        }

    def _ensure_learning_playbook(self) -> None:
        existing = self.store.load_learning_state().get("state", {})
        playbook = dict(existing.get("learning_playbook", {}))
        if playbook:
            return
        playbook = {
            "core_principles": [
                "Protect capital first: avoid forcing entries in weak/uncertain conditions.",
                "A loss is feedback: reduce confidence and size in repeated losing patterns.",
                "Prefer setups aligned with higher-timeframe bias unless confidence is exceptional.",
                "Wait for confirmation and acceptable spread/slippage.",
            ],
            "loss_definition": "Closed trade with net_pnl_zar < 0.",
            "improvement_loop": [
                "Record closed outcomes with session/hour/side context.",
                "Penalize side-hour combinations with persistent loss rates.",
                "Escalate penalty on repeated losses for the same side.",
                "Use rolling outcomes for drift alerts.",
            ],
        }
        self.store.update_learning_state(lambda current: {**current, "learning_playbook": playbook})
        self.store.log_event("learning", "Initialized learning playbook for continuous improvement.")

    def _tail_events_cached(self, limit: int, event_type: str = "") -> list[dict[str, Any]]:
        if not hasattr(self, "_step_event_cache"):
            self._step_event_cache = {}
        key = (max(1, int(limit)), str(event_type or "").strip().lower())
        cached = self._step_event_cache.get(key)
        if cached is not None:
            return list(cached)
        if key[1]:
            events = self.store.tail_events(limit=key[0], event_type=key[1])
        else:
            events = self.store.tail_events(limit=key[0])
        self._step_event_cache[key] = list(events)
        return list(events)

    def _decision_sample_maps(self, limit: int = 5000, ttl_seconds: float = 15.0) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
        if not hasattr(self, "_decision_sample_cache"):
            self._decision_sample_cache = {
                "loaded_at": 0.0,
                "limit": 0,
                "maps": ({}, {}, {}),
            }
        now = time.monotonic()
        cache_limit = int(self._decision_sample_cache.get("limit", 0) or 0)
        loaded_at = float(self._decision_sample_cache.get("loaded_at", 0.0) or 0.0)
        if cache_limit >= int(limit) and (now - loaded_at) <= max(1.0, float(ttl_seconds)):
            maps = self._decision_sample_cache.get("maps", ({}, {}, {}))
            return tuple(dict(x) for x in maps)  # type: ignore[return-value]

        decision_events = self.store.tail_events(limit=limit, event_type="decision_sample")
        decision_strategy_map: dict[str, str] = {}
        decision_id_aliases: dict[str, str] = {}
        decision_meta_by_execution_ref: dict[str, dict[str, Any]] = {}
        for event in decision_events:
            meta = dict(event.get("meta", {}) or {})
            decision_id = str(meta.get("decision_id", "") or "").strip().lower()
            strategy = self._strategy_key(str(meta.get("strategy", "") or ""))
            if decision_id and strategy:
                decision_strategy_map[decision_id] = strategy
                decision_id_aliases[decision_id] = decision_id
                short_id = decision_id[:6]
                if short_id:
                    decision_strategy_map.setdefault(short_id, strategy)
                    decision_id_aliases.setdefault(short_id, decision_id)
            for ref_key in ("deal_id", "order_id"):
                ref_value = str(meta.get(ref_key, "") or "").strip()
                if ref_value and decision_id:
                    contextual = dict(meta.get("contextual_ranking", {}) or {})
                    decision_meta_by_execution_ref[ref_value] = {
                        "decision_id": decision_id,
                        "strategy": strategy,
                        "setup_family": str(meta.get("setup_family", "") or ""),
                        "regime": str(meta.get("regime", "") or ""),
                        "session": str(meta.get("session", "") or ""),
                        "hour_utc": int(meta.get("hour_utc", 0) or 0),
                        "confidence": float(meta.get("confidence", 0.0) or 0.0),
                        "context": dict(contextual.get("context", {}) or {}),
                        "setup_quality": dict(contextual.get("setup_quality", {}) or {}),
                    }

        self._decision_sample_cache = {
            "loaded_at": now,
            "limit": int(limit),
            "maps": (
                dict(decision_strategy_map),
                dict(decision_id_aliases),
                dict(decision_meta_by_execution_ref),
            ),
        }
        return decision_strategy_map, decision_id_aliases, decision_meta_by_execution_ref

    def _market_regime(self, closes: list[float]) -> dict[str, Any]:
        if len(closes) < 30:
            tag = classify_regime_label(closes=closes)
            return {"name": "unknown", "trend_strength": 0.0, "volatility_pct": 0.0, "regime_label": tag["label"], "regime_label_detail": tag}
        first = float(closes[0])
        last = float(closes[-1])
        if first == 0:
            tag = classify_regime_label(closes=closes)
            return {"name": "unknown", "trend_strength": 0.0, "volatility_pct": 0.0, "regime_label": tag["label"], "regime_label_detail": tag}
        drift_pct = ((last - first) / first) * 100.0
        abs_moves = [abs(float(closes[i]) - float(closes[i - 1])) for i in range(1, len(closes))]
        avg_abs = sum(abs_moves) / max(1, len(abs_moves))
        vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0

        if abs(drift_pct) >= max(0.12, vol_pct * 2.2):
            name = "trend"
        elif vol_pct >= 0.06:
            name = "high_volatility_range"
        else:
            name = "range"
        tag = classify_regime_label(closes=closes)
        return {
            "name": name,
            "trend_strength": round(abs(drift_pct), 4),
            "volatility_pct": round(vol_pct, 4),
            "regime_label": tag["label"],
            "regime_label_detail": tag,
        }

    def _session_label(self, when_utc: datetime) -> str:
        h = int(when_utc.hour)
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 13:
            return "london"
        if 13 <= h < 20:
            return "new_york"
        return "late_us"

    def _spread_bucket(self, spread_points: float) -> str:
        if spread_points < 25:
            return "tight"
        if spread_points < 45:
            return "normal"
        return "wide"

    def _volatility_bucket(self, volatility_pct: float) -> str:
        if volatility_pct < 0.02:
            return "low"
        if volatility_pct < 0.05:
            return "medium"
        return "high"

    def _signal_volatility_state(self, volatility_pct: float) -> str:
        low_cut = float(getattr(self.settings, "signal_regime_gate_low_volatility_max_pct", 0.03) or 0.03)
        if volatility_pct < low_cut:
            return "low"
        if volatility_pct < max(low_cut * 2.0, 0.07):
            return "normal"
        return "high"

    def _signal_generation_context(self, df, when_utc: datetime | None = None) -> dict[str, Any]:
        when_utc = when_utc or datetime.now(timezone.utc)
        recent = df.tail(60)
        closes = [float(x) for x in recent["close"].tolist()] if "close" in recent else []
        regime = self._market_regime(closes)
        session = self._session_label(when_utc)
        volatility_pct = float(regime.get("volatility_pct", 0.0) or 0.0)
        trend_strength = float(regime.get("trend_strength", 0.0) or 0.0)
        return {
            "session": session,
            "regime": str(regime.get("name", "unknown") or "unknown"),
            "regime_label": str(regime.get("regime_label", "ranging") or "ranging"),
            "regime_label_detail": dict(regime.get("regime_label_detail", {}) or {}),
            "trend_strength": trend_strength,
            "volatility_pct": volatility_pct,
            "volatility_state": self._signal_volatility_state(volatility_pct),
            "allowed_session": session in {
                str(s).strip().lower()
                for s in getattr(self.settings, "m1_allowed_sessions", ("london", "new_york"))
                if str(s).strip()
            },
            **self._entry_timing_context(df, regime),
        }

    def _entry_timing_context(self, df, regime: dict[str, Any]) -> dict[str, Any]:
        recent = df.tail(90)
        if len(recent) < 25:
            return {
                "entry_location_state": "unclear",
                "entry_momentum_state": "mixed",
                "entry_wickiness": 0.0,
                "entry_expansion": 0.0,
                "entry_position": 0.5,
            }
        closes = [float(x) for x in recent["close"].tolist()]
        highs = [float(x) for x in recent["high"].tolist()]
        lows = [float(x) for x in recent["low"].tolist()]
        opens = [float(x) for x in recent["open"].tolist()]
        sma_fast = sum(closes[-8:]) / 8.0
        sma_slow = sum(closes[-21:]) / 21.0
        trend_state = "range"
        if sma_fast > sma_slow * 1.00015:
            trend_state = "uptrend"
        elif sma_fast < sma_slow * 0.99985:
            trend_state = "downtrend"

        returns = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent_returns = returns[-8:] if len(returns) >= 8 else returns
        pos = sum(1 for r in recent_returns if r > 0)
        neg = sum(1 for r in recent_returns if r < 0)
        last_range = max(1e-9, highs[-1] - lows[-1])
        body_ratio = abs(closes[-1] - opens[-1]) / last_range
        if (pos >= 6 or neg >= 6) and body_ratio >= 0.55:
            momentum_state = "strong"
        elif body_ratio < 0.30:
            momentum_state = "weak"
        else:
            momentum_state = "mixed"

        range_high = max(highs[-20:])
        range_low = min(lows[-20:])
        range_span = max(1e-9, range_high - range_low)
        position = (closes[-1] - range_low) / range_span
        if trend_state == "uptrend":
            location_state = "pullback" if position <= 0.42 else ("breakout" if position >= 0.86 and momentum_state == "strong" else "mid_range")
        elif trend_state == "downtrend":
            location_state = "pullback" if position >= 0.58 else ("breakout" if position <= 0.14 and momentum_state == "strong" else "mid_range")
        else:
            location_state = "pullback" if position <= 0.22 or position >= 0.78 else ("consolidation" if 0.35 <= position <= 0.65 else "mid_range")

        wickiness_values = []
        for idx in range(max(0, len(closes) - 10), len(closes)):
            candle_range = max(1e-9, highs[idx] - lows[idx])
            candle_body = abs(closes[idx] - opens[idx])
            wickiness_values.append(max(0.0, min(1.0, (candle_range - candle_body) / candle_range)))
        recent_range = sum(highs[-5:]) / 5.0 - sum(lows[-5:]) / 5.0
        baseline_range = (sum(highs[-25:-5]) / 20.0) - (sum(lows[-25:-5]) / 20.0)
        expansion_ratio = recent_range / max(1e-9, baseline_range) if baseline_range > 0 else 1.0
        return {
            "entry_location_state": location_state,
            "entry_momentum_state": momentum_state,
            "entry_wickiness": round(sum(wickiness_values) / max(1, len(wickiness_values)), 4),
            "entry_expansion": round(max(0.0, expansion_ratio - 1.0), 4),
            "entry_position": round(max(0.0, min(1.0, position)), 4),
            "entry_trend_state": trend_state,
            "entry_trend_strength": float(regime.get("trend_strength", 0.0) or 0.0),
        }

    def _historical_loss_pattern_counts(self) -> dict[str, int]:
        try:
            state = self.store.load_learning_state().get("state", {})
        except Exception:
            return {}
        analysis = dict(state.get("historical_loss_analysis", {}) or {})
        patterns = dict(analysis.get("patterns", {}) or {})
        rows = list(patterns.get("classification", []) or [])
        return {str(row.get("value", "") or ""): int(row.get("count", 0) or 0) for row in rows}

    def _load_loss_to_win_playbook(self) -> dict[str, Any]:
        try:
            state = self.store.load_learning_state().get("state", {})
        except Exception:
            return {}
        return dict(state.get("loss_to_win_playbook", {}) or {})

    def _load_winner_playbook(self) -> dict[str, Any]:
        try:
            state = self.store.load_learning_state().get("state", {})
        except Exception:
            return {}
        return dict(state.get("winner_playbook", {}) or {})

    @staticmethod
    def _confirmation_status_for_loss(classification: str, context: dict[str, Any]) -> dict[str, Any]:
        code = str(classification or "").strip().lower()
        location = str(context.get("entry_location_state", context.get("location", "")) or "").lower()
        momentum = str(context.get("entry_momentum_state", context.get("momentum", "")) or "").lower()
        wickiness = float(context.get("entry_wickiness", 0.0) or 0.0)
        expansion = float(context.get("entry_expansion", 0.0) or 0.0)
        position = float(context.get("entry_position", 0.5) or 0.5)
        trend_strength = float(context.get("entry_trend_strength", context.get("trend_strength", 0.0)) or 0.0)
        checks: list[dict[str, Any]] = []

        def add(code_name: str, passed: bool, detail: str) -> None:
            checks.append({"code": code_name, "status": "present" if passed else "missing", "detail": detail})

        if code == "weak_pullback":
            add("valid_pullback_location", location == "pullback", f"location={location or 'unknown'}")
            add("strong_or_turning_m1_momentum", momentum == "strong", f"momentum={momentum or 'unknown'}")
            add("controlled_wick_pressure", wickiness < 0.55, f"wickiness={wickiness:.2f}")
        elif code == "stop_hunt":
            add("reduced_wick_pressure", wickiness < 0.50, f"wickiness={wickiness:.2f}")
            add("momentum_confirmation", momentum in {"strong", "mixed"}, f"momentum={momentum or 'unknown'}")
            add("not_entry_into_raw_sweep", location != "mid_range", f"location={location or 'unknown'}")
        elif code == "late_entry":
            add("no_overextension", expansion < 0.45 and 0.10 < position < 0.90, f"expansion={expansion:.2f}, position={position:.2f}")
            add("fresh_entry_location", location in {"pullback", "breakout", "consolidation"}, f"location={location or 'unknown'}")
        elif code == "false_continuation":
            add("strong_m1_momentum", momentum == "strong", f"momentum={momentum or 'unknown'}")
            add("structure_hold", trend_strength >= 0.18, f"trend_strength={trend_strength:.2f}")
        else:
            add("not_mid_range", location != "mid_range", f"location={location or 'unknown'}")
            add("momentum_alignment", momentum == "strong", f"momentum={momentum or 'unknown'}")
            add("clear_structure", trend_strength >= 0.18, f"trend_strength={trend_strength:.2f}")

        missing = [row["code"] for row in checks if row["status"] != "present"]
        return {
            "classification": code or "unknown",
            "confirmed": not missing,
            "checks": checks,
            "missing": missing,
        }

    def _loss_to_win_snapshot(self, signal: TradeSignal, context: dict[str, Any]) -> dict[str, Any]:
        playbook = self._load_loss_to_win_playbook()
        lessons = dict(playbook.get("lessons", {}) or {})
        if not lessons:
            return {"enabled": True, "status": "no_loss_to_win_playbook", "lessons_available": 0}
        counts = self._historical_loss_pattern_counts()
        if not counts:
            return {"enabled": True, "status": "no_historical_loss_counts", "lessons_available": len(lessons)}
        strategy_key = self._strategy_key(str(getattr(signal, "strategy", "") or ""))
        candidates: list[dict[str, Any]] = []
        for classification, count in counts.items():
            lesson = dict(lessons.get(str(classification), {}) or {})
            if not lesson:
                continue
            strategies = {
                str(row.get("value", "") or "").strip().lower()
                for row in list(lesson.get("strategies", []) or [])
            }
            strategy_relevant = not strategies or strategy_key in strategies or "unknown" in strategies
            confirmation = self._confirmation_status_for_loss(str(classification), context)
            candidates.append(
                {
                    "classification": str(classification),
                    "samples": int(count),
                    "strategy_relevant": bool(strategy_relevant),
                    "confirmed": bool(confirmation.get("confirmed", False)),
                    "missing": list(confirmation.get("missing", []) or []),
                    "checks": list(confirmation.get("checks", []) or [])[:6],
                    "corrected_entry": str(lesson.get("corrected_entry", "") or ""),
                    "confirmation_required": list(lesson.get("confirmation_required", []) or [])[:6],
                }
            )
        candidates.sort(key=lambda row: (bool(row.get("strategy_relevant", False)), int(row.get("samples", 0) or 0)), reverse=True)
        top = candidates[0] if candidates else {}
        if top and bool(top.get("confirmed", False)):
            status = "corrected_loss_pattern_present"
            next_action = "allow normal gates to judge the corrected setup"
        elif top:
            status = "loss_pattern_needs_correction"
            next_action = "wait for corrected-entry confirmations before trusting this setup"
        else:
            status = "watching"
            next_action = "no dominant historical loss lesson matched"
        return {
            "enabled": True,
            "schema": str(playbook.get("schema", "atlas_loss_to_win_playbook_v1") or ""),
            "status": status,
            "source": str(playbook.get("source", "") or ""),
            "lessons_available": len(lessons),
            "top_lesson": top,
            "candidates": candidates[:4],
            "next_action": next_action,
            "policy_note": str(playbook.get("policy_note", "") or "Loss-to-win playbook is bounded teaching evidence only."),
        }

    def _winner_playbook_snapshot(self, signal: TradeSignal, context: dict[str, Any]) -> dict[str, Any]:
        playbook = self._load_winner_playbook()
        lessons = dict(playbook.get("lessons", {}) or {})
        if not lessons:
            return {"enabled": True, "status": "no_winner_playbook", "lessons_available": 0}
        strategy_key = self._strategy_key(str(getattr(signal, "strategy", "") or ""))
        setup_family = self._strategy_to_setup_family(str(getattr(signal, "strategy", "") or ""))
        location = str(context.get("entry_location_state", context.get("location", "")) or "").lower()
        momentum = str(context.get("entry_momentum_state", context.get("momentum", "")) or "").lower()
        wickiness = float(context.get("entry_wickiness", 0.0) or 0.0)
        expansion = float(context.get("entry_expansion", 0.0) or 0.0)
        position = float(context.get("entry_position", 0.5) or 0.5)
        trend_strength = float(context.get("entry_trend_strength", context.get("trend_strength", 0.0)) or 0.0)

        def setup_fit(family: str) -> dict[str, Any]:
            checks: list[dict[str, Any]] = []

            def add(code: str, passed: bool, detail: str) -> None:
                checks.append({"code": code, "status": "present" if passed else "missing", "detail": detail})

            if family == "pullback":
                add("valid_pullback_location", location == "pullback", f"location={location or 'unknown'}")
                add("momentum_turn_or_strong_momentum", momentum == "strong", f"momentum={momentum or 'unknown'}")
                add("not_mid_range", location != "mid_range", f"location={location or 'unknown'}")
            elif family == "breakout":
                add("breakout_follow_through_or_retest", location in {"breakout", "pullback"}, f"location={location or 'unknown'}")
                add("no_overextension", expansion < 0.45 and 0.10 < position < 0.90, f"expansion={expansion:.2f}, position={position:.2f}")
                add("execution_friction_acceptable", wickiness < 0.60, f"wickiness={wickiness:.2f}")
            elif family == "continuation":
                add("structure_hold", trend_strength >= 0.18, f"trend_strength={trend_strength:.2f}")
                add("strong_m1_momentum", momentum == "strong", f"momentum={momentum or 'unknown'}")
            elif family == "reversal":
                add("rejection_after_sweep", wickiness < 0.50, f"wickiness={wickiness:.2f}")
                add("structure_reclaim", trend_strength >= 0.12, f"trend_strength={trend_strength:.2f}")
                add("momentum_confirmation", momentum in {"strong", "mixed"}, f"momentum={momentum or 'unknown'}")
            else:
                add("clean_location", location not in {"", "unknown", "mid_range"}, f"location={location or 'unknown'}")
                add("momentum_confirmation", momentum == "strong", f"momentum={momentum or 'unknown'}")
                add("execution_friction_acceptable", wickiness < 0.60, f"wickiness={wickiness:.2f}")
            missing = [row["code"] for row in checks if row["status"] != "present"]
            return {"confirmed": not missing, "missing": missing, "checks": checks}

        candidates: list[dict[str, Any]] = []
        for lesson in lessons.values():
            row = dict(lesson or {})
            strategy_relevant = str(row.get("strategy", "") or "") == strategy_key
            setup_relevant = str(row.get("setup_family", "") or "") == setup_family
            if not strategy_relevant and not setup_relevant:
                continue
            fit = setup_fit(str(row.get("setup_family", "") or "unknown"))
            candidates.append(
                {
                    "key": str(row.get("key", "") or ""),
                    "strategy": str(row.get("strategy", "") or ""),
                    "setup_family": str(row.get("setup_family", "") or ""),
                    "samples": int(row.get("samples", 0) or 0),
                    "strategy_relevant": bool(strategy_relevant),
                    "setup_relevant": bool(setup_relevant),
                    "confirmed": bool(fit.get("confirmed", False)),
                    "missing": list(fit.get("missing", []) or []),
                    "checks": list(fit.get("checks", []) or [])[:6],
                    "winning_signature": str(row.get("winning_signature", "") or ""),
                    "confirmation_required": list(row.get("confirmation_required", []) or [])[:6],
                }
            )
        candidates.sort(
            key=lambda row: (
                bool(row.get("strategy_relevant", False)),
                bool(row.get("setup_relevant", False)),
                int(row.get("samples", 0) or 0),
            ),
            reverse=True,
        )
        top = candidates[0] if candidates else {}
        if top and bool(top.get("confirmed", False)):
            status = "winner_pattern_confirmed"
            next_action = "winner evidence is aligned; normal gates still decide execution"
        elif top:
            status = "winner_pattern_not_confirmed"
            next_action = "wait for the missing winner confirmations"
        else:
            status = "no_relevant_winner_pattern"
            next_action = "no matching winner lesson for current strategy/setup"
        return {
            "enabled": True,
            "schema": str(playbook.get("schema", "atlas_winner_playbook_v1") or ""),
            "status": status,
            "source": str(playbook.get("source", "") or ""),
            "lessons_available": len(lessons),
            "top_lesson": top,
            "candidates": candidates[:4],
            "next_action": next_action,
            "policy_note": str(playbook.get("policy_note", "") or "Winner playbook is bounded teaching evidence only."),
        }

    def _apply_entry_timing_learning(self, signal: TradeSignal, context: dict[str, Any]) -> TradeSignal:
        if str(self.settings.timeframe).upper() != "M1":
            return signal
        side = str(getattr(signal, "side", "flat") or "flat").lower()
        if side not in {"buy", "sell"}:
            return signal
        strategy_key = self._strategy_key(str(getattr(signal, "strategy", "")))
        counts = self._historical_loss_pattern_counts()
        if not counts:
            return signal

        location = str(context.get("entry_location_state", "") or "").lower()
        momentum = str(context.get("entry_momentum_state", "") or "").lower()
        wickiness = float(context.get("entry_wickiness", 0.0) or 0.0)
        expansion = float(context.get("entry_expansion", 0.0) or 0.0)
        position = float(context.get("entry_position", 0.5) or 0.5)
        multiplier = 1.0

        if counts.get("weak_pullback", 0) >= 50 and strategy_key in {"ema_crossover", "pullback_trend", "evolved_rule"}:
            if location == "pullback" and momentum != "strong":
                multiplier *= 0.82
            elif location == "mid_range":
                multiplier *= 0.90
        if counts.get("stop_hunt", 0) >= 40 and wickiness >= 0.55:
            multiplier *= 0.86
        if counts.get("late_entry", 0) >= 40 and (location == "breakout" or expansion >= 0.45 or position <= 0.08 or position >= 0.92):
            multiplier *= 0.88
        if counts.get("false_continuation", 0) >= 3 and strategy_key in {"momentum_continuation", "evolved_rule"}:
            if momentum != "strong":
                multiplier *= 0.90

        if multiplier < 1.0:
            signal.confidence = max(0.0, float(signal.confidence or 0.0) * max(0.60, multiplier))
        lesson_conf, lesson_meta = self._loss_pattern_lesson_adjustment(signal, context)
        if bool(lesson_meta.get("applied", False)):
            signal.confidence = float(lesson_conf)
        return signal

    def _suppress_signal(self, signal: TradeSignal) -> TradeSignal:
        signal.side = "flat"
        signal.confidence = 0.0
        return signal

    def _apply_signal_generation_gating(self, signal: TradeSignal, context: dict[str, Any]) -> TradeSignal:
        side = str(getattr(signal, "side", "flat") or "flat").lower()
        if side not in {"buy", "sell"}:
            return self._sanitize_signal(signal)

        regime = str(context.get("regime", "unknown") or "unknown")
        volatility_state = str(context.get("volatility_state", "low") or "low")
        trend_strength = float(context.get("trend_strength", 0.0) or 0.0)
        strategy_key = self._strategy_key(str(getattr(signal, "strategy", "")))
        learning_mode = bool(getattr(self.settings, "learning_mode", False))
        low_energy = (
            trend_strength < float(getattr(self.settings, "signal_regime_gate_min_trend_strength_pct", 0.18) or 0.18)
            and volatility_state == "low"
            and regime in {"range", "unknown", "high_volatility_range"}
        )
        if low_energy:
            # LEARNING MODE: keep directional candidates alive with a strong confidence haircut
            # instead of hard-flattening every low-energy cycle.
            if learning_mode:
                signal.confidence = float(signal.confidence or 0.0) * 0.78
                return self._sanitize_signal(signal)
            return self._suppress_signal(signal)

        if str(self.settings.timeframe).upper() == "M1" and not bool(context.get("allowed_session", False)):
            if bool(getattr(self.settings, "suppress_directional_signals_outside_sessions", True)):
                return self._suppress_signal(signal)
            signal.confidence = float(signal.confidence or 0.0) * float(
                getattr(self.settings, "signal_outside_session_confidence_multiplier", 0.5) or 0.5
            )

        if strategy_key == "momentum_continuation":
            momentum_requires_suppression = (
                regime != "trend"
                or volatility_state == "low"
                or trend_strength < float(getattr(self.settings, "momentum_signal_min_trend_strength_pct", 0.18) or 0.18)
                or float(context.get("volatility_pct", 0.0) or 0.0)
                < float(getattr(self.settings, "momentum_signal_min_volatility_pct", 0.03) or 0.03)
            )
            if momentum_requires_suppression:
                # LEARNING MODE: do not completely erase momentum candidates;
                # retain them at reduced confidence for safe exploration.
                if learning_mode:
                    signal.confidence = float(signal.confidence or 0.0) * 0.88
                    return self._sanitize_signal(signal)
                return self._suppress_signal(signal)

        weak_trend = float(getattr(self.settings, "signal_trend_strength_weak_pct", 0.12) or 0.12)
        strong_trend = float(getattr(self.settings, "signal_trend_strength_strong_pct", 0.28) or 0.28)
        if trend_strength < weak_trend:
            signal.confidence = float(signal.confidence or 0.0) * float(
                getattr(self.settings, "signal_trend_strength_weak_confidence_multiplier", 0.55) or 0.55
            )
        elif regime == "trend" and volatility_state in {"normal", "high"} and trend_strength >= strong_trend:
            signal.confidence = min(
                1.0,
                float(signal.confidence or 0.0)
                * float(getattr(self.settings, "signal_trend_strength_strong_confidence_boost", 1.12) or 1.12),
            )

        signal = self._apply_entry_timing_learning(signal, context)
        return self._sanitize_signal(signal)

    def _min_conf_threshold(self, regime: dict[str, Any]) -> float:
        name = str(regime.get("name", "unknown"))
        if name == "trend":
            base = float(self.settings.min_confidence_trend)
        elif name == "high_volatility_range":
            base = float(self.settings.min_confidence_high_volatility)
        else:
            base = float(self.settings.min_confidence_range)
        if str(self.settings.timeframe).upper() == "M1":
            base += 0.02
        research_floor = float(getattr(self.settings, "research_confidence_threshold", 0.0) or 0.0)
        return max(base, research_floor)

    def _timeframe_bias(self, timeframe: str) -> str:
        try:
            htf = self.feed.get_rates_for_timeframe(timeframe, bars=220)
        except Exception:
            return "unknown"
        closes = [float(x) for x in htf.tail(80)["close"].tolist()]
        if len(closes) < 25:
            return "unknown"
        fast = sum(closes[-10:]) / 10.0
        slow = sum(closes[-30:]) / 30.0
        if fast > slow * 1.0002:
            return "buy"
        if fast < slow * 0.9998:
            return "sell"
        return "flat"

    def _higher_tf_context(self) -> dict[str, Any]:
        if str(getattr(self.settings, "timeframe", "") or "").upper() == "M1":
            configured = (str(getattr(self.settings, "m1_htf_alignment_timeframe", "M5") or "M5").upper(),)
            fallback_tf = str(getattr(self.settings, "m1_htf_alignment_timeframe", "M5") or "M5").upper()
        else:
            configured = tuple(getattr(self.settings, "higher_timeframe_context", ()) or ())
            if not configured:
                configured = (str(getattr(self.settings, "higher_timeframe_confirm", "H1") or "H1").upper(),)
            fallback_tf = str(getattr(self.settings, "higher_timeframe_confirm", "H1") or "H1").upper()
        timeframes = []
        for tf in configured:
            normalized = str(tf or "").strip().upper()
            if normalized and normalized not in timeframes:
                timeframes.append(normalized)
        details = {tf: self._timeframe_bias(tf) for tf in timeframes}
        known = [bias for bias in details.values() if bias in {"buy", "sell", "flat"}]
        directional = [bias for bias in known if bias in {"buy", "sell"}]
        if not known:
            bias = "unknown"
        elif directional and all(item == directional[0] for item in directional) and "flat" not in known:
            bias = directional[0]
        elif directional:
            bias = "mixed"
        else:
            bias = "flat"
        return {
            "confirm_tf": timeframes[0] if timeframes else fallback_tf,
            "context_tfs": timeframes,
            "bias": bias,
            "details": details,
        }

    def _higher_tf_bias(self) -> str:
        return str(self._higher_tf_context().get("bias", "unknown") or "unknown")

    def _signal_micro_context(self, signal: TradeSignal, df) -> dict[str, float]:
        recent = df.tail(12).copy()
        if len(recent) < 8:
            return {"body_ratio": 0.0, "volume_ratio": 1.0}
        last = recent.iloc[-1]
        candle_range = max(1e-9, float(last["high"]) - float(last["low"]))
        candle_body = abs(float(last["close"]) - float(last["open"]))
        body_ratio = candle_body / candle_range
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        last_vol = float(last.get(vol_col, 0.0) or 0.0)
        avg_vol = float(recent[vol_col].tail(8).mean() or 1.0)
        volume_ratio = last_vol / max(avg_vol, 1.0)
        return {"body_ratio": round(body_ratio, 4), "volume_ratio": round(volume_ratio, 4)}

    def _derive_runtime_chart_features(
        self,
        *,
        df,
        chosen: TradeSignal | None,
        regime: dict[str, Any],
    ) -> dict[str, Any]:
        # purpose: provide deterministic chart-feature context so decision analysis
        # does not fall back to perpetual "unclear/no_trade" when runtime status is otherwise healthy.
        # what it does: derives trend, structure, momentum, and location from recent OHLC bars.
        recent = df.tail(90).copy()
        if len(recent) < 25:
            return {
                "timeframe": str(self.settings.timeframe).upper(),
                "data_source": "mt5_rates_api",
                "visible_terminal_chart_required": False,
                "trend_state": "unclear",
                "trend_structure": "mixed",
                "momentum_state": "mixed",
                "location_state": "unclear",
                "setup_type_hint": "no_trade",
                "reliability": "low",
                "warnings": ["insufficient_chart_bars"],
            }

        closes = [float(x) for x in recent["close"].tolist()]
        highs = [float(x) for x in recent["high"].tolist()]
        lows = [float(x) for x in recent["low"].tolist()]
        opens = [float(x) for x in recent["open"].tolist()]
        if not closes or not highs or not lows or not opens:
            return {
                "timeframe": str(self.settings.timeframe).upper(),
                "data_source": "mt5_rates_api",
                "visible_terminal_chart_required": False,
                "trend_state": "unclear",
                "trend_structure": "mixed",
                "momentum_state": "mixed",
                "location_state": "unclear",
                "setup_type_hint": "no_trade",
                "reliability": "low",
                "warnings": ["missing_ohlc_fields"],
            }

        sma_fast = sum(closes[-8:]) / 8.0
        sma_slow = sum(closes[-21:]) / 21.0
        trend_state = "range"
        if sma_fast > sma_slow * 1.00015:
            trend_state = "uptrend"
        elif sma_fast < sma_slow * 0.99985:
            trend_state = "downtrend"

        recent_high = max(highs[-10:])
        prev_high = max(highs[-20:-10])
        recent_low = min(lows[-10:])
        prev_low = min(lows[-20:-10])
        if recent_high > prev_high and recent_low > prev_low:
            trend_structure = "HH/HL"
        elif recent_high < prev_high and recent_low < prev_low:
            trend_structure = "LH/LL"
        else:
            trend_structure = "mixed"

        returns = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent_returns = returns[-8:] if len(returns) >= 8 else returns
        pos = sum(1 for r in recent_returns if r > 0)
        neg = sum(1 for r in recent_returns if r < 0)
        avg_abs = sum(abs(r) for r in recent_returns) / max(1, len(recent_returns))
        last_range = max(1e-9, highs[-1] - lows[-1])
        last_body = abs(closes[-1] - opens[-1])
        body_ratio = last_body / last_range
        if (pos >= 6 or neg >= 6) and body_ratio >= 0.55 and avg_abs > 0:
            momentum_state = "strong"
        elif body_ratio < 0.30:
            momentum_state = "weak"
        else:
            momentum_state = "mixed"

        range_high = max(highs[-20:])
        range_low = min(lows[-20:])
        range_span = max(1e-9, range_high - range_low)
        position = (closes[-1] - range_low) / range_span
        if trend_state == "uptrend":
            if position <= 0.42:
                location_state = "pullback"
            elif position >= 0.86 and momentum_state == "strong":
                location_state = "breakout"
            else:
                location_state = "mid_range"
        elif trend_state == "downtrend":
            if position >= 0.58:
                location_state = "pullback"
            elif position <= 0.14 and momentum_state == "strong":
                location_state = "breakout"
            else:
                location_state = "mid_range"
        elif trend_state == "range":
            # purpose: detect range-edge behavior so Atlas can classify
            # mean-reversion candidates instead of treating all range states
            # as low-quality mid-range noise.
            if position <= 0.22 or position >= 0.78:
                location_state = "pullback"
            elif 0.35 <= position <= 0.65:
                location_state = "consolidation"
            else:
                location_state = "mid_range"
        else:
            location_state = "unclear"

        setup_hint = self._strategy_to_setup_family(str(getattr(chosen, "strategy", "") if chosen is not None else ""))
        if setup_hint not in {"pullback", "breakout", "consolidation", "mean_reversion"}:
            if trend_state in {"uptrend", "downtrend"} and location_state in {"pullback", "breakout"}:
                setup_hint = location_state
            elif trend_state == "range":
                setup_hint = "mean_reversion" if location_state == "pullback" else "consolidation"
            else:
                setup_hint = "no_trade"

        wickiness_values: list[float] = []
        for idx in range(max(0, len(closes) - 10), len(closes)):
            candle_range = max(1e-9, highs[idx] - lows[idx])
            candle_body = abs(closes[idx] - opens[idx])
            wickiness_values.append(max(0.0, min(1.0, (candle_range - candle_body) / candle_range)))
        wickiness = sum(wickiness_values) / max(1, len(wickiness_values))
        recent_range = sum(highs[-5:]) / 5.0 - sum(lows[-5:]) / 5.0
        baseline_range = (sum(highs[-25:-5]) / 20.0) - (sum(lows[-25:-5]) / 20.0)
        expansion_ratio = recent_range / max(1e-9, baseline_range) if baseline_range > 0 else 1.0
        atr_proxy = float(regime.get("volatility_pct", 0.0) or 0.0)
        compression_score = max(0.0, min(1.0, 1.0 - min(expansion_ratio, 2.0) / 2.0))
        chop_score = max(0.0, min(1.0, wickiness))

        return {
            "timeframe": str(self.settings.timeframe).upper(),
            "data_source": "mt5_rates_api",
            "visible_terminal_chart_required": False,
            "trend_state": trend_state,
            "trend_structure": trend_structure,
            "momentum_state": momentum_state,
            "location_state": location_state,
            "setup_type_hint": setup_hint,
            "reliability": "medium",
            "warnings": [],
            "candle_expansion": round(max(0.0, expansion_ratio - 1.0), 4),
            "compression_score": round(compression_score, 4),
            "wickiness": round(wickiness, 4),
            "chop_score": round(chop_score, 4),
            "atr_ratio": round(max(0.0, atr_proxy), 4),
        }

    def _attach_visual_chart_read(self, status: dict[str, Any]) -> None:
        # Advisory-only visual read. This adds inspectable evidence without
        # changing strategy, risk, gates, or execution policy.
        try:
            image_path: str | None = None
            capture: dict[str, Any] = {
                "ok": False,
                "status": "not_attempted",
                "reason": "capture_throttled_or_disabled",
            }
            visual_dir = Path(str(self.settings.storage_dir)) / "chart_vision"
            capture_enabled = str(os.getenv("ATLAS_VISUAL_CHART_CAPTURE_ENABLED", "1")).strip().lower() not in {"0", "false", "no", "off"}
            cooldown_seconds = max(1.0, float(os.getenv("ATLAS_VISUAL_CHART_CAPTURE_SECONDS", "60") or "60"))
            now = time.time()
            latest = load_latest_visual_read(visual_dir, max_age_seconds=int(os.getenv("ATLAS_VISUAL_WATCH_MAX_AGE_SECONDS", "10") or "10"))
            latest_image = dict(latest.get("image", {}) or {})
            if (
                capture_enabled
                and str(latest.get("source", "") or "") == "screenshot_plus_chart_features"
                and bool(latest.get("live_chart_window_confirmed", False))
                and not bool(latest.get("stale", False))
                and latest_image.get("path")
            ):
                visual = dict(latest)
                visual["visual_watch_used_for_decision"] = True
                status["visual_chart_read"] = visual
                return
            if capture_enabled and now - float(self._last_visual_chart_capture_at or 0.0) >= cooldown_seconds:
                self._last_visual_chart_capture_at = now
                capture = capture_desktop_screenshot(visual_dir, target_window="mt5", fallback_full_desktop=False)
                image_path = str(capture.get("path", "") or "") if bool(capture.get("ok", False)) else None
            elif capture_enabled:
                if bool(latest.get("live_chart_window_confirmed", False)) and not bool(latest.get("stale", False)) and latest_image.get("path"):
                    image_path = str(latest_image.get("path") or "")
                    capture = dict(latest.get("capture", {}) or {})
                    capture["status"] = "reused_recent_live_terminal_capture"
                else:
                    capture = {
                        "ok": False,
                        "status": "waiting_for_live_terminal_capture",
                        "reason": "no_fresh_live_terminal_screenshot",
                        "latest_status": str(latest.get("status", "") or ""),
                    }
            if not image_path:
                status["visual_chart_read"] = {
                    "enabled": True,
                    "advisory_only": True,
                    "available": False,
                    "source": "live_terminal_required",
                    "status": "waiting_for_live_terminal_capture",
                    "symbol": str(status.get("symbol", getattr(self.settings, "symbol", "XAUUSD")) or "XAUUSD"),
                    "timeframe": str(status.get("timeframe", getattr(self.settings, "timeframe", "M1")) or "M1"),
                    "capture": capture,
                    "real_screenshot_attempted": bool(capture.get("status") != "not_attempted"),
                    "live_chart_window_confirmed": False,
                    "visual_confirmation": "unavailable",
                    "confidence_effect": "no_visual_evidence",
                    "reasons": ["live MT5/HFM terminal screenshot is required for visual chart evidence"],
                    "policy_note": "Visual chart reading must use the live terminal screenshot; candle-derived chart features remain separate and cannot be presented as visual evidence.",
                }
                with contextlib.suppress(Exception):
                    save_visual_read(status["visual_chart_read"], visual_dir)
                return
            visual = analyze_chart_visual_context(
                status=status,
                chart_features=dict(status.get("chart_features", {}) or {}),
                image_path=image_path,
            )
            visual["capture"] = capture
            visual["real_screenshot_attempted"] = bool(capture.get("status") != "not_attempted")
            visual["live_chart_window_confirmed"] = bool(capture.get("ok", False)) and str(capture.get("capture_source", "")) == "mt5_window"
            status["visual_chart_read"] = visual
            with contextlib.suppress(Exception):
                save_visual_read(visual, visual_dir)
        except Exception as exc:
            status["visual_chart_read"] = {
                "enabled": True,
                "advisory_only": True,
                "available": False,
                "status": "visual_chart_read_failed",
                "error": str(exc),
                "policy_note": "Visual chart reading is advisory evidence only and cannot affect execution authority.",
            }

    def _build_signal_context(
        self,
        signal: TradeSignal,
        df,
        snapshot: RiskSnapshot,
        regime: dict[str, Any],
        higher_tf_bias: str,
        drift: dict[str, Any],
        same_side_support: int,
        total_signals: int,
        when_utc: datetime,
    ) -> dict[str, Any]:
        micro = self._signal_micro_context(signal, df)
        overlay_context = build_research_regime_overlay_context(
            df=df,
            when_utc=when_utc,
            spread_points=float(getattr(snapshot, "spread_points", 0.0) or 0.0),
        )
        side = str(getattr(signal, "side", "flat"))
        return {
            "strategy": str(getattr(signal, "strategy", "")),
            "side": side,
            "session": self._session_label(when_utc),
            "hour_utc": int(when_utc.hour),
            "regime": str(regime.get("name", "unknown")),
            "regime_label": str(regime.get("regime_label", "ranging") or "ranging"),
            "regime_label_detail": dict(regime.get("regime_label_detail", {}) or {}),
            "higher_tf_bias": str(higher_tf_bias),
            "spread_points": round(float(getattr(snapshot, "spread_points", 0.0) or 0.0), 2),
            "confidence": round(float(getattr(signal, "confidence", 0.0) or 0.0), 4),
            "same_side_support": int(same_side_support),
            "same_side_ratio": round(float(same_side_support) / max(1, int(total_signals)), 4),
            "m1_quality_factor": 1.0,
            "timeframe": self.settings.timeframe,
            "volatility_pct": float(regime.get("volatility_pct", 0.0) or 0.0),
            "trend_strength": float(regime.get("trend_strength", 0.0) or 0.0),
            "body_ratio": float(micro.get("body_ratio", 0.0) or 0.0),
            "volume_ratio": float(micro.get("volume_ratio", 1.0) or 1.0),
            "spike_ratio": float(overlay_context.get("spike_ratio", 0.0) or 0.0),
            "drift_delta": float(drift.get("delta", 0.0) or 0.0),
            "drift_active": bool(drift.get("active", False)),
            "htf_aligned": bool(side in {"buy", "sell"} and side == higher_tf_bias),
            "meta_probability": 0.5,
        }

    def _contextual_signal_scores(
        self,
        signals: list[TradeSignal],
        df,
        snapshot: RiskSnapshot,
        regime: dict[str, Any],
        higher_tf_bias: str,
        drift: dict[str, Any],
        memory: dict[str, Any],
        now_utc: datetime,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
        scores: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        total_signals = max(1, len(signals))
        for signal in signals:
            key = self._strategy_key(str(getattr(signal, "strategy", "")))
            base_score = float(self.allocator.score_signal(signal))
            support = self._same_side_support_count(signals, str(getattr(signal, "side", "flat")))
            context = self._build_signal_context(
                signal=signal,
                df=df,
                snapshot=snapshot,
                regime=regime,
                higher_tf_bias=higher_tf_bias,
                drift=drift,
                same_side_support=support,
                total_signals=total_signals,
                when_utc=now_utc,
            )
            factor = 1.0
            reasons: list[str] = []
            side = str(getattr(signal, "side", "flat"))
            if side in {"buy", "sell"}:
                if support >= max(2, int(self.settings.min_same_side_signals)):
                    factor *= 1.05
                    reasons.append("consensus_support")
                elif support <= 1:
                    factor *= 0.88
                    reasons.append("weak_consensus")
                if bool(context["htf_aligned"]):
                    factor *= 1.06
                    reasons.append("htf_aligned")
                elif higher_tf_bias in {"buy", "sell"}:
                    factor *= 0.9
                    reasons.append("htf_opposed")
                if str(regime.get("name", "")) == "trend" and key in {"macd_trend", "momentum_continuation", "pullback_trend", "ema_crossover", "gold_micro_impulse", "gold_breakout_retest"}:
                    factor *= 1.05
                    reasons.append("trend_strategy_fit")
                if str(regime.get("name", "")) == "range" and key in {"rsi_mean_reversion", "bollinger_mean_reversion", "stochastic_reversal", "gold_liquidity_sweep", "gold_micro_structure_reversion"}:
                    factor *= 1.05
                    reasons.append("range_strategy_fit")
                if float(context["spread_points"]) >= min(float(self.settings.max_spread_points) * 0.5, 40.0):
                    factor *= 0.9
                    reasons.append("spread_pressure")
                if ("htf_opposed" in reasons) and ("spread_pressure" in reasons):
                    factor *= max(0.55, min(0.95, float(getattr(self.settings, "htf_spread_combo_penalty", 0.78))))
                    reasons.append("htf_opposed_spread_combo")
                if bool(drift.get("active", False)) and float(context["volume_ratio"]) < 0.85:
                    factor *= 0.92
                    reasons.append("drift_low_volume")
            p, status = self._meta_model_probability(context)
            if p is not None and self._meta_model_effect_scale() > 0.0:
                context["meta_probability"] = round(float(p), 4)
                effect_scale = self._meta_model_effect_scale()
                meta_delta = (float(p) - 0.5) * 0.4 * effect_scale
                factor *= max(0.94, min(1.06, 1.0 + meta_delta))
                reasons.append("meta_model")
            elif p is not None:
                reasons.append("meta_model_candidate_disabled")
            elif status:
                reasons.append(status)
            setup_quality = score_setup_quality(
                settings=self.settings,
                context=context,
                recent_outcomes=list(dict(memory or {}).get("recent_outcomes", [])),
            )
            if bool(setup_quality.get("enabled", False)):
                quality_factor = float(setup_quality.get("factor", 1.0) or 1.0)
                factor *= quality_factor
                if abs(quality_factor - 1.0) > 1e-6:
                    reasons.append("setup_quality")
            final_score = base_score * factor
            if side == "flat":
                final_score = 0.0
                reasons.append("flat_signal")
            scores[key] = final_score
            details[key] = {
                "strategy": str(getattr(signal, "strategy", "")),
                "base_score": round(base_score, 4),
                "factor": round(factor, 4),
                "final_score": round(final_score, 4),
                "support": int(support),
                "context": context,
                "setup_quality": setup_quality,
                "reasons": reasons[:8],
            }
        return scores, details

    def _confidence_multiplier(self, chosen_side: str, snapshot: RiskSnapshot, guard_state: dict[str, Any], regime: dict[str, Any]) -> tuple[float, str]:
        mult = 1.0
        reasons: list[str] = []

        losses = int(guard_state.get("consecutive_losses", 0))
        if losses >= 1:
            loss_mult = max(0.55, 1.0 - (0.12 * losses))
            mult *= loss_mult
            reasons.append(f"loss_streak={losses}")

        regime_name = str(regime.get("name", "unknown"))
        if regime_name == "high_volatility_range":
            mult *= 0.85
            reasons.append("high_volatility")
        if chosen_side == "flat":
            mult *= 0.0

        spread = float(getattr(snapshot, "spread_points", 0.0) or 0.0)
        if spread > 0 and spread >= float(self.settings.max_spread_points) * 0.7:
            mult *= 0.85
            reasons.append("wide_spread")

        mult = max(0.0, min(1.1, mult))
        return mult, ", ".join(reasons)

    def _volume_multiplier(self, guard_state: dict[str, Any], regime: dict[str, Any]) -> tuple[float, str]:
        mult = 1.0
        reasons: list[str] = []
        losses = int(guard_state.get("consecutive_losses", 0))
        if losses >= 1:
            mult *= max(0.5, 1.0 - 0.15 * losses)
            reasons.append(f"loss_streak={losses}")
        if str(regime.get("name", "")) == "high_volatility_range":
            mult *= 0.8
            reasons.append("high_volatility")
        if str(self.settings.timeframe).upper() == "M1":
            mult *= 0.92
            reasons.append("m1_size_trim")
        if str(guard_state.get("volatility_state", "") or "") == "elevated":
            mult *= 0.78
            reasons.append("elevated_volatility_trim")
        if bool(guard_state.get("protect_gains_mode", False)):
            mult *= 0.8
            reasons.append("protect_gains_mode")
        return max(0.3, min(1.0, mult)), ", ".join(reasons)

    def _account_tier_profile(self, balance: float) -> dict[str, Any]:
        b = float(balance or 0.0)
        if b <= 0 or b < 2500:
            return {
                "account_tier": "micro_account",
                "tier_reason": "small balance requires strict survival-first controls",
                "risk_profile": "capital_preservation_strict",
            }
        if b < 25000:
            return {
                "account_tier": "small_account",
                "tier_reason": "balance still sensitive to drawdown and execution friction",
                "risk_profile": "capital_preservation_moderate",
            }
        return {
            "account_tier": "standard_account",
            "tier_reason": "balance supports standard guarded risk profile",
            "risk_profile": "capital_preservation_standard",
        }

    def _size_tier_profile(self, balance: float) -> dict[str, Any]:
        tiers: list[tuple[str, float, float | None]] = [
            ("tier_0_0p01", 0.01, 750.0),
            ("tier_1_0p02", 0.02, 2500.0),
            ("tier_2_0p03", 0.03, 7500.0),
            ("tier_3_0p05", 0.05, 20000.0),
            ("tier_4_0p08", 0.08, 50000.0),
            ("tier_5_0p10", 0.10, None),
        ]
        current = tiers[0]
        for item in tiers:
            _name, _cap, req = item
            if req is None or float(balance or 0.0) >= req:
                current = item
            else:
                break
        idx = tiers.index(current)
        next_tier = tiers[min(idx + 1, len(tiers) - 1)]
        next_req = next_tier[2]
        return {
            "current_size_tier": current[0],
            "max_lot": float(current[1]),
            "next_size_tier_requirement": f"balance>={next_req:.2f}" if next_req is not None else "max_tier_reached",
            "size_tier_reason": "bounded_stepwise_scaling",
        }

    def _size_band_multiplier(self, structured_analysis: dict[str, Any], guard_state: dict[str, Any], regime: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        trading_response = dict(structured_analysis.get("trading_response", {}) or {})
        size_band = dict(trading_response.get("size_band_rationale", {}) or {})
        multiplier = float(size_band.get("cap_multiplier", 1.0) or 1.0)
        rationale = list(size_band.get("rationale", []) or [])

        losses = int(guard_state.get("consecutive_losses", 0) or 0)
        if losses >= 2:
            multiplier = min(multiplier, 0.55)
            rationale.append("Recent loss streak keeps size in a defensive band.")
        if str(regime.get("name", "") or "") == "high_volatility_range":
            multiplier = min(multiplier, 0.6)
            rationale.append("High-volatility range regime trims size further.")

        band = str(size_band.get("band", "guarded") or "guarded")
        if multiplier <= 0.0:
            band = "blocked"
        elif multiplier <= 0.62 and band not in {"blocked", "probe"}:
            band = "probe"

        return max(0.0, min(1.0, multiplier)), {
            "band": band,
            "cap_multiplier": round(max(0.0, min(1.0, multiplier)), 3),
            "risk_fraction_of_max": round(float(size_band.get("risk_fraction_of_max", multiplier) or multiplier), 3),
            "rationale": rationale[:6],
        }

    def _m1_quality_adjustment(
        self,
        chosen: TradeSignal,
        df,
        snapshot: RiskSnapshot,
        regime: dict[str, Any],
        higher_tf_bias: str,
        now_utc: datetime,
    ) -> tuple[float, dict[str, Any]]:
        if str(self.settings.timeframe).upper() != "M1":
            return float(chosen.confidence), {"enabled": False}
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}
        recent = df.tail(12).copy()
        if len(recent) < 8:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "insufficient_bars"}
        last = recent.iloc[-1]
        candle_range = max(1e-9, float(last["high"]) - float(last["low"]))
        candle_body = abs(float(last["close"]) - float(last["open"]))
        body_ratio = candle_body / candle_range
        vol_col = "tick_volume" if "tick_volume" in recent.columns else "real_volume"
        last_vol = float(last.get(vol_col, 0.0) or 0.0)
        avg_vol = float(recent[vol_col].tail(8).mean() or 1.0)
        spread = float(getattr(snapshot, "spread_points", 0.0) or 0.0)
        factor = 1.0
        reasons: list[str] = []

        if spread >= min(float(self.settings.max_spread_points) * 0.45, 35.0):
            factor *= 0.88
            reasons.append("m1_spread_heavy")
        if last_vol < avg_vol * 0.72:
            factor *= 0.9
            reasons.append("m1_thin_volume")
        if body_ratio < 0.32:
            factor *= 0.9
            reasons.append("m1_weak_body")
        if self._session_label(now_utc) == "asia":
            factor *= 0.95
            reasons.append("asia_session_noise")
        if higher_tf_bias in {"buy", "sell"} and higher_tf_bias == side and str(regime.get("name", "")) == "trend":
            factor *= 1.05
            reasons.append("htf_trend_alignment")
        if higher_tf_bias in {"buy", "sell"} and higher_tf_bias != side and str(regime.get("name", "")) != "range":
            factor *= 0.93
            reasons.append("counter_htf_pressure")

        new_conf = max(0.0, min(1.0, float(chosen.confidence) * factor))
        return new_conf, {
            "enabled": True,
            "applied": abs(factor - 1.0) > 1e-6,
            "factor": round(factor, 4),
            "spread_points": round(spread, 2),
            "body_ratio": round(body_ratio, 4),
            "last_volume": round(last_vol, 2),
            "avg_volume_8": round(avg_vol, 2),
            "session": self._session_label(now_utc),
            "reasons": reasons,
        }

    def _m1_entry_block_reason(self, chosen: TradeSignal, m1_meta: dict[str, Any]) -> str:
        if str(self.settings.timeframe).upper() != "M1":
            return ""
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return ""
        session = str(m1_meta.get("session", "") or "").lower()
        allowed_sessions = {str(x).strip().lower() for x in getattr(self.settings, "m1_allowed_sessions", ()) if str(x).strip()}
        if allowed_sessions and session and session not in allowed_sessions:
            return f"M1 session guard blocked trade during {session} session."
        spread = float(m1_meta.get("spread_points", 0.0) or 0.0)
        # FIX: demo-aware spread gate to avoid over-blocking HFM demo learning flow.
        spread_pips = spread / 10.0
        thresholds = get_active_thresholds(self.settings)
        # LEARNING MODE: threshold from get_active_thresholds()
        spread_max = float(thresholds["spread_max"])
        spread_ok = spread_pips <= spread_max
        logger.info("Spread check: %.2f pips (max: %.2f) -- %s", spread_pips, spread_max, "PASS" if spread_ok else "BLOCK")
        if not spread_ok:
            return f"spread_liquidity_too_poor ({spread_pips:.2f}>{spread_max:.2f} pips)"
        body_ratio = float(m1_meta.get("body_ratio", 0.0) or 0.0)
        avg_vol = float(m1_meta.get("avg_volume_8", 0.0) or 0.0)
        last_vol = float(m1_meta.get("last_volume", 0.0) or 0.0)
        volume_ratio = (last_vol / max(avg_vol, 1e-9)) if avg_vol > 0 else 1.0
        min_body = float(getattr(self.settings, "m1_hard_block_min_body_ratio", 0.28))
        min_vol_ratio = float(getattr(self.settings, "m1_hard_block_min_volume_ratio", 0.72))
        # FIX: demo-aware structure threshold lowers strictness for early learning trades.
        structure_score = float(body_ratio * volume_ratio)
        # LEARNING MODE: threshold from get_active_thresholds()
        structure_min = float(thresholds["structure"])
        structure_ok = structure_score >= structure_min
        logger.info("Structure check: score=%.3f (min: %.3f) -- %s", structure_score, structure_min, "PASS" if structure_ok else "BLOCK")
        if not structure_ok:
            return f"structure_too_weak (score={structure_score:.3f} min={structure_min:.3f})"
        if body_ratio < min_body and volume_ratio < min_vol_ratio:
            return (
                "M1 quality hard-block: weak candle body and thin volume "
                f"(body={body_ratio:.2f}, volume_ratio={volume_ratio:.2f})."
            )
        return ""

    @staticmethod
    def _directional_structure_block_reason(
        *,
        side: str,
        chart_features: dict[str, Any],
        context_snapshot: dict[str, Any],
        reversal_detection: dict[str, Any] | None = None,
    ) -> str:
        side_norm = str(side or "flat").lower()
        if side_norm not in {"buy", "sell"}:
            return ""
        chart = dict(chart_features or {})
        context = dict(context_snapshot or {})
        trend_structure = str(context.get("trend_structure", chart.get("trend_structure", "")) or "").upper()
        trend_state = str(context.get("trend_state", chart.get("trend_state", "")) or "").lower()
        location = str(context.get("location_state", chart.get("location_state", "")) or "").lower()
        momentum = str(context.get("momentum_state", chart.get("momentum_state", "")) or "").lower()
        structure_side = ""
        if trend_structure == "HH/HL" or trend_state == "uptrend":
            structure_side = "buy"
        elif trend_structure == "LH/LL" or trend_state == "downtrend":
            structure_side = "sell"
        if not structure_side or side_norm == structure_side:
            return ""
        reversal = dict(reversal_detection or {})
        confirmed_reversal = bool(reversal.get("confirmed_reversal", False))
        reversal_bias = str(reversal.get("reversal_bias", "neutral") or "neutral").lower()
        structure_break = bool(
            context.get("structure_break_confirmed", chart.get("structure_break_confirmed", False))
            or context.get("reversal_structure_break", chart.get("reversal_structure_break", False))
        )
        momentum_confirmed = str(
            context.get("reversal_momentum_confirmation", chart.get("reversal_momentum_confirmation", ""))
            or ""
        ).lower() in {"yes", "true", "confirmed", "strong"}
        reversal_ok = confirmed_reversal and reversal_bias == side_norm
        if reversal_ok or (structure_break and momentum_confirmed and location != "mid_range"):
            return ""
        return (
            "Directional-structure guard blocked trade: "
            f"chosen={side_norm}, structure={trend_structure or trend_state or 'unknown'}, "
            f"location={location or 'unknown'}, momentum={momentum or 'unknown'}."
        )

    # FIX: expose numeric blocker reasons for dashboard/operator transparency.
    def _build_block_reason_detail(self, checks: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if not bool(checks.get("spread_ok", True)):
            reasons.append(
                f"Spread: {float(checks.get('spread_actual', 0.0)):.2f} pips "
                f"(max: {float(checks.get('spread_max', 0.0)):.2f}) -- BLOCKED"
            )
        if not bool(checks.get("structure_ok", True)):
            reasons.append(
                f"Structure score: {float(checks.get('structure_score', 0.0)):.3f} "
                f"(min: {float(checks.get('structure_min', 0.0)):.3f}) -- BLOCKED"
            )
        if not bool(checks.get("confidence_ok", True)):
            reasons.append(
                f"Signal confidence: {float(checks.get('confidence', 0.0)):.3f} "
                f"(min: {float(checks.get('confidence_threshold', 0.0)):.3f}) -- BLOCKED"
            )
        return reasons

    def _research_regime_overlay_meta(self, df, snapshot: RiskSnapshot, now_utc: datetime) -> dict[str, Any]:
        context = build_research_regime_overlay_context(
            df=df,
            when_utc=now_utc,
            spread_points=float(getattr(snapshot, "spread_points", 0.0) or 0.0),
        )
        return {
            "enabled": bool(getattr(self.settings, "research_regime_filter_enabled", False)),
            "exclude_low_liquidity": bool(getattr(self.settings, "research_regime_filter_low_liquidity", True)),
            "exclude_abnormal_spike": bool(getattr(self.settings, "research_regime_filter_abnormal_spike", True)),
            "exclude_post_spike": bool(getattr(self.settings, "research_regime_filter_post_spike", False)),
            "exclude_london_extreme_volatility": bool(getattr(self.settings, "research_regime_filter_london_extreme_volatility", False)),
            "exclude_weak_trend": bool(getattr(self.settings, "research_regime_filter_weak_trend", False)),
            "exclude_trend_low_volatility": bool(getattr(self.settings, "research_regime_filter_trend_low_volatility", False)),
            "exclude_trend_late_us": bool(getattr(self.settings, "research_regime_filter_trend_late_us", False)),
            **context,
        }

    def _maybe_emit_drift_alert(self, memory: dict[str, Any], flags: dict[str, Any] | None = None) -> None:
        drift = self._drift_state(memory)
        if not drift.get("enough_samples", False):
            return
        if not drift.get("active", False):
            return
        # Debounce drift alerts to reduce event spam during persistent bad regimes.
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        now = datetime.now(timezone.utc)
        raw_last = str(flags.get("drift_alert_last_at", "") or "")
        last_delta = float(flags.get("drift_alert_last_delta", 0.0) or 0.0)
        current_delta = float(drift.get("delta", 0.0) or 0.0)
        if raw_last:
            try:
                last_at = datetime.fromisoformat(raw_last)
                seconds_since = (now - last_at).total_seconds()
                materially_worse = current_delta < (last_delta - 0.05)
                if seconds_since < 10 * 60 and not materially_worse:
                    return
            except Exception:
                pass
        by_session = drift.get("recent_by_session", {})
        self.store.log_event(
            "risk",
            "Performance drift detected: recent win rate materially below baseline.",
            {
                "recent_win_rate_20": round(float(drift.get("recent_win_rate_20", 0.0)), 3),
                "baseline_win_rate": round(float(drift.get("baseline_win_rate", 0.0)), 3),
                "delta": round(float(drift.get("delta", 0.0)), 3),
                "recent_by_session": by_session,
            },
        )
        flags["drift_alert_last_at"] = now.isoformat()
        flags["drift_alert_last_delta"] = round(current_delta, 4)
        self.store.save_runtime_flags(flags)

    def _drift_state(self, memory: dict[str, Any]) -> dict[str, Any]:
        outcomes = list(memory.get("recent_outcomes", []))
        if len(outcomes) < 20:
            return {"enough_samples": False, "active": False, "severe": False, "delta": 0.0}
        recent = outcomes[-20:]
        recent_wins = sum(1 for x in recent if str(x.get("outcome", "")) == "win")
        recent_rate = recent_wins / max(1, len(recent))

        all_stats = dict(memory.get("side_hour_stats", {}))
        total_w = sum(int(v.get("wins", 0)) for v in all_stats.values())
        total_l = sum(int(v.get("losses", 0)) for v in all_stats.values())
        baseline_n = total_w + total_l
        if baseline_n < 30:
            return {"enough_samples": False, "active": False, "severe": False, "delta": 0.0}
        baseline_rate = total_w / max(1, baseline_n)
        delta = recent_rate - baseline_rate
        by_session: dict[str, dict[str, int]] = {}
        for x in recent:
            ses = str(x.get("session", "unknown"))
            out = str(x.get("outcome", "draw"))
            agg = by_session.setdefault(ses, {"win": 0, "loss": 0, "draw": 0})
            if out in agg:
                agg[out] += 1
        return {
            "enough_samples": True,
            "active": bool(delta < -0.15),
            "severe": bool(delta < -0.25),
            "recent_win_rate_20": recent_rate,
            "baseline_win_rate": baseline_rate,
            "delta": delta,
            "recent_by_session": by_session,
        }

    def _apply_drift_signal_adjustment(self, signals: list[TradeSignal], drift: dict[str, Any]) -> list[str]:
        if not bool(drift.get("active", False)):
            return []
        reasons: list[str] = []
        severe = bool(drift.get("severe", False))
        for s in signals:
            name = str(getattr(s, "strategy", ""))
            if name.startswith("vwap_proxy_reversion"):
                before = float(getattr(s, "confidence", 0.0) or 0.0)
                mult = 0.65 if severe else 0.8
                s.confidence = max(0.0, before * mult)
                reasons.append(f"vwap_proxy_reversion x{mult:.2f}")
        return reasons

    def _apply_drift_strategy_weighting(self, signals: list[TradeSignal], drift: dict[str, Any]) -> list[str]:
        if not bool(drift.get("active", False)):
            return []
        severe = bool(drift.get("severe", False))
        reasons: list[str] = []
        stats = self.allocator.stats()
        for s in signals:
            key = self._strategy_key(str(getattr(s, "strategy", "")))
            st = stats.get(key)
            if st is None:
                continue
            wins = int(getattr(st, "wins", 0))
            losses = int(getattr(st, "losses", 0))
            n = wins + losses
            if n < 30:
                continue
            wr = wins / max(1, n)
            before = float(getattr(s, "confidence", 0.0) or 0.0)
            if wr < 0.45:
                mult = 0.78 if severe else 0.88
                s.confidence = max(0.0, before * mult)
                reasons.append(f"{key} wr={wr:.2f} x{mult:.2f}")
            elif severe and wr > 0.56:
                mult = 1.05
                s.confidence = min(1.0, before * mult)
                reasons.append(f"{key} wr={wr:.2f} x{mult:.2f}")
        return reasons

    def _drift_throttle_active(
        self, drift: dict[str, Any], now_utc: datetime, flags: dict[str, Any] | None = None
    ) -> tuple[bool, int]:
        if not bool(drift.get("active", False)):
            return False, 0
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        next_at_raw = str(flags.get("drift_throttle_next_trade_at", "") or "")
        if not next_at_raw:
            return False, 0
        try:
            next_at = datetime.fromisoformat(next_at_raw)
        except Exception:
            return False, 0
        if next_at <= now_utc:
            return False, 0
        secs = int((next_at - now_utc).total_seconds())
        return True, max(1, secs)

    def _set_drift_throttle_after_execution(
        self, drift: dict[str, Any], now_utc: datetime, flags: dict[str, Any] | None = None
    ) -> None:
        flags = dict(flags) if isinstance(flags, dict) else self.store.load_runtime_flags()
        if bool(drift.get("active", False)):
            severe_secs = max(20, int(self.settings.drift_throttle_seconds_severe))
            normal_secs = max(10, int(self.settings.drift_throttle_seconds))
            hold_seconds = severe_secs if bool(drift.get("severe", False)) else normal_secs
            flags["drift_throttle_next_trade_at"] = (now_utc + timedelta(seconds=hold_seconds)).isoformat()
        else:
            flags["drift_throttle_next_trade_at"] = ""
        self.store.save_runtime_flags(flags)

    def _apply_open_position_feedback(self, flags: dict[str, Any] | None = None) -> None:
        if not bool(self.settings.open_feedback_enabled):
            return
        positions = mt5.positions_get(symbol=self.settings.symbol) or []
        if not positions:
            return
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        seen = dict(flags.get("open_feedback_seen", {}) or {})
        threshold = float(self.settings.open_feedback_pnl_threshold)
        reward_mag = abs(float(self.settings.open_feedback_reward))
        now_key = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        changed = False
        decision_strategy_map, decision_id_aliases, _decision_meta_by_execution_ref = self._decision_sample_maps(limit=5000)
        for p in positions:
            if int(getattr(p, "magic", 0) or 0) != int(self.settings.magic_number):
                continue
            ticket = str(getattr(p, "ticket", "") or "")
            if not ticket:
                continue
            mark = f"{ticket}:{now_key}"
            if seen.get(ticket) == now_key:
                continue
            pnl = float(getattr(p, "profit", 0.0) or 0.0)
            if abs(pnl) < threshold:
                continue
            comment = str(getattr(p, "comment", "") or "")
            parsed = parse_broker_comment(comment)
            raw_decision_id = str(parsed.get("decision_id", "") or "").strip().lower()
            decision_id = decision_id_aliases.get(raw_decision_id, raw_decision_id)
            strategy = (
                str(parsed.get("strategy", "") or "").strip()
                or decision_strategy_map.get(str(decision_id or "").lower(), "")
            )
            if not strategy:
                continue
            delta = reward_mag if pnl > 0 else -reward_mag
            self.allocator.update(strategy, delta)
            self.store.log_event(
                "learning",
                "Applied open-position feedback for faster adaptation.",
                {"ticket": ticket, "strategy": strategy, "open_pnl": round(pnl, 2), "delta": round(delta, 4)},
            )
            seen[ticket] = now_key
            changed = True
        if changed:
            flags["open_feedback_seen"] = seen
            self.store.save_runtime_flags(flags)

    def _risk_snapshot(self, flags: dict[str, Any] | None = None) -> RiskSnapshot:
        account = mt5.account_info()
        positions = mt5.positions_get(symbol=self.settings.symbol) or []
        tick = mt5.symbol_info_tick(self.settings.symbol)
        symbol_info = mt5.symbol_info(self.settings.symbol)
        guard_state = self._load_guard_state(flags)
        spread_points = 0.0
        if tick is not None and symbol_info is not None and getattr(symbol_info, "point", 0):
            spread_points = abs(float(tick.ask) - float(tick.bid)) / float(symbol_info.point)
        daily_loss_pct = 0.0
        weekly_loss_pct = 0.0
        if account and account.balance > 0:
            equity = float(account.equity)
            start_balance = float(guard_state.get("start_balance", 0.0) or 0.0)
            if start_balance <= 0:
                start_balance = float(account.balance)
            daily_loss_pct = max(0.0, (start_balance - equity) / start_balance) if start_balance > 0 else 0.0
            week_start_balance = float(guard_state.get("week_start_balance", 0.0) or 0.0)
            if week_start_balance > 0:
                weekly_loss_pct = max(0.0, (week_start_balance - equity) / week_start_balance)
        return RiskSnapshot(
            equity=account.equity if account else 0,
            balance=account.balance if account else 0,
            daily_loss_pct=daily_loss_pct,
            open_positions=len(positions),
            spread_points=spread_points,
            weekly_loss_pct=weekly_loss_pct,
            consecutive_losses=int(guard_state.get("consecutive_losses", 0)),
        )

    def _load_guard_state(self, flags: dict[str, Any] | None = None) -> dict[str, Any]:
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        raw = flags.get("risk_guard_state", {}) if isinstance(flags, dict) else {}
        today = datetime.now(timezone.utc).date().isoformat()
        week_anchor = datetime.now(timezone.utc).date() - timedelta(days=datetime.now(timezone.utc).weekday())
        week_key = week_anchor.isoformat()
        state = {
            "date": str(raw.get("date", today)),
            "trades_today": int(raw.get("trades_today", 0)),
            "consecutive_losses": int(raw.get("consecutive_losses", 0)),
            "cooldown_until": str(raw.get("cooldown_until", "")),
            "start_balance": float(raw.get("start_balance", 0.0) or 0.0),
            "week_key": str(raw.get("week_key", week_key)),
            "week_start_balance": float(raw.get("week_start_balance", 0.0) or 0.0),
            "safety_lock_active": bool(raw.get("safety_lock_active", False)),
            "safety_lock_reason": str(raw.get("safety_lock_reason", "")),
            "safety_lock_started_at": str(raw.get("safety_lock_started_at", "")),
            "safety_lock_scope": str(raw.get("safety_lock_scope", "new_entries")),
            "recovery_requirements": list(raw.get("recovery_requirements", [])),
            "cooldown_level": str(raw.get("cooldown_level", "none")),
            "cooldown_reason": str(raw.get("cooldown_reason", "")),
            "cooldown_marker": str(raw.get("cooldown_marker", "")),
            "cooldown_consumed_marker": str(raw.get("cooldown_consumed_marker", "")),
            "volatility_state": str(raw.get("volatility_state", "normal")),
            "volatility_reason": str(raw.get("volatility_reason", "")),
            "volatility_action": str(raw.get("volatility_action", "allow")),
        }
        if state["date"] != today:
            state["date"] = today
            state["trades_today"] = 0
            state["consecutive_losses"] = 0
            state["cooldown_until"] = ""
            state["start_balance"] = 0.0
        if state["week_key"] != week_key:
            state["week_key"] = week_key
            state["week_start_balance"] = 0.0
        return state

    def _save_guard_state(self, state: dict[str, Any], flags: dict[str, Any] | None = None) -> None:
        flags = dict(flags) if isinstance(flags, dict) else self.store.load_runtime_flags()
        flags["risk_guard_state"] = state
        self.store.save_runtime_flags(flags)

    def _load_loss_memory(self) -> dict[str, Any]:
        raw = self.store.load_learning_state().get("state", {})
        memory = raw.get("loss_memory", {})
        return {
            "processed_deals": list(memory.get("processed_deals", [])),
            "side_hour_stats": dict(memory.get("side_hour_stats", {})),
            "recent_outcomes": list(memory.get("recent_outcomes", [])),
            "pattern_lessons": dict(memory.get("pattern_lessons", {})),
            "lesson_backfill": dict(memory.get("lesson_backfill", {})),
            "quarantined_non_atlas_deals": list(memory.get("quarantined_non_atlas_deals", [])),
            "excluded_manual_test_outcomes": list(memory.get("excluded_manual_test_outcomes", [])),
            "quarantine_summary": dict(memory.get("quarantine_summary", {})),
            "definitions": dict(
                memory.get(
                    "definitions",
                    {
                        "loss": "A closed trade where net_pnl_zar < 0 (profit + swap + commission).",
                        "win": "A closed trade where net_pnl_zar >= 0.",
                        "draw": "A closed trade where net_pnl_zar == 0.",
                    },
                )
            ),
        }

    def _save_loss_memory(self, memory: dict[str, Any]) -> None:
        self.store.update_learning_state(lambda existing: {**existing, "loss_memory": memory})

    def _load_win_memory(self) -> dict[str, Any]:
        raw = self.store.load_learning_state().get("state", {})
        memory = raw.get("win_memory", {})
        return {
            "processed_deals": list(memory.get("processed_deals", [])),
            "context_stats": dict(memory.get("context_stats", {})),
            "recent_wins": list(memory.get("recent_wins", [])),
        }

    def _save_win_memory(self, memory: dict[str, Any]) -> None:
        self.store.update_learning_state(lambda existing: {**existing, "win_memory": memory})

    def _classify_loss_cause(
        self,
        *,
        outcome: str,
        exit_reason: str,
        duration_minutes: float,
        session: str,
        comment: str,
        spread_points: float = 0.0,
        setup_family: str = "unknown",
        regime: str = "unknown",
    ) -> str:
        if str(outcome or "").lower() != "loss":
            return ""
        note = str(comment or "").lower()
        exit_state = str(exit_reason or "").lower()
        family = str(setup_family or "unknown").strip().lower()
        regime_name = str(regime or "unknown").strip().lower()
        if any(token in note for token in ("news", "fomc", "cpi", "nfp")):
            return "news_spike"
        if float(spread_points or 0.0) >= min(float(getattr(self.settings, "max_spread_points", 80.0) or 80.0) * 0.6, 45.0):
            return "spread_liquidity_distortion"
        if family == "continuation" and exit_state == "stop_loss" and float(duration_minutes or 0.0) <= 8.0:
            return "false_continuation"
        if family == "pullback" and exit_state == "stop_loss" and float(duration_minutes or 0.0) <= 12.0:
            return "weak_pullback"
        if exit_state == "stop_loss" and float(duration_minutes or 0.0) >= 10.0 and family in {"pullback", "continuation", "breakout"}:
            return "late_entry"
        if exit_state == "stop_loss" and float(duration_minutes or 0.0) <= 6.0:
            return "stop_hunt"
        if float(duration_minutes or 0.0) >= 45.0:
            return "trend_reversal"
        if regime_name in {"range", "high_volatility_range"}:
            return "weak_structure"
        if str(session or "").lower() in {"late_us", "rollover", "unknown", "other"}:
            return "poor_timing"
        return "weak_structure"

    def _sync_closed_trade_learning(self, flags: dict[str, Any] | None = None) -> dict[str, Any]:
        if not hasattr(self, "_closed_learning_synced_until"):
            self._closed_learning_synced_until = None
        if not hasattr(self, "_closed_learning_sync_persisted_at"):
            self._closed_learning_sync_persisted_at = None
        memory = self._load_loss_memory()
        win_memory = self._load_win_memory()
        processed = set(str(x) for x in memory.get("processed_deals", []))
        win_processed = set(str(x) for x in win_memory.get("processed_deals", []))
        stats = dict(memory.get("side_hour_stats", {}))
        recent_outcomes = list(memory.get("recent_outcomes", []))
        win_stats = dict(win_memory.get("context_stats", {}))
        recent_wins = list(win_memory.get("recent_wins", []))
        original_memory = {
            "processed_deals": list(memory.get("processed_deals", [])),
            "side_hour_stats": dict(stats),
            "recent_outcomes": list(recent_outcomes),
            "pattern_lessons": dict(memory.get("pattern_lessons", {})),
            "definitions": dict(memory.get("definitions", {})),
        }
        original_win_memory = {
            "processed_deals": list(win_memory.get("processed_deals", [])),
            "context_stats": dict(win_stats),
            "recent_wins": list(recent_wins),
        }

        def _comment_marks_ours(raw: Any) -> bool:
            parsed = parse_broker_comment(raw)
            if str(parsed.get("decision_id", "") or "").strip():
                return True
            if str(parsed.get("strategy", "") or "").strip():
                return True
            comment = str(raw or "")
            compact = "".join(ch for ch in comment.lower() if ch.isalnum())
            # Accept legacy/new tags:
            # - AtlasAI / Atlas AI
            # - AAI_D<id>_S<tag> (current compact strategy-tagged comments)
            # - AAI_st_<strategy> (legacy strategy-tagged comments)
            return ("atlasai" in compact) or ("aaist" in compact)

        def _strategy_from_comment(raw: Any) -> str | None:
            parsed = parse_broker_comment(raw)
            strategy = str(parsed.get("strategy", "") or "").strip()
            if not strategy:
                return None
            return canonical_strategy_name(strategy)

        def _decision_id_from_comment(raw: Any) -> str | None:
            parsed = parse_broker_comment(raw)
            decision_id = str(parsed.get("decision_id", "") or "").strip().lower()
            return decision_id or None

        def _is_manual_broker_test_deal(*, comment: str, decision_id: str | None, ref_meta: dict[str, Any]) -> bool:
            from atlas_ai.learning.evidence_provenance import is_manual_broker_test
            if is_manual_broker_test({"strategy": _strategy_from_comment(comment), "decision_id": decision_id, "comment": comment}):
                return True
            haystack = " ".join(
                [
                    str(comment or ""),
                    str(decision_id or ""),
                    str(ref_meta.get("decision_id", "") or ""),
                    str(ref_meta.get("strategy", "") or ""),
                    str(ref_meta.get("requested_by", "") or ""),
                    str(ref_meta.get("request_origin", "") or ""),
                    str(ref_meta.get("payload_summary", "") or ""),
                ]
            ).lower()
            return "manual_broker_test" in haystack or "operator_manual_broker_test" in haystack

        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=14)
        if self._closed_learning_synced_until is None:
            raw_last_sync = str(flags.get("closed_learning_synced_until", "") or "")
            try:
                self._closed_learning_synced_until = datetime.fromisoformat(raw_last_sync) if raw_last_sync else None
            except Exception:
                self._closed_learning_synced_until = None
        if self._closed_learning_synced_until is not None:
            start = max(start, self._closed_learning_synced_until - timedelta(minutes=5))
        deals = mt5.history_deals_get(start, end) or []
        decision_strategy_map, decision_id_aliases, decision_meta_by_execution_ref = self._decision_sample_maps(limit=5000)
        owned_positions: set[str] = set()
        position_strategy: dict[str, str] = {}
        position_decision: dict[str, str] = {}
        position_open_times: dict[str, int] = {}
        # Build a fallback ownership map by position id, because some brokers do not
        # preserve magic/comment consistently on close-out records.
        for d in deals:
            if str(getattr(d, "symbol", "")) != self.settings.symbol:
                continue
            magic = int(getattr(d, "magic", 0) or 0)
            comment = str(getattr(d, "comment", "") or "")
            pos_id = str(getattr(d, "position_id", "") or "")
            order_id = str(getattr(d, "order", "") or "")
            deal_ticket = str(getattr(d, "ticket", "") or "")
            st_name = _strategy_from_comment(comment)
            did = str(_decision_id_from_comment(comment) or "").strip().lower()
            ref_meta = decision_meta_by_execution_ref.get(deal_ticket) or decision_meta_by_execution_ref.get(order_id) or {}
            if not did:
                did = str(ref_meta.get("decision_id", "") or "").strip().lower()
            if not st_name:
                st_name = str(ref_meta.get("strategy", "") or "").strip() or None
            if pos_id and st_name:
                position_strategy[pos_id] = canonical_strategy_name(st_name)
            if pos_id and did:
                position_decision[pos_id] = did
            if pos_id and int(getattr(d, "entry", -1)) not in {int(getattr(mt5, "DEAL_ENTRY_OUT", 1)), int(getattr(mt5, "DEAL_ENTRY_OUT_BY", 3))}:
                opened_at = int(getattr(d, "time", 0) or 0)
                if opened_at > 0:
                    position_open_times[pos_id] = min(opened_at, int(position_open_times.get(pos_id, opened_at)))
            if (magic == int(self.settings.magic_number)) or _comment_marks_ours(comment):
                if pos_id:
                    owned_positions.add(pos_id)

        repaired_outcomes: list[dict[str, Any]] = []
        for item in recent_outcomes:
            row = dict(item or {})
            pos_id = str(row.get("position_id", "") or "")
            raw_decision_id = str(row.get("decision_id", "") or position_decision.get(pos_id, "") or "").strip().lower()
            decision_id = decision_id_aliases.get(raw_decision_id, raw_decision_id) or None
            if decision_id:
                row["decision_id"] = decision_id
            if str(row.get("strategy", "") or "").strip().lower() in {"", "unknown"}:
                row["strategy"] = (
                    position_strategy.get(pos_id)
                    or decision_strategy_map.get(str(decision_id or "").lower(), "")
                    or row.get("strategy", "unknown")
                )
            repaired_outcomes.append(row)
        recent_outcomes = repaired_outcomes

        processed_now = 0
        quarantined_deals: list[dict[str, Any]] = []
        for d in deals:
            ticket = str(getattr(d, "ticket", ""))
            if not ticket or ticket in processed:
                continue
            if str(getattr(d, "symbol", "")) != self.settings.symbol:
                continue
            magic = int(getattr(d, "magic", 0) or 0)
            comment = str(getattr(d, "comment", "") or "")
            pos_id = str(getattr(d, "position_id", "") or "")
            order_id = str(getattr(d, "order", "") or "")
            deal_ticket = str(getattr(d, "ticket", "") or "")
            ref_meta = decision_meta_by_execution_ref.get(deal_ticket) or decision_meta_by_execution_ref.get(order_id) or {}
            is_ours = (magic == int(self.settings.magic_number)) or _comment_marks_ours(comment) or (pos_id in owned_positions)
            if not is_ours:
                quarantined_deals.append(
                    {
                        "ticket": ticket,
                        "order": str(getattr(d, "order", "") or ""),
                        "position_id": pos_id,
                        "time": datetime.fromtimestamp(int(getattr(d, "time", 0) or 0), tz=timezone.utc).isoformat(),
                        "symbol": str(getattr(d, "symbol", "")),
                        "magic": magic,
                        "comment": comment,
                        "volume": float(getattr(d, "volume", 0.0) or 0.0),
                        "profit": round(
                            float(getattr(d, "profit", 0.0) or 0.0)
                            + float(getattr(d, "swap", 0.0) or 0.0)
                            + float(getattr(d, "commission", 0.0) or 0.0),
                            2,
                        ),
                        "reason": "not_atlas_tagged",
                    }
                )
                continue
            entry_type = int(getattr(d, "entry", -1))
            deal_entry_out = int(getattr(mt5, "DEAL_ENTRY_OUT", 1))
            deal_entry_out_by = int(getattr(mt5, "DEAL_ENTRY_OUT_BY", 3))
            if entry_type not in {deal_entry_out, deal_entry_out_by}:
                continue
            raw_decision_id = str(_decision_id_from_comment(comment) or position_decision.get(pos_id, "") or "").strip().lower()
            decision_id = decision_id_aliases.get(raw_decision_id, raw_decision_id) or None
            if _is_manual_broker_test_deal(comment=comment, decision_id=decision_id, ref_meta=ref_meta):
                processed.add(ticket)
                quarantined_deals.append(
                    {
                        "ticket": ticket,
                        "order": str(getattr(d, "order", "") or ""),
                        "position_id": pos_id,
                        "time": datetime.fromtimestamp(int(getattr(d, "time", 0) or 0), tz=timezone.utc).isoformat(),
                        "symbol": str(getattr(d, "symbol", "")),
                        "magic": magic,
                        "comment": comment,
                        "volume": float(getattr(d, "volume", 0.0) or 0.0),
                        "profit": round(
                            float(getattr(d, "profit", 0.0) or 0.0)
                            + float(getattr(d, "swap", 0.0) or 0.0)
                            + float(getattr(d, "commission", 0.0) or 0.0),
                            2,
                        ),
                        "reason": "manual_broker_test_excluded_from_learning",
                    }
                )
                continue

            processed.add(ticket)
            processed_now += 1
            profit = float(getattr(d, "profit", 0.0) or 0.0) + float(getattr(d, "swap", 0.0) or 0.0) + float(getattr(d, "commission", 0.0) or 0.0)
            deal_type = int(getattr(d, "type", -1))
            # Closing SELL means original position was BUY, and vice versa.
            if deal_type == int(getattr(mt5, "DEAL_TYPE_SELL", 1)):
                side = "buy"
            elif deal_type == int(getattr(mt5, "DEAL_TYPE_BUY", 0)):
                side = "sell"
            else:
                side = "flat"
            hour = datetime.fromtimestamp(int(getattr(d, "time", 0) or 0), tz=timezone.utc).hour
            key = f"{side}:{hour}"
            st = dict(stats.get(key, {"wins": 0, "losses": 0, "pnl": 0.0}))
            st["pnl"] = float(st.get("pnl", 0.0)) + profit
            if profit >= 0:
                st["wins"] = int(st.get("wins", 0)) + 1
                outcome = "win" if profit > 0 else "draw"
            else:
                st["losses"] = int(st.get("losses", 0)) + 1
                outcome = "loss"
            stats[key] = st
            strategy_name = (
                position_strategy.get(pos_id)
                or decision_strategy_map.get(str(decision_id or "").lower(), "")
                or _strategy_from_comment(comment)
                or "unknown"
            )
            strategy_name = canonical_strategy_name(strategy_name)
            setup_family = str(ref_meta.get("setup_family", "") or "").strip().lower() or self._strategy_to_setup_family(strategy_name)
            regime_name = str(ref_meta.get("regime", "") or "").strip().lower() or "unknown"
            entry_session = str(ref_meta.get("session", "") or "").strip().lower()
            close_time = datetime.fromtimestamp(int(getattr(d, "time", 0) or 0), tz=timezone.utc)
            session_label = self._session_label(close_time)
            open_time_raw = int(position_open_times.get(pos_id, int(getattr(d, "time", 0) or 0)) or int(getattr(d, "time", 0) or 0))
            duration_minutes = round(max(0.0, (int(getattr(d, "time", 0) or 0) - open_time_raw) / 60.0), 2)
            exit_reason = "stop_loss" if "[sl" in comment.lower() else ("take_profit" if "[tp" in comment.lower() else "other")
            loss_classification = self._classify_loss_cause(
                outcome=outcome,
                exit_reason=exit_reason,
                duration_minutes=duration_minutes,
                session=session_label,
                comment=comment,
                spread_points=0.0,
                setup_family=setup_family,
                regime=regime_name,
            )
            pattern_context = dict(ref_meta.get("context", {}) or {})
            pattern_context.setdefault("regime", regime_name)
            pattern_context.setdefault("session", entry_session or session_label)
            pattern_context.setdefault("higher_tf_bias", str(ref_meta.get("higher_tf_bias", "") or "").strip().lower())
            pattern_context.setdefault("spread_points", float(ref_meta.get("spread_points", 0.0) or 0.0))
            pattern_context.setdefault("htf_aligned", side in {"buy", "sell"} and str(pattern_context.get("higher_tf_bias", "")) == side)
            pattern_fingerprint = self._build_trade_pattern_fingerprint(
                strategy=strategy_name,
                side=side,
                setup_family=setup_family,
                session=entry_session or session_label,
                regime=regime_name,
                context=pattern_context,
            )
            execution_lesson = (
                self._execution_lesson_from_loss(loss_classification or "weak_structure", setup_family)
                if outcome == "loss"
                else {}
            )
            outcome_row = {
                "ticket": ticket,
                "position_id": pos_id,
                "decision_id": decision_id,
                "time": close_time.isoformat(),
                "side": side,
                "hour_utc": hour,
                "net_pnl_zar": round(profit, 2),
                "outcome": outcome,
                "session": session_label,
                "entry_session": entry_session or session_label,
                "regime": regime_name,
                "exit_reason": exit_reason,
                "duration_minutes": duration_minutes,
                "loss_classification": loss_classification,
                "setup_family": setup_family,
                "pattern_fingerprint": pattern_fingerprint,
                "execution_lesson": execution_lesson,
                "post_trade_review": {
                    "classification": loss_classification or ("none" if outcome != "loss" else "weak_structure"),
                    "duration_minutes": duration_minutes,
                    "exit_reason": exit_reason,
                    "setup_family": setup_family,
                    "regime": regime_name,
                    "bounded_learning_only": True,
                    "validation_required": True,
                    "reversible": True,
                    "pattern_fingerprint": pattern_fingerprint,
                    "execution_lesson": execution_lesson,
                },
                "strategy": strategy_name,
            }
            post_trade_lesson = build_post_trade_lesson(outcome_row, pattern_context)
            outcome_row["post_trade_lesson"] = post_trade_lesson
            outcome_row["post_trade_review"]["post_trade_lesson"] = post_trade_lesson
            recent_outcomes.append(outcome_row)
            self._update_pattern_lesson_memory(memory, outcome_row)
            if strategy_name != "unknown":
                self.allocator.update(strategy_name, profit)
            if outcome == "win" and ticket not in win_processed:
                session = self._session_label(datetime.fromtimestamp(int(getattr(d, "time", 0) or 0), tz=timezone.utc))
                context_key = f"{strategy_name}|{side}|{session}|{hour}"
                cur = dict(win_stats.get(context_key, {"wins": 0, "pnl": 0.0}))
                cur["wins"] = int(cur.get("wins", 0)) + 1
                cur["pnl"] = float(cur.get("pnl", 0.0)) + float(profit)
                win_stats[context_key] = cur
                recent_wins.append(
                    {
                        "ticket": ticket,
                        "strategy": strategy_name,
                        "side": side,
                        "session": session,
                        "hour_utc": hour,
                        "net_pnl_zar": round(float(profit), 2),
                    }
                )
                win_processed.add(ticket)

        memory["processed_deals"] = sorted(processed)[-2000:]
        memory["side_hour_stats"] = stats
        memory["recent_outcomes"] = recent_outcomes[-200:]
        memory["pattern_lessons"] = dict(memory.get("pattern_lessons", {}))
        quarantine_existing = list(memory.get("quarantined_non_atlas_deals", []) or [])
        if quarantined_deals:
            quarantine_existing.extend(quarantined_deals)
        memory["quarantined_non_atlas_deals"] = quarantine_existing[-500:]
        memory["quarantine_summary"] = {
            "reason": "Only broker deals tagged with Atlas magic/comment or linked to an Atlas-owned position can enter learning.",
            "latest_quarantined_count": len(quarantined_deals),
            "total_retained_quarantined": len(memory["quarantined_non_atlas_deals"]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        win_memory["processed_deals"] = sorted(win_processed)[-2000:]
        win_memory["context_stats"] = win_stats
        win_memory["recent_wins"] = recent_wins[-300:]
        loss_changed = memory != original_memory
        win_changed = win_memory != original_win_memory
        if loss_changed:
            self._save_loss_memory(memory)
        if win_changed:
            self._save_win_memory(win_memory)
        if processed_now > 0:
            self.store.log_event("learning", f"Processed {processed_now} closed deal(s) for win/loss learning.")
            fresh = recent_outcomes[-processed_now:]
            f_w = sum(1 for x in fresh if str(x.get("outcome", "")) == "win")
            f_l = sum(1 for x in fresh if str(x.get("outcome", "")) == "loss")
            self.store.log_event(
                "learning",
                "Closed-trade learning summary updated.",
                {
                    "processed_now": processed_now,
                    "wins": f_w,
                    "losses": f_l,
                    "learning_action": "Future confidence is penalized for repeating loss-heavy side/hour patterns.",
                },
            )
            try:
                backfill = rebuild_loss_pattern_lessons(self.store)
                self.store.log_event(
                    "learning",
                    "Loss-pattern lesson memory refreshed after closed trade.",
                    {
                        "processed_now": processed_now,
                        "recent_outcomes": int(backfill.get("recent_outcomes", 0) or 0),
                        "pattern_lessons": int(backfill.get("pattern_lessons", 0) or 0),
                        "loss_dominant_lessons": int(backfill.get("loss_dominant_lessons", 0) or 0),
                        "decision_context": dict(backfill.get("decision_context", {}) or {}),
                        "context_quality": dict(backfill.get("context_quality", {}) or {}),
                        "policy_note": str(backfill.get("policy_note", "") or ""),
                    },
                )
                memory = self._load_loss_memory()
            except Exception as exc:
                self.store.log_event("learning", "Loss-pattern lesson memory refresh failed.", {"error": str(exc)[:240]})
            try:
                playbook = build_loss_to_win_playbook(self.store)
                self.store.log_event(
                    "learning",
                    "Loss-to-win playbook refreshed after closed trade.",
                    {
                        "status": str(playbook.get("status", "") or ""),
                        "lessons": len(dict(playbook.get("lessons", {}) or {})),
                        "losses": int(playbook.get("losses", 0) or 0),
                    },
                )
            except Exception as exc:
                self.store.log_event("learning", "Loss-to-win playbook refresh failed.", {"error": str(exc)[:240]})
            try:
                winner_playbook = build_winner_playbook(self.store)
                self.store.log_event(
                    "learning",
                    "Winner playbook refreshed after closed trade.",
                    {
                        "status": str(winner_playbook.get("status", "") or ""),
                        "lessons": len(dict(winner_playbook.get("lessons", {}) or {})),
                        "wins": int(winner_playbook.get("wins", 0) or 0),
                    },
                )
            except Exception as exc:
                self.store.log_event("learning", "Winner playbook refresh failed.", {"error": str(exc)[:240]})
            try:
                intelligence = refresh_trade_outcome_intelligence(self.store)
                self.store.log_event(
                    "learning",
                    "Trade-outcome intelligence refreshed.",
                    {
                        "sample_count": int(intelligence.get("sample_count", 0) or 0),
                        "favorable_regimes": list(dict(intelligence.get("regime_memory", {}) or {}).get("favorable_regimes", [])),
                        "unfavorable_regimes": list(dict(intelligence.get("regime_memory", {}) or {}).get("unfavorable_regimes", [])),
                    },
                )
            except Exception as exc:
                self.store.log_event("learning", "Trade-outcome intelligence refresh failed.", {"error": str(exc)[:240]})
        self._closed_learning_synced_until = end
        should_persist_sync = (
            processed_now > 0
            or self._closed_learning_sync_persisted_at is None
            or (end - self._closed_learning_sync_persisted_at).total_seconds() >= 10 * 60
        )
        if should_persist_sync:
            flags["closed_learning_synced_until"] = end.isoformat()
            self.store.save_runtime_flags(flags)
            self._closed_learning_sync_persisted_at = end
        return memory

    def _win_memory_boost(self, chosen: TradeSignal, now_utc: datetime) -> tuple[float, dict[str, Any]]:
        if not bool(self.settings.win_memory_enabled):
            return float(chosen.confidence), {"enabled": False}
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}
        strategy = self._strategy_key(str(getattr(chosen, "strategy", "")))
        session = self._session_label(now_utc)
        hour = int(now_utc.hour)
        win_memory = self._load_win_memory()
        ctx = dict(win_memory.get("context_stats", {}))
        key = f"{strategy}|{side}|{session}|{hour}"
        st = dict(ctx.get(key, {}))
        wins = int(st.get("wins", 0))
        if wins < 2:
            return float(chosen.confidence), {"enabled": True, "applied": False, "wins": wins}
        avg_pnl = float(st.get("pnl", 0.0)) / max(1, wins)
        boost = min(float(self.settings.win_memory_boost_cap), 0.02 + min(0.06, wins * 0.005))
        if avg_pnl <= 0:
            boost = min(boost, 0.02)
        new_conf = min(1.0, float(chosen.confidence) * (1.0 + boost))
        return new_conf, {"enabled": True, "applied": True, "wins": wins, "avg_pnl": round(avg_pnl, 2), "boost": round(boost, 4)}

    def _online_learning_adjustment(self, chosen: TradeSignal, status: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}
        try:
            state = dict(self.store.load_learning_state().get("state", {}) or {})
        except Exception:
            state = {}
        online = dict(state.get("online_learning", {}) or {})
        latest = dict(online.get("latest", {}) or {})
        allowed_influence = str(latest.get("allowed_influence", "report_only") or "report_only")
        if allowed_influence != "bounded_confidence_haircut":
            return float(chosen.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "report_only_no_confidence_change" if allowed_influence == "report_only" else "not_eligible",
                "allowed_influence": allowed_influence,
                "policy_note": "Report-only online learning records evidence but cannot change confidence or block execution.",
            }
        evidence = float(latest.get("evidence_strength", 0.0) or 0.0)
        samples = int(latest.get("samples", 0) or 0)
        cap = max(0.0, min(0.08, float(latest.get("confidence_haircut_cap", 0.0) or 0.0)))
        if evidence < 0.65 or samples < 8 or cap <= 0.0:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "weak_evidence", "evidence_strength": round(evidence, 4), "samples": samples}

        latest_context = dict(latest.get("context", {}) or {})
        current_context = dict(status.get("context_snapshot", {}) or {})
        latest_blockers = {str(x) for x in list(latest.get("current_blockers", []) or [])}
        current_blockers = {str(x) for x in list(dict(status.get("decision_analysis", {}) or {}).get("hard_blockers", []) or dict(status.get("decision_analysis", {}) or {}).get("blockers", []) or [])}
        match_score = 0
        for key in ("session", "momentum", "location"):
            if str(latest_context.get(key, "") or "") and str(latest_context.get(key, "") or "") == str(current_context.get(key, "") or ""):
                match_score += 1
        if latest_blockers.intersection(current_blockers):
            match_score += 1
        spread = float(current_context.get("spread_points", dict(status.get("risk", {}) or {}).get("spread_points", 0.0)) or 0.0)
        latest_spread = float(latest_context.get("spread_points", 0.0) or 0.0)
        if spread >= 40.0 and latest_spread >= 40.0:
            match_score += 1
        if match_score < 2:
            return float(chosen.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "context_mismatch",
                "match_score": match_score,
                "allowed_influence": "bounded_confidence_haircut",
            }

        haircut = min(cap, 0.02 + min(0.06, evidence * 0.05))
        new_conf = max(0.0, float(chosen.confidence) * (1.0 - haircut))
        return new_conf, {
            "enabled": True,
            "applied": True,
            "allowed_influence": "bounded_confidence_haircut",
            "haircut": round(haircut, 4),
            "factor": round(1.0 - haircut, 4),
            "evidence_strength": round(evidence, 4),
            "overreaction_risk": round(float(latest.get("overreaction_risk", 0.0) or 0.0), 4),
            "samples": samples,
            "match_score": match_score,
            "fingerprint": str(latest.get("top_fingerprint", "") or ""),
            "policy_note": "Online learning can only reduce confidence within cap; it cannot boost, bypass gates, or change risk/execution policy.",
        }

    def _loss_pattern_penalty(self, side: str, now_utc: datetime, memory: dict[str, Any]) -> tuple[float, str]:
        detail = self._loss_pattern_penalty_detail(side, now_utc, memory)
        return float(detail.get("penalty", 0.0) or 0.0), str(detail.get("reason", "") or "")

    def _loss_pattern_penalty_detail(self, side: str, now_utc: datetime, memory: dict[str, Any]) -> dict[str, Any]:
        side_key = str(side or "").strip().lower()
        hour = int(now_utc.hour)
        key = f"{side_key}:{hour}"
        min_samples = 4
        live_hour_candidates = [
            dict(x or {})
            for x in list(memory.get("recent_outcomes", []) or [])
            if str(dict(x or {}).get("side", "") or "").strip().lower() == side_key
            and int(dict(x or {}).get("hour_utc", -1) or -1) == hour
            and not self._is_historical_backtest_outcome(dict(x or {}))
        ]
        live_hour_outcomes, stale_live_excluded = self._fresh_live_outcomes(live_hour_candidates, now_utc=now_utc)
        source = "live_recent" if live_hour_outcomes else "aggregate_side_hour"
        if live_hour_outcomes:
            wins = sum(1 for x in live_hour_outcomes if str(x.get("outcome", "")) == "win")
            losses = sum(1 for x in live_hour_outcomes if str(x.get("outcome", "")) == "loss")
        elif stale_live_excluded:
            wins = 0
            losses = 0
        else:
            st = dict((memory.get("side_hour_stats", {}) or {}).get(key, {}))
            wins = int(st.get("wins", 0))
            losses = int(st.get("losses", 0))
        total = wins + losses
        base = {
            "applied": False,
            "code": "hour_loss_memory",
            "side": side_key,
            "scope": "side_hour",
            "source": source,
            "hour_utc": hour,
            "samples": total,
            "wins": wins,
            "losses": losses,
            "stale_live_excluded": stale_live_excluded,
            "thresholds": {"min_samples": min_samples, "min_loss_rate": 0.55, "max_penalty": 0.25},
            "clears_when": "hour-specific loss rate falls below 0.55 with enough fresh samples",
            "evidence_required": ["fresh live outcomes for the same side and UTC hour", "cleaner entry confirmation"],
        }
        if stale_live_excluded and not live_hour_outcomes:
            base["source"] = "stale_live_memory"
            base["clears_when"] = "stale hour-specific live loss memory has aged out; wait for fresh same-side outcomes"
            return base
        if live_hour_outcomes and total < min_samples:
            base["insufficient_live_samples"] = True
            return base
        if total < min_samples:
            return base
        loss_rate = losses / total
        base["loss_rate"] = round(loss_rate, 4)
        if loss_rate < 0.55:
            return base
        penalty = min(0.25, (loss_rate - 0.5) * 0.5)
        reason = f"Loss memory penalty {penalty:.3f} (hour {now_utc.hour:02d}, loss_rate {loss_rate:.2f})"
        return {**base, "applied": True, "penalty": penalty, "reason": reason}

    @staticmethod
    def _is_historical_backtest_outcome(row: dict[str, Any]) -> bool:
        review = dict(row.get("post_trade_review", {}) or {})
        return str(review.get("source", "") or "").strip().lower() == "historical_backtest" or str(row.get("ticket", "")).startswith("bt_")

    @staticmethod
    def _outcome_age_hours(row: dict[str, Any], now_utc: datetime | None = None) -> float | None:
        raw = str(row.get("time", "") or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        now = now_utc or datetime.now(timezone.utc)
        return max(0.0, (now - parsed.astimezone(timezone.utc)).total_seconds() / 3600.0)

    def _fresh_live_outcomes(
        self,
        rows: list[dict[str, Any]],
        *,
        max_age_hours: float = 72.0,
        now_utc: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        fresh: list[dict[str, Any]] = []
        stale = 0
        for row in rows:
            if self._is_historical_backtest_outcome(row):
                continue
            age = self._outcome_age_hours(row, now_utc=now_utc)
            if age is not None and age > max_age_hours:
                stale += 1
                continue
            fresh.append(row)
        return fresh, stale

    def _recent_side_penalty(self, side: str, memory: dict[str, Any], *, current_session: str = "") -> tuple[float, str]:
        detail = self._recent_side_penalty_detail(side, memory, current_session=current_session)
        return float(detail.get("penalty", 0.0) or 0.0), str(detail.get("reason", "") or "")

    def _recent_side_penalty_detail(self, side: str, memory: dict[str, Any], *, current_session: str = "") -> dict[str, Any]:
        side_key = str(side or "").strip().lower()
        outcomes = [
            dict(x)
            for x in list(memory.get("recent_outcomes", []))
            if str(dict(x or {}).get("side", "")).strip().lower() == side_key
        ]
        base = {
            "applied": False,
            "code": "recent_side_memory",
            "side": side_key,
            "source": "none",
            "scope": "side",
            "session": str(current_session or "").strip().lower(),
            "samples": 0,
            "thresholds": {
                "min_samples": 6,
                "healthy_win_rate": 0.55,
                "watch_win_rate": 0.45,
                "max_penalty": 0.22,
            },
            "clears_when": "recent same-side live win rate recovers or the loss streak breaks",
            "evidence_required": ["fresh live outcomes for the same side", "loss streak reset", "side-specific win rate improvement"],
        }
        if len(outcomes) < 6:
            return base
        live_outcomes, stale_live_excluded = self._fresh_live_outcomes(outcomes)
        source = "live" if live_outcomes else "historical"
        if live_outcomes:
            outcomes = live_outcomes
        elif stale_live_excluded:
            return {
                **base,
                "source": "stale_live_memory",
                "stale_live_excluded": stale_live_excluded,
                "clears_when": "stale live loss memory has aged out; wait for fresh same-side outcomes",
            }
        session_key = str(current_session or "").strip().lower()
        session_scoped = False
        if session_key:
            session_outcomes = [
                x
                for x in outcomes
                if str(x.get("entry_session", x.get("session", "")) or "").strip().lower() == session_key
            ]
            if len(session_outcomes) >= 6:
                outcomes = session_outcomes
                session_scoped = True
        recent = outcomes[-12:]
        wins = sum(1 for x in recent if str(x.get("outcome", "")) == "win")
        losses = sum(1 for x in recent if str(x.get("outcome", "")) == "loss")
        samples = wins + losses
        base.update(
            {
                "source": source,
                "scope": "side_session" if session_scoped else "side",
                "samples": samples,
                "wins": wins,
                "losses": losses,
            }
        )
        if samples < 6:
            return base
        win_rate = wins / max(1, samples)
        base["win_rate"] = round(win_rate, 4)

        streak = 0
        for x in reversed(recent):
            if str(x.get("outcome", "")) == "loss":
                streak += 1
            else:
                break
        base["loss_streak"] = streak

        if win_rate >= 0.55 and streak <= 2:
            return base
        if win_rate >= 0.45 and streak < 2:
            return base

        penalty = min(0.22, max(0.05, (0.45 - win_rate) * 0.35) + min(0.08, streak * 0.02))
        scope_text = f", session={session_key}" if session_scoped else ""
        reason = f"Recent {side_key} penalty {penalty:.3f} ({source}{scope_text}, wr={win_rate:.2f}, loss_streak={streak})"
        return {**base, "applied": True, "penalty": penalty, "reason": reason}

    def _loss_learning_recovery_evidence(
        self,
        *,
        side: str,
        status: dict[str, Any],
        hour_detail: dict[str, Any],
        side_detail: dict[str, Any],
    ) -> dict[str, Any]:
        """Decide whether current evidence corrects a recent loss pattern.

        This keeps loss learning from becoming a blanket side ban. It only
        relaxes the loss-memory haircut when the current setup shows concrete
        evidence that the previous failure mode has changed.
        """
        side_key = str(side or "").strip().lower()
        context = dict(status.get("context_snapshot", {}) or {})
        higher_tf = str(
            dict(status.get("higher_timeframe", {}) or {}).get("bias")
            or context.get("higher_timeframe_bias")
            or ""
        ).strip().lower()
        location = str(context.get("location", context.get("entry_location_state", "")) or "").strip().lower()
        momentum = str(context.get("momentum", context.get("entry_momentum_state", "")) or "").strip().lower()
        trend_structure = str(context.get("trend_structure", "") or "").strip().upper()
        trend_state = str(context.get("trend_state", "") or "").strip().lower()
        setup_hint = str(context.get("setup_type_hint", status.get("setup_family", "")) or "").strip().lower()
        support = dict(status.get("same_side_support", {}) or {})
        support_count = int(support.get("count", support.get("active_count", 0)) or 0)
        confidence = float(dict(status.get("chosen", {}) or {}).get("confidence", 0.0) or 0.0)
        threshold = float(status.get("min_confidence_threshold", dict(status.get("thresholds", {}) or {}).get("confidence", 0.0)) or 0.0)

        confirmations: list[str] = []
        missing: list[str] = []

        if higher_tf == side_key:
            confirmations.append("higher_timeframe_aligned")
        else:
            missing.append("higher_timeframe_alignment")

        if location not in {"", "unknown", "unclear", "mid_range"}:
            confirmations.append(f"entry_location={location}")
        else:
            missing.append("clean_entry_location")

        if momentum == "strong":
            confirmations.append("strong_momentum")
        else:
            missing.append("strong_momentum")

        if support_count >= 2:
            confirmations.append(f"same_side_support={support_count}")
        else:
            missing.append("same_side_support")

        structure_side = "buy" if trend_structure == "HH/HL" or trend_state == "uptrend" else (
            "sell" if trend_structure == "LH/LL" or trend_state == "downtrend" else "neutral"
        )
        if structure_side == side_key:
            confirmations.append("structure_aligned")
        elif setup_hint in {"consolidation", "mean_reversion"} and location not in {"mid_range", "", "unknown", "unclear"}:
            confirmations.append("range_setup_uses_location_not_trend_structure")
        else:
            missing.append("structure_or_range_edge_confirmation")

        if threshold <= 0.0 or confidence >= threshold:
            confirmations.append("confidence_above_threshold")
        else:
            missing.append("confidence_above_threshold")

        required_core = {
            "higher_timeframe_aligned",
            "clean_entry_location",
            "strong_momentum",
            "same_side_support",
            "structure_or_range_edge_confirmation",
        }
        missing_core = [item for item in missing if item in required_core]
        cleared = len(missing_core) == 0
        partial = not cleared and len(confirmations) >= 4 and "higher_timeframe_aligned" in confirmations
        reduction_factor = 0.0 if cleared else (0.45 if partial else 1.0)

        return {
            "evaluated": True,
            "cleared": cleared,
            "partial": partial,
            "reduction_factor": reduction_factor,
            "confirmations": confirmations[:8],
            "missing": missing[:8],
            "next_time_action": (
                "loss memory cleared because current setup shows corrected entry evidence"
                if cleared
                else "trade only when the current setup proves it is different from the recent losing pattern"
            ),
            "details": {
                "hour_penalty_applied": bool(hour_detail.get("applied", False)),
                "side_penalty_applied": bool(side_detail.get("applied", False)),
                "location": location,
                "momentum": momentum,
                "higher_timeframe_bias": higher_tf,
                "same_side_support": support_count,
                "structure_side": structure_side,
            },
            "policy_note": "Recovery evidence can reduce a loss-memory haircut only when corrective confirmations are present; it cannot bypass risk or execution gates.",
        }

    def _strategy_key(self, strategy_name: str) -> str:
        raw = canonical_strategy_name(strategy_name)
        if raw.startswith("evolved_rule("):
            # Keep promoted evolved variants distinct so weak historical
            # aggregate stats for the generic evolved_rule bucket do not hide
            # freshly promoted candidates in live evaluation.
            return raw
        base = raw.split("(", 1)[0].strip()
        return base or raw or "unknown"

    def _loss_pattern_block_reason(self, side: str, memory: dict[str, Any], now_utc: datetime) -> str:
        detail = self._loss_pattern_block_detail(side, memory, now_utc)
        return str(detail.get("reason", "") or "")

    def _loss_pattern_block_detail(self, side: str, memory: dict[str, Any], now_utc: datetime) -> dict[str, Any]:
        base = {
            "blocked": False,
            "code": "m1_loss_pattern_block",
            "side": str(side or ""),
            "scope": "m1_side_memory",
            "hour_utc": int(now_utc.hour),
            "clears_when": "same-side win rate improves, the loss streak breaks, or hour context recovers",
            "evidence_required": ["fresh M1 outcomes", "loss streak reset", "hour-specific win rate improvement"],
            "recovery_checklist": [
                {
                    "code": "loss_streak_breaks",
                    "label": "A same-side or opposite-side win breaks the immediate loss streak.",
                    "status": "waiting",
                },
                {
                    "code": "same_side_win_rate_improves",
                    "label": "Recent same-side win rate rises above the loss-pattern threshold.",
                    "status": "waiting",
                },
                {
                    "code": "hour_context_recovers",
                    "label": "Current hour context stops being loss-dominant.",
                    "status": "waiting",
                },
                {
                    "code": "setup_differs_from_loss_memory",
                    "label": "Next setup has cleaner confirmation or differs from stored weak fingerprints.",
                    "status": "review",
                },
            ],
            "policy_note": "Loss-pattern block detail is read-only observability; it does not change strategy, risk, guards, or execution policy.",
        }
        if str(self.settings.timeframe).upper() != "M1":
            return base
        if not bool(getattr(self.settings, "m1_loss_block_enabled", True)):
            return base
        if side not in {"buy", "sell"}:
            return base

        outcomes = [x for x in list(memory.get("recent_outcomes", [])) if str(x.get("side", "")).lower() == side]
        outcomes = outcomes[-12:]
        if not outcomes:
            return base
        wins = sum(1 for x in outcomes if str(x.get("outcome", "")) == "win")
        losses = sum(1 for x in outcomes if str(x.get("outcome", "")) == "loss")
        samples = wins + losses
        base.update({"samples": samples, "wins": wins, "losses": losses})
        min_samples = int(getattr(self.settings, "m1_loss_block_min_samples", 6))
        max_wr = float(getattr(self.settings, "m1_loss_block_max_win_rate", 0.34))
        min_streak = int(getattr(self.settings, "m1_loss_block_min_streak", 3))
        base["thresholds"] = {"min_samples": min_samples, "max_win_rate": max_wr, "min_loss_streak": min_streak}
        if samples < int(getattr(self.settings, "m1_loss_block_min_samples", 6)):
            return base
        win_rate = wins / max(1, samples)
        base["win_rate"] = round(win_rate, 4)

        streak = 0
        for x in reversed(outcomes):
            if str(x.get("outcome", "")) == "loss":
                streak += 1
            else:
                break
        base["loss_streak"] = streak

        hour_key = f"{side}:{int(now_utc.hour):02d}"
        hour_stats = dict(memory.get("side_hour_stats", {})).get(hour_key, {}) or {}
        hour_wins = int(hour_stats.get("wins", 0) or 0)
        hour_losses = int(hour_stats.get("losses", 0) or 0)
        hour_samples = hour_wins + hour_losses
        hour_wr = hour_wins / max(1, hour_samples) if hour_samples else 0.5
        base["hour_context"] = {
            "key": hour_key,
            "samples": hour_samples,
            "wins": hour_wins,
            "losses": hour_losses,
            "win_rate": round(hour_wr, 4),
        }
        checklist = list(base.get("recovery_checklist", []) or [])
        for row in checklist:
            if row.get("code") == "loss_streak_breaks":
                row["status"] = "blocked" if streak >= min_streak else "clear"
                row["detail"] = f"current_streak={streak}, required_below={min_streak}"
            elif row.get("code") == "same_side_win_rate_improves":
                row["status"] = "blocked" if win_rate <= max_wr else "clear"
                row["detail"] = f"wr={win_rate:.2f}, needs_above={max_wr:.2f}"
            elif row.get("code") == "hour_context_recovers":
                row["status"] = "blocked" if hour_samples >= max(6, min_samples) and hour_wr <= max_wr else "watching"
                row["detail"] = f"{hour_key}: wr={hour_wr:.2f}, samples={hour_samples}"
        base["recovery_checklist"] = checklist
        if streak >= min_streak and win_rate <= max_wr:
            reason = f"M1 loss-pattern block: repeated {side} losses (wr={win_rate:.2f}, streak={streak})."
            return {**base, "blocked": True, "trigger": "repeated_side_losses", "reason": reason}
        if hour_samples >= max(6, int(getattr(self.settings, "m1_loss_block_min_samples", 6))) and hour_wr <= max_wr:
            reason = f"M1 loss-pattern block: weak {side} hour context {hour_key} (wr={hour_wr:.2f}, n={hour_samples})."
            return {**base, "blocked": True, "trigger": "weak_hour_context", "reason": reason}
        return base

    def _recent_win_rate(self, memory: dict[str, Any], lookback: int) -> tuple[float, int]:
        outcomes = list(memory.get("recent_outcomes", []))
        sample = outcomes[-max(1, int(lookback)) :]
        wins = sum(1 for x in sample if str(x.get("outcome", "")) == "win")
        losses = sum(1 for x in sample if str(x.get("outcome", "")) == "loss")
        n = wins + losses
        if n <= 0:
            return 0.5, 0
        return wins / n, n

    def _filter_weak_strategies(self, signals: list[TradeSignal]) -> tuple[list[TradeSignal], list[str]]:
        if not bool(self.settings.win_rate_focus_enabled):
            return signals, []
        stats = self.allocator.stats()
        min_n = max(1, int(self.settings.min_strategy_samples))
        min_wr = max(0.0, min(1.0, float(self.settings.min_strategy_win_rate)))
        hard_prune_enabled = bool(getattr(self.settings, "hard_prune_weak_strategies_enabled", True))
        hard_prune_min_samples = max(min_n, int(getattr(self.settings, "hard_prune_min_samples", 120)))
        hard_prune_max_wr = max(0.0, min(1.0, float(getattr(self.settings, "hard_prune_max_win_rate", 0.43))))
        hard_prune_require_negative_pnl = bool(getattr(self.settings, "hard_prune_require_negative_pnl", True))
        severe_weak_min_samples = max(hard_prune_min_samples, int(getattr(self.settings, "severe_weak_strategy_min_samples", 500)))
        severe_weak_max_wr = max(0.0, min(1.0, float(getattr(self.settings, "severe_weak_strategy_max_win_rate", 0.28))))
        kept: list[TradeSignal] = []
        blocked: list[str] = []
        for s in signals:
            key = self._strategy_key(str(getattr(s, "strategy", "")))
            st = stats.get(key)
            if st is None:
                kept.append(s)
                continue
            n = int(getattr(st, "wins", 0)) + int(getattr(st, "losses", 0))
            wr = float(getattr(st, "win_rate", 0.5))
            pnl = float(getattr(st, "pnl", 0.0) or 0.0)
            if n >= severe_weak_min_samples and wr < severe_weak_max_wr and pnl < 0.0:
                blocked.append(f"{key} severe_blocked wr={wr:.2f} n={n} pnl={pnl:.2f}")
                continue
            hard_prune_blocked = hard_prune_enabled and key not in {"momentum_continuation", "ema_crossover"}
            if hard_prune_blocked and n >= hard_prune_min_samples and wr < hard_prune_max_wr:
                if (not hard_prune_require_negative_pnl) or pnl < 0.0:
                    blocked.append(f"{key} hard_pruned wr={wr:.2f} n={n} pnl={pnl:.2f}")
                    continue
            if n >= min_n and wr < min_wr:
                blocked.append(f"{key} wr={wr:.2f} n={n} pnl={pnl:.2f}")
                continue
            kept.append(s)
        return kept, blocked

    def _dynamic_min_confidence(self, base_min_conf: float, loss_memory: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        if not bool(self.settings.win_rate_focus_enabled):
            return base_min_conf, {"enabled": False}
        wr, n = self._recent_win_rate(loss_memory, int(self.settings.win_rate_lookback))
        target = max(0.0, min(1.0, float(self.settings.target_win_rate)))
        bump = 0.0
        if n >= 10 and wr < target:
            gap = min(0.25, target - wr)
            bump = min(float(self.settings.low_win_rate_confidence_bump), gap * 0.7)
        dynamic = max(0.0, min(1.0, float(base_min_conf) + bump))
        return dynamic, {
            "enabled": True,
            "target": round(target, 3),
            "recent_win_rate": round(wr, 3),
            "samples": int(n),
            "confidence_bump": round(bump, 4),
        }

    def _recent_execution_side_counts(self, lookback: int = 60) -> dict[str, int]:
        events = self._tail_events_cached(limit=max(80, int(lookback) * 3))
        recent_exec = [e for e in events if str(e.get("type", "")).lower() == "execution"][-max(1, int(lookback)) :]
        out = {"buy": 0, "sell": 0}
        for e in recent_exec:
            meta = dict(e.get("meta", {}) or {})
            journal = dict(meta.get("journal", {}) or {})
            side = str(journal.get("side") or dict(meta.get("chosen", {}) or {}).get("side") or "").lower()
            if side in out:
                out[side] += 1
        return out

    def _strategy_family(self, strategy_name: str) -> str:
        key = self._strategy_key(strategy_name)
        if str(key).startswith("evolved_rule("):
            return "evolved_rule"
        return key

    def _recent_execution_cluster(self, lookback: int = 12) -> dict[str, Any]:
        events = self._tail_events_cached(limit=max(80, int(lookback) * 4))
        recent_exec = [e for e in events if str(e.get("type", "")).lower() == "execution"][-max(1, int(lookback)) :]
        side_counts: dict[str, int] = {"buy": 0, "sell": 0}
        family_counts: dict[str, int] = {}
        recent_rows: list[dict[str, str]] = []
        for event in recent_exec:
            meta = dict(event.get("meta", {}) or {})
            chosen = dict(meta.get("chosen", {}) or {})
            strategy = str(chosen.get("strategy", "") or "")
            side = str(chosen.get("side", "") or "").lower()
            family = self._strategy_family(strategy)
            if side in side_counts:
                side_counts[side] += 1
            family_counts[family] = int(family_counts.get(family, 0)) + 1
            recent_rows.append({"strategy": strategy, "family": family, "side": side})
        return {
            "lookback": len(recent_exec),
            "side_counts": side_counts,
            "family_counts": family_counts,
            "recent": recent_rows,
        }

    def _execution_cluster_guard(self, chosen: TradeSignal) -> dict[str, Any]:
        if not bool(getattr(self.settings, "cluster_guard_enabled", True)):
            return {"enabled": False}
        side = str(getattr(chosen, "side", "flat"))
        strategy = str(getattr(chosen, "strategy", ""))
        if side not in {"buy", "sell"}:
            return {"enabled": True, "applied": False, "reason": "flat_side"}
        cluster = self._recent_execution_cluster(int(getattr(self.settings, "cluster_guard_lookback", 12)))
        lookback = int(cluster.get("lookback", 0))
        if lookback < 6:
            return {"enabled": True, "applied": False, "reason": "insufficient_history", "lookback": lookback}
        side_counts = dict(cluster.get("side_counts", {}) or {})
        family_counts = dict(cluster.get("family_counts", {}) or {})
        family = self._strategy_family(strategy)
        same_side = int(side_counts.get(side, 0))
        same_family = int(family_counts.get(family, 0))
        max_same_side = max(2, int(getattr(self.settings, "cluster_guard_max_same_side", 8)))
        max_same_family = max(2, int(getattr(self.settings, "cluster_guard_max_same_family", 5)))
        block = same_side >= max_same_side or same_family >= max_same_family
        return {
            "enabled": True,
            "applied": block,
            "lookback": lookback,
            "side": side,
            "family": family,
            "same_side_count": same_side,
            "same_family_count": same_family,
            "max_same_side": max_same_side,
            "max_same_family": max_same_family,
            "side_counts": side_counts,
            "family_counts": family_counts,
            "reason": (
                f"execution clustering detected for side={side} ({same_side}/{max_same_side}) "
                f"or family={family} ({same_family}/{max_same_family})"
                if block
                else "within_cluster_limits"
            ),
        }

    def _apply_side_balance_penalty(self, chosen: TradeSignal) -> tuple[float, dict[str, Any]]:
        if not bool(self.settings.side_balance_enabled):
            return float(chosen.confidence), {"enabled": False}
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat"}
        counts = self._recent_execution_side_counts(int(self.settings.side_balance_lookback))
        buy_n = int(counts.get("buy", 0))
        sell_n = int(counts.get("sell", 0))
        total = buy_n + sell_n
        if total < 12:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "insufficient_samples", "counts": counts}
        skew = (buy_n - sell_n) / max(1, total)
        max_skew = max(0.05, float(self.settings.side_balance_max_skew))
        penalty = max(0.0, min(0.35, float(self.settings.side_balance_confidence_penalty)))
        over_bought = side == "buy" and skew > max_skew
        over_sold = side == "sell" and skew < -max_skew
        if not (over_bought or over_sold):
            return float(chosen.confidence), {"enabled": True, "applied": False, "counts": counts, "skew": round(skew, 3)}
        new_conf = max(0.0, float(chosen.confidence) * (1.0 - penalty))
        return new_conf, {
            "enabled": True,
            "applied": True,
            "counts": counts,
            "skew": round(skew, 3),
            "penalty": round(penalty, 3),
            "reason": f"recent side skew favors {'buy' if skew > 0 else 'sell'}",
        }

    def _smart_context_adjustment(
        self,
        chosen: TradeSignal,
        memory: dict[str, Any],
        regime: dict[str, Any],
        higher_tf_bias: str,
        now_utc: datetime,
    ) -> tuple[float, dict[str, Any]]:
        if not bool(self.settings.smart_context_enabled):
            return float(chosen.confidence), {"enabled": False}
        side = str(getattr(chosen, "side", "flat"))
        if side not in {"buy", "sell"}:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}

        outcomes = list(memory.get("recent_outcomes", []))[-max(20, int(self.settings.smart_context_lookback)) :]
        if not outcomes:
            return float(chosen.confidence), {"enabled": True, "applied": False, "reason": "no_outcomes"}

        session = self._session_label(now_utc)
        hour = int(now_utc.hour)
        session_side = [x for x in outcomes if str(x.get("session", "")) == session and str(x.get("side", "")) == side]
        hour_side = [x for x in outcomes if int(x.get("hour_utc", -1)) == hour and str(x.get("side", "")) == side]

        def _wr(rows: list[dict[str, Any]]) -> tuple[float, int]:
            wins = sum(1 for x in rows if str(x.get("outcome", "")) == "win")
            losses = sum(1 for x in rows if str(x.get("outcome", "")) == "loss")
            n = wins + losses
            return ((wins / n) if n else 0.5), n

        wr_session, n_session = _wr(session_side)
        wr_hour, n_hour = _wr(hour_side)

        penalty = 0.0
        boost = 0.0
        reasons: list[str] = []

        if n_session >= 6 and wr_session < 0.42:
            d = min(0.18, (0.42 - wr_session) * 0.5)
            penalty += d
            reasons.append(f"session_{session}_wr={wr_session:.2f}")
        elif n_session >= 6 and wr_session > 0.58:
            d = min(0.08, (wr_session - 0.58) * 0.3)
            boost += d
            reasons.append(f"session_{session}_wr={wr_session:.2f}")

        if n_hour >= 4 and wr_hour < 0.40:
            d = min(0.12, (0.40 - wr_hour) * 0.4)
            penalty += d
            reasons.append(f"hour_{hour}_wr={wr_hour:.2f}")
        elif n_hour >= 4 and wr_hour > 0.60:
            d = min(0.05, (wr_hour - 0.60) * 0.25)
            boost += d
            reasons.append(f"hour_{hour}_wr={wr_hour:.2f}")

        if higher_tf_bias in {"buy", "sell"} and higher_tf_bias != side:
            penalty += 0.04
            reasons.append("htf_misaligned")
        if str(regime.get("name", "")) == "trend" and higher_tf_bias == side:
            boost += 0.03
            reasons.append("trend_htf_aligned")

        penalty = min(float(self.settings.smart_context_penalty_cap), penalty)
        boost = min(float(self.settings.smart_context_boost_cap), boost)
        factor = max(0.65, min(1.2, 1.0 - penalty + boost))
        new_conf = max(0.0, min(1.0, float(chosen.confidence) * factor))
        return new_conf, {
            "enabled": True,
            "applied": abs(factor - 1.0) > 1e-6,
            "factor": round(factor, 4),
            "penalty": round(penalty, 4),
            "boost": round(boost, 4),
            "session_samples": int(n_session),
            "hour_samples": int(n_hour),
            "session_win_rate": round(wr_session, 3),
            "hour_win_rate": round(wr_hour, 3),
            "reasons": reasons[:6],
        }

    def _same_side_support_count(self, signals: list[TradeSignal], side: str) -> int:
        target = str(side)
        return sum(1 for s in signals if str(getattr(s, "side", "")) == target and float(getattr(s, "confidence", 0.0) or 0.0) > 0.0)

    def _guard_block_reason(self, state: dict[str, Any]) -> str:
        if bool(state.get("safety_lock_active", False)):
            reason = str(state.get("safety_lock_reason", "") or "safety_lock_active")
            return f"Safety lock active: {reason}"
        if str(state.get("volatility_action", "allow")) == "block_new_entries":
            reason = str(state.get("volatility_reason", "") or "extreme_volatility")
            return f"Volatility protection active: {reason}"
        if state.get("trades_today", 0) >= self.settings.max_trades_per_day:
            return f"Daily trade cap reached ({state.get('trades_today')}/{self.settings.max_trades_per_day})"
        cooldown_until = str(state.get("cooldown_until", "") or "")
        if cooldown_until:
            try:
                until = datetime.fromisoformat(cooldown_until)
                if until > datetime.now(timezone.utc):
                    secs = int((until - datetime.now(timezone.utc)).total_seconds())
                    mins = max(1, secs // 60)
                    level = str(state.get("cooldown_level", "") or "adaptive")
                    reason = str(state.get("cooldown_reason", "") or "loss_protection")
                    return f"Cooldown active ({level}, {mins} min left): {reason}"
            except Exception:
                return ""
        return ""

    def _build_trade_pattern_fingerprint(
        self,
        *,
        strategy: str,
        side: str,
        setup_family: str,
        session: str,
        regime: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        ctx = dict(context or {})
        strategy_key = self._strategy_key(str(strategy or "unknown"))
        location = str(
            ctx.get("entry_location_state")
            or ctx.get("location")
            or ctx.get("entry_location")
            or "unknown"
        ).strip().lower()
        momentum = str(
            ctx.get("entry_momentum_state")
            or ctx.get("momentum")
            or ctx.get("entry_momentum")
            or "unknown"
        ).strip().lower()
        trend_state = str(
            ctx.get("entry_trend_state")
            or ctx.get("trend_state")
            or regime
            or "unknown"
        ).strip().lower()
        volatility_state = str(
            ctx.get("volatility_state")
            or ctx.get("volatility_bucket")
            or "unknown"
        ).strip().lower()
        parts = {
            "strategy": strategy_key or "unknown",
            "side": str(side or "flat").strip().lower() or "flat",
            "setup": str(setup_family or "unknown").strip().lower() or "unknown",
            "session": str(session or "unknown").strip().lower() or "unknown",
            "regime": str(regime or "unknown").strip().lower() or "unknown",
            "trend": trend_state or "unknown",
            "location": location or "unknown",
            "momentum": momentum or "unknown",
            "volatility": volatility_state or "unknown",
        }
        return "|".join(f"{key}={value}" for key, value in parts.items())

    @staticmethod
    def _execution_lesson_from_loss(classification: str, setup_family: str) -> dict[str, Any]:
        code = str(classification or "weak_structure").strip().lower() or "weak_structure"
        family = str(setup_family or "unknown").strip().lower()
        table: dict[str, dict[str, Any]] = {
            "false_continuation": {
                "correction": "Do not enter the first continuation push; wait for momentum to confirm in the trade direction after structure holds.",
                "confirmation_required": ["strong_m1_momentum", "structure_hold", "avoid_first_push"],
            },
            "weak_pullback": {
                "correction": "Wait for the pullback to reach a valid reaction zone and for M1 momentum to turn back with the setup.",
                "confirmation_required": ["valid_pullback_location", "m1_momentum_turn", "controlled_wick_pressure"],
            },
            "late_entry": {
                "correction": "Do not chase after expansion; wait for a retest, pullback, or fresh structure entry before executing.",
                "confirmation_required": ["no_overextension", "retest_or_pullback", "fresh_entry_location"],
            },
            "stop_hunt": {
                "correction": "Treat wick-heavy entries as vulnerable; wait for sweep rejection or structure reclaim before entry.",
                "confirmation_required": ["rejection_after_sweep", "structure_reclaim", "reduced_wick_pressure"],
            },
            "spread_liquidity_distortion": {
                "correction": "The idea may be valid, but execution cost is unsafe; wait for spread/liquidity to normalize.",
                "confirmation_required": ["spread_normalized", "liquidity_acceptable"],
            },
            "trend_reversal": {
                "correction": "Re-check higher-timeframe context and require reversal proof before continuing in the old direction.",
                "confirmation_required": ["htf_recheck", "reversal_or_continuation_proof"],
            },
            "poor_timing": {
                "correction": "Avoid thin or poor timing windows; wait for the next liquid session phase or cleaner momentum.",
                "confirmation_required": ["liquid_session", "cleaner_momentum"],
            },
            "news_spike": {
                "correction": "Do not execute through event-driven volatility unless the event-risk state is clear and stable.",
                "confirmation_required": ["event_risk_clear", "volatility_stabilized"],
            },
            "weak_structure": {
                "correction": "Require clearer HH/HL or LH/LL structure, non-mid-range location, and momentum alignment before entry.",
                "confirmation_required": ["clear_structure", "not_mid_range", "momentum_alignment"],
            },
        }
        lesson = dict(table.get(code, table["weak_structure"]))
        lesson.update(
            {
                "classification": code,
                "setup_family": family,
                "policy_note": "Loss lesson can guide bounded confidence and confirmation checks only; it cannot bypass gates or change risk/execution policy.",
            }
        )
        return lesson

    @staticmethod
    def _correction_confirmed(classification: str, context: dict[str, Any]) -> bool:
        code = str(classification or "").strip().lower()
        location = str(context.get("entry_location_state", context.get("location", "")) or "").lower()
        momentum = str(context.get("entry_momentum_state", context.get("momentum", "")) or "").lower()
        wickiness = float(context.get("entry_wickiness", 0.0) or 0.0)
        expansion = float(context.get("entry_expansion", 0.0) or 0.0)
        position = float(context.get("entry_position", 0.5) or 0.5)
        trend_strength = float(context.get("entry_trend_strength", context.get("trend_strength", 0.0)) or 0.0)
        if code == "weak_pullback":
            return location == "pullback" and momentum == "strong" and wickiness < 0.55
        if code == "late_entry":
            return expansion < 0.45 and 0.10 < position < 0.90 and location in {"pullback", "breakout", "consolidation"}
        if code == "false_continuation":
            return momentum == "strong" and trend_strength >= 0.18
        if code == "stop_hunt":
            return wickiness < 0.50 and momentum in {"strong", "mixed"}
        if code == "weak_structure":
            return trend_strength >= 0.18 and location != "mid_range" and momentum == "strong"
        return False

    def _update_pattern_lesson_memory(self, memory: dict[str, Any], outcome_row: dict[str, Any]) -> None:
        fingerprint = str(outcome_row.get("pattern_fingerprint", "") or "")
        if not fingerprint:
            return
        lessons = dict(memory.get("pattern_lessons", {}) or {})
        existing = dict(lessons.get(fingerprint, {}) or {})
        outcome = str(outcome_row.get("outcome", "") or "").lower()
        wins = int(existing.get("wins", 0) or 0) + (1 if outcome == "win" else 0)
        losses = int(existing.get("losses", 0) or 0) + (1 if outcome == "loss" else 0)
        samples = wins + losses
        lesson = dict(existing.get("execution_lesson", {}) or {})
        if outcome == "loss":
            lesson = dict(outcome_row.get("execution_lesson", {}) or lesson)
        post_trade_lesson = dict(outcome_row.get("post_trade_lesson", {}) or existing.get("post_trade_lesson", {}) or {})
        lessons[fingerprint] = {
            **existing,
            "fingerprint": fingerprint,
            "samples": samples,
            "wins": wins,
            "losses": losses,
            "loss_rate": round(losses / max(1, samples), 4),
            "last_seen": str(outcome_row.get("time", "") or existing.get("last_seen", "")),
            "last_outcome": outcome,
            "strategy": str(outcome_row.get("strategy", existing.get("strategy", "")) or ""),
            "side": str(outcome_row.get("side", existing.get("side", "")) or ""),
            "setup_family": str(outcome_row.get("setup_family", existing.get("setup_family", "")) or ""),
            "execution_lesson": lesson,
            "post_trade_lesson": post_trade_lesson,
            "failure_axis": str(post_trade_lesson.get("failure_axis", existing.get("failure_axis", "")) or ""),
            "idea_quality": str(post_trade_lesson.get("idea_quality", existing.get("idea_quality", "")) or ""),
            "timing_quality": str(post_trade_lesson.get("timing_quality", existing.get("timing_quality", "")) or ""),
            "bounded_learning_only": True,
        }
        memory["pattern_lessons"] = dict(list(lessons.items())[-250:])

    @staticmethod
    def _parse_trade_pattern_fingerprint(fingerprint: str) -> dict[str, str]:
        parts: dict[str, str] = {}
        for item in str(fingerprint or "").split("|"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            parts[str(key or "").strip()] = str(value or "").strip().lower()
        return parts

    def _loss_lesson_match_score(
        self,
        *,
        current_fingerprint: str,
        lesson: dict[str, Any],
    ) -> dict[str, Any]:
        lesson_fingerprint = str(lesson.get("fingerprint", "") or "")
        current_parts = self._parse_trade_pattern_fingerprint(current_fingerprint)
        lesson_parts = self._parse_trade_pattern_fingerprint(lesson_fingerprint)
        if not lesson_parts:
            return {"score": 0.0, "strength": "not_enough_evidence", "matched_fields": [], "different_fields": []}
        if lesson_fingerprint == current_fingerprint:
            return {
                "score": 1.0,
                "strength": "exact_match",
                "matched_fields": sorted(current_parts.keys()),
                "different_fields": [],
            }

        required = ["strategy", "side"]
        if any(current_parts.get(field) != lesson_parts.get(field) for field in required):
            return {"score": 0.0, "strength": "not_enough_evidence", "matched_fields": [], "different_fields": required}

        weights = {
            "strategy": 0.30,
            "side": 0.25,
            "setup": 0.15,
            "session": 0.10,
            "regime": 0.08,
            "trend": 0.04,
            "location": 0.03,
            "momentum": 0.03,
            "volatility": 0.02,
        }
        score = 0.0
        matched: list[str] = []
        different: list[str] = []
        unknown: list[str] = []
        for field, weight in weights.items():
            cur = str(current_parts.get(field, "") or "")
            old = str(lesson_parts.get(field, "") or "")
            if not cur or not old or cur == "unknown" or old == "unknown":
                unknown.append(field)
                continue
            if cur == old:
                score += weight
                matched.append(field)
            else:
                different.append(field)

        if score >= 0.80:
            strength = "strong_partial_match"
        elif score >= 0.55:
            strength = "weak_partial_match"
        else:
            strength = "not_enough_evidence"
        return {
            "score": round(score, 4),
            "strength": strength,
            "matched_fields": matched,
            "different_fields": different,
            "unknown_fields": unknown,
        }

    def _find_loss_pattern_lesson(
        self,
        *,
        fingerprint: str,
        lessons: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        exact = dict(lessons.get(fingerprint, {}) or {})
        if exact:
            return exact, {
                "matched_fingerprint": fingerprint,
                "match_strength": "exact_match",
                "match_score": 1.0,
                "matched_fields": sorted(self._parse_trade_pattern_fingerprint(fingerprint).keys()),
                "different_fields": [],
                "unknown_fields": [],
            }

        best_lesson: dict[str, Any] = {}
        best_match: dict[str, Any] = {}
        for raw in lessons.values():
            lesson = dict(raw or {})
            match = self._loss_lesson_match_score(current_fingerprint=fingerprint, lesson=lesson)
            strength = str(match.get("strength", "") or "")
            if strength == "not_enough_evidence":
                continue
            samples = int(lesson.get("samples", 0) or 0)
            loss_rate = float(lesson.get("loss_rate", 0.0) or 0.0)
            ranking = (float(match.get("score", 0.0) or 0.0), loss_rate, samples)
            current = (
                float(best_match.get("match_score", 0.0) or 0.0),
                float(best_lesson.get("loss_rate", 0.0) or 0.0),
                int(best_lesson.get("samples", 0) or 0),
            )
            if ranking > current:
                best_lesson = lesson
                best_match = {
                    "matched_fingerprint": str(lesson.get("fingerprint", "") or ""),
                    "match_strength": strength,
                    "match_score": float(match.get("score", 0.0) or 0.0),
                    "matched_fields": list(match.get("matched_fields", []) or []),
                    "different_fields": list(match.get("different_fields", []) or []),
                    "unknown_fields": list(match.get("unknown_fields", []) or []),
                }
        return best_lesson, best_match

    def _loss_pattern_lesson_adjustment(self, signal: TradeSignal, context: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        side = str(getattr(signal, "side", "flat") or "flat").lower()
        if side not in {"buy", "sell"}:
            return float(signal.confidence), {"enabled": True, "applied": False, "reason": "flat_side"}
        try:
            memory = self._load_loss_memory()
        except Exception:
            return float(signal.confidence), {"enabled": True, "applied": False, "reason": "memory_unavailable"}
        fingerprint = self._build_trade_pattern_fingerprint(
            strategy=str(getattr(signal, "strategy", "") or ""),
            side=side,
            setup_family=self._strategy_to_setup_family(str(getattr(signal, "strategy", "") or "")),
            session=str(context.get("session", context.get("entry_session", "unknown")) or "unknown"),
            regime=str(context.get("regime", context.get("entry_trend_state", "unknown")) or "unknown"),
            context=context,
        )
        lesson, match_meta = self._find_loss_pattern_lesson(
            fingerprint=fingerprint,
            lessons=dict(memory.get("pattern_lessons", {}) or {}),
        )
        if not lesson:
            return float(signal.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "no_matching_pattern_lesson",
                "fingerprint": fingerprint,
                "match_strength": "not_enough_evidence",
                "match_score": 0.0,
            }
        samples = int(lesson.get("samples", 0) or 0)
        losses = int(lesson.get("losses", 0) or 0)
        loss_rate = float(lesson.get("loss_rate", 0.0) or 0.0)
        if samples < 2 or losses < 1 or loss_rate < 0.50:
            return float(signal.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "lesson_not_loss_dominant",
                "fingerprint": fingerprint,
                **match_meta,
                "samples": samples,
                "loss_rate": round(loss_rate, 4),
            }
        execution_lesson = dict(lesson.get("execution_lesson", {}) or {})
        post_trade_lesson = dict(lesson.get("post_trade_lesson", {}) or {})
        classification = str(execution_lesson.get("classification", "") or "")
        confirmed = self._correction_confirmed(classification, context)
        if confirmed:
            return float(signal.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "corrective_confirmation_present",
                "fingerprint": fingerprint,
                **match_meta,
                "execution_lesson": execution_lesson,
                "post_trade_lesson": post_trade_lesson,
                "decision_critic": {
                    "status": "cleared_by_current_confirmation",
                    "match_strength": str(match_meta.get("match_strength", "exact_match") or "exact_match"),
                    "match_score": float(match_meta.get("match_score", 1.0) or 1.0),
                    "previous_failure_axis": str(post_trade_lesson.get("failure_axis", "") or ""),
                    "idea_quality": str(post_trade_lesson.get("idea_quality", "") or ""),
                    "timing_quality": str(post_trade_lesson.get("timing_quality", "") or ""),
                },
            }
        failure_axis = str(post_trade_lesson.get("failure_axis", "") or "")
        extra = 0.0
        if failure_axis in {"timing", "location_or_structure", "confirmation_quality"}:
            extra = 0.015
        elif failure_axis in {"execution_friction", "higher_timeframe_conflict"}:
            extra = 0.01
        haircut = min(0.12, 0.03 + min(0.07, loss_rate * 0.06) + extra)
        match_strength = str(match_meta.get("match_strength", "exact_match") or "exact_match")
        if match_strength == "strong_partial_match":
            haircut *= 0.75
        elif match_strength == "weak_partial_match":
            return float(signal.confidence), {
                "enabled": True,
                "applied": False,
                "reason": "weak_partial_match_observability_only",
                "fingerprint": fingerprint,
                **match_meta,
                "samples": samples,
                "loss_rate": round(loss_rate, 4),
                "execution_lesson": execution_lesson,
                "post_trade_lesson": post_trade_lesson,
                "decision_critic": {
                    "status": "weak_partial_loss_fingerprint_seen",
                    "match_strength": match_strength,
                    "match_score": float(match_meta.get("match_score", 0.0) or 0.0),
                    "previous_failure_axis": failure_axis,
                    "idea_quality": str(post_trade_lesson.get("idea_quality", "") or ""),
                    "timing_quality": str(post_trade_lesson.get("timing_quality", "") or ""),
                    "next_time_action": str(post_trade_lesson.get("next_time_action", "") or ""),
                    "future_recognition": list(post_trade_lesson.get("future_recognition", []) or [])[:8],
                },
                "policy_note": "Weak partial lesson match is observability only; it does not reduce confidence or change execution policy.",
            }
        return max(0.0, float(signal.confidence) * (1.0 - haircut)), {
            "enabled": True,
            "applied": True,
            "fingerprint": fingerprint,
            **match_meta,
            "haircut": round(haircut, 4),
            "factor": round(1.0 - haircut, 4),
            "samples": samples,
            "loss_rate": round(loss_rate, 4),
            "execution_lesson": execution_lesson,
            "post_trade_lesson": post_trade_lesson,
            "decision_critic": {
                "status": "similar_weak_fingerprint_seen",
                "match_strength": match_strength,
                "match_score": float(match_meta.get("match_score", 1.0) or 1.0),
                "previous_failure_axis": failure_axis,
                "idea_quality": str(post_trade_lesson.get("idea_quality", "") or ""),
                "timing_quality": str(post_trade_lesson.get("timing_quality", "") or ""),
                "next_time_action": str(post_trade_lesson.get("next_time_action", "") or ""),
                "future_recognition": list(post_trade_lesson.get("future_recognition", []) or [])[:8],
            },
            "policy_note": "Pattern lesson teaches confirmation timing only; it cannot force entries, bypass gates, or alter risk policy.",
        }

    def _learning_critic_snapshot(self, signal: TradeSignal, context: dict[str, Any]) -> dict[str, Any]:
        _confidence, meta = self._loss_pattern_lesson_adjustment(signal, context)
        critic = dict(meta.get("decision_critic", {}) or {})
        return {
            "enabled": bool(meta.get("enabled", True)),
            "applied": bool(meta.get("applied", False)),
            "status": str(critic.get("status") or meta.get("reason") or "watching"),
            "fingerprint": str(meta.get("fingerprint", "") or ""),
            "matched_fingerprint": str(meta.get("matched_fingerprint", meta.get("fingerprint", "")) or ""),
            "match_strength": str(critic.get("match_strength") or meta.get("match_strength") or "not_enough_evidence"),
            "match_score": round(float(critic.get("match_score", meta.get("match_score", 0.0)) or 0.0), 4),
            "matched_fields": list(meta.get("matched_fields", []) or [])[:10],
            "different_fields": list(meta.get("different_fields", []) or [])[:10],
            "unknown_fields": list(meta.get("unknown_fields", []) or [])[:10],
            "haircut": float(meta.get("haircut", 0.0) or 0.0),
            "samples": int(meta.get("samples", 0) or 0),
            "loss_rate": round(float(meta.get("loss_rate", 0.0) or 0.0), 4),
            "previous_failure_axis": str(critic.get("previous_failure_axis", "") or ""),
            "idea_quality": str(critic.get("idea_quality", "") or ""),
            "timing_quality": str(critic.get("timing_quality", "") or ""),
            "next_time_action": str(critic.get("next_time_action", "") or ""),
            "future_recognition": list(critic.get("future_recognition", []) or [])[:8],
            "policy_note": str(
                meta.get(
                    "policy_note",
                    "Learning critic is observability only unless an existing bounded lesson adjustment applies.",
                )
                or ""
            ),
        }

    def _timing_readiness_snapshot(self, learning_critic: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        failure_axis = str(learning_critic.get("previous_failure_axis", "") or learning_critic.get("failure_axis", "") or "")
        next_time_action = str(learning_critic.get("next_time_action", "") or "")
        location = str(
            context.get("entry_location_state", context.get("location_state", context.get("location", ""))) or ""
        ).strip().lower()
        momentum = str(
            context.get("entry_momentum_state", context.get("momentum_state", context.get("momentum", ""))) or ""
        ).strip().lower()
        radar = dict(context.get("setup_radar", {}) or {})
        radar_phase = str(radar.get("phase", "") or "").strip().lower()
        trigger_state = str(radar.get("trigger_state", "") or "").strip().lower()
        missing: list[str] = []
        fresh_location = location in {"pullback", "retest", "fresh_retest", "reaction_zone"} or "retest" in radar_phase
        momentum_turning = momentum in {"strong", "turning", "impulsive", "aligned"} or trigger_state in {"armed", "confirmed", "triggered"}
        if not fresh_location:
            missing.append("fresh_entry_or_retest")
        if not momentum_turning:
            missing.append("strong_or_turning_m1_momentum")
        if failure_axis != "timing":
            status = "no_timing_failure_axis"
        elif missing:
            status = "timing_not_ready"
        else:
            status = "timing_ready_for_normal_gates"
        return {
            "enabled": True,
            "status": status,
            "failure_axis": failure_axis,
            "timing_quality": str(learning_critic.get("timing_quality", "") or ""),
            "next_time_action": next_time_action,
            "fresh_location_or_retest": fresh_location,
            "momentum_turning_or_strong": momentum_turning,
            "missing": missing,
            "observed": {
                "location": location or "unknown",
                "momentum": momentum or "unknown",
                "radar_phase": radar_phase or "unknown",
                "trigger_state": trigger_state or "unknown",
            },
            "recommendation": (
                "wait_for_fresher_entry_or_retest"
                if status == "timing_not_ready"
                else "timing_clear_for_existing_gates"
                if status == "timing_ready_for_normal_gates"
                else "monitor_timing_lesson"
            ),
            "policy": {
                "observability_only": True,
                "does_not_execute": True,
                "does_not_change_strategy_logic": True,
                "does_not_change_risk_policy": True,
                "does_not_change_execution_policy": True,
            },
        }

    def _timing_discipline_snapshot(
        self,
        signal: TradeSignal,
        context: dict[str, Any],
        status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status = dict(status or {})
        chart = dict(status.get("chart_features", {}) or {})
        visual = dict(status.get("visual_chart_read", {}) or {})
        visual_features = dict(visual.get("visual_features", {}) or {})
        setup_quality = dict(status.get("setup_quality_observation", {}) or {})
        loss_summary = dict(dict(status.get("learning_summary", {}) or {}).get("loss_reason_summary", {}) or {})
        top_loss_reason = str(loss_summary.get("top_reason", "") or "").strip().lower()
        visual_location = str(visual_features.get("visual_location", "") or "").strip().lower()
        visual_momentum = str(visual_features.get("visual_momentum", "") or "").strip().lower()
        visual_direction = str(visual_features.get("visual_direction", "") or "").strip().lower()
        visual_confirmation = str(visual.get("visual_confirmation", "") or "").strip().lower()
        location = str(
            context.get("entry_location_state")
            or context.get("location_state")
            or context.get("location")
            or visual_location
            or chart.get("location_state")
            or ""
        ).strip().lower()
        momentum = str(
            context.get("entry_momentum_state")
            or context.get("momentum_state")
            or context.get("momentum")
            or visual_momentum
            or chart.get("momentum_state")
            or ""
        ).strip().lower()
        trend_state = str(
            context.get("entry_trend_state") or context.get("trend_state") or chart.get("trend_state") or ""
        ).strip().lower()
        session = str(context.get("session", "") or "").strip().lower()
        setup_hint = str(context.get("setup_type_hint", chart.get("setup_type_hint", "")) or "").strip().lower()
        try:
            trend_strength = float(context.get("trend_strength", chart.get("trend_strength", 0.0)) or 0.0)
        except Exception:
            trend_strength = 0.0

        fresh_location = location in {
            "pullback",
            "retest",
            "fresh_retest",
            "reaction_zone",
            "range_edge",
            "sweep_rejection",
            "structure_reclaim",
        }
        mid_range = location in {"mid_range", "middle", "consolidation", "unknown", ""}
        momentum_confirmed = momentum in {"strong", "turning", "impulsive", "aligned"}
        timing_settings = getattr(self, "settings", None)
        mid_range_strong_allowed = bool(
            getattr(timing_settings, "timing_allow_mid_range_strong_momentum", False)
            and mid_range
            and momentum == "strong"
        )
        structure_hold = trend_strength >= 0.18 or trend_state in {"trend", "uptrend", "downtrend", "bullish", "bearish"}
        stop_hunt_protected = location in {"sweep_rejection", "structure_reclaim", "retest", "fresh_retest"}
        continuation_like = any(token in setup_hint for token in ("continuation", "pullback", "impulse", "trend"))

        missing: list[str] = []
        reasons: list[str] = []
        if mid_range and not mid_range_strong_allowed:
            missing.append("tradable_location")
            reasons.append("entry_location_is_mid_range_or_unclear")
        elif not fresh_location and not mid_range_strong_allowed:
            missing.append("fresh_entry_or_retest")
            reasons.append("entry_location_is_not_fresh_retest_or_reaction_zone")
        elif mid_range_strong_allowed:
            reasons.append("mid_range_accepted_by_strong_momentum")
        if not momentum_confirmed:
            missing.append("strong_or_turning_m1_momentum")
            reasons.append("m1_momentum_not_confirmed")
        if continuation_like and not structure_hold:
            missing.append("structure_hold")
            reasons.append("continuation_structure_not_confirmed")
        if top_loss_reason == "stop_hunt" and not stop_hunt_protected:
            missing.append("sweep_rejection_or_structure_reclaim")
            reasons.append("stop_hunt_history_requires_rejection_or_reclaim")
        if session == "new_york" and not momentum_confirmed:
            missing.append("new_york_extra_momentum_confirmation")
            reasons.append("new_york_recent_performance_requires_stronger_confirmation")
        if bool(visual.get("visual_watch_used_for_decision", False)):
            if visual_confirmation == "does_not_confirm":
                missing.append("visual_confirmation")
                reasons.append("continuous_visual_watch_does_not_confirm_setup")
            if visual_location == "mid_range":
                missing.append("visual_tradable_location")
                reasons.append("continuous_visual_watch_sees_mid_range")
            if visual_momentum in {"weak", "unknown"}:
                missing.append("visual_momentum_confirmation")
                reasons.append("continuous_visual_watch_lacks_momentum_confirmation")
            if visual_direction == "sideways":
                missing.append("visual_directional_impulse")
                reasons.append("continuous_visual_watch_sees_sideways_action")

        unique_missing = list(dict.fromkeys(missing))
        score = 100
        score -= 0 if mid_range_strong_allowed else (35 if mid_range else 0)
        score -= 25 if not momentum_confirmed else 0
        score -= 15 if not fresh_location and not mid_range else 0
        score -= 15 if continuation_like and not structure_hold else 0
        score -= 10 if top_loss_reason == "stop_hunt" and not stop_hunt_protected else 0
        score -= 15 if visual_confirmation == "does_not_confirm" else 0
        score = max(0, min(100, score))
        hard_block = bool(str(getattr(signal, "side", "flat")) in {"buy", "sell"} and unique_missing)
        structured_result = self._build_timing_discipline_result(
            passed=not hard_block,
            side=str(getattr(signal, "side", "flat") or "flat"),
            score=int(score),
            missing=unique_missing,
            reasons=reasons[:8],
            observed={
                "location": location or "unknown",
                "momentum": momentum or "unknown",
                "trend_state": trend_state or "unknown",
                "trend_strength": round(trend_strength, 4),
                "session": session or "unknown",
                "setup_hint": setup_hint or "unknown",
                "top_loss_reason": top_loss_reason or "unknown",
                "visual_watch_used": bool(visual.get("visual_watch_used_for_decision", False)),
                "visual_confirmation": visual_confirmation or "unknown",
                "visual_direction": visual_direction or "unknown",
                "visual_location": visual_location or "unknown",
                "visual_momentum": visual_momentum or "unknown",
            },
        )

        return {
            "enabled": True,
            "status": "timing_block" if hard_block else "timing_clear_for_existing_gates",
            "score": int(score),
            "side": str(getattr(signal, "side", "flat") or "flat"),
            "missing": unique_missing,
            "reasons": reasons[:8],
            "observed": dict(structured_result.get("condition_values", {}).get("observed", {}) or {}),
            "requirements": [
                "fresh_entry_or_retest",
                "strong_or_turning_m1_momentum",
                "avoid_mid_range",
                "structure_hold_for_continuation",
                "sweep_rejection_or_structure_reclaim_when_stop_hunt_risk_is_dominant",
            ],
            "hard_block": hard_block,
            "passed": not hard_block,
            "check": "timing_discipline",
            "reason_code": structured_result.get("reason_code"),
            "detail": structured_result.get("detail"),
            "condition_values": dict(structured_result.get("condition_values", {}) or {}),
            "checked_at": structured_result.get("checked_at"),
            "structured_result": structured_result,
            "recommendation": "wait_for_clean_timing_confirmation" if hard_block else "allow_existing_gates_to_decide",
            "policy": {
                "bounded_timing_gate": True,
                "does_not_change_risk_policy": True,
                "does_not_lower_thresholds": True,
                "does_not_force_entries": True,
                "operator_requested": True,
            },
        }

    @staticmethod
    def _timing_reason_code(missing: list[str]) -> str | None:
        if not missing:
            return None
        priority = [
            ("tradable_location", "AWAITING_TRADABLE_LOCATION"),
            ("fresh_entry_or_retest", "AWAITING_RETEST_CONFIRMATION"),
            ("strong_or_turning_m1_momentum", "AWAITING_M1_MOMENTUM_CONFIRMATION"),
            ("structure_hold", "AWAITING_STRUCTURE_HOLD_CONFIRMATION"),
            ("sweep_rejection_or_structure_reclaim", "AWAITING_SWEEP_REJECTION_OR_RECLAIM"),
            ("new_york_extra_momentum_confirmation", "AWAITING_NEW_YORK_MOMENTUM_CONFIRMATION"),
            ("visual_confirmation", "AWAITING_VISUAL_CONFIRMATION"),
            ("visual_tradable_location", "AWAITING_VISUAL_TRADABLE_LOCATION"),
            ("visual_momentum_confirmation", "AWAITING_VISUAL_MOMENTUM_CONFIRMATION"),
            ("visual_directional_impulse", "AWAITING_VISUAL_DIRECTIONAL_IMPULSE"),
        ]
        missing_set = set(missing)
        for key, code in priority:
            if key in missing_set:
                return code
        return "AWAITING_TIMING_CONFIRMATION"

    @staticmethod
    def _format_timing_missing(missing: list[str]) -> str:
        labels = {
            "tradable_location": "tradable location",
            "fresh_entry_or_retest": "fresh entry or retest",
            "strong_or_turning_m1_momentum": "strong/turning M1 momentum",
            "structure_hold": "structure hold",
            "sweep_rejection_or_structure_reclaim": "sweep rejection or structure reclaim",
            "new_york_extra_momentum_confirmation": "New York momentum confirmation",
            "visual_confirmation": "visual confirmation",
            "visual_tradable_location": "visual tradable location",
            "visual_momentum_confirmation": "visual momentum confirmation",
            "visual_directional_impulse": "visual directional impulse",
        }
        readable = [labels.get(str(item), str(item).replace("_", " ")) for item in missing]
        if not readable:
            return "no timing blockers"
        if len(readable) == 1:
            return readable[0]
        return ", ".join(readable[:-1]) + f", and {readable[-1]}"

    @classmethod
    def _build_timing_discipline_result(
        cls,
        *,
        passed: bool,
        side: str,
        score: int,
        missing: list[str],
        reasons: list[str],
        observed: dict[str, Any],
    ) -> dict[str, Any]:
        reason_code = None if passed else cls._timing_reason_code(missing)
        missing_text = cls._format_timing_missing(missing)
        if passed:
            detail = (
                "Timing discipline passed: clean entry timing is present enough for existing gates "
                f"(location={observed.get('location', 'unknown')}, momentum={observed.get('momentum', 'unknown')})."
            )
        else:
            detail = (
                "Waiting for clean entry timing: "
                f"{missing_text} missing "
                f"(location={observed.get('location', 'unknown')}, momentum={observed.get('momentum', 'unknown')}, "
                f"session={observed.get('session', 'unknown')})."
            )
        return {
            "schema": "atlas_timing_discipline_result_v1",
            "passed": bool(passed),
            "check": "timing_discipline",
            "reason_code": reason_code,
            "detail": detail,
            "condition_values": {
                "side": str(side or "flat"),
                "score": int(score),
                "missing": list(missing),
                "reasons": list(reasons),
                "observed": dict(observed),
            },
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _operator_message_from_timing_discipline(timing_discipline: dict[str, Any]) -> str:
        structured = dict(timing_discipline.get("structured_result", {}) or timing_discipline)
        detail = str(structured.get("detail", "") or "").strip()
        if detail:
            return f"Timing discipline blocked trade: {detail}"
        missing = list(timing_discipline.get("missing", []) or [])
        return "Timing discipline blocked trade: waiting for clean entry timing confirmation" + (
            f" ({AtlasTrader._format_timing_missing([str(x) for x in missing])} missing)." if missing else "."
        )

    def _volatility_safety_state(self, *, context_snapshot: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
        vix_regime = str(context_snapshot.get("vix_regime", context_snapshot.get("vix_state", "")) or "").strip().lower()
        vol_pct = float(regime.get("volatility_pct", 0.0) or 0.0)
        if vix_regime in {"extreme", "panic"} or vol_pct >= 0.07:
            return {
                "volatility_state": "extreme",
                "volatility_reason": "extreme volatility regime; new entries paused",
                "volatility_action": "block_new_entries",
            }
        if vix_regime in {"high", "elevated", "stress", "fear"} or vol_pct >= 0.045:
            return {
                "volatility_state": "elevated",
                "volatility_reason": "elevated volatility; stricter thresholds and reduced size apply",
                "volatility_action": "reduce_aggressiveness",
            }
        return {
            "volatility_state": "normal",
            "volatility_reason": "volatility within acceptable bounds",
            "volatility_action": "allow",
        }

    def _recent_outcomes(self, memory: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
        rows = list(dict(memory or {}).get("recent_outcomes", []) or [])
        filtered = [dict(r) for r in rows if str(dict(r).get("outcome", "")).strip().lower() in {"win", "loss", "draw"}]
        return filtered[-max(1, int(limit)) :]

    def _loss_streak_from_outcomes(self, rows: list[dict[str, Any]]) -> int:
        streak = 0
        for row in reversed(rows):
            if str(row.get("outcome", "")).lower() == "loss":
                streak += 1
            else:
                break
        return streak

    def _rolling_expectancy(self, rows: list[dict[str, Any]]) -> float:
        pnl_rows = [float(r.get("net_pnl_zar", 0.0) or 0.0) for r in rows if "net_pnl_zar" in r]
        if not pnl_rows:
            return 0.0
        return float(sum(pnl_rows) / max(1, len(pnl_rows)))

    def _evaluate_safety_layer(
        self,
        *,
        snapshot: RiskSnapshot,
        regime: dict[str, Any],
        context_snapshot: dict[str, Any],
        guard_state: dict[str, Any],
        loss_memory: dict[str, Any],
        runtime_flags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        recent = self._recent_outcomes(loss_memory, limit=20)

        # FIX: prevent stale undated historical outcomes from keeping a perpetual lock.
        # Use dated recent outcomes for safety-lock evidence; only fall back to undated rows
        # when there is active current-session loss pressure.
        recent_window_hours_raw = str(os.getenv("ATLAS_SAFETY_LOCK_RECENT_HOURS", "72")).strip()
        try:
            recent_window_hours = max(6.0, min(240.0, float(recent_window_hours_raw)))
        except Exception:
            recent_window_hours = 72.0

        def _row_time(row: dict[str, Any]) -> datetime | None:
            for key in ("closed_at", "close_time", "time", "timestamp", "updated_at"):
                raw = row.get(key)
                if raw in (None, ""):
                    continue
                try:
                    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.astimezone(timezone.utc)
                except Exception:
                    continue
            return None

        dated_recent: list[dict[str, Any]] = []
        undated_recent: list[dict[str, Any]] = []
        for row in recent:
            ts = _row_time(dict(row))
            if ts is None:
                undated_recent.append(dict(row))
                continue
            age_hours = (now - ts).total_seconds() / 3600.0
            if age_hours <= recent_window_hours:
                dated_recent.append(dict(row))

        has_current_session_pressure = (
            int(guard_state.get("trades_today", 0) or 0) > 0
            or int(guard_state.get("consecutive_losses", 0) or 0) > 0
            or float(snapshot.daily_loss_pct or 0.0) > 0.0
        )
        recent_for_lock = list(dated_recent[-20:])
        if not recent_for_lock and has_current_session_pressure:
            recent_for_lock = list(undated_recent[-20:])

        streak_recent = self._loss_streak_from_outcomes(recent_for_lock)
        expectancy = self._rolling_expectancy(recent_for_lock[-12:])
        losses = sum(1 for r in recent_for_lock if str(r.get("outcome", "")).lower() == "loss")
        wins = sum(1 for r in recent_for_lock if str(r.get("outcome", "")).lower() == "win")
        sample = max(1, wins + losses)
        win_rate = wins / sample
        loss_class_counts: dict[str, int] = {}
        for row in recent_for_lock[-12:]:
            cls = str(row.get("loss_classification", "") or "").strip().lower()
            if cls:
                loss_class_counts[cls] = int(loss_class_counts.get(cls, 0)) + 1
        dominant_loss_class = max(loss_class_counts.items(), key=lambda kv: kv[1])[0] if loss_class_counts else ""
        dominant_loss_count = int(loss_class_counts.get(dominant_loss_class, 0)) if dominant_loss_class else 0

        volatility = self._volatility_safety_state(context_snapshot=context_snapshot, regime=regime)
        guard_state["volatility_state"] = str(volatility["volatility_state"])
        guard_state["volatility_reason"] = str(volatility["volatility_reason"])
        guard_state["volatility_action"] = str(volatility["volatility_action"])

        lock_reasons: list[str] = []
        if max(int(guard_state.get("consecutive_losses", 0) or 0), streak_recent) >= max(3, int(self.settings.max_consecutive_losses)):
            lock_reasons.append("consecutive_loss_threshold")
        if float(snapshot.daily_loss_pct) >= max(0.012, float(self.settings.max_daily_drawdown) * 0.9):
            lock_reasons.append("rolling_drawdown_limit")
        if len(recent_for_lock) >= 8 and expectancy <= -60.0:
            lock_reasons.append("rolling_expectancy_degradation")
        if dominant_loss_class in {"news_spike", "spread_liquidity_distortion", "trend_reversal"} and dominant_loss_count >= 4:
            lock_reasons.append(f"loss_cluster_{dominant_loss_class}")
        if len(recent_for_lock) >= 10 and win_rate < 0.35:
            lock_reasons.append("supporting_win_rate_weak")

        existing_lock_reason = str(guard_state.get("safety_lock_reason", "") or "")
        manual_lock_active = bool(guard_state.get("safety_lock_active", False)) and (
            "manual" in existing_lock_reason.lower() or existing_lock_reason.strip().lower() == "manual"
        )
        lock_active = bool(guard_state.get("safety_lock_active", False))
        manual_unlock_requested = bool(dict(runtime_flags or {}).get("safety_unlock_requested", False))
        if manual_unlock_requested:
            # FIX: honor explicit operator unlock requests (UI endpoint requires confirmation)
            # without bypassing independent guardrails such as spread/session/volatility checks.
            lock_reasons = []
            guard_state["safety_lock_manual_unlock_at"] = now.isoformat()
            guard_state["safety_lock_manual_unlock_reason"] = str(
                dict(runtime_flags or {}).get("safety_unlock_reason", "operator_requested_unlock") or "operator_requested_unlock"
            )
            if isinstance(runtime_flags, dict):
                runtime_flags["safety_unlock_requested"] = False
                runtime_flags["safety_unlock_consumed_at"] = now.isoformat()
        if manual_lock_active and not manual_unlock_requested:
            lock_active = True
            if not lock_reasons:
                lock_reasons.append("manual_lock_active")
        elif lock_reasons:
            if not lock_active:
                guard_state["safety_lock_started_at"] = now.isoformat()
            lock_active = True
        else:
            # No active lock evidence in the current evaluation window -> unlock safely.
            lock_active = False
            guard_state["safety_lock_reason"] = ""
        guard_state["safety_lock_active"] = bool(lock_active)
        guard_state["safety_lock_reason"] = ", ".join(lock_reasons[:3]) if lock_reasons else str(guard_state.get("safety_lock_reason", "") or "")
        guard_state["safety_lock_scope"] = "new_entries"
        guard_state["recovery_requirements"] = [
            "manual_operator_unlock_required",
            "cooldown_expired",
            "no_active_hard_blocker",
            "volatility_back_to_acceptable",
            "next_setup_meets_high_confluence",
        ]

        cooldown_level = "none"
        cooldown_reason = ""
        cooldown_minutes = 0
        if not lock_active:
            if max(int(guard_state.get("consecutive_losses", 0) or 0), streak_recent) >= 3:
                cooldown_level = "high"
                cooldown_minutes = 120
                cooldown_reason = "repeated_losses"
            elif max(int(guard_state.get("consecutive_losses", 0) or 0), streak_recent) == 2:
                cooldown_level = "medium"
                cooldown_minutes = 45
                cooldown_reason = "loss_streak"
            elif max(int(guard_state.get("consecutive_losses", 0) or 0), streak_recent) == 1:
                cooldown_level = "short"
                cooldown_minutes = 12
                cooldown_reason = "single_loss_cooldown"
            if dominant_loss_class in {"news_spike", "spread_liquidity_distortion"}:
                cooldown_minutes = max(cooldown_minutes, 30)
                if cooldown_reason:
                    cooldown_reason = f"{cooldown_reason}+{dominant_loss_class}"
                else:
                    cooldown_reason = dominant_loss_class

        if cooldown_minutes > 0:
            target_until = now + timedelta(minutes=cooldown_minutes)
            existing_raw = str(guard_state.get("cooldown_until", "") or "")
            try:
                existing = datetime.fromisoformat(existing_raw) if existing_raw else None
            except Exception:
                existing = None
            cooldown_marker = (
                f"losses={max(int(guard_state.get('consecutive_losses', 0) or 0), streak_recent)}|"
                f"level={cooldown_level}|reason={cooldown_reason}"
            )
            existing_marker = str(guard_state.get("cooldown_marker", "") or "")
            consumed_marker = str(guard_state.get("cooldown_consumed_marker", "") or "")
            level_rank = {"none": 0, "short": 1, "medium": 2, "high": 3}
            existing_level = str(guard_state.get("cooldown_level", "none") or "none")
            marker_changed = bool(existing_marker) and cooldown_marker != existing_marker
            should_arm = (
                (existing is None and consumed_marker != cooldown_marker)
                or marker_changed
                or (
                    existing is not None
                    and existing > now
                    and level_rank.get(cooldown_level, 0) > level_rank.get(existing_level, 0)
                )
            )
            if should_arm:
                guard_state["cooldown_until"] = target_until.isoformat()
                guard_state["cooldown_marker"] = cooldown_marker
                guard_state["cooldown_level"] = cooldown_level
                guard_state["cooldown_reason"] = cooldown_reason
            elif not existing_marker and existing is not None and existing > now:
                guard_state["cooldown_marker"] = cooldown_marker
            elif existing is not None and existing <= now:
                guard_state["cooldown_until"] = ""
                guard_state["cooldown_level"] = "none"
                guard_state["cooldown_reason"] = ""
                guard_state["cooldown_consumed_marker"] = cooldown_marker
        elif not str(guard_state.get("cooldown_until", "") or "").strip():
            guard_state["cooldown_level"] = "none"
            guard_state["cooldown_reason"] = ""

        cooldown_active = False
        try:
            until = datetime.fromisoformat(str(guard_state.get("cooldown_until", "") or ""))
            cooldown_active = until > now
        except Exception:
            cooldown_active = False

        recovery_blockers: list[str] = []
        if lock_active:
            recovery_blockers.append("safety_lock_active")
        if cooldown_active:
            recovery_blockers.append("cooldown_active")
        if str(volatility.get("volatility_action", "")) == "block_new_entries":
            recovery_blockers.append("extreme_volatility")
        recovery_ready = not recovery_blockers
        recovery_mode = "normal" if recovery_ready else ("lockdown" if lock_active else "cooldown")

        return {
            "safety_lock_active": bool(lock_active),
            "safety_lock_reason": str(guard_state.get("safety_lock_reason", "") or ""),
            "safety_lock_started_at": str(guard_state.get("safety_lock_started_at", "") or ""),
            "safety_lock_scope": "new_entries",
            "recovery_requirements": list(guard_state.get("recovery_requirements", [])),
            "cooldown_active": bool(cooldown_active),
            "cooldown_reason": str(guard_state.get("cooldown_reason", "") or ""),
            "cooldown_until": str(guard_state.get("cooldown_until", "") or ""),
            "cooldown_level": str(guard_state.get("cooldown_level", "none") or "none"),
            "volatility_state": str(volatility["volatility_state"]),
            "volatility_reason": str(volatility["volatility_reason"]),
            "volatility_action": str(volatility["volatility_action"]),
            "loss_streak": int(max(int(guard_state.get("consecutive_losses", 0) or 0), streak_recent)),
            "rolling_drawdown_pct": round(float(snapshot.daily_loss_pct), 4),
            "rolling_expectancy_zar": round(float(expectancy), 2),
            "supporting_win_rate": round(float(win_rate), 4),
            "dominant_loss_class": dominant_loss_class or "none",
            "dominant_loss_class_count": int(dominant_loss_count),
            "recovery_ready": bool(recovery_ready),
            "recovery_blockers": recovery_blockers,
            "recovery_mode": recovery_mode,
        }

    def _build_signals(self, df) -> list[TradeSignal]:
        signal_context = self._signal_generation_context(df)
        signals = [
            self._apply_signal_generation_gating(self._sanitize_signal(s.generate_signal(df)), signal_context)
            for s in self.base_strategies
            if self._strategy_enabled(str(getattr(s, "name", "")))
        ]
        include_evolved_runtime = bool(getattr(self.settings, "backtest_include_evolved", True))
        evolved_pool = self._load_evolved_strategy_pool() if include_evolved_runtime else []
        if not evolved_pool:
            # Avoid expensive live-time evolution when no historical model is available yet.
            if include_evolved_runtime:
                evolved_pool = [EvolvedRuleStrategy(fast=20, slow=50, rsi_buy=45.0, rsi_sell=55.0)]
            if include_evolved_runtime and not self._logged_evolved_fallback:
                self.store.log_event(
                    "system",
                    "No promoted evolved strategy pool found; using fallback parameters until mutation training completes.",
                )
                self._logged_evolved_fallback = True
        for evolved in evolved_pool:
            evolved_signal = self._apply_signal_generation_gating(
                self._sanitize_signal(evolved.generate_signal(df)),
                signal_context,
            )
            evolved_signal.strategy = self._format_evolved_strategy_name(evolved)
            signals.append(evolved_signal)

        # Active demo-stage chromosomes are operator-reviewed strategy signals.
        # They remain signal-only; execution still routes through ExecutionGate.
        demo_chromosomes = self._load_chromosome_demo_pool()
        for chromosome in demo_chromosomes:
            try:
                strategy = StrategyFactory.build(chromosome)
                chromo_signal = self._apply_signal_generation_gating(
                    self._sanitize_signal(strategy.generate_signal(df)),
                    signal_context,
                )
                chromo_signal.strategy = strategy.name
                signals.append(chromo_signal)
            except Exception as exc:
                self.store.log_event(
                    "system",
                    f"Demo chromosome signal generation failed for {chromosome.uid[:8]}: {exc}",
                )
        return signals

    def _top_signal_rows(
        self,
        signals: list[TradeSignal],
        contextual_scores: dict[str, float] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        include_scores = isinstance(contextual_scores, dict)
        for s in signals:
            row = {
                "strategy": str(getattr(s, "strategy", "")),
                "side": str(getattr(s, "side", "flat")),
                "confidence": round(float(getattr(s, "confidence", 0.0) or 0.0), 4),
            }
            if include_scores:
                row["contextual_score"] = round(float(contextual_scores.get(self._strategy_key(s.strategy), 0.0)), 4)
            rows.append(row)
        rows.sort(
            key=lambda row: (
                1 if str(row.get("side", "flat")) in {"buy", "sell"} else 0,
                float(row.get("contextual_score", row.get("confidence", 0.0)) or 0.0),
                float(row.get("confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return rows[: max(1, int(limit))]

    def _market_status_view(self, recent_closes: list[float], recent_times: list[str], limit: int = 20) -> dict[str, Any]:
        clipped = max(2, int(limit))
        return {
            "recent_close": recent_closes[-clipped:],
            "recent_time": recent_times[-clipped:],
            "last_close": recent_closes[-1] if recent_closes else None,
        }

    def _contextual_ranking_view(self, contextual_details: dict[str, dict[str, Any]], chosen_strategy: str, contextual_scores: dict[str, float]) -> dict[str, Any]:
        top_rows = sorted(
            [
                {
                    "strategy": detail["strategy"],
                    "score": detail["final_score"],
                    "factor": detail["factor"],
                    "setup_quality_score": round(float(dict(detail.get("setup_quality", {}) or {}).get("score", 0.0) or 0.0), 4),
                    "reasons": detail["reasons"][:5],
                }
                for detail in contextual_details.values()
            ],
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )[:3]
        return {
            "chosen_score": round(float(contextual_scores.get(self._strategy_key(chosen_strategy), 0.0)), 4),
            "top": top_rows,
        }

    def _best_actionable_candidate_view(
        self,
        signals: list[TradeSignal],
        contextual_details: dict[str, dict[str, Any]],
        contextual_scores: dict[str, float],
    ) -> dict[str, Any] | None:
        actionable = [
            signal
            for signal in signals
            if str(getattr(signal, "side", "flat")) in {"buy", "sell"}
            and float(getattr(signal, "confidence", 0.0) or 0.0) > 0.0
        ]
        if not actionable:
            return None
        ranked = sorted(
            actionable,
            key=lambda signal: (
                float(contextual_scores.get(self._strategy_key(str(getattr(signal, "strategy", ""))), 0.0) or 0.0),
                float(getattr(signal, "confidence", 0.0) or 0.0),
            ),
            reverse=True,
        )
        best = ranked[0]
        key = self._strategy_key(str(getattr(best, "strategy", "")))
        detail = dict(contextual_details.get(key, {}) or {})
        return {
            "strategy": str(getattr(best, "strategy", "")),
            "side": str(getattr(best, "side", "flat")),
            "confidence": round(float(getattr(best, "confidence", 0.0) or 0.0), 4),
            "score": round(float(contextual_scores.get(key, 0.0) or 0.0), 4),
            "setup_quality_score": round(float(dict(detail.get("setup_quality", {}) or {}).get("score", 0.0) or 0.0), 4),
            "reasons": list(detail.get("reasons", []))[:5],
        }

    def _setup_quality_filter_meta(
        self,
        *,
        chosen: TradeSignal,
        chosen_context: dict[str, Any],
        chosen_detail: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = bool(getattr(self.settings, "setup_quality_filter_enabled", False))
        threshold = float(getattr(self.settings, "setup_quality_filter_min_score", 76.0) or 76.0)
        if not enabled:
            return {"enabled": False, "threshold": round(threshold, 4)}
        side = str(getattr(chosen, "side", "flat") or "flat").lower()
        if side not in {"buy", "sell"}:
            return {"enabled": True, "threshold": round(threshold, 4), "applied": False, "reason": "flat_side"}
        setup_quality = dict(chosen_detail.get("setup_quality", {}) or {})
        scored = score_live_forward_safe_setup(
            session=str(chosen_context.get("session", "unknown") or "unknown"),
            regime=str(chosen_context.get("regime", "unknown") or "unknown"),
            trend_strength=float(chosen_context.get("trend_strength", 0.0) or 0.0),
            volatility_pct=float(chosen_context.get("volatility_pct", 0.0) or 0.0),
            volume_ratio=float(chosen_context.get("volume_ratio", 1.0) or 1.0),
            spike_ratio=float(chosen_context.get("spike_ratio", 0.0) or 0.0),
            same_side_support=int(chosen_context.get("same_side_support", 0) or 0),
            setup_quality_reasons=list(setup_quality.get("reasons", []) or []),
        )
        live_score = float(scored.get("setup_quality_score", 0.0) or 0.0)
        blocked = live_score < threshold
        reason = ""
        if blocked:
            reason = f"Setup-quality filter blocked trade: score={live_score:.1f} < {threshold:.1f}"
            reasons = list(scored.get("reasons", []) or [])
            if reasons:
                reason += f" ({', '.join(reasons[:4])})."
            else:
                reason += "."
        return {
            "enabled": True,
            "applied": blocked,
            "hard_blocked": blocked,
            "threshold": round(threshold, 4),
            "score": round(live_score, 4),
            "bucket": str(scored.get("setup_quality_bucket", "unknown")),
            "reasons": list(scored.get("reasons", []) or []),
            "reason_contributions": dict(scored.get("reason_contributions", {}) or {}),
            "reason": reason,
        }

    def _compact_runtime_event_meta(self, status: dict[str, Any], event_type: str) -> dict[str, Any]:
        chosen = dict(status.get("chosen", {}) or {})
        risk = dict(status.get("risk", {}) or {})
        compact: dict[str, Any] = {
            "symbol": status.get("symbol"),
            "timeframe": status.get("timeframe"),
            "action": status.get("action"),
            "message": status.get("message"),
            "regime_label": status.get("regime_label"),
            "regime_label_detail": dict(status.get("regime_label_detail", {}) or {}),
            "chosen": {
                "strategy": chosen.get("strategy"),
                "side": chosen.get("side"),
                "confidence": chosen.get("confidence"),
            },
            "risk": {
                "equity": risk.get("equity"),
                "balance": risk.get("balance"),
                "daily_loss_pct": risk.get("daily_loss_pct"),
                "weekly_loss_pct": risk.get("weekly_loss_pct"),
                "open_positions": risk.get("open_positions"),
                "spread_points": risk.get("spread_points"),
                "trades_today": risk.get("trades_today"),
                "consecutive_losses": risk.get("consecutive_losses"),
                "daily_profit_zar": risk.get("daily_profit_zar"),
            },
            "signal_summary": dict(status.get("signal_summary", {}) or {}),
            "higher_timeframe": dict(status.get("higher_timeframe", {}) or {}),
            "drift": dict(status.get("drift", {}) or {}),
            "contextual_ranking": dict(status.get("contextual_ranking", {}) or {}),
            "best_actionable_candidate": dict(status.get("best_actionable_candidate", {}) or {}),
            "context_snapshot": dict(status.get("context_snapshot", {}) or {}),
            "decision_explanation": dict(status.get("decision_explanation", {}) or {}),
            "decision_trace": dict(status.get("decision_trace", {}) or {}),
            "timing_discipline": dict(status.get("timing_discipline", {}) or {}),
            "system_health": dict(status.get("system_health", {}) or {}),
            "shadow_advice": dict(status.get("shadow_advice", {}) or {}),
        }
        if event_type == "execution":
            compact["execution"] = dict(status.get("execution", {}) or {})
            compact["volume"] = status.get("volume")
        elif event_type == "paper":
            compact["execution"] = dict(status.get("execution", {}) or {})
        return compact

    @staticmethod
    def _attach_regime_label(status: dict[str, Any], regime: dict[str, Any] | None = None) -> None:
        detail = dict((regime or {}).get("regime_label_detail", {}) or status.get("regime_label_detail", {}) or {})
        label = str((regime or {}).get("regime_label", "") or detail.get("label", "") or status.get("regime_label", "") or "").strip()
        if not label:
            context = dict(status.get("context_snapshot", {}) or {})
            label = str(context.get("regime_label", "") or context.get("regime", "") or "ranging").strip()
        if label == "range":
            label = "ranging"
        if label == "trend":
            label = "trending"
        if label == "high_volatility_range":
            label = "high_volatility"
        if label not in {"trending", "ranging", "high_volatility", "low_volatility"}:
            label = "ranging"
        status["regime_label"] = label
        if detail:
            status["regime_label_detail"] = detail
        context = dict(status.get("context_snapshot", {}) or {})
        context["regime_label"] = label
        if detail:
            context["regime_label_detail"] = detail
        status["context_snapshot"] = context

    def _build_decision_sample_payload(
        self,
        *,
        decision_id: str,
        chosen: TradeSignal,
        setup_family: str,
        status: dict[str, Any],
        regime: dict[str, Any],
        higher_tf_bias: str,
        snapshot: Any,
        support_count: int,
        signal_count: int,
        contextual_detail: dict[str, Any],
        meta_model: dict[str, Any],
        execution: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        chart_features = dict(status.get("chart_features", {}) or {})
        visual = dict(status.get("visual_chart_read", {}) or {})
        analysis = dict(status.get("decision_analysis", {}) or {})
        setup_radar = dict(analysis.get("setup_radar", {}) or {})
        top_watch = dict(setup_radar.get("top_watch", {}) or {})
        setup_quality = dict(contextual_detail.get("setup_quality", {}) or {})
        contextual_context = dict(contextual_detail.get("context", {}) or {})
        setup_quality_reasons = list(setup_quality.get("reasons", []) or [])
        setup_quality_reasons.extend(str(x.get("name", "") or "") for x in list(analysis.get("top_blockers", []) or [])[:4])
        entry_location = str(
            contextual_context.get("entry_location_state")
            or contextual_context.get("location")
            or chart_features.get("location_state")
            or "unknown"
        ).strip().lower() or "unknown"
        entry_momentum = str(
            contextual_context.get("entry_momentum_state")
            or contextual_context.get("momentum")
            or chart_features.get("momentum_state")
            or "unknown"
        ).strip().lower() or "unknown"
        entry_trend = str(
            contextual_context.get("entry_trend_state")
            or contextual_context.get("trend_state")
            or chart_features.get("trend_state")
            or regime.get("name", "unknown")
            or "unknown"
        ).strip().lower() or "unknown"
        volatility_pct = float(regime.get("volatility_pct", contextual_context.get("volatility_pct", 0.0)) or 0.0)
        volatility_state = str(
            contextual_context.get("volatility_state")
            or chart_features.get("volatility_state")
            or self._volatility_bucket(volatility_pct)
        ).strip().lower() or "unknown"
        payload = {
            "schema": "atlas_decision_sample_v2",
            "decision_id": decision_id,
            "strategy": chosen.strategy,
            "setup_family": setup_family,
            "side": chosen.side,
            "confidence": round(float(chosen.confidence), 4),
            "timeframe": self.settings.timeframe,
            "regime": str(regime.get("name", "unknown")),
            "higher_tf_bias": str(higher_tf_bias),
            "htf_aligned": str(chosen.side) in {"buy", "sell"} and str(higher_tf_bias) == str(chosen.side),
            "session": self._session_label(datetime.now(timezone.utc)),
            "hour_utc": int(datetime.now(timezone.utc).hour),
            "spread_points": round(float(snapshot.spread_points), 2),
            "same_side_support": int(support_count),
            "same_side_ratio": round(float(support_count) / max(1, int(signal_count)), 4),
            "volatility_pct": volatility_pct,
            "volatility_state": volatility_state,
            "trend_strength": float(regime.get("trend_strength", contextual_context.get("trend_strength", 0.0)) or 0.0),
            "entry_location_state": entry_location,
            "entry_momentum_state": entry_momentum,
            "entry_trend_state": entry_trend,
            "setup_quality_score": float(setup_quality.get("score", 0.0) or 0.0),
            "setup_quality_reasons": list(dict.fromkeys(str(x) for x in setup_quality_reasons if str(x).strip()))[:8],
            "setup_radar": {
                "top_watch": str(top_watch.get("name", "") or ""),
                "phase": str(top_watch.get("phase", setup_radar.get("timing_state", "")) or ""),
                "trigger_state": str(top_watch.get("trigger_state", "") or ""),
                "missing_to_entry": list(top_watch.get("missing_to_entry", []) or [])[:8],
            },
            "chart_features": {
                "trend_state": str(chart_features.get("trend_state", "") or ""),
                "trend_structure": str(chart_features.get("trend_structure", "") or ""),
                "momentum_state": str(chart_features.get("momentum_state", "") or ""),
                "location_state": str(chart_features.get("location_state", "") or ""),
                "setup_type_hint": str(chart_features.get("setup_type_hint", "") or ""),
                "reliability": str(chart_features.get("reliability", "") or ""),
            },
            "visual_chart_read": {
                "available": bool(visual.get("available", False)),
                "pattern_seen": str(visual.get("pattern_seen", "") or ""),
                "visual_confirmation": str(visual.get("visual_confirmation", "") or ""),
                "setup_quality": visual.get("setup_quality"),
                "visual_bias": str(visual.get("visual_bias", "") or ""),
            },
            "top_blockers": list(analysis.get("top_blockers", []) or [])[:5],
            "m1_quality": dict(status.get("m1_quality", {}) or {}),
            "contextual_ranking": contextual_detail,
            "meta_model": meta_model,
            "bounded_learning_only": True,
            "policy_note": "Decision sample is observability/learning evidence only; it does not change strategy, risk, guards, or execution policy.",
        }
        if execution:
            payload.update(dict(execution))
        return payload

    def _augment_status_layers(self, status: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(status, dict):
            return status
        runtime_flags = dict(self._last_runtime_flags or {})
        rollout = dict(runtime_flags.get("paper_rollout", {}) or {})
        if rollout and not status.get("lane"):
            status["lane"] = str(rollout.get("lane", "") or "default")
        if "context_snapshot" not in status:
            status["context_snapshot"] = build_context_snapshot(status)
        if "decision_explanation" not in status:
            status["decision_explanation"] = build_decision_explanation(
                status=status,
                context_snapshot=dict(status.get("context_snapshot", {}) or {}),
                runtime_flags=runtime_flags,
            )
        status["decision_trace"] = build_decision_trace(
            status=status,
            context_snapshot=dict(status.get("context_snapshot", {}) or {}),
            runtime_flags=runtime_flags,
        )
        blocker_summary = dict(status["decision_trace"].get("blocker_summary", {}) or {})
        status["blocker_classification"] = blocker_summary
        status["blocker_class"] = str(blocker_summary.get("class", "unknown") or "unknown")
        status["primary_blocker"] = str(blocker_summary.get("primary", status.get("message", "")) or status.get("message", ""))
        if "decision_analysis" not in status or "pre_trade_memo" not in status:
            structured = build_structured_decision_analysis(status=status, runtime=runtime_flags)
            status.setdefault("decision_analysis", structured)
            status.setdefault("pre_trade_memo", dict(structured.get("pre_trade_memo", {}) or {}))
        structured = dict(status.get("decision_analysis", {}) or {})
        status.setdefault("market_regime", dict(structured.get("market_regime", {}) or {}))
        status.setdefault("reversal_detection", dict(structured.get("reversal_detection", {}) or {}))
        status.setdefault("setup_playbook", dict(structured.get("setup_playbook", {}) or {}))
        if "setup_hunter" not in status:
            status["setup_hunter"] = build_setup_hunter(
                status=status,
                decision_analysis=structured,
                runtime=runtime_flags,
            )
        if "setup_quality_observation" not in status:
            from atlas_ai.analysis.setup_playbook import build_setup_quality_observation

            context = dict(status.get("context_snapshot", {}) or {})
            chosen = dict(status.get("chosen", {}) or {})
            context["chosen_side"] = str(chosen.get("side", "") or "")
            status["setup_quality_observation"] = build_setup_quality_observation(
                dict(status.get("setup_playbook", {}) or {}),
                context,
            )
        advisory_setup_score = float(dict(status.get("setup_quality_observation", {}) or {}).get("score", 0.0) or 0.0)
        if advisory_setup_score > 0:
            candidate = status.get("best_actionable_candidate")
            if isinstance(candidate, dict) and float(candidate.get("setup_quality_score", 0.0) or 0.0) == 0.0:
                candidate["advisory_setup_quality_score"] = round(advisory_setup_score, 4)
                candidate["setup_quality_source"] = "setup_playbook_advisory"
            ranking = status.get("contextual_ranking")
            if isinstance(ranking, dict):
                for row in list(ranking.get("top", []) or []):
                    if isinstance(row, dict) and float(row.get("setup_quality_score", 0.0) or 0.0) == 0.0:
                        row["advisory_setup_quality_score"] = round(advisory_setup_score, 4)
                        row["setup_quality_source"] = "setup_playbook_advisory"
        if "self_improvement_state" not in status:
            status["self_improvement_state"] = build_self_improvement_summary(
                load_self_improvement_bundle(getattr(self, "store", None))
            )
        if "system_health" not in status:
            status["system_health"] = build_system_health(status, self.settings, runtime_flags=runtime_flags)
        else:
            refreshed_health = build_system_health(status, self.settings, runtime_flags=runtime_flags)
            system_health = dict(status.get("system_health", {}) or {})
            system_health.update(refreshed_health)
            status["system_health"] = system_health
        execution_gate = getattr(self, "execution_gate", None)
        execution_snapshot = dict(execution_gate.snapshot() or {}) if execution_gate is not None else {}
        if execution_snapshot:
            status["execution_state"] = execution_snapshot
            status["execution_gate"] = {
                **dict(status.get("execution_gate", {}) or {}),
                "execution_environment": str(execution_snapshot.get("execution_environment", "") or ""),
                "execution_environment_label": str(execution_snapshot.get("execution_environment_label", "") or ""),
                "execution_gate_enabled": bool(execution_snapshot.get("execution_gate_enabled", False)),
                "rollout_stage": str(execution_snapshot.get("rollout_stage", "") or ""),
                "adapter_type": str(execution_snapshot.get("adapter_type", "") or ""),
            }
            system_health = dict(status.get("system_health", {}) or {})
            mode_truth = dict(system_health.get("mode_truth", {}) or {})
            mode_truth["gate_execution_environment"] = str(execution_snapshot.get("execution_environment", "") or "")
            mode_truth["gate_execution_environment_label"] = str(execution_snapshot.get("execution_environment_label", "") or "")
            mode_truth["execution_gate_enabled"] = bool(execution_snapshot.get("execution_gate_enabled", False))
            system_health["mode_truth"] = mode_truth
            status["system_health"] = system_health
        system_health = dict(status.get("system_health", {}) or {})
        mode_truth = dict(system_health.get("mode_truth", {}) or {})
        status["effective_mode"] = str(system_health.get("effective_mode", status.get("effective_mode", "")) or "")
        status["effective_mode_label"] = str(
            system_health.get("effective_mode_label", status.get("effective_mode_label", "")) or ""
        )
        status["paper_mode_confirmed"] = bool(system_health.get("paper_mode_confirmed", status.get("paper_mode_confirmed", False)))
        gate_environment = str(
            mode_truth.get("gate_execution_environment", "")
            or dict(status.get("execution_gate", {}) or {}).get("execution_environment", "")
            or dict(runtime_flags.get("startup_verification", {}) or {}).get("gate_execution_environment", "")
            or status.get("gate_execution_environment", "")
            or ""
        )
        if gate_environment:
            status["gate_execution_environment"] = gate_environment
        if "shadow_advice" not in status:
            status["shadow_advice"] = build_shadow_advice(
                status=status,
                context_snapshot=dict(status.get("context_snapshot", {}) or {}),
                decision_explanation=dict(status.get("decision_explanation", {}) or {}),
            )
        if "trader_cognition" not in status:
            status["trader_cognition"] = build_trader_cognition(
                status=status,
                runtime=runtime_flags,
                mandate=self._load_autonomous_trader_mandate(),
            )
        return status

    def _load_autonomous_trader_mandate(self) -> dict[str, Any]:
        path = Path(__file__).resolve().parents[1] / "knowledge" / "autonomous_trader_mandate.json"
        if not path.exists():
            path = Path(__file__).resolve().parents[2] / "knowledge" / "autonomous_trader_mandate.json"
        try:
            import json

            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "schema": "atlas_autonomous_trader_mandate_v1",
                "status": "missing_or_unreadable",
                "autonomy_model": {
                    "target_role": "bounded_autonomous_trader",
                    "execution_authority": "ExecutionGate remains the only execution authority.",
                },
            }

    def _log_status_event(self, event_type: str, status: dict[str, Any]) -> None:
        self._augment_status_layers(status)
        self._attach_regime_label(status)
        self.store.log_event(event_type, str(status.get("message", "")), self._compact_runtime_event_meta(status, event_type))

    def _publish_status(self, payload: dict[str, Any]) -> None:
        self._augment_status_layers(payload)
        self._attach_regime_label(payload)
        timing = dict(getattr(self, "_current_cycle_timing", {}) or {})
        if timing:
            now = datetime.now(timezone.utc)
            cycle_started_at = str(timing.get("cycle_started_at", "") or "")
            age_ms = None
            if cycle_started_at:
                try:
                    cycle_dt = datetime.fromisoformat(cycle_started_at)
                    age_ms = round((now - cycle_dt).total_seconds() * 1000.0, 2)
                except Exception:
                    age_ms = None
            timing["status_write_attempted_at"] = now.isoformat()
            timing["backend_cycle_age_ms"] = age_ms
            payload["freshness_timing"] = timing
            payload["freshness_runtime"] = self._freshness_runtime_payload(status_age_ms=age_ms, max_age_ms=120000)
        audit_result = record_missed_entry_audit(self.settings.storage_dir, payload)
        if bool(audit_result.get("recorded", False)):
            audit = dict(audit_result.get("audit", {}) or {})
            payload["missed_entry_audit"] = {
                "recorded": True,
                "path": str(audit_result.get("path", "") or ""),
                "learning_focus": list(audit.get("learning_focus", []) or []),
            }
        elif bool(audit_result.get("deduped", False)):
            payload["missed_entry_audit"] = {
                "recorded": False,
                "deduped": True,
                "reason": str(audit_result.get("reason", "") or ""),
            }
        review_result = review_missed_entry_outcomes(self.settings.storage_dir, payload)
        if bool(review_result.get("reviewed", False)):
            payload["missed_entry_outcome_review"] = {
                "reviewed": True,
                "path": str(review_result.get("path", "") or ""),
                "summary": dict(review_result.get("summary", {}) or {}),
            }
        payload["missed_entry_learning_summary"] = summarize_missed_entry_learning(self.settings.storage_dir)
        payload["blocker_selectivity_review"] = build_blocker_selectivity_review(self.settings.storage_dir)
        payload["blocker_selectivity_replay"] = build_blocker_selectivity_replay(self.settings.storage_dir)
        payload["blocker_selectivity_collection_plan"] = build_blocker_selectivity_collection_plan(self.settings.storage_dir)
        offline_collector = build_blocker_selectivity_offline_collector(self.settings.storage_dir)
        payload["blocker_selectivity_offline_collector"] = offline_collector
        readiness = build_blocker_selectivity_collection_readiness(self.settings.storage_dir)
        payload["blocker_selectivity_collection_readiness"] = readiness
        retry_ready = bool(readiness.get("ready_to_retry_shadow_compare", False)) or bool(
            offline_collector.get("ready_to_retry_shadow_compare", False)
        )
        if retry_ready:
            retry_result = validate_blocker_selectivity_shadow_compare(self.store)
            payload["blocker_selectivity_shadow_compare_retry"] = {
                "attempted": True,
                "ok": bool(retry_result.get("ok", False)),
                "source": "recent_readiness" if bool(readiness.get("ready_to_retry_shadow_compare", False)) else "offline_collector",
                "missing": list(retry_result.get("missing", []) or []),
                "comparison": dict(dict(retry_result.get("report", {}) or {}).get("comparison", {}) or {}),
                "policy": dict(dict(retry_result.get("report", {}) or {}).get("policy", {}) or {}),
            }
        else:
            payload["blocker_selectivity_shadow_compare_retry"] = {
                "attempted": False,
                "reason": "collection_readiness_not_met",
                "remaining_enriched_samples_needed": int(readiness.get("remaining_enriched_samples_needed", 0) or 0),
                "offline_remaining_enriched_samples_needed": int(
                    offline_collector.get("remaining_enriched_samples_needed", 0) or 0
                ),
            }
        self.store.save_status(payload)

    def _execute_trades_enabled(self, flags: dict[str, Any] | None = None) -> bool:
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        return bool(flags.get("execute_trades", self.settings.execute_trades))

    def _gemini_influence_enabled(self, flags: dict[str, Any] | None = None) -> bool:
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        return bool(flags.get("gemini_influence_enabled", self.settings.gemini_influence_enabled))

    def _emit_llm_health_alert_if_needed(self, llm_health: dict[str, Any], flags: dict[str, Any] | None = None) -> None:
        providers = list((llm_health or {}).get("providers", []))
        flagged = []
        for p in providers:
            if bool(p.get("in_cooldown")) and str(p.get("last_error", "")).strip():
                flagged.append({"name": p.get("name", "unknown"), "last_error": str(p.get("last_error"))[:240]})
        if not flagged:
            return
        flags = flags if isinstance(flags, dict) else self.store.load_runtime_flags()
        raw_last = str(flags.get("llm_alert_last_at", "") or "")
        now = datetime.now(timezone.utc)
        if raw_last:
            try:
                last = datetime.fromisoformat(raw_last)
                if (now - last).total_seconds() < 20 * 60:
                    return
            except Exception:
                pass
        self.store.log_event("risk", "LLM provider is rate-limited; fallback/local logic active.", {"providers": flagged})
        flags["llm_alert_last_at"] = now.isoformat()
        self.store.save_runtime_flags(flags)

    def _housekeeping_close_allowed(self) -> tuple[bool, str]:
        # Cross-process safety: refresh shared critical controls before any direct
        # housekeeping close path, then apply conservative block checks.
        sync = self.execution_gate.refresh_critical_controls_from_persisted_state()
        if not bool(sync.get("ok", False)):
            return False, f"shared_state_sync_failed:{sync.get('reason', 'load_failed')}"
        snap = dict(self.execution_gate.snapshot() or {})
        if bool(snap.get("kill_switch_active", False)):
            return False, "kill_switch_active"
        if bool(snap.get("sleep_mode_active", False)):
            return False, "sleep_mode_active"
        if not bool(snap.get("execution_gate_enabled", False)):
            return False, "execution_gate_disabled"
        return True, "allowed"

    def step(self) -> dict[str, Any]:
        cycle_started_at = datetime.now(timezone.utc)
        cycle_started_mono = time.perf_counter()
        self._current_cycle_timing = {
            "cycle_started_at": cycle_started_at.isoformat(),
            "ingest_started_at": cycle_started_at.isoformat(),
        }
        self._step_event_cache = {}
        heartbeat_mode = "live" if self._execute_trades_enabled() else "paper"
        with contextlib.suppress(Exception):
            self.store.write_backend_heartbeat(mode=heartbeat_mode, pid=os.getpid(), cycle_status="running")
        runtime_flags = self.store.load_runtime_flags()
        self._last_runtime_flags = dict(runtime_flags or {})
        execute_trades = self._execute_trades_enabled(runtime_flags)
        gemini_influence_enabled = self._gemini_influence_enabled(runtime_flags)

        # P1 — Worker heartbeat check: block trading if stale
        heartbeat_check = evaluate_heartbeat_state(
            runtime_flags,
            component="backend_trader",
            stale_after_seconds=float(os.getenv("ATLAS_HEARTBEAT_STALE_AFTER_SECONDS", "180") or 180),
        )
        if heartbeat_check.get("is_stale", False):
            stale_reason = heartbeat_check.get("reason", "heartbeat_stale")
            self.store.log_event(
                "risk",
                f"Trading blocked: worker heartbeat is stale ({stale_reason}).",
                {"stale_seconds": heartbeat_check.get("stale_seconds"), "reason": stale_reason},
            )
            self.voice.speak("Attention. Trading blocked. System heartbeat is stale.")
            return {
                "symbol": self.settings.symbol,
                "timeframe": self.settings.timeframe,
                "action": "blocked",
                "message": f"Trading blocked: worker heartbeat is stale ({stale_reason}).",
                "execute_trades": execute_trades,
                "gemini_influence_enabled": gemini_influence_enabled,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

        self._maybe_refresh_external_macro_context()

        loss_memory = self._sync_closed_trade_learning(runtime_flags)
        self._apply_open_position_feedback(runtime_flags)
        self._maybe_emit_drift_alert(loss_memory, runtime_flags)
        drift = self._drift_state(loss_memory)

        if self.settings.enforce_day_trading and execute_trades:
            allow_housekeeping_close, housekeeping_reason = self._housekeeping_close_allowed()
            if not allow_housekeeping_close:
                self.store.log_event(
                    "day_trader",
                    "Skipped overnight housekeeping close due to guarded control state.",
                    {"reason": housekeeping_reason, "guarded_boundary": "housekeeping_exception"},
                )
            else:
                closed = self.executor.close_stale_positions(allow_close=True)
                if closed:
                    self.store.log_event(
                        "day_trader",
                        "Closed overnight positions to enforce day-trading mode.",
                        {"closed": [c.__dict__ for c in closed], "guarded_boundary": "housekeeping_exception"},
                    )

        df = self.feed.get_rates(1200)
        ingest_ended_at = datetime.now(timezone.utc)
        self._current_cycle_timing["ingest_completed_at"] = ingest_ended_at.isoformat()
        self._current_cycle_timing["ingest_duration_ms"] = round((ingest_ended_at - cycle_started_at).total_seconds() * 1000.0, 2)
        self._current_cycle_timing["market_fetch"] = dict(getattr(self.feed, "last_fetch_meta", {}) or {})
        raw_signals = self._build_signals(df)
        self._current_cycle_timing["signal_generation_completed_at"] = datetime.now(timezone.utc).isoformat()
        signals, weak_blocked = self._filter_weak_strategies(raw_signals)
        drift_adjustments = self._apply_drift_signal_adjustment(signals, drift)
        drift_strategy_adjustments = self._apply_drift_strategy_weighting(signals, drift)
        snapshot = self._risk_snapshot(runtime_flags)
        recent = df.tail(60)
        recent_closes = [float(x) for x in recent["close"].tolist()]
        recent_times = [str(x) for x in recent["time"].tolist()]
        regime = self._market_regime(recent_closes)
        higher_tf_context = self._higher_tf_context()
        higher_tf_bias = str(higher_tf_context.get("bias", "unknown") or "unknown")
        contextual_scores, contextual_details = self._contextual_signal_scores(
            signals=signals,
            df=df,
            snapshot=snapshot,
            regime=regime,
            higher_tf_bias=higher_tf_bias,
            drift=drift,
            memory=loss_memory,
            now_utc=datetime.now(timezone.utc),
        )
        if not signals:
            status = {
                "symbol": self.settings.symbol,
                "timeframe": self.settings.timeframe,
                "session": self._session_label(datetime.now(timezone.utc)),
                "gemini_enabled": self.gemini.enabled,
                "risk": {
                    "equity": snapshot.equity,
                    "balance": snapshot.balance,
                    "daily_loss_pct": snapshot.daily_loss_pct,
                    "weekly_loss_pct": snapshot.weekly_loss_pct,
                    "open_positions": snapshot.open_positions,
                    "spread_points": snapshot.spread_points,
                },
                "signals": [],
                "signals_active": [],
                "chosen": {"strategy": "none", "side": "flat", "confidence": 0.0},
                "market": self._market_status_view(recent_closes, recent_times, limit=20),
                "regime": regime,
                "higher_timeframe": higher_tf_context,
                "action": "hold",
                "message": "No eligible strategies after filtering.",
                "execute_trades": execute_trades,
                "gemini_influence_enabled": gemini_influence_enabled,
                "drift": drift,
                "contextual_ranking": {"chosen_score": 0.0, "top": [], "best_actionable": {}},
                "best_actionable_candidate": {},
                "setup_family": "no_trade",
                "chart_features": self._derive_runtime_chart_features(
                    df=df,
                    chosen=None,
                    regime=regime,
                ),
                "signal_summary": {
                    "raw_count": len(raw_signals),
                    "active_count": 0,
                    "filtered_out_count": len(raw_signals),
                    "evolved_raw_count": sum(1 for s in raw_signals if str(getattr(s, "strategy", "")).startswith("evolved_rule(")),
                    "evolved_active_count": 0,
                },
                "health": {"broker": self.connector.health(), "llm": self.gemini.health(), "risk_guard": {
                    "max_trades_per_day": int(self.settings.max_trades_per_day),
                    "max_consecutive_losses": int(self.settings.max_consecutive_losses),
                    "cooldown_minutes_after_loss_streak": int(self.settings.cooldown_minutes_after_loss_streak),
                }},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            if weak_blocked:
                status["win_rate_focus"] = {
                    "enabled": True,
                    "weak_strategies_blocked": weak_blocked[:5],
                }
            self._attach_visual_chart_read(status)
            self._log_status_event("decision", status)
            self._publish_status(status)
            return status
        actionable_signals = [
            s for s in signals if str(getattr(s, "side", "flat")) in {"buy", "sell"} and float(getattr(s, "confidence", 0.0) or 0.0) > 0.0
        ]
        best_actionable_candidate = self._best_actionable_candidate_view(
            signals=signals,
            contextual_details=contextual_details,
            contextual_scores=contextual_scores,
        )
        selection_pool = actionable_signals or signals
        chosen = self.allocator.choose(selection_pool, score_overrides=contextual_scores)
        ema_signal = next((s for s in raw_signals if str(getattr(s, "strategy", "")) == "ema_crossover"), None)
        derived_setup_family = self._strategy_to_setup_family(str(chosen.strategy))
        if str(getattr(chosen, "side", "flat")) not in {"buy", "sell"} or float(getattr(chosen, "confidence", 0.0) or 0.0) <= 0.0:
            # purpose: keep status/setup semantics coherent when no actionable
            # directional candidate exists in the current cycle.
            derived_setup_family = "no_trade"

        raw_strategy_view = self._top_signal_rows(raw_signals, limit=8)
        strategy_view = self._top_signal_rows(signals, contextual_scores=contextual_scores, limit=5)

        status: dict[str, Any] = {
            "symbol": self.settings.symbol,
            "timeframe": self.settings.timeframe,
            # FIX: propagate current UTC session early so all downstream layers consume consistent session context.
            "session": self._session_label(datetime.now(timezone.utc)),
            "gemini_enabled": self.gemini.enabled,
            "risk": {
                "equity": snapshot.equity,
                "balance": snapshot.balance,
                "daily_loss_pct": snapshot.daily_loss_pct,
                "open_positions": snapshot.open_positions,
                "spread_points": round(snapshot.spread_points, 2),
            },
            "signals": raw_strategy_view,
            "signals_active": strategy_view,
            "chosen": {
                "strategy": chosen.strategy,
                "side": chosen.side,
                "confidence": chosen.confidence,
                "signal_confidence_seed": chosen.confidence,
                "stop_loss_points": float(getattr(chosen, "stop_loss_points", 0.0) or 0.0),
                "take_profit_points": float(getattr(chosen, "take_profit_points", 0.0) or 0.0),
            },
            "market": self._market_status_view(recent_closes, recent_times, limit=20),
            "regime": regime,
            "higher_timeframe": higher_tf_context,
            "action": "hold",
            "message": "No execution yet",
            "execute_trades": execute_trades,
            "gemini_influence_enabled": gemini_influence_enabled,
            "drift": {
                "active": bool(drift.get("active", False)),
                "severe": bool(drift.get("severe", False)),
                "delta": round(float(drift.get("delta", 0.0)), 3),
                "recent_win_rate_20": round(float(drift.get("recent_win_rate_20", 0.0)), 3),
                "baseline_win_rate": round(float(drift.get("baseline_win_rate", 0.0)), 3),
                "signal_adjustments": drift_adjustments,
                "strategy_weight_adjustments": drift_strategy_adjustments,
            },
            "setup_family": derived_setup_family,
            "chart_features": self._derive_runtime_chart_features(
                df=df,
                chosen=chosen,
                regime=regime,
            ),
            "contextual_ranking": self._contextual_ranking_view(
                contextual_details=contextual_details,
                chosen_strategy=str(chosen.strategy),
                contextual_scores=contextual_scores,
            ),
        }
        self._attach_visual_chart_read(status)
        self._current_cycle_timing["decision_pre_checks_completed_at"] = datetime.now(timezone.utc).isoformat()
        self._current_cycle_timing["cycle_elapsed_ms_before_gates"] = round((time.perf_counter() - cycle_started_mono) * 1000.0, 2)
        status["best_actionable_candidate"] = best_actionable_candidate
        status["contextual_ranking"]["best_actionable"] = dict(best_actionable_candidate or {})
        status["signal_summary"] = {
            "raw_count": len(raw_signals),
            "active_count": len(signals),
            "filtered_out_count": max(0, len(raw_signals) - len(signals)),
            "evolved_raw_count": sum(1 for s in raw_signals if str(getattr(s, "strategy", "")).startswith("evolved_rule(")),
            "evolved_active_count": sum(1 for s in signals if str(getattr(s, "strategy", "")).startswith("evolved_rule(")),
        }
        learning_status = self._learning_status_snapshot()
        status["learning_status"] = learning_status
        status["learning_status_snapshot"] = learning_status
        guard_state = self._load_guard_state(runtime_flags)
        if "start_balance" not in guard_state or float(guard_state.get("start_balance", 0.0)) <= 0:
            guard_state["start_balance"] = float(snapshot.balance)
        if "week_start_balance" not in guard_state or float(guard_state.get("week_start_balance", 0.0)) <= 0:
            guard_state["week_start_balance"] = float(snapshot.balance)
            self._save_guard_state(guard_state, runtime_flags)
        status["risk"]["trades_today"] = guard_state["trades_today"]
        status["risk"]["consecutive_losses"] = guard_state["consecutive_losses"]
        status["risk"]["daily_profit_zar"] = round(float(snapshot.equity) - float(guard_state.get("start_balance", snapshot.balance)), 2)
        status["risk"]["weekly_loss_pct"] = round(float(snapshot.weekly_loss_pct), 4)
        account_profile = self._account_tier_profile(snapshot.balance)
        size_tier_profile = self._size_tier_profile(snapshot.balance)
        status["account_profile"] = {
            "account_tier": str(account_profile.get("account_tier", "standard_account")),
            "tier_reason": str(account_profile.get("tier_reason", "")),
            "risk_profile": str(account_profile.get("risk_profile", "capital_preservation_standard")),
            "current_size_tier": str(size_tier_profile.get("current_size_tier", "")),
            "next_size_tier_requirement": str(size_tier_profile.get("next_size_tier_requirement", "")),
            "size_tier_reason": str(size_tier_profile.get("size_tier_reason", "")),
        }
        # FIX: build context snapshot before safety/knowledge checks to avoid unknown session/liquidity propagation.
        context_snapshot = build_context_snapshot(status)
        status["context_snapshot"] = dict(context_snapshot)
        self._attach_regime_label(status, regime)
        context_snapshot = dict(status.get("context_snapshot", {}) or {})
        status["learning_critic"] = self._learning_critic_snapshot(chosen, context_snapshot)
        status["timing_readiness"] = self._timing_readiness_snapshot(status["learning_critic"], context_snapshot)
        status["loss_to_win_playbook"] = self._loss_to_win_snapshot(chosen, context_snapshot)
        status["winner_playbook"] = self._winner_playbook_snapshot(chosen, context_snapshot)
        with contextlib.suppress(Exception):
            latest_visual = load_latest_visual_read(
                Path(str(self.settings.storage_dir)) / "chart_vision",
                max_age_seconds=int(os.getenv("ATLAS_VISUAL_WATCH_MAX_AGE_SECONDS", "10") or "10"),
            )
            if (
                str(latest_visual.get("source", "") or "") == "screenshot_plus_chart_features"
                and bool(latest_visual.get("live_chart_window_confirmed", False))
                and not bool(latest_visual.get("stale", False))
            ):
                status["visual_chart_read"] = dict(latest_visual)
                status["visual_chart_read"]["visual_watch_used_for_decision"] = True
        status["timing_discipline"] = self._timing_discipline_snapshot(chosen, context_snapshot, status)
        safety_lock_was_active = bool(guard_state.get("safety_lock_active", False))
        safety_layer = self._evaluate_safety_layer(
            snapshot=snapshot,
            regime=regime,
            context_snapshot=context_snapshot,
            guard_state=guard_state,
            loss_memory=loss_memory,
            runtime_flags=runtime_flags,
        )
        # P2 — Voice announcement when safety lock is newly activated
        if not safety_lock_was_active and bool(guard_state.get("safety_lock_active", False)):
            lock_reason = str(guard_state.get("safety_lock_reason", "unknown") or "unknown")
            self.voice.speak(f"Attention. Safety lock activated. Reason: {lock_reason}.")
        status["safety_layer"] = dict(safety_layer)
        self._save_guard_state(guard_state, runtime_flags)
        status["health"] = {
            "broker": {"connected": bool(mt5.account_info()), "symbol_selected": bool(mt5.symbol_info(self.settings.symbol))},
            "llm": self.gemini.health_summary(),
            "risk_guard": {
                "max_trades_per_day": self.settings.max_trades_per_day,
                "max_consecutive_losses": self.settings.max_consecutive_losses,
                "cooldown_minutes_after_loss_streak": self.settings.cooldown_minutes_after_loss_streak,
            },
        }
        self._emit_llm_health_alert_if_needed(status["health"]["llm"], runtime_flags)
        status["gemini_note"] = "Deferred to post-execution to reduce loop latency."

        allowed, reason = self.risk.evaluate(snapshot)
        if not allowed:
            status["action"] = "blocked"
            status["message"] = f"Risk guard active: {reason}. No trade executed."
            self._log_status_event("risk", status)
            self._publish_status(status)
            logger.warning(status["message"])
            return status

        guard_reason = self._guard_block_reason(guard_state)
        if guard_reason:
            status["action"] = "blocked"
            status["message"] = f"Risk guard active: {guard_reason}. No trade executed."
            self._log_status_event("risk", status)
            self._publish_status(status)
            logger.warning(status["message"])
            return status

        if self.settings.daily_profit_target_zar > 0:
            daily_profit = float(snapshot.equity) - float(guard_state.get("start_balance", snapshot.balance))
            if daily_profit >= float(self.settings.daily_profit_target_zar):
                guard_state["protect_gains_mode"] = True
                status["profit_protection_mode"] = {
                    "active": True,
                    "reason": "daily_profit_target_reached",
                    "detail": (
                        f"Daily profit target reached (+{daily_profit:.2f} >= +{self.settings.daily_profit_target_zar:.2f}); "
                        "risk posture remains defensive but no target-chasing or forced pause is applied."
                    ),
                }
            else:
                guard_state["protect_gains_mode"] = False
            self._save_guard_state(guard_state, runtime_flags)

        throttled, secs_left = self._drift_throttle_active(drift, datetime.now(timezone.utc), runtime_flags)
        if throttled:
            if bool(self.settings.fast_learning_mode):
                ema_side = str(getattr(ema_signal, "side", "flat")) if ema_signal is not None else "flat"
                ema_conf = float(getattr(ema_signal, "confidence", 0.0) or 0.0) if ema_signal is not None else 0.0
                support = self._same_side_support_count(signals, chosen.side)
                can_bypass = (
                    chosen.side in {"buy", "sell"}
                    and higher_tf_bias == chosen.side
                    and ema_side == chosen.side
                    and ema_conf > 0.0
                    and support >= max(2, int(self.settings.min_same_side_signals))
                )
                if can_bypass:
                    status["drift_throttle_bypass"] = {
                        "enabled": True,
                        "reason": "high_quality_alignment",
                        "support": int(support),
                        "ema_confidence": round(ema_conf, 4),
                    }
                else:
                    status["action"] = "hold"
                    status["message"] = f"Drift throttle active: waiting {secs_left}s before next trade."
                    self._log_status_event("risk", status)
                    self._publish_status(status)
                    return status
            else:
                status["action"] = "hold"
                status["message"] = f"Drift throttle active: waiting {secs_left}s before next trade."
                self._log_status_event("risk", status)
                self._publish_status(status)
                return status

        if gemini_influence_enabled and self.gemini.enabled and chosen.side in {"buy", "sell"} and float(chosen.confidence) > 0.0:
            influence = self.gemini.influence_trade_decision(
                {
                    "signals": strategy_view,
                    "chosen": status["chosen"],
                    "risk": status["risk"],
                    "market": {"last_close": status["market"]["last_close"]},
                },
                max_confidence_boost=self.settings.gemini_max_confidence_boost,
                max_confidence_cut=self.settings.gemini_max_confidence_cut,
            )
            status["gemini_influence"] = influence

            adjusted_conf = max(0.0, min(1.0, float(chosen.confidence) + float(influence.get("confidence_delta", 0.0))))
            chosen.confidence = adjusted_conf
            status["chosen"]["confidence"] = adjusted_conf

            if bool(influence.get("veto", False)):
                status["action"] = "blocked"
                status["message"] = f"Gemini vetoed trade: {influence.get('reason', 'no reason provided')}"
                self._log_status_event("risk", status)
                self._publish_status(status)
                logger.warning(status["message"])
                return status

        if chosen.side in {"buy", "sell"} and higher_tf_bias in {"buy", "sell"} and higher_tf_bias != chosen.side:
            chosen.confidence = max(0.0, chosen.confidence * 0.85)
            status["chosen"]["confidence"] = chosen.confidence
            status["higher_timeframe"]["penalty_applied"] = True
            if bool(self.settings.require_higher_tf_alignment):
                status["action"] = "blocked"
                status["message"] = (
                    f"Higher-timeframe alignment required: chosen={chosen.side}, htf_bias={higher_tf_bias}."
                )
                self._log_status_event("risk", status)
                self._publish_status(status)
                return status

        active_support_count = self._same_side_support_count(signals, chosen.side)
        raw_support_count = self._same_side_support_count(raw_signals, chosen.side)
        support_count = max(active_support_count, raw_support_count)
        status["same_side_support"] = {
            "side": chosen.side,
            "count": int(support_count),
            "active_count": int(active_support_count),
            "raw_count": int(raw_support_count),
            "source": "raw_directional_support" if raw_support_count > active_support_count else "active_signals",
        }
        if chosen.side in {"buy", "sell"} and support_count < max(1, int(self.settings.min_same_side_signals)):
            status["action"] = "blocked"
            status["message"] = (
                f"Consensus guard blocked trade: side={chosen.side}, support={support_count}, "
                f"required={int(self.settings.min_same_side_signals)}."
            )
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        conviction_view = build_structured_decision_analysis(status=status, runtime=runtime_flags)
        conviction_score = int(conviction_view.get("confidence_score", 0) or 0)
        chosen.confidence = max(0.0, min(1.0, float(conviction_score) / 100.0))
        status["chosen"]["confidence"] = float(chosen.confidence)
        status["atlas_conviction"] = {
            "score": conviction_score,
            "level": str(conviction_view.get("confidence_level", "low") or "low"),
            "basis": str(conviction_view.get("confidence_basis", "atlas_conviction") or "atlas_conviction"),
            "signal_confidence_seed_score": int(conviction_view.get("signal_confidence_seed_score", 0) or 0),
            "adjustments": list(conviction_view.get("conviction_adjustments", []) or []),
        }

        min_conf = self._min_conf_threshold(regime)
        dynamic_min_conf, wr_focus = self._dynamic_min_confidence(min_conf, loss_memory)
        if str(safety_layer.get("volatility_state", "")) == "elevated":
            dynamic_min_conf = min(1.0, dynamic_min_conf + 0.05)
            status["volatility_threshold_adjustment"] = {
                "applied": True,
                "reason": "elevated_volatility",
                "delta": 0.05,
            }
        if bool(self.settings.severe_drift_hard_mode) and bool(drift.get("severe", False)):
            dynamic_min_conf = min(1.0, dynamic_min_conf + 0.05)
            status["drift_hard_mode"] = {"enabled": True, "min_confidence_bump": 0.05}
        thresholds = get_active_thresholds(self.settings)
        # LEARNING MODE: keep confidence threshold coherence across signal,
        # trader, and analysis layers by exposing both the adaptive raw threshold
        # and the active-profile effective threshold.
        effective_min_conf_threshold = min(float(dynamic_min_conf), float(thresholds["confidence"]))
        status["min_confidence_threshold"] = effective_min_conf_threshold
        status["min_confidence_threshold_raw"] = float(dynamic_min_conf)
        # LEARNING MODE: expose active threshold profile in status for dashboard transparency.
        status["learning_mode"] = bool(getattr(self.settings, "learning_mode", False))
        status["mode_label"] = str(thresholds.get("mode_label", "UNKNOWN"))
        status["thresholds"] = {
            "confidence": float(thresholds["confidence"]),
            "structure": float(thresholds["structure"]),
            "spread_max": float(thresholds["spread_max"]),
        }
        status["win_rate_focus"] = wr_focus
        chosen_score = float(contextual_scores.get(self._strategy_key(str(chosen.strategy)), 0.0) or 0.0)
        confidence_gap = round(float(chosen.confidence) - float(effective_min_conf_threshold), 4)
        status["confidence_diagnostics"] = {
            "chosen_strategy": str(chosen.strategy),
            "chosen_side": str(chosen.side),
            "chosen_confidence": round(float(chosen.confidence), 4),
            "chosen_contextual_score": round(chosen_score, 4),
            "min_confidence_threshold": round(float(effective_min_conf_threshold), 4),
            "raw_adaptive_threshold": round(float(dynamic_min_conf), 4),
            "confidence_gap": confidence_gap,
            "same_side_support": int(support_count),
            "dominant_drag_reasons": list(dict(contextual_details.get(self._strategy_key(str(chosen.strategy)), {}) or {}).get("reasons", []))[:4],
            "suggested_focus": (
                "No high-confidence setup because confidence is below threshold."
                if confidence_gap < 0
                else "Threshold cleared, but the selected candidate is neutral/flat."
                if str(chosen.side) == "flat"
                else "Threshold cleared; downstream safety gates decide final action."
            ),
        }
        if weak_blocked:
            status["win_rate_focus"]["weak_strategies_blocked"] = weak_blocked[:5]
        if chosen.side == "flat" or chosen.confidence < effective_min_conf_threshold:
            status["action"] = "hold"
            status["message"] = (
                "No directional trade signal selected."
                if str(chosen.side) == "flat" and float(chosen.confidence) >= float(effective_min_conf_threshold)
                else "No high-confidence trade found."
            )
            self._log_status_event("decision", status)
            self._publish_status(status)
            logger.info(status["message"])
            return status

        timing_discipline = dict(status.get("timing_discipline", {}) or {})
        if bool(timing_discipline.get("hard_block", False)):
            status["action"] = "blocked"
            status["message"] = self._operator_message_from_timing_discipline(timing_discipline)
            self._log_status_event("risk", status)
            self._publish_status(status)
            logger.info(status["message"])
            return status

        online_conf, online_meta = self._online_learning_adjustment(chosen, status)
        status["online_learning_influence"] = online_meta
        if float(online_conf) != float(chosen.confidence):
            chosen.confidence = float(online_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        if bool(online_meta.get("applied", False)) and chosen.confidence < effective_min_conf_threshold:
            status["action"] = "blocked"
            status["message"] = "Online learning reduced confidence below execution threshold."
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status
        loss_learning_now = datetime.now(timezone.utc)
        hour_detail = self._loss_pattern_penalty_detail(chosen.side, loss_learning_now, loss_memory)
        penalty = float(hour_detail.get("penalty", 0.0) or 0.0)
        current_session = str(dict(status.get("context_snapshot", {}) or {}).get("session", "") or "")
        side_detail = self._recent_side_penalty_detail(chosen.side, loss_memory, current_session=current_session)
        side_penalty = float(side_detail.get("penalty", 0.0) or 0.0)
        recovery_evidence = self._loss_learning_recovery_evidence(
            side=chosen.side,
            status=status,
            hour_detail=hour_detail,
            side_detail=side_detail,
        )
        recovery_factor = float(recovery_evidence.get("reduction_factor", 1.0) or 1.0)
        adjusted_penalty = float(penalty) * recovery_factor
        adjusted_side_penalty = float(side_penalty) * recovery_factor
        total_penalty = min(0.42, adjusted_penalty + adjusted_side_penalty)
        if total_penalty > 0:
            chosen.confidence = max(0.0, chosen.confidence * (1.0 - total_penalty))
            status["chosen"]["confidence"] = chosen.confidence
            reasons = [str(x.get("reason", "") or "") for x in [hour_detail, side_detail] if x.get("reason")]
            applied_details = {
                "hour_pattern": hour_detail,
                "recent_side": side_detail,
            }
            evidence_required = []
            for detail in applied_details.values():
                evidence_required.extend(str(x) for x in list(detail.get("evidence_required", []) or []))
            status["loss_learning"] = {
                "applied": True,
                "penalty": round(total_penalty, 4),
                "components": {
                    "hour_pattern_penalty": round(adjusted_penalty, 4),
                    "recent_side_penalty": round(adjusted_side_penalty, 4),
                    "raw_hour_pattern_penalty": round(penalty, 4),
                    "raw_recent_side_penalty": round(side_penalty, 4),
                },
                "details": applied_details,
                "recovery_evidence": recovery_evidence,
                "reason": " | ".join(reasons),
                "clears_when": " and ".join(
                    str(x.get("clears_when", "") or "") for x in [hour_detail, side_detail] if x.get("clears_when") and x.get("applied")
                ),
                "evidence_required": list(dict.fromkeys(evidence_required))[:6],
                "policy_note": "Loss learning can reduce confidence or block repeated weak patterns unless corrective recovery evidence is present; it does not change strategy, risk, or execution policy.",
            }
            if chosen.confidence < dynamic_min_conf:
                status["action"] = "blocked"
                status["message"] = f"Loss-learning guard reduced confidence. {' | '.join(reasons)}"
                self._log_status_event("risk", status)
                self._publish_status(status)
                return status
        elif penalty > 0 or side_penalty > 0:
            status["loss_learning"] = {
                "applied": False,
                "penalty": 0.0,
                "components": {
                    "hour_pattern_penalty": 0.0,
                    "recent_side_penalty": 0.0,
                    "raw_hour_pattern_penalty": round(penalty, 4),
                    "raw_recent_side_penalty": round(side_penalty, 4),
                },
                "details": {
                    "hour_pattern": hour_detail,
                    "recent_side": side_detail,
                },
                "recovery_evidence": recovery_evidence,
                "reason": "Loss-learning penalty cleared by corrective recovery evidence.",
                "policy_note": "Loss learning observed recent losses but did not reduce confidence because current setup shows corrective evidence.",
            }
        loss_block_detail = self._loss_pattern_block_detail(chosen.side, loss_memory, loss_learning_now)
        status["loss_pattern_block_detail"] = dict(loss_block_detail)
        loss_block_reason = str(loss_block_detail.get("reason", "") or "")
        if loss_block_reason:
            status["action"] = "blocked"
            status["message"] = loss_block_reason
            loss_meta = dict(status.get("loss_learning", {}) or {})
            loss_meta["hard_blocked"] = True
            loss_meta["hard_block"] = loss_block_detail
            loss_meta.setdefault(
                "policy_note",
                "Loss learning can reduce confidence or block repeated weak patterns; it does not change strategy, risk, or execution policy.",
            )
            status["loss_learning"] = loss_meta
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        conf_mult, conf_reason = self._confidence_multiplier(chosen.side, snapshot, guard_state, regime)
        if conf_mult != 1.0:
            chosen.confidence = max(0.0, min(1.0, chosen.confidence * conf_mult))
            status["chosen"]["confidence"] = chosen.confidence
            status["adaptation"] = {
                "confidence_multiplier": round(conf_mult, 3),
                "reason": conf_reason,
            }
            if chosen.confidence < effective_min_conf_threshold:
                status["action"] = "blocked"
                status["message"] = "Adaptive guard reduced confidence below execution threshold."
                self._log_status_event("risk", status)
                self._publish_status(status)
                return status

        # Side-switch trainer: when a fresh EMA cross appears, prefer that side
        # to help the system learn to rotate buy/sell instead of getting sticky.
        if self.settings.adaptive_ema_side_switch and ema_signal is not None:
            ema_side = str(getattr(ema_signal, "side", "flat"))
            ema_conf = float(getattr(ema_signal, "confidence", 0.0) or 0.0)
            if ema_side in {"buy", "sell"} and ema_conf > 0.0 and chosen.side != ema_side:
                chosen.side = ema_side
                chosen.confidence = max(float(chosen.confidence), min(1.0, ema_conf))
                chosen.stop_loss_points = float(getattr(ema_signal, "stop_loss_points", chosen.stop_loss_points))
                chosen.take_profit_points = float(getattr(ema_signal, "take_profit_points", chosen.take_profit_points))
                status["chosen"]["side"] = chosen.side
                status["chosen"]["confidence"] = chosen.confidence
                status["side_switch_training"] = {
                    "applied": True,
                    "reason": "Fresh EMA crossover side override",
                    "ema_side": ema_side,
                    "ema_confidence": round(ema_conf, 4),
                }

        weak_regime = str(regime.get("name", "unknown")) != "trend" or float(regime.get("trend_strength", 0.0) or 0.0) < 0.18
        ema_guard_required = bool(self.settings.require_ema_fresh_cross) and (weak_regime or bool(drift.get("active", False)))
        if ema_guard_required and chosen.side in {"buy", "sell"}:
            ema_side = str(getattr(ema_signal, "side", "flat")) if ema_signal is not None else "flat"
            ema_conf = float(getattr(ema_signal, "confidence", 0.0) or 0.0) if ema_signal is not None else 0.0
            if ema_side != chosen.side or ema_conf <= 0.0:
                if bool(self.settings.severe_drift_hard_mode) and bool(drift.get("severe", False)):
                    status["action"] = "blocked"
                    status["message"] = (
                        f"EMA hard guard blocked during severe drift: chosen={chosen.side}, "
                        f"ema={ema_side}, ema_conf={ema_conf:.3f}."
                    )
                    self._log_status_event("risk", status)
                    self._publish_status(status)
                    return status
                # Soft guard for stronger setups: apply a confidence haircut instead of hard block.
                # Hard-block only if the post-haircut confidence is too close to threshold.
                pre_conf = float(chosen.confidence)
                chosen.confidence = max(0.0, pre_conf * 0.92)
                status["chosen"]["confidence"] = chosen.confidence
                status["ema_fresh_cross_guard"] = {
                    "enabled": True,
                    "mode": "soft",
                    "required": True,
                    "chosen_side": chosen.side,
                    "ema_side": ema_side,
                    "ema_confidence": round(ema_conf, 4),
                    "confidence_before": round(pre_conf, 4),
                    "confidence_after": round(float(chosen.confidence), 4),
                }
                if chosen.confidence < (dynamic_min_conf + 0.05):
                    status["action"] = "blocked"
                    status["message"] = (
                        f"EMA guard blocked weak setup after penalty: chosen={chosen.side}, "
                        f"ema={ema_side}, ema_conf={ema_conf:.3f}, conf={chosen.confidence:.3f}."
                    )
                    self._log_status_event("risk", status)
                    self._publish_status(status)
                    return status

        balanced_conf, balance_meta = self._apply_side_balance_penalty(chosen)
        if float(balanced_conf) != float(chosen.confidence):
            chosen.confidence = float(balanced_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        status["side_balance"] = balance_meta
        if chosen.confidence < dynamic_min_conf:
            status["action"] = "blocked"
            status["message"] = "Side-balance guard reduced confidence below threshold."
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        cluster_meta = self._execution_cluster_guard(chosen)
        status["execution_cluster"] = cluster_meta
        if bool(cluster_meta.get("applied", False)):
            status["action"] = "blocked"
            status["message"] = f"Execution cluster guard blocked trade: {cluster_meta.get('reason', 'cluster risk')}."
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        smart_conf, smart_meta = self._smart_context_adjustment(
            chosen=chosen,
            memory=loss_memory,
            regime=regime,
            higher_tf_bias=higher_tf_bias,
            now_utc=datetime.now(timezone.utc),
        )
        if float(smart_conf) != float(chosen.confidence):
            chosen.confidence = float(smart_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        status["smart_context"] = smart_meta
        if chosen.confidence < dynamic_min_conf:
            status["action"] = "blocked"
            status["message"] = "Smart-context guard reduced confidence below threshold."
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        m1_conf, m1_meta = self._m1_quality_adjustment(
            chosen=chosen,
            df=df,
            snapshot=snapshot,
            regime=regime,
            higher_tf_bias=higher_tf_bias,
            now_utc=datetime.now(timezone.utc),
        )
        if float(m1_conf) != float(chosen.confidence):
            chosen.confidence = float(m1_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        # FIX: apply gold-specific knowledge before threshold checks.
        enhanced = apply_gold_knowledge_to_signal(
            {
                "confidence": float(chosen.confidence),
                "direction": str(chosen.side),
                "setup_type": str(self._strategy_family(str(chosen.strategy))),
            },
            {
                "session": self._session_label(datetime.now(timezone.utc)),
                "dxy_direction": str(context_snapshot.get("dxy_alignment", "flat") or "flat"),
                "us10y_direction": str(context_snapshot.get("us10y_alignment", "flat") or "flat"),
                "vix_state": str(context_snapshot.get("vix_regime", "normal") or "normal"),
            },
        )
        chosen.confidence = float(enhanced.get("confidence", chosen.confidence) or chosen.confidence)
        status["chosen"]["confidence"] = float(chosen.confidence)
        status["knowledge_boosts"] = list(enhanced.get("boosts", []) or [])
        status["knowledge_penalties"] = list(enhanced.get("penalties", []) or [])
        status["m1_quality"] = m1_meta
        thresholds = get_active_thresholds(self.settings)
        # LEARNING MODE: threshold from get_active_thresholds()
        effective_conf_threshold = float(dynamic_min_conf)
        effective_conf_threshold = min(float(dynamic_min_conf), float(thresholds["confidence"]))
        if chosen.confidence < effective_conf_threshold:
            status["action"] = "blocked"
            status["message"] = "no_clear_setup"
            status["block_reason_detail"] = self._build_block_reason_detail(
                {
                    "confidence_ok": False,
                    "confidence": float(chosen.confidence),
                    "confidence_threshold": float(effective_conf_threshold),
                }
            )
            # FIX: force one small demo trade stream until 20 closed outcomes.
            closed = len(
                [
                    x
                    for x in list(dict(loss_memory or {}).get("recent_outcomes", []) or [])
                    if str(dict(x).get("outcome", "")).lower() in {"win", "loss", "draw"}
                ]
            )
            if bool(getattr(self.settings, "learning_mode", False)) and bool(getattr(self.settings, "learning_force_entry_per_session", True)) and closed < int(getattr(self.settings, "learning_graduation_min_trades", 100)) and float(chosen.confidence) >= float(getattr(self.settings, "learning_force_min_confidence", 0.45)):
                status["forced_demo_entry"] = True
                logger.warning("FORCED DEMO ENTRY -- learning mode [confidence=%.2f]", float(chosen.confidence))
            else:
                status["forced_demo_entry"] = False
                self._log_status_event("risk", status)
                self._publish_status(status)
                return status
        structure_block_reason = self._directional_structure_block_reason(
            side=str(chosen.side),
            chart_features=dict(status.get("chart_features", {}) or {}),
            context_snapshot=context_snapshot,
            reversal_detection=dict(status.get("reversal_detection", {}) or {}),
        )
        status["directional_structure_guard"] = {
            "enabled": True,
            "hard_blocked": bool(structure_block_reason),
            "reason": structure_block_reason,
            "chosen_side": str(chosen.side),
            "trend_structure": str(
                dict(status.get("chart_features", {}) or {}).get(
                    "trend_structure",
                    dict(context_snapshot or {}).get("trend_structure", "unknown"),
                )
            ),
            "trend_state": str(
                dict(status.get("chart_features", {}) or {}).get(
                    "trend_state",
                    dict(context_snapshot or {}).get("trend_state", "unknown"),
                )
            ),
        }
        if structure_block_reason:
            status["action"] = "blocked"
            status["message"] = structure_block_reason
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status
        m1_block_reason = self._m1_entry_block_reason(chosen, m1_meta)
        if m1_block_reason:
            status["action"] = "blocked"
            status["message"] = m1_block_reason
            status["m1_quality"]["hard_blocked"] = True
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status
        overlay_meta = self._research_regime_overlay_meta(df, snapshot, datetime.now(timezone.utc))
        status["research_regime_overlay"] = overlay_meta
        overlay_reason = research_regime_filter_reason(self.settings, overlay_meta)
        if overlay_reason:
            status["action"] = "blocked"
            status["message"] = overlay_reason
            status["research_regime_overlay"]["hard_blocked"] = True
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status
        conservative_reason = conservative_pretrade_filter_reason(self.settings, overlay_meta)
        status["conservative_pretrade_filter"] = {
            "enabled": bool(getattr(self.settings, "conservative_filter_enabled", False)),
            "reason": conservative_reason,
            "inputs": {
                "session": overlay_meta.get("session"),
                "regime_name": overlay_meta.get("regime_name"),
                "volume_ratio": overlay_meta.get("volume_ratio"),
                "spike_ratio": overlay_meta.get("spike_ratio"),
            },
        }
        if conservative_reason:
            status["action"] = "blocked"
            status["message"] = conservative_reason
            status["conservative_pretrade_filter"]["hard_blocked"] = True
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status

        win_conf, win_meta = self._win_memory_boost(chosen, datetime.now(timezone.utc))
        if float(win_conf) != float(chosen.confidence):
            chosen.confidence = float(win_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        status["win_memory"] = win_meta

        chosen_key = self._strategy_key(chosen.strategy)
        chosen_detail = dict(contextual_details.get(chosen_key, {}) or {})
        chosen_context = dict(chosen_detail.get("context", {}))
        setup_quality_filter_meta = self._setup_quality_filter_meta(
            chosen=chosen,
            chosen_context=chosen_context,
            chosen_detail=chosen_detail,
        )
        status["setup_quality_filter"] = setup_quality_filter_meta
        if bool(setup_quality_filter_meta.get("hard_blocked", False)):
            status["action"] = "blocked"
            status["message"] = str(setup_quality_filter_meta.get("reason", "Setup-quality filter blocked trade."))
            self._log_status_event("risk", status)
            self._publish_status(status)
            return status
        mm_conf, mm_meta = self._meta_model_adjustment(
            chosen=chosen,
            regime=regime,
            higher_tf_bias=higher_tf_bias,
            snapshot=snapshot,
            when_utc=datetime.now(timezone.utc),
            context=chosen_context,
        )
        if float(mm_conf) != float(chosen.confidence):
            chosen.confidence = float(mm_conf)
            status["chosen"]["confidence"] = float(chosen.confidence)
        status["meta_model"] = mm_meta

        structured_analysis = build_structured_decision_analysis(status=status, runtime=runtime_flags)
        status["decision_analysis"] = structured_analysis
        status["pre_trade_memo"] = dict(structured_analysis.get("pre_trade_memo", {}) or {})
        decision_id = uuid4().hex[:8]

        if not execute_trades:
            status["action"] = "paper"
            status["message"] = "Trade candidate found, but EXECUTE_TRADES is disabled."
            status["execution"] = {
                "success": False,
                "retcode": 0,
                "comment": "Enable EXECUTE_TRADES=true in .env for live order execution",
            }
            self.store.log_event(
                "decision_sample",
                "Captured paper decision sample.",
                self._build_decision_sample_payload(
                    decision_id=decision_id,
                    chosen=chosen,
                    setup_family=self._strategy_to_setup_family(str(chosen.strategy)),
                    status=status,
                    regime=regime,
                    higher_tf_bias=higher_tf_bias,
                    snapshot=snapshot,
                    support_count=support_count,
                    signal_count=len(signals),
                    contextual_detail=dict((contextual_details.get(chosen_key, {}) or {})),
                    meta_model=mm_meta,
                    execution={"execution_success": False, "execution_mode": "paper"},
                ),
            )
            self._log_status_event("paper", status)
            self._publish_status(status)
            logger.info(status["message"])
            return status

        symbol_info = mt5.symbol_info(self.settings.symbol)
        tick_value = symbol_info.trade_tick_value if symbol_info else 1.0
        base_volume = self.risk.lot_size(snapshot.equity, chosen.stop_loss_points, tick_value)
        vol_mult, vol_reason = self._volume_multiplier(guard_state, regime)
        size_band_mult, size_band_meta = self._size_band_multiplier(structured_analysis, guard_state, regime)
        volume = base_volume * vol_mult * size_band_mult
        size_tier_cap = float(size_tier_profile.get("max_lot", 0.01) or 0.01)
        if volume > size_tier_cap:
            size_band_meta.setdefault("rationale", [])
            size_band_meta["rationale"] = list(size_band_meta.get("rationale", []) or []) + [
                f"Size capped by account size tier ({size_tier_profile.get('current_size_tier')}).",
            ]
            size_band_meta["size_tier_cap_lot"] = round(size_tier_cap, 4)
            volume = size_tier_cap
        status["size_tier"] = {
            "current_size_tier": str(size_tier_profile.get("current_size_tier", "")),
            "next_size_tier_requirement": str(size_tier_profile.get("next_size_tier_requirement", "")),
            "size_tier_reason": str(size_tier_profile.get("size_tier_reason", "")),
            "size_tier_cap_lot": round(size_tier_cap, 4),
        }
        # Demo/live-like execution must route through ExecutionGate only.
        gate_env = "demo" if bool(execute_trades) else "paper"
        setup_family = self._strategy_to_setup_family(str(chosen.strategy))
        self.execution_gate.set_execution_environment(gate_env)
        req = self.execution_gate.create_request(
            environment=gate_env,
            action_type="place_trade",
            payload_summary=(
                f"instrument={self.settings.symbol} setup_family={setup_family} setup={setup_family} "
                f"strategy={chosen.strategy} side={chosen.side} size={float(volume):.4f} "
                f"confidence={float(chosen.confidence):.3f} stop_loss_points={float(chosen.stop_loss_points):.2f} "
                f"take_profit_points={float(chosen.take_profit_points):.2f} data_source=broker_runtime "
                f"environment_mode={gate_env} decision_id={decision_id}"
            ),
            requested_by="trader",
            dry_run=False,
            auto_generated=False,
            request_origin="trader_guarded",
            strict_runtime_checks=True,
        )
        # P0 — Fix 1: Trader NEVER self-approves. create_request -> execute only.
        # Guarded demo may auto-approve only inside the bounded policy below.
        req_id = str(req.get("id", "") or "")
        auto_approval = self._guarded_demo_auto_approval_decision(
            environment=gate_env,
            status=status,
            setup_family=setup_family,
            volume=float(volume),
            runtime_flags=runtime_flags,
        )
        status["guarded_demo_auto_approval"] = dict(auto_approval)
        if req_id and bool(auto_approval.get("approved", False)):
            approved_req = self.execution_gate.approve_request(
                req_id,
                approved_by="trader_guarded_auto_policy",
                note=str(auto_approval.get("policy", "guarded_demo_auto_approval")),
                explicit_live_confirmation=True,
            )
            status["guarded_demo_auto_approval"]["request_id"] = req_id
            status["guarded_demo_auto_approval"]["approval_status"] = str(
                dict(approved_req or {}).get("approval_status", "unknown")
            )
        gate_runtime_flags = dict(runtime_flags or {})
        cycle_age_ms = self._current_cycle_age_ms()
        gate_runtime_flags["freshness_runtime"] = self._freshness_runtime_payload(status_age_ms=cycle_age_ms, max_age_ms=120000)
        gate_runtime_flags["max_freshness_age_ms"] = 120000
        gate_out = self.execution_gate.execute(
            req_id,
            runtime_flags=gate_runtime_flags,
            risk_policy={"daily_loss_cap_pct": float(self.settings.max_daily_drawdown * 100.0)},
        ) if req_id else {"ok": False, "reason": "request_not_created"}
        gate_result = gate_out.get("result", {})
        broker_order_id = str(gate_result.get("broker_order_id", "") or "")
        broker_confirmed = bool(
            broker_order_id
            or int(gate_result.get("order_id", 0) or 0) > 0
            or int(gate_result.get("deal_id", 0) or 0) > 0
        )
        gate_success = bool(gate_out.get("ok", False))
        if gate_success and bool(execute_trades) and not broker_confirmed:
            gate_success = False
            gate_out = dict(gate_out)
            gate_out["reason"] = "broker_confirmation_missing"
        result = type(
            "GateExecutionResult",
            (),
            {
                "success": gate_success,
                "retcode": int(gate_result.get("retcode", 0)) if gate_success else -5,
                "comment": str(gate_out.get("reason") or gate_result.get("status", "guarded_execution")),
                "order_id": int(gate_result.get("order_id", 0)),
                "deal_id": int(gate_result.get("deal_id", 0)),
                "broker_order_id": broker_order_id,
            },
        )()
        self.store.log_event(
            "decision_sample",
            "Captured live decision sample.",
            self._build_decision_sample_payload(
                decision_id=decision_id,
                chosen=chosen,
                setup_family=setup_family,
                status=status,
                regime=regime,
                higher_tf_bias=higher_tf_bias,
                snapshot=snapshot,
                support_count=support_count,
                signal_count=len(signals),
                contextual_detail=dict((contextual_details.get(chosen_key, {}) or {})),
                meta_model=mm_meta,
                execution={
                    "execution_success": bool(result.success),
                    "execution_mode": gate_env,
                    "retcode": int(result.retcode),
                    "order_id": int(result.order_id),
                    "deal_id": int(result.deal_id),
                },
            ),
        )

        gate_block_reasons = {"approval_missing", "live_requires_explicit_confirmation"}
        execution_blocked = (not bool(result.success)) and str(result.comment) in gate_block_reasons
        status["action"] = "execute" if result.success else ("blocked" if execution_blocked else "failed")
        status["volume"] = volume
        status["execution"] = {
            "success": result.success,
            "retcode": result.retcode,
            "comment": result.comment,
            "decision_id": decision_id,
            "order_id": int(result.order_id),
            "deal_id": int(result.deal_id),
            "broker_order_id": str(result.broker_order_id),
        }
        status["message"] = (
            f"Executed {chosen.side} via {chosen.strategy}"
            if result.success
            else (
                f"Execution blocked for {chosen.side} via {chosen.strategy}: {result.comment}"
                if execution_blocked
                else f"Execution failed for {chosen.side} via {chosen.strategy}"
            )
        )
        status["journal"] = {
            "strategy": chosen.strategy,
            "side": chosen.side,
            "confidence_final": round(float(chosen.confidence), 4),
            "regime": regime,
            "session": self._session_label(datetime.now(timezone.utc)),
            "spread_bucket": self._spread_bucket(float(snapshot.spread_points)),
            "volatility_bucket": self._volatility_bucket(float(regime.get("volatility_pct", 0.0))),
            "setup_type": "trend_follow" if str(regime.get("name", "")) == "trend" else "mean_revert_or_range",
            "spread_points": round(float(snapshot.spread_points), 2),
            "risk_state": {
                "trades_today": int(guard_state.get("trades_today", 0)),
                "consecutive_losses": int(guard_state.get("consecutive_losses", 0)),
            },
            "volume_plan": {
                "base_volume": round(float(base_volume), 4),
                "multiplier": round(float(vol_mult), 3),
                "size_band_multiplier": round(float(size_band_mult), 3),
                "final_volume": round(float(volume), 4),
                "reason": vol_reason,
                "size_band_rationale": size_band_meta,
            },
            "exit_reason_hint": "broker_managed_after_entry" if result.success else str(result.comment),
        }

        status["learning_status"] = self._learning_status_snapshot()
        status["learning_status_snapshot"] = status["learning_status"]

        gemini_note = self.gemini.summarize_market_context(
            {
                "signals": strategy_view,
                "chosen": status["chosen"],
                "risk": status["risk"],
                "execution": status["execution"],
            }
        )
        status["gemini_note"] = gemini_note

        self._log_status_event("execution", status)
        self._publish_status(status)
        self._set_drift_throttle_after_execution(drift, datetime.now(timezone.utc), runtime_flags)

        # P2 — Voice announcements for execution events (one-way only)
        if result.success:
            self.voice.speak(f"Trade executed. {chosen.side} {chosen.strategy} at confidence {chosen.confidence:.0%}.")
        elif execution_blocked:
            self.voice.speak(f"Trade execution blocked. Reason: {result.comment}.")
        else:
            self.voice.speak(f"Trade execution failed. Reason: {result.comment}.")

        counts_as_trade = self._execution_result_counts_as_trade(result)
        if counts_as_trade:
            guard_state["trades_today"] = int(guard_state.get("trades_today", 0)) + 1
            # Outcome PnL is unknown at send time; use broker retcode as a stability proxy.
            guard_state["consecutive_losses"] = 0
            guard_state["cooldown_until"] = ""
        else:
            # Preflight/gate/broker rejects are not filled trades. They must not advance
            # trade/loss counters or trigger loss-streak cooldowns.
            self.store.log_event(
                "learning",
                "Execution reject not counted as trade outcome.",
                {
                    "strategy": chosen.strategy,
                    "retcode": result.retcode,
                    "comment": str(result.comment),
                    "counted_as_trade": False,
                },
            )
        self._save_guard_state(guard_state, runtime_flags)

        logger.info(
            "Signal=%s side=%s confidence=%.3f volume=%.2f result=%s (%s)",
            chosen.strategy,
            chosen.side,
            chosen.confidence,
            volume,
            result.retcode,
            result.comment,
        )
        return status