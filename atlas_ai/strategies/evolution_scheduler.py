"""EvolutionScheduler: safe background daemon for strategy evolution.

Orchestrates EvolutionEngine, ShadowTester, PromotionGovernor,
EvolutionDrawdownGuard, and EvolutionReporter on a periodic schedule.

Safety invariants:
- Never executes trades directly.
- Never bypasses ExecutionGate.
- Runs in a background daemon thread; never blocks live trading.
- All failures are caught, logged, and recovered from.
- Live-money promotion always requires human approval.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_ai.config.settings import Settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_drawdown_guard import EvolutionDrawdownGuard
from atlas_ai.strategies.evolution_engine import EvolutionEngine
from atlas_ai.strategies.evolution_reporter import EvolutionReporter
from atlas_ai.strategies.promotion_governor import PromotionGovernor
from atlas_ai.brain.ai_decision_core import AIDecisionCore
from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
from atlas_ai.brain.ai_decision_reviewer import AIDecisionReviewer
from atlas_ai.brain.ai_promotion_intelligence import AIPromotionIntelligence
from atlas_ai.brain.ai_research_queue import AIResearchQueue
from atlas_ai.brain.cognitive_core import build_cognitive_core
from atlas_ai.brain.adaptation import build_adaptation_summary
from atlas_ai.decision_support import build_evidence_separation
from atlas_ai.strategies.shadow_tester import ShadowOutcome, ShadowTester

logger = logging.getLogger(__name__)


@dataclass
class SchedulerState:
    """Persisted scheduler runtime state."""

    scheduler_status: str = "idle"  # idle / running / failed
    current_generation: int = 0
    last_evolution_run_at: str = ""
    next_evolution_run_at: str = ""
    best_fitness: float = 0.0
    best_uid: str = ""
    active_demo_count: int = 0
    active_shadow_count: int = 0
    promotions_this_cycle: int = 0
    rejections_this_cycle: int = 0
    last_error: str = ""
    last_error_at: str = ""
    total_runs: int = 0
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_status": self.scheduler_status,
            "current_generation": self.current_generation,
            "last_evolution_run_at": self.last_evolution_run_at,
            "next_evolution_run_at": self.next_evolution_run_at,
            "best_fitness": self.best_fitness,
            "best_uid": self.best_uid,
            "active_demo_count": self.active_demo_count,
            "active_shadow_count": self.active_shadow_count,
            "promotions_this_cycle": self.promotions_this_cycle,
            "rejections_this_cycle": self.rejections_this_cycle,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at,
            "total_runs": self.total_runs,
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SchedulerState":
        return cls(
            scheduler_status=str(payload.get("scheduler_status", "idle")),
            current_generation=int(payload.get("current_generation", 0)),
            last_evolution_run_at=str(payload.get("last_evolution_run_at", "")),
            next_evolution_run_at=str(payload.get("next_evolution_run_at", "")),
            best_fitness=float(payload.get("best_fitness", 0.0)),
            best_uid=str(payload.get("best_uid", "")),
            active_demo_count=int(payload.get("active_demo_count", 0)),
            active_shadow_count=int(payload.get("active_shadow_count", 0)),
            promotions_this_cycle=int(payload.get("promotions_this_cycle", 0)),
            rejections_this_cycle=int(payload.get("rejections_this_cycle", 0)),
            last_error=str(payload.get("last_error", "")),
            last_error_at=str(payload.get("last_error_at", "")),
            total_runs=int(payload.get("total_runs", 0)),
            dry_run=bool(payload.get("dry_run", False)),
        )


class EvolutionScheduler:
    """Background scheduler for autonomous strategy evolution.

    Runs evolve_generation on a periodic interval, applies shadow testing,
    promotion rules, drawdown guards, and writes reporting artifacts.
    """

    def __init__(
        self,
        settings: Settings,
        store: KnowledgeStore,
        stop_event: threading.Event,
        *,
        interval_hours: float | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings
        self.store = store
        self.stop_event = stop_event
        self.dry_run = dry_run
        self.interval_seconds = max(
            300.0,  # Minimum 5 minutes between runs
            (interval_hours if interval_hours is not None else settings.evolution_interval_hours)
            * 3600.0,
        )
        self.engine = EvolutionEngine(settings, store)
        self.ai_decision_core = AIDecisionCore(self.engine.inventor)
        self.ai_decision_journal = AIDecisionJournal(store.base)
        self.ai_decision_reviewer = AIDecisionReviewer(store.base)
        self.shadow_tester = ShadowTester()
        self.governor = PromotionGovernor(
            max_demo_strategies=settings.max_demo_strategies,
            max_risk_per_trade_pct=settings.evolved_max_risk_pct,
            min_shadow_trades=settings.evolution_min_shadow_trades,
            max_shadow_drawdown_pct=settings.evolution_max_shadow_drawdown_pct,
            min_backtest_score=60.0,
            min_demo_trades=settings.evolution_min_demo_trades,
            min_demo_win_rate=settings.evolution_min_demo_win_rate,
            min_demo_profit_factor=settings.evolution_min_demo_profit_factor,
            live_promotion_requires_approval=settings.live_promotion_requires_approval,
        )
        self.ai_promotion_intelligence = AIPromotionIntelligence(store.base, governor=self.governor)
        self.ai_research_queue = AIResearchQueue(store.base)
        self.drawdown_guard = EvolutionDrawdownGuard(
            max_demo_drawdown_pct=settings.evolution_max_demo_drawdown_pct,
            max_shadow_drawdown_pct=settings.evolution_max_shadow_drawdown_pct,
        )
        self.reporter = EvolutionReporter(output_dir=store.base)
        self._state_file = Path(store.base) / "evolution_scheduler_state.json"
        self._shadow_outcomes: dict[str, list[ShadowOutcome]] = {}
        self._thread: threading.Thread | None = None
        self._state = self._load_state()
        self._state.dry_run = dry_run

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> SchedulerState:
        if not self._state_file.exists():
            return SchedulerState()
        try:
            payload = json.loads(self._state_file.read_text(encoding="utf-8"))
            return SchedulerState.from_dict(payload)
        except Exception:
            return SchedulerState()

    def _save_state(self) -> None:
        try:
            self._state_file.write_text(
                json.dumps(self._state.to_dict(), indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save scheduler state: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> SchedulerState:
        return self._state

    def start(self) -> None:
        """Start the scheduler background thread if not already running."""
        if self._thread is not None and self._thread.is_alive():
            logger.info("EvolutionScheduler already running; skipping duplicate start.")
            return
        self._thread = threading.Thread(target=self._loop, name="evolution_scheduler", daemon=True)
        self._thread.start()
        self.store.log_event(
            "evolution",
            f"EvolutionScheduler started. Interval={self.interval_seconds/3600:.1f}h dry_run={self.dry_run}",
            self._state.to_dict(),
        )

    def run_once(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Run a single evolution cycle synchronously (for CLI/testing)."""
        return self._run_cycle(df)

    def shutdown(self) -> None:
        """Signal the scheduler to stop cleanly."""
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                self._state.scheduler_status = "running"
                self._save_state()
                self._run_cycle()
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                logger.exception("Evolution cycle failed: %s", error_msg)
                self._state.scheduler_status = "failed"
                self._state.last_error = error_msg
                self._state.last_error_at = datetime.now(timezone.utc).isoformat()
                self.store.log_event(
                    "evolution",
                    f"Evolution cycle failed: {error_msg}",
                    {"error": error_msg, "traceback": str(exc)[:500]},
                )
                self._save_state()

            # Wait for next interval or stop
            if self.stop_event.wait(timeout=self.interval_seconds):
                break

        self._state.scheduler_status = "idle"
        self._save_state()
        self.store.log_event("evolution", "EvolutionScheduler stopped cleanly.")

    # ------------------------------------------------------------------
    # Cycle orchestration
    # ------------------------------------------------------------------

    def _run_cycle(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Execute one full evolution cycle."""
        start_time = datetime.now(timezone.utc)
        self._state.last_evolution_run_at = start_time.isoformat()
        self._state.promotions_this_cycle = 0
        self._state.rejections_this_cycle = 0

        # --- 1. Fetch market data ---
        if df is None:
            try:
                feed = MarketDataFeed(self.settings)
                df = feed.get_rates(max(500, self.settings.backtest_evolved_lookback_bars))
            except Exception as exc:
                raise RuntimeError(f"Market data fetch failed: {exc}") from exc

        if df is None or len(df) < 50:
            raise RuntimeError("Insufficient market data for evolution")

        # --- 2. Seed AI-invented strategy proposals from current Atlas context ---
        invented_seed_count = 0
        if not self.dry_run:
            try:
                status = self.store.load_status()
                runtime = self.store.load_runtime_flags()
                learning_state = self.store.load_learning_state()
                events = self.store.tail_events(limit=160) if hasattr(self.store, "tail_events") else []
                adaptation_summary = build_adaptation_summary(status, events)
                cognitive_core = build_cognitive_core(
                    status=status,
                    runtime=runtime,
                    events=events,
                    learning_state=learning_state,
                    knowledge_payload={},
                    adaptation_summary=adaptation_summary,
                    evidence_separation=build_evidence_separation(status, runtime),
                )
                ai_decision = self.ai_decision_core.decide(
                    status=status,
                    runtime=runtime,
                    events=events,
                    learning_state=learning_state,
                    cognitive_core=cognitive_core,
                    strategy_preferences=self.ai_decision_reviewer.preferences.load(),
                    mode="evolution_seed",
                ).to_dict()
                self.ai_decision_journal.append(ai_decision, source="evolution_scheduler", query="evolution_seed")
                self.ai_decision_reviewer.review(learning_state=learning_state, events=events, limit=10)
                invented_seed_count = len(
                    self.engine.seed_ai_invented_population(
                        status=status,
                        runtime=runtime,
                        learning_state=learning_state,
                        cognitive_core=cognitive_core,
                        strategy_preferences=self.ai_decision_reviewer.preferences.load(),
                        limit=3,
                    )
                )
                self.store.log_event(
                    "evolution",
                    "AI decision core reviewed evolution seed context.",
                    {
                        "action": dict(ai_decision.get("recommendation", {}) or {}).get("action", ""),
                        "confidence": dict(ai_decision.get("recommendation", {}) or {}).get("confidence", 0.0),
                        "strategy_invention": ai_decision.get("strategy_invention", {}),
                        "safety": ai_decision.get("safety", {}),
                    },
                )
            except Exception as exc:
                self.store.log_event(
                    "evolution",
                    f"AI strategy invention seed skipped: {exc}",
                    {"error": str(exc), "safety": "evolution_continues_without_ai_seed"},
                )

        # --- 3. Run evolution generation ---
        if not self.dry_run:
            population = self.engine.evolve_generation(
                df,
                population_size=getattr(self.settings, "evolved_strategy_population", 18),
                elite_count=3,
            )
        else:
            # Dry-run: just reload existing population without mutation
            population = list(self.engine.population)

        self._state.current_generation = self.engine.generation

        # --- 4. Shadow testing on backtested candidates ---
        shadow_candidates = [
            c for c in population
            if c.promotion_stage in {"backtested", "shadow"}
            and c.backtest_completed
            and c.backtest_score > 0
        ]
        for chromosome in shadow_candidates:
            outcomes = self._run_shadow_testing(chromosome, df)
            if chromosome.uid not in self._shadow_outcomes:
                self._shadow_outcomes[chromosome.uid] = []
            self._shadow_outcomes[chromosome.uid].extend(outcomes)

        # --- 5. PromotionGovernor: backtested → shadow ---
        for chromosome in population:
            if chromosome.promotion_stage == "backtested" and chromosome.backtest_score >= 60.0:
                if not self.dry_run:
                    self.engine.promote_to_shadow(chromosome)

        # --- 6. PromotionGovernor: shadow → demo ---
        for chromosome in population:
            if chromosome.promotion_stage == "shadow":
                outcomes = self._shadow_outcomes.get(chromosome.uid, [])
                ok, reason = self.governor.can_promote_to_demo(
                    chromosome, population, shadow_outcomes=outcomes
                )
                if ok and not self.dry_run:
                    try:
                        self.governor.promote_to_demo(chromosome, population, shadow_outcomes=outcomes)
                        self._state.promotions_this_cycle += 1
                        self.store.log_event(
                            "evolution",
                            f"Promoted {chromosome.uid[:8]} to demo",
                            {"uid": chromosome.uid, "score": chromosome.backtest_score},
                        )
                    except Exception as exc:
                        logger.warning("Promotion to demo failed for %s: %s", chromosome.uid[:8], exc)
                        self._state.rejections_this_cycle += 1
                elif not ok:
                    self._state.rejections_this_cycle += 1

        # --- 7. Drawdown guard on demo strategies ---
        for chromosome in population:
            if chromosome.promotion_stage == "demo" and chromosome.demo_active:
                outcomes = self._shadow_outcomes.get(chromosome.uid, [])
                ok, reason = self.drawdown_guard.check_drawdown(
                    chromosome, outcomes, stage="demo"
                )
                if not ok:
                    self.drawdown_guard.pause_strategy(chromosome, reason)
                    self.store.log_event(
                        "evolution",
                        f"Paused demo strategy {chromosome.uid[:8]}: {reason}",
                        {"uid": chromosome.uid, "reason": reason},
                    )

        # --- 8. Archive rejected ---
        archived = self.engine.archive_rejected()
        if archived:
            self.store.log_event(
                "evolution",
                f"Archived {archived} rejected chromosomes.",
                {"archived": archived},
            )

        # --- 9. Reporter artifacts ---
        if not self.dry_run:
            self.reporter.write_leaderboard(self.engine.population)
            self.reporter.write_evolution_log(self._build_cycle_events(start_time))
            promotion_report = self.ai_promotion_intelligence.evaluate(
                self.engine.population,
                strategy_preferences=self.ai_decision_reviewer.preferences.load(),
            )
        else:
            promotion_report = self.ai_promotion_intelligence.evaluate(
                self.engine.population,
                strategy_preferences=self.ai_decision_reviewer.preferences.load(),
            )
        self.ai_research_queue.sync_from_promotion_report(promotion_report)
        if archived:
            self.ai_research_queue.sync_from_archive(limit=max(archived, 12))

        # --- 10. Update state ---
        best = self._best_chromosome(self.engine.population)
        self._state.best_fitness = best.backtest_score if best else 0.0
        self._state.best_uid = best.uid if best else ""
        self._state.active_demo_count = self.governor.active_demo_count(self.engine.population)
        self._state.active_shadow_count = sum(
            1 for c in self.engine.population if c.promotion_stage == "shadow"
        )
        self._state.total_runs += 1
        next_run = datetime.now(timezone.utc).timestamp() + self.interval_seconds
        self._state.next_evolution_run_at = datetime.fromtimestamp(next_run, tz=timezone.utc).isoformat()
        self._state.scheduler_status = "idle"
        self._save_state()

        cycle_summary = {
            "generation": self._state.current_generation,
            "population_size": len(self.engine.population),
            "best_fitness": self._state.best_fitness,
            "best_uid": self._state.best_uid,
            "promotions": self._state.promotions_this_cycle,
            "rejections": self._state.rejections_this_cycle,
            "active_demo": self._state.active_demo_count,
            "active_shadow": self._state.active_shadow_count,
            "archived": archived,
            "ai_invented_seeded": invented_seed_count,
            "ai_promotion_top": dict(promotion_report.get("top_recommendation", {}) or {}),
            "dry_run": self.dry_run,
            "duration_seconds": round((datetime.now(timezone.utc) - start_time).total_seconds(), 2),
        }
        self.store.log_event("evolution", "Evolution cycle completed.", cycle_summary)
        return cycle_summary

    # ------------------------------------------------------------------
    # Shadow testing helper
    # ------------------------------------------------------------------

    def _run_shadow_testing(
        self,
        chromosome: StrategyChromosome,
        df: pd.DataFrame,
    ) -> list[ShadowOutcome]:
        """Simulate shadow testing over the recent bars of the dataframe.

        Iterates through the last N bars (where N is determined by settings)
        and feeds each bar to ShadowTester.step(). Returns any outcomes
        produced during the walk.
        """
        outcomes: list[ShadowOutcome] = []
        # Use last N bars for shadow testing simulation
        shadow_bars = min(len(df), getattr(self.settings, "backtest_evolved_refresh_bars", 500))
        df_slice = df.tail(shadow_bars).reset_index(drop=True)

        for i in range(1, len(df_slice)):
            if self.stop_event.is_set():
                break
            bar_df = df_slice.iloc[: i + 1]
            current_price = float(bar_df["close"].iloc[-1])
            outcome = self.shadow_tester.step(chromosome, bar_df, current_price)
            if outcome is not None:
                outcomes.append(outcome)

        # Update chromosome shadow metadata
        if outcomes:
            chromosome.shadow_signals = len(outcomes)
            wins = sum(1 for o in outcomes if o.pnl > 0)
            chromosome.shadow_agreement_rate = wins / len(outcomes)
            # Simple degradation: compare first half vs second half win rate
            mid = len(outcomes) // 2
            if mid > 0:
                first_wr = sum(1 for o in outcomes[:mid] if o.pnl > 0) / mid
                second_wr = sum(1 for o in outcomes[mid:] if o.pnl > 0) / max(1, len(outcomes) - mid)
                chromosome.shadow_degradation_pct = max(0.0, first_wr - second_wr)
            chromosome.shadow_passed = True

        return outcomes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_chromosome(population: list[StrategyChromosome]) -> StrategyChromosome | None:
        candidates = [c for c in population if c.backtest_completed and c.backtest_score > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.backtest_score)

    def _build_cycle_events(self, start_time: datetime) -> list[dict[str, Any]]:
        return [
            {
                "time": start_time.isoformat(),
                "generation": self._state.current_generation,
                "best_fitness": self._state.best_fitness,
                "best_uid": self._state.best_uid,
                "population_size": len(self.engine.population),
                "promotions": self._state.promotions_this_cycle,
                "rejections": self._state.rejections_this_cycle,
                "active_demo": self._state.active_demo_count,
            }
        ]


# ------------------------------------------------------------------
# Module-level singleton tracking (prevents duplicate schedulers)
# ------------------------------------------------------------------

_active_schedulers: set[str] = set()
_scheduler_lock = threading.Lock()


def start_evolution_scheduler(
    settings: Settings,
    store: KnowledgeStore,
    stop_event: threading.Event,
    *,
    interval_hours: float | None = None,
    dry_run: bool = False,
) -> EvolutionScheduler | None:
    """Start a singleton EvolutionScheduler if evolution is enabled.

    Returns the scheduler instance, or None if evolution is disabled or
    a scheduler is already active for this storage directory.
    """
    if not settings.evolution_enabled:
        logger.info("Evolution disabled in settings; scheduler not started.")
        return None

    store_key = str(Path(store.base).resolve())
    with _scheduler_lock:
        if store_key in _active_schedulers:
            logger.info("EvolutionScheduler already active for %s; skipping.", store_key)
            return None
        _active_schedulers.add(store_key)

    scheduler = EvolutionScheduler(
        settings=settings,
        store=store,
        stop_event=stop_event,
        interval_hours=interval_hours,
        dry_run=dry_run,
    )
    scheduler.start()
    return scheduler


def stop_evolution_scheduler(scheduler: EvolutionScheduler | None) -> None:
    """Stop an active scheduler and release the singleton lock."""
    if scheduler is None:
        return
    store_key = str(Path(scheduler.store.base).resolve())
    scheduler.shutdown()
    with _scheduler_lock:
        _active_schedulers.discard(store_key)

