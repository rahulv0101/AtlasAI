"""EvolutionEngine: generates, mutates, and backtests strategy chromosomes.

Runs entirely offline. Proposes strategies to the trader; never executes trades.
"""
from __future__ import annotations
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import Settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool
from atlas_ai.strategies.fitness import BacktestResult, FitnessScorer
from atlas_ai.strategies.gold_validator import GoldValidationError, GoldValidator
from atlas_ai.strategies.strategy_inventor import StrategyInventor
from atlas_ai.strategies.strategy_factory import StrategyFactory


class EvolutionEngine:
    """Generates strategy chromosomes via mutation, crossover, and backtest selection.

    All operations are offline. The engine stores its population in
    `storage/evolution_population.json` and logs events to the KnowledgeStore.
    """

    def __init__(
        self,
        settings: Settings,
        store: KnowledgeStore,
        rng: random.Random | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.rng = rng or random.Random()
        self.validator = GoldValidator()
        self.scorer = FitnessScorer(
            min_trades=max(20, getattr(settings, "evolved_candidate_min_trades", 200)),
            min_win_rate=float(getattr(settings, "evolved_candidate_min_win_rate", 0.52)),
            min_profit_factor=1.05,
            max_drawdown_pct=0.10,
        )
        self.inventor = StrategyInventor(self.validator)
        self.population: list[StrategyChromosome] = []
        self.generation = 0
        self._population_file = Path(store.base) / "evolution_population.json"
        self._load_population()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_population(self) -> None:
        if not self._population_file.exists():
            return
        try:
            import json
            payload = json.loads(self._population_file.read_text(encoding="utf-8"))
            self.population = [StrategyChromosome.from_dict(c) for c in payload.get("chromosomes", [])]
            self.generation = int(payload.get("generation", 0))
        except Exception:
            self.population = []
            self.generation = 0

    def _save_population(self) -> None:
        import json
        payload = {
            "generation": self.generation,
            "chromosomes": [c.to_dict() for c in self.population],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._population_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Population operations
    # ------------------------------------------------------------------

    def random_chromosome(self) -> StrategyChromosome:
        genes = make_standard_gene_pool()
        for g in genes:
            g.value = g.allowed_range.random_value(self.rng)
        return StrategyChromosome(
            uid=_new_uid(),
            generation=0,
            genes=genes,
            origin="random",
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def seed_population(self, size: int = 12) -> None:
        """Create an initial random population, validating each member."""
        added = 0
        attempts = 0
        max_attempts = size * 20
        while added < size and attempts < max_attempts:
            attempts += 1
            c = self.random_chromosome()
            ok, reason = self.validator.validate_silent(c)
            if not ok:
                continue
            self.population.append(c)
            added += 1
        self._save_population()
        self.store.log_event(
            "evolution",
            f"Seeded population with {added} valid chromosomes ({attempts} attempts).",
            {"size": added, "attempts": attempts},
        )

    def seed_ai_invented_population(
        self,
        *,
        status: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        learning_state: dict[str, Any] | None = None,
        cognitive_core: dict[str, Any] | None = None,
        strategy_preferences: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> list[StrategyChromosome]:
        """Add Atlas-designed strategy proposals to the offline population.

        This does not backtest, promote, or execute anything. It only inserts
        validator-approved proposal DNA into the normal evolution pipeline.
        """
        plans = self.inventor.invent(
            status=status,
            runtime=runtime,
            learning_state=learning_state,
            cognitive_core=cognitive_core,
            strategy_preferences=strategy_preferences,
            limit=limit,
        )
        added: list[StrategyChromosome] = []
        existing_signatures = {c.signature() for c in self.population}
        for plan in plans:
            chromosome = plan.chromosome
            if chromosome.signature() in existing_signatures:
                continue
            ok, reason = self.validator.validate_silent(chromosome)
            if not ok:
                chromosome.reject_reason = f"ai_inventor_rejected:{reason}"
                continue
            self.population.append(chromosome)
            existing_signatures.add(chromosome.signature())
            added.append(chromosome)

        if added:
            self._save_population()
            self.store.log_event(
                "evolution",
                f"Seeded {len(added)} AI-invented strategy proposals.",
                {
                    "count": len(added),
                    "origins": [c.origin for c in added],
                    "uids": [c.uid for c in added],
                    "safety": "proposal_only_backtest_required",
                },
            )
        return added

    def evolve_generation(
        self,
        df: pd.DataFrame,
        *,
        population_size: int = 12,
        elite_count: int = 3,
        mutation_rate: float = 0.65,
        crossover_rate: float = 0.25,
        mutation_strength: float = 1.0,
    ) -> list[StrategyChromosome]:
        """Run one generation of evolution with backtest fitness scoring.

        Returns the new population sorted by fitness (highest first).
        """
        df = self._bounded_backtest_frame(df)
        if not self.population:
            self.seed_population(population_size)

        # Score any un-backtested members
        for c in self.population:
            if not c.backtest_completed:
                self._backtest_chromosome(c, df)

        # Sort by fitness descending
        scored = [(c, c.backtest_score) for c in self.population if c.backtest_completed and c.backtest_score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Elitism
        new_pop: list[StrategyChromosome] = []
        for c, _ in scored[:elite_count]:
            new_pop.append(c.clone())

        # Breed remainder
        valid_parents = [c for c, _ in scored if c.backtest_score > 0]
        if not valid_parents:
            valid_parents = self.population[:elite_count] if self.population else []

        attempts = 0
        child_backtests = 0
        max_attempts = population_size * 30
        max_child_backtests = max(population_size, int(getattr(self.settings, "evolution_max_child_backtests", population_size * 2) or population_size * 2))
        while len(new_pop) < population_size and attempts < max_attempts:
            attempts += 1
            if child_backtests >= max_child_backtests:
                break
            if valid_parents and self.rng.random() < crossover_rate and len(valid_parents) >= 2:
                p1, p2 = self.rng.sample(valid_parents, 2)
                child = p1.crossover(p2, self.rng)
            elif valid_parents:
                parent = self.rng.choice(valid_parents)
                child = parent.mutate(self.rng, strength=mutation_strength)
            else:
                child = self.random_chromosome()

            if self.rng.random() < mutation_rate:
                child = child.mutate(self.rng, strength=mutation_strength)

            ok, reason = self.validator.validate_silent(child)
            if not ok:
                continue

            child_backtests += 1
            self._backtest_chromosome(child, df)
            if child.backtest_score > 0:
                new_pop.append(child)
            else:
                # Keep a few rejects for diversity
                if len(new_pop) < population_size // 3:
                    new_pop.append(child)

        self.population = new_pop[:population_size]
        self.generation += 1
        self._save_population()

        best = self.population[0] if self.population else None
        self.store.log_event(
            "evolution",
            f"Generation {self.generation} complete. Best score: {best.backtest_score if best else 0}.",
            {
                "generation": self.generation,
                "population_size": len(self.population),
                "best_score": best.backtest_score if best else 0.0,
                "best_uid": best.uid if best else "",
                "child_backtests": child_backtests,
                "max_child_backtests": max_child_backtests,
            },
        )
        return list(self.population)

    def _bounded_backtest_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep evolution cycles bounded for offline candidate evaluation."""
        max_bars = max(300, int(getattr(self.settings, "evolved_backtest_max_bars", 400) or 400))
        if len(df) <= max_bars:
            return df
        return df.tail(max_bars).reset_index(drop=True)

    def _backtest_chromosome(self, chromosome: StrategyChromosome, df: pd.DataFrame) -> None:
        """Run a lightweight backtest on a single chromosome."""
        try:
            strategy = StrategyFactory.build(chromosome)
            result = self._run_single_strategy_backtest(df, strategy)
            self.scorer.update_chromosome(chromosome, result)
        except Exception as exc:
            chromosome.backtest_completed = True
            chromosome.backtest_score = 0.0
            chromosome.promotion_stage = "rejected"
            chromosome.reject_reason = f"backtest_error:{exc}"

    def _run_single_strategy_backtest(self, df: pd.DataFrame, strategy: Any) -> BacktestResult:
        """Evaluate one generated strategy offline using the shared simulator.

        This intentionally stays inside the backtest/simulation surface. It
        creates signals, simulates exits, and summarizes returns; it never
        routes orders or touches the execution gate.
        """
        if df.empty or len(df) < 60:
            return BacktestResult(
                trades=0,
                wins=0,
                losses=0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe=0.0,
                expectancy=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                average_win=0.0,
                average_loss=0.0,
                average_hold_bars=0.0,
            )

        backtester = Backtester(store=None, settings=self.settings)
        initial_equity = max(1.0, float(getattr(self.settings, "backtest_initial_balance", 10000.0) or 10000.0))
        equity = float(initial_equity)
        equity_curve = [equity]
        trade_returns: list[float] = []
        hold_bars: list[int] = []
        regime_stats: dict[str, dict[str, float]] = {}
        min_window = max(50, int(getattr(self.settings, "evolved_candidate_min_window_bars", 120) or 120))
        min_conf = float(getattr(self.settings, "research_confidence_threshold", 0.1) or 0.1)
        strategy_min_conf = float(getattr(strategy, "chromosome", None).gene_value("risk_min_confidence", 0.55) if getattr(strategy, "chromosome", None) is not None else 0.55)
        cooldown_bars = int(float(getattr(strategy, "chromosome", None).gene_value("risk_loss_cooldown_bars", 0.0) if getattr(strategy, "chromosome", None) is not None else 0.0))
        cooldown_lift = float(getattr(strategy, "chromosome", None).gene_value("risk_loss_cooldown_confidence_lift", 0.0) if getattr(strategy, "chromosome", None) is not None else 0.0)
        cooldown_until = -1
        max_start = len(df) - 2
        i = min(min_window, max_start)

        while i < max_start:
            window = df.iloc[: i + 1]
            signal = strategy.generate_signal(window)
            effective_min_conf = max(min_conf, strategy_min_conf + cooldown_lift) if i < cooldown_until else min_conf
            if signal.side == "flat" or float(signal.confidence) < min_conf:
                i += 1
                continue
            if float(signal.confidence) < effective_min_conf:
                i += 1
                continue

            r_multiple, exit_idx, _best_mfe_r = backtester._normalize_simulated_trade(
                backtester._simulate_trade(df, signal, i)
            )
            risk_fraction = float(getattr(self.settings, "max_risk_per_trade", 0.005) or 0.005)
            trade_return = r_multiple * risk_fraction
            spread_cost = float(getattr(self.settings, "backtest_spread_cost_bps", 0.0) or 0.0) / 10000.0
            slippage_cost = float(getattr(self.settings, "backtest_slippage_cost_bps", 0.0) or 0.0) / 10000.0
            trade_return -= spread_cost + slippage_cost

            equity *= max(0.0, 1.0 + trade_return)
            equity_curve.append(equity)
            trade_returns.append(float(trade_return))
            held_bars = max(1, int(exit_idx - i))
            hold_bars.append(held_bars)

            regime_name = backtester._regime_name(list(window["close"].tail(60)))
            bucket = regime_stats.setdefault(
                regime_name,
                {"trades": 0.0, "wins": 0.0, "losses": 0.0, "sum_return": 0.0},
            )
            bucket["trades"] += 1.0
            bucket["sum_return"] += float(trade_return)
            if trade_return > 0:
                bucket["wins"] += 1.0
            elif trade_return < 0:
                bucket["losses"] += 1.0
                if cooldown_bars > 0:
                    cooldown_until = max(cooldown_until, int(exit_idx) + cooldown_bars)

            i = max(i + 1, int(exit_idx))

        stats = backtester._summarize(equity_curve, trade_returns, hold_bars, regime_stats)
        return BacktestResult(
            trades=int(stats["total_trades"]),
            wins=int(stats["wins"]),
            losses=int(stats["losses"]),
            win_rate=float(stats["win_rate"]),
            profit_factor=float(stats["profit_factor"]),
            sharpe=float(stats["sharpe"]),
            expectancy=float(stats["expectancy"]),
            max_drawdown=float(stats["max_drawdown"]),
            total_return=(equity - initial_equity) / max(initial_equity, 1e-9),
            average_win=float(stats["average_win"]),
            average_loss=float(stats["average_loss"]),
            average_hold_bars=float(stats["average_hold_bars"]),
        )

    def get_best_candidates(self, min_score: float = 60.0, limit: int = 4) -> list[StrategyChromosome]:
        """Return top chromosomes that have passed backtesting."""
        candidates = [
            c for c in self.population
            if c.backtest_completed and c.backtest_score >= min_score and c.promotion_stage not in {"rejected", "shadow", "demo", "proposed_live", "live"}
        ]
        candidates.sort(key=lambda c: c.backtest_score, reverse=True)
        return candidates[:limit]

    def promote_to_shadow(self, chromosome: StrategyChromosome) -> None:
        chromosome.promotion_stage = "shadow"
        chromosome.promotion_reason = "backtest_passed_awaiting_shadow"
        self._save_population()

    def promote_to_demo(self, chromosome: StrategyChromosome) -> None:
        chromosome.promotion_stage = "demo"
        chromosome.promotion_reason = "shadow_passed_awaiting_demo"
        self._save_population()

    def promote_to_proposed_live(self, chromosome: StrategyChromosome) -> None:
        chromosome.promotion_stage = "proposed_live"
        chromosome.promotion_reason = "demo_passed_awaiting_operator_approval"
        self._save_population()

    def archive_rejected(self) -> int:
        """Move rejected chromosomes to an archive file to keep population clean."""
        kept: list[StrategyChromosome] = []
        rejected: list[StrategyChromosome] = []
        for c in self.population:
            if c.promotion_stage == "rejected":
                rejected.append(c)
            else:
                kept.append(c)
        if rejected:
            archive_file = Path(self.store.base) / "evolution_archive.jsonl"
            import json
            with archive_file.open("a", encoding="utf-8") as fh:
                for c in rejected:
                    fh.write(json.dumps(c.to_dict()) + "\n")
            self.population = kept
            self._save_population()
        return len(rejected)


def _new_uid() -> str:
    import secrets
    return secrets.token_hex(8)

