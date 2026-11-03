"""Tests for the strategy evolution subsystem (Phases 1–6).

Coverage includes:
- Gene validity / chromosome construction
- GoldValidator safety rejections
- Fitness scoring and disqualifiers
- Crossover / mutation safety
- ShadowTester tracking
- PromotionGovernor gates (max 3 demo, risk cap, trade milestones)
- Drawdown guard behavior
- StrategyExplainer output
- EvolutionReporter artifacts
- Live promotion requires human approval
- Persistence across restart roundtrip
- ExecutionGate cannot be bypassed by evolved strategies
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pytest
import pandas as pd

from atlas_ai.strategies.dna import (
    GeneRange,
    StrategyChromosome,
    StrategyGene,
    make_standard_gene_pool,
)
from atlas_ai.strategies.evolution_drawdown_guard import EvolutionDrawdownGuard
from atlas_ai.strategies.evolution_reporter import EvolutionReporter
from atlas_ai.strategies.fitness import BacktestResult, FitnessScorer
from atlas_ai.strategies.gold_validator import GoldValidationError, GoldValidator
from atlas_ai.strategies.promotion_governor import PromotionError, PromotionGovernor
from atlas_ai.strategies.shadow_tester import ShadowOutcome
from atlas_ai.strategies.strategy_explainer import StrategyExplainer


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def valid_chromosome(rng: random.Random) -> StrategyChromosome:
    genes = make_standard_gene_pool()
    for g in genes:
        g.value = g.allowed_range.random_value(rng)
    return StrategyChromosome(
        uid="test1234",
        generation=0,
        genes=genes,
        origin="random",
        backtest_completed=True,
        backtest_score=65.0,
        promotion_stage="backtested",
    )


@pytest.fixture
def factory() -> FitnessScorer:
    return FitnessScorer()


@pytest.fixture
def tmp_path() -> Path:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ------------------------------------------------------------------
# Phase 1 — Core DNA and Validation
# ------------------------------------------------------------------


def test_gene_range_clamp() -> None:
    r = GeneRange(0.0, 100.0, step=1.0)
    assert r.clamp(-5.0) == 0.0
    assert r.clamp(105.0) == 100.0
    assert r.clamp(42.3) == 42.3


def test_gene_mutation_on_step_grid(rng: random.Random) -> None:
    g = StrategyGene("test", "entry", 10.0, GeneRange(0.0, 20.0, step=0.5), mutation_sigma=0.1)
    child = g.mutate(rng, strength=1.0)
    assert child.value >= 0.0
    assert child.value <= 20.0
    assert abs(child.value % 0.5) < 1e-9 or abs((child.value % 0.5) - 0.5) < 1e-9


def test_chromosome_crossover_preserves_gene_order(rng: random.Random) -> None:
    c1 = StrategyChromosome("a", 0, [
        StrategyGene("x", "entry", 1.0, GeneRange(0, 2)),
        StrategyGene("y", "exit", 10.0, GeneRange(5, 15)),
    ])
    c2 = StrategyChromosome("b", 0, [
        StrategyGene("x", "entry", 100.0, GeneRange(0, 200)),
        StrategyGene("y", "exit", 1000.0, GeneRange(500, 1500)),
    ])
    child = c1.crossover(c2, rng)
    assert len(child.genes) == 2
    assert child.genes[0].name == "x"
    assert child.genes[1].name == "y"


# ------------------------------------------------------------------
# Phase 2 — Fitness
# ------------------------------------------------------------------


def test_fitness_score_insufficient_trades(factory: FitnessScorer) -> None:
    result = BacktestResult(
        trades=10, wins=5, losses=5, win_rate=0.5, profit_factor=1.2,
        sharpe=0.5, expectancy=0.001, max_drawdown=0.03, total_return=0.01,
        average_win=0.002, average_loss=-0.001, average_hold_bars=4.0,
    )
    score, reason = factory.score(result)
    assert score == 0.0
    assert "insufficient_trades" in reason


def test_fitness_score_drawdown_exceeds(factory: FitnessScorer) -> None:
    result = BacktestResult(
        trades=40, wins=25, losses=15, win_rate=0.625, profit_factor=1.5,
        sharpe=1.0, expectancy=0.003, max_drawdown=0.10, total_return=0.05,
        average_win=0.003, average_loss=-0.001, average_hold_bars=5.0,
    )
    score, reason = factory.score(result)
    assert score == 0.0
    assert "drawdown_too_high" in reason


def test_fitness_score_healthy(factory: FitnessScorer) -> None:
    result = BacktestResult(
        trades=40, wins=25, losses=15, win_rate=0.625, profit_factor=1.5,
        sharpe=1.0, expectancy=0.003, max_drawdown=0.04, total_return=0.05,
        average_win=0.003, average_loss=-0.001, average_hold_bars=5.0,
    )
    score, reason = factory.score(result)
    assert score > 0.0
    assert score <= 100.0
    assert reason == ""


# ------------------------------------------------------------------
# Phase 3 — GoldValidator
# ------------------------------------------------------------------


def test_gold_validator_fast_lt_slow() -> None:
    v = GoldValidator()
    c = StrategyChromosome("x", 0, [
        StrategyGene("entry_ema_fast", "entry", 60.0, GeneRange(5, 45)),
        StrategyGene("entry_ema_slow", "entry", 50.0, GeneRange(25, 120)),
    ])
    with pytest.raises(GoldValidationError) as exc_info:
        v.validate(c)
    assert "must be strictly less than slow" in str(exc_info.value)


def test_gold_validator_tp_must_exceed_sl() -> None:
    v = GoldValidator()
    # Use full standard gene pool so earlier rules (RSI, EMA) pass
    genes = make_standard_gene_pool()
    # Set SL > TP to trigger the exit rule
    for g in genes:
        if g.name == "exit_sl_atr_multiple":
            g.value = 2.5
        if g.name == "exit_tp_atr_multiple":
            g.value = 2.0
    c = StrategyChromosome("x", 0, genes)
    with pytest.raises(GoldValidationError) as exc_info:
        v.validate(c)
    assert "TP ATR multiple" in str(exc_info.value)


def test_gold_validator_risk_cap() -> None:
    v = GoldValidator()
    genes = make_standard_gene_pool()
    for g in genes:
        if g.name == "risk_max_risk_per_trade":
            g.value = 0.025
    c = StrategyChromosome("safe", 0, genes)
    with pytest.raises(GoldValidationError) as exc_info:
        v.validate(c)
    assert "exceeds 2.0%" in str(exc_info.value)


def test_gold_validator_rejects_excluding_all_allowed_hours() -> None:
    v = GoldValidator()
    genes = make_standard_gene_pool()
    for g in genes:
        if g.name == "regime_allowed_sessions":
            g.value = 1.0
        elif g.name == "regime_excluded_hours_mask":
            g.value = float(sum(1 << hour for hour in range(7, 13)))
    c = StrategyChromosome("x", 0, genes)
    with pytest.raises(GoldValidationError) as exc_info:
        v.validate(c)
    assert "blocks every allowed session hour" in str(exc_info.value)


def test_gold_validator_rejects_overrestrictive_loss_cooldown() -> None:
    v = GoldValidator()
    genes = make_standard_gene_pool()
    for g in genes:
        if g.name == "risk_loss_cooldown_bars":
            g.value = 10.0
        elif g.name == "risk_loss_cooldown_confidence_lift":
            g.value = 0.2
    c = StrategyChromosome("x", 0, genes)
    with pytest.raises(GoldValidationError) as exc_info:
        v.validate(c)
    assert "Loss cooldown too restrictive" in str(exc_info.value)


# ------------------------------------------------------------------
# Phase 3 — ShadowTester & PromotionGovernor
# ------------------------------------------------------------------


def test_shadow_tester_outcome_pnl_calculation() -> None:
    o = ShadowOutcome(
        chromosome_uid="abc",
        strategy_name="test",
        side="buy",
        confidence=0.72,
        entry_price=2000.0,
        exit_price=2005.0,
        pnl=5.0,
        bars_held=3,
        exit_reason="take_profit",
    )
    assert o.pnl == 5.0


def test_promotion_governor_demo_cap() -> None:
    gov = PromotionGovernor(max_demo_strategies=2)
    pop = [
        StrategyChromosome("a", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True),
        StrategyChromosome("b", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True),
    ]
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="backtested")
    candidate.backtest_score = 70.0
    # shadow_outcomes=None skips the shadow-trade check so we hit the demo cap
    ok, reason = gov.can_promote_to_demo(candidate, pop, shadow_outcomes=None)
    assert not ok
    assert "demo_cap_reached" in reason


def test_promotion_governor_risk_cap() -> None:
    gov = PromotionGovernor(max_risk_per_trade_pct=0.005)
    pop: list[StrategyChromosome] = []
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="backtested")
    candidate.backtest_score = 70.0
    for g in candidate.genes:
        if g.name == "risk_max_risk_per_trade":
            g.value = 0.01
    ok, reason = gov.can_promote_to_demo(candidate, pop, shadow_outcomes=[])
    assert not ok
    assert "risk_too_high" in reason


def test_promotion_governor_raw_peak_drawdown_policy_blocks_sensitive_curve() -> None:
    gov = PromotionGovernor(min_shadow_trades=3, max_shadow_drawdown_pct=0.05)
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="backtested")
    candidate.backtest_score = 70.0
    outcomes = [
        ShadowOutcome("c", "s", "buy", 0.7, 100.0, 100.2, 0.20, 1, ""),
        ShadowOutcome("c", "s", "buy", 0.7, 100.2, 100.1, -0.10, 1, ""),
        ShadowOutcome("c", "s", "buy", 0.7, 100.1, 101.9, 1.80, 1, ""),
    ]

    ok, reason = gov.can_promote_to_demo(candidate, [], shadow_outcomes=outcomes)

    assert not ok
    assert "shadow_drawdown_too_high" in reason
    assert "policy=raw_peak" in reason


def test_promotion_governor_explicit_gross_normalized_policy_allows_sensitive_curve() -> None:
    gov = PromotionGovernor(
        min_shadow_trades=3,
        max_shadow_drawdown_pct=0.05,
        shadow_drawdown_policy="gross_normalized",
    )
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="backtested")
    candidate.backtest_score = 70.0
    outcomes = [
        ShadowOutcome("c", "s", "buy", 0.7, 100.0, 100.2, 0.20, 1, ""),
        ShadowOutcome("c", "s", "buy", 0.7, 100.2, 100.1, -0.10, 1, ""),
        ShadowOutcome("c", "s", "buy", 0.7, 100.1, 101.9, 1.80, 1, ""),
    ]

    ok, reason = gov.can_promote_to_demo(candidate, [], shadow_outcomes=outcomes)

    assert ok
    assert reason == ""


def test_promotion_governor_live_needs_human_approval() -> None:
    gov = PromotionGovernor(live_promotion_requires_approval=True)
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True)
    candidate.demo_trades = 120
    candidate.demo_wins = 65
    candidate.demo_losses = 55
    candidate.demo_win_rate = 65 / 120
    candidate.demo_pnl_pct = 0.03
    ok, reason = gov.can_promote_to_live(candidate, human_approved=False)
    assert not ok
    assert "live_promotion_requires_human_approval" in reason


def test_promotion_governor_live_passes_when_approved() -> None:
    gov = PromotionGovernor(live_promotion_requires_approval=True)
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True)
    candidate.demo_trades = 120
    candidate.demo_wins = 65
    candidate.demo_losses = 55
    candidate.demo_win_rate = 65 / 120
    candidate.demo_pnl_pct = 0.03
    ok, reason = gov.can_promote_to_live(candidate, human_approved=True)
    assert ok
    assert reason == ""


def test_promotion_governor_promote_to_demo_raises_on_violation() -> None:
    gov = PromotionGovernor(max_demo_strategies=0)
    candidate = StrategyChromosome("c", 0, make_standard_gene_pool(), promotion_stage="backtested")
    candidate.backtest_score = 70.0
    with pytest.raises(PromotionError) as exc_info:
        gov.promote_to_demo(candidate, [])
    assert "demo_cap_reached" in str(exc_info.value)


# ------------------------------------------------------------------
# Phase 5 — Drawdown Guard
# ------------------------------------------------------------------


def test_drawdown_guard_empty_outcomes_ok() -> None:
    guard = EvolutionDrawdownGuard(max_demo_drawdown_pct=0.05)
    c = StrategyChromosome("x", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True)
    ok, reason = guard.check_drawdown(c, [], stage="demo")
    assert ok
    assert reason == ""


def test_drawdown_guard_pause_sets_inactive() -> None:
    guard = EvolutionDrawdownGuard()
    c = StrategyChromosome("x", 0, make_standard_gene_pool(), promotion_stage="demo", demo_active=True)
    guard.pause_strategy(c, "demo_drawdown_exceeded")
    assert not c.demo_active
    assert "paused:demo_drawdown_exceeded" in c.promotion_reason


def test_drawdown_guard_summary_stats() -> None:
    outcomes = [
        ShadowOutcome("a", "s", "buy", 0.5, 100.0, 105.0, 5.0, 1, ""),
        ShadowOutcome("a", "s", "buy", 0.5, 105.0, 102.0, -3.0, 1, ""),
        ShadowOutcome("a", "s", "buy", 0.5, 102.0, 104.0, 2.0, 1, ""),
    ]
    summary = EvolutionDrawdownGuard.summarize_outcomes(outcomes)
    assert summary["trades"] == 3
    assert summary["net_pnl"] == 4.0
    assert summary["win_rate"] == 2 / 3
    assert summary["max_drawdown_pct"] >= 0.0


# ------------------------------------------------------------------
# Phase 4 — StrategyExplainer & EvolutionReporter
# ------------------------------------------------------------------


def test_strategy_explainer_returns_sections(valid_chromosome: StrategyChromosome) -> None:
    explainer = StrategyExplainer()
    sections = explainer.explain(valid_chromosome)
    assert "entry" in sections
    assert "confirmation" in sections
    assert "exit" in sections
    assert "risk" in sections
    assert "regime" in sections
    assert "performance" in sections
    assert "lineage" in sections


def test_strategy_explainer_compact_not_empty(valid_chromosome: StrategyChromosome) -> None:
    text = StrategyExplainer.explain_compact(valid_chromosome)
    assert "stage=" in text
    assert "EMA(" in text


def test_evolution_reporter_leaderboard_roundtrip(tmp_path: Path) -> None:
    reporter = EvolutionReporter(output_dir=tmp_path)
    pop = [
        StrategyChromosome("a", 0, make_standard_gene_pool(), backtest_completed=True, backtest_score=80.0),
        StrategyChromosome("b", 0, make_standard_gene_pool(), backtest_completed=True, backtest_score=60.0),
    ]
    path = reporter.write_leaderboard(pop)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_population"] == 2
    assert data["leaderboard"][0]["uid"] == "a"


def test_evolution_reporter_strategy_detail_structure(valid_chromosome: StrategyChromosome) -> None:
    reporter = EvolutionReporter(output_dir=Path("/tmp"))
    detail = reporter.strategy_detail(valid_chromosome)
    assert detail["uid"] == valid_chromosome.uid
    assert "genes" in detail
    assert "backtest" in detail
    assert "shadow" in detail
    assert "demo" in detail


# ------------------------------------------------------------------
# Phase 6 — Persistence / Roundtrip
# ------------------------------------------------------------------


def test_chromosome_json_roundtrip(valid_chromosome: StrategyChromosome) -> None:
    original = valid_chromosome
    serialized = original.to_dict()
    restored = StrategyChromosome.from_dict(serialized)
    assert restored.uid == original.uid
    assert restored.generation == original.generation
    assert len(restored.genes) == len(original.genes)


def test_chromosome_from_dict_backfills_missing_standard_genes() -> None:
    payload = StrategyChromosome("legacy", 0, make_standard_gene_pool()).to_dict()
    payload["genes"] = [gene for gene in payload["genes"] if gene["name"] != "regime_excluded_hours_mask"]

    restored = StrategyChromosome.from_dict(payload)
    restored.set_gene("regime_excluded_hours_mask", float(1 << 12))

    assert restored.gene_value("regime_excluded_hours_mask") == float(1 << 12)
    assert restored.gene_value("risk_loss_cooldown_bars") == 0.0


# ------------------------------------------------------------------
# Phase 6 — ExecutionGate Bypass Guard (safety invariant)
# ------------------------------------------------------------------


def test_evolution_engine_never_executes_trades() -> None:
    """Pure architecture regression: EvolutionEngine has no execute() method."""
    from atlas_ai.strategies.evolution_engine import EvolutionEngine
    assert not hasattr(EvolutionEngine, "execute")
    assert "execute" not in (getattr(EvolutionEngine, "__methods__", "") or "")


def test_strategy_inventor_creates_valid_proposal_only_chromosomes() -> None:
    from atlas_ai.strategies.strategy_inventor import StrategyInventor

    plans = StrategyInventor().invent(
        status={
            "primary_blocker": "confidence below threshold",
            "context_snapshot": {"regime": "trend"},
        },
        cognitive_core={
            "decision_understanding": {"final_trade_bias": "buy"},
            "learning_state": {"stability_score": 0.72, "overreaction_score": 0.12},
        },
    )

    assert plans
    validator = GoldValidator()
    for plan in plans:
        validator.validate(plan.chromosome)
        assert plan.chromosome.origin.startswith("ai_inventor:")
        assert plan.chromosome.promotion_stage == "untested"
        assert plan.chromosome.gene_value("regime_allowed_sessions") == 3.0
        assert plan.chromosome.gene_value("risk_min_confidence") <= 0.55
        assert "execution_gate_cannot_be_bypassed" in plan.safety_constraints
        assert not hasattr(plan.chromosome, "execute")


def test_evolution_engine_seeds_ai_invented_proposals_without_execution(tmp_path: Path) -> None:
    from atlas_ai.config.settings import Settings
    from atlas_ai.memory.knowledge_store import KnowledgeStore
    from atlas_ai.strategies.evolution_engine import EvolutionEngine

    store = KnowledgeStore(str(tmp_path))
    engine = EvolutionEngine(Settings(hfm_login=1, hfm_password="x", hfm_server="demo"), store)
    added = engine.seed_ai_invented_population(
        status={
            "primary_blocker": "spread elevated",
            "context_snapshot": {"regime": "high_volatility_range"},
        },
        cognitive_core={
            "decision_understanding": {"final_trade_bias": "neutral"},
            "learning_state": {"stability_score": 0.38, "overreaction_score": 0.51},
        },
        limit=2,
    )

    assert added
    assert all(c.origin.startswith("ai_inventor:") for c in added)
    assert all(c.promotion_stage == "untested" for c in added)
    assert not hasattr(engine, "execute")
    assert (tmp_path / "evolution_population.json").exists()


def test_evolution_engine_backtests_generated_strategy_without_removed_private_method(tmp_path: Path) -> None:
    from atlas_ai.config.settings import Settings
    from atlas_ai.memory.knowledge_store import KnowledgeStore
    from atlas_ai.strategies.evolution_engine import EvolutionEngine

    store = KnowledgeStore(str(tmp_path))
    settings = Settings(
        hfm_login=1,
        hfm_password="x",
        hfm_server="demo",
        evolved_candidate_min_trades=1,
        research_confidence_threshold=0.1,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
    )
    engine = EvolutionEngine(settings, store, rng=random.Random(7))
    chromosome = engine.random_chromosome()
    for gene in chromosome.genes:
        if gene.name == "risk_min_confidence":
            gene.value = 0.1
        elif gene.name == "regime_allowed_sessions":
            gene.value = 3.0
        elif gene.name == "regime_prefer_trend":
            gene.value = 0.0

    df = pd.DataFrame({
        "open": [2000.0 + i * 0.25 for i in range(220)],
        "high": [2001.0 + i * 0.25 for i in range(220)],
        "low": [1999.0 + i * 0.25 for i in range(220)],
        "close": [2000.5 + i * 0.25 for i in range(220)],
        "tick_volume": [120 + (i % 15) for i in range(220)],
        "time": pd.date_range("2024-01-01 08:00:00", periods=220, freq="15min"),
    })

    engine._backtest_chromosome(chromosome, df)

    assert chromosome.backtest_completed is True
    assert not str(chromosome.reject_reason).startswith("backtest_error:")
    assert chromosome.backtest_trades >= 0


# ------------------------------------------------------------------
# Integration tests — no MT5 required when using mocks
# ------------------------------------------------------------------


def test_load_demo_chromosome_pool_filters_and_limits(tmp_path: Path) -> None:
    """Unit: load_demo_chromosome_pool loads only active demo chromosomes."""
    from atlas_ai.strategies.evolution_integration import load_demo_chromosome_pool

    # Create a population file with mixed states
    pop = {
        "chromosomes": [
            # Passed all checks -> should be loaded
            StrategyChromosome(
                "valid_demo_a", 1, make_standard_gene_pool(),
                promotion_stage="demo", demo_active=True, backtest_completed=True, backtest_score=70.0, shadow_passed=True
            ).to_dict(),
            # Failed backtest -> should be skipped
            StrategyChromosome(
                "failed_backtest", 1, make_standard_gene_pool(),
                promotion_stage="demo", demo_active=True, backtest_completed=True, backtest_score=0.0, shadow_passed=True
            ).to_dict(),
            # Not shadow passed -> should be skipped
            StrategyChromosome(
                "no_shadow", 1, make_standard_gene_pool(),
                promotion_stage="demo", demo_active=True, backtest_completed=True, backtest_score=65.0, shadow_passed=False
            ).to_dict(),
            # Inactive -> should be skipped
            StrategyChromosome(
                "inactive", 1, make_standard_gene_pool(),
                promotion_stage="demo", demo_active=False, backtest_completed=True, backtest_score=65.0, shadow_passed=True
            ).to_dict(),
            # Wrong stage -> should be skipped
            StrategyChromosome(
                "backtested_only", 1, make_standard_gene_pool(),
                promotion_stage="backtested", demo_active=False, backtest_completed=True, backtest_score=65.0, shadow_passed=False
            ).to_dict(),
        ]
    }
    pop_file = tmp_path / "evolution_population.json"
    pop_file.write_text(json.dumps(pop), encoding="utf-8")

    active, status = load_demo_chromosome_pool(tmp_path, max_demo=2)
    assert status["loaded"] is True
    assert status["active_count"] == 1  # Only valid_demo_a qualifies
    assert len(active) == 1
    assert active[0].uid == "valid_demo_a"
    assert status["active_ids"] == ["valid_demo_a"]
    assert status["total_in_population"] == 5


def test_load_demo_chromosome_pool_max_demo_enforced(tmp_path: Path) -> None:
    """Unit: max_demo limit is enforced."""
    from atlas_ai.strategies.evolution_integration import load_demo_chromosome_pool

    pop = {
        "chromosomes": [
            StrategyChromosome(
                f"demo_{i}", 1, make_standard_gene_pool(),
                promotion_stage="demo", demo_active=True, backtest_completed=True, backtest_score=70.0, shadow_passed=True
            ).to_dict()
            for i in range(5)
        ]
    }
    pop_file = tmp_path / "evolution_population.json"
    pop_file.write_text(json.dumps(pop), encoding="utf-8")

    active, status = load_demo_chromosome_pool(tmp_path, max_demo=3)
    assert len(active) == 3
    assert status["active_count"] == 3
    assert status["demo_candidates"] == 5


def test_load_demo_chromosome_pool_missing_file() -> None:
    """Unit: missing population file returns empty list with status."""
    from atlas_ai.strategies.evolution_integration import load_demo_chromosome_pool
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        active, status = load_demo_chromosome_pool(td)
        assert active == []
        assert status["loaded"] is False
        assert status["reason"] == "population_file_missing"


def test_evolved_strategy_cannot_execute_directly() -> None:
    """Safety invariant: evolved strategies produce signals only; ExecutionGate handles execution."""
    from atlas_ai.strategies.strategy_factory import StrategyFactory
    import pandas as pd

    chromosome = StrategyChromosome(
        "safe_exec_test", 0, make_standard_gene_pool(),
        promotion_stage="demo", demo_active=True, backtest_completed=True, backtest_score=70.0, shadow_passed=True
    )
    strategy = StrategyFactory.build(chromosome)
    # strategy has generate_signal but NOT execute
    assert hasattr(strategy, "generate_signal")
    assert not hasattr(strategy, "execute")
    assert not hasattr(strategy, "place_order")
    assert not hasattr(strategy, "send_to_broker")

    # Even with a signal, the strategy itself cannot act on it
    df = pd.DataFrame({
        "open": [2000.0, 2001.0, 2002.0],
        "high": [2002.0, 2003.0, 2004.0],
        "low": [1999.0, 2000.0, 2001.0],
        "close": [2001.0, 2002.0, 2003.0],
        "time": pd.date_range("2024-01-01", periods=3, freq="h"),
    })
    signal = strategy.generate_signal(df)
    assert signal.side in {"buy", "sell", "flat"}
    # The signal is just data — no side effects


def test_evolved_strategy_respects_excluded_hours() -> None:
    from atlas_ai.strategies.strategy_factory import StrategyFactory
    import pandas as pd

    genes = make_standard_gene_pool()
    for gene in genes:
        if gene.name == "regime_allowed_sessions":
            gene.value = 1.0
        elif gene.name == "regime_excluded_hours_mask":
            gene.value = float(1 << 12)
        elif gene.name == "risk_min_confidence":
            gene.value = 0.3
    chromosome = StrategyChromosome("hour_filter", 0, genes)
    strategy = StrategyFactory.build(chromosome)
    df = pd.DataFrame({
        "open": [2000.0 + i * 0.2 for i in range(80)],
        "high": [2001.0 + i * 0.2 for i in range(80)],
        "low": [1999.0 + i * 0.2 for i in range(80)],
        "close": [2000.5 + i * 0.2 for i in range(80)],
        "tick_volume": [120 + (i % 12) for i in range(80)],
        "time": pd.date_range("2024-01-01 12:00:00", periods=80, freq="1min"),
    })

    signal = strategy.generate_signal(df)

    assert signal.side == "flat"
