"""Strategy DNA encoding and manipulation for evolution engine.

Each strategy is represented as a chromosome composed of genes.
Every gene carries its type, value, allowed range, and mutation rules.
The strategy engine produces proposals only — ExecutionGate remains
the sole authority for trade execution.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass(frozen=True)
class GeneRange:
    """Allowed numeric range for a gene and its mutation step size."""
    min_val: float
    max_val: float
    step: float = 1.0

    def clamp(self, value: float) -> float:
        return max(self.min_val, min(self.max_val, value))

    def random_value(self, rng: random.Random) -> float:
        steps = int((self.max_val - self.min_val) / max(self.step, 1e-9))
        if steps <= 0:
            return self.min_val
        return self.min_val + rng.randint(0, steps) * self.step


@dataclass
class StrategyGene:
    """Individual gene of a strategy chromosome.

    Attributes:
        name: Human-readable identifier (e.g. "ema_fast_period").
        gene_type: Category: "entry", "confirmation", "exit", "risk", "regime".
        value: Current gene value.
        allowed_range: Numeric bounds and step size.
        mutation_sigma: Standard deviation for mutation relative to range span.
    """
    name: str
    gene_type: str
    value: float
    allowed_range: GeneRange
    mutation_sigma: float = 0.1

    def clone(self) -> "StrategyGene":
        return StrategyGene(
            name=self.name,
            gene_type=self.gene_type,
            value=self.value,
            allowed_range=GeneRange(
                self.allowed_range.min_val,
                self.allowed_range.max_val,
                self.allowed_range.step,
            ),
            mutation_sigma=self.mutation_sigma,
        )

    def mutate(self, rng: random.Random, strength: float = 1.0) -> "StrategyGene":
        span = self.allowed_range.max_val - self.allowed_range.min_val
        sigma = span * self.mutation_sigma * strength
        new_value = self.value + rng.gauss(0.0, sigma)
        new_value = self.allowed_range.clamp(new_value)
        # Snap to step grid
        if self.allowed_range.step > 0:
            new_value = round(new_value / self.allowed_range.step) * self.allowed_range.step
        return StrategyGene(
            name=self.name,
            gene_type=self.gene_type,
            value=new_value,
            allowed_range=GeneRange(
                self.allowed_range.min_val,
                self.allowed_range.max_val,
                self.allowed_range.step,
            ),
            mutation_sigma=self.mutation_sigma,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "gene_type": self.gene_type,
            "value": self.value,
            "allowed_range": {
                "min": self.allowed_range.min_val,
                "max": self.allowed_range.max_val,
                "step": self.allowed_range.step,
            },
            "mutation_sigma": self.mutation_sigma,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyGene":
        r = payload.get("allowed_range", {})
        return cls(
            name=str(payload["name"]),
            gene_type=str(payload["gene_type"]),
            value=float(payload["value"]),
            allowed_range=GeneRange(
                min_val=float(r.get("min", 0.0)),
                max_val=float(r.get("max", 0.0)),
                step=float(r.get("step", 1.0)),
            ),
            mutation_sigma=float(payload.get("mutation_sigma", 0.1)),
        )


@dataclass
class StrategyChromosome:
    """Complete strategy encoded as a collection of genes plus metadata.

    The chromosome can be materialized into an executable strategy object.
    Never executes trades itself — proposals go through EvolutionEngine
    → ShadowTester → Promotion → AtlasTrader → ExecutionGate.
    """
    uid: str
    generation: int
    genes: list[StrategyGene]
    origin: str = "random"
    parent_uids: list[str] = field(default_factory=list)
    created_at: str = ""

    # Performance record (updated live)
    backtest_trades: int = 0
    backtest_wins: int = 0
    backtest_losses: int = 0
    backtest_win_rate: float = 0.0
    backtest_profit_factor: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_expectancy: float = 0.0
    backtest_max_drawdown: float = 0.0
    backtest_score: float = 0.0
    backtest_completed: bool = False

    shadow_signals: int = 0
    shadow_agreement_rate: float = 0.0
    shadow_degradation_pct: float = 0.0
    shadow_passed: bool = False

    demo_trades: int = 0
    demo_wins: int = 0
    demo_losses: int = 0
    demo_win_rate: float = 0.0
    demo_pnl_pct: float = 0.0
    demo_active: bool = False

    # Promotion state
    promotion_stage: str = "untested"  # untested → backtested → shadow → demo → proposed_live → live
    promotion_reason: str = ""
    reject_reason: str = ""

    def gene_value(self, name: str, default: float = 0.0) -> float:
        for g in self.genes:
            if g.name == name:
                return g.value
        return default

    def set_gene(self, name: str, value: float) -> None:
        for g in self.genes:
            if g.name == name:
                g.value = value
                break
        else:
            for standard in make_standard_gene_pool():
                if standard.name == name:
                    standard.value = standard.allowed_range.clamp(float(value))
                    self.genes.append(standard)
                    break

    def signature(self) -> str:
        parts = [f"{g.name}={g.value:.3f}" for g in sorted(self.genes, key=lambda x: x.name)]
        return f"gen{self.generation}|" + "|".join(parts)

    def clone(self) -> "StrategyChromosome":
        return StrategyChromosome(
            uid=_new_uid(),
            generation=self.generation,
            genes=[g.clone() for g in self.genes],
            origin=self.origin,
            parent_uids=[self.uid],
            created_at=self.created_at,
        )

    def mutate(self, rng: random.Random, strength: float = 1.0) -> "StrategyChromosome":
        child = StrategyChromosome(
            uid=_new_uid(),
            generation=self.generation + 1,
            genes=[g.mutate(rng, strength) for g in self.genes],
            origin="mutation",
            parent_uids=[self.uid],
            created_at=self.created_at,
        )
        return child

    def crossover(self, other: "StrategyChromosome", rng: random.Random) -> "StrategyChromosome":
        child_genes: list[StrategyGene] = []
        for g1, g2 in zip(self.genes, other.genes):
            assert g1.name == g2.name, "Chromosome gene mismatch"
            if rng.random() < 0.5:
                child_genes.append(g1.clone())
            else:
                child_genes.append(g2.clone())
        return StrategyChromosome(
            uid=_new_uid(),
            generation=max(self.generation, other.generation) + 1,
            genes=child_genes,
            origin="crossover",
            parent_uids=[self.uid, other.uid],
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["genes"] = [g.to_dict() for g in self.genes]
        payload["signature"] = self.signature()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyChromosome":
        raw_genes = payload.get("genes", [])
        genes = [StrategyGene.from_dict(g) for g in raw_genes]
        existing = {gene.name for gene in genes}
        for standard in make_standard_gene_pool():
            if standard.name not in existing:
                genes.append(standard)
                existing.add(standard.name)
        chromosome = cls(
            uid=str(payload.get("uid", _new_uid())),
            generation=int(payload.get("generation", 0)),
            genes=genes,
            origin=str(payload.get("origin", "unknown")),
            parent_uids=list(payload.get("parent_uids", [])),
            created_at=str(payload.get("created_at", "")),
            backtest_trades=int(payload.get("backtest_trades", 0)),
            backtest_wins=int(payload.get("backtest_wins", 0)),
            backtest_losses=int(payload.get("backtest_losses", 0)),
            backtest_win_rate=float(payload.get("backtest_win_rate", 0.0)),
            backtest_profit_factor=float(payload.get("backtest_profit_factor", 0.0)),
            backtest_sharpe=float(payload.get("backtest_sharpe", 0.0)),
            backtest_expectancy=float(payload.get("backtest_expectancy", 0.0)),
            backtest_max_drawdown=float(payload.get("backtest_max_drawdown", 0.0)),
            backtest_score=float(payload.get("backtest_score", 0.0)),
            backtest_completed=bool(payload.get("backtest_completed", False)),
            shadow_signals=int(payload.get("shadow_signals", 0)),
            shadow_agreement_rate=float(payload.get("shadow_agreement_rate", 0.0)),
            shadow_degradation_pct=float(payload.get("shadow_degradation_pct", 0.0)),
            shadow_passed=bool(payload.get("shadow_passed", False)),
            demo_trades=int(payload.get("demo_trades", 0)),
            demo_wins=int(payload.get("demo_wins", 0)),
            demo_losses=int(payload.get("demo_losses", 0)),
            demo_win_rate=float(payload.get("demo_win_rate", 0.0)),
            demo_pnl_pct=float(payload.get("demo_pnl_pct", 0.0)),
            demo_active=bool(payload.get("demo_active", False)),
            promotion_stage=str(payload.get("promotion_stage", "untested")),
            promotion_reason=str(payload.get("promotion_reason", "")),
            reject_reason=str(payload.get("reject_reason", "")),
        )
        return chromosome


# ---------------------------------------------------------------------------
# Standard gene library for XAUUSD mean-reversion and trend strategies
# ---------------------------------------------------------------------------


def make_standard_gene_pool() -> list[StrategyGene]:
    """Returns the canonical gene pool for XAUUSD strategy evolution."""
    genes: list[StrategyGene] = [
        # --- ENTRY GENES ---
        StrategyGene(
            name="entry_ema_fast",
            gene_type="entry",
            value=20.0,
            allowed_range=GeneRange(5.0, 45.0, step=1.0),
            mutation_sigma=0.12,
        ),
        StrategyGene(
            name="entry_ema_slow",
            gene_type="entry",
            value=50.0,
            allowed_range=GeneRange(25.0, 120.0, step=1.0),
            mutation_sigma=0.10,
        ),
        StrategyGene(
            name="rsi_buy_threshold",
            gene_type="entry",
            value=42.0,
            allowed_range=GeneRange(20.0, 55.0, step=1.0),
            mutation_sigma=0.08,
        ),
        StrategyGene(
            name="rsi_sell_threshold",
            gene_type="entry",
            value=58.0,
            allowed_range=GeneRange(45.0, 80.0, step=1.0),
            mutation_sigma=0.08,
        ),

        # --- CONFIRMATION GENES ---
        StrategyGene(
            name="confirm_min_trend_strength",
            gene_type="confirmation",
            value=0.22,
            allowed_range=GeneRange(0.08, 0.45, step=0.01),
            mutation_sigma=0.10,
        ),
        StrategyGene(
            name="confirm_min_body_ratio",
            gene_type="confirmation",
            value=0.35,
            allowed_range=GeneRange(0.20, 0.60, step=0.01),
            mutation_sigma=0.08,
        ),
        StrategyGene(
            name="confirm_volume_min_ratio",
            gene_type="confirmation",
            value=0.85,
            allowed_range=GeneRange(0.50, 1.30, step=0.01),
            mutation_sigma=0.06,
        ),
        StrategyGene(
            name="confirm_htf_alignment",
            gene_type="confirmation",
            value=1.0,
            allowed_range=GeneRange(0.0, 1.0, step=1.0),
            mutation_sigma=0.0,
        ),

        # --- EXIT GENES ---
        StrategyGene(
            name="exit_sl_atr_multiple",
            gene_type="exit",
            value=1.6,
            allowed_range=GeneRange(0.8, 3.5, step=0.1),
            mutation_sigma=0.12,
        ),
        StrategyGene(
            name="exit_tp_atr_multiple",
            gene_type="exit",
            value=2.8,
            allowed_range=GeneRange(1.0, 5.0, step=0.1),
            mutation_sigma=0.10,
        ),
        StrategyGene(
            name="exit_trail_enable",
            gene_type="exit",
            value=1.0,
            allowed_range=GeneRange(0.0, 1.0, step=1.0),
            mutation_sigma=0.0,
        ),
        StrategyGene(
            name="exit_max_hold_bars",
            gene_type="exit",
            value=8.0,
            allowed_range=GeneRange(3.0, 20.0, step=1.0),
            mutation_sigma=0.10,
        ),

        # --- RISK / SIZING GENES ---
        StrategyGene(
            name="risk_max_risk_per_trade",
            gene_type="risk",
            value=0.005,
            allowed_range=GeneRange(0.001, 0.020, step=0.001),
            mutation_sigma=0.05,
        ),
        StrategyGene(
            name="risk_confidence_multiplier",
            gene_type="risk",
            value=1.0,
            allowed_range=GeneRange(0.50, 1.50, step=0.01),
            mutation_sigma=0.06,
        ),
        StrategyGene(
            name="risk_min_confidence",
            gene_type="risk",
            value=0.55,
            allowed_range=GeneRange(0.30, 0.85, step=0.01),
            mutation_sigma=0.06,
        ),
        StrategyGene(
            name="risk_max_trades_per_day",
            gene_type="risk",
            value=8.0,
            allowed_range=GeneRange(2.0, 20.0, step=1.0),
            mutation_sigma=0.08,
        ),

        # --- REGIME FILTER GENES ---
        StrategyGene(
            name="regime_prefer_trend",
            gene_type="regime",
            value=0.0,
            allowed_range=GeneRange(0.0, 1.0, step=1.0),
            mutation_sigma=0.0,
        ),
        StrategyGene(
            name="regime_min_trend_strength",
            gene_type="regime",
            value=0.18,
            allowed_range=GeneRange(0.10, 0.40, step=0.01),
            mutation_sigma=0.08,
        ),
        StrategyGene(
            name="regime_max_volatility_pct",
            gene_type="regime",
            value=0.07,
            allowed_range=GeneRange(0.03, 0.14, step=0.01),
            mutation_sigma=0.08,
        ),
        StrategyGene(
            name="regime_allowed_sessions",
            gene_type="regime",
            value=2.0,  # 0=asia, 1=london, 2=new_york, 3=both NY+London
            allowed_range=GeneRange(1.0, 3.0, step=1.0),
            mutation_sigma=0.0,
        ),
        StrategyGene(
            name="regime_excluded_hours_mask",
            gene_type="regime",
            value=0.0,
            allowed_range=GeneRange(0.0, float((1 << 24) - 1), step=1.0),
            mutation_sigma=0.0,
        ),
        StrategyGene(
            name="risk_loss_cooldown_bars",
            gene_type="risk",
            value=0.0,
            allowed_range=GeneRange(0.0, 12.0, step=1.0),
            mutation_sigma=0.0,
        ),
        StrategyGene(
            name="risk_loss_cooldown_confidence_lift",
            gene_type="risk",
            value=0.0,
            allowed_range=GeneRange(0.0, 0.25, step=0.01),
            mutation_sigma=0.0,
        ),
    ]
    return genes


def _new_uid() -> str:
    import secrets
    return secrets.token_hex(8)

