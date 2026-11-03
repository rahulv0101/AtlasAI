from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_reporter import EvolutionReporter
from atlas_ai.strategies.promotion_governor import PromotionGovernor
from atlas_ai.strategies.shadow_tester import ShadowOutcome, ShadowTester


class AIShadowReviewRunner:
    """Run focused paper-only shadow review for a prepared candidate."""

    def __init__(self, settings: Settings, store: KnowledgeStore) -> None:
        self.settings = settings
        self.store = store
        self.base = Path(store.base)
        self.population_path = self.base / "evolution_population.json"
        self.results_path = self.base / "ai_shadow_reviews.jsonl"
        self.governor = PromotionGovernor(
            max_demo_strategies=int(getattr(settings, "max_demo_strategies", 3)),
            max_risk_per_trade_pct=float(getattr(settings, "evolved_max_risk_pct", 0.005)),
            min_shadow_trades=int(getattr(settings, "evolution_min_shadow_trades", 100)),
            max_shadow_drawdown_pct=float(getattr(settings, "evolution_max_shadow_drawdown_pct", 0.05)),
            shadow_drawdown_policy=str(getattr(settings, "evolution_shadow_drawdown_policy", "raw_peak")),
            min_backtest_score=60.0,
            min_demo_trades=int(getattr(settings, "evolution_min_demo_trades", 100)),
            min_demo_win_rate=float(getattr(settings, "evolution_min_demo_win_rate", 0.50)),
            min_demo_profit_factor=float(getattr(settings, "evolution_min_demo_profit_factor", 1.10)),
            live_promotion_requires_approval=bool(getattr(settings, "live_promotion_requires_approval", True)),
        )

    def run(self, uid: str, df: pd.DataFrame, *, bars: int | None = None) -> dict[str, Any]:
        population, generation = self._load_population()
        chromosome = next((c for c in population if c.uid == uid), None)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}
        if chromosome.promotion_stage not in {"backtested", "shadow"}:
            return {"ok": False, "uid": uid, "reason": f"invalid_stage:{chromosome.promotion_stage}", "executed": False}
        if df.empty:
            return {"ok": False, "uid": uid, "reason": "empty_market_data", "executed": False}

        review_bars = int(bars or getattr(self.settings, "ai_shadow_review_bars", 1200) or 1200)
        df_slice = df.tail(max(60, review_bars)).reset_index(drop=True)
        outcomes = self._run_shadow(chromosome, df_slice)
        summary = self._summarize(outcomes)

        ok_demo, demo_reason = self.governor.can_promote_to_demo(chromosome, population, shadow_outcomes=outcomes)
        recommendation = "eligible_for_demo_review" if ok_demo else "continue_shadow_review"
        chromosome.promotion_stage = "shadow"
        chromosome.promotion_reason = (
            "shadow_review_passed_awaiting_demo_review"
            if ok_demo
            else f"shadow_review_blocked:{demo_reason or 'needs_more_evidence'}"
        )
        chromosome.shadow_signals = int(summary["trades"])
        chromosome.shadow_agreement_rate = float(summary["win_rate"])
        chromosome.shadow_degradation_pct = float(summary["degradation_pct"])
        chromosome.shadow_passed = bool(ok_demo)
        self._save_population(population, generation)
        EvolutionReporter(self.base).write_leaderboard(population)

        payload = {
            "schema": "atlas_ai_shadow_review_v1",
            "uid": uid,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "bars": int(len(df_slice)),
            "stage": chromosome.promotion_stage,
            "summary": summary,
            "promotion_governor": {
                "can_promote_to_demo": bool(ok_demo),
                "reason": demo_reason,
                "shadow_drawdown_policy": self.governor.shadow_drawdown_policy,
                "max_shadow_drawdown_pct": self.governor.max_shadow_drawdown_pct,
            },
            "recommendation": recommendation,
            "safety": {
                "paper_only": True,
                "does_not_execute": True,
                "does_not_promote_live": True,
                "requires_promotion_governor": True,
            },
            "executed": False,
        }
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self.store.log_event("evolution", "AI shadow review completed.", payload)
        return {"ok": True, **payload}

    @staticmethod
    def _run_shadow(chromosome: StrategyChromosome, df: pd.DataFrame) -> list[ShadowOutcome]:
        tester = ShadowTester()
        outcomes: list[ShadowOutcome] = []
        cooldown_bars = int(chromosome.gene_value("risk_loss_cooldown_bars", 0.0))
        cooldown_lift = float(chromosome.gene_value("risk_loss_cooldown_confidence_lift", 0.0))
        base_min_conf = float(chromosome.gene_value("risk_min_confidence", 0.55))
        cooldown_until = -1
        for i in range(1, len(df)):
            bar_df = df.iloc[: i + 1]
            current_price = float(bar_df["close"].iloc[-1])
            if i < cooldown_until:
                from atlas_ai.strategies.strategy_factory import StrategyFactory

                signal = StrategyFactory.build(chromosome).generate_signal(bar_df)
                if signal.side in {"buy", "sell"} and float(signal.confidence) < base_min_conf + cooldown_lift:
                    continue
            outcome = tester.step(chromosome, bar_df, current_price)
            if outcome is not None:
                if "time" in bar_df.columns:
                    try:
                        outcome.timestamp = str(pd.Timestamp(bar_df["time"].iloc[-1]).isoformat())
                    except Exception:
                        pass
                outcomes.append(outcome)
                if outcome.pnl <= 0 and cooldown_bars > 0:
                    cooldown_until = max(cooldown_until, i + cooldown_bars)
        return outcomes

    @staticmethod
    def _summarize(outcomes: list[ShadowOutcome]) -> dict[str, Any]:
        trades = len(outcomes)
        wins = sum(1 for row in outcomes if row.pnl > 0)
        losses = trades - wins
        net_pnl = sum(float(row.pnl) for row in outcomes)
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        for row in outcomes:
            running += float(row.pnl)
            peak = max(peak, running)
            max_dd = max(max_dd, peak - running)
        mid = trades // 2
        degradation = 0.0
        if mid > 0:
            first = sum(1 for row in outcomes[:mid] if row.pnl > 0) / mid
            second = sum(1 for row in outcomes[mid:] if row.pnl > 0) / max(1, trades - mid)
            degradation = max(0.0, first - second)
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades if trades else 0.0,
            "net_pnl": net_pnl,
            "max_drawdown_pct": max_dd / max(1e-9, abs(peak)) if peak else 0.0,
            "degradation_pct": degradation,
        }

    def _load_population(self) -> tuple[list[StrategyChromosome], int]:
        if not self.population_path.exists():
            return [], 0
        try:
            payload = json.loads(self.population_path.read_text(encoding="utf-8"))
        except Exception:
            return [], 0
        return [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])], int(payload.get("generation", 0) or 0)

    def _save_population(self, population: list[StrategyChromosome], generation: int) -> None:
        payload = {
            "generation": int(generation),
            "chromosomes": [c.to_dict() for c in population],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.population_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
