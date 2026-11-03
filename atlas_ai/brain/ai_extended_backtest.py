from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.memory.jsonl_tail import read_jsonl_tail
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_engine import EvolutionEngine
from atlas_ai.strategies.fitness import FitnessScorer
from atlas_ai.strategies.strategy_factory import StrategyFactory


class AIExtendedBacktestRunner:
    """Research-only extended backtest runner for archived AI candidates."""

    def __init__(self, settings: Settings, store: KnowledgeStore) -> None:
        self.settings = settings
        self.store = store
        self.base = Path(store.base)
        self.results_path = self.base / "ai_extended_backtests.jsonl"

    def run_for_uid(self, uid: str, df: pd.DataFrame) -> dict[str, Any]:
        chromosome = self._load_archived(uid)
        if chromosome is None:
            return {"ok": False, "reason": "candidate_not_found", "uid": uid, "executed": False}
        if df.empty:
            return {"ok": False, "reason": "empty_market_data", "uid": uid, "executed": False}

        engine = EvolutionEngine(self.settings, self.store)
        strategy = StrategyFactory.build(chromosome)
        result = engine._run_single_strategy_backtest(df.reset_index(drop=True), strategy)
        scorer = FitnessScorer(
            min_trades=max(20, int(getattr(self.settings, "evolved_candidate_min_trades", 200) or 200)),
            min_win_rate=float(getattr(self.settings, "evolved_candidate_min_win_rate", 0.52) or 0.52),
            min_profit_factor=1.05,
            max_drawdown_pct=0.10,
        )
        score, reject_reason = scorer.score(result)
        passed = score > 0.0
        payload = {
            "schema": "atlas_ai_extended_backtest_v1",
            "uid": chromosome.uid,
            "origin": chromosome.origin,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "bars": int(len(df)),
            "first_bar": _bar_time(df, 0),
            "last_bar": _bar_time(df, len(df) - 1),
            "result": {
                "trades": int(result.trades),
                "wins": int(result.wins),
                "losses": int(result.losses),
                "win_rate": float(result.win_rate),
                "profit_factor": float(result.profit_factor),
                "sharpe": float(result.sharpe),
                "expectancy": float(result.expectancy),
                "max_drawdown": float(result.max_drawdown),
                "total_return": float(result.total_return),
                "average_win": float(result.average_win),
                "average_loss": float(result.average_loss),
                "average_hold_bars": float(result.average_hold_bars),
            },
            "score": float(score),
            "passed_extended_sample": bool(passed),
            "recommendation": "eligible_for_shadow_review" if passed else "continue_research",
            "reject_reason": reject_reason,
            "safety": {
                "research_only": True,
                "does_not_execute": True,
                "does_not_promote": True,
                "requires_promotion_governor": True,
            },
            "executed": False,
        }
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self.store.log_event("evolution", "AI extended backtest completed.", payload)
        return {"ok": True, **payload}

    def _load_archived(self, uid: str) -> StrategyChromosome | None:
        archive_path = self.base / "evolution_archive.jsonl"
        rows = read_jsonl_tail(archive_path, 1, matcher=lambda row: str(row.get("uid", "")) == str(uid))
        if not rows:
            return None
        return StrategyChromosome.from_dict(rows[-1])


def _bar_time(df: pd.DataFrame, idx: int) -> str:
    if df.empty or "time" not in df.columns:
        return ""
    try:
        return str(pd.Timestamp(df.iloc[idx]["time"]).isoformat())
    except Exception:
        return ""
