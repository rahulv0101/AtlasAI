"""Integration helpers for wiring evolved strategies into the trader.

No trade execution here — only chromosome loading, filtering, and status tracking.
All execution authority remains with ExecutionGate.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_drawdown_guard import EvolutionDrawdownGuard


def load_demo_chromosome_pool(
    storage_dir: str | Path,
    *,
    max_demo: int = 3,
    max_demo_drawdown_pct: float = 0.05,
    max_shadow_drawdown_pct: float = 0.05,
) -> tuple[list[StrategyChromosome], dict[str, Any]]:
    """Load active demo-stage chromosomes from the evolution population file.

    Filters:
        - promotion_stage == "demo"
        - demo_active == True
        - backtest_completed and backtest_score > 0
        - shadow_passed == True
        - drawdown within limits

    Returns:
        (active_chromosomes, status_metadata)
    """
    pop_file = Path(storage_dir) / "evolution_population.json"
    if not pop_file.exists():
        return [], {
            "loaded": False,
            "reason": "population_file_missing",
            "active_count": 0,
            "active_ids": [],
        }

    try:
        payload = json.loads(pop_file.read_text(encoding="utf-8"))
        all_chromosomes = [
            StrategyChromosome.from_dict(c)
            for c in payload.get("chromosomes", [])
        ]
    except Exception as exc:
        return [], {
            "loaded": False,
            "reason": f"parse_error:{exc}",
            "active_count": 0,
            "active_ids": [],
        }

    # Filter: only demo stage, active, passed validation/backtest/shadow
    candidates: list[StrategyChromosome] = []
    for c in all_chromosomes:
        if c.promotion_stage != "demo":
            continue
        if not c.demo_active:
            continue
        if not c.backtest_completed or c.backtest_score <= 0:
            continue
        if not c.shadow_passed:
            continue
        candidates.append(c)

    # Apply drawdown guard
    guard = EvolutionDrawdownGuard(
        max_demo_drawdown_pct=max_demo_drawdown_pct,
        max_shadow_drawdown_pct=max_shadow_drawdown_pct,
    )
    active: list[StrategyChromosome] = []
    paused: list[dict[str, str]] = []
    for c in candidates:
        ok, reason = guard.check_drawdown(c, [], stage="demo")
        if ok:
            active.append(c)
        else:
            guard.pause_strategy(c, reason)
            paused.append({"uid": c.uid[:8], "reason": reason})

    # Respect max demo limit
    active = active[:max(0, max_demo)]

    status: dict[str, Any] = {
        "loaded": True,
        "population_file": str(pop_file),
        "total_in_population": len(all_chromosomes),
        "demo_candidates": len(candidates),
        "active_count": len(active),
        "paused_count": len(paused),
        "paused": paused,
        "active_ids": [c.uid for c in active],
    }
    return active, status
