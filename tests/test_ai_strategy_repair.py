from __future__ import annotations

import json
from pathlib import Path

from atlas_ai.brain.ai_strategy_repair import AIStrategyRepairer
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def test_ai_strategy_repair_generates_proposal_only_variants(tmp_path: Path) -> None:
    parent = StrategyChromosome(
        uid="shadow_failed",
        generation=2,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_blocked:shadow_drawdown_too_high",
        shadow_signals=160,
        shadow_agreement_rate=0.5,
        shadow_passed=False,
        backtest_completed=True,
        backtest_score=62.4,
    )
    payload = {"generation": 2, "chromosomes": [parent.to_dict()]}
    (tmp_path / "evolution_population.json").write_text(json.dumps(payload), encoding="utf-8")
    repairer = AIStrategyRepairer(KnowledgeStore(str(tmp_path)))

    result = repairer.repair_shadow_failure("shadow_failed")

    assert result["ok"] is True
    assert result["executed"] is False
    assert len(result["variants"]) == 3
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))
    variants = [row for row in population["chromosomes"] if str(row["uid"]).startswith("repair_")]
    assert len(variants) == 3
    assert all(row["promotion_stage"] == "untested" for row in variants)
    assert all(row["backtest_completed"] is False for row in variants)
    assert result["safety"]["does_not_execute"] is True
    assert (tmp_path / "ai_strategy_repairs.json").exists()


def test_ai_strategy_repair_refuses_passed_shadow_candidate(tmp_path: Path) -> None:
    parent = StrategyChromosome(
        uid="shadow_passed",
        generation=2,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        shadow_passed=True,
    )
    payload = {"generation": 2, "chromosomes": [parent.to_dict()]}
    (tmp_path / "evolution_population.json").write_text(json.dumps(payload), encoding="utf-8")
    repairer = AIStrategyRepairer(KnowledgeStore(str(tmp_path)))

    result = repairer.repair_shadow_failure("shadow_passed")

    assert result == {"ok": False, "uid": "shadow_passed", "reason": "candidate_not_shadow_failed", "executed": False}


def test_ai_strategy_micro_repair_generates_near_threshold_variants(tmp_path: Path) -> None:
    parent = StrategyChromosome(
        uid="near_threshold",
        generation=3,
        genes=make_standard_gene_pool(),
        promotion_stage="backtested",
        promotion_reason="extended_sample_positive_below_shadow_threshold",
        backtest_completed=True,
        backtest_score=59.22,
    )
    payload = {"generation": 3, "chromosomes": [parent.to_dict()]}
    (tmp_path / "evolution_population.json").write_text(json.dumps(payload), encoding="utf-8")
    repairer = AIStrategyRepairer(KnowledgeStore(str(tmp_path)))

    result = repairer.micro_repair_below_threshold("near_threshold")

    assert result["ok"] is True
    assert result["executed"] is False
    assert len(result["variants"]) == 4
    assert all(v["origin"].startswith("ai_micro_repair:") for v in result["variants"])
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))
    variants = [row for row in population["chromosomes"] if str(row["uid"]).startswith("repair_")]
    assert len(variants) == 4
    assert all(row["promotion_stage"] == "untested" for row in variants)
