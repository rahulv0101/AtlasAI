from __future__ import annotations

import json
from pathlib import Path

from atlas_ai.brain.ai_shadow_review import AIShadowReviewPreparer
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def test_ai_shadow_review_prepares_passed_candidate_without_execution(tmp_path: Path) -> None:
    chromosome = StrategyChromosome(
        uid="candidate_pass",
        generation=2,
        genes=make_standard_gene_pool(),
        origin="ai_inventor:test",
        promotion_stage="rejected",
        reject_reason="insufficient_trades:31<200",
    )
    (tmp_path / "evolution_archive.jsonl").write_text(json.dumps(chromosome.to_dict()) + "\n", encoding="utf-8")
    (tmp_path / "ai_extended_backtests.jsonl").write_text(
        json.dumps({
            "uid": "candidate_pass",
            "passed_extended_sample": True,
            "score": 62.4,
            "result": {
                "trades": 255,
                "wins": 139,
                "losses": 116,
                "win_rate": 0.545,
                "profit_factor": 2.65,
                "sharpe": 0.47,
                "expectancy": 0.004,
                "max_drawdown": 0.029,
            },
        })
        + "\n",
        encoding="utf-8",
    )
    preparer = AIShadowReviewPreparer(KnowledgeStore(str(tmp_path)))

    result = preparer.prepare("candidate_pass")

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["stage"] == "backtested"
    assert result["safety"]["does_not_execute"] is True
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))
    restored = population["chromosomes"][0]
    assert restored["uid"] == "candidate_pass"
    assert restored["promotion_stage"] == "backtested"
    assert restored["promotion_reason"] == "extended_backtest_passed_awaiting_shadow_review"
    assert restored["backtest_score"] == 62.4
    assert (tmp_path / "leaderboard.json").exists()


def test_ai_shadow_review_refuses_unpassed_candidate(tmp_path: Path) -> None:
    (tmp_path / "ai_extended_backtests.jsonl").write_text(
        json.dumps({"uid": "candidate_fail", "passed_extended_sample": False})
        + "\n",
        encoding="utf-8",
    )
    preparer = AIShadowReviewPreparer(KnowledgeStore(str(tmp_path)))

    result = preparer.prepare("candidate_fail")

    assert result == {
        "ok": False,
        "uid": "candidate_fail",
        "reason": "passed_extended_backtest_not_found",
        "executed": False,
    }
