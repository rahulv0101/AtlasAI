from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas_ai.brain.ai_shadow_review_runner import AIShadowReviewRunner
from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def _prepared_population(tmp_path: Path, stage: str = "backtested") -> None:
    genes = make_standard_gene_pool()
    for gene in genes:
        if gene.name == "risk_min_confidence":
            gene.value = 0.35
        elif gene.name == "regime_allowed_sessions":
            gene.value = 3.0
        elif gene.name == "confirm_min_body_ratio":
            gene.value = 0.25
        elif gene.name == "confirm_volume_min_ratio":
            gene.value = 0.6
    chromosome = StrategyChromosome(
        uid="shadow_candidate",
        generation=1,
        genes=genes,
        promotion_stage=stage,
        promotion_reason="extended_backtest_passed_awaiting_shadow_review",
        backtest_completed=True,
        backtest_score=65.0,
        backtest_trades=220,
        backtest_win_rate=0.55,
        backtest_profit_factor=2.0,
    )
    payload = {"generation": 1, "chromosomes": [chromosome.to_dict()]}
    (tmp_path / "evolution_population.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ai_shadow_review_runner_records_paper_only_review(tmp_path: Path) -> None:
    _prepared_population(tmp_path)
    df = pd.DataFrame({
        "open": [2000.0 + i * 0.2 for i in range(260)],
        "high": [2001.0 + i * 0.2 for i in range(260)],
        "low": [1999.0 + i * 0.2 for i in range(260)],
        "close": [2000.5 + i * 0.2 for i in range(260)],
        "tick_volume": [120 + (i % 12) for i in range(260)],
        "time": pd.date_range("2024-01-01 08:00:00", periods=260, freq="15min"),
    })
    settings = Settings(
        hfm_login=1,
        hfm_password="x",
        hfm_server="demo",
        evolution_min_shadow_trades=1,
    )
    runner = AIShadowReviewRunner(settings, KnowledgeStore(str(tmp_path)))

    result = runner.run("shadow_candidate", df, bars=220)

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["safety"]["does_not_execute"] is True
    assert result["stage"] == "shadow"
    assert "promotion_governor" in result
    assert (tmp_path / "ai_shadow_reviews.jsonl").exists()
    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))
    restored = population["chromosomes"][0]
    assert restored["promotion_stage"] == "shadow"
    assert restored["shadow_signals"] == result["summary"]["trades"]
    assert restored["shadow_passed"] == result["promotion_governor"]["can_promote_to_demo"]


def test_ai_shadow_review_runner_refuses_missing_candidate(tmp_path: Path) -> None:
    settings = Settings(hfm_login=1, hfm_password="x", hfm_server="demo")
    runner = AIShadowReviewRunner(settings, KnowledgeStore(str(tmp_path)))

    result = runner.run("missing", pd.DataFrame({"close": []}))

    assert result == {"ok": False, "uid": "missing", "reason": "candidate_not_in_population", "executed": False}
