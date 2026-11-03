from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas_ai.brain.ai_extended_backtest import AIExtendedBacktestRunner
from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import make_standard_gene_pool, StrategyChromosome


def test_ai_extended_backtest_runs_archived_candidate_without_execution(tmp_path: Path) -> None:
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
        uid="candidate_1",
        generation=1,
        genes=genes,
        origin="ai_inventor:test",
        promotion_stage="rejected",
        reject_reason="insufficient_trades:10<200",
    )
    (tmp_path / "evolution_archive.jsonl").write_text(json.dumps(chromosome.to_dict()) + "\n", encoding="utf-8")
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
        evolved_candidate_min_trades=1,
        research_confidence_threshold=0.1,
        backtest_spread_cost_bps=0.0,
        backtest_slippage_cost_bps=0.0,
    )
    runner = AIExtendedBacktestRunner(settings, KnowledgeStore(str(tmp_path)))

    result = runner.run_for_uid("candidate_1", df)

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["safety"]["does_not_execute"] is True
    assert "result" in result
    assert (tmp_path / "ai_extended_backtests.jsonl").exists()


def test_ai_extended_backtest_missing_candidate_is_safe(tmp_path: Path) -> None:
    settings = Settings(hfm_login=1, hfm_password="x", hfm_server="demo")
    runner = AIExtendedBacktestRunner(settings, KnowledgeStore(str(tmp_path)))

    result = runner.run_for_uid("missing", pd.DataFrame({"close": []}))

    assert result == {"ok": False, "reason": "candidate_not_found", "uid": "missing", "executed": False}
