from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from atlas_ai.brain.ai_shadow_risk_audit import AIShadowRiskAuditor
from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool
from atlas_ai.brain.ai_shadow_risk_audit import _session_from_hour


def test_shadow_risk_audit_records_metrics_without_execution(tmp_path: Path) -> None:
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
        uid="audit_candidate",
        generation=1,
        genes=genes,
        promotion_stage="shadow",
        backtest_completed=True,
        backtest_score=65.0,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"generation": 1, "chromosomes": [chromosome.to_dict()]}),
        encoding="utf-8",
    )
    df = pd.DataFrame({
        "open": [2000.0 + i * 0.2 for i in range(260)],
        "high": [2001.0 + i * 0.2 for i in range(260)],
        "low": [1999.0 + i * 0.2 for i in range(260)],
        "close": [2000.5 + i * 0.2 for i in range(260)],
        "tick_volume": [120 + (i % 12) for i in range(260)],
        "time": pd.date_range("2024-01-01 08:00:00", periods=260, freq="15min"),
    })
    auditor = AIShadowRiskAuditor(
        Settings(hfm_login=1, hfm_password="x", hfm_server="demo"),
        KnowledgeStore(str(tmp_path)),
    )

    result = auditor.audit("audit_candidate", df, bars=220)

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["safety"]["does_not_execute"] is True
    assert "metrics" in result
    assert "gross_normalized_drawdown_pct" in result["metrics"]
    assert "session_breakdown" in result["metrics"]
    assert "side_breakdown" in result["metrics"]
    assert "diagnosis" in result
    assert (tmp_path / "ai_shadow_risk_audits.jsonl").exists()


def test_shadow_risk_audit_refuses_missing_candidate(tmp_path: Path) -> None:
    auditor = AIShadowRiskAuditor(
        Settings(hfm_login=1, hfm_password="x", hfm_server="demo"),
        KnowledgeStore(str(tmp_path)),
    )

    result = auditor.audit("missing", pd.DataFrame({"close": []}))

    assert result == {"ok": False, "uid": "missing", "reason": "candidate_not_in_population", "executed": False}


def test_shadow_risk_audit_session_labels_match_evolved_strategy() -> None:
    assert _session_from_hour(6) == "asia"
    assert _session_from_hour(7) == "london"
    assert _session_from_hour(12) == "london"
    assert _session_from_hour(13) == "new_york"
    assert _session_from_hour(19) == "new_york"
    assert _session_from_hour(20) == "late_us"
