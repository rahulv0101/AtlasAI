from __future__ import annotations

import json

from atlas_ai.brain.ai_demo_activation_proposal import AIDemoActivationProposal
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def _seed_candidate(tmp_path, *, queue_status: str = "demo_review_approved") -> StrategyChromosome:
    chromosome = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        origin="ai_regime_repair:test",
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        backtest_trades=260,
        backtest_win_rate=0.61,
        shadow_passed=True,
        shadow_signals=260,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"generation": 1, "chromosomes": [chromosome.to_dict()]}),
        encoding="utf-8",
    )
    (tmp_path / "ai_research_queue.json").write_text(
        json.dumps({
            "items": [{
                "id": "demo_candidate:eligible_for_demo_review",
                "uid": "demo_candidate",
                "recommendation": "eligible_for_demo_review",
                "status": queue_status,
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_review_packet_demo_candidate.json").write_text(
        json.dumps({
            "schema": "atlas_ai_demo_review_packet_v1",
            "safety": {"does_not_promote_demo": True, "does_not_promote_live": True},
        }),
        encoding="utf-8",
    )
    return chromosome


def test_ai_demo_activation_proposal_is_non_executing(tmp_path) -> None:
    _seed_candidate(tmp_path)

    proposal = AIDemoActivationProposal(tmp_path).build("demo_candidate")

    assert proposal["ok"] is True
    assert proposal["executed"] is False
    assert proposal["candidate"]["demo_active"] is False
    assert proposal["operator_boundary"]["does_not_call_promote_to_demo"] is True
    assert proposal["safety"]["does_not_promote_demo"] is True
    assert (tmp_path / "ai_demo_activation_proposal_demo_candidate.json").exists()


def test_ai_demo_activation_proposal_requires_review_approval(tmp_path) -> None:
    _seed_candidate(tmp_path, queue_status="demo_review_requested")

    proposal = AIDemoActivationProposal(tmp_path).build("demo_candidate")

    assert proposal["ok"] is False
    assert proposal["reason"] == "demo_review_not_approved"
    assert proposal["executed"] is False
