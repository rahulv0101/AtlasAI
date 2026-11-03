from __future__ import annotations

import json

from atlas_ai.brain.ai_demo_activation_preflight import AIDemoActivationPreflight
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def _seed_preflight(tmp_path, *, queue_status: str = "demo_activation_proposed") -> None:
    chromosome = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        shadow_passed=True,
    )
    (tmp_path / "evolution_population.json").write_text(
        json.dumps({"chromosomes": [chromosome.to_dict()]}),
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
        json.dumps({"safety": {"does_not_promote_demo": True}}),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_activation_proposal_demo_candidate.json").write_text(
        json.dumps({"safety": {"does_not_promote_demo": True}}),
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_reviews.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "recommendation": "eligible_for_demo_review",
            "summary": {"trades": 260, "win_rate": 0.61},
            "promotion_governor": {"can_promote_to_demo": True},
        })
        + "\n",
        encoding="utf-8",
    )


def test_demo_activation_preflight_passes_without_execution(tmp_path) -> None:
    _seed_preflight(tmp_path)

    result = AIDemoActivationPreflight(tmp_path).run("demo_candidate")

    assert result["ok"] is True
    assert result["executed"] is False
    assert result["safety"]["does_not_promote_demo"] is True
    assert result["operator_boundary"]["does_not_call_promote_to_demo"] is True
    assert not result["blockers"]
    assert (tmp_path / "ai_demo_activation_preflight_demo_candidate.json").exists()


def test_demo_activation_preflight_blocks_wrong_queue_state(tmp_path) -> None:
    _seed_preflight(tmp_path, queue_status="demo_review_approved")

    result = AIDemoActivationPreflight(tmp_path).run("demo_candidate")

    assert result["ok"] is False
    assert "queue_status_demo_activation_proposed" in result["blockers"]
    assert result["executed"] is False
