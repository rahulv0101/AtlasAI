from __future__ import annotations

import json

from atlas_ai.brain.ai_demo_review_packet import AIDemoReviewPacket
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def test_ai_demo_review_packet_builds_research_only_handoff(tmp_path) -> None:
    chromosome = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        origin="ai_regime_repair:test",
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=66.7,
        backtest_trades=250,
        backtest_win_rate=0.61,
        shadow_passed=True,
        shadow_signals=250,
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
                "status": "demo_review_requested",
                "reasons": ["drawdown_policy=gross_normalized,limit=5.00%"],
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_reviews.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "recommendation": "eligible_for_demo_review",
            "summary": {"trades": 250, "win_rate": 0.61},
            "promotion_governor": {"can_promote_to_demo": True, "shadow_drawdown_policy": "gross_normalized"},
        })
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "ai_shadow_risk_audits.jsonl").write_text(
        json.dumps({
            "uid": "demo_candidate",
            "metrics": {"gross_normalized_drawdown_pct": 0.047},
            "diagnosis": {"primary": "raw_peak_drawdown_metric_over_sensitive"},
            "guard": {"gross_normalized_drawdown_passed": True},
        })
        + "\n",
        encoding="utf-8",
    )

    packet = AIDemoReviewPacket(tmp_path).build("demo_candidate")

    assert packet["ok"] is True
    assert packet["executed"] is False
    assert packet["safety"]["does_not_promote_demo"] is True
    assert packet["safety"]["does_not_promote_live"] is True
    assert packet["queue_item"]["status"] == "demo_review_requested"
    assert packet["shadow_review"]["promotion_governor"]["shadow_drawdown_policy"] == "gross_normalized"
    assert (tmp_path / "ai_demo_review_packet_demo_candidate.json").exists()


def test_ai_demo_review_packet_missing_candidate(tmp_path) -> None:
    packet = AIDemoReviewPacket(tmp_path).build("missing")

    assert packet == {"ok": False, "uid": "missing", "reason": "candidate_not_in_population", "executed": False}
