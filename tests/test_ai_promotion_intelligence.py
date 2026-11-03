from __future__ import annotations

from atlas_ai.brain.ai_promotion_intelligence import AIPromotionIntelligence
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def test_ai_promotion_intelligence_recommends_backtest_for_untested(tmp_path) -> None:
    c = StrategyChromosome(
        uid="ai_untested",
        generation=0,
        genes=make_standard_gene_pool(),
        origin="ai_inventor:trend_pullback_continuation",
        promotion_stage="untested",
    )
    report = AIPromotionIntelligence(tmp_path).evaluate(
        [c],
        strategy_preferences={"scores": {"trend_pullback_continuation": 0.5}},
    )

    top = report["top_recommendation"]
    assert top["uid"] == "ai_untested"
    assert top["recommendation"] == "run_offline_backtest"
    assert top["preference_score"] == 0.5
    assert top["safety"]["does_not_promote"] is True
    assert (tmp_path / "ai_promotion_recommendations.json").exists()


def test_ai_promotion_intelligence_recommends_demo_review_for_backtested(tmp_path) -> None:
    c = StrategyChromosome(
        uid="ai_backtested",
        generation=1,
        genes=make_standard_gene_pool(),
        origin="ai_inventor:trend_pullback_continuation",
        promotion_stage="backtested",
        backtest_completed=True,
        backtest_score=72.0,
        backtest_trades=250,
        backtest_win_rate=0.57,
        backtest_profit_factor=1.3,
    )
    report = AIPromotionIntelligence(tmp_path).evaluate(
        [c],
        strategy_preferences={"scores": {"trend_pullback_continuation": 0.2}},
    )

    assert report["top_recommendation"]["recommendation"] == "eligible_for_shadow_or_demo_review"
    assert "backtest_score:72.0" in report["top_recommendation"]["reasons"]


def test_ai_promotion_intelligence_flags_live_human_approval_boundary(tmp_path) -> None:
    c = StrategyChromosome(
        uid="demo_ready",
        generation=2,
        genes=make_standard_gene_pool(),
        origin="ai_inventor:range_reversion_filter",
        promotion_stage="demo",
        demo_active=True,
        demo_trades=140,
        demo_wins=80,
        demo_losses=60,
        demo_win_rate=80 / 140,
        demo_pnl_pct=0.04,
    )
    report = AIPromotionIntelligence(tmp_path).evaluate([c])

    assert report["top_recommendation"]["recommendation"] in {
        "operator_review_for_live_proposal",
        "continue_demo_validation",
    }
    assert report["safety"]["requires_human_approval_for_live"] is True
    assert report["safety"]["does_not_execute"] is True
