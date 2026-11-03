from __future__ import annotations

import json

from atlas_ai.brain.ai_demo_activation import AIDemoActivator
from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool


def _seed_activation(tmp_path, *, preflight_ok: bool = True, queue_status: str = "demo_activation_proposed") -> None:
    chromosome = StrategyChromosome(
        uid="demo_candidate",
        generation=1,
        genes=make_standard_gene_pool(),
        promotion_stage="shadow",
        promotion_reason="shadow_review_passed_awaiting_demo_review",
        backtest_completed=True,
        backtest_score=67.0,
        backtest_trades=260,
        backtest_win_rate=0.61,
        shadow_passed=True,
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
                "safety": {"does_not_execute": True, "does_not_promote_live": True},
            }]
        }),
        encoding="utf-8",
    )
    (tmp_path / "ai_demo_activation_preflight_demo_candidate.json").write_text(
        json.dumps({"ok": preflight_ok, "blockers": [] if preflight_ok else ["blocked"]}),
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


def test_demo_activation_requires_explicit_command(tmp_path) -> None:
    _seed_activation(tmp_path)

    result = AIDemoActivator(tmp_path).activate("demo_candidate", "activate")

    assert result["ok"] is False
    assert result["reason"] == "explicit_demo_activation_command_required"
    assert result["state_changed"] is False
    assert result["trade_executed"] is False


def test_demo_activation_rejects_loose_command_with_uid(tmp_path) -> None:
    _seed_activation(tmp_path)

    result = AIDemoActivator(tmp_path).activate("demo_candidate", "Activate demo immediately for demo_candidate")

    assert result["ok"] is False
    assert result["reason"] == "explicit_demo_activation_command_required"
    assert result["state_changed"] is False


def test_demo_activation_requires_passed_preflight(tmp_path) -> None:
    _seed_activation(tmp_path, preflight_ok=False)

    result = AIDemoActivator(tmp_path).activate("demo_candidate", "Activate demo for demo_candidate")

    assert result["ok"] is False
    assert result["reason"] == "preflight_not_passed"
    assert result["state_changed"] is False


def test_demo_activation_sets_demo_only_and_updates_queue(tmp_path) -> None:
    _seed_activation(tmp_path)

    result = AIDemoActivator(tmp_path).activate("demo_candidate", "Activate demo for demo_candidate")

    population = json.loads((tmp_path / "evolution_population.json").read_text(encoding="utf-8"))
    candidate = population["chromosomes"][0]
    queue = json.loads((tmp_path / "ai_research_queue.json").read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert result["state_changed"] is True
    assert result["demo_activated"] is True
    assert result["trade_executed"] is False
    assert result["live_promoted"] is False
    assert result["safety"]["does_not_promote_live"] is True
    assert candidate["promotion_stage"] == "demo"
    assert candidate["demo_active"] is True
    assert candidate["promotion_reason"] == "demo_activated_by_explicit_operator_command"
    assert candidate.get("demo_trades", 0) == 0
    assert candidate["promotion_stage"] != "proposed_live"
    assert queue["items"][0]["status"] == "demo_activated"
    assert (tmp_path / "ai_demo_activation_demo_candidate.json").exists()
