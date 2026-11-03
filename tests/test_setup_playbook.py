from __future__ import annotations

from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.analysis.setup_playbook import build_setup_quality_observation, build_setup_radar, score_setup_playbook


def test_pullback_continuation_is_top_playbook_for_clear_trend_pullback() -> None:
    out = score_setup_playbook(
        status={
            "chosen": {"side": "buy"},
            "setup_family": "pullback",
            "context_snapshot": {
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "spread_state": "tight",
            },
        },
        runtime={},
    )
    top = out["top_candidate"]
    assert top["name"] == "pullback_continuation"
    assert top["side"] == "buy"
    assert top["score"] >= 80
    assert top["readiness"] == "ready_for_existing_gates"
    assert "good_example" in top["teaching"]


def test_mid_range_mixed_context_surfaces_missing_execution_ingredients() -> None:
    out = score_setup_playbook(
        status={
            "chosen": {"side": "flat"},
            "setup_family": "continuation",
            "context_snapshot": {
                "trend_state": "uptrend",
                "trend_structure": "HH/HL",
                "momentum": "weak",
                "location": "mid_range",
            },
        },
        runtime={},
    )
    top = out["top_candidate"]
    assert top["name"] == "pullback_continuation"
    assert "pullback_entry_location" in top["missing"]
    assert "momentum_confirmation" in top["missing"]


def test_range_context_prefers_range_edge_reversal_over_continuation() -> None:
    out = score_setup_playbook(
        status={
            "chosen": {"side": "flat"},
            "setup_family": "mean_reversion",
            "context_snapshot": {
                "trend_state": "range",
                "momentum": "mixed",
                "location": "consolidation",
            },
        },
        runtime={},
    )
    assert out["top_candidate"]["name"] == "range_edge_reversal"


def test_macro_and_spread_risks_reduce_playbook_score_without_becoming_policy() -> None:
    out = score_setup_playbook(
        status={
            "chosen": {"side": "sell"},
            "setup_family": "breakout",
            "context_snapshot": {
                "trend_structure": "LH/LL",
                "momentum": "strong",
                "location": "breakout",
                "spread_state": "wide",
                "dxy_alignment": "conflict",
                "us10y_alignment": "conflict",
            },
        },
        runtime={"max_spread_points": 30},
    )
    top = out["top_candidate"]
    assert top["name"] == "breakout_retest"
    assert top["score"] < 80
    assert "execution_friction_elevated" in top["risk_notes"]
    assert "macro_alignment_conflict" in top["risk_notes"]
    assert out["mode"] == "advisory_only"
    assert "first break" in top["teaching"]["bad_example"].lower()


def test_decision_analysis_exposes_setup_playbook_and_protocol_summary() -> None:
    out = build_structured_decision_analysis(
        status={
            "action": "buy",
            "chosen": {"confidence": 0.82, "side": "buy"},
            "setup_family": "pullback",
            "context_snapshot": {
                "trend_structure": "HH/HL",
                "momentum": "strong",
                "location": "pullback",
                "spread_state": "tight",
            },
        },
        runtime={},
    )
    assert out["setup_playbook"]["top_candidate"]["name"] == "pullback_continuation"
    assert out["pre_trade_memo"]["setup_playbook"]["name"] == "pullback_continuation"
    orient = out["decision_protocol"]["orient"]["summary"]
    assert any("playbook=pullback_continuation" in item for item in orient)


def test_setup_radar_exposes_forming_setup_without_changing_policy() -> None:
    playbook = score_setup_playbook(
        status={
            "chosen": {"side": "buy"},
            "setup_family": "pullback",
            "context_snapshot": {
                "trend_structure": "HH/HL",
                "momentum": "weak",
                "location": "pullback",
                "spread_state": "tight",
            },
        },
        runtime={},
    )
    radar = build_setup_radar(
        playbook=playbook,
        context={"momentum": "weak", "location": "pullback", "spread_state": "tight", "trend_strength": 0.12},
        hard_blockers=[],
    )
    assert radar["mode"] == "observability_only"
    assert radar["timing_state"] == "setup_forming"
    assert radar["top_watch"]["name"] == "pullback_continuation"
    assert "momentum_confirmation" in radar["top_watch"]["missing_to_entry"]
    assert radar["top_watch"]["trigger_state"] == "armed_waiting_for_final_confirmation"
    assert "momentum" in radar["top_watch"]["next_confirmation"].lower()
    assert "does not lower thresholds" in radar["policy_note"]


def test_setup_radar_explains_slow_setup_conditions() -> None:
    playbook = score_setup_playbook(
        status={
            "chosen": {"side": "flat"},
            "setup_family": "continuation",
            "context_snapshot": {
                "trend_structure": "LH/LL",
                "momentum": "mixed",
                "location": "mid_range",
                "spread_state": "wide",
            },
        },
        runtime={},
    )
    radar = build_setup_radar(
        playbook=playbook,
        context={
            "momentum": "mixed",
            "location": "mid_range",
            "spread_state": "wide",
            "spread_points": 45,
            "trend_strength": 0.01,
        },
        hard_blockers=["spread_liquidity_too_poor", "no_clear_setup"],
    )
    joined = " ".join(radar["conditions_slowing_setups"])
    assert "spread/liquidity" in joined
    assert "mid-range" in joined
    assert "trend strength" in joined


def test_setup_radar_trigger_checklist_explains_mid_range_pullback_wait() -> None:
    playbook = score_setup_playbook(
        status={
            "chosen": {"side": "flat"},
            "setup_family": "continuation",
            "context_snapshot": {
                "trend_structure": "LH/LL",
                "momentum": "mixed",
                "location": "mid_range",
                "spread_state": "normal",
            },
        },
        runtime={},
    )
    radar = build_setup_radar(
        playbook=playbook,
        context={"momentum": "mixed", "location": "mid_range", "spread_state": "normal", "trend_strength": 0.18},
        hard_blockers=["no_clear_setup"],
    )

    top = radar["top_watch"]
    labels = " ".join(str(x.get("label", "")) for x in top["trigger_checklist"])
    assert top["trigger_state"] == "forming_waiting_for_confirmations"
    assert "pull back into a valid structure" in labels
    assert "M1 momentum" in labels
    assert "mid-range" in " ".join(radar["conditions_slowing_setups"])


def test_setup_quality_observation_turns_playbook_into_advisory_lesson() -> None:
    playbook = score_setup_playbook(
        status={
            "chosen": {"side": "buy"},
            "setup_family": "continuation",
            "context_snapshot": {
                "trend_structure": "LH/LL",
                "momentum": "weak",
                "location": "mid_range",
            },
        },
        runtime={},
    )
    out = build_setup_quality_observation(
        playbook,
        {"trend_structure": "LH/LL", "momentum": "weak", "location": "mid_range", "chosen_side": "buy"},
    )
    assert out["mode"] == "advisory_only"
    assert out["score"] > 0
    assert out["bias_entry_conflict"] is True
    assert "mid-range is observation, not entry" in out["lessons"]
    assert "does not change strategy" in out["policy_note"]
