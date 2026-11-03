from atlas_ai.analysis.setup_hunter import build_setup_hunter


def test_setup_hunter_tracks_forming_radar_candidate_without_policy_change() -> None:
    status = {
        "chosen": {"side": "flat", "confidence": 0.0},
        "thresholds": {"confidence": 0.44},
        "signals": [{"strategy": "ema_crossover", "side": "buy", "confidence": 0.11}],
    }
    analysis = {
        "hard_blockers": ["no_clear_setup"],
        "setup_radar": {
            "watchlist": [
                {
                    "name": "pullback_continuation",
                    "side": "buy",
                    "score": 67,
                    "phase": "forming_confirmations_needed",
                    "trigger_state": "armed_waiting_for_final_confirmation",
                    "missing_to_entry": ["momentum_confirmation"],
                    "next_confirmation": "M1 momentum must turn back in the setup direction",
                }
            ]
        },
    }

    out = build_setup_hunter(status=status, decision_analysis=analysis)

    assert out["mode"] == "proactive_scan"
    assert out["state"] == "forming"
    assert out["closest_setup"]["name"] == "pullback_continuation"
    assert out["closest_setup"]["side"] == "buy"
    assert "momentum" in out["next_action"]
    assert "does not create fake setups" in out["policy_note"]


def test_setup_hunter_uses_raw_signal_when_no_radar_watchlist() -> None:
    out = build_setup_hunter(
        status={
            "thresholds": {"confidence": 0.44},
            "signals": [
                {"strategy": "gold_liquidity_sweep", "side": "sell", "confidence": 0.39},
                {"strategy": "flat_noise", "side": "flat", "confidence": 0.9},
            ],
        },
        decision_analysis={"setup_radar": {"watchlist": []}},
    )

    assert out["candidate_count"] == 1
    assert out["closest_setup"]["source"] == "strategy_signal"
    assert out["closest_setup"]["name"] == "gold_liquidity_sweep"
    assert out["closest_setup"]["phase"] == "forming"
