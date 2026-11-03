from __future__ import annotations

from pathlib import Path

from atlas_ai.testing.replay_harness import (
    ReplayDiffError,
    assert_behavior_matches_baseline,
    default_replay_scenarios,
    load_replay_baselines,
    run_replay_scenario,
)


def test_replay_harness_matches_behavior_baselines():
    baseline_path = Path(__file__).parent / "fixtures" / "replay_baselines.json"
    baselines = load_replay_baselines(baseline_path)

    for scenario in default_replay_scenarios():
        actual = run_replay_scenario(scenario)
        assert_behavior_matches_baseline(scenario.name, actual, baselines[scenario.name])


def test_replay_harness_reports_exact_diffs():
    actual = {
        "action": "paper",
        "execution": {"success": False},
    }
    baseline = {
        "action": "blocked",
        "execution": {"success": False},
    }

    try:
        assert_behavior_matches_baseline("demo", actual, baseline)
        assert False, "expected ReplayDiffError"
    except ReplayDiffError as exc:
        text = str(exc)
        assert "scenario=demo" in text
        assert "action: expected 'blocked', got 'paper'" in text
