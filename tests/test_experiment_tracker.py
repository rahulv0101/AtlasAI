from atlas_ai.learning.experiment_tracker import ExperimentTracker


def test_tracker_logs_and_summarizes(tmp_path):
    tracker = ExperimentTracker(str(tmp_path))
    tracker.log_run("exp_a", "baseline", {"accuracy": 0.5, "brier": 0.25}, sample_size=20)
    tracker.log_run("exp_a", "candidate", {"accuracy": 0.6, "brier": 0.22}, sample_size=20)
    tracker.log_run("exp_a", "candidate", {"accuracy": 0.7, "brier": 0.20}, sample_size=25)

    rows = tracker.list_runs(name="exp_a")
    assert len(rows) == 3
    summary = tracker.summarize("exp_a")
    assert summary["total_runs"] == 3
    assert "baseline" in summary["variants"]
    assert "candidate" in summary["variants"]

