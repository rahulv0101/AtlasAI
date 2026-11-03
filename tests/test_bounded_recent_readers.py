from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from atlas_ai.analysis.signal_correlation import load_signal_rows_from_store
from atlas_ai.brain.ai_decision_journal import AIDecisionJournal
from atlas_ai.brain.ai_decision_reviewer import AIDecisionReviewer
from atlas_ai.brain.ai_research_queue import AIResearchQueue
from atlas_ai.learning.experiment_runner import build_joined_decision_outcome_samples
from atlas_ai.learning.experiment_tracker import ExperimentTracker
from atlas_ai.learning.loss_lesson_backfill import _decision_events
from atlas_ai.learning.missed_entry_audit import (
    build_blocker_selectivity_review,
    recent_missed_entry_audits,
    summarize_missed_entry_learning,
)
from atlas_ai.learning.outcome_intelligence import build_trade_outcome_memory
from atlas_ai.ui_server import _latest_successful_desktop_start


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _forbid_jsonl_read_text(monkeypatch) -> None:
    original = Path.read_text

    def guarded(path: Path, *args, **kwargs):
        if path.suffix == ".jsonl" or path.name.endswith(".log"):
            raise AssertionError(f"full-file read forbidden for bounded reader: {path}")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)


def test_large_status_and_decision_readers_never_read_full_jsonl(tmp_path: Path, monkeypatch) -> None:
    _write_jsonl(
        tmp_path / "missed_entry_outcome_reviews.jsonl",
        [{"reviews": [{"primary_blocker": "timing", "outcome": "missed_opportunity", "move_points": 90}]}],
    )
    _write_jsonl(tmp_path / "missed_entry_audits.jsonl", [{"created_at": "one"}, {"created_at": "two"}])
    _write_jsonl(tmp_path / "ai_decision_reviews.jsonl", [{"id": 1}, {"id": 2}])
    _write_jsonl(tmp_path / "ai_decision_journal.jsonl", [{"id": 1}, {"id": 2}])
    _forbid_jsonl_read_text(monkeypatch)

    assert summarize_missed_entry_learning(tmp_path, limit=1)["available"] is True
    assert build_blocker_selectivity_review(tmp_path, primary_blocker="timing", limit=1)["available"] is True
    assert recent_missed_entry_audits(tmp_path, limit=1)[0]["created_at"] == "two"
    assert AIDecisionReviewer(tmp_path).recent_reviews(1)[0]["id"] == 2
    journal = AIDecisionJournal(tmp_path)
    assert journal.recent(1)[0]["id"] == 2
    assert journal.latest()["id"] == 2


class _TailOnlyStore:
    def __init__(self, base: Path) -> None:
        self.base = base
        self.events_file = base / "events.jsonl"
        self.calls: list[tuple[int, str | None]] = []

    def load_events(self, *args, **kwargs):
        raise AssertionError("bounded caller must not use load_events")

    def tail_events(self, limit: int, event_type: str | None = None):
        self.calls.append((limit, event_type))
        return []

    def load_learning_state(self):
        return {"state": {"loss_memory": {"recent_outcomes": []}}}


def test_research_callers_never_use_full_event_loader(tmp_path: Path) -> None:
    store = _TailOnlyStore(tmp_path)

    assert load_signal_rows_from_store(store, rolling_window=12) == []
    assert build_joined_decision_outcome_samples(store) == []
    assert build_trade_outcome_memory(store) == []
    assert _decision_events(store, event_limit=25) == []

    assert (12, None) in store.calls
    assert store.calls.count((5000, "decision_sample")) == 2
    assert (25, "decision_sample") in store.calls


def test_small_recent_readers_never_read_full_files(tmp_path: Path, monkeypatch) -> None:
    tracker = ExperimentTracker(str(tmp_path))
    tracker.log_run("one", "a", {}, 1)
    tracker.log_run("two", "b", {}, 1)
    _write_jsonl(tmp_path / "evolution_archive.jsonl", [{"uid": "old"}, {"uid": "new", "reject_reason": "x"}])
    _write_jsonl(tmp_path / "ai_extended_backtests.jsonl", [{"uid": "new", "recommendation": "review"}])
    _write_jsonl(tmp_path / "ai_shadow_reviews.jsonl", [{"uid": "new", "recommendation": "review"}])
    _write_jsonl(tmp_path / "ai_shadow_risk_audits.jsonl", [{"uid": "new", "diagnosis": {}}])
    log = tmp_path / "logs" / "ui_startup_diagnostics.log"
    log.parent.mkdir(parents=True)
    log.write_text("2026-09-29T00:00:00+00:00 | ok=True | stage=started | error=\n", encoding="utf-8")
    _forbid_jsonl_read_text(monkeypatch)

    assert tracker.list_runs(limit=1)[0]["name"] == "two"
    queue = AIResearchQueue(tmp_path)
    queue.sync_from_archive(limit=1)
    queue.sync_from_extended_backtests(limit=1)
    queue.sync_from_shadow_reviews(limit=1)
    queue.sync_from_shadow_risk_audits(limit=1)
    assert _latest_successful_desktop_start(SimpleNamespace(base=tmp_path)) is not None


def test_desktop_startup_reader_source_has_no_full_log_read() -> None:
    source = (Path(__file__).parents[1] / "atlas_ai" / "desktop.py").read_text(encoding="utf-8")
    assert "diag_log.read_text" not in source
