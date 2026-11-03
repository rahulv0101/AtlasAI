from __future__ import annotations

import json
from pathlib import Path
import tracemalloc

import pytest

from atlas_ai.learning import missed_entry_audit as audit


@pytest.mark.parametrize("target", ["", "consensus"])
def test_collector_streams_all_review_batches_without_full_file_read(tmp_path, monkeypatch, target):
    path = tmp_path / "missed_entry_outcome_reviews.jsonl"
    rows = [
        {"primary_blocker": blocker, "outcome": outcome}
        for blocker in ("consensus", "other")
        for outcome in ("missed_opportunity", "correct_hold_or_block", "inconclusive",
                        "reference_price_unavailable", "direction_unavailable", "other", None)
    ]
    batches = [{"reviews": rows[:8] + [None, "invalid"]}, {"reviews": rows[8:]}, {}]
    path.write_text("\ninvalid\nnull\n[]\n" + "\n".join(json.dumps(row) for row in batches), encoding="utf-8")
    original_read_text = Path.read_text
    original_list_reader = audit._read_jsonl_dicts

    def no_full_read(self, *args, **kwargs):
        assert self != path, "Review journal must not be loaded as full text"
        return original_read_text(self, *args, **kwargs)

    def no_full_list(source):
        assert source != path, "Review journal must not be materialized as a list"
        return original_list_reader(source)

    monkeypatch.setattr(Path, "read_text", no_full_read)
    monkeypatch.setattr(audit, "_read_jsonl_dicts", no_full_list)
    monkeypatch.setattr(audit, "_proposal_target_blocker", lambda *args: target)
    result = audit.build_blocker_selectivity_offline_collector(tmp_path)
    expected = [row for row in rows if (not target or row["primary_blocker"] == target)
                and row["outcome"] not in {"reference_price_unavailable", "direction_unavailable"}]
    counts = {key: sum(row["outcome"] == key for row in expected) for key in
              ("missed_opportunity", "correct_hold_or_block", "inconclusive")}
    assert result["reviewed_outcome_samples"] == len(expected)
    assert result["outcome_counts"] == counts
    assert result["metrics"] == {
        "missed_rate": round(counts["missed_opportunity"] / len(expected), 4),
        "correct_block_rate": round(counts["correct_hold_or_block"] / len(expected), 4),
        "unknown_rate": round(counts["inconclusive"] / len(expected), 4),
        "net_review_edge": 0.0,
    }
    assert result["policy"]["does_not_change_risk_policy"] is True


def test_collector_review_memory_does_not_scale_with_history(tmp_path):
    path = tmp_path / "missed_entry_outcome_reviews.jsonl"
    line = json.dumps({"reviews": [{"outcome": "missed_opportunity", "evidence": "x" * 8192}]}) + "\n"
    peaks = []
    for count in (100, 4000):
        with path.open("w", encoding="utf-8") as handle:
            for _ in range(count):
                handle.write(line)
        tracemalloc.start()
        try:
            result = audit.build_blocker_selectivity_offline_collector(tmp_path)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result["reviewed_outcome_samples"] == count
        assert result["outcome_counts"]["missed_opportunity"] == count
    assert path.stat().st_size > 32_000_000
    assert peaks[1] < 1_000_000
    assert peaks[1] < peaks[0] + 200_000


def test_collector_missing_review_file_has_zero_totals(tmp_path):
    result = audit.build_blocker_selectivity_offline_collector(tmp_path)
    assert result["reviewed_outcome_samples"] == 0
    assert result["metrics"]["missed_rate"] == 0
