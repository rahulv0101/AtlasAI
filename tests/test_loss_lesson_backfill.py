from __future__ import annotations

import json

from atlas_ai.learning.loss_lesson_backfill import rebuild_loss_pattern_lessons
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_rebuild_loss_pattern_lessons_enriches_outcomes_and_lessons(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "time": "2026-05-19T12:06:09+00:00",
                        "outcome": "loss",
                        "strategy": "gold_micro_impulse",
                        "side": "sell",
                        "session": "london",
                        "entry_session": "london",
                        "regime": "range",
                        "setup_family": "continuation",
                        "net_pnl_zar": -14.14,
                    },
                    {
                        "time": "2026-05-19T12:24:09+00:00",
                        "outcome": "loss",
                        "strategy": "gold_micro_impulse",
                        "side": "sell",
                        "session": "london",
                        "entry_session": "london",
                        "regime": "range",
                        "setup_family": "continuation",
                        "net_pnl_zar": -12.0,
                    },
                ]
            }
        }
    )

    summary = rebuild_loss_pattern_lessons(store)
    state = store.load_learning_state()["state"]
    memory = state["loss_memory"]
    lessons = memory["pattern_lessons"]
    lesson = next(iter(lessons.values()))

    assert summary["pattern_lessons"] == 1
    assert summary["loss_dominant_lessons"] == 1
    assert len(memory["recent_outcomes"]) == 2
    assert memory["recent_outcomes"][0]["pattern_fingerprint"].startswith("strategy=gold_micro_impulse")
    assert memory["recent_outcomes"][0]["post_trade_lesson"]["bounded_learning_only"] is True
    assert lesson["samples"] == 2
    assert lesson["losses"] == 2
    assert lesson["loss_rate"] == 1.0
    assert lesson["bounded_learning_only"] is True
    assert lesson["post_trade_lesson"]["failure_axis"]


def test_rebuild_loss_pattern_lessons_preserves_existing_learning_state(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "allocator": {"ema_crossover": {"wins": 1, "losses": 0, "pnl": 5.0}},
            "loss_memory": {"recent_outcomes": []},
        }
    )

    summary = rebuild_loss_pattern_lessons(store)
    state = store.load_learning_state()["state"]

    assert summary["pattern_lessons"] == 0
    assert state["allocator"]["ema_crossover"]["wins"] == 1
    assert state["loss_memory"]["pattern_lessons"] == {}


def test_rebuild_loss_pattern_lessons_normalizes_momentum_alias(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "time": "2026-05-15T16:38:03+00:00",
                        "outcome": "loss",
                        "strategy": "mom",
                        "side": "sell",
                        "session": "new_york",
                        "regime": "unknown",
                        "setup_family": "continuation",
                    }
                ]
            }
        }
    )

    rebuild_loss_pattern_lessons(store)
    lessons = store.load_learning_state()["state"]["loss_memory"]["pattern_lessons"]
    fingerprint = next(iter(lessons))

    assert fingerprint.startswith("strategy=momentum_continuation")


def test_rebuild_loss_pattern_lessons_links_nearby_decision_context(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.events_file.write_text(
        json.dumps(
            {
                "time": "2026-05-19T11:58:00+00:00",
                "type": "decision_sample",
                "message": "sample",
                "meta": {
                    "decision_id": "abc123",
                    "strategy": "gold_micro_impulse",
                    "side": "sell",
                    "session": "london",
                    "regime": "trend",
                    "confidence": 0.56,
                    "trend_strength": 0.19,
                    "volatility_pct": 0.033,
                    "setup_family": "continuation",
                    "higher_tf_bias": "sell",
                    "spread_points": 34.0,
                    "contextual_ranking": {
                        "context": {
                            "body_ratio": 0.72,
                            "trend_strength": 0.19,
                            "volatility_pct": 0.033,
                            "htf_aligned": True,
                        },
                        "setup_quality": {"score": 68.0, "reasons": ["pullback location improving"]},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "time": "2026-05-19T12:06:09+00:00",
                        "outcome": "loss",
                        "strategy": "gold_micro_impulse",
                        "side": "sell",
                        "session": "london",
                        "regime": "unknown",
                        "setup_family": "continuation",
                    }
                ]
            }
        }
    )

    summary = rebuild_loss_pattern_lessons(store)
    memory = store.load_learning_state()["state"]["loss_memory"]
    row = memory["recent_outcomes"][0]
    fingerprint = row["pattern_fingerprint"]

    assert summary["decision_context"]["outcomes_linked"] == 1
    assert summary["context_quality"]["momentum_known"] == 1
    assert row["decision_context_link"]["linked"] is True
    assert row["context"]["entry_momentum_state"] == "strong"
    assert row["context"]["entry_location_state"] == "pullback"
    assert row["context"]["volatility_state"] == "normal"
    assert "momentum=strong" in fingerprint
    assert "location=pullback" in fingerprint
    assert "volatility=normal" in fingerprint


def test_rebuild_loss_pattern_lessons_reads_archived_decision_samples(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "events_20260519_120000.jsonl").write_text(
        json.dumps(
            {
                "time": "2026-05-19T11:58:00+00:00",
                "type": "decision_sample",
                "message": "sample",
                "meta": {
                    "decision_id": "archived",
                    "strategy": "momentum_continuation",
                    "side": "buy",
                    "session": "new_york",
                    "regime": "range",
                    "confidence": 0.51,
                    "trend_strength": 0.09,
                    "volatility_pct": 0.018,
                    "setup_family": "continuation",
                    "higher_tf_bias": "buy",
                    "contextual_ranking": {"context": {"body_ratio": 0.42, "trend_strength": 0.09, "volatility_pct": 0.018}},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    store.save_learning_state(
        {
            "loss_memory": {
                "recent_outcomes": [
                    {
                        "time": "2026-05-19T12:10:00+00:00",
                        "outcome": "loss",
                        "strategy": "momentum_continuation",
                        "side": "buy",
                        "session": "new_york",
                        "setup_family": "continuation",
                    }
                ]
            }
        }
    )

    summary = rebuild_loss_pattern_lessons(store)
    row = store.load_learning_state()["state"]["loss_memory"]["recent_outcomes"][0]

    assert summary["decision_context"]["decision_samples_scanned"] == 1
    assert summary["decision_context"]["outcomes_linked"] == 1
    assert row["decision_context_link"]["decision_id"] == "archived"
    assert row["context"]["volatility_state"] == "low"
