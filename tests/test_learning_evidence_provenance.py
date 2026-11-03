from copy import deepcopy

from atlas_ai.learning.evidence_provenance import exclude_manual_tests, outcome_source, source_counts
from atlas_ai.learning.loss_lesson_backfill import rebuild_loss_pattern_lessons
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_manual_test_removal_is_exact_and_idempotent(tmp_path):
    manual = {"ticket": "99516875", "strategy": "mabr", "decision_id": "manual",
              "side": "buy", "hour_utc": 14, "outcome": "loss", "net_pnl_zar": -16.52,
              "pattern_fingerprint": "manual"}
    normal = {"ticket": "123", "strategy": "ema_crossover", "outcome": "loss", "side": "buy"}
    state = {"allocator": {"mabr": {"wins": 1, "losses": 2, "pnl": -16.52}},
             "loss_memory": {"recent_outcomes": [manual, normal], "processed_deals": ["99516875"],
                             "side_hour_stats": {"buy:14": {"wins": 0, "losses": 2, "pnl": -1498.04}},
                             "pattern_lessons": {"manual": {"samples": 1, "losses": 1, "loss_rate": 1}}}}
    original = deepcopy(state)
    clean = exclude_manual_tests(state)
    assert state == original
    assert clean == exclude_manual_tests(clean)
    assert clean["loss_memory"]["recent_outcomes"] == [normal]
    assert clean["loss_memory"]["processed_deals"] == ["99516875"]
    assert clean["loss_memory"]["side_hour_stats"]["buy:14"]["losses"] == 1
    assert round(clean["loss_memory"]["side_hour_stats"]["buy:14"]["pnl"], 2) == -1481.52
    assert clean["allocator"]["mabr"] == {"wins": 1, "losses": 1, "pnl": 0}
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(state)
    rebuild_loss_pattern_lessons(store)
    saved = store.load_learning_state()["state"]
    assert len(saved["loss_memory"]["excluded_manual_test_outcomes"]) == 1
    assert all("mabr" not in key for key in saved["loss_memory"]["pattern_lessons"])
    store.update_learning_state(lambda current: current)
    assert store.load_learning_state()["state"] == saved


def test_source_labels_do_not_claim_numeric_tickets_are_verified():
    rows = [{"ticket": "bt_1"}, {"ticket": "123"}, {},
            {"ticket": "456", "forced_demo_entry": True},
            {"strategy": "manual_broker_test"}]
    assert [outcome_source(row) for row in rows] == ["historical_backtest", "broker_recorded", "unknown", "forced_probe", "manual_test"]
    assert source_counts({"recent_outcomes": rows}) == {
        "historical_backtest": 1, "broker_recorded": 1, "unknown": 1,
        "forced_probe": 1, "manual_test": 1, "excluded_manual_tests": 0}
    assert outcome_source({"decision_id": "manual", "strategy": "other"}) == "unknown"


def test_ui_reports_sources_separately(tmp_path, monkeypatch):
    from atlas_ai.ui_server import _build_learning_loop_status
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state({"loss_memory": {"recent_outcomes": [
        {"ticket": "bt_1"}, {"ticket": "123"},
        {"ticket": "456", "decision_id": "manual", "strategy": "mabr"}]}})
    monkeypatch.setattr("atlas_ai.ui_server._tail_learning_events_fast", lambda store: [])
    result = _build_learning_loop_status(store)
    assert result["outcome_sources"]["historical_backtest"] == 1
    assert result["outcome_sources"]["broker_recorded"] == 1
    assert result["outcome_sources"]["excluded_manual_tests"] == 1
