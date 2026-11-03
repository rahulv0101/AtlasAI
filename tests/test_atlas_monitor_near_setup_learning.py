from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_monitor_module():
    path = Path(__file__).resolve().parents[1] / "storage" / "monitoring" / "atlas_monitor.py"
    spec = importlib.util.spec_from_file_location("atlas_monitor_for_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_near_setup_learning_classifies_repeated_unclear_m1_setup() -> None:
    monitor = _load_monitor_module()
    status = {
        "action": "hold",
        "blocker_class": "signal_quality",
        "primary_blocker": "No high-confidence trade found.",
        "decision_analysis": {
            "confluence_score": 45,
            "confluence_threshold": 50,
            "hard_blockers": ["no_clear_setup"],
        },
    }
    market = {
        "regime": "trending",
        "trend_structure": "LH/LL",
        "momentum": "mixed",
        "location": "mid_range",
        "spread_points": 35.0,
    }
    sample = monitor.classify_near_setup(
        status,
        market,
        ["setup_family_no_trade", "momentum_mixed", "entry_location_mid_range"],
        ["momentum_quality:50.0/65.0", "entry_location:-10.0/0.0", "confluence_gap:45.0/50.0"],
    )

    assert sample["mode"] == "observability_only"
    assert sample["confluence_gap"] == 5.0
    assert sample["fingerprint"]
    assert "does not change strategy" in sample["policy_note"]
    assert any("mid-range" in item for item in sample["teaching_focus"])
    assert any("direction bias" in item for item in sample["teaching_focus"])


def test_near_setup_learning_updates_pattern_counts_without_policy_change() -> None:
    monitor = _load_monitor_module()
    state = {}
    sample = {
        "fingerprint": "regime=trending|structure=LH/LL|momentum=mixed|location=mid_range",
        "teaching_focus": ["separate direction bias from entry momentum"],
    }

    first = monitor.update_near_setup_learning(state, sample)
    second = monitor.update_near_setup_learning(state, sample)

    assert first["mode"] == "observability_only"
    assert second["total_samples"] == 2
    assert second["top_patterns"][0]["count"] == 2
    assert "does not alter live trading decisions" in second["policy_note"]


def test_near_setup_learning_teaches_breakout_confirmation_gap() -> None:
    monitor = _load_monitor_module()
    status = {
        "action": "hold",
        "blocker_class": "signal_quality",
        "primary_blocker": "No high-confidence trade found.",
        "decision_analysis": {
            "confluence_score": 45,
            "confluence_threshold": 50,
            "hard_blockers": ["no_clear_setup"],
        },
    }
    sample = monitor.classify_near_setup(
        status,
        {
            "regime": "trending",
            "trend_structure": "LH/LL",
            "momentum": "strong",
            "location": "breakout",
            "spread_points": 35.0,
        },
        ["setup_family_no_trade"],
        ["confluence_gap:45.0/50.0", "setup_clarity:75.0/60.0"],
    )

    assert any("confirmed breakout/retest" in item for item in sample["teaching_focus"])


def test_execution_diagnostics_classifies_failures() -> None:
    monitor = _load_monitor_module()
    diag = monitor.build_execution_diagnostics(
        {
            "action": "failed",
            "message": "Execution failed",
            "execution": {"success": False, "comment": "broker_confirmation_missing", "broker_order_id": ""},
        }
    )

    assert diag["failure_class"] == "broker_confirmation"
    assert "observability only" in diag["policy_note"]


def test_execution_diagnostics_ignores_stale_approval_truth(monkeypatch) -> None:
    monitor = _load_monitor_module()
    monkeypatch.setattr(monitor.time, "time", lambda: 1_000_000.0)

    diag = monitor.build_execution_diagnostics(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "execution_truth": {
                "ok": False,
                "reason": "approval_missing",
                "failure_class": "approval_or_authority",
                "request_id": "old_request",
                "ts": 998_000.0,
            },
        }
    )

    assert diag["truth_stale"] is True
    assert diag["truth"] == {}
    assert diag["ignored_stale_truth"]["reason"] == "approval_missing"
    assert diag["reason"] == "No high-confidence trade found."
    assert diag["failure_class"] == "setup_quality"


def test_execution_diagnostics_classifies_consensus_guard_as_quality_gate() -> None:
    monitor = _load_monitor_module()
    diag = monitor.build_execution_diagnostics(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
        }
    )

    assert diag["failure_class"] == "quality_gate"


def test_post_trade_learning_connects_closed_deal_to_context_without_policy_change() -> None:
    monitor = _load_monitor_module()
    state = {}
    learning = monitor.update_post_trade_learning(
        state,
        [
            {
                "ticket": 123,
                "order": 456,
                "position_id": 789,
                "time": "2026-05-15T12:28:05",
                "symbol": "XAUUSD",
                "volume": 0.01,
                "price": 4581.98,
                "profit": 27.06,
            }
        ],
        {"action": "execute", "execution": {"success": True, "broker_order_id": "456"}},
        {"regime": "trending", "trend_structure": "LH/LL", "momentum": "mixed", "location": "mid_range"},
        {"fingerprint": "regime=trending|structure=LH/LL|momentum=mixed|location=mid_range"},
    )

    assert learning["mode"] == "observability_only"
    assert learning["latest_outcomes"][0]["outcome"] == "win"
    assert learning["top_patterns"][0]["profit"] == 27.06
    assert "does not change strategy" in learning["policy_note"]


def test_post_trade_learning_ignores_zero_profit_entry_legs() -> None:
    monitor = _load_monitor_module()
    state = {}
    learning = monitor.update_post_trade_learning(
        state,
        [{"ticket": 123, "profit": 0.0, "comment": "", "symbol": "XAUUSD"}],
        {},
        {"regime": "trending"},
        {},
    )

    assert learning == {}
