from __future__ import annotations

from atlas_ai.analysis.signal_correlation import (
    build_signal_correlation_report,
    build_signal_correlation_report_from_store,
    signal_correlation_matrix_csv,
)
from atlas_ai.main import parse_args
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui_server import create_ui_app


def _rows() -> list[dict]:
    sides = ["buy", "buy", "sell", "sell", "flat", "buy"]
    inverse = ["sell", "sell", "buy", "buy", "flat", "sell"]
    mixed = ["buy", "flat", "sell", "flat", "buy", "sell"]
    return [
        {
            "time": f"t{idx}",
            "signals": [
                {"strategy": "alpha", "side": side, "confidence": 0.7},
                {"strategy": "alpha_clone", "side": side, "confidence": 0.6},
                {"strategy": "inverse_alpha", "side": inverse[idx], "confidence": 0.6},
                {"strategy": "mixed", "side": mixed[idx], "confidence": 0.5},
            ],
        }
        for idx, side in enumerate(sides)
    ]


def test_signal_correlation_report_flags_redundant_strategy_pairs() -> None:
    report = build_signal_correlation_report(_rows(), rolling_window=20, threshold=0.8)

    assert report["read_only"] is True
    assert report["matrix"]["alpha"]["alpha_clone"]["pearson"] == 1.0
    assert any(
        row["strategy_a"] == "alpha" and row["strategy_b"] == "alpha_clone"
        for row in report["flagged_pairs"]
    )
    assert "ranking weights are not changed" in report["policy_note"]


def test_signal_correlation_report_writes_json_and_csv(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_status(_rows()[-1])
    for row in _rows()[:-1]:
        store.log_event("signal_snapshot", "paper signal snapshot", {"signals": row["signals"]})

    report = build_signal_correlation_report_from_store(store, rolling_window=20, threshold=0.8)
    csv_text = signal_correlation_matrix_csv(report)

    assert (tmp_path / "signal_correlation_report.json").exists()
    assert (tmp_path / "signal_correlation_matrix.csv").exists()
    assert "alpha,1.0,1.0" in csv_text


def test_signal_correlation_report_api(tmp_path, monkeypatch) -> None:
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_status(_rows()[-1])
    for row in _rows()[:-1]:
        store.log_event("signal_snapshot", "paper signal snapshot", {"signals": row["signals"]})

    payload = create_ui_app(store).test_client().get("/api/reports/signal-correlation?threshold=0.8&window=20").get_json()

    assert payload["ok"] is True
    assert payload["report"]["matrix"]["alpha"]["alpha_clone"]["pearson"] == 1.0
    assert payload["report"]["flagged_pairs"]
    assert (tmp_path / "signal_correlation_report.json").exists()


def test_signal_correlation_report_mode_is_available() -> None:
    args = parse_args(
        [
            "signal-correlation-report",
            "--signal-correlation-window",
            "20",
            "--signal-correlation-threshold",
            "0.75",
        ]
    )

    assert args.mode == "signal-correlation-report"
    assert args.signal_correlation_window == 20
    assert args.signal_correlation_threshold == 0.75
