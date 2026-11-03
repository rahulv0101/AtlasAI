from __future__ import annotations

from atlas_ai.learning.fast_path_audit import build_fast_path_promotion_audit_report
from atlas_ai.main import parse_args
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui_server import create_ui_app


def _observation(fingerprint: str, mode: str, updated_at: str) -> dict:
    return {
        "top_fingerprint": fingerprint,
        "mode": mode,
        "updated_at": updated_at,
        "evidence_strength": 0.6,
        "overreaction_risk": 0.2,
    }


def test_fast_path_audit_counts_path_entries_transitions_and_promotions(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "online_learning": {
                "recent_observations": [
                    _observation("alpha", "slow_path_observe", "2026-08-09T00:00:00+00:00"),
                    _observation("alpha", "medium_path_observe", "2026-08-09T01:00:00+00:00"),
                    _observation("alpha", "fast_path_observe", "2026-08-09T02:00:00+00:00"),
                    _observation("beta", "slow_path_observe", "2026-08-09T00:15:00+00:00"),
                ],
            }
        }
    )
    store.append_self_improvement_audit(
        {
            "time": "2026-08-09T03:00:00+00:00",
            "event": "rollout_transitioned",
            "proposal_id": "alpha",
            "status": "promotion_candidate",
            "reason": "guarded_rollout_passed",
        }
    )
    store.append_self_improvement_audit(
        {
            "time": "2026-08-09T04:00:00+00:00",
            "event": "promotion_completed",
            "proposal_id": "alpha",
            "status": "promoted",
            "reason": "approved_rule_registry_updated",
        }
    )

    report = build_fast_path_promotion_audit_report(store, window_hours=0)

    assert report["read_only"] is True
    assert report["path_counts"]["slow_path"]["entered"] == 2
    assert report["path_counts"]["medium_path"]["entered"] == 1
    assert report["path_counts"]["fast_path"]["entered"] == 1
    assert report["transitions"]["slow_to_medium"] == 1
    assert report["transitions"]["medium_to_fast"] == 1
    assert report["promotion"]["promotion_candidate_events"] == 1
    assert report["promotion"]["promoted_events"] == 1
    assert report["path_counts"]["fast_path"]["reached_promotion"] == 1
    assert "does not change promotion logic" in report["policy_note"]


def test_fast_path_audit_writes_report_file(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "online_learning": {
                "recent_observations": [
                    _observation("alpha", "slow_path_observe", "2026-08-09T00:00:00+00:00"),
                ],
            }
        }
    )

    report = build_fast_path_promotion_audit_report(store, window_hours=24, write_file=True)

    assert (tmp_path / "fast_path_promotion_audit_report.json").exists()
    assert report["report_path"].endswith("fast_path_promotion_audit_report.json")


def test_fast_path_audit_report_api(tmp_path, monkeypatch) -> None:
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.ui_server.announce_ui_ready_once", lambda *_args, **_kwargs: True)
    store.save_learning_state(
        {
            "online_learning": {
                "recent_observations": [
                    _observation("alpha", "slow_path_observe", "2026-08-09T00:00:00+00:00"),
                    _observation("alpha", "medium_path_observe", "2026-08-09T01:00:00+00:00"),
                ],
            }
        }
    )

    payload = create_ui_app(store).test_client().get("/api/reports/fast-path-audit?window_hours=0").get_json()

    assert payload["ok"] is True
    assert payload["report"]["transitions"]["slow_to_medium"] == 1
    assert (tmp_path / "fast_path_promotion_audit_report.json").exists()


def test_fast_path_audit_cli_accepts_window_hours() -> None:
    args = parse_args(["fast-path-audit-report", "--fast-path-audit-window-hours", "12"])

    assert args.mode == "fast-path-audit-report"
    assert args.fast_path_audit_window_hours == 12
