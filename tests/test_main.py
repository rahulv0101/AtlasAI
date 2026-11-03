from atlas_ai.main import (
    _LIVE_UI_OWNER_ENV,
    _ensure_live_worker_running,
    _record_ui_process_lifecycle,
    _record_self_improvement_runtime_guard,
    _record_ui_wrapper_exit,
    _ensure_ui_server_running,
    _ensure_ui_process_running,
    _is_spawned_live_ui_child,
    _mode_requires_venv,
    _allow_non_venv_main,
    _mode_process_count,
    _prime_desktop_wrapper_state,
    _start_live_worker_process,
    _start_ui_process,
    _start_ui_server_process,
    _should_run_startup_historical_training,
    _should_exit_non_venv_duplicate,
    _should_terminate_duplicate_process,
    ModeReconciliationError,
    guarded_demo_startup_sanity,
    reconcile_runtime_flags,
)
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import append_or_update_proposal
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta


def test_should_terminate_duplicate_process_for_non_venv_live_ui_child():
    assert _should_terminate_duplicate_process(
        current_pid=100,
        mode="live-ui",
        pid=200,
        cmdline='"C:\\Python311\\python.exe" -m atlas_ai.main live-ui --poll-seconds 60',
    )


def test_should_not_terminate_current_or_venv_process():
    assert not _should_terminate_duplicate_process(
        current_pid=100,
        mode="live-ui",
        pid=100,
        cmdline='"C:\\Python311\\python.exe" -m atlas_ai.main live-ui',
    )
    assert not _should_terminate_duplicate_process(
        current_pid=100,
        mode="live-ui",
        pid=200,
        cmdline='"C:\\repo\\.venv\\Scripts\\python.exe" -m atlas_ai.main live-ui',
    )


def test_should_not_terminate_unrelated_mode():
    assert not _should_terminate_duplicate_process(
        current_pid=100,
        mode="live-ui",
        pid=200,
        cmdline='"C:\\Python311\\python.exe" -m atlas_ai.main backtest',
    )


def test_live_duplicate_reaper_does_not_kill_ui_processes_with_mode_override_live():
    assert not _should_terminate_duplicate_process(
        current_pid=100,
        mode="live",
        pid=200,
        cmdline='"C:\\Python311\\python.exe" -m atlas_ai.main ui --mode-override live',
    )
    assert not _should_terminate_duplicate_process(
        current_pid=100,
        mode="live",
        pid=300,
        cmdline='"C:\\Python311\\python.exe" -m atlas_ai.main ui-server --mode-override live',
    )


def test_is_spawned_live_ui_child(monkeypatch):
    monkeypatch.setenv(_LIVE_UI_OWNER_ENV, "1234")
    monkeypatch.setattr("atlas_ai.main.os.name", "nt")
    monkeypatch.setattr("atlas_ai.main.os.getpid", lambda: 5678)
    assert _is_spawned_live_ui_child() is True


def test_is_spawned_live_ui_child_false_for_owner(monkeypatch):
    monkeypatch.setenv(_LIVE_UI_OWNER_ENV, "1234")
    monkeypatch.setattr("atlas_ai.main.os.name", "nt")
    monkeypatch.setattr("atlas_ai.main.os.getpid", lambda: 1234)
    assert _is_spawned_live_ui_child() is False


def test_ensure_live_worker_running_skips_when_live_exists(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.main._mode_process_count", lambda mode: 1 if mode == "live" else 0)
    called = {"spawn": 0}
    monkeypatch.setattr("atlas_ai.main._start_live_worker_process", lambda poll: called.__setitem__("spawn", called["spawn"] + 1))

    _ensure_live_worker_running(60, store)

    assert called["spawn"] == 0
    events = store.tail_events(1)
    assert events[0]["message"].startswith("Live worker already running")


def test_ensure_live_worker_running_spawns_when_missing(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.main._mode_process_count", lambda mode: 0)
    called = {"spawn": 0}
    monkeypatch.setattr("atlas_ai.main._start_live_worker_process", lambda poll: called.__setitem__("spawn", poll))

    _ensure_live_worker_running(45, store)

    assert called["spawn"] == 45
    events = store.tail_events(1)
    assert events[0]["message"].startswith("Started detached live worker")


def test_mode_process_count_does_not_confuse_live_ui_with_live(monkeypatch):
    monkeypatch.setattr("atlas_ai.main.os.name", "nt")
    monkeypatch.setattr("atlas_ai.main.os.getpid", lambda: 100)
    monkeypatch.setattr(
        "atlas_ai.main._list_windows_python_processes",
        lambda: [
            (200, '"C:\\repo\\.venv\\Scripts\\python.exe" -m atlas_ai.main live-ui --poll-seconds 60'),
            (300, '"C:\\repo\\.venv\\Scripts\\python.exe" -m atlas_ai.main live --poll-seconds 60'),
        ],
    )

    assert _mode_process_count("live") == 1


def test_non_venv_duplicate_guard_blocks_non_venv_ui_modes(monkeypatch):
    monkeypatch.setattr("atlas_ai.main.os.name", "nt")
    monkeypatch.setattr("atlas_ai.main._is_venv_python", lambda: False)
    monkeypatch.setattr("atlas_ai.main.sys.argv", ["atlas_ai.main", "live-ui"])
    monkeypatch.delenv("ATLAS_ALLOW_NON_VENV_MAIN", raising=False)
    monkeypatch.setattr("atlas_ai.main._list_windows_python_processes", lambda: [(222, '"C:\\repo\\.venv\\Scripts\\python.exe" -m atlas_ai.main live-ui')])

    assert _should_exit_non_venv_duplicate() is True


def test_ensure_ui_process_running_spawns_when_missing(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.main._mode_process_count", lambda mode: 0)
    called = {"spawn": 0}
    monkeypatch.setattr("atlas_ai.main._start_ui_process", lambda: called.__setitem__("spawn", called["spawn"] + 1))

    _ensure_ui_process_running(store)

    assert called["spawn"] == 1
    events = store.tail_events(1)
    assert events[0]["message"].startswith("Started detached desktop UI process")


def test_spawn_helpers_use_preferred_python_executable(monkeypatch):
    seen: dict[str, list[str]] = {}

    def _capture(cmd: list[str]) -> None:
        seen["cmd"] = list(cmd)

    monkeypatch.setattr("atlas_ai.main._preferred_python_executable", lambda: "C:\\repo\\.venv\\Scripts\\python.exe")
    monkeypatch.setattr("atlas_ai.main._start_background_process", _capture)

    _start_live_worker_process(60)
    assert seen["cmd"][0].endswith(".venv\\Scripts\\python.exe")
    assert seen["cmd"][3] == "live"

    _start_ui_server_process(mode_override="live")
    assert seen["cmd"][0].endswith(".venv\\Scripts\\python.exe")
    assert seen["cmd"][3] == "ui-server"

    _start_ui_process()
    assert seen["cmd"][0].endswith(".venv\\Scripts\\python.exe")
    assert seen["cmd"][3] == "ui"


def test_mode_requires_venv_for_runtime_modes() -> None:
    assert _mode_requires_venv("live") is True
    assert _mode_requires_venv("ui-server") is True
    assert _mode_requires_venv("backtest") is False


def test_allow_non_venv_main_flag(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_ALLOW_NON_VENV_MAIN", raising=False)
    assert _allow_non_venv_main() is False
    monkeypatch.setenv("ATLAS_ALLOW_NON_VENV_MAIN", "true")
    assert _allow_non_venv_main() is True


def test_ensure_ui_server_running_reuses_existing_server(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setattr("atlas_ai.main._wait_for_url_ready", lambda url, timeout_seconds=20.0: True)
    called = {"spawn": 0}
    monkeypatch.setattr("atlas_ai.main._start_ui_server_process", lambda **kwargs: called.__setitem__("spawn", called["spawn"] + 1))

    ok = _ensure_ui_server_running(store, source_mode="ui")

    assert ok is True
    assert called["spawn"] == 0
    events = store.tail_events(1)
    assert events[0]["message"].startswith("UI server already running")
    flags = store.load_runtime_flags()
    last = dict(flags.get("ui_process_roles", {}).get("ui_wrapper", {}).get("last", {}) or {})
    assert last["stage"] == "ui_server_probe_existing"
    assert last["status"] == "started"


def test_ensure_ui_server_running_spawns_when_missing(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    calls = {"wait": 0, "spawn": 0}

    def _wait(url, timeout_seconds=20.0):
        calls["wait"] += 1
        return calls["wait"] >= 2

    monkeypatch.setattr("atlas_ai.main._wait_for_url_ready", _wait)
    monkeypatch.setattr("atlas_ai.main._start_ui_server_process", lambda **kwargs: calls.__setitem__("spawn", calls["spawn"] + 1))

    ok = _ensure_ui_server_running(store, source_mode="ui")

    assert ok is True
    assert calls["spawn"] == 1
    events = store.tail_events(10)
    assert any(event["message"].startswith("Started durable UI server process") for event in events)
    flags = store.load_runtime_flags()
    last = dict(flags.get("ui_process_roles", {}).get("ui_wrapper", {}).get("last", {}) or {})
    assert last["stage"] == "ui_server_spawn"
    assert last["status"] == "started"


def test_record_ui_wrapper_exit_persists_reason(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    _record_ui_wrapper_exit(store, stage="wrapper_exit", status="failed", error="boom", meta={"exit_reason": "wrapper_exception"})

    flags = store.load_runtime_flags()
    last = dict(flags.get("ui_process_roles", {}).get("ui_wrapper", {}).get("last", {}) or {})
    assert last["stage"] == "wrapper_exit"
    assert last["status"] == "failed"
    assert last["meta"]["exit_reason"] == "wrapper_exception"


def test_record_ui_process_lifecycle_persists_per_role_history(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    _record_ui_process_lifecycle(store, process_role="ui_server", stage="startup", status="starting")
    _record_ui_process_lifecycle(store, process_role="ui_wrapper", stage="desktop_shell_attach", status="running")

    flags = store.load_runtime_flags()
    roles = dict(flags.get("ui_process_roles", {}) or {})

    assert roles["ui_server"]["last"]["stage"] == "startup"
    assert roles["ui_wrapper"]["last"]["stage"] == "desktop_shell_attach"
    assert roles["ui_server"]["history"][0]["status"] == "starting"


def test_prime_desktop_wrapper_state_resets_stale_ready_snapshot(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags(
        {
            "ui_desktop_window": {
                "ok": True,
                "ready": True,
                "stage": "started",
                "frontend_mode": "desktop",
                "desktop_window_ready": True,
                "wrapper_pid": 1234,
                "updated_at": "2026-04-20T00:00:00+00:00",
            }
        }
    )

    _prime_desktop_wrapper_state(store, stage="wrapper_starting")

    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert payload["ok"] is False
    assert payload["ready"] is False
    assert payload["stage"] == "wrapper_starting"
    assert payload["frontend_mode"] == "desktop_starting"
    assert int(payload["wrapper_pid"]) > 0


def test_startup_historical_training_skips_when_pool_is_fresh(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "evolved_strategy_pool": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
    )
    settings = SimpleNamespace(enable_historical_training=True, historical_retrain_hours=6)

    should_run, meta = _should_run_startup_historical_training(settings, store)

    assert should_run is False
    assert meta["reason"] == "recent_pool"


def test_startup_historical_training_runs_when_pool_is_stale(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_learning_state(
        {
            "evolved_strategy_pool": {
                "generated_at": (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat(),
            }
        }
    )
    settings = SimpleNamespace(enable_historical_training=True, historical_retrain_hours=6)

    should_run, meta = _should_run_startup_historical_training(settings, store)

    assert should_run is True
    assert meta["reason"] == "stale_pool"


def test_record_self_improvement_runtime_guard_persists_registry_summary(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    append_or_update_proposal(
        store,
        {
            "proposal_id": "p1",
            "status": "awaiting_approval",
            "category": "filters",
            "scope": {"target_blocker": "Consensus guard blocked trade."},
        },
    )

    payload = _record_self_improvement_runtime_guard(store, source_mode="live")
    flags = store.load_runtime_flags()

    assert payload["execution_rule_source"] == "approved_registry_only"
    assert payload["draft_rules_loaded_into_execution"] is False
    assert payload["approval_pending_count"] == 1
    assert payload["pending_proposal_id"] == "p1"
    assert payload["pending_proposal_target_blocker"] == "Consensus guard blocked trade."
    assert flags["self_improvement_runtime"]["active_rule_registry_version"] == "atlas_rules_v1"


def test_reconcile_runtime_flags_accepts_matching_modes(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": False})

    result = reconcile_runtime_flags(store, execute_trades=False, source_mode="live")

    assert result["execute_trades"] is False
    assert result["meta"]["reason"] == "env_runtime_match"
    assert store.tail_events(1)[0]["message"] == "Execution mode reconciled at startup."


def test_reconcile_runtime_flags_blocks_mismatched_modes(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": True})

    try:
        reconcile_runtime_flags(store, execute_trades=False, source_mode="live")
        assert False, "expected ModeReconciliationError"
    except ModeReconciliationError:
        pass

    events = store.tail_events(1)
    assert events[0]["message"].startswith("Startup blocked: execute_trades mismatch")
    status = store.load_status()
    assert status["action"] == "error"
    assert "mode-override" in status["message"]


def test_reconcile_runtime_flags_initializes_missing_runtime_from_env(tmp_path):
    store = KnowledgeStore(str(tmp_path))

    result = reconcile_runtime_flags(store, execute_trades=True, source_mode="live")

    assert result["execute_trades"] is True
    assert result["meta"]["reason"] == "runtime_missing_initialized_from_env"
    assert store.load_runtime_flags()["execute_trades"] is True


def test_reconcile_runtime_flags_explicit_override_wins(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    store.save_runtime_flags({"execute_trades": True})

    result = reconcile_runtime_flags(
        store,
        execute_trades=False,
        source_mode="live",
        mode_override="paper",
    )

    assert result["execute_trades"] is False
    assert result["meta"]["reason"] == "explicit_paper_override"
    assert store.load_runtime_flags()["execute_trades"] is False


def test_guarded_demo_startup_sanity_blocks_paper_runtime(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setenv("ATLAS_WAKE_LAUNCH_PROFILE", "guarded_demo")
    try:
        guarded_demo_startup_sanity(
            store=store,
            execute_trades=False,
            source_mode="live",
            gate_environment="paper",
        )
        assert False, "expected ModeReconciliationError"
    except ModeReconciliationError:
        pass
    status = store.load_status()
    assert status["action"] == "error"
    assert "runtime resolved to paper" in str(status["message"]).lower()
    event = store.tail_events(1)[0]
    assert event["message"] == "STARTUP FAILED: NOT IN GUARDED DEMO MODE"
    flags = store.load_runtime_flags()
    assert flags["startup_verification"]["startup_ok"] is False


def test_guarded_demo_startup_sanity_accepts_live_demo_path(tmp_path, monkeypatch):
    store = KnowledgeStore(str(tmp_path))
    monkeypatch.setenv("ATLAS_WAKE_LAUNCH_PROFILE", "guarded_demo")
    meta = guarded_demo_startup_sanity(
        store=store,
        execute_trades=True,
        source_mode="live",
        gate_environment="demo",
    )
    assert meta["execute_trades"] is True
    assert meta["effective_mode"] == "live"
    assert meta["trader_mode"] == "guarded_demo_live_path"
    assert meta["gate_execution_environment"] == "demo"
    assert meta["startup_ok"] is True
    assert meta["startup_message"] == "ATLAS STARTED IN GUARDED DEMO MODE"
    event = store.tail_events(1)[0]
    assert event["message"] == "ATLAS STARTED IN GUARDED DEMO MODE"
    flags = store.load_runtime_flags()
    assert flags["startup_verification"]["startup_message"] == "ATLAS STARTED IN GUARDED DEMO MODE"
