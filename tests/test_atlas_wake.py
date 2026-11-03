from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.atlas_full_launcher import run_full_launch, verify_health
from atlas_ai.atlas_full_shutdown import verify_data_integrity
from atlas_ai.atlas_audio_listener import _contains_phrase, _should_trigger_phrase, ClapState
from atlas_ai.atlas_process_guard import is_frontend_healthy
from atlas_ai.atlas_wake_config import AtlasWakeConfig, load_atlas_wake_config
from atlas_ai.atlas_wake_listener import handle_trigger, is_wake_trigger, normalize_phrase


def _config() -> AtlasWakeConfig:
    return AtlasWakeConfig(
        project_root="C:\\Atlas",
        python_executable="C:\\Atlas\\.venv\\Scripts\\python.exe",
        mt5_executable_path="C:\\Program Files\\HFM Metatrader 5\\terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "paper-compare"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root="C:\\Atlas\\storage_compare_runs",
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path="C:\\Atlas\\models\\vosk-model-small-en-us-0.15",
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )


def test_normalize_phrase_handles_case_and_spacing():
    assert normalize_phrase("  Atlas   Wake!! ") == "atlas wake"


def test_is_wake_trigger_matches_expected_phrase():
    assert is_wake_trigger("atlas wake", "Atlas wake") is True
    assert is_wake_trigger("Atlas, wake!", "Atlas wake") is True
    assert is_wake_trigger("atlas awake", "Atlas wake") is False


def test_handle_trigger_uses_async_dispatch(monkeypatch):
    called = {}

    def _fake_async(trigger_text: str):
        called["trigger"] = trigger_text
        return {"matched": True, "pending": True}

    monkeypatch.setattr("atlas_ai.atlas_wake_listener.handle_trigger_async", _fake_async)

    payload = handle_trigger("Atlas wake")

    assert payload["pending"] is True
    assert called["trigger"] == "Atlas wake"


def test_verify_health_reports_full_system_running(monkeypatch):
    config = _config()
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.is_mt5_ready", lambda cfg: (True, "mt5_ready"))
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_backend_healthy",
        lambda cfg: (True, "backend_ok", {"status": {"execute_trades": True, "system_health": {"paper_mode_confirmed": False}}}),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_frontend_healthy",
        lambda cfg: (
            True,
            "frontend_ok",
            {"comparison_active": False, "execute_trades": True, "system_health": {"paper_mode_confirmed": False}},
        ),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.fetch_json",
        lambda url, timeout=2.0: {"comparison_active": False, "execute_trades": True, "system_health": {"paper_mode_confirmed": False}},
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.mt5_process_state", lambda cfg: type("S", (), {"running": True})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.backend_process_state", lambda cfg=None: type("S", (), {"running": True})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.frontend_process_state", lambda: type("S", (), {"running": True})())

    summary = verify_health(config)

    assert summary["overall_result"] == "SUCCESS"
    assert summary["api_status_health"]["comparison_active"] is False
    assert summary["launch_profile"] == "guarded_demo"


def test_verify_health_reports_partial_start(monkeypatch):
    config = _config()
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.is_mt5_ready", lambda cfg: (True, "mt5_ready"))
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_backend_healthy",
        lambda cfg: (False, "backend_down", {"status": {}}),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_frontend_healthy",
        lambda cfg: (False, "frontend_down", {}),
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.fetch_json", lambda url, timeout=2.0: {})
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.mt5_process_state", lambda cfg: type("S", (), {"running": True})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.backend_process_state", lambda cfg=None: type("S", (), {"running": False})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.frontend_process_state", lambda: type("S", (), {"running": False})())

    summary = verify_health(config)

    assert summary["overall_result"] == "FAILED"


def test_verify_health_reports_degraded_when_backend_healthy_and_http_ui_alive(monkeypatch):
    config = _config()
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.is_mt5_ready", lambda cfg: (True, "mt5_ready"))
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_backend_healthy",
        lambda cfg: (True, "backend_ok", {"status": {"execute_trades": True, "system_health": {"paper_mode_confirmed": False}}}),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_frontend_healthy",
        lambda cfg: (
            False,
            "desktop_down",
            {
                "comparison_active": False,
                "execute_trades": True,
                "ui_desktop_window_ready": False,
                "ui_desktop_window": {
                    "fallback_active": True,
                    "frontend_mode": "browser_fallback",
                    "failure_reason": "desktop_host_accessibility_recursion",
                    "degraded_since": "2026-04-22T16:31:05+00:00",
                },
                "system_health": {"paper_mode_confirmed": False},
            },
        ),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.fetch_json",
        lambda url, timeout=2.0: {
            "comparison_active": False,
            "execute_trades": True,
            "ui_desktop_window": {
                "fallback_active": True,
                "frontend_mode": "browser_fallback",
                "failure_reason": "desktop_host_accessibility_recursion",
                "degraded_since": "2026-04-22T16:31:05+00:00",
            },
        },
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.mt5_process_state", lambda cfg: type("S", (), {"running": True})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.backend_process_state", lambda cfg=None: type("S", (), {"running": True})())
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.frontend_process_state", lambda: type("S", (), {"running": True})())

    summary = verify_health(config)

    assert summary["overall_result"] == "DEGRADED"
    assert summary["overall_reason"] == "desktop_host_accessibility_recursion"
    assert summary["desktop_shell_health"]["fallback_active"] is True


def test_guarded_demo_wake_config_defaults_do_not_use_paper_compare(monkeypatch, tmp_path):
    monkeypatch.setattr("atlas_ai.atlas_wake_config._project_root", lambda: tmp_path)
    monkeypatch.setattr("atlas_ai.atlas_wake_config._load_json_config", lambda project_root: {})
    monkeypatch.setattr("atlas_ai.atlas_wake_config._default_python", lambda project_root: tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(
        "atlas_ai.atlas_wake_config.load_settings",
        lambda: type(
            "S",
            (),
            {
                "hfm_terminal_path": "terminal64.exe",
            },
        )(),
    )

    config = load_atlas_wake_config()

    assert config.launch_profile == "guarded_demo"
    assert "paper-compare" not in " ".join(config.backend_launch_command)
    assert config.backend_launch_command[3] == "live"
    assert "--mode-override" in config.backend_launch_command
    assert "live" in config.backend_launch_command
    assert "--comparison-root" not in config.frontend_launch_command
    assert "--comparison-run-id" not in config.frontend_launch_command


def test_paper_compare_launch_profile_stays_paper_only(monkeypatch, tmp_path):
    monkeypatch.setattr("atlas_ai.atlas_wake_config._project_root", lambda: tmp_path)
    monkeypatch.setattr("atlas_ai.atlas_wake_config._load_json_config", lambda project_root: {"launch_profile": "paper_compare"})
    monkeypatch.setattr("atlas_ai.atlas_wake_config._default_python", lambda project_root: tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(
        "atlas_ai.atlas_wake_config.load_settings",
        lambda: type(
            "S",
            (),
            {
                "hfm_terminal_path": "terminal64.exe",
            },
        )(),
    )

    config = load_atlas_wake_config()

    assert config.launch_profile == "paper_compare"
    assert "paper-compare" in " ".join(config.backend_launch_command)
    assert "--comparison-root" in config.frontend_launch_command
    assert "--comparison-run-id" in config.frontend_launch_command


def test_guarded_demo_rejects_forbidden_backend_command(monkeypatch, tmp_path):
    monkeypatch.setattr("atlas_ai.atlas_wake_config._project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "atlas_ai.atlas_wake_config._load_json_config",
        lambda project_root: {
            "launch_profile": "guarded_demo",
            "backend_launch_command": ["python", "-m", "atlas_ai.main", "paper-compare"],
        },
    )
    monkeypatch.setattr("atlas_ai.atlas_wake_config._default_python", lambda project_root: tmp_path / ".venv" / "Scripts" / "python.exe")
    monkeypatch.setattr(
        "atlas_ai.atlas_wake_config.load_settings",
        lambda: type(
            "S",
            (),
            {
                "hfm_terminal_path": "terminal64.exe",
            },
        )(),
    )
    try:
        load_atlas_wake_config()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "forbidden paper routing token" in str(exc).lower()


def test_run_full_launch_fails_fast_when_guarded_demo_command_is_paper(tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "paper-compare"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    summary = run_full_launch(cfg)
    assert summary["health"]["overall_result"] == "FAILED"
    assert "forbidden token" in str(summary["health"]["failure_reason"]).lower()


def test_run_full_launch_restarts_after_failed_start(monkeypatch, tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    call_count = {"n": 0}

    def _fake_cycle(config, attempt_index=1, max_attempts=1):
        call_count["n"] += 1
        result = "FAILED" if call_count["n"] == 1 else "SUCCESS"
        return {
            "config": config.to_safe_dict(),
            "startup_sequence": [],
            "health": {"overall_result": result, "overall_reason": "test"},
        }

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._run_launch_cycle", _fake_cycle)
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.run_full_shutdown",
        lambda config, **kwargs: {"overall_result": "FULL SYSTEM STOPPED CLEANLY"},
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.setup_logging", lambda config: None)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.time.sleep", lambda seconds: None)

    summary = run_full_launch(cfg)

    assert summary["health"]["overall_result"] == "SUCCESS"
    assert len(summary["attempts"]) == 2
    assert len(summary["recovery_actions"]) == 1
    assert summary["recovery_actions"][0]["shutdown"]["overall_result"] == "FULL SYSTEM STOPPED CLEANLY"


def test_launch_cycle_skips_frontend_if_backend_unhealthy(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _run_launch_cycle

    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._validate_guarded_demo_commands", lambda config: SimpleNamespace(status="ok", detail="ok", already_running=False, pid=None))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._start_mt5_if_needed", lambda config: SimpleNamespace(status="started", detail="ok", already_running=False, pid=None))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._wait_for_mt5", lambda config: SimpleNamespace(status="healthy", detail="ok", already_running=True, pid=1))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._start_backend_if_needed", lambda config: SimpleNamespace(status="started", detail="ok", already_running=False, pid=2))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._wait_for_backend", lambda config: SimpleNamespace(status="failed", detail="backend_down", already_running=False, pid=2))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.verify_health", lambda config: {"overall_result": "FAILED"})
    called = {"frontend_started": False}

    def _frontend_start(config):
        called["frontend_started"] = True
        return SimpleNamespace(status="started", detail="ok", already_running=False, pid=3)

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._start_frontend_if_needed", _frontend_start)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._wait_for_frontend", lambda config: SimpleNamespace(status="healthy", detail="ok", already_running=True, pid=3))

    summary = _run_launch_cycle(cfg, attempt_index=1, max_attempts=2)

    assert called["frontend_started"] is False
    steps = summary["startup_sequence"]
    frontend_step = [s for s in steps if s["step"] == "frontend_launch_check"][0]["result"]
    assert frontend_step["status"] == "skipped"


def test_guarded_demo_backend_starts_when_only_stale_status_exists(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _start_backend_if_needed

    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_backend_healthy",
        lambda config: (True, "process_running=False, runtime_execute=True, fresh=True, age_seconds=167.5", {}),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.backend_process_state",
        lambda config=None: SimpleNamespace(running=False, pid=None),
    )
    called = {"n": 0}

    def _spawn(command, cwd, log_label=""):
        called["n"] += 1
        return SimpleNamespace(poll=lambda: None)

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._spawn", _spawn)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.time.sleep", lambda *_args, **_kwargs: None)

    result = _start_backend_if_needed(cfg)

    assert result.status == "started"
    assert called["n"] == 1


def test_frontend_force_browser_only_when_repeated_failures_disabled(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _start_frontend_if_needed

    project_root = tmp_path
    storage_dir = project_root / "storage"
    storage_dir.mkdir(parents=True)
    (storage_dir / "runtime_flags.json").write_text(
        '{"ui_desktop_window":{"disabled_due_to_repeated_failures":true}}',
        encoding="utf-8",
    )
    cfg = AtlasWakeConfig(
        project_root=str(project_root),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.is_frontend_healthy",
        lambda config: (False, "frontend_down", {}),
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher.frontend_process_state",
        lambda: SimpleNamespace(running=False, pid=None),
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.time.sleep", lambda *_args, **_kwargs: None)
    captured: list[dict] = []

    def _spawn(command, cwd, log_label="", extra_env=None):
        captured.append(
            {
                "command": list(command),
                "cwd": cwd,
                "log_label": log_label,
                "extra_env": dict(extra_env or {}),
            }
        )
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._spawn", _spawn)

    result = _start_frontend_if_needed(cfg)

    assert result.status == "started"
    assert captured[0]["log_label"] == "ui_server"
    assert captured[0]["command"][3] == "ui-server"
    assert captured[1]["log_label"] == "frontend"
    assert captured[1]["extra_env"]["ATLAS_UI_FORCE_BROWSER_ONLY"] == "1"
    assert captured[1]["command"][3] == "ui-browser"


def test_frontend_prefers_desktop_mode_when_no_known_instability(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _start_frontend_if_needed

    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.is_frontend_healthy", lambda config: (False, "frontend_down", {}))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.frontend_process_state", lambda: SimpleNamespace(running=False, pid=None))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher._frontend_should_force_browser_only", lambda config: (False, ""))
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.time.sleep", lambda *_args, **_kwargs: None)
    captured: list[dict] = []

    def _spawn(command, cwd, log_label="", extra_env=None):
        captured.append({"command": list(command), "extra_env": dict(extra_env or {}), "log_label": log_label})
        return SimpleNamespace(pid=123)

    monkeypatch.setattr("atlas_ai.atlas_full_launcher._spawn", _spawn)

    result = _start_frontend_if_needed(cfg)

    assert result.status == "started"
    assert captured[0]["command"][3] == "ui-server"
    assert captured[1]["command"][3] == "ui"
    assert captured[1]["extra_env"] == {}


def test_spawn_avoids_detached_flag_for_ui_processes(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _spawn

    captured: dict = {}

    class _Proc:
        def __init__(self):
            self.pid = 123

    def _fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return _Proc()

    monkeypatch.setattr("atlas_ai.atlas_full_launcher.os.name", "nt")
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.DETACHED_PROCESS", 8, raising=False)

    _spawn(["python", "-m", "atlas_ai.main", "ui-server"], cwd=str(tmp_path))

    assert captured["kwargs"]["creationflags"] == 512


def test_spawn_keeps_detached_flag_for_backend_processes(monkeypatch, tmp_path):
    from atlas_ai.atlas_full_launcher import _spawn

    captured: dict = {}

    class _Proc:
        def __init__(self):
            self.pid = 123

    def _fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = dict(kwargs)
        return _Proc()

    monkeypatch.setattr("atlas_ai.atlas_full_launcher.os.name", "nt")
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.Popen", _fake_popen)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.subprocess.DETACHED_PROCESS", 8, raising=False)

    _spawn(["python", "-m", "atlas_ai.main", "live"], cwd=str(tmp_path))

    assert captured["kwargs"]["creationflags"] == 520


def test_run_full_launch_final_cleanup_runs_on_persistent_failure(monkeypatch, tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher._run_launch_cycle",
        lambda config, attempt_index=1, max_attempts=1: {
            "config": config.to_safe_dict(),
            "startup_sequence": [],
            "health": {"overall_result": "FAILED", "overall_reason": "no_usable_control_surface"},
        },
    )
    calls = {"n": 0}

    def _shutdown(_config, *, keep_mt5: bool = False):
        calls["n"] += 1
        return {"overall_result": "FULL SYSTEM STOPPED CLEANLY", "listener_preserved": False, "mt5_hfm_stopped": {"detail": "mt5 preserved" if keep_mt5 else "mt5 stopped"}}

    monkeypatch.setattr("atlas_ai.atlas_full_launcher.run_full_shutdown", _shutdown)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.setup_logging", lambda config: None)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.time.sleep", lambda seconds: None)

    summary = run_full_launch(cfg)

    assert summary["health"]["overall_result"] == "FAILED"
    assert calls["n"] == 2
    assert len(summary["recovery_actions"]) == 2
    assert summary["recovery_actions"][-1]["reason"] == "final_launch_unhealthy_cleanup"
    assert summary["recovery_actions"][-1]["preserved_mt5"] is True


def test_run_full_launch_keeps_backend_alive_on_degraded_frontend(monkeypatch, tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_full_launcher._run_launch_cycle",
        lambda config, attempt_index=1, max_attempts=1: {
            "config": config.to_safe_dict(),
            "startup_sequence": [],
            "health": {
                "overall_result": "DEGRADED",
                "overall_reason": "desktop_host_accessibility_recursion",
            },
        },
    )
    calls = {"n": 0}

    def _shutdown(_config):
        calls["n"] += 1
        return {"overall_result": "FULL SYSTEM STOPPED CLEANLY"}

    monkeypatch.setattr("atlas_ai.atlas_full_launcher.run_full_shutdown", _shutdown)
    monkeypatch.setattr("atlas_ai.atlas_full_launcher.setup_logging", lambda config: None)

    summary = run_full_launch(cfg)

    assert summary["health"]["overall_result"] == "DEGRADED"
    assert calls["n"] == 0


def test_guarded_demo_frontend_health_accepts_browser_fallback(monkeypatch, tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_api_status",
        lambda config: {
            "comparison_active": False,
            "execute_trades": True,
            "ui_desktop_window_ready": False,
            "ui_desktop_window": {
                "fallback_active": True,
                "disabled_due_to_repeated_failures": True,
            },
        },
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_process_state",
        lambda: SimpleNamespace(running=True),
    )

    ok, detail, _ = is_frontend_healthy(cfg)
    assert ok is True
    assert "desktop_window_ready=False" in detail
    assert "fallback_active=True" in detail


def test_guarded_demo_frontend_health_fails_when_no_desktop_and_no_fallback(monkeypatch, tmp_path):
    cfg = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=["python", "-m", "atlas_ai.main", "live"],
        frontend_launch_command=["python", "-m", "atlas_ai.main", "ui"],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_api_status",
        lambda config: {
            "comparison_active": False,
            "execute_trades": True,
            "ui_desktop_window_ready": False,
            "ui_desktop_window": {"fallback_active": False},
        },
    )
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_process_state",
        lambda: SimpleNamespace(running=True),
    )

    ok, detail, _ = is_frontend_healthy(cfg)
    assert ok is False
    assert "desktop_window_ready=False" in detail
    assert "fallback_active=False" in detail

def test_verify_data_integrity_reads_required_json_files(tmp_path):
    run_root = tmp_path / "storage_compare_runs" / "atlas_wake_latest"
    run_root.mkdir(parents=True)
    for name in ["comparison_summary.json", "comparison_report.json", "comparison_tracker.json"]:
        (run_root / name).write_text("{}", encoding="utf-8")
    config = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="paper_compare",
        backend_launch_command=[],
        frontend_launch_command=[],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )

    result = verify_data_integrity(config)

    assert result["intact"] is True
    assert result["files"]["comparison_summary"]["readable"] is True


def test_handle_trigger_routes_sleep(monkeypatch):
    config = _config()
    monkeypatch.setattr("atlas_ai.atlas_wake_listener.load_atlas_wake_config", lambda: config)
    monkeypatch.setattr(
        "atlas_ai.atlas_wake_listener.close_system",
        lambda **kwargs: {"ok": True, "summary": {"overall_result": "FULL SYSTEM STOPPED CLEANLY"}},
    )

    result = handle_trigger("Atlas sleep")

    assert result["matched"] is True
    assert result["action"] == "sleep"
    assert result["shutdown_summary"]["overall_result"] == "FULL SYSTEM STOPPED CLEANLY"


def test_audio_phrase_detection_supports_embedded_phrase():
    assert _contains_phrase("please atlas wake now", "Atlas wake") is True


def test_audio_trigger_allows_sleep_without_clap_gate():
    config = _config()
    config = AtlasWakeConfig(**{**config.to_safe_dict(), "audio_clap_enabled": True})
    phrase = _should_trigger_phrase("atlas sleep", config, ClapState(clap_times=[], armed_until=0.0))
    assert phrase == "Atlas sleep"


def test_verify_data_integrity_guarded_demo_not_bound_to_comparison_files(tmp_path):
    config = AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile="guarded_demo",
        backend_launch_command=[],
        frontend_launch_command=[],
        comparison_root=str(tmp_path / "storage_compare_runs"),
        comparison_run_id="atlas_wake_latest",
        paper_preset="conservative_filter_rollout_v1",
        poll_seconds=60,
        ui_host="127.0.0.1",
        ui_port=8765,
        backend_ready_timeout_seconds=180,
        frontend_ready_timeout_seconds=90,
        mt5_ready_timeout_seconds=90,
        health_stale_after_seconds=180,
        audio_model_path=str(tmp_path / "model"),
        audio_sample_rate=16000,
        audio_block_size=8000,
        audio_device_name="",
        audio_clap_enabled=False,
        audio_double_clap_window_seconds=1.5,
        audio_clap_arm_seconds=8.0,
        audio_clap_peak_threshold=12000.0,
        sleep_phrase="Atlas sleep",
    )

    result = verify_data_integrity(config)

    assert result["intact"] is True
    assert result["files"] == {}
