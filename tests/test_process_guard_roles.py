from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.atlas_process_guard import (
    backend_process_state,
    classify_atlas_process_role,
    frontend_api_status,
    frontend_process_state,
    is_frontend_healthy,
)
from atlas_ai.atlas_wake_config import AtlasWakeConfig


def _cfg(tmp_path, launch_profile: str = "guarded_demo") -> AtlasWakeConfig:
    return AtlasWakeConfig(
        project_root=str(tmp_path),
        python_executable="python",
        mt5_executable_path="terminal64.exe",
        launch_profile=launch_profile,
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


def test_classify_atlas_process_role_distinguishes_live_vs_ui():
    ui_cmd = '"C:\\x\\.venv\\Scripts\\python.exe" -m atlas_ai.main ui --mode-override live'
    ui_server_cmd = '"C:\\x\\.venv\\Scripts\\python.exe" -m atlas_ai.main ui-server --mode-override live'
    live_cmd = '"C:\\x\\.venv\\Scripts\\python.exe" -m atlas_ai.main live --poll-seconds 60 --mode-override live'
    assert classify_atlas_process_role(ui_cmd)["role"] == "ui_desktop_wrapper"
    assert classify_atlas_process_role(ui_server_cmd)["role"] == "ui_server"
    assert classify_atlas_process_role(live_cmd)["role"] == "backend_trader"


def test_backend_process_state_never_treats_ui_process_as_backend(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")

    def _fake_processes():
        return [
            {"ProcessId": 100, "Name": "python.exe", "CommandLine": '"python" -m atlas_ai.main ui --mode-override live'},
            {"ProcessId": 200, "Name": "python.exe", "CommandLine": '"python" -m atlas_ai.main live --poll-seconds 60 --mode-override live'},
        ]

    monkeypatch.setattr("atlas_ai.atlas_process_guard._processes", _fake_processes)
    state = backend_process_state(cfg)
    assert state.running is True
    assert state.pid == 200
    assert "role=backend_trader" in str(state.detail)


def test_pythonw_processes_are_recognized_for_backend_and_frontend(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")

    def _fake_processes():
        return [
            {"ProcessId": 300, "Name": "pythonw.exe", "CommandLine": '"pythonw" -m atlas_ai.main ui-server --mode-override live'},
            {"ProcessId": 400, "Name": "pythonw.exe", "CommandLine": '"pythonw" -m atlas_ai.main ui --mode-override live'},
            {"ProcessId": 500, "Name": "pythonw.exe", "CommandLine": '"pythonw" -m atlas_ai.main live --poll-seconds 60 --mode-override live'},
        ]

    monkeypatch.setattr("atlas_ai.atlas_process_guard._processes", _fake_processes)

    backend = backend_process_state(cfg)
    frontend = frontend_process_state()

    assert backend.running is True
    assert backend.pid == 500
    assert "role=backend_trader" in str(backend.detail)
    assert frontend.running is True
    assert frontend.pid == 300
    assert "role=ui_server" in str(frontend.detail)


def test_frontend_health_allows_fallback_mode_with_explicit_reason(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_api_status",
        lambda config: {
            "comparison_active": False,
            "execute_trades": True,
            "ui_desktop_window_ready": False,
            "ui_desktop_window": {
                "fallback_active": True,
                "frontend_mode": "browser_fallback",
                "failure_reason": "desktop_host_accessibility_recursion",
            },
        },
    )
    monkeypatch.setattr("atlas_ai.atlas_process_guard.frontend_process_state", lambda: SimpleNamespace(running=True))
    ok, detail, _ = is_frontend_healthy(cfg)
    assert ok is True
    assert "fallback_active=True" in detail
    assert "failure_reason=desktop_host_accessibility_recursion" in detail


def test_frontend_health_rejects_stale_desktop_ready_snapshot_without_live_wrapper(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_api_status",
        lambda config: {
            "comparison_active": False,
            "execute_trades": True,
            "ui_desktop_window_ready": True,
            "ui_desktop_window": {
                "desktop_window_ready": True,
                "frontend_mode": "desktop",
                "updated_at": "2026-04-20T00:00:00+00:00",
                "wrapper_pid": 99999,
            },
        },
    )
    monkeypatch.setattr("atlas_ai.atlas_process_guard.frontend_process_state", lambda: SimpleNamespace(running=True))
    monkeypatch.setattr("atlas_ai.atlas_process_guard.process_exists", lambda pid: False)
    ok, detail, _ = is_frontend_healthy(cfg)
    assert ok is False
    assert "desktop_state_fresh=False" in detail
    assert "wrapper_running=False" in detail


def test_frontend_api_status_prefers_compact_health_probe(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")
    calls: list[str] = []

    def _fetch(url: str, timeout: float = 2.0):
        calls.append(url)
        if url.endswith("/api/status/compact"):
            return {
                "compact": True,
                "action": "connected",
                "ui_desktop_window_ready": True,
                "ui_desktop_window": {"ready": True},
            }
        return None

    monkeypatch.setattr("atlas_ai.atlas_process_guard.fetch_json", _fetch)

    payload = frontend_api_status(cfg)

    assert payload is not None
    assert payload["_health_probe_source"] == "compact"
    assert payload["comparison_active"] is False
    assert calls == ["http://127.0.0.1:8765/api/status/compact"]


def test_guarded_demo_compact_probe_does_not_require_heavy_status_execute_flag(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, launch_profile="guarded_demo")
    monkeypatch.setattr(
        "atlas_ai.atlas_process_guard.frontend_api_status",
        lambda config: {
            "_health_probe_source": "compact",
            "comparison_active": False,
            "ui_desktop_window_ready": False,
            "ui_desktop_window": {
                "fallback_active": True,
                "frontend_mode": "browser_fallback",
                "failure_reason": "desktop_startup_slow",
            },
        },
    )
    monkeypatch.setattr("atlas_ai.atlas_process_guard.frontend_process_state", lambda: SimpleNamespace(running=True))

    ok, detail, _ = is_frontend_healthy(cfg)

    assert ok is True
    assert "api_ok=True" in detail
    assert "fallback_active=True" in detail
