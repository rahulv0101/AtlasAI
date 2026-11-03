"""Read-only UI health and supervisor recovery; never launch a real process."""
import io
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atlas_ai import ui_server
from atlas_ai import runtime_processes
from atlas_ai.atlas_process_guard import ProcessState
from tools import backend_heartbeat_watchdog as watchdog


@pytest.fixture
def isolated_watchdog(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "_UI_HTTP_FAILURES", 0)
    for name in ("_append_event", "_write_alert", "_write_ui_alert",
                 "_clear_alert_if_recovered", "_clear_ui_alert_if_recovered"):
        monkeypatch.setattr(watchdog, name, Mock())
    for name in ("_start_backend", "_start_ui_server", "stop_process"):
        monkeypatch.setattr(watchdog, name, Mock(side_effect=AssertionError("Real process control forbidden")))
    monkeypatch.setattr(watchdog, "get_ui_health", lambda *a, **k: {"status": "live", "is_fresh": True})
    monkeypatch.setattr(watchdog, "_ui_server_process_state", lambda: ProcessState(name="python.exe", running=True, pid=123))


def test_fresh_heartbeat_does_not_hide_hung_status_endpoint(isolated_watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "probe_ui_http", lambda: {
        "ok": False, "checks": [{"route": "/api/health", "ok": True},
                                 {"route": "/api/status/compact", "ok": False, "error": "timeout"}]})
    restart = Mock(return_value={"started_ui_server": {"started": True}})
    monkeypatch.setattr(watchdog, "_restart_ui_server", restart)
    assert watchdog.check_ui_once()["action"] == "http_retry_pending"
    restart.assert_not_called()
    result = watchdog.check_ui_once()
    assert result["action"] == "restart_ui_server"
    restart.assert_called_once_with("http_unresponsive")
    assert result["http"]["checks"][1]["ok"] is False


def test_http_recovery_resets_failure_count(isolated_watchdog, monkeypatch):
    probe = Mock(side_effect=[{"ok": False}, {"ok": True}, {"ok": False}])
    monkeypatch.setattr(watchdog, "probe_ui_http", probe)
    assert watchdog.check_ui_once()["action"] == "http_retry_pending"
    assert watchdog.check_ui_once()["action"] == "healthy"
    assert watchdog.check_ui_once()["action"] == "http_retry_pending"


def test_no_restart_mode_only_alerts(isolated_watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "probe_ui_http", lambda: {"ok": False})
    watchdog.check_ui_once(restart=False)
    assert watchdog.check_ui_once(restart=False)["action"] == "alert_only"


def test_backend_process_missing_even_with_fresh_heartbeat(isolated_watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "get_backend_health", lambda *_: {"status": "live", "is_fresh": True})
    monkeypatch.setattr(watchdog, "_live_backend_process_state", lambda: ProcessState(name="python.exe", running=False))
    assert watchdog.check_once(restart=False)["action"] == "alert_only"


@pytest.mark.parametrize("service", ["backend", "ui_server"])
def test_failed_stop_never_starts_duplicate(isolated_watchdog, monkeypatch, service):
    monkeypatch.setattr(watchdog, "load_settings", lambda: SimpleNamespace())
    state_name = "_live_backend_process_state" if service == "backend" else "_ui_server_process_state"
    monkeypatch.setattr(watchdog, state_name, lambda: ProcessState(name="python.exe", running=True, pid=123))
    monkeypatch.setattr(watchdog, "stop_process", Mock())
    monkeypatch.setattr(watchdog, "wait_for", lambda *a, **k: (False, None))
    result = getattr(watchdog, "_restart_" + service)("test")
    assert result["started_" + service]["started"] is False


def test_ui_port_must_be_free_before_start(isolated_watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "_ui_server_process_state", lambda: ProcessState(name="python.exe", running=False))
    monkeypatch.setattr(watchdog, "_ui_port_listening", lambda: True)
    assert watchdog._restart_ui_server("test")["started_ui_server"]["error"] == "ui_port_still_in_use"


def test_ui_check_runs_while_backend_check_waits(isolated_watchdog, monkeypatch):
    ui_checked = threading.Event()
    def backend(**kwargs):
        assert ui_checked.wait(3), "UI check was delayed by backend check"
        raise RuntimeError("backend probe failed")
    def ui(**kwargs):
        ui_checked.set()
        return {"action": "healthy"}
    monkeypatch.setattr(watchdog, "check_once", backend)
    monkeypatch.setattr(watchdog, "check_ui_once", ui)
    result = watchdog.check_all_once(restart=False)
    assert result["backend"]["action"] == "watchdog_error"
    assert result["ui_server"]["action"] == "healthy"


def test_once_restart_obeys_supervisor_lock(isolated_watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "_acquire_single_instance_lock", lambda: False)
    check = Mock(side_effect=AssertionError("Duplicate check must not run"))
    monkeypatch.setattr(watchdog, "check_all_once", check)
    assert watchdog.main(["--once"]) == 0


def test_worker_selection_skips_virtualenv_launcher():
    rows = [
        {"ProcessId": 10, "ParentProcessId": 1, "Name": "python.exe", "CommandLine": "venv launcher"},
        {"ProcessId": 20, "ParentProcessId": 10, "Name": "python.exe", "CommandLine": "real worker"},
    ]
    assert watchdog._select_worker(rows).pid == 20


def test_independent_duplicate_workers_require_operator_review():
    rows = [{"ProcessId": 10, "ParentProcessId": 1}, {"ProcessId": 20, "ParentProcessId": 2}]
    with pytest.raises(RuntimeError, match="Expected one worker, found 2"):
        watchdog._select_worker(rows)


@pytest.mark.parametrize("compact", [{"compact": True}, {"unexpected": True}, "timeout"])
def test_http_probe_validates_both_routes_and_uses_timeout(monkeypatch, compact):
    monkeypatch.setattr(watchdog, "_ui_address", lambda: ("127.0.0.1", 8765))
    def open_url(url, *, timeout):
        assert timeout == 0.25
        if url.endswith("/api/health"):
            return io.BytesIO(json.dumps({"ok": True, "service": "atlas-ui"}).encode())
        assert url.endswith("/api/status/compact")
        if compact == "timeout":
            raise TimeoutError("test timeout")
        return io.BytesIO(json.dumps(compact).encode())
    monkeypatch.setattr(watchdog.urllib.request, "build_opener", lambda *_: SimpleNamespace(open=open_url))
    result = watchdog.probe_ui_http(timeout_seconds=0.25)
    assert result["ok"] is (compact == {"compact": True})
    assert len(result["checks"]) == 2


def test_desktop_process_queries_reused_and_expire(monkeypatch):
    monkeypatch.setattr(ui_server, "python_process_exists", lambda *_: None)
    monkeypatch.setattr(ui_server, "atlas_mode_processes", lambda *_: None)
    ui_server._cached_process_exists.cache_clear()
    ui_server._cached_ui_wrapper_pid.cache_clear()
    clock = [100.0]
    monkeypatch.setattr(ui_server.time, "monotonic", lambda: clock[0])
    query = Mock(side_effect=['"python.exe","123"', '[{"ProcessId":456,"CommandLine":"python -m atlas_ai.main ui"}]', '', ''])
    monkeypatch.setattr(ui_server.subprocess, "check_output", query)
    for _ in range(3):
        assert ui_server._process_exists(123)
        assert ui_server._discover_ui_wrapper_pid() == 456
    assert query.call_count == 2
    clock[0] += 6
    assert not ui_server._process_exists(123)
    assert ui_server._discover_ui_wrapper_pid() == 0
    assert query.call_count == 4
    ui_server._cached_process_exists.cache_clear()
    ui_server._cached_ui_wrapper_pid.cache_clear()


def test_native_inventory_matches_exact_modes(monkeypatch):
    class Gone(Exception):
        pass
    def proc(pid, mode):
        return SimpleNamespace(pid=pid, info={"pid": pid, "name": "python.exe"},
                               cmdline=lambda: ["python", "-m", "atlas_ai.main", mode], ppid=lambda: 1)
    monkeypatch.setattr(runtime_processes, "psutil", SimpleNamespace(
        NoSuchProcess=Gone,
        process_iter=lambda *_: [proc(10, "live"), proc(20, "live-ui"), proc(30, "ui-server")]))
    assert [p["ProcessId"] for p in runtime_processes.atlas_mode_processes("live")] == [10]
    assert [p["ProcessId"] for p in runtime_processes.atlas_mode_processes("ui-server")] == [30]


def test_unreadable_native_process_does_not_mean_missing(monkeypatch):
    class Gone(Exception):
        pass
    def denied():
        raise PermissionError("denied")
    proc = SimpleNamespace(info={"name": "python.exe"}, cmdline=denied)
    monkeypatch.setattr(runtime_processes, "psutil", SimpleNamespace(
        NoSuchProcess=Gone, process_iter=lambda *_: [proc]))
    with pytest.raises(PermissionError):
        runtime_processes.atlas_mode_processes("live")


def test_native_ui_inventory_never_spawns_shell(monkeypatch):
    ui_server._cached_ui_wrapper_pid.cache_clear()
    ui_server._cached_process_exists.cache_clear()
    monkeypatch.setattr(ui_server, "atlas_mode_processes", lambda *_: [
        {"ProcessId": 10, "ParentProcessId": 1}, {"ProcessId": 20, "ParentProcessId": 10}])
    monkeypatch.setattr(ui_server, "python_process_exists", lambda pid: pid == 20)
    monkeypatch.setattr(ui_server.subprocess, "check_output", Mock(side_effect=AssertionError("No shell needed")))
    assert ui_server._discover_ui_wrapper_pid() == 20
    assert ui_server._process_exists(20)
    ui_server._cached_ui_wrapper_pid.cache_clear()
    ui_server._cached_process_exists.cache_clear()
