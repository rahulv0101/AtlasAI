from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atlas_ai.atlas_process_guard import ProcessState, backend_process_state, process_exists, stop_process, wait_for
from atlas_ai.config.settings import load_settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.runtime_heartbeat import get_backend_health


ALERT_PATH = PROJECT_ROOT / "storage" / "backend_staleness_alert.json"
UI_ALERT_PATH = PROJECT_ROOT / "storage" / "ui_staleness_alert.json"
EVENTS_PATH = PROJECT_ROOT / "logs" / "backend_heartbeat_watchdog.jsonl"
LOCK_PATH = PROJECT_ROOT / "storage" / "backend_heartbeat_watchdog.lock"
_LOCK_HANDLE: Any | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _preferred_python() -> str:
    pinned = str(os.getenv("ATLAS_PINNED_PYTHON", "") or "").strip()
    if pinned and Path(pinned).exists():
        return pinned
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _creationflags() -> int:
    if os.name != "nt":
        return 0
    flags = 0
    for name in ("CREATE_NO_WINDOW", "CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS"):
        flags |= int(getattr(subprocess, name, 0) or 0)
    return flags


def _live_backend_process_state() -> ProcessState:
    if os.name != "nt":
        return backend_process_state()
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and "
        "$_.CommandLine -match '-m\\s+atlas_ai\\.main\\s+live' } | "
        "Select-Object -First 1 ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0),
        ).strip()
    except Exception:
        return backend_process_state()
    if not out:
        return ProcessState(name="python.exe", running=False)
    try:
        data = json.loads(out)
    except Exception:
        return backend_process_state()
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        return backend_process_state()
    try:
        pid = int(row.get("ProcessId") or 0)
    except Exception:
        pid = 0
    return ProcessState(
        name=str(row.get("Name", "python.exe") or "python.exe"),
        running=pid > 0,
        pid=pid or None,
        detail=str(row.get("CommandLine", "") or ""),
    )


def _ui_server_process_state() -> ProcessState:
    if os.name != "nt":
        return ProcessState(name="python.exe", running=False)
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') -and "
        "$_.CommandLine -match '-m\\s+atlas_ai\\.main\\s+ui-server' } | "
        "Select-Object -First 1 ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0),
        ).strip()
    except Exception:
        return ProcessState(name="python.exe", running=False)
    if not out:
        return ProcessState(name="python.exe", running=False)
    try:
        data = json.loads(out)
    except Exception:
        return ProcessState(name="python.exe", running=False)
    row = data[0] if isinstance(data, list) and data else data
    if not isinstance(row, dict):
        return ProcessState(name="python.exe", running=False)
    try:
        pid = int(row.get("ProcessId") or 0)
    except Exception:
        pid = 0
    return ProcessState(
        name=str(row.get("Name", "python.exe") or "python.exe"),
        running=pid > 0,
        pid=pid or None,
        detail=str(row.get("CommandLine", "") or ""),
    )


def _append_event(payload: dict[str, Any]) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _acquire_single_instance_lock() -> bool:
    global _LOCK_HANDLE
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK_PATH.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            fh.seek(0)
            with contextlib.suppress(Exception):
                fh.truncate(1)
            try:
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                fh.close()
                return False
        else:
            import fcntl

            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                return False
    except Exception:
        fh.close()
        return False
    fh.seek(0)
    fh.truncate(0)
    fh.write(str(os.getpid()))
    fh.flush()
    _LOCK_HANDLE = fh
    return True


def _write_alert(payload: dict[str, Any]) -> None:
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALERT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_ui_alert(payload: dict[str, Any]) -> None:
    UI_ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    UI_ALERT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _clear_alert_if_recovered(health: dict[str, Any]) -> None:
    if not ALERT_PATH.exists():
        return
    payload = {
        "schema": "atlas_backend_staleness_alert_v1",
        "status": "recovered",
        "updated_at": _now(),
        "backend_health": health,
        "message": "Backend heartbeat is fresh again.",
    }
    _write_alert(payload)


def _clear_ui_alert_if_recovered(health: dict[str, Any]) -> None:
    if not UI_ALERT_PATH.exists():
        return
    payload = {
        "schema": "atlas_ui_staleness_alert_v1",
        "status": "recovered",
        "updated_at": _now(),
        "ui_health": health,
        "message": "UI server heartbeat is fresh again.",
    }
    _write_ui_alert(payload)


def _start_backend() -> dict[str, Any]:
    python_exe = _preferred_python()
    command = [
        python_exe,
        "-m",
        "atlas_ai.main",
        "live",
        "--poll-seconds",
        "60",
        "--mode-override",
        "live",
    ]
    env = os.environ.copy()
    env["ATLAS_PINNED_PYTHON"] = python_exe
    kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    flags = _creationflags()
    if flags:
        kwargs["creationflags"] = flags
    proc = subprocess.Popen(command, **kwargs)
    return {"started": True, "pid": int(proc.pid), "command": command}


def _start_ui_server() -> dict[str, Any]:
    python_exe = _preferred_python()
    command = [
        python_exe,
        "-m",
        "atlas_ai.main",
        "ui-server",
    ]
    env = os.environ.copy()
    env["ATLAS_PINNED_PYTHON"] = python_exe
    kwargs: dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "env": env,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    flags = _creationflags()
    if flags:
        kwargs["creationflags"] = flags
    proc = subprocess.Popen(command, **kwargs)
    return {"started": True, "pid": int(proc.pid), "command": command}


def _restart_backend(reason: str) -> dict[str, Any]:
    settings = load_settings()
    state = _live_backend_process_state()
    stopped: dict[str, Any] = {"attempted": False, "was_running": bool(state.running), "pid": state.pid}
    if state.running and state.pid:
        stop_process(int(state.pid), force=False)
        graceful, _ = wait_for(lambda: not process_exists(int(state.pid)), timeout_seconds=20, poll_seconds=1.0)
        forced = False
        if not graceful:
            stop_process(int(state.pid), force=True)
            forced = True
            graceful, _ = wait_for(lambda: not process_exists(int(state.pid)), timeout_seconds=10, poll_seconds=0.5)
        stopped = {
            "attempted": True,
            "was_running": True,
            "pid": int(state.pid),
            "stopped": bool(graceful),
            "forced": bool(forced),
        }
    start = _start_backend()
    return {
        "reason": reason,
        "settings": {
            "timeframe": settings.timeframe,
            "execute_trades": bool(settings.execute_trades),
            "enabled_strategies": list(settings.enabled_strategies),
        },
        "stopped_existing_backend": stopped,
        "started_backend": start,
    }


def _restart_ui_server(reason: str) -> dict[str, Any]:
    state = _ui_server_process_state()
    stopped: dict[str, Any] = {"attempted": False, "was_running": bool(state.running), "pid": state.pid}
    if state.running and state.pid:
        stop_process(int(state.pid), force=False)
        graceful, _ = wait_for(lambda: not process_exists(int(state.pid)), timeout_seconds=15, poll_seconds=1.0)
        forced = False
        if not graceful:
            stop_process(int(state.pid), force=True)
            forced = True
            graceful, _ = wait_for(lambda: not process_exists(int(state.pid)), timeout_seconds=8, poll_seconds=0.5)
        stopped = {
            "attempted": True,
            "was_running": True,
            "pid": int(state.pid),
            "stopped": bool(graceful),
            "forced": bool(forced),
        }
    start = _start_ui_server()
    return {
        "reason": reason,
        "stopped_existing_ui_server": stopped,
        "started_ui_server": start,
    }


def get_ui_health(store: KnowledgeStore | None = None, *, threshold_seconds: int = 180, now_ts: float | None = None) -> dict[str, Any]:
    active_store = store if store is not None else KnowledgeStore(str(PROJECT_ROOT / "storage"))
    heartbeat = active_store.load_ui_heartbeat()
    threshold = max(1, int(threshold_seconds))
    if not heartbeat:
        return {
            "is_fresh": False,
            "last_heartbeat": "",
            "seconds_since_heartbeat": 0,
            "staleness_threshold_seconds": threshold,
            "cycle_count": 0,
            "status": "unknown",
        }
    last_ts = float(heartbeat.get("last_heartbeat_ts", 0.0) or 0.0)
    if last_ts <= 0:
        return {
            "is_fresh": False,
            "last_heartbeat": str(heartbeat.get("last_heartbeat", "") or ""),
            "seconds_since_heartbeat": 0,
            "staleness_threshold_seconds": threshold,
            "cycle_count": int(heartbeat.get("cycle_count", 0) or 0),
            "status": "unknown",
        }
    now = float(now_ts if now_ts is not None else time.time())
    seconds_since = max(0, int(now - last_ts))
    is_fresh = seconds_since <= threshold
    return {
        "is_fresh": is_fresh,
        "last_heartbeat": str(heartbeat.get("last_heartbeat", "") or ""),
        "seconds_since_heartbeat": seconds_since,
        "staleness_threshold_seconds": threshold,
        "cycle_count": int(heartbeat.get("cycle_count", 0) or 0),
        "status": "live" if is_fresh else "stale",
        "pid": int(heartbeat.get("pid", 0) or 0),
        "bind_host": str(heartbeat.get("bind_host", "") or ""),
        "bind_port": int(heartbeat.get("bind_port", 0) or 0),
    }


def check_once(*, restart: bool = True) -> dict[str, Any]:
    store = KnowledgeStore(str(PROJECT_ROOT / "storage"))
    health = get_backend_health(store)
    status = str(health.get("status", "") or "unknown").lower()
    state = _live_backend_process_state()
    event: dict[str, Any] = {
        "schema": "atlas_backend_heartbeat_watchdog_v1",
        "checked_at": _now(),
        "backend_health": health,
        "backend_process": {
            "running": bool(state.running),
            "pid": state.pid,
            "detail": state.detail,
        },
        "restart_enabled": bool(restart),
        "action": "none",
    }
    if status == "live" and bool(health.get("is_fresh", False)):
        event["action"] = "healthy"
        _clear_alert_if_recovered(health)
        _append_event(event)
        return event

    alert = {
        "schema": "atlas_backend_staleness_alert_v1",
        "status": "active",
        "created_at": _now(),
        "message": "Backend heartbeat is stale or unknown; backend may be down or hung.",
        "backend_health": health,
        "backend_process": event["backend_process"],
        "restart_enabled": bool(restart),
        "restart_attempt": {},
    }
    if restart:
        event["action"] = "restart_backend"
        restart_result = _restart_backend(status)
        event["restart_result"] = restart_result
        alert["restart_attempt"] = restart_result
    else:
        event["action"] = "alert_only"
    _write_alert(alert)
    _append_event(event)
    return event


def check_ui_once(*, restart: bool = True) -> dict[str, Any]:
    store = KnowledgeStore(str(PROJECT_ROOT / "storage"))
    threshold = int(os.getenv("ATLAS_UI_HEARTBEAT_STALE_AFTER_SECONDS", os.getenv("ATLAS_HEARTBEAT_STALE_AFTER_SECONDS", "180")) or 180)
    health = get_ui_health(store, threshold_seconds=threshold)
    status = str(health.get("status", "") or "unknown").lower()
    state = _ui_server_process_state()
    process_running = bool(state.running)
    event: dict[str, Any] = {
        "schema": "atlas_ui_heartbeat_watchdog_v1",
        "checked_at": _now(),
        "ui_health": health,
        "ui_process": {
            "running": process_running,
            "pid": state.pid,
            "detail": state.detail,
        },
        "restart_enabled": bool(restart),
        "action": "none",
    }
    if status == "live" and bool(health.get("is_fresh", False)) and process_running:
        event["action"] = "healthy"
        _clear_ui_alert_if_recovered(health)
        _append_event(event)
        return event

    reason = status
    if not process_running:
        reason = "process_missing"
    alert = {
        "schema": "atlas_ui_staleness_alert_v1",
        "status": "active",
        "created_at": _now(),
        "message": "UI server heartbeat is stale, unknown, or the ui-server process is missing.",
        "ui_health": health,
        "ui_process": event["ui_process"],
        "restart_enabled": bool(restart),
        "restart_attempt": {},
    }
    if restart:
        event["action"] = "restart_ui_server"
        restart_result = _restart_ui_server(reason)
        event["restart_result"] = restart_result
        alert["restart_attempt"] = restart_result
    else:
        event["action"] = "alert_only"
    _write_ui_alert(alert)
    _append_event(event)
    return event


def check_all_once(*, restart: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "atlas_combined_heartbeat_watchdog_v1",
        "checked_at": _now(),
        "restart_enabled": bool(restart),
        "backend": {},
        "ui_server": {},
    }
    try:
        result["backend"] = check_once(restart=restart)
    except Exception as exc:
        result["backend"] = {"action": "watchdog_error", "error": str(exc)}
        _append_event({"schema": "atlas_backend_heartbeat_watchdog_v1", "checked_at": _now(), "action": "watchdog_error", "error": str(exc), "restart_enabled": bool(restart)})
    try:
        result["ui_server"] = check_ui_once(restart=restart)
    except Exception as exc:
        result["ui_server"] = {"action": "watchdog_error", "error": str(exc)}
        _append_event({"schema": "atlas_ui_heartbeat_watchdog_v1", "checked_at": _now(), "action": "watchdog_error", "error": str(exc), "restart_enabled": bool(restart)})
    return result


def run_forever(*, interval_seconds: float, restart: bool) -> None:
    if not _acquire_single_instance_lock():
        _append_event(
            {
                "schema": "atlas_backend_heartbeat_watchdog_v1",
                "checked_at": _now(),
                "action": "duplicate_watchdog_exit",
                "restart_enabled": bool(restart),
            }
        )
        return
    interval = max(30.0, float(interval_seconds))
    while True:
        check_all_once(restart=restart)
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch Atlas backend and UI heartbeats and alert/restart when stale.")
    parser.add_argument("--once", action="store_true", help="Run one check and exit.")
    parser.add_argument("--interval-seconds", type=float, default=60.0)
    parser.add_argument("--no-restart", action="store_true", help="Only write alert artifacts; do not restart backend.")
    args = parser.parse_args(argv)
    restart = not bool(args.no_restart)
    if args.once:
        print(json.dumps(check_all_once(restart=restart), indent=2, sort_keys=True))
        return 0
    run_forever(interval_seconds=float(args.interval_seconds), restart=restart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
