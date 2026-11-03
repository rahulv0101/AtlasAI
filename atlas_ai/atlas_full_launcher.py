from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.atlas_process_guard import (
    backend_process_state,
    fetch_json,
    frontend_process_state,
    is_backend_healthy,
    is_frontend_healthy,
    is_mt5_ready,
    mt5_process_state,
    process_exists,
    stop_process,
    wait_for,
)
from atlas_ai.atlas_full_shutdown import run_full_shutdown
from atlas_ai.atlas_wake_config import AtlasWakeConfig, load_atlas_wake_config
from atlas_ai.memory.knowledge_store import KnowledgeStore


@dataclass
class ComponentSummary:
    status: str
    detail: str
    already_running: bool = False
    pid: int | None = None


def _log_path(config: AtlasWakeConfig) -> Path:
    return Path(config.project_root) / "storage" / "atlas_wake.log"


def _append_launcher_trace(event: str, extra: dict[str, Any] | None = None) -> None:
    try:
        trace_dir = Path.cwd() / "storage"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "launcher_trace.jsonl"
        payload: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "pid": int(os.getpid()),
            "ppid": int(os.getppid()),
            "argv": list(sys.argv),
            "python": str(sys.executable),
            "base_python": str(getattr(sys, "_base_executable", sys.executable)),
            "is_venv_python": bool(_is_venv_python()),
            "allow_non_venv": str(os.getenv("ATLAS_ALLOW_NON_VENV_LAUNCHER", "") or ""),
            "pinned_python": str(os.getenv("ATLAS_PINNED_PYTHON", "") or ""),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _is_venv_python() -> bool:
    exe = str(Path(sys.executable)).lower()
    return ".venv\\scripts\\python.exe" in exe or ".venv\\scripts\\pythonw.exe" in exe


def _mode_requires_venv() -> bool:
    # Full launcher is responsible for live/ui worker orchestration.
    return True


def _allow_non_venv_launcher() -> bool:
    return str(os.getenv("ATLAS_ALLOW_NON_VENV_LAUNCHER", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _list_windows_python_processes() -> list[tuple[int, str]]:
    if os.name != "nt":
        return []
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -in @('python.exe','pythonw.exe') } | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=_hidden_creationflags(),
        ).strip()
    except Exception:
        return []
    if not out:
        return []
    try:
        data = json.loads(out)
    except Exception:
        return []
    items = data if isinstance(data, list) else [data]
    results: list[tuple[int, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            pid_i = int(item.get("ProcessId"))
        except Exception:
            continue
        cmdline = str(item.get("CommandLine", "")).lower()
        results.append((pid_i, cmdline))
    return results


def _terminate_non_venv_launcher_duplicates() -> None:
    if os.name != "nt" or not _is_venv_python():
        return
    current_pid = os.getpid()
    for pid, cmdline in _list_windows_python_processes():
        if pid == current_pid:
            continue
        if "atlas_ai.atlas_full_launcher" not in cmdline:
            continue
        if ".venv\\scripts\\python.exe" in cmdline or ".venv\\scripts\\pythonw.exe" in cmdline:
            continue
        with contextlib.suppress(Exception):
            stop_process(pid, force=True)


def _ensure_single_launcher_instance(project_root: Path) -> bool:
    lock_dir = project_root / "storage"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / "atlas_full_launcher.lock"
    current_pid = str(os.getpid())
    if lock_file.exists():
        with contextlib.suppress(Exception):
            existing = lock_file.read_text(encoding="utf-8").strip()
            if existing and existing != current_pid:
                # Verify existing pid is a running launcher process.
                for pid, cmdline in _list_windows_python_processes():
                    if str(pid) == existing and "atlas_ai.atlas_full_launcher" in cmdline:
                        return False
    with contextlib.suppress(Exception):
        lock_file.write_text(current_pid, encoding="utf-8")
    return True


def _spawn_log_dir(cwd: str) -> Path:
    path = Path(cwd) / "storage" / "wake_spawn_logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ps_single_quote(value: object) -> str:
    return str(value or "").replace("'", "''")


def _update_launch_runtime_flags(config: AtlasWakeConfig, **fields: Any) -> None:
    store = KnowledgeStore(str(Path(config.project_root) / "storage"))
    ts = datetime.now(timezone.utc).isoformat()

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        flags = dict(current or {})
        state = dict(flags.get("launcher_lifecycle", {}) or {})
        state.update({k: v for k, v in fields.items()})
        state["updated_at"] = ts
        flags["launcher_lifecycle"] = state
        return flags

    with contextlib.suppress(Exception):
        store.update_runtime_flags(_mutator)


def setup_logging(config: AtlasWakeConfig) -> None:
    log_path = _log_path(config)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _spawn(
    command: list[str],
    cwd: str,
    log_label: str = "",
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    # FIX: pin child interpreter for all descendants to prevent mixed .venv/system-python respawn chains.
    if command:
        pinned_python = str(command[0])
        env["ATLAS_PINNED_PYTHON"] = pinned_python
        # FIX: propagate pinned interpreter to child spawn contexts.
        env["PYTHONEXECUTABLE"] = pinned_python
    env["ATLAS_WAKE_LAUNCH_PROFILE"] = str(env.get("ATLAS_WAKE_LAUNCH_PROFILE", "") or "")
    for key, value in dict(extra_env or {}).items():
        env[str(key)] = str(value)

    # Avoid persistent Windows venv redirector parent processes:
    # spawn the base python directly but preserve venv identity.
    spawn_command = list(command or [])
    if os.name == "nt" and spawn_command:
        raw = str(spawn_command[0] or "").strip().strip('"')
        exe = Path(raw) if raw else None
        lowered = str(exe).replace("/", "\\").lower() if exe is not None else ""
        if exe is not None and exe.name.lower() == "python.exe" and "\\.venv\\scripts\\python.exe" in lowered:
            base_python = str(getattr(sys, "_base_executable", "") or "").strip()
            if not base_python or not Path(base_python).exists():
                cfg = exe.parent.parent / "pyvenv.cfg"
                if cfg.exists():
                    with contextlib.suppress(Exception):
                        for raw_line in cfg.read_text(encoding="utf-8").splitlines():
                            line = str(raw_line or "").strip()
                            if not line.lower().startswith("home"):
                                continue
                            _, _, rhs = line.partition("=")
                            home = rhs.strip().strip('"').strip("'")
                            candidate = Path(home) / "python.exe"
                            if candidate.exists():
                                base_python = str(candidate)
                                break
            if base_python and Path(base_python).exists():
                env["__PYVENV_LAUNCHER__"] = str(exe)
                spawn_command[0] = str(base_python)

    python_exe = Path(command[0]) if command else None
    if python_exe and python_exe.name.lower() == "python.exe":
        scripts_dir = python_exe.parent
        venv_dir = scripts_dir.parent if scripts_dir.name.lower() == "scripts" else None
        if venv_dir is not None:
            env["VIRTUAL_ENV"] = str(venv_dir)
            env["PATH"] = f"{scripts_dir}{os.pathsep}{env.get('PATH', '')}"
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    subcommand = _atlas_main_subcommand(command)
    if log_label:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        log_path = _spawn_log_dir(cwd) / f"{log_label}_{stamp}.log"
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write(f"# command: {' '.join(command)}\n")
        if spawn_command != list(command or []):
            log_handle.write(f"# spawn_command: {' '.join(spawn_command)}\n")
        log_handle.flush()
        kwargs["stdout"] = log_handle
        kwargs["stderr"] = log_handle
    if os.name == "nt":
        creationflags = 0
        ui_like = subcommand in {"ui", "ui-server", "ui-browser"}
        if not ui_like:
            creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if creationflags:
            kwargs["creationflags"] = creationflags
    proc = subprocess.Popen(spawn_command, **kwargs)
    if log_label:
        logging.info("Spawn logs for %s redirected to %s", log_label, str(log_path))
    return proc


def _atlas_main_subcommand(command: list[str]) -> str:
    parts = [str(x or "").strip().lower() for x in list(command or [])]
    try:
        main_index = parts.index("atlas_ai.main")
    except ValueError:
        return ""
    if main_index + 1 >= len(parts):
        return ""
    return str(parts[main_index + 1] or "").strip().lower()


def _frontend_should_force_browser_only(config: AtlasWakeConfig) -> tuple[bool, str]:
    try:
        runtime_path = Path(config.project_root) / "storage" / "runtime_flags.json"
        if runtime_path.exists():
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            ui_state = dict(payload.get("ui_desktop_window", {}) or {})
            if bool(ui_state.get("disabled_due_to_repeated_failures", False)):
                return True, "runtime_flags_repeated_failures_disabled"
    except Exception:
        pass
    return False, ""


def _contains_forbidden_guarded_demo_routing(command: list[str]) -> str:
    joined = " ".join(str(x) for x in list(command or [])).strip().lower()
    if not joined:
        return ""
    if "paper-compare" in joined:
        return "paper-compare"
    if "--comparison-root" in joined:
        return "--comparison-root"
    if "storage_compare_runs" in joined:
        return "storage_compare_runs"
    return ""


def _validate_guarded_demo_commands(config: AtlasWakeConfig) -> ComponentSummary:
    profile = str(getattr(config, "launch_profile", "") or "").strip().lower()
    if profile != "guarded_demo":
        return ComponentSummary(status="ok", detail="profile_not_guarded_demo")
    backend_forbidden = _contains_forbidden_guarded_demo_routing(config.backend_launch_command)
    if backend_forbidden:
        return ComponentSummary(
            status="failed",
            detail=(
                "guarded_demo startup rejected: backend command contains forbidden token "
                f"'{backend_forbidden}'"
            ),
        )
    frontend_forbidden = _contains_forbidden_guarded_demo_routing(config.frontend_launch_command)
    if frontend_forbidden:
        return ComponentSummary(
            status="failed",
            detail=(
                "guarded_demo startup rejected: frontend command contains forbidden token "
                f"'{frontend_forbidden}'"
            ),
        )
    return ComponentSummary(status="ok", detail="guarded_demo_command_validation_passed")


def _start_mt5_if_needed(config: AtlasWakeConfig) -> ComponentSummary:
    state = mt5_process_state(config)
    if state.running:
        logging.info("MT5 already running")
        return ComponentSummary(status="running", detail="MT5 already running", already_running=True, pid=state.pid)
    if not config.mt5_executable_path:
        return ComponentSummary(status="failed", detail="MT5 executable path is not configured")
    _spawn([config.mt5_executable_path], cwd=config.project_root)
    logging.info("Started MT5 terminal from %s", config.mt5_executable_path)
    return ComponentSummary(status="started", detail=f"Started MT5 terminal from {config.mt5_executable_path}")


def _wait_for_mt5(config: AtlasWakeConfig) -> ComponentSummary:
    _update_launch_runtime_flags(config, mt5_readiness_started_at=datetime.now(timezone.utc).isoformat())
    ok, result = wait_for(
        lambda: is_mt5_ready(config),
        timeout_seconds=config.mt5_ready_timeout_seconds,
        poll_seconds=2.0,
    )
    detail = result[1] if isinstance(result, tuple) and len(result) > 1 else "mt5_readiness_unknown"
    if ok:
        _update_launch_runtime_flags(config, mt5_readiness_completed_at=datetime.now(timezone.utc).isoformat(), mt5_readiness_result="healthy")
        logging.info("MT5 readiness confirmed: %s", detail)
        return ComponentSummary(status="healthy", detail=detail, already_running=mt5_process_state(config).running)
    _update_launch_runtime_flags(config, mt5_readiness_completed_at=datetime.now(timezone.utc).isoformat(), mt5_readiness_result="failed")
    logging.error("MT5 readiness failed: %s", detail)
    return ComponentSummary(status="failed", detail=detail)


def _start_backend_if_needed(config: AtlasWakeConfig) -> ComponentSummary:
    healthy, detail, health_payload = is_backend_healthy(config)
    state = backend_process_state(config)
    launch_profile = str(getattr(config, "launch_profile", "") or "").strip().lower()
    if healthy and (launch_profile != "guarded_demo" or state.running):
        logging.info("Backend already running")
        return ComponentSummary(status="running", detail=f"backend already running ({detail})", already_running=True, pid=state.pid)
    if healthy and launch_profile == "guarded_demo" and not state.running:
        status = dict((health_payload or {}).get("status", {}) or {})
        status_process = dict(status.get("process", {}) or {})
        try:
            status_pid = int(status_process.get("pid", 0) or 0)
        except Exception:
            status_pid = 0
        if status_pid > 0 and process_exists(status_pid):
            # In some Windows sessions, command-line process visibility is restricted even when
            # the backend is healthy and updating status. If the status-linked PID is alive,
            # skip forced restart to avoid duplicate spawn loops.
            logging.info(
                "Backend health is fresh in guarded-demo mode; process visibility is limited, skipping forced restart (%s)",
                detail,
            )
            return ComponentSummary(status="running", detail=f"backend healthy (visibility_limited) ({detail})", already_running=True, pid=status_pid)
        logging.warning(
            "Backend health looked fresh but status PID is not alive; starting backend to avoid stale-snapshot false positive (%s)",
            detail,
        )
    if state.running:
        logging.warning("Backend process exists but is unhealthy; restarting pid=%s (%s)", state.pid, detail)
        stop_process(state.pid, force=False)
        wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
        if process_exists(state.pid):
            stop_process(state.pid, force=True)
            wait_for(lambda: not process_exists(state.pid), timeout_seconds=6, poll_seconds=0.5)
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        proc = _spawn(config.backend_launch_command, cwd=config.project_root, log_label="backend")
        _update_launch_runtime_flags(
            config,
            backend_spawned_at=datetime.now(timezone.utc).isoformat(),
            backend_spawn_command=list(config.backend_launch_command),
            launch_profile=str(config.launch_profile or ""),
        )
        logging.info("Started backend (attempt %s/%s): %s", attempt, max_attempts, " ".join(config.backend_launch_command))
        time.sleep(6.0)
        if proc.poll() is None:
            return ComponentSummary(status="started", detail="Started Atlas backend / paper comparison worker")
        logging.warning("Backend exited immediately on attempt %s with exit_code=%s", attempt, proc.returncode)
        if attempt < max_attempts:
            mt5_ok, mt5_detail = is_mt5_ready(config)
            logging.info("Revalidated MT5 before backend retry: ok=%s detail=%s", mt5_ok, mt5_detail)
            time.sleep(2.0)
    return ComponentSummary(status="failed", detail="Backend exited immediately after launch attempts")


def _wait_for_backend(config: AtlasWakeConfig) -> ComponentSummary:
    _update_launch_runtime_flags(config, backend_readiness_started_at=datetime.now(timezone.utc).isoformat())
    ok, result = wait_for(
        lambda: is_backend_healthy(config),
        timeout_seconds=config.backend_ready_timeout_seconds,
        poll_seconds=max(2.0, min(10.0, config.poll_seconds / 2)),
    )
    detail = result[1] if isinstance(result, tuple) and len(result) > 1 else "backend_readiness_unknown"
    state = backend_process_state(config)
    if ok:
        _update_launch_runtime_flags(config, backend_readiness_completed_at=datetime.now(timezone.utc).isoformat(), backend_readiness_result="healthy")
        logging.info("Backend readiness confirmed: %s", detail)
        return ComponentSummary(status="healthy", detail=detail, already_running=state.running, pid=state.pid)
    _update_launch_runtime_flags(config, backend_readiness_completed_at=datetime.now(timezone.utc).isoformat(), backend_readiness_result="failed")
    logging.error("Backend readiness failed: %s", detail)
    return ComponentSummary(status="failed", detail=detail, already_running=state.running, pid=state.pid)


def _start_frontend_if_needed(config: AtlasWakeConfig) -> ComponentSummary:
    healthy, detail, _ = is_frontend_healthy(config)
    state = frontend_process_state()
    if healthy:
        logging.info("Frontend already running")
        return ComponentSummary(status="running", detail=f"frontend already running ({detail})", already_running=True, pid=state.pid)
    if state.running:
        logging.warning("Frontend process exists but is unhealthy; restarting pid=%s (%s)", state.pid, detail)
        stop_process(state.pid, force=False)
        wait_for(lambda: not process_exists(state.pid), timeout_seconds=10, poll_seconds=0.5)
        if process_exists(state.pid):
            stop_process(state.pid, force=True)
            wait_for(lambda: not process_exists(state.pid), timeout_seconds=6, poll_seconds=0.5)
    force_browser_only, force_reason = _frontend_should_force_browser_only(config)
    extra_env = {"ATLAS_UI_FORCE_BROWSER_ONLY": "1"} if force_browser_only else None
    frontend_command = list(config.frontend_launch_command or [])
    ui_server_command = list(frontend_command)
    if len(ui_server_command) >= 4 and str(ui_server_command[3]).strip().lower() in {"ui", "ui-browser"}:
        ui_server_command[3] = "ui-server"
    if force_browser_only and len(frontend_command) >= 4 and str(frontend_command[3]).strip().lower() == "ui":
        frontend_command[3] = "ui-browser"
    if force_browser_only:
        logging.warning("Starting frontend in forced browser-only mode due to %s", force_reason or "known_desktop_instability")
    _spawn(ui_server_command, cwd=config.project_root, log_label="ui_server", extra_env=extra_env)
    _update_launch_runtime_flags(
        config,
        ui_server_spawned_at=datetime.now(timezone.utc).isoformat(),
        ui_server_spawn_command=list(ui_server_command),
    )
    time.sleep(2.0)
    _spawn(frontend_command, cwd=config.project_root, log_label="frontend", extra_env=extra_env)
    _update_launch_runtime_flags(
        config,
        frontend_spawned_at=datetime.now(timezone.utc).isoformat(),
        frontend_spawn_command=list(frontend_command),
        frontend_force_browser_only=bool(force_browser_only),
        frontend_force_browser_reason=str(force_reason or ""),
    )
    logging.info("Started frontend: %s", " ".join(frontend_command))
    return ComponentSummary(status="started", detail="Started Atlas frontend / UI")


def _wait_for_frontend(config: AtlasWakeConfig) -> ComponentSummary:
    _update_launch_runtime_flags(config, frontend_readiness_started_at=datetime.now(timezone.utc).isoformat())
    ok, result = wait_for(
        lambda: is_frontend_healthy(config),
        timeout_seconds=config.frontend_ready_timeout_seconds,
        poll_seconds=2.0,
    )
    detail = result[1] if isinstance(result, tuple) and len(result) > 1 else "frontend_readiness_unknown"
    state = frontend_process_state()
    if ok:
        _update_launch_runtime_flags(config, frontend_readiness_completed_at=datetime.now(timezone.utc).isoformat(), frontend_readiness_result="healthy")
        logging.info("Frontend readiness confirmed: %s", detail)
        return ComponentSummary(status="healthy", detail=detail, already_running=state.running, pid=state.pid)
    _update_launch_runtime_flags(config, frontend_readiness_completed_at=datetime.now(timezone.utc).isoformat(), frontend_readiness_result="failed")
    logging.error("Frontend readiness failed: %s", detail)
    return ComponentSummary(status="failed", detail=detail, already_running=state.running, pid=state.pid)


def _desktop_ready_from_status(config: AtlasWakeConfig) -> tuple[bool, str]:
    payload = fetch_json(config.status_url, timeout=2.0) or {}
    window = dict(payload.get("ui_desktop_window", {}) or {})
    desktop_ready = bool(window.get("desktop_window_ready", payload.get("ui_desktop_window_ready", False)))
    frontend_mode = str(window.get("frontend_mode", "") or "")
    failure_reason = str(window.get("failure_reason", "") or "")
    fallback_active = bool(window.get("fallback_active", False))
    if desktop_ready and not fallback_active:
        return True, "desktop_ready"
    if frontend_mode == "desktop" and not failure_reason and not fallback_active:
        return True, "desktop_mode_healthy"
    return False, f"desktop_not_ready:mode={frontend_mode or 'unknown'} fallback={fallback_active} failure={failure_reason or 'none'}"


def _wait_for_desktop_stabilization(config: AtlasWakeConfig) -> ComponentSummary:
    _update_launch_runtime_flags(config, desktop_stabilization_started_at=datetime.now(timezone.utc).isoformat())
    ok, result = wait_for(
        lambda: _desktop_ready_from_status(config),
        timeout_seconds=20.0,
        poll_seconds=1.0,
    )
    detail = result[1] if isinstance(result, tuple) and len(result) > 1 else "desktop_stabilization_unknown"
    if ok:
        _update_launch_runtime_flags(config, desktop_stabilization_completed_at=datetime.now(timezone.utc).isoformat(), desktop_stabilization_result="ready")
        return ComponentSummary(status="healthy", detail=detail)
    _update_launch_runtime_flags(config, desktop_stabilization_completed_at=datetime.now(timezone.utc).isoformat(), desktop_stabilization_result="timeout")
    return ComponentSummary(status="degraded", detail=detail)


def _fetch_status_for_health(config: AtlasWakeConfig, attempts: int = 3, timeout: float = 6.0) -> dict[str, Any]:
    urls = [f"{config.api_base_url}/api/status/compact", config.status_url]
    for attempt in range(max(1, int(attempts))):
        for url in urls:
            payload = fetch_json(url, timeout=timeout) or {}
            if payload:
                payload = dict(payload)
                payload.setdefault("_health_probe_source", "compact" if url.endswith("/api/status/compact") else "status")
                payload.setdefault("comparison_active", False)
                payload.setdefault("comparison_run_id", "")
                return payload
        if attempt < attempts - 1:
            time.sleep(1.0)
    return {}


def verify_health(config: AtlasWakeConfig) -> dict[str, Any]:
    mt5_ready, mt5_detail = is_mt5_ready(config)
    backend_ok, backend_detail, backend_payload = is_backend_healthy(config)
    frontend_ok, frontend_detail, frontend_payload = is_frontend_healthy(config)
    api_payload = dict(frontend_payload or {})
    if not api_payload:
        api_payload = _fetch_status_for_health(config)
    api_ok = bool(api_payload)
    comparison_active = bool(api_payload.get("comparison_active", False))
    ui_window = dict(api_payload.get("ui_desktop_window", {}) or {})
    fallback_active = bool(ui_window.get("fallback_active", False))
    desktop_window_ready = bool(
        ui_window.get(
            "desktop_window_ready",
            api_payload.get("ui_desktop_window_ready", bool(frontend_ok)),
        )
    )
    frontend_mode = str(ui_window.get("frontend_mode", "") or "")
    frontend_failure_reason = str(ui_window.get("failure_reason", "") or "")
    degraded_since = str(ui_window.get("degraded_since", "") or "")
    launch_profile = str(getattr(config, "launch_profile", "paper_compare") or "paper_compare").strip().lower()
    profile_ok = comparison_active if launch_profile == "paper_compare" else (not comparison_active)
    core_runtime_ok = bool(mt5_ready and backend_ok)
    http_ui_ok = bool(api_ok and profile_ok)
    desktop_shell_ok = bool(desktop_window_ready)
    control_surface_usable = bool((http_ui_ok or fallback_active) and profile_ok)
    degraded_reason = ""
    if core_runtime_ok and desktop_shell_ok and http_ui_ok:
        overall = "SUCCESS"
        overall_reason = "backend_healthy_desktop_ready"
    elif core_runtime_ok and control_surface_usable:
        overall = "DEGRADED"
        degraded_reason = frontend_failure_reason or "desktop_shell_unavailable_browser_fallback_active"
        overall_reason = degraded_reason
    else:
        overall = "FAILED"
        if not mt5_ready:
            overall_reason = f"mt5_not_ready:{mt5_detail}"
        elif not backend_ok:
            overall_reason = f"backend_unhealthy:{backend_detail}"
        elif not profile_ok:
            overall_reason = "profile_mismatch"
        elif not control_surface_usable:
            overall_reason = "no_usable_control_surface"
        else:
            overall_reason = "unknown_launch_failure"
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "launch_profile": launch_profile,
        "core_runtime_health": {
            "ok": core_runtime_ok,
            "mt5_ready": bool(mt5_ready),
            "backend_ready": bool(backend_ok),
            "detail": backend_detail,
        },
        "http_ui_health": {
            "ok": http_ui_ok,
            "api_ok": bool(api_ok),
            "profile_ok": bool(profile_ok),
            "status_url": config.status_url,
        },
        "desktop_shell_health": {
            "ok": desktop_shell_ok,
            "fallback_active": fallback_active,
            "frontend_mode": frontend_mode,
            "failure_reason": frontend_failure_reason,
            "degraded_since": degraded_since,
        },
        "control_surface_health": {
            "ok": control_surface_usable,
            "desktop_window_ready": desktop_window_ready,
            "api_ok": bool(api_ok),
            "fallback_active": fallback_active,
        },
        "mt5_hfm_status": {
            "ok": mt5_ready,
            "detail": mt5_detail,
            "process_running": mt5_process_state(config).running,
        },
        "backend_status": {
            "ok": backend_ok,
            "detail": backend_detail,
            "process_running": backend_process_state(config).running,
            "comparison_run_id": config.comparison_run_id,
            "paper_mode_confirmed": bool(
                dict((backend_payload.get("status") or {}).get("system_health", {}) or {}).get("paper_mode_confirmed", False)
            ),
            "runtime_execute_trades": bool(dict(backend_payload.get("status", {}) or {}).get("execute_trades", False)),
        },
        "frontend_status": {
            "ok": control_surface_usable,
            "detail": frontend_detail,
            "process_running": frontend_process_state().running,
            "frontend_mode": frontend_mode,
            "desktop_window_ready": desktop_window_ready,
            "fallback_active": fallback_active,
            "failure_reason": frontend_failure_reason,
            "degraded_since": degraded_since,
        },
        "api_status_health": {
            "ok": api_ok,
            "status_url": config.status_url,
            "comparison_active": comparison_active,
            "comparison_run_id": str(api_payload.get("comparison_run_id", "")),
            "paper_mode_confirmed": bool(dict(api_payload.get("system_health", {}) or {}).get("paper_mode_confirmed", False)),
            "runtime_execute_trades": bool(api_payload.get("execute_trades", False)),
            "frontend_mode": frontend_mode,
            "fallback_active": fallback_active,
            "failure_reason": frontend_failure_reason,
        },
        "overall_result": overall,
        "overall_reason": overall_reason,
        "degraded_reason": degraded_reason,
        "api_status_payload": frontend_payload,
    }


def _run_launch_cycle(config: AtlasWakeConfig, attempt_index: int = 1, max_attempts: int = 1) -> dict[str, Any]:
    logging.info(
        "Atlas wake launch attempt %s/%s requested for profile=%s run_id=%s preset=%s",
        int(attempt_index),
        int(max_attempts),
        config.launch_profile,
        config.comparison_run_id,
        config.paper_preset,
    )
    _update_launch_runtime_flags(
        config,
        startup_requested_at=datetime.now(timezone.utc).isoformat(),
        launch_profile=str(config.launch_profile or ""),
        wake_attempt_index=int(attempt_index),
        wake_max_attempts=int(max_attempts),
    )
    command_validation = _validate_guarded_demo_commands(config)
    if command_validation.status == "failed":
        logging.error(command_validation.detail)
        return {
            "config": config.to_safe_dict(),
            "startup_sequence": [
                {"step": "guarded_demo_command_validation", "result": command_validation.__dict__},
            ],
            "health": {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "launch_profile": str(config.launch_profile or "").strip().lower(),
                "overall_result": "FAILED",
                "failure_reason": command_validation.detail,
            },
        }
    mt5_start = _start_mt5_if_needed(config)
    mt5_health = _wait_for_mt5(config)
    if mt5_health.status != "healthy":
        logging.error("Aborting backend/frontend startup because MT5 is not ready: %s", mt5_health.detail)
        backend_start = ComponentSummary(status="skipped", detail="Skipped: MT5 readiness failed")
        backend_health = ComponentSummary(status="failed", detail="backend skipped due to MT5 readiness failure")
        frontend_start = ComponentSummary(status="skipped", detail="Skipped: MT5 readiness failed")
        frontend_health = ComponentSummary(status="failed", detail="frontend skipped due to MT5 readiness failure")
        desktop_stabilization = ComponentSummary(status="skipped", detail="Skipped: MT5 readiness failed")
    else:
        backend_start = _start_backend_if_needed(config)
        backend_health = _wait_for_backend(config)
        if backend_health.status != "healthy":
            logging.error("Skipping frontend startup because backend is not healthy: %s", backend_health.detail)
            frontend_start = ComponentSummary(status="skipped", detail="Skipped: backend readiness failed")
            frontend_health = ComponentSummary(status="failed", detail="frontend skipped due to backend readiness failure")
            desktop_stabilization = ComponentSummary(status="skipped", detail="Skipped: backend readiness failed")
        else:
            frontend_start = _start_frontend_if_needed(config)
            frontend_health = _wait_for_frontend(config)
            if frontend_health.status == "healthy":
                desktop_stabilization = _wait_for_desktop_stabilization(config)
            else:
                desktop_stabilization = ComponentSummary(status="skipped", detail="Skipped: frontend readiness failed")
    health = verify_health(config)
    summary = {
        "config": config.to_safe_dict(),
        "startup_sequence": [
            {"step": "wake_attempt", "result": {"attempt_index": int(attempt_index), "max_attempts": int(max_attempts)}},
            {"step": "guarded_demo_command_validation", "result": command_validation.__dict__},
            {"step": "mt5_launch_check", "result": mt5_start.__dict__},
            {"step": "mt5_readiness", "result": mt5_health.__dict__},
            {"step": "backend_launch_check", "result": backend_start.__dict__},
            {"step": "backend_readiness", "result": backend_health.__dict__},
            {"step": "frontend_launch_check", "result": frontend_start.__dict__},
            {"step": "frontend_readiness", "result": frontend_health.__dict__},
            {"step": "desktop_stabilization", "result": desktop_stabilization.__dict__},
        ],
        "health": health,
    }
    logging.info(
        "Atlas wake attempt %s/%s completed with result=%s reason=%s",
        int(attempt_index),
        int(max_attempts),
        health["overall_result"],
        health.get("overall_reason", ""),
    )
    _update_launch_runtime_flags(
        config,
        final_startup_verdict=str(health.get("overall_result", "")),
        final_startup_reason=str(health.get("overall_reason", "")),
        fallback_reason=str(dict(health.get("desktop_shell_health", {}) or {}).get("failure_reason", "") or health.get("degraded_reason", "") or ""),
        launch_completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return summary


def run_full_launch(config: AtlasWakeConfig | None = None) -> dict[str, Any]:
    config = config or load_atlas_wake_config()
    setup_logging(config)
    os.environ["ATLAS_WAKE_LAUNCH_PROFILE"] = str(config.launch_profile or "").strip().lower()
    max_attempts = 2
    attempts: list[dict[str, Any]] = []
    recovery_actions: list[dict[str, Any]] = []
    for attempt_index in range(1, max_attempts + 1):
        summary = _run_launch_cycle(config, attempt_index=attempt_index, max_attempts=max_attempts)
        attempts.append(summary)
        overall = str(dict(summary.get("health", {}) or {}).get("overall_result", "") or "")
        overall_reason = str(dict(summary.get("health", {}) or {}).get("overall_reason", "") or "")
        if overall in {"SUCCESS", "DEGRADED"}:
            break
        if attempt_index >= max_attempts:
            break
        logging.error(
            "Atlas wake attempt %s/%s unhealthy (result=%s reason=%s). Triggering full shutdown before restart.",
            attempt_index,
            max_attempts,
            overall or "UNKNOWN",
            overall_reason,
        )
        preserve_mt5 = overall_reason == "no_usable_control_surface"
        shutdown_summary = run_full_shutdown(config, keep_mt5=preserve_mt5)
        recovery_actions.append(
            {
                "reason": "startup_unhealthy",
                "attempt_index": attempt_index,
                "preserved_mt5": bool(preserve_mt5),
                "shutdown": shutdown_summary,
            }
        )
        time.sleep(2.0)
    final = dict(attempts[-1] if attempts else {})
    final_overall = str(dict(final.get("health", {}) or {}).get("overall_result", "") or "")
    if final_overall == "FAILED":
        final_reason = str(dict(final.get("health", {}) or {}).get("overall_reason", "") or "")
        logging.error(
            "Atlas wake final result unhealthy (result=%s reason=%s). Running final cleanup shutdown.",
            final_overall or "UNKNOWN",
            final_reason,
        )
        preserve_mt5 = final_reason == "no_usable_control_surface"
        shutdown_summary = run_full_shutdown(config, keep_mt5=preserve_mt5)
        recovery_actions.append(
            {
                "reason": "final_launch_unhealthy_cleanup",
                "attempt_index": len(attempts),
                "preserved_mt5": bool(preserve_mt5),
                "shutdown": shutdown_summary,
            }
        )
    final["attempts"] = attempts
    if recovery_actions:
        final["recovery_actions"] = recovery_actions
    logging.info("Atlas wake completed with final result=%s", dict(final.get("health", {}) or {}).get("overall_result", "UNKNOWN"))
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the full local Atlas paper stack.")
    parser.add_argument("--json", action="store_true", help="Print the final structured summary as JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _append_launcher_trace("main_enter")
    if _mode_requires_venv() and not _is_venv_python() and not _allow_non_venv_launcher():
        _append_launcher_trace("blocked_non_venv")
        print("atlas_full_launcher blocked: non-venv interpreter is not allowed")
        return 0
    if not _ensure_single_launcher_instance(Path(__file__).resolve().parents[1]):
        _append_launcher_trace("blocked_singleton")
        print("atlas_full_launcher duplicate blocked by singleton lock")
        return 0
    _terminate_non_venv_launcher_duplicates()
    args = parse_args(argv)
    summary = run_full_launch()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps(summary["health"], indent=2))
    return 0 if summary["health"]["overall_result"] in {"SUCCESS", "DEGRADED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
