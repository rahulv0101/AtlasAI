from __future__ import annotations

import contextlib
import ctypes
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from atlas_ai.atlas_wake_config import AtlasWakeConfig
from atlas_ai.config.settings import load_settings


def _hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _terminate_process_windows(pid: int, *, force: bool = False) -> bool:
    if os.name != "nt" or pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32
    process = None
    try:
        process = kernel32.OpenProcess(0x0001 | 0x00100000, False, int(pid))
        if not process:
            return not process_exists(pid)
        if not force:
            with contextlib.suppress(Exception):
                os.kill(int(pid), signal.SIGTERM)
            for _ in range(20):
                if not process_exists(pid):
                    return True
                time.sleep(0.1)
        if kernel32.TerminateProcess(process, 1) == 0:
            return not process_exists(pid)
        kernel32.WaitForSingleObject(process, 2000)
        return not process_exists(pid)
    except Exception:
        return not process_exists(pid)
    finally:
        if process:
            with contextlib.suppress(Exception):
                kernel32.CloseHandle(process)


@dataclass(frozen=True)
class ProcessState:
    name: str
    running: bool
    detail: str = ""
    pid: int | None = None


def classify_atlas_process_role(command: str) -> dict[str, str]:
    cmd = str(command or "")
    low = cmd.lower()
    if "atlas_ai.main" not in low:
        return {"role": "unknown", "reason": "not_atlas_main"}
    match = re.search(r"(?:^|\s)-m\s+atlas_ai\.main\s+([a-z0-9_-]+)", low)
    if not match:
        return {"role": "unknown", "reason": "atlas_main_subcommand_missing"}
    subcommand = str(match.group(1) or "").strip().lower()
    if subcommand == "live":
        return {"role": "backend_trader", "reason": "atlas_main_live"}
    if subcommand == "paper-compare":
        return {"role": "backend_paper_compare", "reason": "atlas_main_paper_compare"}
    if subcommand == "ui-server":
        return {"role": "ui_server", "reason": "atlas_main_ui_server"}
    if subcommand == "ui-browser":
        return {"role": "ui_browser_wrapper", "reason": "atlas_main_ui_browser"}
    if subcommand in {"ui", "ui-browser"}:
        return {"role": "ui_desktop_wrapper", "reason": "atlas_main_ui"}
    return {"role": "unknown", "reason": f"atlas_main_{subcommand}"}


def _parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds_from_timestamp(value: Any) -> float | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _processes() -> list[dict[str, Any]]:
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=_hidden_creationflags(),
        ).strip()
        if output:
            data = json.loads(output)
            items = data if isinstance(data, list) else [data]
            return [item for item in items if isinstance(item, dict)]
    except Exception:
        pass

    # Fallback for restricted Windows sessions where Win32_Process can return access denied.
    # We keep at least pid+name visibility so readiness checks stay reliable.
    fallback_cmd = (
        "Get-Process | "
        "Select-Object @{Name='ProcessId';Expression={$_.Id}},"
        "@{Name='Name';Expression={$_.ProcessName}},"
        "@{Name='CommandLine';Expression={''}} | "
        "ConvertTo-Json -Depth 3"
    )
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", fallback_cmd],
            text=True,
            stderr=subprocess.DEVNULL,
            creationflags=_hidden_creationflags(),
        ).strip()
        if not output:
            return []
        data = json.loads(output)
        items = data if isinstance(data, list) else [data]
        return [item for item in items if isinstance(item, dict)]
    except Exception:
        return []


def _find_process(name_predicate, command_predicate=None) -> ProcessState:
    try:
        for item in _processes():
            name = str(item.get("Name", "") or "")
            command = str(item.get("CommandLine", "") or "")
            if not name_predicate(name):
                continue
            if command_predicate is not None and not command_predicate(command):
                continue
            pid = item.get("ProcessId")
            try:
                pid_value = int(pid)
            except Exception:
                pid_value = None
            return ProcessState(name=name, running=True, detail=command, pid=pid_value)
    except Exception as exc:
        return ProcessState(name="unknown", running=False, detail=str(exc))
    return ProcessState(name="unknown", running=False, detail="")


def mt5_process_state(config: AtlasWakeConfig) -> ProcessState:
    exe_name = Path(config.mt5_executable_path).stem.lower()
    return _find_process(
        lambda name: name.lower().removesuffix(".exe") in {exe_name, "terminal64"}
    )


def backend_process_state(config: AtlasWakeConfig | None = None) -> ProcessState:
    launch_profile = str(getattr(config, "launch_profile", "paper_compare") or "paper_compare").strip().lower()
    if launch_profile == "guarded_demo":
        state = _find_process(
            lambda name: name.lower().removesuffix(".exe") in {"python", "pythonw"},
            lambda cmd: str(classify_atlas_process_role(cmd).get("role", "")) == "backend_trader",
        )
    else:
        state = _find_process(
            lambda name: name.lower().removesuffix(".exe") in {"python", "pythonw"},
            lambda cmd: str(classify_atlas_process_role(cmd).get("role", "")) == "backend_paper_compare",
        )
    if state.running:
        role_meta = classify_atlas_process_role(state.detail)
        return ProcessState(
            name=state.name,
            running=True,
            detail=f"{state.detail} | role={role_meta.get('role', '')} reason={role_meta.get('reason', '')}",
            pid=state.pid,
        )
    # Fallback when command-line visibility is restricted: rely on Atlas lock files.
    settings = load_settings()
    for lock_name in ("atlas_live.pid", "atlas_live_worker.pid"):
        lock_path = Path(settings.storage_dir) / lock_name
        if not lock_path.exists():
            continue
        try:
            pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            continue
        if process_exists(pid):
            return ProcessState(name="python.exe", running=True, detail=f"pid_lock:{lock_name}", pid=pid)
    return state


def frontend_process_state() -> ProcessState:
    state = _find_process(
        lambda name: name.lower().removesuffix(".exe") in {"python", "pythonw"},
        lambda cmd: str(classify_atlas_process_role(cmd).get("role", "")) in {"ui_server", "ui_desktop_wrapper", "ui_browser_wrapper"},
    )
    if state.running:
        role_meta = classify_atlas_process_role(state.detail)
        return ProcessState(
            name=state.name,
            running=True,
            detail=f"{state.detail} | role={role_meta.get('role', '')} reason={role_meta.get('reason', '')}",
            pid=state.pid,
        )
    settings = load_settings()
    lock_path = Path(settings.storage_dir) / "atlas_ui.pid"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if process_exists(pid):
            return ProcessState(name="python.exe", running=True, detail="pid_lock:atlas_ui.pid", pid=pid)
    return state


def fetch_json(url: str, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def frontend_api_status(config: AtlasWakeConfig) -> dict[str, Any] | None:
    compact_url = f"{config.api_base_url}/api/status/compact"
    compact = fetch_json(compact_url, timeout=2.0)
    if compact:
        payload = dict(compact)
        payload.setdefault("_health_probe_source", "compact")
        payload.setdefault("comparison_active", False)
        payload.setdefault("comparison_run_id", "")
        return payload
    payload = fetch_json(config.status_url, timeout=2.0)
    if payload:
        payload.setdefault("_health_probe_source", "status")
    return payload


def comparison_summary(config: AtlasWakeConfig) -> dict[str, Any]:
    summary_path = Path(config.comparison_root) / config.comparison_run_id / "comparison_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def experiment_status(config: AtlasWakeConfig) -> dict[str, Any]:
    status_path = Path(config.comparison_root) / config.comparison_run_id / "experiment" / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def runtime_status() -> dict[str, Any]:
    settings = load_settings()
    status_path = Path(settings.storage_dir) / "status.json"
    if not status_path.exists():
        return {}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_backend_healthy(config: AtlasWakeConfig) -> tuple[bool, str, dict[str, Any]]:
    if str(getattr(config, "launch_profile", "paper_compare")).strip().lower() == "guarded_demo":
        status = runtime_status()
        updated_at = _parse_timestamp(status.get("updated_at"))
        now = datetime.now(timezone.utc)
        age_seconds = (now - updated_at).total_seconds() if updated_at else None
        runtime_execute = bool(status.get("execute_trades", False))
        process_running = backend_process_state(config).running
        fresh = age_seconds is not None and age_seconds <= config.health_stale_after_seconds
        healthy = (process_running or fresh) and runtime_execute and fresh
        detail = (
            f"process_running={process_running}, runtime_execute={runtime_execute}, "
            f"fresh={fresh}, age_seconds={None if age_seconds is None else round(age_seconds, 1)}"
        )
        return healthy, detail, {"status": status}

    status = experiment_status(config)
    summary = comparison_summary(config)
    updated_at = _parse_timestamp(status.get("updated_at") or summary.get("updated_at"))
    now = datetime.now(timezone.utc)
    age_seconds = (now - updated_at).total_seconds() if updated_at else None
    comparison_ok = str(summary.get("run_id", "")) == config.comparison_run_id
    paper_mode = bool(dict(status.get("system_health", {}) or {}).get("paper_mode_confirmed", False))
    process_running = backend_process_state(config).running
    fresh = age_seconds is not None and age_seconds <= config.health_stale_after_seconds
    # In restricted Windows sessions command-line process visibility can be limited.
    # Treat a fresh run summary as sufficient evidence of backend activity.
    healthy = (process_running or fresh) and comparison_ok and paper_mode and fresh
    detail = (
        f"process_running={process_running}, comparison_ok={comparison_ok}, "
        f"paper_mode={paper_mode}, fresh={fresh}, age_seconds={None if age_seconds is None else round(age_seconds, 1)}"
    )
    return healthy, detail, {"status": status, "summary": summary}


def is_frontend_healthy(config: AtlasWakeConfig) -> tuple[bool, str, dict[str, Any]]:
    payload = frontend_api_status(config)
    process_running = frontend_process_state().running
    if str(getattr(config, "launch_profile", "paper_compare")).strip().lower() == "guarded_demo":
        comparison_active = bool(payload) and bool(payload.get("comparison_active", False))
        compact_probe = bool(payload) and str(payload.get("_health_probe_source", "") or "") == "compact"
        runtime_execute = bool(payload) and bool(payload.get("execute_trades", False) or compact_probe)
        desktop_window_ready = bool(payload) and bool(payload.get("ui_desktop_window_ready", False))
        desktop_window = dict((payload or {}).get("ui_desktop_window", {}) or {})
        desktop_updated_age_seconds = _age_seconds_from_timestamp(desktop_window.get("updated_at"))
        desktop_state_fresh = desktop_updated_age_seconds is not None and desktop_updated_age_seconds <= max(
            10.0,
            float(getattr(config, "health_stale_after_seconds", 30) or 30),
        )
        wrapper_pid_raw = desktop_window.get("wrapper_pid", 0)
        try:
            wrapper_pid = int(wrapper_pid_raw or 0)
        except Exception:
            wrapper_pid = 0
        wrapper_running = bool(wrapper_pid and process_exists(wrapper_pid))
        fallback_active = bool(desktop_window.get("fallback_active", False))
        desktop_disabled = bool(desktop_window.get("disabled_due_to_repeated_failures", False))
        frontend_mode = str(desktop_window.get("frontend_mode", "") or "")
        failure_reason = str(desktop_window.get("failure_reason", "") or "")
        desktop_ready_now = bool(desktop_window_ready and desktop_state_fresh and wrapper_running)
        ui_available = bool(desktop_ready_now or fallback_active)
        healthy = bool(payload) and not comparison_active and runtime_execute and ui_available
        detail = (
            f"process_running={process_running}, api_ok={bool(payload)}, "
            f"comparison_active={comparison_active}, runtime_execute={runtime_execute}, "
            f"desktop_window_ready={desktop_window_ready}, desktop_state_fresh={desktop_state_fresh}, "
            f"desktop_state_age_seconds={None if desktop_updated_age_seconds is None else round(desktop_updated_age_seconds, 1)}, "
            f"wrapper_pid={wrapper_pid}, wrapper_running={wrapper_running}, "
            f"fallback_active={fallback_active}, "
            f"desktop_disabled={desktop_disabled}, frontend_mode={frontend_mode}, failure_reason={failure_reason}"
        )
        return healthy, detail, payload or {}
    comparison_ok = bool(payload) and str(payload.get("comparison_run_id", "")) == config.comparison_run_id
    paper_mode = bool(payload) and bool(dict(payload.get("system_health", {}) or {}).get("paper_mode_confirmed", False))
    # API responsiveness is the practical health signal for the frontend.
    healthy = bool(payload) and comparison_ok and paper_mode
    detail = f"process_running={process_running}, api_ok={bool(payload)}, comparison_ok={comparison_ok}, paper_mode={paper_mode}"
    return healthy, detail, payload or {}


def is_mt5_ready(config: AtlasWakeConfig) -> tuple[bool, str]:
    settings = load_settings()
    attempts = 3
    last_detail = "mt5_readiness_unknown"
    for attempt in range(1, attempts + 1):
        initialized = False
        try:
            initialized = mt5.initialize(path=config.mt5_executable_path) if config.mt5_executable_path else mt5.initialize()
            if not initialized:
                last_detail = f"initialize_failed={mt5.last_error()}"
                continue
            authorized = mt5.login(
                login=settings.hfm_login,
                password=settings.hfm_password,
                server=settings.hfm_server,
            )
            if not authorized:
                last_detail = f"login_failed={mt5.last_error()}"
                continue
            terminal = mt5.terminal_info()
            if terminal is None:
                last_detail = "terminal_info_unavailable"
                continue
            if not bool(getattr(terminal, "trade_allowed", False)):
                last_detail = "autotrading_disabled"
                continue
            if not mt5.symbol_select(settings.symbol, True):
                last_detail = f"symbol_select_failed={settings.symbol}"
                continue
            rates = mt5.copy_rates_from_pos(settings.symbol, mt5.TIMEFRAME_M1, 0, 5)
            if rates is None or len(rates) == 0:
                last_detail = f"rates_unavailable={mt5.last_error()}"
                continue
            return True, "mt5_ready"
        except Exception as exc:
            last_detail = str(exc)
        finally:
            if initialized:
                mt5.shutdown()
        if attempt < attempts:
            time.sleep(1.5)
    return False, last_detail


def wait_for(predicate, timeout_seconds: int, poll_seconds: float) -> tuple[bool, Any]:
    deadline = time.time() + max(1, timeout_seconds)
    last_result: Any = None
    while time.time() < deadline:
        result = predicate()
        last_result = result
        if isinstance(result, tuple):
            if result and bool(result[0]):
                return True, result
        elif result:
            return True, result
        time.sleep(max(0.2, poll_seconds))
    return False, last_result


def process_exists(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {int(pid)} | Out-Null"],
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=_hidden_creationflags(),
        )
        return True
    except Exception:
        return False


def stop_process(pid: int | None, force: bool = False) -> bool:
    if not pid:
        return False
    if not process_exists(pid):
        return True
    if os.name == "nt":
        return _terminate_process_windows(int(pid), force=force)
    try:
        cmd = ["taskkill", "/PID", str(int(pid))]
        cmd.append("/T")
        if force:
            cmd.append("/F")
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=_hidden_creationflags(),
        )
        if int(result.returncode or 0) == 0:
            return True
        # Treat "already exited / teardown race" as success if the pid is gone now.
        return not process_exists(pid)
    except Exception:
        return not process_exists(pid)


def json_file_status(path: Path) -> dict[str, Any]:
    payload = {
        "path": str(path),
        "exists": path.exists(),
        "readable": False,
        "timestamp": "",
        "error": "",
    }
    if not path.exists():
        return payload
    try:
        json.loads(path.read_text(encoding="utf-8"))
        payload["readable"] = True
        payload["timestamp"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception as exc:
        payload["error"] = str(exc)
    return payload
