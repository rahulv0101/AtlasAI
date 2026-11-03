from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import json
import logging
import os
import re
import subprocess
import threading
import sys
import time
import traceback
import urllib.request
import webbrowser
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from atlas_ai.analysis.signal_correlation import build_signal_correlation_report_from_store
from atlas_ai.backtest import Backtester
from atlas_ai.config.settings import load_settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.desktop import launch_desktop_ui
from atlas_ai.execution.broker import HFMConnector
from atlas_ai.learning.edge_building import build_edge_building_report
from atlas_ai.learning.experiment_runner import readiness_report, run_meta_model_experiment
from atlas_ai.learning.fast_path_audit import build_fast_path_promotion_audit_report
from atlas_ai.learning.historical_trainer import HistoricalTrainer
from atlas_ai.learning.label_review import build_label_review
from atlas_ai.learning.setup_quality_score import build_offline_setup_quality_score
from atlas_ai.learning.trade_labeling import run_trade_labeling
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.atlas_process_guard import stop_process
from atlas_ai.paper_compare import run_shadow_paper_comparison
from atlas_ai.runtime_heartbeat import start_runtime_heartbeat
from atlas_ai.self_improvement import build_self_improvement_runtime_guard
from atlas_ai.market_scanner import MarketScanner
from atlas_ai.trader import AtlasTrader
from atlas_ai.ui_server import create_ui_app, serve_ui_app, start_background_live_loop, start_visual_watch_loop
from atlas_ai.strategies.evolution_scheduler import (
    EvolutionScheduler,
    start_evolution_scheduler,
    stop_evolution_scheduler,
)

_LOCK_HANDLES: dict[str, object] = {}
_MUTEX_HANDLES: dict[str, object] = {}
_LIVE_UI_OWNER_ENV = "ATLAS_LIVE_UI_OWNER_PID"


class ModeReconciliationError(RuntimeError):
    """Raised when persisted runtime mode conflicts with launch intent."""


def _hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _process_meta(mode: str = "", role: str = "") -> dict[str, object]:
    return {
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()),
        "mode": str(mode or ""),
        "role": str(role or ""),
        "python": str(sys.executable),
        "base_python": str(getattr(sys, "_base_executable", sys.executable)),
        "is_venv_python": bool(_is_venv_python()),
        "thread": str(threading.current_thread().name),
        "live_ui_owner_pid_env": str(os.getenv(_LIVE_UI_OWNER_ENV, "") or ""),
    }


def _append_startup_trace(event: str, extra: dict[str, object] | None = None) -> None:
    try:
        trace_dir = Path.cwd() / "storage"
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file = trace_dir / "startup_trace.jsonl"
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "pid": int(os.getpid()),
            "ppid": int(os.getppid()),
            "argv": list(sys.argv),
            "python": str(sys.executable),
            "base_python": str(getattr(sys, "_base_executable", sys.executable)),
            "is_venv_python": bool(_is_venv_python()),
            "live_ui_owner_pid_env": str(os.getenv(_LIVE_UI_OWNER_ENV, "") or ""),
        }
        if isinstance(extra, dict):
            payload.update(extra)
        with trace_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_runtime_flags(store: KnowledgeStore, execute_trades: bool) -> None:
    flags = store.load_runtime_flags()
    if "execute_trades" not in flags:
        flags["execute_trades"] = execute_trades
        store.save_runtime_flags(flags)


def _mode_label(enabled: bool | None) -> str:
    if enabled is None:
        return "unset"
    return "live" if bool(enabled) else "paper"


def reconcile_runtime_flags(
    store: KnowledgeStore,
    execute_trades: bool,
    source_mode: str = "",
    mode_override: str = "",
) -> dict[str, object]:
    flags = store.load_runtime_flags()
    env_mode = bool(execute_trades)
    runtime_mode = bool(flags["execute_trades"]) if "execute_trades" in flags else None
    override = str(mode_override or "").strip().lower()
    meta: dict[str, object] = {
        "source_mode": str(source_mode or ""),
        "env_mode": _mode_label(env_mode),
        "runtime_mode": _mode_label(runtime_mode),
        "override": override or "",
    }
    effective_mode: bool
    reason: str

    if override:
        if override == "env":
            effective_mode = env_mode
            reason = "explicit_env_override"
        elif override == "paper":
            effective_mode = False
            reason = "explicit_paper_override"
        elif override == "live":
            effective_mode = True
            reason = "explicit_live_override"
        else:
            raise ValueError(f"Unsupported mode override: {mode_override}")
        flags["execute_trades"] = effective_mode
        store.save_runtime_flags(flags)
    elif runtime_mode is None:
        effective_mode = env_mode
        reason = "runtime_missing_initialized_from_env"
        flags["execute_trades"] = effective_mode
        store.save_runtime_flags(flags)
    elif runtime_mode == env_mode:
        effective_mode = env_mode
        reason = "env_runtime_match"
    else:
        effective_mode = runtime_mode
        reason = "env_runtime_mismatch_blocked"
        meta.update(
            {
                "effective_mode": _mode_label(effective_mode),
                "reason": reason,
            }
        )
        store.log_event(
            "error",
            "Startup blocked: execute_trades mismatch between .env and runtime flags.",
            meta,
        )
        store.save_status(
            {
                "action": "error",
                "message": (
                    "Startup blocked: EXECUTE_TRADES in .env does not match runtime_flags.json. "
                    "Pass --mode-override env|paper|live to make the launch intent explicit."
                ),
                "execute_trades": bool(effective_mode),
                "process": _process_meta(mode=source_mode, role="mode_reconciliation_blocked"),
                "mode_reconciliation": meta,
            }
        )
        raise ModeReconciliationError(
            "execute_trades mismatch between .env and runtime flags; "
            "use --mode-override env|paper|live to choose explicitly"
        )

    meta.update(
        {
            "effective_mode": _mode_label(effective_mode),
            "reason": reason,
        }
    )
    store.log_event("system", "Execution mode reconciled at startup.", meta)
    return {
        "execute_trades": bool(effective_mode),
        "meta": meta,
    }


def guarded_demo_startup_sanity(
    *,
    store: KnowledgeStore,
    execute_trades: bool,
    source_mode: str = "live",
    gate_environment: str = "",
) -> dict[str, object]:
    profile = str(os.getenv("ATLAS_WAKE_LAUNCH_PROFILE", "") or "").strip().lower()
    gate_env = str(gate_environment or "").strip().lower()
    effective_mode = "live" if bool(execute_trades) else "paper"
    trader_mode = "guarded_demo_live_path" if bool(execute_trades) else "paper_early_return_path"
    sanity = {
        "profile": profile or "",
        "execute_trades": bool(execute_trades),
        "effective_mode": effective_mode,
        "trader_mode": trader_mode,
        "gate_execution_environment": gate_env,
        "paper_early_return_blocked": bool(execute_trades),
        "gate_demo_ready": gate_env == "demo",
    }
    startup_ok = bool(execute_trades) and gate_env == "demo" and trader_mode != "paper_early_return_path"
    startup_message = (
        "ATLAS STARTED IN GUARDED DEMO MODE"
        if startup_ok
        else "STARTUP FAILED: NOT IN GUARDED DEMO MODE"
    )
    startup_verification = {
        **sanity,
        "startup_ok": startup_ok,
        "startup_message": startup_message,
    }
    store.log_event("system", startup_message, startup_verification)
    flags = dict(store.load_runtime_flags() or {})
    flags["startup_verification"] = startup_verification
    store.save_runtime_flags(flags)
    if profile == "guarded_demo" and not bool(execute_trades):
        store.save_status(
            {
                "action": "error",
                "message": (
                    "Guarded demo startup rejected: runtime resolved to paper "
                    "(execute_trades=false)."
                ),
                "execute_trades": False,
                "process": _process_meta(mode=source_mode, role="guarded_demo_startup_blocked"),
                "guarded_demo_startup_sanity": startup_verification,
            }
        )
        raise ModeReconciliationError(
            "guarded_demo startup rejected because execute_trades resolved to false"
        )
    if profile == "guarded_demo" and gate_env != "demo":
        store.save_status(
            {
                "action": "error",
                "message": (
                    "Guarded demo startup rejected: execution gate environment is not demo."
                ),
                "execute_trades": bool(execute_trades),
                "process": _process_meta(mode=source_mode, role="guarded_demo_startup_blocked"),
                "guarded_demo_startup_sanity": startup_verification,
            }
        )
        raise ModeReconciliationError(
            "guarded_demo startup rejected because gate execution environment is not demo"
        )
    return startup_verification


def _record_self_improvement_runtime_guard(store: KnowledgeStore, source_mode: str) -> dict[str, object]:
    runtime_guard = build_self_improvement_runtime_guard(store)
    flags = dict(store.load_runtime_flags() or {})
    flags["self_improvement_runtime"] = runtime_guard
    store.save_runtime_flags(flags)
    store.log_event(
        "system",
        "Self-improvement runtime guard loaded approved rule registry.",
        {
            **runtime_guard,
            "source_mode": str(source_mode or ""),
        },
    )
    return runtime_guard


def run_historical_training_if_enabled(settings, store) -> None:
    if not settings.enable_historical_training:
        return
    try:
        store.log_event(
            "training",
            "Starting historical training",
            {
                "months": settings.historical_train_months,
                "max_bars": settings.historical_train_max_bars,
                **_process_meta(role="historical_training"),
            },
        )
        HistoricalTrainer(settings, store).train()
    except Exception as exc:
        store.log_event("error", f"Historical training failed: {exc}")


def start_historical_training_background(settings, store) -> None:
    if not settings.enable_historical_training:
        return
    threading.Thread(
        target=run_historical_training_if_enabled,
        args=(settings, store),
        daemon=True,
    ).start()


def start_periodic_historical_training_background(settings, store) -> None:
    if not settings.enable_historical_training:
        return
    interval_seconds = max(1800.0, float(settings.historical_retrain_hours) * 3600.0)

    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            run_historical_training_if_enabled(settings, store)

    threading.Thread(target=_loop, daemon=True).start()


def _parse_utc_datetime(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _should_run_startup_historical_training(settings, store: KnowledgeStore) -> tuple[bool, dict[str, object]]:
    if not settings.enable_historical_training:
        return False, {"enabled": False, "reason": "disabled"}
    state = dict(store.load_learning_state().get("state", {}))
    evolved_pool = dict(state.get("evolved_strategy_pool", {}))
    generated_at = _parse_utc_datetime(evolved_pool.get("generated_at"))
    if generated_at is None:
        return True, {"enabled": True, "reason": "missing_generated_at"}
    age_hours = max(0.0, (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600.0)
    threshold_hours = max(0.5, float(getattr(settings, "historical_retrain_hours", 24.0)))
    if age_hours >= threshold_hours:
        return True, {
            "enabled": True,
            "reason": "stale_pool",
            "age_hours": round(age_hours, 2),
            "threshold_hours": round(threshold_hours, 2),
        }
    return False, {
        "enabled": True,
        "reason": "recent_pool",
        "age_hours": round(age_hours, 2),
        "threshold_hours": round(threshold_hours, 2),
        "generated_at": generated_at.isoformat(),
    }


def start_periodic_housekeeping_background(settings, store) -> None:
    interval_seconds = 15 * 60

    def _loop() -> None:
        while True:
            time.sleep(interval_seconds)
            result = store.maintain_events(max_lines=int(settings.event_max_lines))
            if result.get("archived", 0) or result.get("dropped_malformed", 0) or result.get("dropped_duplicates", 0):
                store.log_event("system", "Event log maintenance completed.", result)

    threading.Thread(target=_loop, daemon=True).start()


def _terminate_duplicate_mode_processes(mode: str) -> None:
    """Best-effort duplicate cleanup on Windows to enforce one mode instance."""
    if os.name != "nt":
        return
    # Never let a non-venv helper process terminate the main venv process.
    if not _is_venv_python():
        return
    current_pid = os.getpid()
    mode = str(mode).strip().lower()
    if not mode:
        return
    for pid, cmdline in _list_windows_python_processes():
        if not _should_terminate_duplicate_process(current_pid, mode, pid, cmdline):
            continue
        try:
            if stop_process(pid, force=True):
                logging.warning("Stopped duplicate atlas process pid=%s for mode=%s", pid, mode)
        except Exception:
            pass


def _should_terminate_duplicate_process(current_pid: int, mode: str, pid: int, cmdline: str) -> bool:
    mode = str(mode).strip().lower()
    cmdline = str(cmdline or "").lower()
    if not mode:
        return False
    if pid == current_pid:
        return False
    if "atlas_ai.main" not in cmdline:
        return False
    match = re.search(r"atlas_ai\.main(?:\.py)?\s+([a-z0-9\-]+)(?:\s|$)", cmdline)
    subcommand = str(match.group(1) if match else "").strip().lower()
    expected_subcommand = {
        "live": "live",
        "ui": "ui",
        "ui-server": "ui-server",
        "ui-browser": "ui-browser",
        "live-ui": "live-ui",
        "paper-compare": "paper-compare",
    }.get(mode, mode)
    if subcommand != expected_subcommand:
        return False
    if ".venv\\scripts\\python.exe" in cmdline or ".venv\\scripts\\pythonw.exe" in cmdline:
        return False
    return True


def _start_duplicate_reaper(mode: str, interval_seconds: float = 2.0) -> None:
    """Continuously prune non-venv duplicates that may be auto-spawned by GUI libs."""
    if os.name != "nt" or not _is_venv_python():
        return

    def _loop() -> None:
        while True:
            _terminate_duplicate_mode_processes(mode)
            time.sleep(max(1.0, float(interval_seconds)))

    threading.Thread(target=_loop, daemon=True).start()


def _mode_process_count(mode: str) -> int:
    """Count running Atlas mode processes on Windows."""
    if os.name != "nt":
        return 0
    mode = str(mode).strip().lower()
    if not mode:
        return 0
    current_pid = os.getpid()
    count = 0
    for pid, cmdline in _list_windows_python_processes():
        if pid == current_pid:
            continue
        if "atlas_ai.main" not in cmdline:
            continue
        lowered = str(cmdline).lower()
        if not re.search(rf"atlas_ai\.main(?:\.py)?\s+{re.escape(mode)}(?:\s|$)", lowered):
            continue
        count += 1
    return count


def _start_live_worker_process(poll_seconds: int) -> None:
    cmd = [_preferred_python_executable(), "-m", "atlas_ai.main", "live", "--poll-seconds", str(int(poll_seconds))]
    _start_background_process(cmd)


def _start_ui_server_process(mode_override: str = "", comparison_root: str = "", comparison_run_id: str = "") -> None:
    cmd = [_preferred_python_executable(), "-m", "atlas_ai.main", "ui-server"]
    if mode_override:
        cmd.extend(["--mode-override", str(mode_override)])
    if comparison_root:
        cmd.extend(["--comparison-root", str(comparison_root)])
    if comparison_run_id:
        cmd.extend(["--comparison-run-id", str(comparison_run_id)])
    _start_background_process(cmd)


def _start_ui_process() -> None:
    cmd = [_preferred_python_executable(), "-m", "atlas_ai.main", "ui"]
    _start_background_process(cmd)


def _preferred_python_executable() -> str:
    """Prefer project venv python for spawned Atlas mode workers."""
    # FIX: honor pinned interpreter from launcher to prevent mixed-runtime spawn chains.
    pinned = str(os.getenv("ATLAS_PINNED_PYTHON", "") or "").strip()
    if pinned:
        with contextlib.suppress(Exception):
            p = Path(pinned)
            if p.exists():
                return str(p)
    with contextlib.suppress(Exception):
        candidate = Path.cwd() / ".venv" / "Scripts" / "python.exe"
        if candidate.exists():
            return str(candidate)
    return str(sys.executable)


def _resolve_windows_venv_base_python(venv_python: Path) -> str:
    """Resolve base python for a Windows venv redirector executable."""
    base = str(getattr(sys, "_base_executable", "") or "").strip()
    if base and Path(base).exists():
        return base
    cfg = venv_python.parent.parent / "pyvenv.cfg"
    if cfg.exists():
        with contextlib.suppress(Exception):
            for raw in cfg.read_text(encoding="utf-8").splitlines():
                line = str(raw or "").strip()
                if not line.lower().startswith("home"):
                    continue
                _, _, rhs = line.partition("=")
                home = rhs.strip().strip('"').strip("'")
                if not home:
                    continue
                candidate = Path(home) / "python.exe"
                if candidate.exists():
                    return str(candidate)
    return ""


def _normalize_windows_venv_spawn_command(cmd: list[str], env: dict[str, str]) -> list[str]:
    """
    Avoid persistent venv redirector parent processes on Windows.
    Use base python directly while preserving venv identity via __PYVENV_LAUNCHER__.
    """
    command = list(cmd or [])
    if os.name != "nt" or not command:
        return command
    raw = str(command[0] or "").strip().strip('"')
    if not raw:
        return command
    exe = Path(raw)
    lowered = str(exe).replace("/", "\\").lower()
    if exe.name.lower() not in {"python.exe", "pythonw.exe"}:
        return command
    if "\\.venv\\scripts\\python.exe" not in lowered and "\\.venv\\scripts\\pythonw.exe" not in lowered:
        return command
    base_python = _resolve_windows_venv_base_python(exe)
    if not base_python:
        return command
    env["__PYVENV_LAUNCHER__"] = str(exe)
    command[0] = str(base_python)
    return command


def _start_background_process(cmd: list[str]) -> None:
    env = os.environ.copy()
    # Enforce a single interpreter lineage across all Atlas child processes.
    pinned_python = str(_preferred_python_executable())
    env["ATLAS_PINNED_PYTHON"] = pinned_python
    # FIX: force multiprocessing/spawn descendants to resolve to the pinned venv interpreter.
    env["PYTHONEXECUTABLE"] = pinned_python
    python_exe = Path(sys.executable)
    scripts_dir = python_exe.parent
    venv_dir = scripts_dir.parent if scripts_dir.name.lower() == "scripts" else None
    if venv_dir is not None and python_exe.exists():
        env["VIRTUAL_ENV"] = str(venv_dir)
        existing_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([str(scripts_dir), existing_path]) if existing_path else str(scripts_dir)
        env["ATLAS_EXPECTED_PYTHON"] = str(python_exe)
        env.pop("PYTHONHOME", None)
    spawn_cmd = _normalize_windows_venv_spawn_command(cmd, env)
    kwargs: dict[str, object] = {
        "cwd": str(Path.cwd()),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": env,
    }
    if os.name == "nt":
        creationflags = 0
        with contextlib.suppress(Exception):
            creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
        with contextlib.suppress(Exception):
            creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if creationflags:
            kwargs["creationflags"] = creationflags
    subprocess.Popen(spawn_cmd, **kwargs)


def _ensure_live_worker_running(poll_seconds: int, store: KnowledgeStore, source_mode: str = "live-ui") -> None:
    if _mode_process_count("live") > 0:
        store.log_event(
            "system",
            "Live worker already running; UI attached without starting a second broker loop.",
            _process_meta(mode=source_mode, role="live_worker_already_running"),
        )
        return
    _start_live_worker_process(poll_seconds)
    store.log_event(
        "system",
        "Started detached live worker for desktop UI session.",
        {**_process_meta(mode=source_mode, role="live_worker_spawn"), "poll_seconds": int(poll_seconds)},
    )


def _ensure_ui_process_running(store: KnowledgeStore, source_mode: str = "live-ui") -> None:
    if _mode_process_count("ui") > 0:
        store.log_event(
            "system",
            "Desktop UI already running; reusing existing window host.",
            _process_meta(mode=source_mode, role="ui_already_running"),
        )
        return
    _start_ui_process()
    store.log_event(
        "system",
        "Started detached desktop UI process.",
        _process_meta(mode=source_mode, role="ui_spawn"),
    )


def _wait_for_url_ready(url: str, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if int(getattr(resp, "status", 0) or 0) == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def _ui_server_url(host: str = "127.0.0.1", port: int = 8765) -> str:
    return f"http://{host}:{int(port)}"


def _record_ui_process_lifecycle(
    store: KnowledgeStore,
    *,
    process_role: str,
    stage: str,
    status: str,
    error: str = "",
    exc: BaseException | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    role_key = re.sub(r"[^a-z0-9_]+", "_", str(process_role or "").strip().lower()).strip("_") or "unknown"
    payload = {
        "process_role": str(process_role or ""),
        "stage": str(stage or ""),
        "status": str(status or ""),
        "pid": int(os.getpid()),
        "ppid": int(os.getppid()),
        "error": str(error or (str(exc) if exc is not None else "")),
        "exception_type": type(exc).__name__ if exc is not None else "",
        "traceback": traceback.format_exc() if exc is not None else "",
        "meta": dict(meta or {}),
        "ts": ts,
        "updated_at": ts,
    }
    try:
        flags = dict(store.load_runtime_flags() or {})
        flags["ui_process_lifecycle"] = payload
        role_payloads = dict(flags.get("ui_process_roles", {}) or {})
        role_entry = dict(role_payloads.get(role_key, {}) or {})
        history = list(role_entry.get("history", []) or [])
        history.append(payload)
        role_payloads[role_key] = {
            "last": payload,
            "history": history[-20:],
        }
        flags["ui_process_roles"] = role_payloads
        store.save_runtime_flags(flags)
    except Exception:
        pass
    with contextlib.suppress(Exception):
        store.log_event(
            "system" if status in {"starting", "started", "running", "stopped", "intentional_exit"} else "error",
            "UI process lifecycle updated.",
            {"ui_process_lifecycle": payload},
        )
    _append_startup_trace("ui_process_lifecycle", payload)


def _record_ui_wrapper_exit(
    store: KnowledgeStore,
    *,
    stage: str,
    status: str,
    error: str = "",
    exc: BaseException | None = None,
    meta: dict[str, object] | None = None,
) -> None:
    payload_meta = dict(meta or {})
    payload_meta.setdefault("wrapper_pid", int(os.getpid()))
    _record_ui_process_lifecycle(
        store,
        process_role="ui_wrapper",
        stage=str(stage or "wrapper_exit"),
        status=str(status or "stopped"),
        error=error,
        exc=exc,
        meta=payload_meta,
    )


def _prime_desktop_wrapper_state(
    store: KnowledgeStore,
    *,
    stage: str,
    error: str = "",
    frontend_mode: str = "desktop_starting",
) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    wrapper_pid = int(os.getpid())
    try:
        flags = dict(store.load_runtime_flags() or {})
    except Exception:
        flags = {}
    previous = dict(flags.get("ui_desktop_window", {}) or {})
    payload = {
        "ok": False,
        "ready": False,
        "stage": str(stage or ""),
        "backend": str(previous.get("backend", "webview") or "webview"),
        "error": str(error or ""),
        "exception_type": "",
        "traceback": "",
        "classification": {
            "code": "",
            "category": "starting",
            "operator_cause": "",
            "operator_action": "",
        },
        "failure_reason": "",
        "frontend_mode": str(frontend_mode or "desktop_starting"),
        "desktop_window_ready": False,
        "desktop_shell_healthy": False,
        "desktop_ui_available": False,
        "fallback_active": False,
        "fallback_url": str(previous.get("fallback_url", "http://127.0.0.1:8765") or "http://127.0.0.1:8765"),
        "degraded_since": str(previous.get("degraded_since", "") or ts),
        "failure_evidence": {
            "source": "ui_wrapper_startup",
            "wrapper_pid": wrapper_pid,
            "wrapper_stage": str(stage or ""),
            "detected_at": ts,
        },
        "preferred_gui": str(previous.get("preferred_gui", "") or ""),
        "failure_count": int(previous.get("failure_count", 0) or 0),
        "failure_threshold": int(previous.get("failure_threshold", 2) or 2),
        "disabled_due_to_repeated_failures": bool(previous.get("disabled_due_to_repeated_failures", False)),
        "wrapper_pid": wrapper_pid,
        "wrapper_started_at": ts,
        "updated_at": ts,
        "ts": ts,
    }
    flags["ui_desktop_window"] = payload
    with contextlib.suppress(Exception):
        store.save_runtime_flags(flags)
        store.log_event("system", "Desktop wrapper state primed for new run.", {"ui_desktop_window": payload})


def _ensure_ui_server_running(
    store: KnowledgeStore,
    *,
    mode_override: str = "",
    comparison_root: str = "",
    comparison_run_id: str = "",
    source_mode: str = "ui",
    timeout_seconds: float = 20.0,
) -> bool:
    url = _ui_server_url()
    _record_ui_process_lifecycle(
        store,
        process_role="ui_wrapper" if source_mode == "ui" else "ui_browser_wrapper",
        stage="ui_server_probe_existing",
        status="starting",
        meta={"url": url, "timeout_seconds": 2.0, "source_mode": source_mode},
    )
    existing_ready = _wait_for_url_ready(url, timeout_seconds=2.0)
    _record_ui_process_lifecycle(
        store,
        process_role="ui_wrapper" if source_mode == "ui" else "ui_browser_wrapper",
        stage="ui_server_probe_existing",
        status="started" if existing_ready else "not_ready",
        meta={"url": url, "timeout_seconds": 2.0, "source_mode": source_mode},
    )
    if existing_ready:
        store.log_event(
            "system",
            "UI server already running; attaching presentation layer.",
            _process_meta(mode=source_mode, role="ui_server_already_running"),
        )
        return True
    _record_ui_process_lifecycle(
        store,
        process_role="ui_wrapper" if source_mode == "ui" else "ui_browser_wrapper",
        stage="ui_server_spawn",
        status="starting",
        meta={"url": url, "source_mode": source_mode},
    )
    _start_ui_server_process(
        mode_override=mode_override,
        comparison_root=comparison_root,
        comparison_run_id=comparison_run_id,
    )
    store.log_event(
        "system",
        "Started durable UI server process.",
        _process_meta(mode=source_mode, role="ui_server_spawn"),
    )
    ready = _wait_for_url_ready(url, timeout_seconds=timeout_seconds)
    _record_ui_process_lifecycle(
        store,
        process_role="ui_wrapper" if source_mode == "ui" else "ui_browser_wrapper",
        stage="ui_server_spawn",
        status="started" if ready else "failed",
        error="" if ready else "durable_ui_server_not_reachable",
        meta={"url": url, "timeout_seconds": float(timeout_seconds), "source_mode": source_mode},
    )
    return ready


def _should_exit_non_venv_duplicate() -> bool:
    """On Windows, block spawned non-venv duplicates of Atlas runtime."""
    if os.name != "nt":
        return False
    mode = ""
    if len(sys.argv) >= 2:
        mode = str(sys.argv[1]).strip().lower()
    # FIX: hard block non-venv runtime for modes that require deterministic single-runtime startup.
    if _mode_requires_venv(mode) and not _is_venv_python():
        return True
    if _is_venv_python():
        return False
    if _allow_non_venv_main():
        return False
    # If this process was spawned by another Atlas process, it must not run.
    parent_pid = os.getppid()
    parent_cmd = _get_windows_process_cmdline(parent_pid).lower()
    if "atlas_ai.main" in parent_cmd:
        return True
    # Or if any Atlas process already exists, treat this non-venv one as duplicate.
    return any("atlas_ai.main" in cmd for _, cmd in _list_windows_python_processes())


def _is_spawned_live_ui_child() -> bool:
    if os.name != "nt":
        return False
    owner_pid = str(os.getenv(_LIVE_UI_OWNER_ENV, "") or "").strip()
    if not owner_pid:
        return False
    return owner_pid != str(os.getpid())


def _mode_requires_venv(mode: str) -> bool:
    target = str(mode or "").strip().lower()
    return target in {"live", "ui", "ui-server", "ui-browser", "live-ui", "paper-compare"}


def _allow_non_venv_main() -> bool:
    return str(os.getenv("ATLAS_ALLOW_NON_VENV_MAIN", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _is_venv_python() -> bool:
    exe = str(Path(sys.executable)).lower()
    return ".venv\\scripts\\python.exe" in exe or ".venv\\scripts\\pythonw.exe" in exe


def _get_windows_process_cmdline(pid: int) -> str:
    if os.name != "nt" or pid <= 0:
        return ""
    for process_pid, cmd in _list_windows_python_processes():
        if process_pid == pid:
            return cmd
    return ""


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
        pid = item.get("ProcessId")
        cmdline = str(item.get("CommandLine", "")).lower()
        try:
            pid_i = int(pid)
        except Exception:
            continue
        results.append((pid_i, cmdline))
    return results


def _ensure_single_instance(lock_file: Path) -> bool:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    def _is_active_python_pid(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        if os.name == "nt":
            try:
                info = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    creationflags=_hidden_creationflags(),
                ).strip().lower()
                return "python.exe" in info or "pythonw.exe" in info
            except Exception:
                return False
        return True

    # Cross-process file locking to avoid duplicate live/ui loops from launcher quirks.
    lock_key = str(lock_file.resolve())
    lock_fh = open(lock_file, "a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            lock_fh.seek(0)
            with contextlib.suppress(Exception):
                lock_fh.truncate(1)
            try:
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                try:
                    lock_fh.seek(0)
                    existing = lock_fh.read().strip()
                except Exception:
                    existing = ""
                lock_fh.close()
                if existing:
                    logging.warning("Duplicate start blocked for %s; active pid hint: %s", lock_file.name, existing)
                return False
        else:
            import fcntl

            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                lock_fh.close()
                return False
    except Exception:
        lock_fh.close()
        return False

    # Keep handle alive for lifetime of process.
    _LOCK_HANDLES[lock_key] = lock_fh
    with contextlib.suppress(Exception):
        lock_fh.seek(0)
        lock_fh.truncate(0)
        lock_fh.write(str(current_pid))
        lock_fh.flush()

    def _cleanup() -> None:
        try:
            fh = _LOCK_HANDLES.pop(lock_key, None)
            if fh is not None:
                with contextlib.suppress(Exception):
                    if os.name == "nt":
                        import msvcrt

                        fh.seek(0)
                        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    fh.close()
            if lock_file.exists() and lock_file.read_text(encoding="utf-8").strip() == str(current_pid):
                lock_file.unlink()
        except Exception:
            pass

    atexit.register(_cleanup)
    return True


def _ensure_mode_mutex(mode: str) -> bool:
    """Windows named mutex guard to block duplicate Atlas mode processes."""
    if os.name != "nt":
        return True
    mode_key = str(mode).strip().lower()
    if not mode_key:
        return True
    mutex_name = f"Global\\AtlasAI_{mode_key}"
    # Win32 CreateMutexW returns existing handle and sets last_error=183 if already exists.
    k32 = ctypes.windll.kernel32
    k32.SetLastError(0)
    handle = k32.CreateMutexW(None, False, mutex_name)
    if not handle:
        return False
    already_exists = k32.GetLastError() == 183
    if already_exists:
        with contextlib.suppress(Exception):
            k32.CloseHandle(handle)
        return False
    _MUTEX_HANDLES[mode_key] = handle

    def _cleanup() -> None:
        h = _MUTEX_HANDLES.pop(mode_key, None)
        if h:
            with contextlib.suppress(Exception):
                k32.ReleaseMutex(h)
            with contextlib.suppress(Exception):
                k32.CloseHandle(h)

    atexit.register(_cleanup)
    return True


def run_live(poll_seconds: int, mode_override: str = "") -> None:
    _terminate_duplicate_mode_processes("live")
    _start_duplicate_reaper("live")
    if not _ensure_mode_mutex("live"):
        logging.warning("Another live process already holds the mode mutex. Blocking duplicate startup.")
        return
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_live.pid"):
        store.log_event("system", "Another live process is already running. Skipping duplicate start.", _process_meta(mode="live", role="startup_blocked"))
        return
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_live_worker.pid"):
        store.log_event("system", "Another live worker is already running. Skipping duplicate broker start.", _process_meta(mode="live", role="worker_blocked"))
        return
    reconcile_runtime = reconcile_runtime_flags(
        store,
        settings.execute_trades,
        source_mode="live",
        mode_override=mode_override,
    )
    _record_self_improvement_runtime_guard(store, source_mode="live")
    connector = HFMConnector(settings)
    store.log_event("system", "Starting broker connection...", _process_meta(mode="live", role="broker_connect"))
    store.save_status(
        {
            "action": "connecting",
            "message": "Starting broker connection...",
            "execute_trades": bool(reconcile_runtime["execute_trades"]),
            "gemini_enabled": bool(settings.gemini_api_key),
            "symbol": settings.symbol,
            "timeframe": settings.timeframe,
            "process": _process_meta(mode="live", role="broker_connect"),
        }
    )
    try:
        connector.connect()
        run_startup_training, startup_training_meta = _should_run_startup_historical_training(settings, store)
        live_bg_training_enabled = str(os.getenv("ATLAS_ENABLE_LIVE_BACKGROUND_TRAINING", "0")).strip().lower() in {"1", "true", "yes", "on"}
        if live_bg_training_enabled:
            if run_startup_training:
                start_historical_training_background(settings, store)
            else:
                store.log_event(
                    "training",
                    "Skipping immediate historical training; promoted strategy pool is still fresh.",
                    startup_training_meta,
                )
            start_periodic_historical_training_background(settings, store)
        else:
            # TODO(py3.13/mt5-threading): move training to a fully isolated worker process before re-enabling by default.
            store.log_event(
                "training",
                "Live background historical training disabled to protect trading-loop responsiveness.",
                {"enabled": False, "reason": "live_loop_responsiveness_guard", **dict(startup_training_meta or {})},
            )
        start_periodic_housekeeping_background(settings, store)
        store.log_event("system", "Broker connected. Live loop started.", _process_meta(mode="live", role="broker_connected"))
        store.save_status(
            {
                "action": "connected",
                "message": (
                    "Broker connected. Live loop started. Historical training running in background."
                    if (live_bg_training_enabled and run_startup_training)
                    else (
                        "Broker connected. Live loop started. Historical training deferred until scheduled retrain."
                        if live_bg_training_enabled
                        else "Broker connected. Live loop started. Historical training disabled in live runtime."
                    )
                ),
                "execute_trades": bool(reconcile_runtime["execute_trades"]),
                "gemini_enabled": bool(settings.gemini_api_key),
                "symbol": settings.symbol,
                "timeframe": settings.timeframe,
                "process": _process_meta(mode="live", role="broker_connected"),
                "training": startup_training_meta,
            }
        )
    except Exception as exc:
        store.log_event("error", f"Broker startup failed: {exc}", _process_meta(mode="live", role="broker_error"))
        store.save_status(
            {
                "action": "error",
                "message": f"Broker startup failed: {exc}",
                "execute_trades": bool(reconcile_runtime["execute_trades"]),
                "gemini_enabled": bool(settings.gemini_api_key),
                "symbol": settings.symbol,
                "timeframe": settings.timeframe,
                "process": _process_meta(mode="live", role="broker_error"),
            }
        )
        connector.shutdown()
        return

    trader = AtlasTrader(settings, store=store)
    gate_env = "demo" if bool(reconcile_runtime["execute_trades"]) else "paper"
    trader.execution_gate.set_execution_environment(gate_env)
    # purpose: keep scanning and decision scoring continuous in background.
    # what it does: runs scan-only pipeline and request creation via ExecutionGate.
    scanner = MarketScanner(
        execution_gate=trader.execution_gate,
        broker_adapter=None,
        settings=settings,
    )
    scanner.start()

    # Start evolution scheduler if enabled
    evolution_stop = threading.Event()
    evolution_scheduler = start_evolution_scheduler(
        settings=settings,
        store=store,
        stop_event=evolution_stop,
    )

    heartbeat_stop = start_runtime_heartbeat(
        store,
        component="backend_trader",
        interval_seconds=float(os.getenv("ATLAS_HEARTBEAT_INTERVAL_SECONDS", "120") or 120),
        stale_after_seconds=float(os.getenv("ATLAS_HEARTBEAT_STALE_AFTER_SECONDS", "180") or 180),
    )
    startup_sanity = guarded_demo_startup_sanity(
        store=store,
        execute_trades=bool(reconcile_runtime["execute_trades"]),
        source_mode="live",
        gate_environment=gate_env,
    )
    store.save_status(
        {
            **dict(store.load_status() or {}),
            "execute_trades": bool(reconcile_runtime["execute_trades"]),
            "guarded_demo_startup_sanity": startup_sanity,
        }
    )

    try:
        while True:
            cycle_started = time.perf_counter()
            try:
                trader.step()
            except Exception as step_exc:
                store.log_event(
                    "error",
                    f"Live cycle failed: {step_exc}",
                    {
                        **_process_meta(mode="live", role="cycle_error"),
                        "cycle_duration_ms": round((time.perf_counter() - cycle_started) * 1000.0, 2),
                    },
                )
                store.save_status(
                    {
                        **dict(store.load_status() or {}),
                        "action": "degraded",
                        "message": f"Live cycle error: {step_exc}",
                        "cycle_error": {
                            "error": str(step_exc),
                            "at": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                )
            else:
                cycle_ms = round((time.perf_counter() - cycle_started) * 1000.0, 2)
                if cycle_ms > 45000.0:
                    store.log_event(
                        "risk",
                        "Live cycle exceeded latency watchdog threshold.",
                        {
                            **_process_meta(mode="live", role="cycle_watchdog"),
                            "cycle_duration_ms": cycle_ms,
                            "threshold_ms": 45000.0,
                        },
                    )
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        logging.info("Atlas AI stopped by user")
    finally:
        scanner.stop()
        heartbeat_stop.set()
        evolution_stop.set()
        stop_evolution_scheduler(evolution_scheduler)
        connector.shutdown()


def run_backtest(mode_override: str = "") -> None:
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    reconcile_runtime_flags(
        store,
        settings.execute_trades,
        source_mode="backtest",
        mode_override=mode_override,
    )
    connector = HFMConnector(settings)
    connector.connect()

    try:
        feed = MarketDataFeed(settings)
        df = feed.get_rates(3000)
        result = Backtester(store=store, settings=settings).run(df)
        print("Atlas AI Backtest")
        print(f"Total return: {result.total_return:.2%}")
        print(f"Total trades: {result.trades}")
        print(f"Wins: {result.wins}")
        print(f"Losses: {result.losses}")
        print(f"Win rate: {result.win_rate:.2%}")
        print(f"Expectancy per trade: {result.expectancy:.4%}")
        print(f"Profit factor: {result.profit_factor:.3f}")
        print(f"Average win: {result.average_win:.4%}")
        print(f"Average loss: {result.average_loss:.4%}")
        print(f"Max drawdown: {result.max_drawdown:.2%}")
        print(f"Sharpe: {result.sharpe:.3f}")
        print(f"Sortino: {result.sortino:.3f}")
        print(f"Risk of ruin: {result.risk_of_ruin:.2%}")
        print(f"Average hold bars: {result.average_hold_bars:.2f}")
    finally:
        connector.shutdown()


def run_ui_only(mode_override: str = "", comparison_root: str = "", comparison_run_id: str = "") -> None:
    if not _ensure_mode_mutex("ui"):
        logging.warning("Another ui process already holds the mode mutex. Blocking duplicate startup.")
        return
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_ui.pid"):
        store.log_event("system", "Another ui process is already running. Skipping duplicate start.", _process_meta(mode="ui", role="startup_blocked"))
        return
    try:
        reconcile_runtime_flags(
            store,
            settings.execute_trades,
            source_mode="ui",
            mode_override=mode_override,
        )
        _record_self_improvement_runtime_guard(store, source_mode="ui")
        _record_ui_process_lifecycle(store, process_role="ui_wrapper", stage="startup", status="starting")
        _prime_desktop_wrapper_state(store, stage="wrapper_starting")
        _record_ui_process_lifecycle(
            store,
            process_role="ui_wrapper",
            stage="ui_server_wait",
            status="starting",
            meta={"url": _ui_server_url(), "timeout_seconds": 25.0},
        )
        if not _ensure_ui_server_running(
            store,
            mode_override=mode_override,
            comparison_root=comparison_root,
            comparison_run_id=comparison_run_id,
            source_mode="ui",
            timeout_seconds=25.0,
        ):
            _prime_desktop_wrapper_state(
                store,
                stage="ui_server_wait_failed",
                error="durable_ui_server_not_reachable",
                frontend_mode="desktop_unavailable",
            )
            _record_ui_wrapper_exit(
                store,
                stage="ui_server_wait",
                status="failed",
                error="durable_ui_server_not_reachable",
                meta={"url": _ui_server_url(), "timeout_seconds": 25.0},
            )
            raise RuntimeError("Durable UI server did not become reachable for desktop shell attach.")
        _record_ui_process_lifecycle(store, process_role="ui_wrapper", stage="ui_server_wait", status="started")
        store.log_event("system", "Launching desktop UI shell.", _process_meta(mode="ui", role="ui_launch"))
        try:
            _prime_desktop_wrapper_state(store, stage="desktop_shell_attach", frontend_mode="desktop_attaching")
            _record_ui_process_lifecycle(store, process_role="ui_wrapper", stage="desktop_shell_attach", status="running")
            launch_desktop_ui(
                store,
                comparison_root=comparison_root or None,
                comparison_run_id=comparison_run_id or None,
                server_already_running=True,
            )
            _record_ui_wrapper_exit(
                store,
                stage="desktop_shell_attach",
                status="stopped",
                meta={"exit_reason": "desktop_shell_returned"},
            )
        except BaseException as exc:
            _prime_desktop_wrapper_state(
                store,
                stage="desktop_shell_attach_failed",
                error=str(exc),
                frontend_mode="desktop_unavailable",
            )
            _record_ui_wrapper_exit(
                store,
                stage="desktop_shell_attach",
                status="failed",
                exc=exc,
                meta={"exit_reason": "desktop_shell_exception"},
            )
            raise
    except BaseException as exc:
        if str(type(exc).__name__) != "SystemExit":
            _record_ui_wrapper_exit(
                store,
                stage="wrapper_exit",
                status="failed",
                exc=exc,
                meta={"exit_reason": "wrapper_exception_before_desktop_ready"},
            )
        raise


def run_ui_browser_only(mode_override: str = "", comparison_root: str = "", comparison_run_id: str = "") -> None:
    if not _ensure_mode_mutex("ui"):
        logging.warning("Another ui process already holds the mode mutex. Blocking duplicate startup.")
        return
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_ui.pid"):
        store.log_event("system", "Another ui process is already running. Skipping duplicate start.", _process_meta(mode="ui", role="startup_blocked"))
        return
    reconcile_runtime_flags(
        store,
        settings.execute_trades,
        source_mode="ui",
        mode_override=mode_override,
    )
    _record_self_improvement_runtime_guard(store, source_mode="ui-browser")
    _record_ui_process_lifecycle(store, process_role="ui_browser_wrapper", stage="startup", status="starting")
    if not _ensure_ui_server_running(
        store,
        mode_override=mode_override,
        comparison_root=comparison_root,
        comparison_run_id=comparison_run_id,
        source_mode="ui-browser",
        timeout_seconds=25.0,
    ):
        _record_ui_process_lifecycle(
            store,
            process_role="ui_browser_wrapper",
            stage="ui_server_wait",
            status="failed",
            error="durable_ui_server_not_reachable",
        )
        raise RuntimeError("Durable UI server did not become reachable for browser fallback attach.")
    fallback_url = "http://127.0.0.1:8765"
    flags = dict(store.load_runtime_flags() or {})
    ts = datetime.now(timezone.utc).isoformat()
    flags["ui_desktop_window"] = {
        "ok": False,
        "ready": False,
        "stage": "browser_only_server",
        "backend": "browser",
        "error": "browser-only frontend mode forced by launcher due to known desktop instability",
        "exception_type": "",
        "traceback": "",
        "classification": {
            "code": "desktop_host_accessibility_recursion",
            "category": "desktop_host_runtime",
            "operator_cause": "Desktop host hit a pywebview/native accessibility recursion failure.",
            "operator_action": "Use browser fallback mode on this machine and disable desktop shell until environment is repaired.",
        },
        "failure_reason": "desktop_host_accessibility_recursion",
        "frontend_mode": "browser_fallback",
        "desktop_window_ready": False,
        "desktop_shell_healthy": False,
        "desktop_ui_available": False,
        "fallback_active": True,
        "fallback_url": fallback_url,
        "degraded_since": ts,
        "failure_evidence": {
            "classified_reason": "desktop_host_accessibility_recursion",
            "source": "launcher_browser_only_mode",
            "matched_pattern": "ui-browser",
            "detected_at": ts,
            "fallback_triggered_reason": "launcher_browser_only_mode",
            "fallback_activated_at": ts,
        },
        "preferred_gui": "browser_only",
        "failure_count": 0,
        "failure_threshold": 2,
        "disabled_due_to_repeated_failures": False,
        "ts": ts,
        "updated_at": ts,
    }
    flags["desktop_host_stability"] = {
        "accessibility_recursion_seen": True,
        "force_browser_fallback": True,
        "last_reason": "desktop_host_accessibility_recursion",
        "last_detected_at": ts,
        "recommended_frontend_mode": "browser_fallback",
        "updated_at": ts,
    }
    store.save_runtime_flags(flags)
    store.log_event("system", "Launching browser fallback wrapper.", _process_meta(mode="ui", role="browser_only_launch"))

    def _open_browser() -> None:
        time.sleep(0.5)
        with contextlib.suppress(Exception):
            webbrowser.open(fallback_url)

    threading.Thread(target=_open_browser, daemon=True).start()
    _record_ui_process_lifecycle(
        store,
        process_role="ui_browser_wrapper",
        stage="browser_open",
        status="intentional_exit",
        meta={"fallback_url": fallback_url, "wrapper_behavior": "server_remains_durable"},
    )


def run_ui_server_only(mode_override: str = "", comparison_root: str = "", comparison_run_id: str = "") -> None:
    if not _ensure_mode_mutex("ui-server"):
        logging.warning("Another ui-server process already holds the mode mutex. Blocking duplicate startup.")
        return
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_ui_server.pid"):
        store.log_event("system", "Another ui-server process is already running. Skipping duplicate start.", _process_meta(mode="ui-server", role="startup_blocked"))
        return
    reconcile_runtime_flags(
        store,
        settings.execute_trades,
        source_mode="ui-server",
        mode_override=mode_override,
    )
    _record_self_improvement_runtime_guard(store, source_mode="ui-server")
    heartbeat_stop = start_runtime_heartbeat(
        store,
        component="ui_server",
        interval_seconds=float(os.getenv("ATLAS_UI_HEARTBEAT_INTERVAL_SECONDS", "60") or 60),
        stale_after_seconds=float(os.getenv("ATLAS_HEARTBEAT_STALE_AFTER_SECONDS", "180") or 180),
    )
    visual_watch_stop = start_visual_watch_loop(store, enabled_default=False)
    _record_ui_process_lifecycle(store, process_role="ui_server", stage="startup", status="starting")
    try:
        app = create_ui_app(
            store,
            comparison_root=comparison_root or None,
            comparison_run_id=comparison_run_id or None,
        )
        _record_ui_process_lifecycle(store, process_role="ui_server", stage="app_created", status="started")
        serve_ui_app(app, host="127.0.0.1", port=8765)
        _record_ui_process_lifecycle(store, process_role="ui_server", stage="serve_forever", status="stopped")
    except BaseException as exc:
        _record_ui_process_lifecycle(store, process_role="ui_server", stage="serve_forever", status="failed", exc=exc)
        raise
    finally:
        heartbeat_stop.set()
        visual_watch_stop.set()


def run_live_with_ui(poll_seconds: int, mode_override: str = "") -> None:
    if not _ensure_mode_mutex("live-ui"):
        logging.warning("Another live-ui process already holds the mode mutex. Blocking duplicate startup.")
        return
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    if not _ensure_single_instance(Path(settings.storage_dir) / "atlas_live_ui.pid"):
        store.log_event("system", "Another live-ui process is already running. Skipping duplicate start.", _process_meta(mode="live-ui", role="startup_blocked"))
        return
    os.environ[_LIVE_UI_OWNER_ENV] = str(os.getpid())
    reconcile_runtime_flags(
        store,
        settings.execute_trades,
        source_mode="live-ui",
        mode_override=mode_override,
    )
    _record_self_improvement_runtime_guard(store, source_mode="live-ui")
    # Start evolution scheduler if enabled
    evolution_stop = threading.Event()
    start_evolution_scheduler(
        settings=settings,
        store=store,
        stop_event=evolution_stop,
    )
    _ensure_live_worker_running(poll_seconds, store, source_mode="live-ui")
    _ensure_ui_process_running(store, source_mode="live-ui")
    store.log_event("system", "Live worker and desktop UI launch coordinated.", _process_meta(mode="live-ui", role="launch_complete"))


def run_evolution_mode(
    once: bool = False,
    generations: int = 1,
    dry_run: bool = False,
    interval_hours: float | None = None,
) -> None:
    """Offline/manual evolution run.

    Loads market data, runs one or more evolution generations,
    writes reporter artifacts, and prints a summary.
    """
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    store.log_event("evolution", "Starting offline evolution run.", {
        "once": once,
        "generations": generations,
        "dry_run": dry_run,
        "interval_hours": interval_hours,
    })

    stop_event = threading.Event()
    scheduler = EvolutionScheduler(
        settings=settings,
        store=store,
        stop_event=stop_event,
        interval_hours=interval_hours,
        dry_run=dry_run,
    )

    try:
        if once:
            summary = scheduler.run_once()
            print("Atlas AI Evolution (single cycle)")
            print(f"Generation: {summary.get('generation')}")
            print(f"Population size: {summary.get('population_size')}")
            print(f"Best fitness: {summary.get('best_fitness', 0.0):.2f}")
            print(f"Best UID: {summary.get('best_uid', '')}")
            print(f"Promotions: {summary.get('promotions', 0)}")
            print(f"Rejections: {summary.get('rejections', 0)}")
            print(f"Active demo: {summary.get('active_demo', 0)}")
            print(f"Active shadow: {summary.get('active_shadow', 0)}")
            print(f"Archived: {summary.get('archived', 0)}")
            print(f"Dry run: {summary.get('dry_run', False)}")
            print(f"Duration: {summary.get('duration_seconds', 0):.2f}s")
            next_run = scheduler.state.next_evolution_run_at
            print(f"Next scheduled run: {next_run or 'N/A (manual)'}")
        else:
            # Continuous scheduled evolution
            scheduler.start()
            print(f"Atlas AI Evolution Scheduler started.")
            print(f"Interval: {scheduler.interval_seconds / 3600:.1f} hours")
            print(f"Generations target: {generations}")
            print(f"Dry run: {dry_run}")
            print("Press Ctrl+C to stop.")
            try:
                while scheduler.state.total_runs < generations:
                    time.sleep(1.0)
                    if stop_event.is_set():
                        break
            except KeyboardInterrupt:
                print("\nStopping evolution scheduler...")
            finally:
                scheduler.shutdown()
                final = scheduler.state
                print("\nAtlas AI Evolution Summary")
                print(f"Total runs: {final.total_runs}")
                print(f"Current generation: {final.current_generation}")
                print(f"Best fitness: {final.best_fitness:.2f}")
                print(f"Best UID: {final.best_uid}")
                print(f"Active demo strategies: {final.active_demo_count}")
                print(f"Last error: {final.last_error or 'none'}")
    except Exception as exc:
        store.log_event("evolution", f"Offline evolution failed: {exc}", {"error": str(exc)})
        print(f"Evolution failed: {exc}")
        raise


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atlas AI Gold Trading System")
    parser.add_argument(
        "mode",
        choices=["live", "backtest", "ui", "ui-server", "ui-browser", "live-ui", "paper-compare", "experiment-report", "readiness-report", "fast-path-audit-report", "signal-correlation-report", "label-report", "label-review", "setup-quality-report", "edge-building-report", "evolution"],
        help="Execution mode",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Live loop interval")
    parser.add_argument("--comparison-root", type=str, default="", help="Optional root directory for paper comparison runs")
    parser.add_argument("--comparison-run-id", type=str, default="", help="Explicit run id for paper comparison")
    parser.add_argument("--resume", action="store_true", help="Resume an existing paper comparison run id")
    parser.add_argument("--allow-market-closed", action="store_true", help="Bypass paper comparison market-open readiness check")
    parser.add_argument(
        "--paper-preset",
        type=str,
        default="conservative_filter_rollout_v1",
        help="Paper comparison rollout preset",
    )
    parser.add_argument(
        "--mode-override",
        choices=["env", "paper", "live"],
        default="",
        help="Resolve execute_trades startup mismatches explicitly",
    )
    parser.add_argument("--once", action="store_true", help="Run a single evolution cycle and exit (evolution mode only)")
    parser.add_argument("--generations", type=int, default=1, help="Number of evolution generations to run (evolution mode only)")
    parser.add_argument("--dry-run", action="store_true", help="Run evolution without persisting mutations (evolution mode only)")
    parser.add_argument("--interval-hours", type=float, default=0.0, help="Override evolution interval hours (evolution mode only)")
    parser.add_argument("--fast-path-audit-window-hours", type=float, default=168.0, help="Lookback window for fast-path-audit-report")
    parser.add_argument("--signal-correlation-window", type=int, default=500, help="Rolling row window for signal-correlation-report")
    parser.add_argument("--signal-correlation-threshold", type=float, default=0.8, help="Redundancy threshold for signal-correlation-report")
    return parser.parse_args(argv)


def main() -> None:
    _append_startup_trace("main_enter")
    args = parse_args(sys.argv[1:])
    if os.name == "nt" and _mode_requires_venv(args.mode) and not _is_venv_python() and not _allow_non_venv_main():
        _append_startup_trace("non_venv_mode_exit", {"mode": args.mode})
        os._exit(0)
    if args.mode == "live-ui" and _is_spawned_live_ui_child():
        _append_startup_trace("spawned_live_ui_child_exit", {"mode": args.mode})
        os._exit(0)
    if _should_exit_non_venv_duplicate():
        _append_startup_trace("non_venv_duplicate_exit", {"mode": args.mode})
        os._exit(0)
    setup_logging()

    if args.mode == "live":
        run_live(args.poll_seconds, mode_override=args.mode_override)
    elif args.mode == "backtest":
        run_backtest(mode_override=args.mode_override)
    elif args.mode == "paper-compare":
        settings = replace(load_settings(), execute_trades=False)
        store = KnowledgeStore(settings.storage_dir)
        ensure_runtime_flags(store, False)
        connector = HFMConnector(settings)
        connector_connected = False
        try:
            connector.connect()
            connector_connected = True
        except RuntimeError as exc:
            if "Another Atlas broker session is already active for this account" not in str(exc):
                raise
            logging.warning(
                "Paper-compare is continuing without broker session ownership because another Atlas broker session is active. "
                "This is allowed only for the read-only paper-compare lane."
            )
            store.log_event(
                "system",
                "Paper-compare continuing without broker session ownership; existing broker session is active.",
                {"mode": "paper-compare", "execute_trades": False, "reason": "broker_session_already_active"},
            )
        try:
            run_shadow_paper_comparison(
                settings,
                poll_seconds=args.poll_seconds,
                comparison_root=args.comparison_root or None,
                run_id=args.comparison_run_id or None,
                resume=bool(args.resume),
                require_market_open=not bool(args.allow_market_closed),
                preset=args.paper_preset,
            )
        finally:
            if connector_connected:
                connector.shutdown()
    elif args.mode == "ui":
        run_ui_only(
            mode_override=args.mode_override,
            comparison_root=args.comparison_root,
            comparison_run_id=args.comparison_run_id,
        )
    elif args.mode == "ui-server":
        run_ui_server_only(
            mode_override=args.mode_override,
            comparison_root=args.comparison_root,
            comparison_run_id=args.comparison_run_id,
        )
    elif args.mode == "ui-browser":
        run_ui_browser_only(
            mode_override=args.mode_override,
            comparison_root=args.comparison_root,
            comparison_run_id=args.comparison_run_id,
        )
    elif args.mode == "experiment-report":
        settings = load_settings()
        store = KnowledgeStore(settings.storage_dir)
        result = run_meta_model_experiment(settings, store)
        print("Atlas AI Experiment Report")
        print(f"ok: {bool(result.get('ok', False))}")
        if result.get("ok", False):
            print(f"run_id: {result.get('run_id')}")
            print(f"samples_total: {result.get('samples_total')}")
            print(f"train_samples: {result.get('train_samples')}")
            print(f"test_samples: {result.get('test_samples')}")
            print(f"dataset_source: {result.get('dataset_source', 'unknown')}")
            metrics = dict(result.get("metrics", {}) or {})
            print(f"accuracy: {float(metrics.get('accuracy', 0.0)):.4f}")
            print(f"brier: {float(metrics.get('brier', 1.0)):.4f}")
            walk = dict(result.get("walk_forward", {}) or {})
            print(f"walk_forward_folds: {int(walk.get('fold_count', 0))}")
            print(f"walk_forward_accuracy: {float(walk.get('avg_accuracy', 0.0)):.4f}")
            print(f"walk_forward_brier: {float(walk.get('avg_brier', 1.0)):.4f}")
            promotion = dict(result.get("promotion", {}) or {})
            print(f"promoted: {bool(promotion.get('promoted', False))}")
            print(f"cooldown_active: {bool(promotion.get('cooldown_active', False))}")
        else:
            print(f"reason: {result.get('reason', 'unknown')}")
    elif args.mode == "readiness-report":
        settings = load_settings()
        store = KnowledgeStore(settings.storage_dir)
        report = readiness_report(store)
        print("Atlas AI Readiness Report")
        print(f"decision_samples: {int(report.get('decision_samples', 0))}")
        print(f"closed_outcomes: {int(report.get('closed_outcomes', 0))}")
        print(f"joined_samples: {int(report.get('joined_samples', 0))}")
        print(f"joined_training_ready: {bool(report.get('joined_training_ready', False))}")
        print(f"samples_needed_for_joined_training: {int(report.get('samples_needed_for_joined_training', 0))}")
        print(f"meta_model_dataset_source: {report.get('meta_model_dataset_source', 'untrained')}")
        print(f"meta_model_status: {report.get('meta_model_status', 'untrained')}")
    elif args.mode == "fast-path-audit-report":
        settings = load_settings()
        store = KnowledgeStore(settings.storage_dir)
        report = build_fast_path_promotion_audit_report(
            store,
            window_hours=max(0.0, float(args.fast_path_audit_window_hours)),
            write_file=True,
        )
        print("Atlas AI Fast-Path Promotion Audit")
        print(f"report_path: {report.get('report_path', '')}")
        print(f"window_hours: {float(report.get('window_hours', 0.0)):.1f}")
        print(f"observation_count: {int(report.get('observation_count', 0))}")
        print(f"evidence_count: {int(report.get('evidence_count', 0))}")
        transitions = dict(report.get("transitions", {}) or {})
        print(f"slow_to_medium: {int(transitions.get('slow_to_medium', 0))}")
        print(f"medium_to_fast: {int(transitions.get('medium_to_fast', 0))}")
        print(f"slow_to_fast: {int(transitions.get('slow_to_fast', 0))}")
        promotion = dict(report.get("promotion", {}) or {})
        print(f"promotion_candidate_events: {int(promotion.get('promotion_candidate_events', 0))}")
        print(f"promoted_events: {int(promotion.get('promoted_events', 0))}")
        print(f"summary: {report.get('summary', '')}")
    elif args.mode == "signal-correlation-report":
        settings = replace(load_settings(), execute_trades=False)
        store = KnowledgeStore(settings.storage_dir)
        report = build_signal_correlation_report_from_store(
            store,
            rolling_window=max(1, int(args.signal_correlation_window)),
            threshold=max(0.0, min(1.0, float(args.signal_correlation_threshold))),
            write_files=True,
        )
        print("Atlas AI Signal Correlation Report")
        print(f"report_path: {str(Path(settings.storage_dir) / 'signal_correlation_report.json')}")
        print(f"matrix_path: {str(Path(settings.storage_dir) / 'signal_correlation_matrix.csv')}")
        print(f"rolling_window: {int(report.get('rolling_window', 0))}")
        print(f"threshold: {float(report.get('threshold', 0.0)):.2f}")
        print(f"source_rows: {int(report.get('source_rows', 0))}")
        print(f"usable_rows: {int(report.get('usable_rows', 0))}")
        print(f"strategy_count: {len(list(report.get('strategies', []) or []))}")
        print(f"flagged_pairs: {len(list(report.get('flagged_pairs', []) or []))}")
        print(f"summary: {report.get('summary', '')}")
    elif args.mode == "label-report":
        settings = replace(load_settings(), execute_trades=False)
        result = run_trade_labeling(settings)
        print("Atlas AI Trade Label Report")
        print(f"ok: {bool(result.get('ok', False))}")
        if result.get("ok", False):
            print(f"dataset_path: {result.get('dataset_path')}")
            summary = dict(result.get("summary_by_label", {}) or {})
            for label, metrics in summary.items():
                print(
                    f"{label}: count={int(metrics.get('count', 0))}, "
                    f"win_rate={float(metrics.get('win_rate', 0.0)):.2%}, "
                    f"expectancy={float(metrics.get('expectancy', 0.0)):.4%}"
                )
        else:
            print(f"reason: {result.get('reason', 'unknown')}")
    elif args.mode == "label-review":
        settings = replace(load_settings(), execute_trades=False)
        label_path = Path(settings.storage_dir) / "trade_labels_report.json"
        result = build_label_review(label_path)
        print("Atlas AI Label Review")
        print(f"report_path: {str(Path(label_path).with_name('trade_label_review.json'))}")
        overview = dict(result.get("label_bucket_overview", {}) or {})
        for label, meta in overview.items():
            print(
                f"{label}: count={int(meta.get('count', 0))}, "
                f"suspicious_examples={int(meta.get('suspicious_example_count', 0))}"
            )
    elif args.mode == "setup-quality-report":
        settings = replace(load_settings(), execute_trades=False)
        label_path = Path(settings.storage_dir) / "trade_labels_report.json"
        review_path = Path(settings.storage_dir) / "trade_label_review.json"
        result = build_offline_setup_quality_score(label_path, review_report_path=review_path)
        print("Atlas AI Setup Quality Report")
        print(f"ok: {bool(result.get('ok', False))}")
        if result.get("ok", False):
            print(f"report_path: {str(Path(label_path).with_name('trade_setup_quality_report.json'))}")
            for label, metrics in dict(result.get("summary_by_label", {}) or {}).items():
                print(
                    f"{label}: count={int(metrics.get('count', 0))}, "
                    f"avg_score={float(metrics.get('average_score', 0.0)):.4f}"
                )
        else:
            print(f"reason: {result.get('reason', 'unknown')}")
    elif args.mode == "edge-building-report":
        settings = replace(load_settings(), execute_trades=False)
        label_path = Path(settings.storage_dir) / "trade_labels_report.json"
        result = build_edge_building_report(label_path)
        print("Atlas AI Edge Building Report")
        print(f"ok: {bool(result.get('ok', False))}")
        if result.get("ok", False):
            print(f"report_path: {str(Path(label_path).with_name('edge_building_report.json'))}")
            experiment = dict(result.get("filter_experiment", {}) or {})
            control = dict(experiment.get("control", {}) or {})
            filtered = dict(experiment.get("experiment", {}) or {})
            print(
                "control: "
                f"trades={int(control.get('trades', 0))}, "
                f"win_rate={float(control.get('win_rate', 0.0)):.2%}, "
                f"expectancy={float(control.get('expectancy', 0.0)):.4%}"
            )
            print(
                "experiment: "
                f"trades={int(filtered.get('trades', 0))}, "
                f"win_rate={float(filtered.get('win_rate', 0.0)):.2%}, "
                f"expectancy={float(filtered.get('expectancy', 0.0)):.4%}"
            )
            print(
                f"trades_removed={int(experiment.get('trades_removed', 0))}, "
                f"filter_precision={float(experiment.get('filter_precision', 0.0)):.2%}"
            )
        else:
            print(f"reason: {result.get('reason', 'unknown')}")
    elif args.mode == "evolution":
        run_evolution_mode(
            once=args.once,
            generations=args.generations,
            dry_run=args.dry_run,
            interval_hours=args.interval_hours if args.interval_hours > 0 else None,
        )
    else:
        run_live_with_ui(args.poll_seconds, mode_override=args.mode_override)


if __name__ == "__main__":
    main()
