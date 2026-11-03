from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path


# Purpose: start the Atlas full stack without creating a visible cmd window.
ROOT = Path(__file__).resolve().parent
VENV_ROOT = ROOT / ".venv"
LOG_PATH = ROOT / "storage" / "atlas_fullstack_startup.log"


def _hidden_creationflags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _bootstrap_venv_imports() -> None:
    # Purpose: allow this launcher to run under base pythonw.exe while preserving venv packages.
    if str(getattr(sys, "prefix", "")) != str(VENV_ROOT):
        site_packages = VENV_ROOT / "Lib" / "site-packages"
        if site_packages.exists():
            sys.path.insert(0, str(site_packages))
    scripts_dir = VENV_ROOT / "Scripts"
    os.environ.setdefault("VIRTUAL_ENV", str(VENV_ROOT))
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )


def _preferred_pythonw() -> str:
    # Purpose: resolve a stable pythonw.exe path for Atlas background children.
    current = Path(sys.executable)
    if current.name.lower() == "pythonw.exe" and current.exists():
        return str(current)
    candidate = VENV_ROOT / "Scripts" / "pythonw.exe"
    if candidate.exists():
        return str(candidate)
    base = str(getattr(sys, "_base_executable", "") or "").strip()
    if base:
        pyw = Path(base.replace("python.exe", "pythonw.exe"))
        if pyw.exists():
            return str(pyw)
    return str(candidate)


def _is_core_stack_running() -> bool:
    # Purpose: avoid duplicate relaunch when live and UI are already up.
    ps_cmd = (
        "$live = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and "
        "$_.CommandLine -match '-m\\s+atlas_ai\\.main\\s+live' }; "
        "$ui = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and "
        "$_.CommandLine -match '-m\\s+atlas_ai\\.main\\s+ui(\\s|$)' }; "
        "if ($live -and $ui) { exit 0 } else { exit 1 }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=_hidden_creationflags(),
        )
        return int(result.returncode) == 0
    except Exception:
        return False


def _is_audio_path_running() -> bool:
    # Purpose: detect an existing Atlas listener/watchdog before spawning another one.
    ps_cmd = (
        "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and "
        "(($_.CommandLine -match '-m\\s+atlas_ai\\.atlas_audio_listener(\\s|$)') -or "
        "($_.CommandLine -match 'atlas_audio_watchdog\\.py')) }; "
        "if ($p) { exit 0 } else { exit 1 }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=_hidden_creationflags(),
        )
        return int(result.returncode) == 0
    except Exception:
        return False


def _ensure_audio_path() -> None:
    # Purpose: keep the listener watchdog running without using a cmd wrapper.
    if _is_audio_path_running():
        logging.info("Atlas audio path already running.")
        return
    cmd = [_preferred_pythonw(), str(ROOT / "atlas_audio_watchdog.py")]
    kwargs: dict[str, object] = {
        "cwd": str(ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": dict(os.environ),
    }
    if os.name == "nt":
        creationflags = 0
        with contextlib.suppress(Exception):
            creationflags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
        with contextlib.suppress(Exception):
            creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if creationflags:
            kwargs["creationflags"] = creationflags
    subprocess.Popen(cmd, **kwargs)
    logging.info("Started Atlas audio watchdog without cmd wrapper.")


_bootstrap_venv_imports()

from atlas_ai.atlas_full_launcher import run_full_launch
from atlas_ai.atlas_wake_config import load_atlas_wake_config


def main() -> int:
    # Purpose: launch Atlas full stack in background mode and write status to a log file.
    os.chdir(str(ROOT))
    _setup_logging()
    if not (VENV_ROOT / "Scripts" / "python.exe").exists():
        logging.error("Atlas startup failed: .venv\\Scripts\\python.exe not found.")
        return 1

    if _is_core_stack_running():
        logging.info("Atlas core stack already running. Skipping relaunch.")
        _ensure_audio_path()
        return 0

    logging.info("Starting Atlas full stack via no-console guarded launcher.")
    config = load_atlas_wake_config()
    summary = run_full_launch(config)
    overall = str(dict(summary.get("health", {}) or {}).get("overall_result", "")).strip().upper()
    logging.info("Atlas full stack launcher result=%s", overall or "UNKNOWN")
    _ensure_audio_path()
    with contextlib.suppress(Exception):
        result_path = ROOT / "storage" / "atlas_fullstack_startup_result.json"
        result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if overall in {"SUCCESS", "DEGRADED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
