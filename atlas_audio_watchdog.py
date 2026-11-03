from __future__ import annotations

import logging
import os
import ctypes
import sys
import time
from pathlib import Path


# Purpose: run Atlas audio listener continuously without a visible cmd window.
# This file is intended to be launched by Task Scheduler via pythonw.exe.
ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "logs" / "audio_listener_watchdog.log"
VENV_ROOT = ROOT / ".venv"


def _bootstrap_venv_imports() -> None:
    # Purpose: allow the watchdog to run under base pythonw.exe without losing venv packages.
    if str(getattr(sys, "prefix", "")) != str(VENV_ROOT):
        site_packages = VENV_ROOT / "Lib" / "site-packages"
        if site_packages.exists():
            sys.path.insert(0, str(site_packages))
    scripts_dir = VENV_ROOT / "Scripts"
    os.environ.setdefault("VIRTUAL_ENV", str(VENV_ROOT))
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")


_bootstrap_venv_imports()

from atlas_ai.atlas_audio_listener import main as listener_main


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8")],
    )


def _listener_mutex_exists() -> bool:
    # Purpose: detect an already-running voice loop without spawning another loop attempt.
    if os.name != "nt":
        return False
    kernel32 = ctypes.windll.kernel32
    for name in ("Local\\AtlasAI_VoiceLoop", "AtlasAI_VoiceLoop"):
        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            continue
        try:
            if kernel32.GetLastError() == 183:
                return True
        finally:
            kernel32.CloseHandle(handle)
    return False


def _watchdog_mutex() -> object | None:
    # Purpose: ensure we never run more than one watchdog loop at a time.
    if os.name != "nt":
        return None
    kernel32 = ctypes.windll.kernel32
    for name in ("Local\\AtlasAI_AudioWatchdog", "AtlasAI_AudioWatchdog"):
        kernel32.SetLastError(0)
        handle = kernel32.CreateMutexW(None, False, name)
        if not handle:
            continue
        if kernel32.GetLastError() == 183:
            kernel32.CloseHandle(handle)
            return False
        return handle
    return None


def run_forever() -> int:
    # Keep the project root as cwd so listener/storage paths remain stable.
    os.chdir(str(ROOT))
    _setup_logging()
    guard = _watchdog_mutex()
    if guard is False:
        logging.info("Atlas audio watchdog already running; exiting duplicate watchdog instance.")
        return 0
    delay = float(str(os.getenv("ATLAS_AUDIO_WATCHDOG_RESTART_SEC", "3")).strip() or "3")
    delay = max(1.0, min(30.0, delay))
    logging.info("Atlas audio watchdog started. restart_delay_sec=%.1f", delay)
    while True:
        if _listener_mutex_exists():
            # Another listener process is already alive; keep watchdog idle.
            time.sleep(max(5.0, delay))
            continue
        try:
            code = int(listener_main([]) or 0)
            logging.warning("Atlas audio listener exited with code=%s; restarting in %.1fs", code, delay)
        except Exception:
            logging.exception("Atlas audio listener crashed; restarting in %.1fs", delay)
        time.sleep(delay)


if __name__ == "__main__":
    raise SystemExit(run_forever())
