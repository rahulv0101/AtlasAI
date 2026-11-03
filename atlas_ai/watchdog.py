from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from atlas_ai.memory.knowledge_store import KnowledgeStore


class SystemWatchdog:
    PROCESSES = [
        {"name": "AtlasMain", "module": "atlas_ai.main", "args": ["live"], "critical": True},
        {"name": "AtlasAudio", "module": "atlas_ai.audio_listener_bridge", "args": [], "critical": False},
    ]

    def __init__(self) -> None:
        base = Path(__file__).resolve().parents[1]
        self.cwd = str(base)
        self.store = KnowledgeStore(str(base / "storage"))
        log_dir = base / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_dir / "watchdog.log", encoding="utf-8"), logging.StreamHandler()],
        )

    def _is_running(self, marker: str) -> bool:
        import subprocess as sp

        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-CimInstance Win32_Process | Where-Object {{ $_.Name -match 'python' -and $_.CommandLine -match '{marker}' }} | Measure-Object | Select-Object -ExpandProperty Count",
        ]
        kwargs = {
            "capture_output": True,
            "text": True,
            "timeout": 10,
        }
        if os.name == "nt":
            kwargs["creationflags"] = int(getattr(sp, "CREATE_NO_WINDOW", 0))
        out = sp.run(cmd, **kwargs)
        try:
            return int((out.stdout or "0").strip() or "0") > 0
        except Exception:
            return False

    def _restart(self, proc: dict) -> None:
        for _ in range(3):
            # Keep watchdog restarts on the same interpreter lineage to avoid duplicate runtime chains.
            python_exe = self._preferred_python()
            args = [python_exe, "-m", str(proc["module"])] + list(proc.get("args", []))
            env = os.environ.copy()
            env["ATLAS_PINNED_PYTHON"] = python_exe
            subprocess.Popen(args, cwd=self.cwd, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            time.sleep(5.0)
            if self._is_running(str(proc["module"])):
                logging.info("%s restarted successfully", proc["name"])
                return
        logging.critical("%s failed to restart after 3 attempts", proc["name"])

    def _preferred_python(self) -> str:
        pinned = str(os.getenv("ATLAS_PINNED_PYTHON", "") or "").strip()
        if pinned and Path(pinned).exists():
            return pinned
        candidate = Path(self.cwd) / ".venv" / "Scripts" / "python.exe"
        if candidate.exists():
            return str(candidate)
        return str(sys.executable)

    def watch_forever(self) -> None:
        while True:
            for proc in self.PROCESSES:
                marker = str(proc["module"])
                if not self._is_running(marker):
                    logging.warning("Process %s not found — restarting...", proc["name"])
                    self._restart(proc)
            time.sleep(15.0)


def main() -> int:
    SystemWatchdog().watch_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
