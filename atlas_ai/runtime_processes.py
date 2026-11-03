"""Cheap, read-only process inventory for UI and service supervision."""
from __future__ import annotations

try:
    import psutil
except ImportError:
    psutil = None


def python_process_exists(pid: int) -> bool | None:
    """None means native inspection is unavailable, not that a process is absent."""
    if psutil is None:
        return None
    try:
        return psutil.Process(pid).name().lower() in {"python.exe", "pythonw.exe", "python", "python3"}
    except psutil.NoSuchProcess:
        return False


def atlas_mode_processes(mode: str) -> list[dict] | None:
    if psutil is None:
        return None
    rows = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            if str(process.info.get("name") or "").lower() not in {"python.exe", "pythonw.exe", "python", "python3"}:
                continue
            command = process.cmdline()
            if not any(command[i:i + 3] == ["-m", "atlas_ai.main", mode] for i in range(len(command) - 2)):
                continue
            rows.append({"ProcessId": process.pid, "ParentProcessId": process.ppid(),
                         "Name": process.info["name"], "CommandLine": " ".join(command)})
        except psutil.NoSuchProcess:
            continue
        # AccessDenied must propagate: an unreadable process is not evidence
        # that no worker exists, and must never authorize a duplicate launch.
    return rows
