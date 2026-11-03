from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FLAGS = ROOT / "storage" / "runtime_flags.json"
AUTOMATION_STATE = ROOT / "storage" / "automation_state.json"
LOG = ROOT / "storage" / "monitoring" / "atlas_weekend_learning_enforcer.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _log(event: str, **payload: Any) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _live_worker_pids() -> list[int]:
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -match 'atlas_ai[.]main live' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", script], text=True, cwd=str(ROOT))
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _stop_pid(pid: int) -> None:
    subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"], cwd=str(ROOT), check=False)


def enforce_once() -> None:
    flags = _read_json(RUNTIME_FLAGS)
    weekend = bool(flags.get("weekend_learning_only", False))
    if not weekend:
        return

    changed_flags = False
    if bool(flags.get("execute_trades", False)):
        flags["execute_trades"] = False
        changed_flags = True
    if bool(flags.get("gemini_influence_enabled", False)):
        flags["gemini_influence_enabled"] = False
        changed_flags = True
    if changed_flags:
        _write_json(RUNTIME_FLAGS, flags)
        _log("runtime_flags_reenforced", execute_trades=False, gemini_influence_enabled=False)

    state = _read_json(AUTOMATION_STATE)
    payload = dict(state.get("payload", {}) or {})
    changed_auto = False
    if not bool(payload.get("automation_paused", False)):
        payload["automation_paused"] = True
        changed_auto = True
    if str(payload.get("automation_pause_reason", "") or "") != "weekend_learning_only_operator_requested":
        payload["automation_pause_reason"] = "weekend_learning_only_operator_requested"
        changed_auto = True
    if bool(payload.get("automation_explicit_resume_active", False)):
        payload["automation_explicit_resume_active"] = False
        payload["automation_explicit_resume_at"] = None
        payload["automation_explicit_resume_policy_id"] = ""
        changed_auto = True
    pilot = payload.get("tiny_live_pilot")
    if isinstance(pilot, dict):
        if not bool(pilot.get("paused", False)):
            pilot["paused"] = True
            changed_auto = True
        if str(pilot.get("pause_reason", "") or "") != "weekend_learning_only_operator_requested":
            pilot["pause_reason"] = "weekend_learning_only_operator_requested"
            changed_auto = True
    if changed_auto:
        payload["saved_at"] = time.time()
        state["payload"] = payload
        state["event"] = "weekend_learning_only_enforced"
        state["saved_at"] = time.time()
        state["revision"] = int(state.get("revision", 0) or 0) + 1
        _write_json(AUTOMATION_STATE, state)
        _log("automation_state_reenforced", automation_paused=True)

    pids = _live_worker_pids()
    for pid in pids:
        _stop_pid(pid)
        _log("live_worker_stopped", pid=pid)


def main() -> None:
    _log("enforcer_started")
    while True:
        enforce_once()
        time.sleep(30)


if __name__ == "__main__":
    main()
