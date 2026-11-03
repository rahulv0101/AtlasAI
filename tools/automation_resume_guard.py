from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "storage" / "automation_state.json"
FLAGS_PATH = ROOT / "storage" / "runtime_flags.json"
LOG_PATH = ROOT / "storage" / "monitoring" / "atlas_automation_resume_guard.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _log(event: str, **payload: Any) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def enforce_once() -> None:
    flags = _read_json(FLAGS_PATH)
    if not bool(flags.get("automation_resume_sticky", False)):
        return

    state = _read_json(STATE_PATH)
    payload = dict(state.get("payload", {}) or {})
    reason = str(payload.get("automation_pause_reason", "") or "")
    kill_active = bool(dict(payload.get("kill_switch", {}) or {}).get("active", False))
    sleep_active = bool(payload.get("sleep_mode_active", False))
    policy_id = str(payload.get("active_automation_policy_id", "") or payload.get("automation_explicit_resume_policy_id", "") or "p1").strip()

    if (
        bool(payload.get("automation_paused", False))
        and reason == "unclean_shutdown_forced_pause"
        and policy_id
        and not kill_active
        and not sleep_active
    ):
        now = time.time()
        payload["automation_paused"] = False
        payload["automation_pause_reason"] = ""
        payload["automation_explicit_resume_active"] = True
        payload["automation_explicit_resume_at"] = now
        payload["automation_explicit_resume_policy_id"] = policy_id
        pilot = dict(payload.get("tiny_live_pilot", {}) or {})
        if str(pilot.get("pause_reason", "") or "") in {"", "unclean_shutdown_forced_pause", "operator_requested"}:
            pilot["paused"] = False
            pilot["pause_reason"] = ""
            payload["tiny_live_pilot"] = pilot
        payload["saved_at"] = now
        state["payload"] = payload
        state["event"] = "automation_resume_guard_repaired_unclean_pause"
        state["saved_at"] = now
        state["revision"] = int(state.get("revision", 0) or 0) + 1
        _write_json(STATE_PATH, state)
        _log("repaired_unclean_shutdown_pause", policy_id=policy_id)
        return

    pilot = dict(payload.get("tiny_live_pilot", {}) or {})
    pilot_reason = str(pilot.get("pause_reason", "") or "")
    if (
        not bool(payload.get("automation_paused", False))
        and bool(payload.get("automation_explicit_resume_active", False))
        and bool(pilot.get("paused", False))
        and pilot_reason in {"weekend_learning_only_operator_requested", "operator_requested", "operator_pause"}
        and not kill_active
        and not sleep_active
    ):
        now = time.time()
        pilot["paused"] = False
        pilot["pause_reason"] = ""
        payload["tiny_live_pilot"] = pilot
        payload["saved_at"] = now
        state["payload"] = payload
        state["event"] = "automation_resume_guard_repaired_pilot_pause"
        state["saved_at"] = now
        state["revision"] = int(state.get("revision", 0) or 0) + 1
        _write_json(STATE_PATH, state)
        _log("repaired_stale_pilot_pause", previous_reason=pilot_reason)


def main() -> None:
    _log("guard_started")
    while True:
        enforce_once()
        time.sleep(5)


if __name__ == "__main__":
    main()
