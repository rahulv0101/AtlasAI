from __future__ import annotations

import json
import os
import tempfile
import time
import contextlib
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


class AutomationStateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextlib.contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+b") as lock_fh:
            if os.name == "nt":
                import msvcrt

                while True:
                    try:
                        lock_fh.seek(0)
                        msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
                        break
                    except OSError:
                        time.sleep(0.01)
            else:
                import fcntl

                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt

                    lock_fh.seek(0)
                    msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def load(self) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._locked():
            if not self.path.exists():
                return {}, {"ok": False, "reason": "missing_state_file"}
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {}, {"ok": False, "reason": f"corrupt_json:{exc.__class__.__name__}"}

            if not isinstance(raw, dict):
                return {}, {"ok": False, "reason": "invalid_root_type"}
            if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
                return {}, {"ok": False, "reason": "schema_version_mismatch"}
            payload = dict(raw.get("payload", {}) or {})
            if not isinstance(payload, dict):
                return {}, {"ok": False, "reason": "invalid_payload_type"}
            return payload, {
                "ok": True,
                "reason": "loaded",
                "saved_at": raw.get("saved_at"),
                "revision": int(raw.get("revision", 0) or 0),
            }

    def _runtime_flags(self) -> dict[str, Any]:
        candidates = [self.path.parent / "runtime_flags.json"]
        if not os.getenv("PYTEST_CURRENT_TEST"):
            candidates.append(Path.cwd() / "storage" / "runtime_flags.json")
        for candidate in candidates:
            try:
                if candidate.exists():
                    raw = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        return raw
            except Exception:
                continue
        return {}

    def _apply_sticky_resume_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = dict(payload or {})
        flags = self._runtime_flags()
        sticky_resume = bool(flags.get("automation_resume_sticky", False))
        explicit_resume_active = bool(payload.get("automation_explicit_resume_active", False))
        automation_paused = bool(payload.get("automation_paused", False))
        resume_protected = sticky_resume or (explicit_resume_active and not automation_paused)
        if not resume_protected:
            return payload

        kill_switch = payload.get("kill_switch", {}) or {}
        if not isinstance(kill_switch, dict):
            kill_switch = {}
        kill_active = bool(kill_switch.get("active", False))
        sleep_active = bool(payload.get("sleep_mode_active", False))
        if kill_active or sleep_active:
            return payload

        policy_id = str(
            payload.get("automation_explicit_resume_policy_id")
            or payload.get("active_automation_policy_id")
            or "p1"
        ).strip()
        if not policy_id:
            return payload

        pause_reason = str(payload.get("automation_pause_reason", "") or "")
        if automation_paused and pause_reason == "unclean_shutdown_forced_pause":
            payload["automation_paused"] = False
            payload["automation_pause_reason"] = ""
            payload["automation_explicit_resume_active"] = True
            payload["automation_explicit_resume_policy_id"] = policy_id
            payload["automation_explicit_resume_at"] = time.time()

        stale_pilot_reasons = {
            "operator_pause",
            "operator_requested",
            "weekend_learning_only_operator_requested",
        }
        pilot = payload.get("tiny_live_pilot", {}) or {}
        if isinstance(pilot, dict) and str(pilot.get("pause_reason", "") or "") in stale_pilot_reasons:
            pilot = dict(pilot)
            pilot["paused"] = False
            pilot["pause_reason"] = ""
            payload["tiny_live_pilot"] = pilot

        return payload

    def save(self, payload: dict[str, Any], *, event: str = "state_save") -> dict[str, Any]:
        target = self.path
        with self._locked():
            previous_saved_at = None
            previous_revision = 0
            if target.exists():
                try:
                    current = json.loads(target.read_text(encoding="utf-8"))
                    previous_saved_at = current.get("saved_at")
                    previous_revision = int(current.get("revision", 0) or 0)
                except Exception:
                    previous_saved_at = None
                    previous_revision = 0
            clean_payload = self._apply_sticky_resume_policy(dict(payload or {}))
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "saved_at": time.time(),
                "previous_saved_at": previous_saved_at,
                "revision": previous_revision + 1,
                "event": str(event or "state_save"),
                "payload": clean_payload,
            }
            fd, tmp = tempfile.mkstemp(prefix="atlas_state_", suffix=".json", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(envelope, fh, ensure_ascii=True, indent=2, sort_keys=True)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, target)
                return {
                    "ok": True,
                    "reason": "saved",
                    "saved_at": envelope["saved_at"],
                    "revision": envelope["revision"],
                }
            except Exception as exc:
                try:
                    os.remove(tmp)
                except Exception:
                    pass
                return {"ok": False, "reason": f"save_failed:{exc.__class__.__name__}"}
