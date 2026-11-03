from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from atlas_ai.atlas_wake_config import load_atlas_wake_config
from atlas_ai.commands.system import close_system, start_system
from atlas_ai.memory.knowledge_store import KnowledgeStore


def normalize_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _is_wake_trigger(heard_text: str, expected_phrase: str) -> bool:
    normalized = normalize_phrase(heard_text)
    expected = normalize_phrase(expected_phrase)
    wake_aliases = {
        expected,
        "atlas wake",
        "atlas wake up",
        "wake atlas",
        "wake up atlas",
    }
    alias_set = {x for x in wake_aliases if x}
    if normalized in alias_set:
        return True
    return any(alias in normalized for alias in alias_set)


# Public alias for testability
is_wake_trigger = _is_wake_trigger


def _is_sleep_trigger(heard_text: str, expected_phrase: str) -> bool:
    normalized = normalize_phrase(heard_text)
    expected = normalize_phrase(expected_phrase)
    sleep_aliases = {
        expected,
        "atlas sleep",
        "sleep atlas",
        "go to sleep atlas",
    }
    alias_set = {x for x in sleep_aliases if x}
    if normalized in alias_set:
        return True
    return any(alias in normalized for alias in alias_set)


def _update_listener_state(project_root: str, **fields: Any) -> None:
    store = KnowledgeStore(str(Path(project_root) / "storage"))
    ts = datetime.now().isoformat()

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        flags = dict(current or {})
        state = dict(flags.get("voice_listener_orchestration", {}) or {})
        state.update({k: v for k, v in fields.items()})
        state["listener_path"] = "atlas_wake_listener"
        state["updated_at"] = ts
        flags["voice_listener_orchestration"] = state
        return flags

    try:
        store.update_runtime_flags(_mutator)
    except Exception:
        pass


def _set_startup_orchestration_state(project_root: str, **fields: Any) -> None:
    store = KnowledgeStore(str(Path(project_root) / "storage"))
    ts = datetime.now().isoformat()

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        flags = dict(current or {})
        orchestration = dict(flags.get("startup_orchestration", {}) or {})
        orchestration.update({k: v for k, v in fields.items()})
        orchestration["updated_at"] = ts
        flags["startup_orchestration"] = orchestration
        return flags

    try:
        store.update_runtime_flags(_mutator)
    except Exception:
        pass


def _startup_state_is_stale(startup_state: dict[str, Any], *, stale_after_seconds: int = 600) -> bool:
    if not bool(startup_state.get("startup_in_progress", False)):
        return False
    started_at_raw = str(startup_state.get("startup_started_at", "") or "").strip()
    if not started_at_raw:
        return True
    try:
        started_at = datetime.fromisoformat(started_at_raw)
    except Exception:
        return True
    age = (datetime.now(started_at.tzinfo) - started_at).total_seconds()
    return age > max(60, int(stale_after_seconds))


def _complete_async_start() -> None:
    config = load_atlas_wake_config()
    project_root = str(getattr(config, "project_root", Path.cwd()))
    try:
        _update_listener_state(
            project_root,
            launch_dispatch_state="running",
            launch_dispatched_at=datetime.now().isoformat(),
        )
        result = start_system()
        summary = dict(result.get("summary", {}) or {})
        health = dict(summary.get("health", {}) or {})
        desktop_health = dict(health.get("desktop_shell_health", {}) or {})
        _update_listener_state(
            project_root,
            launch_dispatch_state="completed",
            final_startup_verdict=str(result.get("overall_result", "") or health.get("overall_result", "")),
            final_startup_reason=str(health.get("overall_reason", "") or result.get("reason", "")),
            browser_fallback_used=bool(desktop_health.get("fallback_active", False)),
            fallback_reason=str(desktop_health.get("failure_reason", "") or health.get("degraded_reason", "") or ""),
            launch_completed_at=datetime.now().isoformat(),
        )
        _set_startup_orchestration_state(
            project_root,
            startup_in_progress=False,
            reason="launch_completed",
            startup_finished_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        _update_listener_state(
            project_root,
            launch_dispatch_state="failed",
            final_startup_verdict="FAILED",
            final_startup_reason=str(exc),
            launch_completed_at=datetime.now().isoformat(),
        )
        _set_startup_orchestration_state(
            project_root,
            startup_in_progress=False,
            reason=f"launch_failed:{exc}",
            startup_finished_at=datetime.now().isoformat(),
        )


def _dispatch_start_in_background(config, trigger_text: str) -> dict[str, Any]:
    project_root = str(getattr(config, "project_root", Path.cwd()))
    state_store = KnowledgeStore(str(Path(project_root) / "storage"))
    flags_snapshot = dict(state_store.load_runtime_flags() or {})
    startup_state = dict(flags_snapshot.get("startup_orchestration", {}) or {})
    listener_state = dict(flags_snapshot.get("voice_listener_orchestration", {}) or {})
    launch_dispatch_state = str(listener_state.get("launch_dispatch_state", "") or "").strip().lower()
    if launch_dispatch_state in {"started_async", "running"}:
        _update_listener_state(
            project_root,
            launch_dispatch_state="already_running",
            final_startup_reason="startup_dispatch_in_progress",
        )
        return {
            "trigger_received": trigger_text,
            "trigger_expected": config.trigger_phrase,
            "matched": True,
            "action": "wake",
            "launched": False,
            "pending": True,
            "dispatch_state": "already_running",
            "launch_summary": {},
            "reason": "startup_dispatch_in_progress",
        }
    if _startup_state_is_stale(startup_state):
        _set_startup_orchestration_state(
            project_root,
            startup_in_progress=False,
            reason="stale_startup_state_recovered",
            startup_finished_at=datetime.now().isoformat(),
        )
        startup_state = dict((state_store.load_runtime_flags() or {}).get("startup_orchestration", {}) or {})
    if bool(startup_state.get("startup_in_progress", False)):
        _update_listener_state(
            project_root,
            launch_dispatch_state="already_running",
            final_startup_reason="startup_already_in_progress",
        )
        return {
            "trigger_received": trigger_text,
            "trigger_expected": config.trigger_phrase,
            "matched": True,
            "action": "wake",
            "launched": False,
            "pending": True,
            "dispatch_state": "already_running",
            "launch_summary": {},
            "reason": "startup_already_in_progress",
        }
    dispatch_started_at = datetime.now().isoformat()
    cmd = [sys.executable, "-c", "from atlas_ai.atlas_wake_listener import _complete_async_start; _complete_async_start()"]
    kwargs: dict[str, Any] = {
        "cwd": str(project_root),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": dict(os.environ),
    }
    if os.name == "nt":
        creationflags = 0
        creationflags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        if creationflags:
            kwargs["creationflags"] = creationflags
    subprocess.Popen(cmd, **kwargs)
    _update_listener_state(
        project_root,
        launch_dispatch_state="started_async",
        launch_dispatched_at=dispatch_started_at,
    )
    return {
        "trigger_received": trigger_text,
        "trigger_expected": config.trigger_phrase,
        "matched": True,
        "action": "wake",
        "launched": True,
        "pending": True,
        "dispatch_state": "started_async",
        "launch_summary": {},
        "reason": "",
    }


def handle_trigger(trigger_text: str) -> dict[str, Any]:
    config = load_atlas_wake_config()
    normalized = normalize_phrase(trigger_text)
    project_root = str(getattr(config, "project_root", Path.cwd()))
    _update_listener_state(
        project_root,
        last_detected_transcript=trigger_text,
        last_detected_transcript_normalized=normalized,
        last_detected_at=datetime.now().isoformat(),
    )
    if _is_wake_trigger(trigger_text, config.trigger_phrase):
        return handle_trigger_async(trigger_text)
    if _is_sleep_trigger(trigger_text, config.sleep_phrase):
        logging.getLogger(__name__).info("Sleep trigger received via atlas_wake_listener: %s", trigger_text)
        _update_listener_state(
            project_root,
            matched_command="close_system",
            confirmation_required=False,
            confirmation_satisfied=True,
            shutdown_requested_at=datetime.now().isoformat(),
        )
        result = close_system(keep_listener=True)
        _update_listener_state(
            project_root,
            shutdown_result=str(result.get("overall_result", "") or ("blocked" if not result.get("ok", False) else "ok")),
            shutdown_reason=str(result.get("reason", "") or ""),
            listener_preserved=bool(result.get("listener_preserved", True)),
        )
        return {
            "trigger_received": trigger_text,
            "trigger_expected": config.sleep_phrase,
            "matched": True,
            "action": "sleep",
            "launched": False,
            "shutdown_summary": dict(result.get("summary", {}) or {}),
            "reason": str(result.get("reason", "") or ""),
        }
    return {
        "trigger_received": trigger_text,
        "trigger_expected": f"{config.trigger_phrase} | {config.sleep_phrase}",
        "matched": False,
        "launched": False,
    }


def handle_trigger_async(trigger_text: str) -> dict[str, Any]:
    config = load_atlas_wake_config()
    normalized = normalize_phrase(trigger_text)
    project_root = str(getattr(config, "project_root", Path.cwd()))
    _update_listener_state(
        project_root,
        last_detected_transcript=trigger_text,
        last_detected_transcript_normalized=normalized,
        last_detected_at=datetime.now().isoformat(),
    )
    if _is_wake_trigger(trigger_text, config.trigger_phrase):
        logging.getLogger(__name__).info("Wake trigger received via atlas_wake_listener(async): %s", trigger_text)
        _update_listener_state(
            project_root,
            matched_command="start_system",
            confirmation_required=False,
            confirmation_satisfied=True,
            startup_requested_at=datetime.now().isoformat(),
        )
        return _dispatch_start_in_background(config, trigger_text)
    if _is_sleep_trigger(trigger_text, config.sleep_phrase):
        return handle_trigger(trigger_text)
    return {
        "trigger_received": trigger_text,
        "trigger_expected": f"{config.trigger_phrase} | {config.sleep_phrase}",
        "matched": False,
        "launched": False,
        "pending": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Listen for the Atlas wake phrase and call the launcher.")
    parser.add_argument("--trigger", type=str, default="", help="One-shot trigger phrase for v1 automation.")
    parser.add_argument("--stdin", action="store_true", help="Read trigger phrases from stdin, one per line.")
    parser.add_argument("--json", action="store_true", help="Print JSON result.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload: dict[str, Any]
    if args.trigger:
        payload = handle_trigger_async(args.trigger)
    elif args.stdin:
        config = load_atlas_wake_config()
        print(f"Listening for wake phrase: {config.trigger_phrase}", flush=True)
        for line in sys.stdin:
            payload = handle_trigger_async(line.strip())
            if args.json:
                print(json.dumps(payload, indent=2), flush=True)
            else:
                if payload.get("action") == "wake":
                    if payload.get("pending"):
                        print("STARTING", flush=True)
                    else:
                        launch_health = dict(dict(payload.get("launch_summary", {}) or {}).get("health", {}) or {})
                        print(str(launch_health.get("overall_result", payload.get("reason", "IGNORED"))), flush=True)
                elif payload.get("action") == "sleep":
                    print(str(dict(payload.get("shutdown_summary", {}) or {}).get("overall_result", payload.get("reason", "IGNORED"))), flush=True)
                else:
                    print("IGNORED", flush=True)
            if payload.get("matched"):
                return 0
        return 1
    else:
        payload = handle_trigger_async("Atlas wake")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if payload.get("action") == "wake":
            if payload.get("pending"):
                print("STARTING")
            else:
                launch_health = dict(dict(payload.get("launch_summary", {}) or {}).get("health", {}) or {})
                print(str(launch_health.get("overall_result", payload.get("reason", "IGNORED"))))
        elif payload.get("action") == "sleep":
            print(str(dict(payload.get("shutdown_summary", {}) or {}).get("overall_result", payload.get("reason", "IGNORED"))))
        else:
            print("IGNORED")
    return 0 if payload.get("matched") else 1


if __name__ == "__main__":
    raise SystemExit(main())
