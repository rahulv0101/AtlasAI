from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.atlas_full_launcher import run_full_launch, verify_health
from atlas_ai.atlas_full_shutdown import run_full_shutdown
from atlas_ai.atlas_wake_config import load_atlas_wake_config
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
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


def _debounce_seconds() -> float:
    raw = str(os.getenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "8")).strip()
    try:
        return max(1.0, min(60.0, float(raw)))
    except Exception:
        return 8.0


def _startup_stale_after_seconds() -> float:
    raw = str(os.getenv("ATLAS_STARTUP_STALE_AFTER_SECONDS", "600")).strip()
    try:
        return max(60.0, min(3600.0, float(raw)))
    except Exception:
        return 600.0


def _state_store(config) -> KnowledgeStore:
    return KnowledgeStore(str(Path(config.project_root) / "storage"))


def _state_payload(flags: dict[str, Any]) -> dict[str, Any]:
    return dict(flags.get("startup_orchestration", {}) or {})


def _startup_state_is_stale(state: dict[str, Any], *, now: datetime | None = None) -> bool:
    if not bool(state.get("startup_in_progress", False)):
        return False
    now = now or _now_utc()
    started_at = _parse_ts(state.get("startup_started_at", ""))
    if started_at is None:
        return True
    finished_at = _parse_ts(state.get("startup_finished_at", ""))
    if finished_at is not None and finished_at >= started_at:
        return True
    age_seconds = (now - started_at).total_seconds()
    return age_seconds > _startup_stale_after_seconds()


def _set_startup_state(store: KnowledgeStore, *, in_progress: bool, source: str, reason: str = "") -> dict[str, Any]:
    now = _now_utc().isoformat()

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        flags = dict(current or {})
        state = _state_payload(flags)
        state["startup_in_progress"] = bool(in_progress)
        state["startup_source"] = str(source or "")
        state["reason"] = str(reason or "")
        state["updated_at"] = now
        if in_progress:
            state["startup_started_at"] = now
        else:
            state["startup_finished_at"] = now
        flags["startup_orchestration"] = state
        return flags

    return dict(store.update_runtime_flags(_mutator) or {})


def _try_begin_startup(store: KnowledgeStore, *, source: str) -> tuple[bool, dict[str, Any], str]:
    now = _now_utc()
    debounce = _debounce_seconds()
    block_reason = ""

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        nonlocal block_reason
        flags = dict(current or {})
        state = _state_payload(flags)
        in_progress = bool(state.get("startup_in_progress", False))
        if in_progress:
            if _startup_state_is_stale(state, now=now):
                state["startup_in_progress"] = False
                state["reason"] = "stale_startup_state_recovered"
                state["startup_finished_at"] = now.isoformat()
            else:
                block_reason = "startup_already_in_progress"
                return flags
        last_wake_at = _parse_ts(state.get("last_wake_trigger_at", ""))
        if last_wake_at is not None and (now - last_wake_at).total_seconds() < debounce:
            block_reason = "wake_debounced"
            return flags
        state["startup_in_progress"] = True
        state["startup_source"] = str(source or "")
        state["startup_started_at"] = now.isoformat()
        state["last_wake_trigger_at"] = now.isoformat()
        state["updated_at"] = now.isoformat()
        flags["startup_orchestration"] = state
        return flags

    updated = dict(store.update_runtime_flags(_mutator) or {})
    return (not bool(block_reason), updated, block_reason)


def start_system() -> dict[str, Any]:
    config = load_atlas_wake_config()
    store = _state_store(config)
    source = "commands.system.start_system"
    ok_to_start, flags, block_reason = _try_begin_startup(store, source=source)
    if not ok_to_start:
        logging.getLogger(__name__).info("Wake trigger ignored: %s", block_reason)
        store.log_event(
            "system",
            "Wake trigger ignored.",
            {"reason": block_reason, "startup_orchestration": dict(flags.get("startup_orchestration", {}) or {})},
        )
        return {
            "ok": False,
            "command": "start_system",
            "reason": str(block_reason or "startup_blocked"),
            "summary": {},
        }
    with_health_probe = all(hasattr(config, attr) for attr in ("comparison_root", "comparison_run_id", "launch_profile"))
    if with_health_probe:
        try:
            current_health = verify_health(config)
        except Exception:
            current_health = {}
        overall = str(current_health.get("overall_result", "") or "").strip().upper()
        control_surface_ok = bool(dict(current_health.get("control_surface_health", {}) or {}).get("ok", False))
        core_runtime_ok = bool(dict(current_health.get("core_runtime_health", {}) or {}).get("ok", False))
        if overall in {"SUCCESS", "DEGRADED"} and core_runtime_ok and control_surface_ok:
            logging.getLogger(__name__).info("Wake trigger ignored: Atlas stack already healthy and running.")
            store.log_event(
                "system",
                "Wake trigger ignored because Atlas stack is already healthy.",
                {"overall_result": overall, "health": current_health},
            )
            _set_startup_state(store, in_progress=False, source=source, reason="already_running_healthy")
            return {
                "ok": True,
                "command": "start_system",
                "overall_result": overall,
                "reason": "already_running_healthy",
                "summary": {"health": current_health},
            }
    logging.getLogger(__name__).info("Wake trigger accepted. Launch sequence started.")
    store.log_event("system", "Wake command received. Launch sequence started.", {"source": source})
    try:
        summary = run_full_launch(config)
        overall_result = str(dict(summary.get("health", {}) or {}).get("overall_result", "")).strip().upper()
        logging.getLogger(__name__).info("Launch sequence ended with result=%s", overall_result)
        store.log_event(
            "system",
            "Launch sequence ended.",
            {"overall_result": overall_result},
        )
    finally:
        _set_startup_state(store, in_progress=False, source=source, reason="launch_completed")
    overall_result = str(dict(summary.get("health", {}) or {}).get("overall_result", "")).strip().upper()
    ok = overall_result in {"SUCCESS", "DEGRADED"}
    return {
        "ok": bool(ok),
        "command": "start_system",
        "overall_result": overall_result,
        "summary": summary,
    }


def close_system(*, keep_listener: bool = False) -> dict[str, Any]:
    config = load_atlas_wake_config()
    store = _state_store(config)
    flags = dict(store.load_runtime_flags() or {})
    state = _state_payload(flags)
    if bool(state.get("startup_in_progress", False)):
        logging.getLogger(__name__).warning(
            "Close system requested during startup; forcing full shutdown while preserving listener=%s.",
            bool(keep_listener),
        )
        store.log_event(
            "system",
            "Close command received during startup. Forcing shutdown.",
            {"reason": "startup_in_progress_forced_shutdown", "startup_orchestration": state},
        )
    summary = run_full_shutdown(config, keep_listener=keep_listener)
    _set_startup_state(store, in_progress=False, source="commands.system.close_system", reason="shutdown_completed")
    return {
        "ok": True,
        "command": "close_system",
        "overall_result": str(summary.get("overall_result", "")),
        "listener_preserved": bool(summary.get("listener_preserved", keep_listener)),
        "summary": summary,
    }
