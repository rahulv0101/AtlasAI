from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

from atlas_ai.config.settings import Settings, load_settings
from atlas_ai.memory.knowledge_store import KnowledgeStore


def evaluate_heartbeat_state(
    runtime_flags: dict[str, Any],
    *,
    component: str,
    now_ts: float | None = None,
    stale_after_seconds: float = 180.0,
) -> dict[str, Any]:
    flags = dict(runtime_flags or {})
    heartbeat = dict(flags.get("runtime_heartbeat", {}) or {})
    row = dict(heartbeat.get(component, {}) or {})
    now = float(now_ts if now_ts is not None else time.time())
    last_ts = float(row.get("last_heartbeat_ts", 0.0) or 0.0)
    age_seconds = max(0.0, now - last_ts) if last_ts > 0 else float("inf")
    is_stale = bool(last_ts <= 0 or age_seconds > float(stale_after_seconds))
    return {
        "component": component,
        "last_heartbeat_at": str(row.get("last_heartbeat_at", "") or ""),
        "last_heartbeat_ts": last_ts,
        "age_seconds": age_seconds,
        "cycle_status": str(row.get("cycle_status", "unknown") or "unknown"),
        "is_stale": is_stale,
        "stale_after_seconds": float(stale_after_seconds),
    }


def _resolve_staleness_threshold(settings: Settings | None = None, threshold_seconds: int | None = None) -> int:
    if threshold_seconds is not None:
        return max(1, int(threshold_seconds))
    active_settings = settings if settings is not None else load_settings()
    configured = int(getattr(active_settings, "backend_heartbeat_stale_after_seconds", 0) or 0)
    scan_interval = int(getattr(active_settings, "scan_interval_seconds", 30) or 30)
    return max(1, configured or max(180, scan_interval * 3))


def get_backend_health(
    store: KnowledgeStore | None = None,
    *,
    settings: Settings | None = None,
    threshold_seconds: int | None = None,
    now_ts: float | None = None,
) -> dict[str, Any]:
    active_settings = settings if settings is not None else load_settings()
    active_store = store if store is not None else KnowledgeStore(active_settings.storage_dir)
    threshold = _resolve_staleness_threshold(active_settings, threshold_seconds)
    heartbeat = active_store.load_heartbeat()
    if not heartbeat:
        return {
            "is_fresh": False,
            "last_heartbeat": "",
            "seconds_since_heartbeat": 0,
            "staleness_threshold_seconds": threshold,
            "cycle_count": 0,
            "status": "unknown",
        }

    last_ts = float(heartbeat.get("last_heartbeat_ts", 0.0) or 0.0)
    if last_ts <= 0:
        return {
            "is_fresh": False,
            "last_heartbeat": str(heartbeat.get("last_heartbeat", "") or ""),
            "seconds_since_heartbeat": 0,
            "staleness_threshold_seconds": threshold,
            "cycle_count": int(heartbeat.get("cycle_count", 0) or 0),
            "status": "unknown",
        }

    now = float(now_ts if now_ts is not None else time.time())
    seconds_since = max(0, int(now - last_ts))
    is_fresh = seconds_since <= threshold
    return {
        "is_fresh": is_fresh,
        "last_heartbeat": str(heartbeat.get("last_heartbeat", "") or ""),
        "seconds_since_heartbeat": seconds_since,
        "staleness_threshold_seconds": threshold,
        "cycle_count": int(heartbeat.get("cycle_count", 0) or 0),
        "status": "live" if is_fresh else "stale",
    }


def start_runtime_heartbeat(
    store: KnowledgeStore,
    *,
    component: str,
    interval_seconds: float = 120.0,
    stale_after_seconds: float = 180.0,
) -> threading.Event:
    stop_event = threading.Event()
    component_name = str(component or "unknown").strip() or "unknown"
    interval = max(1.0, float(interval_seconds))
    stale_after = max(interval, float(stale_after_seconds))

    def _tick(status: str) -> None:
        now = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        if component_name == "ui_server":
            store.write_ui_heartbeat(cycle_status=status)

        def _mutator(current: dict[str, Any]) -> dict[str, Any]:
            flags = dict(current or {})
            heartbeat = dict(flags.get("runtime_heartbeat", {}) or {})
            row = dict(heartbeat.get(component_name, {}) or {})
            row.update(
                {
                    "component": component_name,
                    "last_heartbeat_at": now_iso,
                    "last_heartbeat_ts": now,
                    "cycle_status": str(status),
                    "interval_seconds": interval,
                    "stale_after_seconds": stale_after,
                }
            )
            heartbeat[component_name] = row
            flags["runtime_heartbeat"] = heartbeat
            return flags

        store.update_runtime_flags(_mutator)

    def _loop() -> None:
        while not stop_event.wait(interval):
            _tick("running")
        _tick("stopped")

    _tick("running")
    threading.Thread(target=_loop, daemon=True).start()
    return stop_event
