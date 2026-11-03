from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from atlas_ai.memory.jsonl_tail import read_jsonl_tail
from atlas_ai.learning.evidence_provenance import exclude_manual_tests

from atlas_ai.learning.missed_entry_audit import (
    backfill_missed_entry_collection_context,
    backfill_missed_entry_audits_from_events,
    build_blocker_selectivity_collection_plan,
    build_blocker_selectivity_collection_readiness,
    build_blocker_selectivity_offline_collector,
    build_blocker_selectivity_replay,
    build_blocker_selectivity_review,
    recent_missed_entry_audits,
    summarize_missed_entry_learning,
)
from atlas_ai.strategy_aliases import canonical_context_key, canonical_strategy_name


class KnowledgeStore:
    def __init__(self, storage_dir: str = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).resolve().parents[2]
        try:
            self.base.resolve().relative_to(self.project_root.resolve())
            self.knowledge_dir = self.project_root / "knowledge"
        except Exception:
            # Keep external/test stores isolated from each other.
            self.knowledge_dir = self.base / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.base / "learning_state.json"
        self.events_file = self.base / "events.jsonl"
        self.status_file = self.base / "status.json"
        self.runtime_file = self.base / "runtime_flags.json"
        self.heartbeat_file = self.base / "heartbeat.json"
        self.ui_heartbeat_file = self.base / "ui_heartbeat.json"
        self.self_improvement_audit_file = self.base / "self_improvement_audit.jsonl"
        self.rule_snapshot_dir = self.base / "rule_snapshots"
        self.rule_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._state_lock_file = self.base / "learning_state.lock"
        self._runtime_lock_file = self.base / "runtime_flags.lock"
        self._status_lock_file = self.base / "status.lock"
        self._heartbeat_lock_file = self.base / "heartbeat.lock"
        self._ui_heartbeat_lock_file = self.base / "ui_heartbeat.lock"
        self._status_cache_key: tuple[int, int] | None = None
        self._status_cache: dict[str, Any] = {}
        self._runtime_cache_key: tuple[int, int] | None = None
        self._runtime_cache: dict[str, Any] = {}
        self._events_cache_key: tuple[int, int] | None = None
        self._events_cache_offset = 0
        self._events_cache: list[dict[str, Any]] = []

    def _write_json_atomic(self, target: Path, payload: dict[str, Any]) -> None:
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{target.stem}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())

            last_err: Exception | None = None
            for _ in range(12):
                try:
                    os.replace(tmp, target)
                    last_err = None
                    return
                except PermissionError as exc:
                    last_err = exc
                    time.sleep(0.05)

            # Windows can deny os.replace if another process temporarily has the
            # target open for reading. Under the same process/file lock, fall back
            # to an in-place overwrite so training results are still committed.
            try:
                with target.open("w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                last_err = None
                return
            except Exception as exc:
                last_err = exc
                raise last_err
        finally:
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)

    def _locked_json_update(
        self,
        target: Path,
        lock_path: Path,
        mutator: Callable[[dict[str, Any]], dict[str, Any]],
        include_timestamp: bool = True,
    ) -> dict[str, Any]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with lock_path.open("a+b") as lock_fh:
                locked = False
                if os.name == "nt":
                    import msvcrt

                    lock_started = time.perf_counter()
                    while True:
                        try:
                            lock_fh.seek(0)
                            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
                            locked = True
                            break
                        except OSError:
                            if (time.perf_counter() - lock_started) > 3.0:
                                raise TimeoutError(f"Timed out acquiring lock for {target.name}")
                            time.sleep(0.01)
                            continue
                try:
                    current: dict[str, Any] = {}
                    if target.exists():
                        try:
                            with target.open("r", encoding="utf-8") as fh:
                                current = json.load(fh)
                        except Exception:
                            current = {}
                    updated = mutator(dict(current)) or {}
                    payload = self._normalize_learning_payload(dict(updated)) if target == self.state_file else dict(updated)
                    if include_timestamp:
                        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_json_atomic(target, payload)
                    return payload
                finally:
                    if os.name == "nt" and locked:
                        with contextlib.suppress(OSError):
                            lock_fh.seek(0)
                            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)

    def load_learning_state(self) -> dict[str, Any]:
        if not self.state_file.exists():
            return {}
        try:
            with self.state_file.open("r", encoding="utf-8") as fh:
                return self._normalize_learning_payload(json.load(fh))
        except Exception:
            # Guard against transient partial writes while concurrent threads persist state.
            return {}

    @staticmethod
    def _normalize_learning_payload(payload: dict[str, Any]) -> dict[str, Any]:
        root = dict(payload or {})
        state = dict(root.get("state", {}) or {})

        allocator = dict(state.get("allocator", {}) or {})
        merged_allocator: dict[str, dict[str, Any]] = {}
        for name, row in allocator.items():
            key = canonical_strategy_name(name)
            current = merged_allocator.setdefault(key, {"wins": 0, "losses": 0, "pnl": 0.0})
            value = dict(row or {})
            current["wins"] += int(value.get("wins", 0) or 0)
            current["losses"] += int(value.get("losses", 0) or 0)
            current["pnl"] += float(value.get("pnl", 0.0) or 0.0)
        if merged_allocator:
            state["allocator"] = merged_allocator

        loss_memory = dict(state.get("loss_memory", {}) or {})
        if loss_memory:
            outcomes = []
            for item in list(loss_memory.get("recent_outcomes", []) or []):
                row = dict(item or {})
                row["strategy"] = canonical_strategy_name(str(row.get("strategy", "") or "unknown"))
                outcomes.append(row)
            loss_memory["recent_outcomes"] = outcomes
            state["loss_memory"] = loss_memory

        win_memory = dict(state.get("win_memory", {}) or {})
        if win_memory:
            context_stats = dict(win_memory.get("context_stats", {}) or {})
            merged_context: dict[str, dict[str, Any]] = {}
            for key, row in context_stats.items():
                merged_key = canonical_context_key(str(key or ""))
                current = merged_context.setdefault(merged_key, {"wins": 0, "pnl": 0.0})
                value = dict(row or {})
                current["wins"] += int(value.get("wins", 0) or 0)
                current["pnl"] += float(value.get("pnl", 0.0) or 0.0)
            if merged_context:
                win_memory["context_stats"] = merged_context

            wins = []
            for item in list(win_memory.get("recent_wins", []) or []):
                row = dict(item or {})
                row["strategy"] = canonical_strategy_name(str(row.get("strategy", "") or "unknown"))
                wins.append(row)
            win_memory["recent_wins"] = wins
            state["win_memory"] = win_memory

        root["state"] = exclude_manual_tests(state)
        return root

    def save_learning_state(self, state: dict[str, Any]) -> None:
        self._locked_json_update(
            self.state_file,
            self._state_lock_file,
            lambda _current: {"state": exclude_manual_tests(state)},
        )

    def update_learning_state(self, mutator: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        return self._locked_json_update(
            self.state_file,
            self._state_lock_file,
            lambda current: {"state": exclude_manual_tests(mutator(exclude_manual_tests(dict(current.get("state", {})))))},
        )

    def save_status(self, status: dict[str, Any]) -> None:
        payload = self._locked_json_update(
            self.status_file,
            self._status_lock_file,
            lambda _current: dict(status),
        )
        with contextlib.suppress(Exception):
            stat = self.status_file.stat()
            self._status_cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
            self._status_cache = dict(payload)

    def load_status(self) -> dict[str, Any]:
        if not self.status_file.exists():
            return {}
        for _ in range(8):
            try:
                stat = self.status_file.stat()
                cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
                if self._status_cache_key == cache_key:
                    return copy.deepcopy(self._status_cache)
                with self.status_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                self._status_cache_key = cache_key
                self._status_cache = dict(payload)
                return copy.deepcopy(self._status_cache)
            except (json.JSONDecodeError, PermissionError, OSError):
                time.sleep(0.02)
                continue
            except Exception:
                return {}
        return {}

    def load_context_snapshot(self) -> dict[str, Any]:
        return dict(self.load_status().get("context_snapshot", {}) or {})

    def load_decision_explanation(self) -> dict[str, Any]:
        return dict(self.load_status().get("decision_explanation", {}) or {})

    def load_decision_trace(self) -> dict[str, Any]:
        return dict(self.load_status().get("decision_trace", {}) or {})

    def load_system_health(self) -> dict[str, Any]:
        return dict(self.load_status().get("system_health", {}) or {})

    def load_shadow_advice(self) -> dict[str, Any]:
        return dict(self.load_status().get("shadow_advice", {}) or {})

    def tail_missed_entry_audits(self, limit: int = 20) -> list[dict[str, Any]]:
        return recent_missed_entry_audits(self.base, limit=limit)

    def load_missed_entry_learning_summary(self, limit: int = 120) -> dict[str, Any]:
        return summarize_missed_entry_learning(self.base, limit=limit)

    def load_blocker_selectivity_review(self, primary_blocker: str | None = None, limit: int = 240) -> dict[str, Any]:
        return build_blocker_selectivity_review(self.base, primary_blocker=primary_blocker, limit=limit)

    def load_blocker_selectivity_replay(self, primary_blocker: str | None = None, limit: int = 240) -> dict[str, Any]:
        return build_blocker_selectivity_replay(self.base, primary_blocker=primary_blocker, limit=limit)

    def load_blocker_selectivity_collection_plan(self, proposal_id: str = "proposal_consensus_selectivity_context_review") -> dict[str, Any]:
        return build_blocker_selectivity_collection_plan(self, proposal_id=proposal_id)

    def load_blocker_selectivity_collection_readiness(self, limit: int = 240) -> dict[str, Any]:
        return build_blocker_selectivity_collection_readiness(self.base, limit=limit)

    def load_blocker_selectivity_offline_collector(self) -> dict[str, Any]:
        return build_blocker_selectivity_offline_collector(self.base)

    def backfill_missed_entry_audits_from_events(self, include_archives: bool = False, limit_files: int = 40) -> dict[str, Any]:
        return backfill_missed_entry_audits_from_events(
            self.base,
            include_archives=include_archives,
            limit_files=limit_files,
        )

    def backfill_missed_entry_collection_context(self, limit: int = 1000) -> dict[str, Any]:
        return backfill_missed_entry_collection_context(self.base, limit=limit)

    def log_event(self, event_type: str, message: str, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            record = {
                "time": datetime.now(timezone.utc).isoformat(),
                "type": event_type,
                "message": message,
                "meta": meta or {},
            }
            with self.events_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            if self._events_cache_key is not None or self._events_cache_offset > 0:
                self._events_cache.append(dict(record))
                with contextlib.suppress(Exception):
                    stat = self.events_file.stat()
                    self._events_cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
                    self._events_cache_offset = int(stat.st_size)
            else:
                self._events_cache_key = None
                self._events_cache_offset = 0
                self._events_cache = []

    def tail_events(self, limit: int = 80, event_type: str | None = None) -> list[dict[str, Any]]:
        return read_jsonl_tail(self.events_file, limit, event_type_filter=event_type)

    def _load_events_cache(self) -> list[dict[str, Any]]:
        if not self.events_file.exists():
            self._events_cache_key = None
            self._events_cache_offset = 0
            self._events_cache = []
            return []

        stat = self.events_file.stat()
        cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
        if self._events_cache_key == cache_key:
            return list(self._events_cache)

        if int(stat.st_size) >= int(self._events_cache_offset) > 0 and self._events_cache:
            try:
                with self.events_file.open("r", encoding="utf-8") as fh:
                    fh.seek(self._events_cache_offset)
                    for line in fh:
                        text = str(line or "").strip()
                        if not text:
                            continue
                        try:
                            self._events_cache.append(json.loads(text))
                        except Exception:
                            continue
                self._events_cache_key = cache_key
                self._events_cache_offset = int(stat.st_size)
                return list(self._events_cache)
            except Exception:
                pass

        events: list[dict[str, Any]] = []
        with self.events_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                text = str(line or "").strip()
                if not text:
                    continue
                try:
                    events.append(json.loads(text))
                except Exception:
                    continue
        self._events_cache = events
        self._events_cache_key = cache_key
        self._events_cache_offset = int(stat.st_size)
        return list(self._events_cache)

    def load_events(self, limit: int | None = None, event_type: str | None = None) -> list[dict[str, Any]]:
        if limit is not None:
            return self.tail_events(limit=limit, event_type=event_type)
        events = self._load_events_cache()
        wanted = str(event_type or "").strip().lower()
        if wanted:
            events = [row for row in events if str(row.get("type", "")).strip().lower() == wanted]
        return list(events)

    def load_runtime_flags(self) -> dict[str, Any]:
        if not self.runtime_file.exists():
            return {}
        for _ in range(8):
            try:
                stat = self.runtime_file.stat()
                cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
                if self._runtime_cache_key == cache_key:
                    return copy.deepcopy(self._runtime_cache)
                with self.runtime_file.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                self._runtime_cache_key = cache_key
                self._runtime_cache = dict(payload)
                return copy.deepcopy(self._runtime_cache)
            except (json.JSONDecodeError, PermissionError, OSError):
                time.sleep(0.02)
                continue
            except Exception:
                return {}
        return {}

    def save_runtime_flags(self, flags: dict[str, Any]) -> None:
        try:
            payload = self._locked_json_update(
                self.runtime_file,
                self._runtime_lock_file,
                lambda _current: dict(flags),
            )
        except TimeoutError:
            return
        with contextlib.suppress(Exception):
            stat = self.runtime_file.stat()
            self._runtime_cache_key = (int(stat.st_mtime_ns), int(stat.st_size))
            self._runtime_cache = dict(payload)

    def update_runtime_flags(self, mutator: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        try:
            return self._locked_json_update(
                self.runtime_file,
                self._runtime_lock_file,
                lambda current: mutator(dict(current)),
            )
        except TimeoutError:
            return self.load_runtime_flags()

    def load_heartbeat(self) -> dict[str, Any]:
        if not self.heartbeat_file.exists():
            return {}
        try:
            with self.heartbeat_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def load_ui_heartbeat(self) -> dict[str, Any]:
        if not self.ui_heartbeat_file.exists():
            return {}
        try:
            with self.ui_heartbeat_file.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    def write_backend_heartbeat(
        self,
        *,
        mode: str,
        pid: int | None = None,
        cycle_status: str = "running",
    ) -> dict[str, Any]:
        now_ts = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        process_id = int(pid if pid is not None else os.getpid())

        def _heartbeat_mutator(current: dict[str, Any]) -> dict[str, Any]:
            previous = int(dict(current or {}).get("cycle_count", 0) or 0)
            return {
                "schema": "atlas_backend_heartbeat_v1",
                "component": "backend_trader",
                "last_heartbeat": now_iso,
                "last_heartbeat_ts": now_ts,
                "mode": str(mode or "unknown"),
                "pid": process_id,
                "cycle_count": previous + 1,
                "cycle_status": str(cycle_status or "running"),
            }

        payload = self._locked_json_update(
            self.heartbeat_file,
            self._heartbeat_lock_file,
            _heartbeat_mutator,
            include_timestamp=False,
        )

        def _runtime_mutator(current: dict[str, Any]) -> dict[str, Any]:
            flags = dict(current or {})
            heartbeat = dict(flags.get("runtime_heartbeat", {}) or {})
            row = dict(heartbeat.get("backend_trader", {}) or {})
            row.update(
                {
                    "component": "backend_trader",
                    "last_heartbeat_at": now_iso,
                    "last_heartbeat_ts": now_ts,
                    "cycle_status": str(cycle_status or "running"),
                    "mode": str(mode or "unknown"),
                    "pid": process_id,
                    "cycle_count": int(payload.get("cycle_count", 0) or 0),
                    "heartbeat_file": str(self.heartbeat_file),
                }
            )
            heartbeat["backend_trader"] = row
            flags["runtime_heartbeat"] = heartbeat
            return flags

        with contextlib.suppress(Exception):
            self.update_runtime_flags(_runtime_mutator)
        return payload

    def write_ui_heartbeat(
        self,
        *,
        pid: int | None = None,
        cycle_status: str = "running",
        bind_host: str = "127.0.0.1",
        bind_port: int = 8765,
    ) -> dict[str, Any]:
        now_ts = time.time()
        now_iso = datetime.now(timezone.utc).isoformat()
        process_id = int(pid if pid is not None else os.getpid())

        def _heartbeat_mutator(current: dict[str, Any]) -> dict[str, Any]:
            previous = int(dict(current or {}).get("cycle_count", 0) or 0)
            return {
                "schema": "atlas_ui_heartbeat_v1",
                "component": "ui_server",
                "last_heartbeat": now_iso,
                "last_heartbeat_ts": now_ts,
                "pid": process_id,
                "cycle_count": previous + 1,
                "cycle_status": str(cycle_status or "running"),
                "bind_host": str(bind_host or "127.0.0.1"),
                "bind_port": int(bind_port),
            }

        payload = self._locked_json_update(
            self.ui_heartbeat_file,
            self._ui_heartbeat_lock_file,
            _heartbeat_mutator,
            include_timestamp=False,
        )

        def _runtime_mutator(current: dict[str, Any]) -> dict[str, Any]:
            flags = dict(current or {})
            heartbeat = dict(flags.get("runtime_heartbeat", {}) or {})
            row = dict(heartbeat.get("ui_server", {}) or {})
            row.update(
                {
                    "component": "ui_server",
                    "last_heartbeat_at": now_iso,
                    "last_heartbeat_ts": now_ts,
                    "cycle_status": str(cycle_status or "running"),
                    "pid": process_id,
                    "cycle_count": int(payload.get("cycle_count", 0) or 0),
                    "heartbeat_file": str(self.ui_heartbeat_file),
                    "bind_host": str(bind_host or "127.0.0.1"),
                    "bind_port": int(bind_port),
                }
            )
            heartbeat["ui_server"] = row
            flags["runtime_heartbeat"] = heartbeat
            return flags

        with contextlib.suppress(Exception):
            self.update_runtime_flags(_runtime_mutator)
        return payload

    def _resolve_structured_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] == "knowledge":
            return self.project_root / candidate
        return self.knowledge_dir / candidate

    def load_structured_knowledge_file(self, path: str | Path, default: Any | None = None) -> Any:
        target = self._resolve_structured_path(path)
        if not target.exists():
            return copy.deepcopy(default)
        try:
            with target.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return copy.deepcopy(default)

    def save_structured_knowledge_file(self, path: str | Path, payload: Any) -> None:
        target = self._resolve_structured_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(payload, dict):
            self._write_json_atomic(target, dict(payload))
            return
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f"{target.stem}.",
            suffix=".tmp",
            dir=str(target.parent),
            text=True,
        )
        tmp = Path(tmp_name)
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, target)
        finally:
            with contextlib.suppress(Exception):
                tmp.unlink(missing_ok=True)

    def append_self_improvement_audit(self, event: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(event or {})
            payload.setdefault("time", datetime.now(timezone.utc).isoformat())
            payload.setdefault("type", "self_improvement")
            with self.self_improvement_audit_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")

    def tail_self_improvement_audit(self, limit: int = 80) -> list[dict[str, Any]]:
        if not self.self_improvement_audit_file.exists():
            return []
        with self.self_improvement_audit_file.open("r", encoding="utf-8") as fh:
            lines = list(deque(fh, maxlen=max(1, int(limit))))
        rows: list[dict[str, Any]] = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        return rows

    def save_rule_snapshot(self, snapshot_id: str, payload: dict[str, Any]) -> Path:
        safe_id = str(snapshot_id or "").strip() or f"snapshot_{int(time.time())}"
        target = self.rule_snapshot_dir / f"{safe_id}.json"
        self._write_json_atomic(target, dict(payload))
        return target

    def load_rule_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        target = self.rule_snapshot_dir / f"{str(snapshot_id or '').strip()}.json"
        if not target.exists():
            return {}
        try:
            with target.open("r", encoding="utf-8") as fh:
                return dict(json.load(fh) or {})
        except Exception:
            return {}

    def maintain_events(self, max_lines: int = 20000) -> dict[str, int]:
        if not self.events_file.exists():
            return {"kept": 0, "dropped_malformed": 0, "dropped_duplicates": 0, "archived": 0}
        with self._lock:
            dropped_malformed = 0
            dropped_duplicates = 0
            seen: dict[bytes, list[tuple[int, int]]] = {}
            retained_offsets = deque(maxlen=max(1, int(max_lines)))
            unique_count = 0
            archive_dir = self.base / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            archive_file = archive_dir / f"events_{stamp}.jsonl"
            # Spool payloads to disk. Keep only digest/offset metadata in memory;
            # verify matching digests against the original normalized JSON so a
            # hash collision cannot discard a distinct event.
            with tempfile.TemporaryFile(mode="w+b", dir=self.base) as spool:
                write_offset = 0
                with self.events_file.open("r", encoding="utf-8") as fh:
                    for raw in fh:
                        line = str(raw or "").strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            normalized = json.dumps(obj, sort_keys=True)
                        except Exception:
                            dropped_malformed += 1
                            continue
                        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
                        duplicate = False
                        matching_offsets = seen.get(digest, ())
                        for offset, length in matching_offsets:
                            spool.seek(offset)
                            previous = json.loads(spool.read(length))
                            if json.dumps(previous, sort_keys=True) == normalized:
                                duplicate = True
                                break
                        if matching_offsets:
                            spool.seek(write_offset)
                        if duplicate:
                            dropped_duplicates += 1
                            continue
                        row = (line + os.linesep).encode("utf-8")
                        offset = write_offset
                        spool.write(row)
                        write_offset += len(row)
                        seen.setdefault(digest, []).append((offset, len(row)))
                        retained_offsets.append(offset)
                        unique_count += 1
                archived = unique_count - len(retained_offsets)
                retained_start = retained_offsets[0] if retained_offsets else 0
                spool.seek(0)
                if archived:
                    with archive_file.open("wb") as af:
                        remaining = retained_start
                        while remaining:
                            block = spool.read(min(64 * 1024, remaining))
                            if not block:
                                raise OSError("Event maintenance spool ended before archive boundary")
                            af.write(block)
                            remaining -= len(block)

                tmp_fd, tmp_name = tempfile.mkstemp(
                    prefix=f"{self.events_file.stem}.",
                    suffix=".tmp",
                    dir=str(self.events_file.parent),
                )
                tmp = Path(tmp_name)
                try:
                    with os.fdopen(tmp_fd, "wb") as fh:
                        shutil.copyfileobj(spool, fh, length=64 * 1024)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.events_file)
                finally:
                    with contextlib.suppress(Exception):
                        tmp.unlink(missing_ok=True)
            self._events_cache_key = None
            self._events_cache_offset = 0
            self._events_cache = []
            return {
                "kept": len(retained_offsets),
                "dropped_malformed": dropped_malformed,
                "dropped_duplicates": dropped_duplicates,
                "archived": archived,
            }
