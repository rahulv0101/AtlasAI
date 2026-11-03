from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEntry:
    ts: float
    event: str
    request_id: str
    details: dict[str, Any]


class AuditLog:
    def __init__(self, *, max_entries: int = 2000) -> None:
        self._entries: list[AuditEntry] = []
        self._max_entries = max(100, int(max_entries))

    def append(self, event: str, request_id: str, details: dict[str, Any] | None = None) -> AuditEntry:
        entry = AuditEntry(
            ts=time.time(),
            event=str(event or "unknown_event"),
            request_id=str(request_id or ""),
            details=dict(details or {}),
        )
        self._entries.append(entry)
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]
        return entry

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [asdict(x) for x in self._entries[-max(1, int(limit)) :]]

    def last(self) -> dict[str, Any] | None:
        if not self._entries:
            return None
        return asdict(self._entries[-1])

