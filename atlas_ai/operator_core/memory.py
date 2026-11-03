from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryTurn:
    query: str
    intent: str
    answer: str


class ConversationMemory:
    def __init__(self, max_turns: int = 12) -> None:
        self._turns: deque[MemoryTurn] = deque(maxlen=max_turns)

    def add(self, query: str, intent: str, answer: str) -> None:
        self._turns.append(MemoryTurn(query=str(query or ""), intent=str(intent or ""), answer=str(answer or "")))

    def latest(self) -> MemoryTurn | None:
        return self._turns[-1] if self._turns else None

    def recent(self, limit: int = 5) -> list[MemoryTurn]:
        return list(self._turns)[-max(1, int(limit)) :]

