from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import re

from atlas_ai.ui.ask_atlas import AskAtlasService


@dataclass(frozen=True)
class ConversationTurn:
    raw_query: str
    normalized_query: str
    intent: str
    reasoning_mode: str
    result: str


@dataclass
class ConversationMemory:
    turns: deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=5))

    def remember(self, turn: ConversationTurn) -> None:
        self.turns.append(turn)

    def recent(self, limit: int = 3) -> list[ConversationTurn]:
        return list(self.turns)[-limit:]

    def latest(self) -> ConversationTurn | None:
        return self.turns[-1] if self.turns else None


class ConversationalAtlasInterface:
    """Conversational wrapper over the canonical Ask Atlas service.

    Older tests may still inject an object with ``respond()``; the default path
    now uses ``AskAtlasService.answer()`` so informal typed queries share the
    same Brain v2 route as the UI and voice surfaces.
    """

    def __init__(self, source: Any, assistant: Any | None = None) -> None:
        self.core = assistant or AskAtlasService(source)
        self.memory = ConversationMemory()

    def respond(self, raw_query: str) -> dict[str, Any]:
        normalized = self._normalize_query(raw_query)
        if hasattr(self.core, "answer"):
            payload = dict(self.core.answer(normalized))
        else:
            payload = dict(self.core.respond(normalized))
        payload["conversation_query"] = normalized
        payload["conversation_memory_size"] = len(self.memory.turns)
        payload["acknowledge"] = self._rewrite_acknowledge(str(payload.get("acknowledge", "") or ""))
        payload["understanding"] = self._rewrite_understanding(str(payload.get("understanding", "") or ""))
        payload["result"] = self._rewrite_result(payload)
        self.memory.remember(
            ConversationTurn(
                raw_query=str(raw_query or ""),
                normalized_query=normalized,
                intent=str(payload.get("intent", "") or ""),
                reasoning_mode=str(payload.get("reasoning_mode", "") or ""),
                result=str(payload.get("result", "") or ""),
            )
        )
        return payload

    def _normalize_query(self, raw_query: str) -> str:
        query = " ".join(str(raw_query or "").strip().split())
        lowered = query.lower()
        if not lowered:
            return query
        latest = self.memory.latest()

        explicit_map = {
            "are we close to a trade": "are we close to a trade or not",
            "whats going on": "what is happening now",
            "what's going on": "what is happening now",
            "is the market good or bad": "tell me whether the market is favorable or unfavorable right now",
            "is the market good": "tell me whether the market is favorable or unfavorable right now",
            "is the market bad": "tell me whether the market is favorable or unfavorable right now",
            "what should i do": "what is the best approach right now",
        }
        if lowered in explicit_map:
            return explicit_map[lowered]

        if self._contains_any(lowered, ("are we close to a trade", "close to a trade", "close to trading")):
            return "are we close to a trade or not"
        if self._contains_any(lowered, ("what's going on", "whats going on", "what is going on")):
            return "what is happening now"
        if self._contains_any(lowered, ("market good or bad", "market good", "market bad", "market clean or messy")):
            return "tell me whether the market is favorable or unfavorable right now"
        if self._contains_any(lowered, ("what should i do", "what should we do", "best approach")):
            if latest and latest.intent == "action_guidance" and self._contains_any(latest.result.lower(), ("consensus", "signals", "guard", "loss-pattern", "strategy")):
                if "consensus" in latest.result.lower():
                    return "what should I do about consensus"
                if "signal" in latest.result.lower():
                    return "how should we improve signals"
            return "what is the best approach right now"
        if latest and self._contains_any(lowered, ("what about that", "and that", "about that", "about this")):
            if latest.intent == "action_guidance":
                return f"what should I do about {self._component_from_turn(latest)}"
        return query

    def _component_from_turn(self, turn: ConversationTurn) -> str:
        lowered = turn.result.lower()
        for component in ("consensus", "signal quality", "signals", "guards", "loss-pattern", "strategy"):
            if component in lowered:
                return component
        return "the current blocker"

    def _rewrite_acknowledge(self, text: str) -> str:
        lowered = text.lower()
        if "read-only recommendation" in lowered:
            return "I’m answering from the read-only Atlas layer, sir."
        if "ready for a task" in lowered:
            return "I’m ready when you are, sir."
        return text

    def _rewrite_understanding(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return raw
        raw = raw.replace("You want", "You’re asking for").replace("You are asking", "You’re asking")
        return raw

    def _rewrite_result(self, payload: dict[str, Any]) -> str:
        result = str(payload.get("result", "") or "").strip()
        if not result:
            return result
        intent = str(payload.get("intent", "") or "")
        if intent == "action_guidance":
            result = re.sub(r"\bCurrent role:\s*", "", result, count=1)
            result = result.replace("Diagnosis:", "Right now,")
            result = result.replace("Recommendation:", "Recommendation:")
            result = result.replace("Reasoning:", "Why:")
            return result
        if intent == "general_operator_query":
            return result.replace("I can answer from current state even when the request is only partly specified.", "Here’s the closest grounded answer from current Atlas state.")
        return result

    @staticmethod
    def _contains_any(lowered: str, phrases: tuple[str, ...]) -> bool:
        return any(phrase in lowered for phrase in phrases)
