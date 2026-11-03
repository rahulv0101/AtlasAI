from __future__ import annotations

import time
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BrainTurn:
    query: str
    intent: str
    domain: str
    plan: str
    answer: str
    main_conclusion: str
    supporting_reasons: list[str]
    confidence: float
    entities: list[str]
    timestamp: float


class BrainMemory:
    def __init__(self, max_turns: int = 20) -> None:
        self._turns: deque[BrainTurn] = deque(maxlen=max_turns)
        self.recent_questions: deque[str] = deque(maxlen=max_turns)
        self.recent_conclusions: deque[str] = deque(maxlen=max_turns)
        self.repeated_blockers: Counter[str] = Counter()
        # Legacy compatibility for modules/tests that inspect this attribute directly.
        self.recent_turns = self._turns

    def add(
        self,
        *,
        query: str,
        intent: str,
        domain: str,
        plan: str,
        answer: str,
        confidence: float = 0.0,
    ) -> None:
        clean_answer = str(answer or "").strip()
        lines = [ln.strip() for ln in clean_answer.splitlines() if ln.strip()]
        main = lines[0] if lines else ""
        reasons = [ln.lstrip("- ").strip() for ln in lines[1:3] if ln.startswith("-")]
        entities = [x for x in ("comparison", "readiness", "blocker", "volatility", "market", "strategy") if x in clean_answer.lower()]
        self._turns.append(
            BrainTurn(
                query=str(query or ""),
                intent=str(intent or ""),
                domain=str(domain or ""),
                plan=str(plan or ""),
                answer=clean_answer,
                main_conclusion=main,
                supporting_reasons=reasons,
                confidence=float(confidence or 0.0),
                entities=entities,
                timestamp=time.time(),
            )
        )
        self.remember_question(query)
        self.remember_conclusion(clean_answer)

    # Legacy compatibility methods used by operator_assistant/older tests.
    def remember_question(self, query: str) -> None:
        text = " ".join(str(query or "").split())
        if text:
            self.recent_questions.append(text)

    def remember_conclusion(self, conclusion: str) -> None:
        text = " ".join(str(conclusion or "").split())
        if text:
            self.recent_conclusions.append(text)

    def remember_turn(self, turn: Any) -> None:
        self._turns.append(turn)

    def recent_relevant_turns(self, limit: int = 5) -> list[Any]:
        relevant = [turn for turn in self._turns if bool(getattr(turn, "follow_up_relevance", True))]
        if not relevant:
            relevant = list(self._turns)
        return relevant[-max(1, int(limit)) :]

    def recent(self, limit: int = 5) -> list[BrainTurn]:
        return list(self._turns)[-max(1, int(limit)) :]

    def latest(self) -> BrainTurn | None:
        return self._turns[-1] if self._turns else None

    @staticmethod
    def is_follow_up(query: str) -> bool:
        q = str(query or "").strip().lower()
        if not q:
            return False
        compact = q.replace("?", "").strip()
        if compact in {
            "why",
            "why not",
            "what changed",
            "compared to what",
            "is that still true",
            "what do you mean",
            "and the next step",
            "what about that",
            "about that",
            "what about this",
            "about this",
            "explain that",
            "explain that simpler",
            "why does that matter",
            "based on that what next",
            "based on that what should happen next",
        }:
            return True
        return any(
            marker in q
            for marker in (
                "why?",
                "what changed?",
                "compared to what",
                "is that still true",
                "what do you mean",
                "how confident are you",
                "and the next step",
                "what about that",
                "what about this",
                "explain that",
                "why does that matter",
                "based on that",
                "why not",
            )
        )

    def build_follow_up_context(self, query: str) -> str:
        latest = self.latest()
        if latest is None:
            return str(query or "")
        q = str(query or "").strip()
        return (
            f"{q} (follow-up context: prior_query='{latest.query}'; "
            f"prior_intent={latest.intent}; prior_domain={latest.domain}; "
            f"prior_plan={latest.plan}; prior_conclusion='{latest.main_conclusion}')"
        )
