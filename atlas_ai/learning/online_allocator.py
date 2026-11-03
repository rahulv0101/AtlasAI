from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.models import TradeSignal
from atlas_ai.strategy_aliases import canonical_strategy_name


@dataclass
class StrategyState:
    wins: int = 1
    losses: int = 1
    pnl: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / (self.wins + self.losses)


class OnlineAllocator:
    """Self-learning portfolio allocator using reward-updated strategy weights."""

    def __init__(self, store: KnowledgeStore | None = None) -> None:
        self._state: dict[str, StrategyState] = {}
        self.store = store
        self.exploration_epsilon = 0.0
        self._rng = random.Random(17)
        self._restore()

    def set_exploration(self, epsilon: float) -> None:
        self.exploration_epsilon = max(0.0, min(0.5, float(epsilon)))

    def _restore(self) -> None:
        if self.store is None:
            return
        raw = self.store.load_learning_state().get("state", {})
        allocator = raw.get("allocator", {})
        for name, value in allocator.items():
            self._state[name] = StrategyState(**value)

    def _persist(self) -> None:
        if self.store is None:
            return
        allocator_payload = {name: asdict(state) for name, state in self._state.items()}
        self.store.update_learning_state(lambda existing: {**existing, "allocator": allocator_payload})

    def _strategy_multiplier(self, strategy_name: str) -> float:
        name = canonical_strategy_name(strategy_name).lower()
        if name == "macd_trend":
            return 0.82
        if name.startswith("evolved_rule"):
            return 0.78
        if name == "ema_crossover":
            return 1.08
        return 1.0

    def _key(self, strategy_name: str) -> str:
        raw = canonical_strategy_name(strategy_name)
        if raw.startswith("evolved_rule("):
            return raw
        base = raw.split("(", 1)[0].strip()
        return base or raw or "unknown"

    def score_signal(self, signal: TradeSignal) -> float:
        key = self._key(signal.strategy)
        st = self._state.setdefault(key, StrategyState())
        base = signal.confidence * 0.7 + st.win_rate * 0.3
        return base * self._strategy_multiplier(key)

    def choose(self, signals: list[TradeSignal], score_overrides: dict[str, float] | None = None) -> TradeSignal:
        if not signals:
            raise ValueError("No signals")

        scored = []
        overrides = dict(score_overrides or {})
        for signal in signals:
            key = self._key(signal.strategy)
            score = float(overrides.get(key, self.score_signal(signal)))
            scored.append((score, signal))

        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) > 1 and self.exploration_epsilon > 0.0 and self._rng.random() < self.exploration_epsilon:
            pool = scored[: min(3, len(scored))]
            return self._rng.choice(pool)[1]
        return scored[0][1]

    def update(self, strategy_name: str, trade_pnl: float) -> None:
        key = self._key(strategy_name)
        st = self._state.setdefault(key, StrategyState())
        st.pnl += trade_pnl
        if trade_pnl >= 0:
            st.wins += 1
        else:
            st.losses += 1
        self._persist()

    def stats(self) -> dict[str, StrategyState]:
        return dict(self._state)
