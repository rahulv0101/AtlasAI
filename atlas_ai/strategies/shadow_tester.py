"""ShadowTester: runs evolved strategies in paper mode alongside the main trader.

Tracks paper-only outcomes without affecting live positions. Results feed into
the promotion pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.strategy_factory import StrategyFactory


@dataclass
class ShadowOutcome:
    chromosome_uid: str
    strategy_name: str
    side: str
    confidence: float
    entry_price: float
    exit_price: float
    pnl: float
    bars_held: int
    exit_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ShadowTester:
    """Paper-mode shadow tracker for evolved strategies.

    Each candidate gets a virtual position. When the candidate signals flat
    or a new signal in the opposite direction, the virtual position is closed
    and an outcome is recorded.
    """

    def __init__(self) -> None:
        self.positions: dict[str, dict[str, Any]] = {}
        self.outcomes: list[ShadowOutcome] = []

    def reset(self) -> None:
        self.positions.clear()
        self.outcomes.clear()

    def step(
        self,
        chromosome: StrategyChromosome,
        df: pd.DataFrame,
        current_price: float,
    ) -> ShadowOutcome | None:
        """Process one bar for a shadow candidate.

        Returns an outcome if a virtual position was closed this bar.
        """
        strategy = StrategyFactory.build(chromosome)
        signal = strategy.generate_signal(df)
        uid = chromosome.uid
        pos = self.positions.get(uid)

        # Open new position if none exists and signal is directional
        if pos is None and signal.side in ("buy", "sell"):
            self.positions[uid] = {
                "side": signal.side,
                "entry_price": current_price,
                "confidence": signal.confidence,
                "strategy_name": strategy.name,
                "bars_held": 0,
                "sl_points": signal.stop_loss_points,
                "tp_points": signal.take_profit_points,
            }
            return None

        if pos is None:
            return None

        # Update bars held
        pos["bars_held"] += 1

        # Check exit conditions
        exited = False
        exit_reason = ""
        pnl = 0.0

        # Signal reversal or flat
        if signal.side == "flat" or signal.side != pos["side"]:
            exited = True
            exit_reason = "signal_reversal" if signal.side != "flat" else "signal_flat"

        # Stop loss
        if not exited and pos["sl_points"] > 0:
            if pos["side"] == "buy" and current_price <= pos["entry_price"] - pos["sl_points"]:
                exited = True
                exit_reason = "stop_loss"
            elif pos["side"] == "sell" and current_price >= pos["entry_price"] + pos["sl_points"]:
                exited = True
                exit_reason = "stop_loss"

        # Take profit
        if not exited and pos["tp_points"] > 0:
            if pos["side"] == "buy" and current_price >= pos["entry_price"] + pos["tp_points"]:
                exited = True
                exit_reason = "take_profit"
            elif pos["side"] == "sell" and current_price <= pos["entry_price"] - pos["tp_points"]:
                exited = True
                exit_reason = "take_profit"

        # Max bars
        if not exited and pos["bars_held"] >= 12:
            exited = True
            exit_reason = "max_bars"

        if exited:
            if pos["side"] == "buy":
                pnl = current_price - pos["entry_price"]
            else:
                pnl = pos["entry_price"] - current_price

            outcome = ShadowOutcome(
                chromosome_uid=uid,
                strategy_name=pos["strategy_name"],
                side=pos["side"],
                confidence=pos["confidence"],
                entry_price=pos["entry_price"],
                exit_price=current_price,
                pnl=pnl,
                bars_held=pos["bars_held"],
                exit_reason=exit_reason,
            )
            self.outcomes.append(outcome)
            del self.positions[uid]
            return outcome

        return None

    def summary(self, chromosome_uid: str) -> dict[str, Any]:
        """Return win rate, PnL, and trade count for a specific candidate."""
        rows = [o for o in self.outcomes if o.chromosome_uid == chromosome_uid]
        if not rows:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pnl": 0.0}
        wins = sum(1 for o in rows if o.pnl > 0)
        losses = sum(1 for o in rows if o.pnl <= 0)
        net_pnl = sum(o.pnl for o in rows)
        return {
            "trades": len(rows),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(rows),
            "net_pnl": net_pnl,
        }

    def all_summaries(self) -> dict[str, dict[str, Any]]:
        """Return summaries keyed by chromosome UID."""
        uids = {o.chromosome_uid for o in self.outcomes}
        return {uid: self.summary(uid) for uid in uids}

