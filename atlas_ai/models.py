from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SignalSide = Literal["buy", "sell", "flat"]


@dataclass
class TradeSignal:
    strategy: str
    side: SignalSide
    confidence: float
    stop_loss_points: float
    take_profit_points: float


@dataclass
class PositionSnapshot:
    ticket: int
    symbol: str
    side: SignalSide
    volume: float
    entry_price: float
    profit: float


@dataclass
class StrategyScore:
    name: str
    score: float
