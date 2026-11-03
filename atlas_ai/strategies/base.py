from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from atlas_ai.models import TradeSignal


class Strategy(ABC):
    name: str

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        raise NotImplementedError
