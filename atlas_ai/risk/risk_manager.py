from __future__ import annotations

from dataclasses import dataclass

from atlas_ai.config.settings import Settings, get_active_thresholds


@dataclass
class RiskSnapshot:
    equity: float
    balance: float
    daily_loss_pct: float
    open_positions: int
    spread_points: float = 0.0
    weekly_loss_pct: float = 0.0
    consecutive_losses: int = 0


class RiskManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def can_trade(self, snapshot: RiskSnapshot) -> bool:
        allowed, _ = self.evaluate(snapshot)
        return allowed

    def evaluate(self, snapshot: RiskSnapshot) -> tuple[bool, str]:
        if snapshot.daily_loss_pct >= self.settings.max_daily_drawdown:
            return False, "Daily drawdown limit reached"
        if snapshot.weekly_loss_pct >= getattr(self.settings, "max_weekly_drawdown", 1.0):
            return False, "Weekly drawdown limit reached"
        if snapshot.open_positions >= self.settings.max_open_trades:
            return False, "Max open trades reached"
        if snapshot.spread_points > 0 and snapshot.spread_points > self.settings.max_spread_points:
            return False, f"Spread too high ({snapshot.spread_points:.1f} pts)"
        return True, ""

    def lot_size(self, equity: float, stop_loss_points: float, tick_value: float) -> float:
        thresholds = get_active_thresholds(self.settings)
        # LEARNING MODE: lot size fixed from get_active_thresholds().
        if bool(getattr(self.settings, "learning_mode", False)):
            return float(thresholds["max_lot"])
        risk_money = equity * self.settings.max_risk_per_trade
        if stop_loss_points <= 0 or tick_value <= 0:
            return self.settings.base_lot

        raw = risk_money / (stop_loss_points * tick_value)
        return max(self.settings.base_lot, round(raw, 2))
