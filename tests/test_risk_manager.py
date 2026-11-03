from atlas_ai.config.settings import Settings
from atlas_ai.risk.risk_manager import RiskManager, RiskSnapshot


def _settings(**overrides):
    base = {
        "hfm_login": 1,
        "hfm_password": "x",
        "hfm_server": "demo",
        "max_risk_per_trade": 0.01,
        "max_daily_drawdown": 0.03,
        "max_open_trades": 2,
        "base_lot": 0.01,
    }
    base.update(overrides)
    return Settings(**base)


def test_can_trade_blocks_on_daily_drawdown():
    risk = RiskManager(_settings(max_daily_drawdown=0.03))
    snap = RiskSnapshot(equity=1000, balance=1000, daily_loss_pct=0.03, open_positions=0)
    assert risk.can_trade(snap) is False


def test_can_trade_blocks_on_open_positions_limit():
    risk = RiskManager(_settings(max_open_trades=2))
    snap = RiskSnapshot(equity=1000, balance=1000, daily_loss_pct=0.01, open_positions=2)
    assert risk.can_trade(snap) is False


def test_evaluate_blocks_on_spread_limit():
    risk = RiskManager(_settings())
    snap = RiskSnapshot(equity=1000, balance=1000, daily_loss_pct=0.0, open_positions=0, spread_points=999)
    allowed, reason = risk.evaluate(snap)
    assert allowed is False
    assert "Spread too high" in reason


def test_lot_size_respects_base_lot_and_formula():
    risk = RiskManager(_settings(max_risk_per_trade=0.01, base_lot=0.01))
    lot = risk.lot_size(equity=10000, stop_loss_points=200, tick_value=1)
    assert lot == 0.5

    fallback = risk.lot_size(equity=10000, stop_loss_points=0, tick_value=1)
    assert fallback == 0.01
