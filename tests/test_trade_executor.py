from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from atlas_ai.config.settings import Settings
from atlas_ai.execution.comment_codec import MAX_BROKER_COMMENT_LEN, build_broker_comment, parse_broker_comment
from atlas_ai.execution.trade_executor import TradeExecutor
import atlas_ai.execution.trade_executor as trade_executor_module


def _settings():
    return Settings(
        hfm_login=1,
        hfm_password="x",
        hfm_server="demo",
    )


def test_safe_comment_ascii_and_length():
    ex = TradeExecutor(_settings())
    comment = ex._safe_comment("AtlasAI:evolved_rule(fast=123,slow=999,rsi_buy=44.2)")
    assert len(comment) <= 31
    assert ":" not in comment
    assert "(" not in comment
    assert ")" not in comment
    assert all(ord(c) < 128 for c in comment)


def test_strategy_comment_tag_prioritizes_decision_id():
    ex = TradeExecutor(_settings())
    comment = ex._strategy_comment_tag("gold_micro_impulse", decision_id="ab12cd34")
    assert comment == "AAI_Dab12cd_Sgmi"
    assert len(comment) <= MAX_BROKER_COMMENT_LEN


def test_build_broker_comment_preserves_priority_and_never_overflows():
    comment = build_broker_comment(
        strategy_name="momentum_continuation",
        decision_id="ab12cd34",
        lane_tag="experiment",
        overlay_tag="meta_model",
        regime_tag="late_us",
    )

    assert comment == "AAI_Dab12cd_Smomc_Lexpe_Ometa"
    assert len(comment) <= MAX_BROKER_COMMENT_LEN


def test_parse_broker_comment_round_trips_new_schema():
    parsed = parse_broker_comment("AAI_Dab12cd_Smomc_Lctl_Ometa")

    assert parsed["decision_id"] == "ab12cd"
    assert parsed["strategy"] == "momentum_continuation"
    assert parsed["lane_tag"] == "ctl"
    assert parsed["overlay_tag"] == "meta"


def test_parse_broker_comment_supports_legacy_strategy_tags():
    parsed = parse_broker_comment("AAI_ST_gold_micro_impulse")

    assert parsed["strategy"] == "gold_micro_impulse"


class _SymbolInfo:
    volume_min = 0.01
    volume_max = 50.0
    volume_step = 0.01
    point = 0.01
    digits = 2
    trade_mode = 1
    visible = True
    trade_stops_level = 25
    trade_freeze_level = 0


def test_normalize_volume_aligns_to_step():
    ex = TradeExecutor(_settings())
    normalized = ex._normalize_volume(1.1132, _SymbolInfo())
    assert normalized == 1.11


def test_normalize_volume_respects_max_without_silently_upsizing_to_min():
    ex = TradeExecutor(_settings())
    low = ex._normalize_volume(0.001, _SymbolInfo())
    high = ex._normalize_volume(100.0, _SymbolInfo())
    assert low == 0.0
    assert high == 50.0


def test_preflight_blocks_volume_below_broker_minimum(monkeypatch):
    ex = TradeExecutor(_settings())

    class _Tick:
        ask = 100.05
        bid = 100.0

    class _Account:
        trade_allowed = True
        margin_free = 100000.0

    monkeypatch.setattr(trade_executor_module.mt5, "symbol_info_tick", lambda _symbol: _Tick())
    monkeypatch.setattr(trade_executor_module.mt5, "account_info", lambda: _Account())

    ok, note = ex._preflight_order_request(
        {
            "type": 1,
            "symbol": "XAUUSD",
            "volume": 0.0,
            "price": 100.0,
            "magic": ex.settings.magic_number,
            "comment": "AAI_Dtest_Sgmi",
        },
        _SymbolInfo(),
    )

    assert ok is False
    assert "Volume below min" in note


def test_preflight_blocks_untagged_entry_requests(monkeypatch):
    ex = TradeExecutor(_settings())

    class _Tick:
        ask = 100.05
        bid = 100.0

    class _Account:
        trade_allowed = True
        margin_free = 100000.0

    monkeypatch.setattr(trade_executor_module.mt5, "symbol_info_tick", lambda _symbol: _Tick())
    monkeypatch.setattr(trade_executor_module.mt5, "account_info", lambda: _Account())

    request = {"type": 1, "symbol": "XAUUSD", "volume": 0.01, "price": 100.0, "magic": 0, "comment": ""}
    ok, note = ex._preflight_order_request(request, _SymbolInfo())

    assert ok is False
    assert "magic" in note.lower()

    request["magic"] = ex.settings.magic_number
    ok, note = ex._preflight_order_request(request, _SymbolInfo())

    assert ok is False
    assert "comment" in note.lower()


def test_normalize_stops_pushes_buy_stops_outside_broker_minimum():
    ex = TradeExecutor(_settings())
    price = 100.00
    sl, tp = ex._normalize_stops("buy", price=price, sl=99.90, tp=100.10, symbol_info=_SymbolInfo())
    assert sl <= 99.73
    assert tp >= 100.27


def test_normalize_stops_pushes_sell_stops_outside_broker_minimum():
    ex = TradeExecutor(_settings())
    price = 100.00
    sl, tp = ex._normalize_stops("sell", price=price, sl=100.10, tp=99.90, symbol_info=_SymbolInfo())
    assert sl >= 100.27
    assert tp <= 99.73


def test_normalize_stops_adds_spread_buffer_and_post_rounding_margin():
    ex = TradeExecutor(_settings())
    price = 100.00

    sl, tp = ex._normalize_stops(
        "buy",
        price=price,
        sl=99.80,
        tp=100.20,
        symbol_info=_SymbolInfo(),
        spread_points=40.0,
    )

    min_points = ex._stop_buffer_points(_SymbolInfo(), spread_points=40.0)
    assert ((price - sl) / _SymbolInfo.point) >= min_points
    assert ((tp - price) / _SymbolInfo.point) >= min_points


def test_close_stale_positions_fails_closed_without_explicit_allow(monkeypatch):
    ex = TradeExecutor(_settings())

    def _boom(*args, **kwargs):
        raise AssertionError("positions_get should not be called when allow_close=False")

    monkeypatch.setattr(trade_executor_module.mt5, "positions_get", _boom)
    out = ex.close_stale_positions()
    assert out == []


def test_close_stale_positions_closes_buy_type_zero_with_sell_order(monkeypatch):
    ex = TradeExecutor(_settings())
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    sent: list[dict] = []

    monkeypatch.setattr(trade_executor_module.mt5, "POSITION_TYPE_BUY", 0, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "ORDER_TYPE_BUY", 0, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "ORDER_TYPE_SELL", 1, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "TRADE_ACTION_DEAL", 1, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "ORDER_TIME_GTC", 0, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "ORDER_FILLING_IOC", 1, raising=False)
    monkeypatch.setattr(trade_executor_module.mt5, "TRADE_RETCODE_DONE", 10009, raising=False)
    monkeypatch.setattr(
        trade_executor_module.mt5,
        "positions_get",
        lambda symbol=None: [
            SimpleNamespace(
                ticket=123,
                time=int(yesterday.timestamp()),
                symbol="XAUUSD",
                type=0,
                volume=0.01,
            )
        ],
    )
    monkeypatch.setattr(
        trade_executor_module.mt5,
        "symbol_info_tick",
        lambda symbol: SimpleNamespace(bid=4434.61, ask=4434.95),
    )

    def _order_send(request):
        sent.append(dict(request))
        return SimpleNamespace(retcode=10009, comment="Request executed", order=456, deal=789)

    monkeypatch.setattr(trade_executor_module.mt5, "order_send", _order_send)

    out = ex.close_stale_positions(allow_close=True)

    assert out[0].success is True
    assert sent[0]["type"] == 1
    assert sent[0]["price"] == 4434.61
