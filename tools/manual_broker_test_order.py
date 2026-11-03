from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.config.settings import load_settings
from atlas_ai.execution.broker import HFMConnector
from atlas_ai.execution.execution_gate import ExecutionGate
from atlas_ai.execution.mt5_broker_adapter import MT5BrokerAdapter
from atlas_ai.execution.trade_executor import TradeExecutor
from atlas_ai.memory.knowledge_store import KnowledgeStore


def _position_row(pos: Any) -> dict[str, Any]:
    pos_type = int(getattr(pos, "type", -1))
    return {
        "ticket": int(getattr(pos, "ticket", 0) or 0),
        "symbol": str(getattr(pos, "symbol", "") or ""),
        "type": pos_type,
        "side": "buy" if pos_type == int(mt5.POSITION_TYPE_BUY) else "sell",
        "volume": float(getattr(pos, "volume", 0.0) or 0.0),
        "price_open": float(getattr(pos, "price_open", 0.0) or 0.0),
        "price_current": float(getattr(pos, "price_current", 0.0) or 0.0),
        "profit": float(getattr(pos, "profit", 0.0) or 0.0),
        "sl": float(getattr(pos, "sl", 0.0) or 0.0),
        "tp": float(getattr(pos, "tp", 0.0) or 0.0),
        "magic": int(getattr(pos, "magic", 0) or 0),
        "comment": str(getattr(pos, "comment", "") or ""),
        "time": int(getattr(pos, "time", 0) or 0),
    }


def _close_position(pos: Any, settings: Any) -> dict[str, Any]:
    tick = mt5.symbol_info_tick(str(getattr(pos, "symbol", settings.symbol) or settings.symbol))
    if tick is None:
        return {"ok": False, "reason": "No tick for close", "last_error": str(mt5.last_error())}
    close_type = mt5.ORDER_TYPE_SELL if int(getattr(pos, "type", -1)) == int(mt5.POSITION_TYPE_BUY) else mt5.ORDER_TYPE_BUY
    price = float(tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": str(getattr(pos, "symbol", settings.symbol) or settings.symbol),
        "position": int(getattr(pos, "ticket", 0) or 0),
        "volume": float(getattr(pos, "volume", 0.0) or 0.0),
        "type": close_type,
        "price": price,
        "deviation": max(1, int(settings.max_slippage_points)),
        "magic": int(settings.magic_number),
        "comment": "AAI_manual_test_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        return {"ok": False, "reason": "order_send returned None", "last_error": str(mt5.last_error()), "request": request}
    return {
        "ok": int(getattr(result, "retcode", 0) or 0) == int(mt5.TRADE_RETCODE_DONE),
        "retcode": int(getattr(result, "retcode", 0) or 0),
        "comment": str(getattr(result, "comment", "") or ""),
        "order": int(getattr(result, "order", 0) or 0),
        "deal": int(getattr(result, "deal", 0) or 0),
        "price": float(getattr(result, "price", 0.0) or 0.0),
        "volume": float(getattr(result, "volume", 0.0) or 0.0),
        "last_error": str(mt5.last_error()),
        "request": request,
    }


def main() -> None:
    settings = load_settings()
    store = KnowledgeStore(settings.storage_dir)
    connector = HFMConnector(settings)
    output: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "symbol": settings.symbol,
        "side": "buy",
        "size": 0.01,
        "stop_loss_points": 150.0,
        "take_profit_points": 240.0,
        "steps": [],
    }
    try:
        connector.connect()
        output["steps"].append({"step": "broker_connect", "ok": True, "last_error": str(mt5.last_error())})
        before_positions = list(mt5.positions_get(symbol=settings.symbol) or [])
        before_tickets = {int(getattr(p, "ticket", 0) or 0) for p in before_positions}
        output["positions_before"] = [_position_row(p) for p in before_positions]

        gate = ExecutionGate(state_store_path=str(Path(settings.storage_dir) / "automation_state.json"))
        gate.set_broker_adapter(MT5BrokerAdapter(TradeExecutor(settings)))
        gate.set_execution_environment("demo")
        runtime_flags = dict(store.load_runtime_flags() or {})
        runtime_flags["execute_trades"] = True
        runtime_flags["effective_mode"] = "live"
        runtime_flags["market_state_mode"] = "broker_runtime"
        payload_summary = (
            "instrument=XAUUSD setup_family=continuation setup=continuation "
            "strategy=manual_broker_test side=buy size=0.0100 confidence=0.990 "
            "stop_loss_points=150.00 take_profit_points=240.00 "
            "data_source=broker_runtime environment_mode=demo decision_id=manual_broker_test"
        )
        req = gate.create_request(
            environment="demo",
            action_type="place_trade",
            payload_summary=payload_summary,
            requested_by="operator_manual_broker_test",
            dry_run=False,
            auto_generated=False,
            request_origin="operator_manual_test",
            strict_runtime_checks=False,
        )
        output["request"] = req
        req_id = str(req.get("id", "") or "")
        if req_id:
            approved = gate.approve_request(
                req_id,
                approved_by="operator_explicit_manual_broker_test",
                note="single 0.01 demo broker mechanics test",
                explicit_live_confirmation=True,
            )
            output["approval"] = approved
            result = gate.execute(
                req_id,
                runtime_flags=runtime_flags,
                risk_policy={"daily_loss_cap_pct": float(settings.max_daily_drawdown * 100.0)},
            )
            output["execution_gate_result"] = result
        else:
            output["approval"] = None
            output["execution_gate_result"] = {"ok": False, "reason": str(req.get("last_block_reason", "request_not_created"))}

        time.sleep(1.0)
        after_positions = list(mt5.positions_get(symbol=settings.symbol) or [])
        output["positions_after_execution"] = [_position_row(p) for p in after_positions]
        new_positions = [p for p in after_positions if int(getattr(p, "ticket", 0) or 0) not in before_tickets]
        output["new_positions"] = [_position_row(p) for p in new_positions]
        close_results = []
        for pos in new_positions:
            close_results.append({"position": _position_row(pos), "close_result": _close_position(pos, settings)})
        output["close_results"] = close_results
        time.sleep(1.0)
        final_positions = list(mt5.positions_get(symbol=settings.symbol) or [])
        output["positions_after_close"] = [_position_row(p) for p in final_positions]
        output["finished_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # noqa: BLE001 - operator needs exact failure.
        output["error"] = f"{exc.__class__.__name__}: {exc}"
        output["last_error"] = str(mt5.last_error())
        output["finished_at"] = datetime.now(timezone.utc).isoformat()
    finally:
        try:
            connector.shutdown()
        finally:
            print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
