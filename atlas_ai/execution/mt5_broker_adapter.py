from __future__ import annotations

from typing import Any
import MetaTrader5 as mt5

from atlas_ai.execution.broker_adapter import BrokerAdapter, BrokerOrderResponse
from atlas_ai.execution.trade_executor import TradeExecutor
from atlas_ai.models import TradeSignal


class MT5BrokerAdapter(BrokerAdapter):
    """Production BrokerAdapter that delegates to TradeExecutor for real MT5 execution."""

    def __init__(self, executor: TradeExecutor) -> None:
        self._executor = executor

    def submit_order(self, *, idempotency_key: str, request: dict[str, Any]) -> BrokerOrderResponse:
        """Translate ExecutionGate request into TradeExecutor call."""
        payload = dict(request or {})
        # Reconstruct a minimal TradeSignal from the payload_summary or request fields
        side = str(payload.get("side", payload.get("action_type", "flat"))).lower()
        if side not in {"buy", "sell"}:
            return BrokerOrderResponse(
                ok=False,
                status="rejected",
                broker_order_id="",
                message="invalid_side",
            )

        strategy = str(payload.get("strategy", payload.get("setup_family", "unknown")))
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        stop_loss_points = float(payload.get("stop_loss_points", 0.0) or 0.0)
        take_profit_points = float(payload.get("take_profit_points", 0.0) or 0.0)
        volume = float(payload.get("size", payload.get("volume", 0.0)) or 0.0)
        decision_id = str(payload.get("decision_id", idempotency_key) or idempotency_key)

        signal = TradeSignal(
            strategy=strategy,
            side=side,  # type: ignore[arg-type]
            confidence=confidence,
            stop_loss_points=stop_loss_points,
            take_profit_points=take_profit_points,
        )

        result = self._executor.execute(signal, volume=volume, decision_id=decision_id)

        if result.success:
            broker_order_id = str(result.order_id or result.deal_id or "")
            if not broker_order_id:
                return BrokerOrderResponse(
                    ok=False,
                    status="unknown",
                    broker_order_id="",
                    message=f"missing_broker_confirmation retcode={result.retcode} comment={result.comment}",
                )
            return BrokerOrderResponse(
                ok=True,
                status="filled",
                broker_order_id=broker_order_id,
                message=result.comment,
                fill_qty=volume,
                avg_fill_price=0.0,
            )
        return BrokerOrderResponse(
            ok=False,
            status="rejected",
            broker_order_id="",
            message=f"retcode={result.retcode} comment={result.comment}",
        )

    def cancel_order(self, *, broker_order_id: str) -> BrokerOrderResponse:
        # TradeExecutor does not expose cancel directly; MT5 positions are managed
        # through the position lifecycle. Return not-supported for now.
        return BrokerOrderResponse(
            ok=False,
            status="cancel_failed",
            broker_order_id=broker_order_id,
            message="mt5_cancel_not_implemented_use_position_close",
        )

    def query_order_status(self, *, broker_order_id: str) -> BrokerOrderResponse:
        # Best-effort: we do not maintain a separate order book in this adapter.
        return BrokerOrderResponse(
            ok=True,
            status="unknown",
            broker_order_id=broker_order_id,
            message="mt5_status_query_not_implemented",
        )

    def list_open_orders(self) -> list[dict[str, Any]]:
        if mt5.terminal_info() is None and not mt5.initialize():
            raise RuntimeError(f"mt5_initialize_failed:{mt5.last_error()}")
        try:
            mt5.symbol_select(self._executor.settings.symbol, True)
        except Exception:
            pass
        positions_raw = mt5.positions_get(symbol=self._executor.settings.symbol)
        if positions_raw is None:
            raise RuntimeError(f"mt5_positions_get_failed:{mt5.last_error()}")
        positions = list(positions_raw or [])
        out: list[dict[str, Any]] = []
        for pos in positions:
            pos_type = int(getattr(pos, "type", -1))
            side = "buy" if pos_type == int(getattr(mt5, "POSITION_TYPE_BUY", 0)) else "sell"
            out.append(
                {
                    "broker_order_id": str(getattr(pos, "ticket", "") or ""),
                    "instrument": str(getattr(pos, "symbol", self._executor.settings.symbol) or self._executor.settings.symbol),
                    "side": side,
                    "size": float(getattr(pos, "volume", 0.0) or 0.0),
                    "status": "open",
                    "profit": float(getattr(pos, "profit", 0.0) or 0.0),
                    "sl": float(getattr(pos, "sl", 0.0) or 0.0),
                    "tp": float(getattr(pos, "tp", 0.0) or 0.0),
                    "magic": int(getattr(pos, "magic", 0) or 0),
                    "comment": str(getattr(pos, "comment", "") or ""),
                }
            )
        return out

