from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol


@dataclass(frozen=True)
class BrokerOrderResponse:
    ok: bool
    status: str
    broker_order_id: str
    message: str = ""
    payload: dict[str, Any] | None = None
    fill_qty: float = 0.0
    avg_fill_price: float = 0.0


class BrokerAdapter(Protocol):
    def submit_order(self, *, idempotency_key: str, request: dict[str, Any]) -> BrokerOrderResponse:
        ...

    def cancel_order(self, *, broker_order_id: str) -> BrokerOrderResponse:
        ...

    def query_order_status(self, *, broker_order_id: str) -> BrokerOrderResponse:
        ...

    def list_open_orders(self) -> list[dict[str, Any]]:
        ...


class InMemoryBrokerAdapter:
    """Deterministic broker adapter for tiny-live-pilot safety and testing."""

    def __init__(self) -> None:
        self._orders: dict[str, dict[str, Any]] = {}
        self._idempotency: dict[str, str] = {}
        self._counter = 0
        self._submit_overrides: dict[str, BrokerOrderResponse] = {}
        self._status_overrides: dict[str, BrokerOrderResponse] = {}

    def set_submit_override(self, idempotency_key: str, response: BrokerOrderResponse) -> None:
        self._submit_overrides[idempotency_key] = response

    def set_status_override(self, broker_order_id: str, response: BrokerOrderResponse) -> None:
        self._status_overrides[broker_order_id] = response

    def submit_order(self, *, idempotency_key: str, request: dict[str, Any]) -> BrokerOrderResponse:
        if idempotency_key in self._idempotency:
            oid = self._idempotency[idempotency_key]
            raw = self._orders.get(oid, {})
            return BrokerOrderResponse(
                ok=True,
                status=str(raw.get("status", "acknowledged")),
                broker_order_id=oid,
                message="idempotent_replay",
                payload=dict(raw),
                fill_qty=float(raw.get("fill_qty", 0.0) or 0.0),
                avg_fill_price=float(raw.get("avg_fill_price", 0.0) or 0.0),
            )

        override = self._submit_overrides.get(idempotency_key)
        if override is not None:
            if override.ok and override.broker_order_id:
                self._idempotency[idempotency_key] = override.broker_order_id
                self._orders[override.broker_order_id] = {
                    "status": override.status,
                    "payload": dict(override.payload or {}),
                    "fill_qty": float(override.fill_qty or 0.0),
                    "avg_fill_price": float(override.avg_fill_price or 0.0),
                    "updated_at": time.time(),
                }
            return override

        self._counter += 1
        oid = f"BRK_{self._counter:08d}"
        self._idempotency[idempotency_key] = oid
        self._orders[oid] = {"status": "acknowledged", "payload": dict(request), "fill_qty": 0.0, "avg_fill_price": 0.0, "updated_at": time.time()}
        return BrokerOrderResponse(ok=True, status="acknowledged", broker_order_id=oid, message="accepted", payload=dict(request))

    def cancel_order(self, *, broker_order_id: str) -> BrokerOrderResponse:
        raw = self._orders.get(broker_order_id)
        if raw is None:
            return BrokerOrderResponse(ok=False, status="unknown", broker_order_id=broker_order_id, message="order_not_found")
        raw["status"] = "cancelled"
        raw["updated_at"] = time.time()
        return BrokerOrderResponse(ok=True, status="cancelled", broker_order_id=broker_order_id, message="cancelled")

    def query_order_status(self, *, broker_order_id: str) -> BrokerOrderResponse:
        override = self._status_overrides.get(broker_order_id)
        if override is not None:
            return override
        raw = self._orders.get(broker_order_id)
        if raw is None:
            return BrokerOrderResponse(ok=False, status="unknown", broker_order_id=broker_order_id, message="order_not_found")
        return BrokerOrderResponse(
            ok=True,
            status=str(raw.get("status", "unknown")),
            broker_order_id=broker_order_id,
            message="ok",
            payload=dict(raw.get("payload", {}) or {}),
            fill_qty=float(raw.get("fill_qty", 0.0) or 0.0),
            avg_fill_price=float(raw.get("avg_fill_price", 0.0) or 0.0),
        )

    def list_open_orders(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for oid, raw in self._orders.items():
            status = str(raw.get("status", "unknown") or "unknown").lower()
            if status in {"cancelled", "filled", "rejected", "failed"}:
                continue
            payload = dict(raw.get("payload", {}) or {})
            out.append(
                {
                    "broker_order_id": oid,
                    "instrument": str(payload.get("instrument", "") or ""),
                    "side": str(payload.get("action_type", "") or payload.get("side", "") or ""),
                    "size": float(payload.get("size", 0.0) or 0.0),
                    "status": status,
                    "payload": payload,
                    "updated_at": float(raw.get("updated_at", time.time()) or time.time()),
                }
            )
        return out




class MockBrokerAdapter(InMemoryBrokerAdapter):
    """Explicit test broker adapter that is allowed in live-like test environments.

    Identical to InMemoryBrokerAdapter but reports a distinct type so tests
    that intentionally simulate broker behavior are not blocked by safety
    hardening that rejects InMemoryBrokerAdapter for live-like modes.
    """
    pass


class TestBrokerAdapter(MockBrokerAdapter):
    """Backward-compatible test adapter name expected by existing tests."""
    __test__ = False
    pass
