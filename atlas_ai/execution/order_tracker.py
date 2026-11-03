from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any


TERMINAL_STATUSES = {"filled", "cancelled", "rejected", "failed"}


@dataclass
class TrackedOrder:
    internal_request_id: str
    broker_order_id: str
    environment: str
    instrument: str
    side: str
    size: float
    submitted_at: float
    last_update_at: float
    status: str = "created"
    fill_qty: float = 0.0
    avg_fill_price: float = 0.0
    rejection_reason: str = ""
    last_broker_payload_ref: str = ""
    reconciliation_status: str = "clear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderTracker:
    def __init__(self) -> None:
        self._orders: dict[str, TrackedOrder] = {}
        self._by_broker_id: dict[str, str] = {}

    def restore(self, payload: dict[str, Any]) -> None:
        self._orders = {}
        self._by_broker_id = {}
        for request_id, raw in dict(payload or {}).items():
            try:
                o = TrackedOrder(
                    internal_request_id=str(raw.get("internal_request_id", request_id)),
                    broker_order_id=str(raw.get("broker_order_id", "")),
                    environment=str(raw.get("environment", "paper")),
                    instrument=str(raw.get("instrument", "")),
                    side=str(raw.get("side", "")),
                    size=float(raw.get("size", 0.0) or 0.0),
                    submitted_at=float(raw.get("submitted_at", time.time()) or time.time()),
                    last_update_at=float(raw.get("last_update_at", time.time()) or time.time()),
                    status=str(raw.get("status", "unknown")),
                    fill_qty=float(raw.get("fill_qty", 0.0) or 0.0),
                    avg_fill_price=float(raw.get("avg_fill_price", 0.0) or 0.0),
                    rejection_reason=str(raw.get("rejection_reason", "")),
                    last_broker_payload_ref=str(raw.get("last_broker_payload_ref", "")),
                    reconciliation_status=str(raw.get("reconciliation_status", "clear")),
                )
            except Exception:
                continue
            self._orders[o.internal_request_id] = o
            if o.broker_order_id:
                self._by_broker_id[o.broker_order_id] = o.internal_request_id

    def snapshot(self) -> dict[str, Any]:
        return {rid: order.to_dict() for rid, order in self._orders.items()}

    def get_by_request(self, request_id: str) -> TrackedOrder | None:
        return self._orders.get(str(request_id))

    def upsert_submission(
        self,
        *,
        request_id: str,
        broker_order_id: str,
        environment: str,
        instrument: str,
        side: str,
        size: float,
        status: str,
        rejection_reason: str = "",
        reconciliation_status: str = "clear",
    ) -> TrackedOrder:
        now = time.time()
        existing = self._orders.get(request_id)
        if existing is None:
            existing = TrackedOrder(
                internal_request_id=request_id,
                broker_order_id=broker_order_id,
                environment=environment,
                instrument=instrument,
                side=side,
                size=float(size or 0.0),
                submitted_at=now,
                last_update_at=now,
                status=status,
                rejection_reason=rejection_reason,
                reconciliation_status=reconciliation_status,
            )
            self._orders[request_id] = existing
        else:
            existing.broker_order_id = broker_order_id or existing.broker_order_id
            existing.last_update_at = now
            existing.status = status
            existing.rejection_reason = rejection_reason
            existing.reconciliation_status = reconciliation_status
        if existing.broker_order_id:
            self._by_broker_id[existing.broker_order_id] = request_id
        return existing

    def update_status(
        self,
        *,
        request_id: str,
        status: str,
        fill_qty: float = 0.0,
        avg_fill_price: float = 0.0,
        rejection_reason: str = "",
        reconciliation_status: str = "clear",
    ) -> TrackedOrder | None:
        order = self._orders.get(request_id)
        if order is None:
            return None
        order.status = status
        order.last_update_at = time.time()
        order.fill_qty = float(fill_qty or 0.0)
        order.avg_fill_price = float(avg_fill_price or 0.0)
        order.rejection_reason = rejection_reason
        order.reconciliation_status = reconciliation_status
        return order

    def mark_reconciliation_required(self, request_id: str, *, reason: str) -> TrackedOrder | None:
        order = self._orders.get(request_id)
        if order is None:
            return None
        order.reconciliation_status = "required"
        order.status = "reconciliation_required"
        order.rejection_reason = reason
        order.last_update_at = time.time()
        return order

    def pending(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.status not in TERMINAL_STATUSES]

    def counts(self) -> dict[str, int]:
        pending = 0
        partial = 0
        recon = 0
        for o in self._orders.values():
            if o.status not in TERMINAL_STATUSES:
                pending += 1
            if o.status == "partially_filled":
                partial += 1
            if o.reconciliation_status == "required" or o.status == "reconciliation_required":
                recon += 1
        return {"pending": pending, "partially_filled": partial, "reconciliation_required": recon}

    def last_order(self) -> TrackedOrder | None:
        if not self._orders:
            return None
        return max(self._orders.values(), key=lambda x: x.last_update_at)

    def known_broker_order_ids(self) -> set[str]:
        return {str(x) for x in self._by_broker_id.keys() if str(x)}

    def get_by_broker_order_id(self, broker_order_id: str) -> TrackedOrder | None:
        rid = self._by_broker_id.get(str(broker_order_id))
        if not rid:
            return None
        return self._orders.get(rid)
