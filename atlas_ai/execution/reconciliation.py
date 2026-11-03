from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atlas_ai.execution.broker_adapter import BrokerAdapter
from atlas_ai.execution.order_tracker import OrderTracker


@dataclass(frozen=True)
class ReconciliationResult:
    ok: bool
    status: str
    reason: str
    requires_review: bool
    details: dict[str, Any]


class OrderReconciliation:
    def __init__(self, adapter: BrokerAdapter, tracker: OrderTracker) -> None:
        self._adapter = adapter
        self._tracker = tracker

    def reconcile_request(self, request_id: str) -> ReconciliationResult:
        tracked = self._tracker.get_by_request(request_id)
        if tracked is None:
            return ReconciliationResult(False, "unknown", "tracked_order_not_found", True, {})
        if not tracked.broker_order_id:
            self._tracker.mark_reconciliation_required(request_id, reason="missing_broker_order_id")
            return ReconciliationResult(False, "reconciliation_required", "missing_broker_order_id", True, {})
        broker = self._adapter.query_order_status(broker_order_id=tracked.broker_order_id)
        if not broker.ok:
            self._tracker.mark_reconciliation_required(request_id, reason=f"broker_query_failed:{broker.message}")
            return ReconciliationResult(False, "reconciliation_required", f"broker_query_failed:{broker.message}", True, {"broker_status": broker.status})
        self._tracker.update_status(
            request_id=request_id,
            status=str(broker.status or "unknown"),
            fill_qty=float(broker.fill_qty or 0.0),
            avg_fill_price=float(broker.avg_fill_price or 0.0),
            rejection_reason=str(broker.message or ""),
            reconciliation_status="clear",
        )
        return ReconciliationResult(True, str(broker.status or "unknown"), "reconciled", False, {"broker_status": broker.status})
