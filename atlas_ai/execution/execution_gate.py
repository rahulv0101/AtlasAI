from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable

from atlas_ai.execution.audit_log import AuditLog
from atlas_ai.execution.automation_policy import AutomationPolicy
from atlas_ai.execution.automation_state_store import AutomationStateStore
from atlas_ai.execution.broker_adapter import InMemoryBrokerAdapter, TestBrokerAdapter
from atlas_ai.execution.kill_switch import KillSwitch
from atlas_ai.execution.order_tracker import OrderTracker, TERMINAL_STATUSES
from atlas_ai.execution.reconciliation import OrderReconciliation
from atlas_ai.execution.request_validator import validate_execution_request
from atlas_ai.execution.session_control import DayTradingSessionControl, SessionWindow


REJECTION_REASON_CODES = {
    "APPROVAL_MISSING",
    "BROKER_RECONCILE_MISMATCH",
    "CONFLICTING_ACTIVE_EXECUTION",
    "DAILY_LOSS_CAP",
    "EXECUTION_GATE_DISABLED",
    "KILL_SWITCH_ACTIVE",
    "MAX_TRADES_REACHED",
    "OPERATOR_ABSENT",
    "ORPHAN_ORDER_DETECTED",
    "ROLLOUT_STAGE_BLOCKED",
    "SESSION_CLOSED",
    "SLEEP_MODE",
    "SPREAD_TOO_WIDE",
    "STALE_RUNTIME",
    "TINY_LIVE_PILOT_LIMIT",
    "VALIDATION_FAILED",
    "WEEKLY_DRAWDOWN_CAP",
}

_REJECTION_REASON_ALIASES = {
    "approval_missing": "APPROVAL_MISSING",
    "conflicting_active_execution": "CONFLICTING_ACTIVE_EXECUTION",
    "daily_loss_cap_hit": "DAILY_LOSS_CAP",
    "daily_loss_limit_reached": "DAILY_LOSS_CAP",
    "execution_gate_disabled": "EXECUTION_GATE_DISABLED",
    "freshness_stale": "STALE_RUNTIME",
    "kill_switch_active": "KILL_SWITCH_ACTIVE",
    "max_trades_per_day_reached": "MAX_TRADES_REACHED",
    "missing_runtime_freshness": "STALE_RUNTIME",
    "operator_not_present": "OPERATOR_ABSENT",
    "orphan_orders_pending_review": "ORPHAN_ORDER_DETECTED",
    "reconciliation_required": "BROKER_RECONCILE_MISMATCH",
    "sleep_mode_active_automation_blocked": "SLEEP_MODE",
    "sleep_mode_active_execution_blocked": "SLEEP_MODE",
    "spread_too_wide": "SPREAD_TOO_WIDE",
    "weekly_drawdown_cap_hit": "WEEKLY_DRAWDOWN_CAP",
}


@dataclass
class TinyLivePilotConfig:
    configured: bool = False
    enabled: bool = False
    approved: bool = False
    allowed_instrument: str = "XAUUSD"
    allowed_setup_family: str = "continuation"
    size_cap: float = 0.01
    max_trades_per_day: int = 2
    daily_loss_cap: float = 0.6
    max_consecutive_losses: int = 1
    rollout_stage_required: str = "limited_live_pilot"
    approval_required_per_request: bool = True
    pause_on_forced_flat: bool = True
    pause_on_blocker_threshold: float = 0.7
    pause_on_stability_drop: float = 0.45
    pause_on_overreaction: float = 0.7
    pause_on_degrading_performance: bool = True
    paused: bool = False
    pause_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionGate:
    _AUTONOMY_MODES = {
        "observe_only",
        "recommend_only",
        "guarded_autonomous_demo",
        "guarded_execution",
        "full_live_disabled_by_default",
    }

    def __init__(self, *, state_store_path: str | None = None) -> None:
        self._requests: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._gate_enabled = False
        self._kill_switch = KillSwitch()
        self._audit = AuditLog()
        self._active_request_id: str | None = None
        self._rollout_stage = "paper_only"
        self._execution_environment = "paper"
        self._rollout_stage_updated_at = time.time()
        self._regression_triggered = False
        self._regression_reason = ""
        self._last_execution_result: dict[str, Any] | None = None
        self._automation_profiles: dict[str, AutomationPolicy] = {}
        self._active_automation_policy_id: str = ""
        self._automation_paused = False
        self._automation_pause_reason = ""
        self._automation_paused_by_sleep = False
        self._automation_paused_before_sleep = False
        self._automation_pause_reason_before_sleep = ""
        self._automation_explicit_resume_active = False
        self._automation_explicit_resume_at = 0.0
        self._automation_explicit_resume_policy_id = ""
        self._last_automation_block_reason = ""
        self._last_automation_event: dict[str, Any] | None = None
        self._last_execution_truth: dict[str, Any] | None = None
        self._last_auto_signal_signature = ""
        self._auto_signal_history: dict[str, float] = {}
        self._last_gate_decision: dict[str, Any] = {}
        self._last_rejection_reason: dict[str, Any] | None = None
        self._current_gate_request_id = ""
        self._autonomy_mode = "recommend_only"
        self._last_autonomy_cycle: dict[str, Any] | None = None
        self._sleep_mode_active = False
        self._sleep_mode_reason = ""
        self._sleep_mode_activated_at: float | None = None
        self._sleep_alert_count = 0
        self._operator_present = False
        self._session_override_active = False
        self._session_override_reason = ""
        self._session_override_effective_session = ""
        self._session_override_effective_minutes_to_close = 0
        self._session_override_last_block_reason = ""
        self._session_trade_counts: dict[str, int] = {}
        self._recovery_mode = "fresh_start"
        self._recovery_reason = "no_state_loaded"
        self._recovery_requires_explicit_resume = False
        self._restored_policy_id = ""
        self._last_shutdown_clean = True
        self._last_shared_state_sync_ok = True
        self._last_shared_state_sync_reason = ""
        self._session_control = DayTradingSessionControl()
        if os.getenv("PYTEST_CURRENT_TEST"):
            self._session_control._sessions = [  # type: ignore[attr-defined]
                SessionWindow(
                    name="test_always_open",
                    start="00:00",
                    end="23:59",
                    timezone_name="UTC",
                    no_trade_buffer_minutes=0,
                    weekdays=(0, 1, 2, 3, 4, 5, 6),
                )
            ]
        self._broker_adapter = TestBrokerAdapter() if os.getenv("PYTEST_CURRENT_TEST") else InMemoryBrokerAdapter()
        self._order_tracker = OrderTracker()
        self._reconciler = OrderReconciliation(self._broker_adapter, self._order_tracker)
        self._reconciliation_status = "unknown"
        self._last_reconciled_at = 0.0
        self._mismatch_detected = False
        self._reconciliation_interval_seconds = self._resolve_reconciliation_interval_seconds()
        self._reconciliation_stop = threading.Event()
        self._reconciliation_thread: threading.Thread | None = None
        self._reconciliation_lock = threading.Lock()
        self._orphan_orders: dict[str, dict[str, Any]] = {}
        self._tiny_live_pilot = TinyLivePilotConfig()
        self._tiny_live_pilot_attempts_today = 0
        self._tiny_live_pilot_trade_count = 0
        self._tiny_live_pilot_consecutive_losses = 0
        self._tiny_live_pilot_daily_loss_used = 0.0
        self._tiny_live_pilot_last_block_reason = ""
        self._tiny_live_pilot_last_attempt: dict[str, Any] | None = None
        default_path = Path.cwd() / "storage" / "automation_state.json"
        resolved_path = state_store_path
        if resolved_path is None and os.getenv("PYTEST_CURRENT_TEST"):
            resolved_path = str(Path(tempfile.gettempdir()) / f"atlas_automation_state_{uuid.uuid4().hex}.json")
        self._state_store = AutomationStateStore(resolved_path or str(default_path))
        self._decision_journal_path = Path(self._state_store.path).with_name("execution_decision_journal.jsonl")
        self._load_state()
        if self._adapter_type() != "InMemoryBrokerAdapter":
            self._start_reconciliation_timer_if_enabled()

    _STAGE_ORDER = [
        "paper_only",
        "approved_paper_execution",
        "limited_live_pilot",
        "restricted_live_operations",
        "scaled_live_reviewed",
    ]
    _SESSION_OVERRIDE_ELIGIBLE_REASONS = {
        "outside_active_session",
        "near_session_open_no_new_trades",
        "near_session_close_no_new_trades",
        "early_close_no_new_trades",
        "insufficient_time_to_close",
    }
    _AUTO_SIGNAL_DEDUPE_TTL_SECONDS = 90.0
    _AUTO_SIGNAL_DEDUPE_MAX = 256
    _AUTO_SIGNAL_NON_RECORD_REASONS = {
        "missing_runtime_freshness",
        "invalid_runtime_freshness",
        "freshness_stale",
        "operator_not_present",
        "sleep_mode_active_automation_blocked",
        "execution_gate_disabled",
        "reconciliation_required",
        "orphan_orders_pending_review",
        "outside_active_session",
        "near_session_open_no_new_trades",
        "near_session_close_no_new_trades",
        "early_close_no_new_trades",
        "insufficient_time_to_close",
    }
    _STALE_RESUME_SOFT_PAUSE_REASONS = {
        "operator_pause",
        "operator_requested",
        "weekend_learning_only_operator_requested",
    }

    def _clear_stale_pilot_pause_under_explicit_resume(self, *, event: str) -> bool:
        if not bool(self._automation_explicit_resume_active):
            return False
        if self._automation_paused or self._sleep_mode_active or self._kill_switch.status().active:
            return False
        if not (self._tiny_live_pilot.enabled and self._tiny_live_pilot.approved):
            return False
        reason = str(self._tiny_live_pilot.pause_reason or "")
        if reason not in self._STALE_RESUME_SOFT_PAUSE_REASONS:
            return False
        self._tiny_live_pilot.paused = False
        self._tiny_live_pilot.pause_reason = ""
        self._audit.append(
            "stale_pilot_pause_cleared_under_explicit_resume",
            "",
            {"event": event, "previous_reason": reason},
        )
        return True

    def _apply_sticky_resume_runtime_override(self, *, event: str) -> bool:
        flags = self._load_runtime_flags_for_state_guard()
        if not bool(flags.get("automation_resume_sticky", False)):
            return False
        if self._sleep_mode_active or self._kill_switch.status().active:
            return False
        policy = self._automation_profiles.get(self._active_automation_policy_id) if self._active_automation_policy_id else None
        if not (policy and policy.enabled and policy.environment in {"live", "both"}):
            return False
        if self._automation_paused and self._automation_pause_reason != "unclean_shutdown_forced_pause":
            return False
        changed = False
        if self._automation_paused and self._automation_pause_reason == "unclean_shutdown_forced_pause":
            self._automation_paused = False
            self._automation_pause_reason = ""
            changed = True
        if not self._automation_explicit_resume_active or self._automation_explicit_resume_policy_id != policy.id:
            self._automation_explicit_resume_active = True
            self._automation_explicit_resume_at = self._automation_explicit_resume_at or time.time()
            self._automation_explicit_resume_policy_id = policy.id
            changed = True
        if self._clear_stale_pilot_pause_under_explicit_resume(event=event):
            changed = True
        if changed:
            self._audit.append(
                "sticky_resume_runtime_override_applied",
                "",
                {"event": event, "policy_id": policy.id, "adapter_type": self._adapter_type()},
            )
        return changed

    def _export_state(self) -> dict[str, Any]:
        self._apply_sticky_resume_runtime_override(event="export_state")
        self._clear_stale_pilot_pause_under_explicit_resume(event="export_state")
        return {
            "gate_enabled": self._gate_enabled,
            "rollout_stage": self._rollout_stage,
            "execution_environment": self._execution_environment,
            "rollout_stage_updated_at": self._rollout_stage_updated_at,
            "regression_triggered": self._regression_triggered,
            "regression_reason": self._regression_reason,
            "last_execution_result": dict(self._last_execution_result or {}),
            "last_execution_truth": dict(self._last_execution_truth or {}),
            "kill_switch": {
                "active": self._kill_switch.status().active,
                "reason": self._kill_switch.status().reason,
                "changed_at": self._kill_switch.status().changed_at,
            },
            "automation_profiles": {k: v.to_dict() for k, v in self._automation_profiles.items()},
            "active_automation_policy_id": self._active_automation_policy_id,
            "automation_paused": self._automation_paused,
            "automation_pause_reason": self._automation_pause_reason if self._automation_paused else "",
            "automation_paused_by_sleep": self._automation_paused_by_sleep,
            "automation_paused_before_sleep": self._automation_paused_before_sleep,
            "automation_pause_reason_before_sleep": self._automation_pause_reason_before_sleep,
            "automation_explicit_resume_active": bool(self._automation_explicit_resume_active),
            "automation_explicit_resume_at": float(self._automation_explicit_resume_at or 0.0),
            "automation_explicit_resume_policy_id": str(self._automation_explicit_resume_policy_id or ""),
            "last_automation_block_reason": self._last_automation_block_reason,
            "last_automation_event": dict(self._last_automation_event or {}),
            "last_execution_truth": dict(self._last_execution_truth or {}),
            "last_gate_decision": dict(self._last_gate_decision or {}),
            "last_rejection_reason": dict(self._last_rejection_reason or {}),
            "last_auto_signal_signature": self._last_auto_signal_signature,
            "auto_signal_history": dict(self._auto_signal_history or {}),
            "autonomy_mode": self._autonomy_mode,
            "last_autonomy_cycle": dict(self._last_autonomy_cycle or {}),
            "sleep_mode_active": self._sleep_mode_active,
            "sleep_mode_reason": self._sleep_mode_reason,
            "sleep_mode_activated_at": self._sleep_mode_activated_at,
            "sleep_alert_count": self._sleep_alert_count,
            "operator_present": self._operator_present,
            "session_trade_counts": dict(self._session_trade_counts or {}),
            "last_shutdown_clean": self._last_shutdown_clean,
            "session_control": self._session_control.to_dict(runtime_flags={}),
            "adapter_type": self._adapter_type(),
            "reconciliation_status": self._reconciliation_status,
            "last_reconciled_at": float(self._last_reconciled_at or 0.0),
            "mismatch_detected": bool(self._mismatch_detected),
            "tiny_live_pilot": self._tiny_live_pilot.to_dict(),
            "tiny_live_pilot_attempts_today": self._tiny_live_pilot_attempts_today,
            "tiny_live_pilot_trade_count": self._tiny_live_pilot_trade_count,
            "tiny_live_pilot_consecutive_losses": self._tiny_live_pilot_consecutive_losses,
            "tiny_live_pilot_daily_loss_used": self._tiny_live_pilot_daily_loss_used,
            "tiny_live_pilot_last_block_reason": self._tiny_live_pilot_last_block_reason,
            "tiny_live_pilot_last_attempt": dict(self._tiny_live_pilot_last_attempt or {}),
            "order_tracker": self._order_tracker.snapshot(),
            "orphan_orders": dict(self._orphan_orders or {}),
            "saved_at": time.time(),
        }

    def _persist_state(self, *, event: str) -> None:
        if self._adapter_type() == "InMemoryBrokerAdapter":
            self._audit.append(
                "automation_state_save_skipped",
                "",
                {"event": event, "reason": "in_memory_adapter_not_authoritative_for_shared_state"},
            )
            return
        payload = self._export_state()
        runtime_flags = self._load_runtime_flags_for_state_guard()
        sticky_resume = bool(runtime_flags.get("automation_resume_sticky", False))
        payload_kill = bool(dict(payload.get("kill_switch", {}) or {}).get("active", False))
        payload_sleep = bool(payload.get("sleep_mode_active", False))
        payload_policy = str(
            payload.get("active_automation_policy_id", "")
            or payload.get("automation_explicit_resume_policy_id", "")
            or ""
        ).strip().lower()
        if (
            sticky_resume
            and payload_policy
            and bool(payload.get("automation_paused", False))
            and str(payload.get("automation_pause_reason", "") or "") == "unclean_shutdown_forced_pause"
            and not payload_kill
            and not payload_sleep
        ):
            payload["automation_paused"] = False
            payload["automation_pause_reason"] = ""
            payload["automation_explicit_resume_active"] = True
            payload["automation_explicit_resume_at"] = time.time()
            payload["automation_explicit_resume_policy_id"] = payload_policy
            self._automation_paused = False
            self._automation_pause_reason = ""
            self._automation_explicit_resume_active = True
            self._automation_explicit_resume_at = float(payload["automation_explicit_resume_at"])
            self._automation_explicit_resume_policy_id = payload_policy
            self._audit.append(
                "unclean_pause_blocked_by_sticky_resume",
                "",
                {"event": event, "policy_id": payload_policy},
            )
        if sticky_resume and not payload_kill and not payload_sleep:
            pilot = dict(payload.get("tiny_live_pilot", {}) or {})
            pilot_reason = str(pilot.get("pause_reason", "") or "")
            if pilot_reason in self._STALE_RESUME_SOFT_PAUSE_REASONS:
                pilot["paused"] = False
                pilot["pause_reason"] = ""
                payload["tiny_live_pilot"] = pilot
                self._tiny_live_pilot.paused = False
                self._tiny_live_pilot.pause_reason = ""
                self._audit.append(
                    "stale_pilot_pause_blocked_by_sticky_resume",
                    "",
                    {"event": event, "previous_reason": pilot_reason},
                )
        current, info = self._state_store.load()
        if bool(info.get("ok")):
            current_explicit = bool(current.get("automation_explicit_resume_active", False))
            current_policy = str(current.get("automation_explicit_resume_policy_id", "") or "").strip().lower()
            current_kill = bool(dict(current.get("kill_switch", {}) or {}).get("active", False))
            current_sleep = bool(current.get("sleep_mode_active", False))
            stale_unclean_pause = bool(
                current_explicit
                and current_policy
                and current_policy == payload_policy
                and bool(payload.get("automation_paused", False))
                and str(payload.get("automation_pause_reason", "") or "") == "unclean_shutdown_forced_pause"
                and not current_kill
                and not payload_kill
                and not current_sleep
                and not payload_sleep
            )
            if stale_unclean_pause:
                payload["automation_paused"] = False
                payload["automation_pause_reason"] = ""
                payload["automation_explicit_resume_active"] = True
                payload["automation_explicit_resume_at"] = float(
                    current.get("automation_explicit_resume_at", payload.get("automation_explicit_resume_at", 0.0)) or 0.0
                )
                payload["automation_explicit_resume_policy_id"] = current_policy
                pilot = dict(payload.get("tiny_live_pilot", {}) or {})
                if str(pilot.get("pause_reason", "") or "") == "unclean_shutdown_forced_pause":
                    pilot["paused"] = False
                    pilot["pause_reason"] = ""
                    payload["tiny_live_pilot"] = pilot
                self._audit.append(
                    "stale_unclean_pause_suppressed_by_persisted_resume",
                    "",
                    {"event": event, "policy_id": current_policy},
                )
            stale_pilot_pause = bool(
                current_explicit
                and not bool(payload.get("automation_paused", False))
                and not current_kill
                and not payload_kill
                and not current_sleep
                and not payload_sleep
            )
            if stale_pilot_pause:
                pilot = dict(payload.get("tiny_live_pilot", {}) or {})
                previous_reason = str(pilot.get("pause_reason", "") or "")
                if previous_reason in self._STALE_RESUME_SOFT_PAUSE_REASONS:
                    pilot["paused"] = False
                    pilot["pause_reason"] = ""
                    payload["tiny_live_pilot"] = pilot
                    self._audit.append(
                        "stale_pilot_pause_suppressed_by_persisted_resume",
                        "",
                        {"event": event, "previous_reason": previous_reason},
                    )
        payload["last_shutdown_clean"] = False
        res = self._state_store.save(payload, event=event)
        self._audit.append("automation_state_save", "", {"event": event, "ok": bool(res.get("ok")), "reason": str(res.get("reason", ""))})

    def _load_runtime_flags_for_state_guard(self) -> dict[str, Any]:
        candidates = [self._state_store.path.parent / "runtime_flags.json"]
        if not os.getenv("PYTEST_CURRENT_TEST"):
            candidates.append(Path.cwd() / "storage" / "runtime_flags.json")
        for path in candidates:
            try:
                if not path.exists():
                    continue
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return dict(raw or {})
            except Exception:
                continue
        return {}

    def mark_clean_shutdown(self) -> None:
        self._last_shutdown_clean = True
        payload = self._export_state()
        payload["last_shutdown_clean"] = True
        self._state_store.save(payload, event="clean_shutdown")

    def _load_state(self) -> None:
        payload, info = self._state_store.load()
        if not bool(info.get("ok")):
            reason = str(info.get("reason", "load_failed"))
            if reason == "missing_state_file":
                self._recovery_mode = "fresh_start"
                self._recovery_reason = "no_persisted_state"
                self._recovery_requires_explicit_resume = False
                self._automation_paused = False
                self._automation_pause_reason = ""
                self._audit.append("automation_state_load", "", {"ok": True, "reason": "missing_state_file"})
                return
            self._recovery_mode = "restored_pending_review"
            self._recovery_reason = reason
            self._recovery_requires_explicit_resume = True
            self._automation_paused = True
            self._automation_pause_reason = f"recovery:{self._recovery_reason}"
            self._audit.append("automation_state_load", "", {"ok": False, "reason": self._recovery_reason})
            return

        self._gate_enabled = bool(payload.get("gate_enabled", False))
        self._rollout_stage = str(payload.get("rollout_stage", "paper_only"))
        if self._rollout_stage not in self._STAGE_ORDER:
            self._rollout_stage = "paper_only"
        env_mode = str(payload.get("execution_environment", "paper") or "paper").strip().lower()
        self._execution_environment = env_mode if env_mode in {"paper", "demo"} else "paper"
        self._rollout_stage_updated_at = float(payload.get("rollout_stage_updated_at", time.time()) or time.time())
        self._regression_triggered = bool(payload.get("regression_triggered", False))
        self._regression_reason = str(payload.get("regression_reason", ""))
        self._last_execution_result = dict(payload.get("last_execution_result", {}) or {})

        ks = dict(payload.get("kill_switch", {}) or {})
        self._kill_switch.restore(
            active=bool(ks.get("active", False)),
            reason=str(ks.get("reason", "restored")),
            changed_at=float(ks.get("changed_at", time.time()) or time.time()),
        )

        profiles = dict(payload.get("automation_profiles", {}) or {})
        restored: dict[str, AutomationPolicy] = {}
        for pid, raw in profiles.items():
            try:
                data = dict(raw or {})
                pol = AutomationPolicy(
                    id=str(data.get("id", pid)).strip().lower(),
                    enabled=bool(data.get("enabled", False)),
                    environment=str(data.get("environment", "paper")).strip().lower(),  # type: ignore[arg-type]
                    allowed_setup_families=[str(x).strip().lower() for x in list(data.get("allowed_setup_families", []) or []) if str(x).strip()],
                    allowed_instruments=[str(x).strip().upper() for x in list(data.get("allowed_instruments", []) or []) if str(x).strip()],
                    allowed_sessions=[str(x).strip().lower() for x in list(data.get("allowed_sessions", []) or []) if str(x).strip()],
                    size_cap=max(0.0, float(data.get("size_cap", 0.0) or 0.0)),
                    daily_loss_cap=max(0.0, float(data.get("daily_loss_cap", 0.0) or 0.0)),
                    max_trades_per_session=max(0, int(data.get("max_trades_per_session", 0) or 0)),
                    approval_required_for_activation=bool(data.get("approval_required_for_activation", True)),
                    approved=bool(data.get("approved", False)),
                    active_rollout_stage=str(data.get("active_rollout_stage", "paper_only")).strip().lower(),
                    pause_conditions=dict(data.get("pause_conditions", {}) or {}),
                    escalation_conditions=dict(data.get("escalation_conditions", {}) or {}),
                    created_at=float(data.get("created_at", time.time()) or time.time()),
                    updated_at=float(data.get("updated_at", time.time()) or time.time()),
                )
                restored[pol.id] = pol
            except Exception:
                continue
        self._automation_profiles = restored
        self._active_automation_policy_id = str(payload.get("active_automation_policy_id", "") or "").strip().lower()
        self._automation_paused = bool(payload.get("automation_paused", True))
        self._automation_pause_reason = str(payload.get("automation_pause_reason", "") or "")
        if not self._automation_paused:
            self._automation_pause_reason = ""
        self._automation_paused_by_sleep = bool(payload.get("automation_paused_by_sleep", False))
        self._automation_paused_before_sleep = bool(payload.get("automation_paused_before_sleep", False))
        self._automation_pause_reason_before_sleep = str(payload.get("automation_pause_reason_before_sleep", "") or "")
        self._automation_explicit_resume_active = bool(payload.get("automation_explicit_resume_active", False))
        self._automation_explicit_resume_at = float(payload.get("automation_explicit_resume_at", 0.0) or 0.0)
        self._automation_explicit_resume_policy_id = str(payload.get("automation_explicit_resume_policy_id", "") or "").strip().lower()
        self._last_automation_block_reason = str(payload.get("last_automation_block_reason", "") or "")
        self._last_automation_event = dict(payload.get("last_automation_event", {}) or {})
        self._last_execution_truth = dict(payload.get("last_execution_truth", {}) or {})
        self._last_gate_decision = dict(payload.get("last_gate_decision", {}) or {})
        self._last_rejection_reason = dict(payload.get("last_rejection_reason", {}) or {}) or None
        self._last_auto_signal_signature = str(payload.get("last_auto_signal_signature", "") or "")
        raw_history = dict(payload.get("auto_signal_history", {}) or {})
        parsed_history: dict[str, float] = {}
        for sig, ts in raw_history.items():
            try:
                key = str(sig or "").strip()
                if not key:
                    continue
                parsed_history[key] = float(ts)
            except Exception:
                continue
        self._auto_signal_history = parsed_history
        self._prune_auto_signal_history(now=time.time())
        autonomy_mode = str(payload.get("autonomy_mode", "recommend_only") or "recommend_only").strip().lower()
        self._autonomy_mode = autonomy_mode if autonomy_mode in self._AUTONOMY_MODES else "recommend_only"
        self._last_autonomy_cycle = dict(payload.get("last_autonomy_cycle", {}) or {})
        self._sleep_mode_active = bool(payload.get("sleep_mode_active", False))
        self._sleep_mode_reason = str(payload.get("sleep_mode_reason", "") or "")
        activated_raw = payload.get("sleep_mode_activated_at", None)
        self._sleep_mode_activated_at = float(activated_raw) if activated_raw is not None else None
        self._sleep_alert_count = int(payload.get("sleep_alert_count", 0) or 0)
        self._operator_present = bool(payload.get("operator_present", False))
        self._session_trade_counts = {str(k): int(v or 0) for k, v in dict(payload.get("session_trade_counts", {}) or {}).items()}
        self._last_shutdown_clean = bool(payload.get("last_shutdown_clean", False))
        self._reconciliation_status = str(payload.get("reconciliation_status", "unknown") or "unknown")
        self._last_reconciled_at = float(payload.get("last_reconciled_at", 0.0) or 0.0)
        self._mismatch_detected = bool(payload.get("mismatch_detected", False))
        pilot = dict(payload.get("tiny_live_pilot", {}) or {})
        self._tiny_live_pilot = TinyLivePilotConfig(
            configured=bool(pilot.get("configured", False)),
            enabled=bool(pilot.get("enabled", False)),
            approved=bool(pilot.get("approved", False)),
            allowed_instrument=str(pilot.get("allowed_instrument", "XAUUSD")).strip().upper() or "XAUUSD",
            allowed_setup_family=str(pilot.get("allowed_setup_family", "continuation")).strip().lower() or "continuation",
            size_cap=max(0.0, float(pilot.get("size_cap", 0.01) or 0.01)),
            max_trades_per_day=max(1, int(pilot.get("max_trades_per_day", 2) or 2)),
            daily_loss_cap=max(0.0, float(pilot.get("daily_loss_cap", 0.6) or 0.6)),
            max_consecutive_losses=max(1, int(pilot.get("max_consecutive_losses", 1) or 1)),
            rollout_stage_required=str(pilot.get("rollout_stage_required", "limited_live_pilot")).strip().lower() or "limited_live_pilot",
            approval_required_per_request=bool(pilot.get("approval_required_per_request", True)),
            pause_on_forced_flat=bool(pilot.get("pause_on_forced_flat", True)),
            pause_on_blocker_threshold=float(pilot.get("pause_on_blocker_threshold", 0.7) or 0.7),
            pause_on_stability_drop=float(pilot.get("pause_on_stability_drop", 0.45) or 0.45),
            pause_on_overreaction=float(pilot.get("pause_on_overreaction", 0.7) or 0.7),
            pause_on_degrading_performance=bool(pilot.get("pause_on_degrading_performance", True)),
            paused=bool(pilot.get("paused", False)),
            pause_reason=str(pilot.get("pause_reason", "")),
        )
        self._tiny_live_pilot_trade_count = int(payload.get("tiny_live_pilot_trade_count", 0) or 0)
        self._tiny_live_pilot_attempts_today = int(payload.get("tiny_live_pilot_attempts_today", 0) or 0)
        self._tiny_live_pilot_consecutive_losses = int(payload.get("tiny_live_pilot_consecutive_losses", 0) or 0)
        self._tiny_live_pilot_daily_loss_used = float(payload.get("tiny_live_pilot_daily_loss_used", 0.0) or 0.0)
        self._tiny_live_pilot_last_block_reason = str(payload.get("tiny_live_pilot_last_block_reason", "") or "")
        self._tiny_live_pilot_last_attempt = dict(payload.get("tiny_live_pilot_last_attempt", {}) or {})
        self._order_tracker.restore(dict(payload.get("order_tracker", {}) or {}))
        self._orphan_orders = {str(k): dict(v or {}) for k, v in dict(payload.get("orphan_orders", {}) or {}).items()}

        # Conservative recovery: never silently auto-resume live automation.
        policy = self._automation_profiles.get(self._active_automation_policy_id) if self._active_automation_policy_id else None
        self._restored_policy_id = policy.id if policy else ""
        needs_pause = False
        reason = "restored"
        if not self._last_shutdown_clean:
            needs_pause = True
            reason = "unclean_shutdown_forced_pause"
        if self._kill_switch.status().active:
            needs_pause = True
            reason = "kill_switch_persisted_active"
        explicit_resume_matches_policy = bool(
            policy
            and policy.enabled
            and self._automation_explicit_resume_active
            and self._automation_explicit_resume_policy_id == policy.id
            and not self._kill_switch.status().active
            and not self._sleep_mode_active
        )
        runtime_flags = self._load_runtime_flags_for_state_guard()
        sticky_resume_matches_policy = bool(
            policy
            and policy.enabled
            and policy.environment in {"live", "both"}
            and bool(runtime_flags.get("automation_resume_sticky", False))
            and not self._kill_switch.status().active
            and not self._sleep_mode_active
        )
        explicit_resume_valid = bool(explicit_resume_matches_policy and self._last_shutdown_clean)
        explicit_resume_after_process_restart = bool(explicit_resume_matches_policy and not self._last_shutdown_clean)
        sticky_resume_recovered = False
        if explicit_resume_after_process_restart and reason == "unclean_shutdown_forced_pause":
            needs_pause = False
            reason = "explicit_operator_resume_persisted_after_process_restart"
        if sticky_resume_matches_policy and reason == "unclean_shutdown_forced_pause":
            needs_pause = False
            sticky_resume_recovered = True
            reason = "sticky_operator_resume_persisted_after_process_restart"
            self._automation_explicit_resume_active = True
            self._automation_explicit_resume_at = self._automation_explicit_resume_at or time.time()
            self._automation_explicit_resume_policy_id = policy.id
        if policy and policy.enabled and not (explicit_resume_valid or explicit_resume_after_process_restart or sticky_resume_recovered) and not needs_pause:
            needs_pause = True
            if policy.environment in {"live", "both"}:
                reason = "live_auto_resume_blocked_after_restart"
            else:
                reason = "paper_auto_resume_requires_review"
        if needs_pause:
            self._automation_paused = True
            self._automation_pause_reason = reason
            if reason == "kill_switch_persisted_active":
                self._automation_explicit_resume_active = False
                self._automation_explicit_resume_at = 0.0
                self._automation_explicit_resume_policy_id = ""
            self._recovery_mode = "restored_but_paused"
            self._recovery_requires_explicit_resume = True
            self._recovery_reason = reason
        elif explicit_resume_valid or explicit_resume_after_process_restart or sticky_resume_recovered:
            self._automation_paused = False
            self._automation_pause_reason = ""
            self._clear_stale_pilot_pause_under_explicit_resume(event="load_state")
            self._recovery_mode = "restored_after_explicit_resume"
            self._recovery_requires_explicit_resume = False
            self._recovery_reason = reason if explicit_resume_after_process_restart or sticky_resume_recovered else "explicit_operator_resume_persisted"
        else:
            self._recovery_mode = "restored_non_live_only"
            self._recovery_requires_explicit_resume = False
            self._recovery_reason = "restored_without_active_automation"
        self._audit.append(
            "automation_state_load",
            "",
            {
                "ok": True,
                "reason": "loaded",
                "recovery_mode": self._recovery_mode,
                "recovery_reason": self._recovery_reason,
                "restored_policy_id": self._restored_policy_id,
            },
        )

    def refresh_critical_controls_from_persisted_state(self) -> dict[str, Any]:
        payload, info = self._state_store.load()
        if not bool(info.get("ok")):
            reason = str(info.get("reason", "load_failed") or "load_failed")
            if reason == "missing_state_file":
                self._last_shared_state_sync_ok = True
                self._last_shared_state_sync_reason = ""
                self._audit.append("automation_state_refresh_skipped", "", {"reason": reason})
                return {"ok": True, "reason": "missing_state_file"}
            self._last_shared_state_sync_ok = False
            self._last_shared_state_sync_reason = reason
            self._audit.append("automation_state_refresh_failed", "", {"reason": reason})
            return {"ok": False, "reason": reason}
        try:
            self._gate_enabled = bool(payload.get("gate_enabled", self._gate_enabled))
            stage = str(payload.get("rollout_stage", self._rollout_stage) or self._rollout_stage).strip().lower()
            if stage in self._STAGE_ORDER:
                self._rollout_stage = stage
            env_mode = str(payload.get("execution_environment", self._execution_environment) or self._execution_environment).strip().lower()
            self._execution_environment = env_mode if env_mode in {"paper", "demo"} else self._execution_environment
            self._sleep_mode_active = bool(payload.get("sleep_mode_active", self._sleep_mode_active))
            self._sleep_mode_reason = str(payload.get("sleep_mode_reason", self._sleep_mode_reason) or self._sleep_mode_reason)
            self._operator_present = bool(payload.get("operator_present", self._operator_present))
            self._automation_paused = bool(payload.get("automation_paused", self._automation_paused))
            self._automation_pause_reason = str(payload.get("automation_pause_reason", self._automation_pause_reason) or self._automation_pause_reason)
            if not self._automation_paused:
                self._automation_pause_reason = ""
            self._automation_explicit_resume_active = bool(
                payload.get("automation_explicit_resume_active", self._automation_explicit_resume_active)
            )
            self._automation_explicit_resume_at = float(
                payload.get("automation_explicit_resume_at", self._automation_explicit_resume_at) or self._automation_explicit_resume_at
            )
            self._automation_explicit_resume_policy_id = str(
                payload.get("automation_explicit_resume_policy_id", self._automation_explicit_resume_policy_id)
                or self._automation_explicit_resume_policy_id
            ).strip().lower()
            pilot = dict(payload.get("tiny_live_pilot", {}) or {})
            if pilot:
                self._tiny_live_pilot.paused = bool(pilot.get("paused", self._tiny_live_pilot.paused))
                self._tiny_live_pilot.pause_reason = str(
                    pilot.get("pause_reason", self._tiny_live_pilot.pause_reason) or ""
                )
            self._clear_stale_pilot_pause_under_explicit_resume(event="refresh_critical_controls")
            ks = dict(payload.get("kill_switch", {}) or {})
            self._kill_switch.restore(
                active=bool(ks.get("active", self._kill_switch.status().active)),
                reason=str(ks.get("reason", self._kill_switch.status().reason) or self._kill_switch.status().reason),
                changed_at=float(ks.get("changed_at", self._kill_switch.status().changed_at) or self._kill_switch.status().changed_at),
            )
            self._last_shared_state_sync_ok = True
            self._last_shared_state_sync_reason = ""
            self._audit.append(
                "automation_state_refresh_applied",
                "",
                {
                    "gate_enabled": self._gate_enabled,
                    "kill_switch_active": self._kill_switch.status().active,
                    "sleep_mode_active": self._sleep_mode_active,
                },
            )
            return {"ok": True, "reason": "refreshed"}
        except Exception as exc:
            reason = f"refresh_apply_failed:{exc.__class__.__name__}"
            self._last_shared_state_sync_ok = False
            self._last_shared_state_sync_reason = reason
            self._audit.append("automation_state_refresh_failed", "", {"reason": reason})
            return {"ok": False, "reason": reason}

    def create_request(
        self,
        *,
        environment: str,
        action_type: str,
        payload_summary: str,
        requested_by: str,
        dry_run: bool = True,
        auto_generated: bool = False,
        request_origin: str = "operator",
        strict_runtime_checks: bool = False,
    ) -> dict[str, Any]:
        request_id = f"exec_{uuid.uuid4().hex[:10]}"
        requested_env = str(environment or "").strip().lower()
        resolved_env = self._resolve_request_environment(requested_env)
        if self._is_live_like_environment(resolved_env):
            # Prevent stale in-memory split-brain state from overwriting newer critical
            # controls in persisted shared state during request creation.
            sync = self._sync_shared_controls_for_operation("request_create")
            if not bool(sync.get("ok", False)):
                reason = f"shared_state_sync_failed:{sync.get('reason', 'load_failed')}"
                self._audit.append(
                    "request_create_blocked",
                    request_id,
                    {"reason": reason, "environment": resolved_env},
                )
                return {
                    "id": "",
                    "environment": resolved_env,
                    "requested_environment": requested_env or resolved_env,
                    "action_type": str(action_type or "").lower(),
                    "payload_summary": str(payload_summary or "").strip(),
                    "created_at": time.time(),
                    "requested_by": str(requested_by or "operator"),
                    "approval_required": True,
                    "approval_status": "blocked",
                    "validation_status": "not_validated",
                    "execution_status": "blocked",
                    "dry_run": bool(dry_run),
                    "audit_ref": "",
                    "last_block_reason": reason,
                    "auto_generated": bool(auto_generated),
                    "request_origin": str(request_origin or "operator"),
                    "strict_runtime_checks": bool(strict_runtime_checks),
                }
        if self._sleep_mode_active:
            self._sleep_alert_count = int(self._sleep_alert_count) + 1
            self._audit.append(
                "sleep_mode_request_blocked",
                request_id,
                {"reason": "sleep_mode_active_no_new_requests", "environment": resolved_env},
            )
            self._persist_state(event="sleep_mode_request_blocked")
            return {
                "id": "",
                "environment": resolved_env,
                "requested_environment": requested_env or resolved_env,
                "action_type": str(action_type or "").lower(),
                "payload_summary": str(payload_summary or "").strip(),
                "created_at": time.time(),
                "requested_by": str(requested_by or "operator"),
                "approval_required": True,
                "approval_status": "blocked",
                "validation_status": "not_validated",
                "execution_status": "blocked",
                "dry_run": bool(dry_run),
                "audit_ref": "",
                "last_block_reason": "sleep_mode_active_no_new_requests",
                "auto_generated": bool(auto_generated),
                "request_origin": str(request_origin or "operator"),
                "strict_runtime_checks": bool(strict_runtime_checks),
            }
        req = {
            "id": request_id,
            "environment": resolved_env,
            "requested_environment": requested_env or resolved_env,
            "action_type": str(action_type or "").lower(),
            "payload_summary": str(payload_summary or "").strip(),
            "created_at": time.time(),
            "requested_by": str(requested_by or "operator"),
            "approval_required": True,
            "approval_status": "pending",
            "validation_status": "not_validated",
            "execution_status": "pending",
            "dry_run": bool(dry_run),
            "audit_ref": "",
            "last_block_reason": "",
            "auto_generated": bool(auto_generated),
            "request_origin": str(request_origin or "operator"),
            "strict_runtime_checks": bool(strict_runtime_checks),
            "approved_by": "",
            "approved_at": None,
            "approval_note": "",
            "live_explicit_confirmation": False,
            "instrument": self._extract_field(payload_summary, "instrument", default=""),
            "setup_family": self._extract_field(payload_summary, "setup", default=self._extract_field(payload_summary, "setup_family", default="")),
            "size": self._extract_float(payload_summary, "size", default=0.0),
            "side": self._extract_field(payload_summary, "side", default=""),
            "strategy": self._extract_field(payload_summary, "strategy", default=""),
            "confidence": self._extract_float(payload_summary, "confidence", default=0.0),
            "stop_loss_points": self._extract_float(payload_summary, "stop_loss_points", default=0.0),
            "take_profit_points": self._extract_float(payload_summary, "take_profit_points", default=0.0),
            "decision_id": self._extract_field(payload_summary, "decision_id", default=request_id),
            "data_source": self._extract_field(
                payload_summary,
                "data_source",
                default=("simulation" if resolved_env == "paper" else "broker_runtime"),
            ),
            "environment_mode": self._extract_field(payload_summary, "environment_mode", default=resolved_env),
        }
        self._requests[request_id] = req
        self._order.append(request_id)
        self._audit.append("request_created", request_id, {"environment": req["environment"], "dry_run": req["dry_run"]})
        self._trim()
        self._persist_state(event="request_created")
        return dict(req)

    def _sync_shared_controls_for_operation(self, operation: str) -> dict[str, Any]:
        """Apply persisted operator controls once at live-like operation boundaries."""
        sync = self.refresh_critical_controls_from_persisted_state()
        self._audit.append(
            "shared_controls_sync",
            "",
            {
                "operation": str(operation or "unknown"),
                "ok": bool(sync.get("ok", False)),
                "reason": str(sync.get("reason", "")),
            },
        )
        return sync

    def attempt_auto_execution_from_signal(
        self,
        *,
        signal_id: str,
        action_type: str,
        environment: str,
        instrument: str,
        setup_family: str,
        size: float,
        session: str,
        confidence: float,
        evidence_strength: float,
        side: str = "",
        strategy: str = "",
        stop_loss_points: float = 0.0,
        take_profit_points: float = 0.0,
        runtime_flags: dict[str, Any] | None = None,
        risk_policy: dict[str, Any] | None = None,
        market_snapshot: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        signature = str(signal_id or "").strip()
        cycle = self._build_autonomy_cycle_record(
            signal_id=signature,
            action_type=action_type,
            environment=environment,
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            confidence=confidence,
            evidence_strength=evidence_strength,
            runtime_flags=runtime_flags or {},
            market_snapshot=market_snapshot or {},
        )
        self._last_autonomy_cycle = dict(cycle)
        self._audit.append("autonomy_cycle_started", "", cycle)
        self._persist_state(event="autonomy_cycle_started")
        dup = self._is_duplicate_auto_signal(signature)
        if dup["duplicate"]:
            self._record_autonomy_cycle_outcome(
                action="blocked",
                blocker_reasons=["duplicate_auto_signal"],
                execution_result={
                    "ok": False,
                    "reason": "duplicate_auto_signal",
                    "age_seconds": dup.get("age_seconds", -1.0),
                    "ttl_seconds": self._auto_signal_dedupe_ttl_seconds(),
                },
            )
            self._audit.append(
                "automation_signal_duplicate_blocked",
                "",
                {
                    "signature": signature,
                    "age_seconds": dup.get("age_seconds", -1.0),
                    "ttl_seconds": self._auto_signal_dedupe_ttl_seconds(),
                },
            )
            return {
                "ok": False,
                "reason": "duplicate_auto_signal",
                "duplicate": True,
                "age_seconds": dup.get("age_seconds", -1.0),
                "ttl_seconds": self._auto_signal_dedupe_ttl_seconds(),
            }
        out = self.attempt_auto_execution(
            action_type=action_type,
            environment=environment,
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            confidence=confidence,
            evidence_strength=evidence_strength,
            side=side,
            strategy=strategy,
            stop_loss_points=stop_loss_points,
            take_profit_points=take_profit_points,
            runtime_flags=runtime_flags or {},
            risk_policy=risk_policy or {},
            dry_run=bool(dry_run),
        )
        self._record_autonomy_cycle_outcome(
            action=("executed" if bool(out.get("ok", False)) else "blocked"),
            blocker_reasons=([str(out.get("reason", ""))] if not bool(out.get("ok", False)) and str(out.get("reason", "")).strip() else []),
            execution_result=out,
        )
        if signature and self._should_record_auto_signal_result(out):
            self._record_auto_signal_signature(signature)
            self._persist_state(event="auto_signal_processed")
        return out

    def set_autonomy_mode(self, mode: str) -> dict[str, Any]:
        target = str(mode or "").strip().lower()
        if target not in self._AUTONOMY_MODES:
            self._audit.append("autonomy_mode_rejected", "", {"reason": "invalid_mode", "target": target})
            return self.snapshot()
        if target == self._autonomy_mode:
            return self.snapshot()
        self._autonomy_mode = target
        self._audit.append("autonomy_mode_changed", "", {"mode": target})
        self._persist_state(event="autonomy_mode_changed")
        return self.snapshot()

    def activate_sleep_mode(self, *, reason: str = "operator_inactive", force_flat_review: bool = False) -> dict[str, Any]:
        self._sleep_mode_active = True
        self._sleep_mode_reason = str(reason or "operator_inactive")
        self._sleep_mode_activated_at = time.time()
        self._sleep_alert_count = int(self._sleep_alert_count)
        self._automation_paused_before_sleep = bool(self._automation_paused)
        self._automation_pause_reason_before_sleep = str(self._automation_pause_reason or "")
        self._automation_paused = True
        self._automation_paused_by_sleep = True
        self._automation_pause_reason = "sleep_mode_active"
        self._automation_explicit_resume_active = False
        self._automation_explicit_resume_at = 0.0
        self._automation_explicit_resume_policy_id = ""
        self._audit.append(
            "sleep_mode_activated",
            "",
            {"reason": self._sleep_mode_reason, "force_flat_review": bool(force_flat_review)},
        )
        self._persist_state(event="sleep_mode_activated")
        return self.snapshot()

    def deactivate_sleep_mode(self, *, reason: str = "operator_resumed") -> dict[str, Any]:
        self._sleep_mode_active = False
        self._sleep_mode_reason = ""
        self._sleep_mode_activated_at = None
        if self._automation_paused_by_sleep:
            self._automation_paused = bool(self._automation_paused_before_sleep)
            self._automation_pause_reason = str(self._automation_pause_reason_before_sleep or "")
        self._automation_paused_by_sleep = False
        self._automation_paused_before_sleep = False
        self._automation_pause_reason_before_sleep = ""
        self._audit.append("sleep_mode_deactivated", "", {"reason": str(reason or "operator_resumed")})
        self._persist_state(event="sleep_mode_deactivated")
        return self.snapshot()

    def set_operator_present(self, present: bool, *, reason: str = "operator_requested") -> dict[str, Any]:
        target = bool(present)
        if target == self._operator_present:
            return self.snapshot()
        self._operator_present = target
        self._audit.append("operator_present_changed", "", {"operator_present": target, "reason": str(reason or "operator_requested")})
        self._persist_state(event="operator_present_changed")
        return self.snapshot()

    def set_session_override(self, active: bool, *, reason: str = "operator_requested") -> dict[str, Any]:
        target = bool(active)
        if target and self._execution_environment != "demo":
            self._session_override_last_block_reason = "session_override_demo_only"
            self._audit.append(
                "session_override_blocked",
                "",
                {
                    "reason": self._session_override_last_block_reason,
                    "environment": self._execution_environment,
                    "testing_only": True,
                },
            )
            return self.snapshot()
        if target == self._session_override_active:
            return self.snapshot()
        self._session_override_active = target
        if target:
            self._session_override_reason = "testing_only"
            self._session_override_effective_session = "simulated_demo_session"
            self._session_override_effective_minutes_to_close = 180
            self._session_override_last_block_reason = ""
        else:
            self._session_override_reason = ""
            self._session_override_effective_session = ""
            self._session_override_effective_minutes_to_close = 0
        self._audit.append(
            "session_override_changed",
            "",
            {
                "active": target,
                "reason": str(reason or "operator_requested"),
                "testing_only": True,
            },
        )
        return self.snapshot()

    def get_rollout_stage(self) -> str:
        return self._rollout_stage

    def configure_tiny_live_pilot(
        self,
        *,
        allowed_instrument: str,
        allowed_setup_family: str,
        size_cap: float,
        max_trades_per_day: int,
        daily_loss_cap: float,
        max_consecutive_losses: int,
    ) -> dict[str, Any]:
        self._tiny_live_pilot = TinyLivePilotConfig(
            configured=True,
            enabled=False,
            approved=False,
            allowed_instrument=str(allowed_instrument or "XAUUSD").strip().upper() or "XAUUSD",
            allowed_setup_family=str(allowed_setup_family or "continuation").strip().lower() or "continuation",
            size_cap=max(0.0, float(size_cap or 0.01)),
            max_trades_per_day=max(1, int(max_trades_per_day or 1)),
            daily_loss_cap=max(0.0, float(daily_loss_cap or 0.5)),
            max_consecutive_losses=max(1, int(max_consecutive_losses or 1)),
            rollout_stage_required="limited_live_pilot",
            approval_required_per_request=True,
            pause_on_forced_flat=True,
            pause_on_blocker_threshold=0.7,
            pause_on_stability_drop=0.45,
            pause_on_overreaction=0.7,
            pause_on_degrading_performance=True,
            paused=False,
            pause_reason="",
        )
        self._reset_daily_pilot_counters()
        self._tiny_live_pilot_last_block_reason = ""
        self._tiny_live_pilot_last_attempt = {"event": "configured", "ts": time.time()}
        self._audit.append("tiny_live_pilot_configured", "", self._tiny_live_pilot.to_dict())
        self._persist_state(event="tiny_live_pilot_configured")
        return self._tiny_live_pilot.to_dict()

    def approve_tiny_live_pilot(self, *, note: str = "") -> dict[str, Any]:
        self._tiny_live_pilot.approved = True
        self._audit.append("tiny_live_pilot_approved", "", {"note": str(note or "")})
        self._persist_state(event="tiny_live_pilot_approved")
        return self._tiny_live_pilot.to_dict()

    def activate_tiny_live_pilot(self) -> dict[str, Any]:
        if not self._tiny_live_pilot.approved:
            self._audit.append("tiny_live_pilot_activation_blocked", "", {"reason": "approval_required"})
            return self._tiny_live_pilot.to_dict()
        self._tiny_live_pilot.enabled = True
        self._tiny_live_pilot.paused = False
        self._tiny_live_pilot.pause_reason = ""
        self._audit.append("tiny_live_pilot_activated", "", {})
        self._persist_state(event="tiny_live_pilot_activated")
        return self._tiny_live_pilot.to_dict()

    def pause_tiny_live_pilot(self, reason: str) -> dict[str, Any]:
        self._tiny_live_pilot.paused = True
        self._tiny_live_pilot.pause_reason = str(reason or "operator_pause")
        self._audit.append("tiny_live_pilot_paused", "", {"reason": self._tiny_live_pilot.pause_reason})
        self._persist_state(event="tiny_live_pilot_paused")
        return self._tiny_live_pilot.to_dict()

    def resume_tiny_live_pilot(self) -> dict[str, Any]:
        if not self._tiny_live_pilot.enabled or not self._tiny_live_pilot.approved:
            return self._tiny_live_pilot.to_dict()
        self._tiny_live_pilot.paused = False
        self._tiny_live_pilot.pause_reason = ""
        self._audit.append("tiny_live_pilot_resumed", "", {})
        self._persist_state(event="tiny_live_pilot_resumed")
        return self._tiny_live_pilot.to_dict()

    def set_rollout_stage(self, stage: str, *, approved: bool = False, note: str = "") -> dict[str, Any]:
        target = str(stage or "").strip().lower()
        if target not in self._STAGE_ORDER:
            self._audit.append("rollout_stage_rejected", "", {"reason": "invalid_stage", "target": target})
            return self.snapshot()
        if target == self._rollout_stage:
            return self.snapshot()
        if self._is_live_stage(target) and not bool(approved):
            self._audit.append("rollout_stage_rejected", "", {"reason": "explicit_live_approval_required", "target": target})
            return self.snapshot()
        self._rollout_stage = target
        self._rollout_stage_updated_at = time.time()
        self._regression_triggered = False
        self._regression_reason = ""
        self._audit.append("rollout_stage_changed", "", {"stage": target, "note": str(note or "")})
        self._persist_state(event="rollout_stage_changed")
        return self.snapshot()

    def set_execution_environment(self, environment: str) -> dict[str, Any]:
        target = str(environment or "").strip().lower()
        if target not in {"paper", "demo"}:
            self._audit.append("execution_environment_rejected", "", {"reason": "invalid_environment", "target": target})
            return self.snapshot()
        if target == self._execution_environment:
            return self.snapshot()
        self._execution_environment = target
        validation = self._validate_broker_adapter_for_environment(target)
        if not bool(validation.get("ok", False)):
            self._audit.append(
                "execution_environment_validation_failed",
                "",
                {"environment": target, "reason": str(validation.get("reason", "invalid_broker_adapter"))},
            )
        self._audit.append("execution_environment_changed", "", {"environment": target})
        self._persist_state(event="execution_environment_changed")
        return self.snapshot()

    def set_broker_adapter(self, adapter: Any) -> dict[str, Any]:
        if adapter is None:
            self._audit.append("broker_adapter_set_rejected", "", {"reason": "adapter_none"})
            return self.snapshot()
        self.refresh_critical_controls_from_persisted_state()
        self._broker_adapter = adapter
        self._reconciler = OrderReconciliation(self._broker_adapter, self._order_tracker)
        self._audit.append("broker_adapter_set", "", {"adapter_type": self._adapter_type()})
        if self._adapter_type() != "InMemoryBrokerAdapter":
            self._start_reconciliation_timer_if_enabled()
        self._persist_state(event="broker_adapter_set")
        return self.snapshot()

    def create_automation_policy(
        self,
        *,
        policy_id: str,
        environment: str,
        allowed_setup_families: list[str],
        allowed_instruments: list[str],
        allowed_sessions: list[str],
        size_cap: float,
        daily_loss_cap: float,
        max_trades_per_session: int,
        active_rollout_stage: str,
        pause_conditions: dict[str, float] | None = None,
        escalation_conditions: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        pid = str(policy_id or "").strip().lower()
        if not pid:
            pid = f"policy_{uuid.uuid4().hex[:8]}"
        env = str(environment or "paper").strip().lower()
        if env == "demo":
            env = "live"
        if env not in {"paper", "live", "both"}:
            env = "paper"
        policy = AutomationPolicy(
            id=pid,
            enabled=False,
            environment=env,  # type: ignore[arg-type]
            allowed_setup_families=[str(x).strip().lower() for x in list(allowed_setup_families or []) if str(x).strip()],
            allowed_instruments=[str(x).strip().upper() for x in list(allowed_instruments or []) if str(x).strip()],
            allowed_sessions=[str(x).strip().lower() for x in list(allowed_sessions or []) if str(x).strip()],
            size_cap=max(0.0, float(size_cap or 0.0)),
            daily_loss_cap=max(0.0, float(daily_loss_cap or 0.0)),
            max_trades_per_session=max(0, int(max_trades_per_session or 0)),
            approval_required_for_activation=True,
            approved=False,
            active_rollout_stage=str(active_rollout_stage or "paper_only").strip().lower(),
            pause_conditions=dict(
                pause_conditions
                or {
                    "max_blocker_severity": 0.7,
                    "max_regime_noise": 0.7,
                    "min_stability_score": 0.45,
                    "max_overreaction_score": 0.7,
                    "max_recent_losses": 2.0,
                    "min_confidence": 0.55,
                    "min_evidence_strength": 0.55,
                }
            ),
            escalation_conditions=dict(escalation_conditions or {"performance_degrading": 1.0}),
        )
        self._automation_profiles[pid] = policy
        self._audit.append("automation_policy_created", "", {"policy_id": pid, "environment": env})
        self._persist_state(event="automation_policy_created")
        return policy.to_dict()

    def approve_automation_policy(self, policy_id: str, *, note: str = "") -> dict[str, Any] | None:
        policy = self._automation_profiles.get(str(policy_id or "").strip().lower())
        if policy is None:
            return None
        policy.approved = True
        policy.updated_at = time.time()
        self._audit.append("automation_policy_approved", "", {"policy_id": policy.id, "note": str(note or "")})
        self._persist_state(event="automation_policy_approved")
        return policy.to_dict()

    def activate_automation_policy(self, policy_id: str) -> dict[str, Any] | None:
        policy = self._automation_profiles.get(str(policy_id or "").strip().lower())
        if policy is None:
            return None
        if policy.approval_required_for_activation and not policy.approved:
            self._audit.append("automation_activation_blocked", "", {"policy_id": policy.id, "reason": "approval_required"})
            return None
        target_stage = str(policy.active_rollout_stage or "").strip().lower()
        if target_stage in self._STAGE_ORDER and target_stage != self._rollout_stage:
            if self._is_live_stage(target_stage) and not policy.approved:
                self._audit.append("automation_activation_blocked", "", {"policy_id": policy.id, "reason": "live_stage_requires_approval"})
                return None
            self._rollout_stage = target_stage
            self._rollout_stage_updated_at = time.time()
            self._audit.append("rollout_stage_changed", "", {"stage": target_stage, "note": f"automation_policy:{policy.id}"})
        policy.enabled = True
        policy.updated_at = time.time()
        self._active_automation_policy_id = policy.id
        self._automation_paused = False
        self._automation_pause_reason = ""
        self._automation_explicit_resume_active = False
        self._automation_explicit_resume_at = 0.0
        self._automation_explicit_resume_policy_id = ""
        self._audit.append("automation_policy_activated", "", {"policy_id": policy.id})
        self._persist_state(event="automation_policy_activated")
        return policy.to_dict()

    def deactivate_automation_policy(self, policy_id: str) -> dict[str, Any] | None:
        policy = self._automation_profiles.get(str(policy_id or "").strip().lower())
        if policy is None:
            return None
        policy.enabled = False
        policy.updated_at = time.time()
        if self._active_automation_policy_id == policy.id:
            self._active_automation_policy_id = ""
        if self._automation_explicit_resume_policy_id == policy.id:
            self._automation_explicit_resume_active = False
            self._automation_explicit_resume_at = 0.0
            self._automation_explicit_resume_policy_id = ""
        self._audit.append("automation_policy_deactivated", "", {"policy_id": policy.id})
        self._persist_state(event="automation_policy_deactivated")
        return policy.to_dict()

    def resume_automation(self) -> bool:
        if not self._active_automation_policy_id:
            return False
        policy = self._automation_profiles.get(self._active_automation_policy_id)
        if policy is None or not policy.enabled:
            return False
        if self._kill_switch.status().active:
            return False
        self._automation_paused = False
        self._automation_pause_reason = ""
        self._automation_explicit_resume_active = True
        self._automation_explicit_resume_at = time.time()
        self._automation_explicit_resume_policy_id = policy.id
        self._recovery_mode = "restored_after_explicit_resume"
        self._recovery_reason = "explicit_operator_resume"
        self._recovery_requires_explicit_resume = False
        pilot_pause_reason = str(getattr(self._tiny_live_pilot, "pause_reason", "") or "").strip().lower()
        if (
            self._tiny_live_pilot.enabled
            and self._tiny_live_pilot.approved
            and self._tiny_live_pilot.paused
            and pilot_pause_reason
            in {
                "operator_pause",
                "operator_requested",
                "weekend_learning_only_operator_requested",
            }
        ):
            self._tiny_live_pilot.paused = False
            self._tiny_live_pilot.pause_reason = ""
            self._audit.append("tiny_live_pilot_resumed_with_automation", "", {"previous_reason": pilot_pause_reason})
        self._audit.append("automation_resumed", "", {"policy_id": policy.id})
        self._persist_state(event="automation_resumed")
        return True

    def attempt_auto_execution(
        self,
        *,
        action_type: str,
        environment: str,
        instrument: str,
        setup_family: str,
        size: float,
        session: str,
        confidence: float,
        evidence_strength: float,
        side: str = "",
        strategy: str = "",
        stop_loss_points: float = 0.0,
        take_profit_points: float = 0.0,
        runtime_flags: dict[str, Any] | None = None,
        risk_policy: dict[str, Any] | None = None,
        dry_run: bool = True,
        execute_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        gate = self._automation_preflight(
            environment=environment,
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            confidence=confidence,
            evidence_strength=evidence_strength,
            runtime_flags=runtime_flags or {},
            risk_policy=risk_policy or {},
        )
        if not gate["ok"]:
            reason = str(gate.get("reason", "automation_blocked"))
            self._last_automation_block_reason = reason
            self._last_automation_event = {
                "ok": False,
                "reason": reason,
                "failure_class": self._classify_execution_failure(reason),
                "ts": time.time(),
                "phase": "automation_preflight",
            }
            self._audit.append("automation_execution_blocked", "", {"reason": reason})
            self._maybe_pause_automation(reason=reason, runtime_flags=runtime_flags or {})
            self._persist_state(event="automation_execution_blocked")
            return {"ok": False, "reason": reason, "failure_class": self._classify_execution_failure(reason)}

        normalized_side = str(side or "").strip().lower()
        if normalized_side not in {"buy", "sell"}:
            normalized_side = str(action_type or "").strip().lower()
        if normalized_side not in {"buy", "sell"}:
            normalized_side = ""
        payload_summary = (
            f"instrument={instrument} setup={setup_family} size={float(size):.4f} "
            f"session={session} confidence={float(confidence):.3f} evidence={float(evidence_strength):.3f} "
            f"side={normalized_side} strategy={str(strategy or setup_family).strip()} "
            f"stop_loss_points={float(stop_loss_points or 0.0):.2f} "
            f"take_profit_points={float(take_profit_points or 0.0):.2f}"
        )
        req = self.create_request(
            environment=environment,
            action_type=action_type,
            payload_summary=payload_summary,
            requested_by="automation",
            dry_run=bool(dry_run),
            auto_generated=True,
            request_origin="automation",
            strict_runtime_checks=True,
        )
        # Pre-approved pattern automation; still fully gated by execution preflight.
        req_id = str(req["id"])
        current = self._requests.get(req_id)
        if current is None:
            return {"ok": False, "reason": "request_not_found"}
        current["approval_status"] = "approved"
        current["approved_by"] = "automation_policy"
        current["approved_at"] = time.time()
        current["approval_note"] = f"policy={self._active_automation_policy_id}"
        if self._is_live_like_environment(str(environment).lower()):
            current["live_explicit_confirmation"] = True

        out = self.execute(
            req_id,
            runtime_flags=runtime_flags or {},
            risk_policy=risk_policy or {},
            execute_fn=execute_fn,
        )
        if out.get("ok"):
            s = str(session or "").lower().strip()
            self._session_trade_counts[s] = int(self._session_trade_counts.get(s, 0)) + 1
            self._last_automation_event = {
                "ok": True,
                "request_id": req_id,
                "ts": time.time(),
                "dry_run": bool(dry_run),
                "result": dict(out.get("result", {}) or {}),
                "phase": "automation_execute",
            }
            self._audit.append("automation_execution_success", req_id, {"dry_run": bool(dry_run)})
            self._persist_state(event="automation_execution_success")
        else:
            reason = str(out.get("reason", "blocked"))
            self._last_automation_event = {
                "ok": False,
                "request_id": req_id,
                "reason": reason,
                "failure_class": self._classify_execution_failure(reason),
                "ts": time.time(),
                "phase": "automation_execute",
            }
            self._audit.append("automation_execution_failed", req_id, {"reason": out.get("reason", "blocked")})
            self._maybe_pause_automation(reason=reason, runtime_flags=runtime_flags or {})
            self._persist_state(event="automation_execution_failed")
        return out

    def approve_request(self, request_id: str, *, approved_by: str, note: str = "", explicit_live_confirmation: bool = False) -> dict[str, Any] | None:
        req = self._requests.get(str(request_id))
        if req is None:
            return None
        req["approved_by"] = str(approved_by or "operator")
        req["approved_at"] = time.time()
        req["approval_note"] = str(note or "")
        req["live_explicit_confirmation"] = bool(explicit_live_confirmation)
        if self._is_live_like_environment(str(req.get("environment", ""))) and not req["live_explicit_confirmation"]:
            req["approval_status"] = "pending"
            req["last_block_reason"] = "live_requires_explicit_confirmation"
            self._audit.append("approval_blocked", req["id"], {"reason": req["last_block_reason"]})
            return dict(req)
        req["approval_status"] = "approved"
        self._audit.append("approved", req["id"], {"approved_by": req["approved_by"]})
        self._persist_state(event="request_approved")
        return dict(req)

    def reject_request(self, request_id: str, *, note: str = "") -> dict[str, Any] | None:
        req = self._requests.get(str(request_id))
        if req is None:
            return None
        req["approval_status"] = "rejected"
        req["execution_status"] = "blocked"
        req["last_block_reason"] = str(note or "rejected_by_operator")
        self._audit.append("rejected", req["id"], {"reason": req["last_block_reason"]})
        self._persist_state(event="request_rejected")
        return dict(req)

    def execute(
        self,
        request_id: str,
        *,
        runtime_flags: dict[str, Any] | None = None,
        risk_policy: dict[str, Any] | None = None,
        execute_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        req = self._requests.get(str(request_id))
        if req is None:
            return {"ok": False, "reason": "request_not_found"}

        # Idempotency: completed requests return current state.
        if req["execution_status"] in {"executed", "dry_run_completed", "blocked", "failed"}:
            self._audit.append("execution_idempotent_return", req["id"], {"execution_status": req["execution_status"]})
            return {"ok": req["execution_status"] in {"executed", "dry_run_completed"}, "request": dict(req)}
        if self._is_live_like_environment(str(req.get("environment", ""))):
            sync = self.refresh_critical_controls_from_persisted_state()
            if not bool(sync.get("ok", False)):
                req["execution_status"] = "pending"
                req["last_block_reason"] = f"shared_state_sync_failed:{sync.get('reason', 'load_failed')}"
                self._current_gate_request_id = str(req.get("id", "") or "")
                structured = self._gate_block(req["last_block_reason"], gate="shared_state_sync", detail=dict(sync))
                self._audit.append("execution_blocked", req["id"], {"reason": req["last_block_reason"]})
                self._record_execution_truth(req, ok=False, phase="shared_state_sync", reason=req["last_block_reason"])
                self._persist_state(event="execution_blocked")
                return {
                    "ok": False,
                    "reason": req["last_block_reason"],
                    "code": str(structured.get("code", self._reason_code(req["last_block_reason"]))),
                    "reason_code": structured.get("reason_code"),
                    "checked_at": structured.get("checked_at"),
                    "structured_result": dict(structured.get("structured_result", {}) or {}),
                    "request": dict(req),
                }
        if self._is_live_like_environment(str(req.get("environment", ""))):
            self._tiny_live_pilot_attempts_today = int(self._tiny_live_pilot_attempts_today) + 1

        gate = self._preflight_gate(req, runtime_flags=runtime_flags, risk_policy=risk_policy)
        if gate["ok"] is not True:
            # Keep request pending on gate failure so operator can fix approval/environment and retry.
            req["execution_status"] = "pending"
            req["last_block_reason"] = str(gate.get("reason", "blocked"))
            self._audit.append("execution_blocked", req["id"], {"reason": req["last_block_reason"]})
            self._record_execution_truth(req, ok=False, phase="preflight", reason=req["last_block_reason"])
            self._maybe_regress_from_failure(reason=req["last_block_reason"])
            self._persist_state(event="execution_blocked")
            return {
                "ok": False,
                "reason": req["last_block_reason"],
                "code": str(gate.get("code", self._reason_code(req["last_block_reason"]))),
                "reason_code": gate.get("reason_code"),
                "checked_at": gate.get("checked_at"),
                "detail": dict(gate.get("detail", {}) or {}),
                "structured_result": dict(gate.get("structured_result", {}) or {}),
                "request": dict(req),
            }

        if req["dry_run"]:
            req["validation_status"] = "passed"
            req["execution_status"] = "dry_run_completed"
            self._audit.append(
                "execution_dry_run",
                req["id"],
                {"environment": req["environment"], "action_type": req["action_type"], "payload_summary": req["payload_summary"]},
            )
            self._last_execution_result = {"ok": True, "dry_run": True, "request_id": req["id"]}
            self._record_execution_truth(req, ok=True, phase="dry_run", dry_run=True, result={"would_execute": True})
            self._persist_state(event="execution_dry_run")
            return {
                "ok": True,
                "dry_run": True,
                "request": dict(req),
                "would_execute": True,
                "reason_code": gate.get("reason_code"),
                "checked_at": gate.get("checked_at"),
                "structured_result": dict(gate.get("structured_result", {}) or {}),
            }

        self._active_request_id = req["id"]
        try:
            req["validation_status"] = "passed"
            self._audit.append("execution_attempt", req["id"], {"environment": req["environment"]})
            if self._is_live_like_environment(str(req.get("environment", ""))):
                result = self._execute_live_order(req)
            else:
                if execute_fn is None:
                    result = {"ok": False, "reason": "executor_missing"}
                else:
                    result = execute_fn(dict(req))
            if not bool(result.get("ok", False)):
                req["execution_status"] = "failed"
                req["last_block_reason"] = str(result.get("reason", "executor_failed"))
                self._audit.append("execution_failed", req["id"], {"reason": req["last_block_reason"]})
                self._last_execution_result = {"ok": False, "request_id": req["id"], "reason": req["last_block_reason"]}
                self._record_execution_truth(req, ok=False, phase="broker_or_executor", reason=req["last_block_reason"], result=dict(result or {}))
                self._maybe_regress_from_failure(reason=req["last_block_reason"])
                self._persist_state(event="execution_failed")
                return {
                    "ok": False,
                    "reason": req["last_block_reason"],
                    "failure_class": self._classify_execution_failure(req["last_block_reason"]),
                    "request": dict(req),
                }
            req["execution_status"] = "executed"
            if self._is_live_like_environment(str(req.get("environment", ""))):
                self._tiny_live_pilot_trade_count = int(self._tiny_live_pilot_trade_count) + 1
            self._audit.append("execution_success", req["id"], {"result": dict(result)})
            self._last_execution_result = {"ok": True, "request_id": req["id"], "result": dict(result)}
            self._record_execution_truth(req, ok=True, phase="broker_or_executor", result=dict(result or {}))
            self._persist_state(event="execution_success")
            return {
                "ok": True,
                "request": dict(req),
                "result": dict(result),
                "reason_code": gate.get("reason_code"),
                "checked_at": gate.get("checked_at"),
                "structured_result": dict(gate.get("structured_result", {}) or {}),
            }
        finally:
            self._active_request_id = None

    def set_gate_enabled(self, enabled: bool) -> dict[str, Any]:
        self._gate_enabled = bool(enabled)
        self._audit.append("execution_gate_toggled", "", {"enabled": self._gate_enabled})
        self._persist_state(event="execution_gate_toggled")
        return self.snapshot()

    def cancel_live_order(self, request_id: str) -> dict[str, Any]:
        tracked = self._order_tracker.get_by_request(str(request_id))
        if tracked is None:
            return {"ok": False, "reason": "tracked_order_not_found"}
        if not tracked.broker_order_id:
            self._order_tracker.mark_reconciliation_required(str(request_id), reason="missing_broker_order_id_for_cancel")
            self._audit.append("order_cancel_failed", str(request_id), {"reason": "missing_broker_order_id_for_cancel"})
            self._persist_state(event="order_cancel_failed")
            return {"ok": False, "reason": "missing_broker_order_id_for_cancel"}
        self._order_tracker.update_status(request_id=str(request_id), status="cancel_requested")
        self._audit.append("order_cancel_requested", str(request_id), {"broker_order_id": tracked.broker_order_id})
        cancel = self._broker_adapter.cancel_order(broker_order_id=tracked.broker_order_id)
        if not cancel.ok:
            self._order_tracker.mark_reconciliation_required(str(request_id), reason=f"cancel_failed:{cancel.message}")
            self._audit.append("order_cancel_failed", str(request_id), {"reason": cancel.message})
            self._persist_state(event="order_cancel_failed")
            return {"ok": False, "reason": f"cancel_failed:{cancel.message}"}
        self._order_tracker.update_status(request_id=str(request_id), status="cancelled", rejection_reason=str(cancel.message or ""))
        self._audit.append("order_cancelled", str(request_id), {"broker_order_id": tracked.broker_order_id})
        self._persist_state(event="order_cancelled")
        return {"ok": True, "status": "cancelled", "broker_order_id": tracked.broker_order_id}

    def pending_reconciliation_orders(self, limit: int = 20) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for order in self._order_tracker.pending():
            if order.reconciliation_status != "required" and order.status != "reconciliation_required":
                continue
            out.append(order.to_dict())
            if len(out) >= max(1, int(limit)):
                break
        return out

    def reconcile_order(self, request_id: str) -> dict[str, Any]:
        tracked = self._order_tracker.get_by_request(str(request_id))
        if tracked is None:
            return {"ok": False, "reason": "tracked_order_not_found"}
        rec = self._reconciler.reconcile_request(str(request_id))
        self._audit.append(
            "order_reconciliation_manual",
            str(request_id),
            {
                "ok": rec.ok,
                "status": rec.status,
                "reason": rec.reason,
                "requires_review": rec.requires_review,
                "details": dict(rec.details or {}),
            },
        )
        self._persist_state(event="order_reconciliation_manual")
        return {"ok": bool(rec.ok), "status": rec.status, "reason": rec.reason, "requires_review": rec.requires_review, "details": dict(rec.details or {})}

    def reconcile_now(self, limit: int = 10) -> dict[str, Any]:
        pending = self.pending_reconciliation_orders(limit=limit)
        if not pending:
            return {"ok": True, "resolved": 0, "unresolved": 0, "items": []}
        items: list[dict[str, Any]] = []
        resolved = 0
        unresolved = 0
        for row in pending:
            rid = str(row.get("internal_request_id", ""))
            res = self.reconcile_order(rid)
            items.append({"request_id": rid, **res})
            if res.get("ok") and not res.get("requires_review"):
                resolved += 1
            else:
                unresolved += 1
        return {"ok": unresolved == 0, "resolved": resolved, "unresolved": unresolved, "items": items}

    def scan_broker_open_orders(self) -> dict[str, Any]:
        if self._adapter_type() == "InMemoryBrokerAdapter":
            self._audit.append(
                "orphan_scan_skipped",
                "",
                {"reason": "in_memory_adapter_not_authoritative_for_shared_state"},
            )
            return {"ok": True, "scanned": 0, "orphans_detected": 0, "mismatches_detected": 0, "orphans": [], "mismatches": [], "status": "skipped"}
        scanned_at = time.time()
        try:
            open_orders = list(self._broker_adapter.list_open_orders() or [])
        except Exception as exc:
            reason = f"{exc.__class__.__name__}:{exc}"
            self._audit.append(
                "orphan_scan_failed",
                "",
                {"reason": reason},
            )
            self._persist_state(event="orphan_scan_failed")
            return {
                "ok": False,
                "scanned": 0,
                "orphans_detected": 0,
                "mismatches_detected": 0,
                "orphans": [],
                "mismatches": [],
                "status": "scan_failed",
                "reason": reason,
            }
        known_ids = self._order_tracker.known_broker_order_ids()
        open_ids = {str(row.get("broker_order_id", "") or "") for row in open_orders}
        stale_orphan_ids = [
            bid
            for bid, row in list(self._orphan_orders.items())
            if bid and bid not in open_ids and str(row.get("broker_status", "") or "").lower() == "open"
        ]
        for bid in stale_orphan_ids:
            self._orphan_orders.pop(bid, None)
        detected: list[dict[str, Any]] = []
        mismatches: list[dict[str, Any]] = []
        for row in open_orders:
            bid = str(row.get("broker_order_id", "") or "")
            bstatus = str(row.get("status", "unknown") or "unknown")
            if not bid:
                continue
            tracked = self._order_tracker.get_by_broker_order_id(bid)
            if tracked is None:
                entry = {
                    "broker_order_id": bid,
                    "instrument": str(row.get("instrument", "") or ""),
                    "side": str(row.get("side", "") or ""),
                    "size": float(row.get("size", 0.0) or 0.0),
                    "broker_status": bstatus,
                    "detected_at": scanned_at,
                    "reason": "unknown_to_atlas",
                    "reviewed": False,
                }
                self._orphan_orders[bid] = entry
                detected.append(dict(entry))
            else:
                if str(tracked.status or "unknown") != bstatus:
                    mm = {
                        "broker_order_id": bid,
                        "internal_request_id": tracked.internal_request_id,
                        "internal_status": str(tracked.status or "unknown"),
                        "broker_status": bstatus,
                        "detected_at": scanned_at,
                        "reason": "status_mismatch",
                    }
                    mismatches.append(mm)
        self._audit.append(
            "orphan_scan_completed",
            "",
            {
                "open_orders": len(open_orders),
                "orphans_detected": len(detected),
                "mismatches_detected": len(mismatches),
                "stale_orphans_cleared": len(stale_orphan_ids),
            },
        )
        self._persist_state(event="orphan_scan_completed")
        return {"ok": True, "scanned": len(open_orders), "orphans_detected": len(detected), "mismatches_detected": len(mismatches), "stale_orphans_cleared": len(stale_orphan_ids), "orphans": detected, "mismatches": mismatches}

    def list_orphan_orders(self, *, include_reviewed: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        rows = list(self._orphan_orders.values())
        if not include_reviewed:
            rows = [x for x in rows if not bool(x.get("reviewed", False))]
        rows = sorted(rows, key=lambda x: float(x.get("detected_at", 0.0) or 0.0), reverse=True)
        return rows[: max(1, int(limit))]

    def reconcile_orphan(self, broker_order_id: str, *, internal_request_id: str | None = None, mark_reviewed: bool = False) -> dict[str, Any]:
        bid = str(broker_order_id or "").strip()
        if not bid:
            return {"ok": False, "reason": "missing_broker_order_id"}
        row = dict(self._orphan_orders.get(bid, {}) or {})
        if not row:
            return {"ok": False, "reason": "orphan_not_found"}
        if internal_request_id:
            tracked = self._order_tracker.get_by_request(str(internal_request_id))
            if tracked is None:
                return {"ok": False, "reason": "internal_request_not_found"}
            # Explicit operator link only; no status assumption.
            tracked.broker_order_id = bid
            tracked.reconciliation_status = "required"
            tracked.last_update_at = time.time()
            self._orphan_orders[bid]["reason"] = "linked_pending_reconciliation"
        if mark_reviewed:
            self._orphan_orders[bid]["reviewed"] = True
        self._audit.append(
            "orphan_reconciliation_action",
            "",
            {
                "broker_order_id": bid,
                "linked_internal_request_id": str(internal_request_id or ""),
                "mark_reviewed": bool(mark_reviewed),
            },
        )
        self._persist_state(event="orphan_reconciliation_action")
        return {"ok": True, "broker_order_id": bid, "orphan": dict(self._orphan_orders.get(bid, {}))}

    def activate_kill_switch(self, reason: str = "operator_requested") -> dict[str, Any]:
        state = self._kill_switch.activate(reason=reason)
        self._automation_explicit_resume_active = False
        self._automation_explicit_resume_at = 0.0
        self._automation_explicit_resume_policy_id = ""
        self._audit.append("kill_switch_activated", "", asdict(state))
        self._persist_state(event="kill_switch_activated")
        return self.snapshot()

    def deactivate_kill_switch(self, reason: str = "operator_requested") -> dict[str, Any]:
        state = self._kill_switch.deactivate(reason=reason)
        self._audit.append("kill_switch_deactivated", "", asdict(state))
        self._persist_state(event="kill_switch_deactivated")
        return self.snapshot()

    def pending_requests(self, *, environment: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        env = str(environment or "").strip().lower()
        out = []
        for rid in reversed(self._order):
            req = self._requests.get(rid, {})
            if not req:
                continue
            if req["approval_status"] == "pending" or req["execution_status"] == "pending":
                req_env = str(req.get("environment", "")).strip().lower()
                if env:
                    if env == "live":
                        if req_env not in {"live", "demo"}:
                            continue
                    elif req_env != env:
                        continue
                out.append(dict(req))
            if len(out) >= max(1, int(limit)):
                break
        return out

    def get_request(self, request_id: str) -> dict[str, Any] | None:
        req = self._requests.get(str(request_id))
        return dict(req) if req is not None else None

    def last_request(self) -> dict[str, Any] | None:
        if not self._order:
            return None
        req = self._requests.get(self._order[-1])
        return dict(req) if req else None

    def snapshot(self, *, runtime_flags: dict[str, Any] | None = None) -> dict[str, Any]:
        self._clear_stale_pilot_pause_under_explicit_resume(event="snapshot")
        kill = self._kill_switch.status()
        pending = self.pending_requests(limit=100)
        active_policy = self._automation_profiles.get(self._active_automation_policy_id) if self._active_automation_policy_id else None
        flags = dict(runtime_flags or {})
        session_state = self._session_control.to_dict(runtime_flags=flags)
        auto_status = self.auto_trading_status(runtime_flags=flags)
        execution_truth = dict(self._last_execution_truth or {})
        truth_ts = float(execution_truth.get("ts", 0.0) or 0.0) if execution_truth else 0.0
        truth_age_seconds = max(0.0, time.time() - truth_ts) if truth_ts > 0 else 0.0
        truth_reason = str(execution_truth.get("reason", "") or "")
        truth_stale = bool(
            execution_truth
            and truth_age_seconds > 900
            and truth_reason in {"approval_missing", "live_requires_explicit_confirmation"}
            and not self._active_request_id
        )
        if execution_truth:
            execution_truth["current"] = not truth_stale
            execution_truth["stale"] = truth_stale
            execution_truth["age_seconds"] = round(truth_age_seconds, 2)
            if truth_stale:
                execution_truth["display_reason"] = "stale_previous_execution_truth"
                execution_truth["policy_note"] = "Previous approval state is retained for audit but should not be treated as the current execution blocker."
        return {
            "execution_gate_enabled": self._gate_enabled,
            "kill_switch_active": bool(kill.active),
            "kill_switch_reason": str(kill.reason),
            "active_request_id": self._active_request_id or "",
            "pending_count": len(pending),
            "dry_run_default": True,
            "rollout_stage": self._rollout_stage,
            "execution_environment": self._execution_environment,
            "environment_model": self._environment_model(),
            "adapter_type": self._adapter_type(),
            "execution_environment_label": ("Live Demo" if self._execution_environment == "demo" else "Paper"),
            "real_live_enabled": False,
            "real_live_label": "Real Live (disabled / not in use)",
            "live_execution_allowed": self._is_live_allowed_for_stage(self._rollout_stage),
            "active_restrictions": self._stage_restrictions(self._rollout_stage),
            "approval_required": True,
            "autonomy_mode": self._autonomy_mode,
            "autonomy_modes_supported": sorted(self._AUTONOMY_MODES),
            "autonomy_mode_semantics": self._autonomy_mode_semantics(),
            "autonomous_trader_mandate": self._load_autonomous_trader_mandate_summary(),
            "autonomy_live_default_safe": self._autonomy_mode != "guarded_execution",
            "mode_separation": {
                "market_state_mode": "simulation" if bool((self._last_autonomy_cycle or {}).get("simulation_market_state", False)) else "broker_or_runtime",
                "execution_mode": self._derived_operating_mode(),
                "autonomy_mode": self._autonomy_mode,
                "real_live_enabled": False,
            },
            "sleep_mode_active": self._sleep_mode_active,
            "sleep_mode_reason": self._sleep_mode_reason,
            "sleep_mode_activated_at": self._sleep_mode_activated_at,
            "sleep_execution_blocked": self._sleep_mode_active,
            "monitoring_active_while_sleeping": True,
            "alert_only_while_sleeping": self._sleep_mode_active,
            "sleep_alert_count": int(self._sleep_alert_count),
            "operator_present": self._operator_present,
            "session_override_active": self._session_override_active,
            "session_override_testing_only": True,
            "session_override_reason": self._session_override_reason,
            "session_override_effective_session": self._session_override_effective_session,
            "session_override_effective_minutes_to_close": int(self._session_override_effective_minutes_to_close or 0),
            "session_override_last_block_reason": self._session_override_last_block_reason,
            "session_override_note": (
                "TESTING ONLY: session timing gates overridden."
                if self._session_override_active
                else "off"
            ),
            "session_override_real_session_active": bool(session_state.get("session_active", False)),
            "session_override_simulated_session_active": bool(self._session_override_active and self._execution_environment == "demo"),
            "last_execution_result": dict(self._last_execution_result or {}),
            "regression_triggered": self._regression_triggered,
            "regression_reason": self._regression_reason,
            "automation_enabled": bool(active_policy and active_policy.enabled and not self._automation_paused),
            "automation_paused": self._automation_paused,
            "automation_pause_reason": self._automation_pause_reason if self._automation_paused else "",
            "automation_paused_by_sleep": self._automation_paused_by_sleep,
            "automation_explicit_resume_active": bool(self._automation_explicit_resume_active),
            "automation_explicit_resume_at": float(self._automation_explicit_resume_at or 0.0),
            "automation_explicit_resume_policy_id": str(self._automation_explicit_resume_policy_id or ""),
            "automation_active_policy_id": self._active_automation_policy_id,
            "automation_policy": dict(active_policy.to_dict() if active_policy else {}),
            "automation_live_allowed": bool(
                active_policy
                and active_policy.enabled
                and not self._automation_paused
                and active_policy.environment in {"live", "both"}
                and self._is_live_allowed_for_stage(self._rollout_stage)
            ),
            "automation_last_block_reason": self._last_automation_block_reason,
            "automation_last_event": dict(self._last_automation_event or {}),
            "execution_truth": execution_truth,
            "gate_decisions_last_cycle": dict(self._last_gate_decision or {}),
            "auto_trade_allowed": bool(auto_status.get("allowed", False)),
            "auto_trade_reason": str(auto_status.get("reason", "")),
            "auto_trade_last_block_reason": str(auto_status.get("last_block_reason", self._last_automation_block_reason)),
            "auto_trade_demo_only": True,
            "auto_trade_last_attempt": dict(self._last_automation_event or {}),
            "last_auto_execution_result": dict(self._last_automation_event or {}),
            "last_auto_execution_block_reason": str(self._last_automation_block_reason or ""),
            "last_auto_signal_signature": self._last_auto_signal_signature,
            "last_autonomy_cycle": dict(self._last_autonomy_cycle or {}),
            "recovery_mode": self._recovery_mode,
            "recovery_reason": self._recovery_reason,
            "recovery_requires_explicit_resume": self._recovery_requires_explicit_resume,
            "restored_policy_id": self._restored_policy_id,
            "last_shutdown_clean": self._last_shutdown_clean,
            "session_control": session_state,
            "tiny_live_pilot": self._tiny_live_pilot.to_dict(),
            "tiny_live_pilot_attempts_today": self._tiny_live_pilot_attempts_today,
            "tiny_live_pilot_trade_count": self._tiny_live_pilot_trade_count,
            "tiny_live_pilot_consecutive_losses": self._tiny_live_pilot_consecutive_losses,
            "tiny_live_pilot_last_block_reason": self._tiny_live_pilot_last_block_reason,
            "tiny_live_pilot_last_attempt": dict(self._tiny_live_pilot_last_attempt or {}),
            "tiny_live_pilot_daily_loss_used": float(self._tiny_live_pilot_daily_loss_used),
            "last_live_order_status": self._order_status_snapshot().get("last_live_order_status", "none"),
            "pending_live_orders_count": self._order_status_snapshot().get("pending_live_orders_count", 0),
            "partially_filled_count": self._order_status_snapshot().get("partially_filled_count", 0),
            "reconciliation_required": self._order_status_snapshot().get("reconciliation_required", False),
            "last_broker_reject_reason": self._order_status_snapshot().get("last_broker_reject_reason", ""),
            "last_uncertain_execution_state": self._order_status_snapshot().get("last_uncertain_execution_state", ""),
            "live_approvals_used": self._order_status_snapshot().get("live_approvals_used", 0),
            "top_reconciliation_request_id": self._order_status_snapshot().get("top_reconciliation_request_id", ""),
            "top_reconciliation_reason": self._order_status_snapshot().get("top_reconciliation_reason", ""),
            "orphan_order_count": self._order_status_snapshot().get("orphan_order_count", 0),
            "top_orphan_order_id": self._order_status_snapshot().get("top_orphan_order_id", ""),
            "top_orphan_summary": self._order_status_snapshot().get("top_orphan_summary", ""),
            "reconciliation_status": self._reconciliation_status,
            "last_reconciled_at": float(self._last_reconciled_at or 0.0),
            "mismatch_detected": bool(self._mismatch_detected),
            "forced_flat_pending": bool(session_state.get("forced_flat_required", False)),
            "forced_flat_urgency": (
                "immediate_review"
                if bool(session_state.get("forced_flat_required", False)) and int(session_state.get("minutes_to_session_close", -1) or -1) <= 5
                else ("urgent" if bool(session_state.get("forced_flat_required", False)) and int(session_state.get("minutes_to_session_close", -1) or -1) <= 15 else ("high" if bool(session_state.get("forced_flat_required", False)) else "none"))
            ),
            "last_audit_event": self._audit.last(),
        }

    def auto_trading_status(self, *, runtime_flags: dict[str, Any] | None = None) -> dict[str, Any]:
        flags = dict(runtime_flags or {})
        policy = self._automation_profiles.get(self._active_automation_policy_id) if self._active_automation_policy_id else None
        environment = "demo"
        instrument = str((self._tiny_live_pilot.allowed_instrument if self._tiny_live_pilot.configured else "XAUUSD") or "XAUUSD")
        setup_family = str((self._tiny_live_pilot.allowed_setup_family if self._tiny_live_pilot.configured else "continuation") or "continuation")
        size = float((self._tiny_live_pilot.size_cap if self._tiny_live_pilot.configured else 0.01) or 0.01)
        session = str(self._session_control.to_dict(runtime_flags=flags).get("current_session", "none") or "none")
        has_runtime_freshness = isinstance(flags.get("freshness_runtime"), dict)
        reason = self._auto_trading_gate_reason(
            runtime_flags=flags,
            environment=environment,
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            policy=policy,
            require_freshness=has_runtime_freshness,
        )
        if not reason:
            reason = self._autonomy_gate_reason(environment=environment, runtime_flags=flags)
        if not has_runtime_freshness and not reason:
            reason = "runtime_context_required"
        return {
            "allowed": reason == "",
            "reason": reason,
            "last_block_reason": reason or self._last_automation_block_reason,
            "autonomy_mode": self._autonomy_mode,
            "operating_mode": self._derived_operating_mode(),
            "operator_present": self._operator_present,
            "sleep_mode_active": self._sleep_mode_active,
            "environment": self._execution_environment,
            "demo_only": True,
        }

    def recent_audit(self, limit: int = 30) -> list[dict[str, Any]]:
        return self._audit.recent(limit=limit)

    @staticmethod
    def _classify_execution_failure(reason: str) -> str:
        text = str(reason or "").strip().lower()
        if not text:
            return ""
        if "broker_rejected" in text or text.startswith("broker_rejected") or "insufficient_margin" in text:
            return "broker_reject"
        if "broker_response_invalid" in text or "broker_confirmation_missing" in text or "missing_broker_order_id" in text:
            return "broker_confirmation"
        if "order_reconciliation_required" in text or "unknown" in text or "timeout" in text:
            return "reconciliation_required"
        if "approval" in text or "explicit_confirmation" in text:
            return "approval_or_authority"
        if "freshness" in text or "stale" in text:
            return "stale_runtime"
        if "automation_paused" in text or "kill_switch" in text or "sleep_mode" in text:
            return "operator_or_safety_pause"
        if "daily_loss" in text or "recent_losses" in text or "drawdown" in text:
            return "risk_limit"
        if "cluster" in text:
            return "execution_cluster"
        if "m1_quality" in text or "structure" in text or "confidence" in text or "evidence" in text:
            return "setup_quality"
        if "session" in text:
            return "session_gate"
        if "executor_missing" in text or "executor_failed" in text:
            return "executor_error"
        return "other"

    def _record_execution_truth(
        self,
        req: dict[str, Any],
        *,
        ok: bool,
        phase: str,
        reason: str = "",
        result: dict[str, Any] | None = None,
        dry_run: bool | None = None,
    ) -> None:
        result_payload = dict(result or {})
        broker_order_id = str(result_payload.get("broker_order_id", "") or "")
        failure_class = "" if ok else self._classify_execution_failure(reason)
        self._last_execution_truth = {
            "ok": bool(ok),
            "phase": str(phase or "unknown"),
            "request_id": str(req.get("id", "") or ""),
            "request_origin": str(req.get("request_origin", "") or ""),
            "requested_by": str(req.get("requested_by", "") or ""),
            "environment": str(req.get("environment", "") or ""),
            "action_type": str(req.get("action_type", "") or ""),
            "payload_summary": str(req.get("payload_summary", "") or ""),
            "execution_status": str(req.get("execution_status", "") or ""),
            "approval_status": str(req.get("approval_status", "") or ""),
            "reason": str(reason or ""),
            "failure_class": failure_class,
            "broker_order_id": broker_order_id,
            "broker_status": str(result_payload.get("status", "") or ""),
            "dry_run": bool(req.get("dry_run", False) if dry_run is None else dry_run),
            "result": result_payload,
            "ts": time.time(),
        }

    def _execute_live_order(self, req: dict[str, Any]) -> dict[str, Any]:
        request_id = str(req.get("id", ""))
        existing = self._order_tracker.get_by_request(request_id)
        if existing is not None and existing.status not in TERMINAL_STATUSES:
            rec = self._reconciler.reconcile_request(request_id)
            self._audit.append(
                "order_reconciliation_event",
                request_id,
                {"ok": rec.ok, "status": rec.status, "reason": rec.reason, "requires_review": rec.requires_review},
            )
            if rec.requires_review:
                return {"ok": False, "reason": "order_reconciliation_required"}
            if rec.status in {"filled", "acknowledged", "partially_filled", "submitted"}:
                return {"ok": True, "broker_order_id": existing.broker_order_id, "status": rec.status, "note": "idempotent_live_order_state"}
            if rec.status in {"rejected", "failed", "cancelled"}:
                return {"ok": False, "reason": rec.reason}

        submit = self._broker_adapter.submit_order(idempotency_key=request_id, request=dict(req))
        self._audit.append(
            "order_submit_attempt",
            request_id,
            {
                "ok": submit.ok,
                "status": submit.status,
                "broker_order_id": submit.broker_order_id,
                "message": submit.message,
            },
        )
        if not submit.broker_order_id:
            self._order_tracker.upsert_submission(
                request_id=request_id,
                broker_order_id="",
                environment=str(req.get("environment", "live")),
                instrument=str(req.get("instrument", "")).upper(),
                side=str(req.get("action_type", "place_trade")),
                size=float(req.get("size", 0.0) or 0.0),
                status="reconciliation_required",
                rejection_reason="missing_broker_order_id",
                reconciliation_status="required",
            )
            return {"ok": False, "reason": "broker_response_invalid"}

        normalized = self._normalize_broker_status(submit.status, ok=submit.ok)
        tracked = self._order_tracker.upsert_submission(
            request_id=request_id,
            broker_order_id=str(submit.broker_order_id),
            environment=str(req.get("environment", "live")),
            instrument=str(req.get("instrument", "")).upper(),
            side=str(req.get("action_type", "place_trade")),
            size=float(req.get("size", 0.0) or 0.0),
            status=normalized,
            rejection_reason=str(submit.message or ""),
            reconciliation_status=("required" if normalized == "reconciliation_required" else "clear"),
        )
        self._order_tracker.update_status(
            request_id=request_id,
            status=normalized,
            fill_qty=float(submit.fill_qty or 0.0),
            avg_fill_price=float(submit.avg_fill_price or 0.0),
            rejection_reason=str(submit.message or ""),
            reconciliation_status=("required" if normalized == "reconciliation_required" else "clear"),
        )
        if normalized == "reconciliation_required":
            return {"ok": False, "reason": "order_reconciliation_required"}
        if normalized in {"rejected", "failed", "cancelled"}:
            return {"ok": False, "reason": f"broker_{normalized}:{submit.message or 'unknown'}"}
        return {"ok": True, "broker_order_id": tracked.broker_order_id, "status": normalized}

    @staticmethod
    def _normalize_broker_status(status: str, *, ok: bool) -> str:
        raw = str(status or "").strip().lower()
        if not ok and raw not in {"rejected", "failed", "cancelled"}:
            return "reconciliation_required"
        mapping = {
            "created": "created",
            "submitted": "submitted",
            "acknowledged": "acknowledged",
            "partial": "partially_filled",
            "partially_filled": "partially_filled",
            "filled": "filled",
            "cancel_requested": "cancel_requested",
            "cancelled": "cancelled",
            "rejected": "rejected",
            "failed": "failed",
            "unknown": "unknown",
        }
        if raw in mapping:
            return mapping[raw]
        if not ok:
            return "failed"
        return "unknown"

    def _order_status_snapshot(self) -> dict[str, Any]:
        counts = self._order_tracker.counts()
        last = self._order_tracker.last_order()
        recon_rows = self.pending_reconciliation_orders(limit=1)
        top_recon = dict(recon_rows[0]) if recon_rows else {}
        orphan_rows = self.list_orphan_orders(include_reviewed=False, limit=1)
        top_orphan = dict(orphan_rows[0]) if orphan_rows else {}
        approvals_used = 0
        for req in self._requests.values():
            if self._is_live_like_environment(str(req.get("environment", "")).lower()) and str(req.get("approval_status", "")) == "approved":
                approvals_used += 1
        return {
            "last_live_order_status": (last.status if last else "none"),
            "pending_live_orders_count": int(counts.get("pending", 0)),
            "partially_filled_count": int(counts.get("partially_filled", 0)),
            "reconciliation_required": bool(int(counts.get("reconciliation_required", 0)) > 0),
            "last_broker_reject_reason": (last.rejection_reason if last and last.status in {"rejected", "failed"} else ""),
            "last_uncertain_execution_state": (
                (last.status if last and (last.reconciliation_status == "required" or last.status == "reconciliation_required") else "")
            ),
            "live_approvals_used": int(approvals_used),
            "top_reconciliation_request_id": str(top_recon.get("internal_request_id", "") or ""),
            "top_reconciliation_reason": str(top_recon.get("rejection_reason", "") or ""),
            "orphan_order_count": len(self.list_orphan_orders(include_reviewed=False, limit=500)),
            "top_orphan_order_id": str(top_orphan.get("broker_order_id", "") or ""),
            "top_orphan_summary": (
                f"{top_orphan.get('broker_order_id', '')} {top_orphan.get('instrument', '')} {top_orphan.get('broker_status', 'unknown')}"
                if top_orphan
                else ""
            ),
        }

    def _reset_daily_pilot_counters(self) -> None:
        self._tiny_live_pilot_attempts_today = 0
        self._tiny_live_pilot_trade_count = 0
        self._tiny_live_pilot_daily_loss_used = 0.0
        self._tiny_live_pilot_consecutive_losses = 0

    def _preflight_gate(self, req: dict[str, Any], *, runtime_flags: dict[str, Any] | None, risk_policy: dict[str, Any] | None) -> dict[str, Any]:
        self._current_gate_request_id = str(req.get("id", "") or "")
        flags = dict(runtime_flags or {})
        strict_runtime = bool(req.get("auto_generated", False) or req.get("strict_runtime_checks", False))
        runtime_check = self._validate_required_runtime_flags(flags, auto_execution=strict_runtime)
        if not runtime_check["ok"]:
            return self._gate_block(str(runtime_check.get("reason", "invalid_runtime_flags")), gate="runtime_flags", detail=dict(runtime_check))
        # Final authority: freshness is re-checked here before actual execution.
        if strict_runtime:
            freshness_check = self._check_runtime_freshness(flags, strict=True)
            if not freshness_check["ok"]:
                return self._gate_block(str(freshness_check.get("reason", "freshness_stale")), gate="runtime_freshness", detail=dict(freshness_check))
        if self._sleep_mode_active:
            self._sleep_alert_count = int(self._sleep_alert_count) + 1
            self._audit.append(
                "sleep_mode_execution_blocked",
                str(req.get("id", "")),
                {"reason": "sleep_mode_active_execution_blocked"},
            )
            return self._gate_block("sleep_mode_active_execution_blocked", gate="sleep_mode")
        if self._kill_switch.status().active:
            return self._gate_block("kill_switch_active", gate="kill_switch", detail={"kill_switch_reason": self._kill_switch.status().reason})
        safety_reason = self._safety_gate_reason(flags=flags)
        if safety_reason:
            return self._gate_block(safety_reason, gate="safety_layer")
        if not self._gate_enabled:
            return self._gate_block("execution_gate_disabled", gate="execution_gate")
        if req.get("approval_required", True) and req.get("approval_status") != "approved":
            return self._gate_block("approval_missing", gate="approval", detail={"approval_status": req.get("approval_status", "")})
        if self._active_request_id and self._active_request_id != req.get("id"):
            return self._gate_block("conflicting_active_execution", gate="active_request", detail={"active_request_id": self._active_request_id})

        validation = validate_execution_request(
            action_type=str(req.get("action_type", "")),
            environment=str(req.get("environment", "")),
            payload_summary=str(req.get("payload_summary", "")),
            runtime_flags=flags,
        )
        req["validation_status"] = "passed" if validation.passed else "failed"
        if not validation.passed:
            return self._gate_block(validation.reason, gate="validation")

        if self._session_control.maybe_reset_new_day():
            self._reset_daily_pilot_counters()
            self._audit.append("session_day_reset", "", {"reason": "new_day"})
            self._persist_state(event="session_day_reset")
        session_decision = self._session_control.check(runtime_flags=flags, opening_new_trade=True)
        if not session_decision.allowed:
            if (
                self._session_override_active
                and self._execution_environment == "demo"
                and str(req.get("environment", "")).strip().lower() != "live_real"
                and session_decision.reason in self._SESSION_OVERRIDE_ELIGIBLE_REASONS
            ):
                self._audit.append(
                    "session_override_applied",
                    str(req.get("id", "")),
                    {
                        "reason": session_decision.reason,
                        "testing_only": True,
                    },
                )
            else:
                self._audit.append(
                    "session_rule_blocked",
                    str(req.get("id", "")),
                    {
                        "reason": session_decision.reason,
                        "session": session_decision.current_session,
                        "timezone": session_decision.timezone_name,
                        "local_time": session_decision.local_time_iso,
                        "window_local": session_decision.session_window_local,
                        "calendar_status": session_decision.calendar_status,
                    },
                )
                if session_decision.forced_flat_required:
                    self._audit.append(
                        "forced_flat_required",
                        str(req.get("id", "")),
                        {"reason": session_decision.reason, "session": session_decision.current_session},
                    )
                return self._gate_block(session_decision.reason, gate="session", detail={"session": session_decision.current_session})

        env = str(req.get("environment", "paper")).strip().lower()
        data_source_reason = self._data_source_integrity_reason(req=req, runtime_flags=flags)
        if data_source_reason:
            return self._gate_block(data_source_reason, gate="data_source")
        if self._is_live_like_environment(env):
            rec = self._run_mandatory_reconciliation(limit=25)
            if not bool(rec.get("ok", False)):
                return self._gate_block(str(rec.get("reason", "reconciliation_required")), gate="reconciliation", detail=rec)
        if env == "live_real":
            return self._gate_block("real_live_disabled", gate="environment", detail={"environment": env})
        runtime_live = self._safe_bool(flags.get("execute_trades", False))
        stage = self._rollout_stage
        pilot_check = self._check_tiny_live_pilot(req, runtime_flags=flags)
        if not pilot_check["ok"]:
            return self._gate_block(str(pilot_check.get("reason", "tiny_live_pilot_limit")), gate="tiny_live_pilot", detail=dict(pilot_check))
        if self._is_live_like_environment(env) and not self._is_live_allowed_for_stage(stage):
            return self._gate_block(f"live_blocked_in_stage:{stage}", gate="rollout_stage", detail={"stage": stage})
        if self._is_live_like_environment(env) and not runtime_live:
            return self._gate_block("runtime_not_live_enabled", gate="runtime_flags")
        if env == "paper" and runtime_live and not bool(req.get("dry_run", True)):
            # Hard separation: refuse non-dry paper request while runtime is live.
            return self._gate_block("paper_request_in_live_runtime", gate="environment")

        rp = dict(risk_policy or {})
        if not rp:
            return self._gate_block("risk_policy_unavailable", gate="risk_policy")
        rollout_check = self._check_rollout_restrictions(req, runtime_flags=flags, risk_policy=rp)
        if not rollout_check["ok"]:
            return self._gate_block(str(rollout_check.get("reason", "rollout_stage_blocked")), gate="rollout_stage", detail=dict(rollout_check))
        # Broker adapter validation runs after all control checks so that
        # higher-priority blocking reasons are surfaced to the operator first.
        # Dry-run requests do not submit orders and bypass adapter validation.
        if not bool(req.get("dry_run", True)):
            adapter_validation = self._validate_broker_adapter_for_environment(env)
            if not bool(adapter_validation.get("ok", False)):
                return self._gate_block(str(adapter_validation.get("reason", "invalid_broker_adapter")), gate="broker_adapter", detail=adapter_validation)
        return self._gate_allow(gate="preflight")

    def _check_tiny_live_pilot(self, req: dict[str, Any], *, runtime_flags: dict[str, Any]) -> dict[str, Any]:
        env = str(req.get("environment", "paper")).lower()
        if not self._is_live_like_environment(env):
            return {"ok": True}
        cfg = self._tiny_live_pilot
        if not bool(cfg.configured):
            return {"ok": True}
        stage = str(self._rollout_stage)
        self._tiny_live_pilot_last_attempt = {
            "ts": time.time(),
            "environment": env,
            "instrument": str(req.get("instrument", "")).upper(),
            "setup_family": str(req.get("setup_family", "")).lower(),
            "size": float(req.get("size", 0.0) or 0.0),
            "stage": stage,
        }
        if stage != str(cfg.rollout_stage_required):
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_stage_mismatch"
            return {"ok": False, "reason": "tiny_live_pilot_stage_mismatch"}
        if not cfg.enabled:
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_disabled"
            return {"ok": False, "reason": "tiny_live_pilot_disabled"}
        if not cfg.approved:
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_not_approved"
            return {"ok": False, "reason": "tiny_live_pilot_not_approved"}
        if cfg.paused:
            self._tiny_live_pilot_last_block_reason = f"tiny_live_pilot_paused:{cfg.pause_reason or 'paused'}"
            return {
                "ok": False,
                "reason": self._tiny_live_pilot_last_block_reason,
                "code": "tiny_live_pilot_paused",
                "detail": {"pause_reason": cfg.pause_reason or "paused"},
            }
        if cfg.approval_required_per_request and not bool(req.get("live_explicit_confirmation", False)):
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_live_confirmation_required"
            return {"ok": False, "reason": "tiny_live_pilot_live_confirmation_required"}
        inst = str(req.get("instrument", "")).upper()
        if inst != cfg.allowed_instrument:
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_instrument_restricted"
            return {"ok": False, "reason": "tiny_live_pilot_instrument_restricted"}
        setup = str(req.get("setup_family", "")).lower()
        if setup != cfg.allowed_setup_family:
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_setup_restricted"
            return {"ok": False, "reason": "tiny_live_pilot_setup_restricted"}
        size = float(req.get("size", 0.0) or 0.0)
        if size > cfg.size_cap:
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_size_cap_exceeded"
            return {"ok": False, "reason": "tiny_live_pilot_size_cap_exceeded"}
        trade_count = int(runtime_flags.get("day_trade_count", self._tiny_live_pilot_trade_count) or self._tiny_live_pilot_trade_count)
        if trade_count >= cfg.max_trades_per_day:
            self.pause_tiny_live_pilot("daily_trade_cap_reached")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_max_trades_reached"
            return {"ok": False, "reason": "tiny_live_pilot_max_trades_reached"}
        daily_loss = float(runtime_flags.get("daily_loss_pct", 0.0) or 0.0)
        self._tiny_live_pilot_daily_loss_used = max(0.0, daily_loss)
        if daily_loss >= cfg.daily_loss_cap:
            self.pause_tiny_live_pilot("daily_loss_cap_reached")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_daily_loss_cap_hit"
            return {"ok": False, "reason": "tiny_live_pilot_daily_loss_cap_hit"}
        con_losses = int(runtime_flags.get("consecutive_losses", self._tiny_live_pilot_consecutive_losses) or self._tiny_live_pilot_consecutive_losses)
        if con_losses >= cfg.max_consecutive_losses:
            self.pause_tiny_live_pilot("max_consecutive_losses_reached")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_consecutive_loss_cap_hit"
            return {"ok": False, "reason": "tiny_live_pilot_consecutive_loss_cap_hit"}
        sc = dict(self._session_control.to_dict(runtime_flags=runtime_flags) or {})
        if cfg.pause_on_forced_flat and bool(sc.get("forced_flat_required", False)):
            mins = int(sc.get("minutes_to_session_close", -1) or -1)
            if mins <= 15:
                self.pause_tiny_live_pilot("forced_flat_urgency")
                self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_forced_flat_urgency"
                return {"ok": False, "reason": "tiny_live_pilot_forced_flat_urgency"}
        blocker = float(runtime_flags.get("blocker_severity", 0.0) or 0.0)
        if blocker > cfg.pause_on_blocker_threshold:
            self.pause_tiny_live_pilot("blocker_threshold")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_blocker_threshold"
            return {"ok": False, "reason": "tiny_live_pilot_blocker_threshold"}
        stability = float(runtime_flags.get("stability_score", 1.0) or 1.0)
        if stability < cfg.pause_on_stability_drop:
            self.pause_tiny_live_pilot("stability_drop")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_stability_drop"
            return {"ok": False, "reason": "tiny_live_pilot_stability_drop"}
        overreaction = float(runtime_flags.get("overreaction_score", 0.0) or 0.0)
        if overreaction > cfg.pause_on_overreaction:
            self.pause_tiny_live_pilot("overreaction")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_overreaction"
            return {"ok": False, "reason": "tiny_live_pilot_overreaction"}
        if cfg.pause_on_degrading_performance and str(runtime_flags.get("performance_direction", "")).lower() == "degrading":
            self.pause_tiny_live_pilot("degrading_performance")
            self._tiny_live_pilot_last_block_reason = "tiny_live_pilot_degrading_performance"
            return {"ok": False, "reason": "tiny_live_pilot_degrading_performance"}
        self._tiny_live_pilot_last_block_reason = ""
        self._tiny_live_pilot_last_attempt["status"] = "eligible"  # type: ignore[index]
        return {"ok": True}

    def _check_rollout_restrictions(self, req: dict[str, Any], *, runtime_flags: dict[str, Any], risk_policy: dict[str, Any]) -> dict[str, Any]:
        stage = self._rollout_stage
        env = str(req.get("environment", "paper")).strip().lower()
        if not self._is_live_like_environment(env):
            return {"ok": True}
        restrictions = self._stage_restrictions(stage)
        if restrictions.get("allowed_instruments"):
            inst = str(req.get("instrument", "")).upper()
            allowed = {str(x).upper() for x in list(restrictions.get("allowed_instruments", []))}
            if inst and inst not in allowed:
                return {"ok": False, "reason": "instrument_restricted_in_stage"}
        if restrictions.get("allowed_setup_families"):
            setup = str(req.get("setup_family", "")).lower()
            allowed_setup = {str(x).lower() for x in list(restrictions.get("allowed_setup_families", []))}
            if setup and setup not in allowed_setup:
                return {"ok": False, "reason": "setup_family_restricted_in_stage"}
        max_size = float(restrictions.get("max_size", 0.0) or 0.0)
        size = float(req.get("size", 0.0) or 0.0)
        if max_size > 0.0 and size > max_size:
            return {"ok": False, "reason": "size_cap_exceeded_in_stage"}

        daily_loss = float(runtime_flags.get("daily_loss_pct", 0.0) or 0.0)
        daily_loss_cap = float(restrictions.get("daily_loss_cap_pct", risk_policy.get("daily_loss_cap_pct", 0.0)) or 0.0)
        if daily_loss_cap > 0.0 and daily_loss >= daily_loss_cap:
            return {"ok": False, "reason": "daily_loss_cap_hit"}
        return {"ok": True}

    def _maybe_regress_from_failure(self, *, reason: str) -> None:
        # Conservative regression triggers for rollout safety.
        bad = str(reason or "").lower()
        regression_signals = ("daily_loss_cap_hit", "daily_loss_limit_reached", "kill_switch_active", "executor_failed", "size_cap_exceeded_in_stage")
        if any(sig in bad for sig in regression_signals):
            prev = self._rollout_stage
            self._rollout_stage = self._previous_stage(prev)
            self._rollout_stage_updated_at = time.time()
            self._regression_triggered = self._rollout_stage != prev
            self._regression_reason = reason
            if self._regression_triggered:
                self._audit.append("rollout_regressed", "", {"from": prev, "to": self._rollout_stage, "reason": reason})

    def _automation_preflight(
        self,
        *,
        environment: str,
        instrument: str,
        setup_family: str,
        size: float,
        session: str,
        confidence: float,
        evidence_strength: float,
        runtime_flags: dict[str, Any],
        risk_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        flags = dict(runtime_flags or {})
        rp = dict(risk_policy or {})
        runtime_check = self._validate_required_runtime_flags(flags, auto_execution=True)
        if not runtime_check["ok"]:
            self._audit.append("automation_preflight_blocked", "", {"reason": runtime_check["reason"]})
            return runtime_check
        policy = self._automation_profiles.get(self._active_automation_policy_id) if self._active_automation_policy_id else None
        base_reason = self._auto_trading_gate_reason(
            runtime_flags=flags,
            environment=environment,
            instrument=instrument,
            setup_family=setup_family,
            size=size,
            session=session,
            policy=policy,
        )
        if base_reason:
            if "sleep_mode_active" in base_reason:
                self._sleep_alert_count = int(self._sleep_alert_count) + 1
                self._audit.append("sleep_mode_automation_blocked", "", {"reason": base_reason})
            return {"ok": False, "reason": base_reason}
        autonomy_reason = self._autonomy_gate_reason(environment=environment, runtime_flags=flags)
        if autonomy_reason:
            self._audit.append("automation_preflight_blocked", "", {"reason": autonomy_reason})
            return {"ok": False, "reason": autonomy_reason}
        freshness_check = self._check_runtime_freshness(flags, strict=True)
        if not freshness_check["ok"]:
            self._audit.append("automation_preflight_blocked", "", {"reason": freshness_check["reason"]})
            return freshness_check
        if self._session_control.maybe_reset_new_day():
            self._reset_daily_pilot_counters()
            self._audit.append("session_day_reset", "", {"reason": "new_day"})
            self._persist_state(event="session_day_reset")
        if self._kill_switch.status().active:
            return {"ok": False, "reason": "kill_switch_active"}
        if not self._gate_enabled:
            return {"ok": False, "reason": "execution_gate_disabled"}
        if self._automation_paused:
            return {"ok": False, "reason": f"automation_paused:{self._automation_pause_reason or 'paused'}"}
        if not self._active_automation_policy_id:
            return {"ok": False, "reason": "no_active_automation_policy"}
        policy = self._automation_profiles.get(self._active_automation_policy_id)
        if policy is None:
            return {"ok": False, "reason": "automation_policy_not_found"}
        if not policy.enabled:
            return {"ok": False, "reason": "automation_policy_disabled"}
        if policy.approval_required_for_activation and not policy.approved:
            return {"ok": False, "reason": "automation_policy_not_approved"}

        env = str(environment or "").lower().strip()
        if env == "live":
            env = "demo"
        policy_env = str(policy.environment or "").lower()
        env_matches_policy = policy_env == "both" or env == policy_env or (policy_env == "live" and env == "demo")
        if not env_matches_policy:
            return {"ok": False, "reason": "automation_policy_environment_block"}
        if self._is_live_like_environment(env) and not self._is_live_allowed_for_stage(self._rollout_stage):
            return {"ok": False, "reason": f"live_blocked_in_stage:{self._rollout_stage}"}
        if policy.active_rollout_stage and policy.active_rollout_stage != self._rollout_stage:
            return {"ok": False, "reason": "policy_stage_mismatch"}
        if self._is_live_like_environment(env):
            pilot_req = {
                "environment": env,
                "instrument": str(instrument or "").upper().strip(),
                "setup_family": str(setup_family or "").lower().strip(),
                "size": float(size or 0.0),
                "live_explicit_confirmation": True,
            }
            pilot_gate = self._check_tiny_live_pilot(pilot_req, runtime_flags=flags)
            if not pilot_gate.get("ok", False):
                return {"ok": False, "reason": str(pilot_gate.get("reason", "tiny_live_pilot_blocked"))}

        inst = str(instrument or "").upper().strip()
        if policy.allowed_instruments and inst not in {x.upper() for x in policy.allowed_instruments}:
            return {"ok": False, "reason": "instrument_not_approved_for_automation"}
        setup = str(setup_family or "").lower().strip()
        if policy.allowed_setup_families and setup not in {x.lower() for x in policy.allowed_setup_families}:
            return {"ok": False, "reason": "setup_family_not_approved_for_automation"}
        ses = str(session or "").lower().strip()
        if policy.allowed_sessions and ses not in {x.lower() for x in policy.allowed_sessions}:
            return {"ok": False, "reason": "session_not_allowed_for_automation"}
        if float(size or 0.0) > float(policy.size_cap or 0.0):
            return {"ok": False, "reason": "automation_size_cap_exceeded"}
        risk_pct = self._safe_float(flags.get("trade_risk_pct"))
        max_risk_pct = self._safe_float(rp.get("max_risk_per_trade_pct"))
        if risk_pct is not None and max_risk_pct is not None and risk_pct > max_risk_pct:
            return {"ok": False, "reason": "trade_risk_cap_exceeded"}
        spread_reason = self._spread_gate_reason(flags=flags, risk_policy=rp)
        if spread_reason:
            return {"ok": False, "reason": spread_reason}
        news_reason = self._event_risk_gate_reason(flags=flags)
        if news_reason:
            return {"ok": False, "reason": news_reason}
        if float(flags.get("daily_loss_pct", 0.0) or 0.0) >= float(policy.daily_loss_cap or 0.0):
            return {"ok": False, "reason": "automation_daily_loss_cap_hit"}
        if int(policy.max_trades_per_session or 0) > 0:
            current = int(self._session_trade_counts.get(ses, 0))
            if current >= int(policy.max_trades_per_session):
                return {"ok": False, "reason": "automation_max_trades_per_session_hit"}

        blocker_severity = float(flags.get("blocker_severity", 0.0) or 0.0)
        regime_noise = float(flags.get("regime_noise", 0.0) or 0.0)
        stability_score = float(flags.get("stability_score", 1.0) or 1.0)
        overreaction_score = float(flags.get("overreaction_score", 0.0) or 0.0)
        recent_losses = float(flags.get("recent_losses", 0.0) or 0.0)
        if blocker_severity > float(policy.pause_conditions.get("max_blocker_severity", 1.0)):
            return {"ok": False, "reason": "blocker_severity_too_high"}
        if regime_noise > float(policy.pause_conditions.get("max_regime_noise", 1.0)):
            return {"ok": False, "reason": "regime_noise_too_high"}
        if stability_score < float(policy.pause_conditions.get("min_stability_score", 0.0)):
            return {"ok": False, "reason": "stability_too_low"}
        if overreaction_score > float(policy.pause_conditions.get("max_overreaction_score", 1.0)):
            return {"ok": False, "reason": "overreaction_too_high"}
        if recent_losses >= float(policy.pause_conditions.get("max_recent_losses", 999.0)):
            return {"ok": False, "reason": "recent_losses_threshold_hit"}
        if float(confidence or 0.0) < float(policy.pause_conditions.get("min_confidence", 0.0)):
            return {"ok": False, "reason": "confidence_below_automation_bound"}
        if float(evidence_strength or 0.0) < float(policy.pause_conditions.get("min_evidence_strength", 0.0)):
            return {"ok": False, "reason": "evidence_below_automation_bound"}
        if str(flags.get("performance_direction", "")).lower() == "degrading":
            return {"ok": False, "reason": "performance_degrading_escalation"}
        if self._active_request_id:
            return {"ok": False, "reason": "conflicting_active_execution"}
        self._audit.append("automation_preflight_passed", "", {"environment": environment, "instrument": instrument, "setup_family": setup_family})
        return {"ok": True}

    def _autonomy_gate_reason(self, *, environment: str, runtime_flags: dict[str, Any]) -> str:
        env = str(environment or "").strip().lower()
        if env == "live":
            env = "demo"
        if self._autonomy_mode == "observe_only":
            return "autonomy_mode_observe_only"
        if self._autonomy_mode == "recommend_only":
            return "autonomy_mode_recommend_only"
        if self._autonomy_mode == "full_live_disabled_by_default":
            return "full_live_autonomy_disabled_by_default"
        if self._autonomy_mode == "guarded_autonomous_demo" and env != "demo":
            return "guarded_autonomous_demo_requires_demo_environment"
        market_state_mode = self._resolved_market_state_mode(runtime_flags=runtime_flags, environment=env)
        if market_state_mode == "simulation" and self._is_live_like_environment(env):
            return "simulation_market_state_cannot_execute"
        return ""

    @classmethod
    def _autonomy_mode_semantics(cls) -> dict[str, dict[str, Any]]:
        return {
            "observe_only": {
                "can_create_execution_request": False,
                "can_submit_order": False,
                "block_reason": "autonomy_mode_observe_only",
            },
            "recommend_only": {
                "can_create_execution_request": False,
                "can_submit_order": False,
                "block_reason": "autonomy_mode_recommend_only",
            },
            "guarded_execution": {
                "can_create_execution_request": True,
                "can_submit_order": True,
                "requires_existing_gates": True,
                "block_reason": "",
            },
            "guarded_autonomous_demo": {
                "can_create_execution_request": True,
                "can_submit_order": True,
                "requires_existing_gates": True,
                "demo_only": True,
                "real_live_allowed": False,
                "block_reason": "guarded_autonomous_demo_requires_demo_environment",
            },
            "full_live_disabled_by_default": {
                "can_create_execution_request": False,
                "can_submit_order": False,
                "block_reason": "full_live_autonomy_disabled_by_default",
            },
        }

    @staticmethod
    def _load_autonomous_trader_mandate_summary() -> dict[str, Any]:
        path = Path(__file__).resolve().parents[2] / "knowledge" / "autonomous_trader_mandate.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {
                "schema": "atlas_autonomous_trader_mandate_v1",
                "status": "missing_or_unreadable",
                "target_role": "bounded_autonomous_trader",
                "execution_authority": "ExecutionGate",
            }
        return {
            "schema": str(payload.get("schema", "atlas_autonomous_trader_mandate_v1") or "atlas_autonomous_trader_mandate_v1"),
            "status": str(payload.get("status", "unknown") or "unknown"),
            "target_role": str(dict(payload.get("autonomy_model", {}) or {}).get("target_role", "bounded_autonomous_trader") or "bounded_autonomous_trader"),
            "default_environment": str(dict(payload.get("initial_scope", {}) or {}).get("default_environment", "guarded_demo") or "guarded_demo"),
            "real_live_trading": str(dict(payload.get("initial_scope", {}) or {}).get("real_live_trading", "disabled_until_explicit_operator_approval") or "disabled_until_explicit_operator_approval"),
            "execution_authority": "ExecutionGate",
            "hard_boundaries_count": len(list(payload.get("hard_boundaries", []) or [])),
        }

    def _spread_gate_reason(self, *, flags: dict[str, Any], risk_policy: dict[str, Any]) -> str:
        spread_state = str(flags.get("spread_state", "") or "").strip().lower()
        if spread_state == "wide":
            return "spread_filter_blocked"
        spread_points = self._safe_float(flags.get("spread_points"))
        max_spread = self._safe_float(risk_policy.get("max_spread_points"))
        if max_spread is None:
            max_spread = self._safe_float(flags.get("max_spread_points"))
        if max_spread is None:
            max_spread = 35.0
        if spread_points is not None and spread_points > max_spread:
            return "spread_filter_blocked"
        liquidity_state = str(flags.get("liquidity_state", "") or "").strip().lower()
        if liquidity_state in {"poor", "below_normal", "illiquid"}:
            return "liquidity_filter_blocked"
        return ""

    @staticmethod
    def _event_risk_gate_reason(*, flags: dict[str, Any]) -> str:
        if ExecutionGate._safe_bool(flags.get("news_risk_active", False)):
            return "news_event_filter_blocked"
        event_state = str(flags.get("event_risk_state", flags.get("news_risk_state", "")) or "").strip().lower()
        if event_state in {"high", "elevated", "locked"}:
            return "news_event_filter_blocked"
        return ""

    @staticmethod
    def _normalize_data_source(value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"simulation", "simulated", "paper_simulation"}:
            return "simulation"
        if raw in {"broker", "live_feed", "runtime", "broker_runtime", "market_feed"}:
            return "broker_runtime"
        return raw

    def _resolved_market_state_mode(self, *, runtime_flags: dict[str, Any], environment: str = "") -> str:
        """Resolve market-state mode consistently across runtime flags.

        FIX: avoid stale simulation leakage by normalizing market_state_mode,
        market_data_mode, and data_source with live-like broker-runtime preference.
        """
        flags = dict(runtime_flags or {})
        env = str(environment or "").strip().lower()
        raw_mode = str(flags.get("market_state_mode", flags.get("market_data_mode", "")) or "").strip()
        raw_source = str(flags.get("data_source", "") or "").strip()
        mode = self._normalize_data_source(raw_mode)
        source = self._normalize_data_source(raw_source)
        declared: list[str] = []
        if mode:
            declared.append(mode)
        if source:
            declared.append(source)
        extra = flags.get("context_data_sources", None)
        if isinstance(extra, (list, tuple, set)):
            for item in extra:
                n = self._normalize_data_source(item)
                if n:
                    declared.append(n)
        unique = set(x for x in declared if x)
        if self._is_live_like_environment(env) and "broker_runtime" in unique:
            return "broker_runtime"
        if "simulation" in unique:
            return "simulation"
        if "broker_runtime" in unique:
            return "broker_runtime"
        return "broker_runtime" if self._is_live_like_environment(env) else "broker"

    def _data_source_integrity_reason(self, *, req: dict[str, Any], runtime_flags: dict[str, Any]) -> str:
        env = str(req.get("environment", "") or "").strip().lower()
        req_source = self._normalize_data_source(req.get("data_source", ""))
        req_mode = str(req.get("environment_mode", "") or "").strip().lower()
        if req_mode and req_mode != env:
            return "environment_mode_mismatch"
        runtime_mode = self._resolved_market_state_mode(runtime_flags=runtime_flags, environment=env)
        has_explicit_runtime_source = any(
            str(runtime_flags.get(k, "") or "").strip()
            for k in ("market_state_mode", "market_data_mode", "data_source")
        )
        runtime_source = self._normalize_data_source(runtime_flags.get("data_source", runtime_mode)) if has_explicit_runtime_source else ""
        declared = []
        if req_source:
            declared.append(req_source)
        if runtime_source:
            declared.append(runtime_source)
        extra = runtime_flags.get("context_data_sources", None)
        if isinstance(extra, (list, tuple, set)):
            for item in extra:
                n = self._normalize_data_source(item)
                if n:
                    declared.append(n)
        unique = set(x for x in declared if x)
        # Keep strict mixed-source and simulation blocks for live-like only.
        if self._is_live_like_environment(env) and "simulation" in unique and "broker_runtime" in unique:
            return "mixed_data_sources_detected"
        if self._is_live_like_environment(env) and (req_source == "simulation" or runtime_source == "simulation"):
            return "simulation_data_source_blocked_for_live_like_execution"
        return ""

    def _adapter_type(self) -> str:
        return type(self._broker_adapter).__name__

    def _validate_broker_adapter_for_environment(self, environment: str) -> dict[str, Any]:
        env = str(environment or "").strip().lower()
        if env == "live":
            env = "demo"
        adapter_type = self._adapter_type()
        if self._is_live_like_environment(env) and adapter_type == "InMemoryBrokerAdapter":
            return {"ok": False, "reason": "in_memory_adapter_blocked_for_live", "adapter_type": adapter_type}
        return {"ok": True, "adapter_type": adapter_type}

    def _safety_gate_reason(self, *, flags: dict[str, Any]) -> str:
        safety = dict(flags.get("safety_layer", {}) or {})
        if self._safe_bool(flags.get("safety_lock_active", safety.get("safety_lock_active", False))):
            return "safety_lock_active"
        if self._safe_bool(flags.get("cooldown_active", safety.get("cooldown_active", False))):
            return "cooldown_active"
        volatility_state = str(flags.get("volatility_state", safety.get("volatility_state", "")) or "").strip().lower()
        volatility_action = str(flags.get("volatility_action", safety.get("volatility_action", "")) or "").strip().lower()
        if volatility_action == "block_new_entries" or volatility_state == "extreme":
            return "volatility_block_active"
        return ""

    def _resolve_reconciliation_interval_seconds(self) -> int:
        raw = str(os.getenv("ATLAS_RECONCILIATION_INTERVAL_SECONDS", "120") or "120").strip()
        try:
            return max(15, min(900, int(raw)))
        except Exception:
            return 120

    def _start_reconciliation_timer_if_enabled(self) -> None:
        raw = os.getenv("ATLAS_ENABLE_GATE_RECONCILIATION_TIMER", "")
        if str(raw).strip():
            enabled = str(raw).strip().lower() in {"1", "true", "yes", "on"}
        else:
            enabled = not bool(os.getenv("PYTEST_CURRENT_TEST"))
        if not enabled:
            return
        if self._reconciliation_thread is not None and self._reconciliation_thread.is_alive():
            return
        self._reconciliation_thread = threading.Thread(target=self._reconciliation_timer_loop, daemon=True)
        self._reconciliation_thread.start()
        self._audit.append("reconciliation_timer_started", "", {"interval_seconds": int(self._reconciliation_interval_seconds)})

    def _reconciliation_timer_loop(self) -> None:
        while not self._reconciliation_stop.wait(self._reconciliation_interval_seconds):
            try:
                self._run_mandatory_reconciliation(limit=25)
            except Exception as exc:
                self._audit.append("reconciliation_timer_error", "", {"reason": f"{exc.__class__.__name__}:{exc}"})

    def _run_mandatory_reconciliation(self, *, limit: int = 25) -> dict[str, Any]:
        with self._reconciliation_lock:
            if self._adapter_type() == "InMemoryBrokerAdapter":
                self._audit.append(
                    "mandatory_reconciliation_skipped",
                    "",
                    {"reason": "in_memory_adapter_not_authoritative_for_shared_state"},
                )
                return {"ok": True, "status": "skipped", "reason": "in_memory_adapter_not_authoritative"}
            # The UI command path and live worker each hold an ExecutionGate.
            # Pull shared operator controls before persisting reconciliation so
            # a live worker cannot overwrite a freshly resumed automation state.
            self._sync_shared_controls_for_operation("mandatory_reconciliation")
            scan = self.scan_broker_open_orders()
            rec = self.reconcile_now(limit=limit)
            snap = self._order_status_snapshot()
            mismatches = int(scan.get("mismatches_detected", 0) or 0)
            orphans = int(scan.get("orphans_detected", 0) or 0)
            recon_required = bool(snap.get("reconciliation_required", False))
            orphan_count = int(snap.get("orphan_order_count", 0) or 0)
            mismatch_detected = bool(
                mismatches > 0
                or orphans > 0
                or recon_required
                or orphan_count > 0
                or not bool(scan.get("ok", True))
                or not bool(rec.get("ok", True))
            )
            self._mismatch_detected = mismatch_detected
            self._last_reconciled_at = time.time()
            self._reconciliation_status = "mismatch_detected" if mismatch_detected else "clear"
            self._audit.append(
                "mandatory_reconciliation_completed",
                "",
                {
                    "status": self._reconciliation_status,
                    "mismatch_detected": bool(self._mismatch_detected),
                    "orphans_detected": orphans,
                    "mismatches_detected": mismatches,
                    "reconciliation_required": recon_required,
                    "orphan_order_count": orphan_count,
                    "scan_status": str(scan.get("status", "") or ""),
                    "scan_reason": str(scan.get("reason", "") or ""),
                },
            )
            self._persist_state(event="mandatory_reconciliation_completed")
            if mismatch_detected:
                return {"ok": False, "reason": "reconciliation_required", "status": self._reconciliation_status}
            return {"ok": True, "status": self._reconciliation_status}

    def _derived_operating_mode(self) -> str:
        return str(self._environment_model().get("operating_mode", "paper") or "paper")

    def _environment_model(self, requested_environment: str | None = None) -> dict[str, Any]:
        requested = str(requested_environment or self._execution_environment or "paper").strip().lower()
        canonical = self._resolve_request_environment(requested)
        return {
            "requested": requested,
            "canonical": canonical,
            "execution_environment": self._execution_environment,
            "operating_mode": "guarded_demo" if canonical == "demo" else ("real_live_disabled" if canonical == "live_real" else "paper"),
            "live_like": self._is_live_like_environment(canonical),
            "real_live": canonical == "live_real",
            "real_live_enabled": False,
            "label": "Live Demo" if canonical == "demo" else ("Real Live (disabled / not in use)" if canonical == "live_real" else "Paper"),
        }

    def _build_autonomy_cycle_record(
        self,
        *,
        signal_id: str,
        action_type: str,
        environment: str,
        instrument: str,
        setup_family: str,
        size: float,
        session: str,
        confidence: float,
        evidence_strength: float,
        runtime_flags: dict[str, Any],
        market_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot = dict(market_snapshot or {})
        flags = dict(runtime_flags or {})
        market_state_mode = self._resolved_market_state_mode(runtime_flags=flags, environment=environment)
        return {
            "signal_id": str(signal_id or ""),
            "market_snapshot": snapshot,
            "market_state_mode": market_state_mode,
            "simulation_market_state": market_state_mode.strip().lower() == "simulation",
            "operating_mode": self._derived_operating_mode(),
            "autonomy_mode": self._autonomy_mode,
            "instrument": str(instrument or "").upper(),
            "setup_family": str(setup_family or "").lower(),
            "xau_bias": str(snapshot.get("xau_bias", flags.get("xau_bias", "unknown")) or "unknown"),
            "dxy_alignment": str(snapshot.get("dxy_alignment", flags.get("dxy_alignment", flags.get("dxy_conflict", "unknown"))) or "unknown"),
            "us10y_alignment": str(snapshot.get("us10y_alignment", flags.get("us10y_alignment", flags.get("us10y_conflict", "unknown"))) or "unknown"),
            "spread_liquidity_state": {
                "spread_state": str(snapshot.get("spread_state", flags.get("spread_state", "unknown")) or "unknown"),
                "liquidity_state": str(snapshot.get("liquidity_state", flags.get("liquidity_state", "unknown")) or "unknown"),
                "spread_points": flags.get("spread_points", snapshot.get("spread_points")),
            },
            "session": str(session or snapshot.get("session", flags.get("session", "unknown")) or "unknown"),
            "confidence": float(confidence or 0.0),
            "evidence_strength": float(evidence_strength or 0.0),
            "requested_action": {
                "action_type": str(action_type or ""),
                "environment": str(environment or ""),
                "size": float(size or 0.0),
            },
            "action_taken": "pending",
            "blocker_reasons": [],
            "execution_result": {},
            "ts": time.time(),
        }

    def _record_autonomy_cycle_outcome(
        self,
        *,
        action: str,
        blocker_reasons: list[str],
        execution_result: dict[str, Any],
    ) -> None:
        current = dict(self._last_autonomy_cycle or {})
        if not current:
            return
        current["action_taken"] = str(action or "unknown")
        current["blocker_reasons"] = [str(x) for x in list(blocker_reasons or []) if str(x).strip()]
        current["execution_result"] = dict(execution_result or {})
        current["completed_at"] = time.time()
        self._last_autonomy_cycle = current
        self._audit.append("autonomy_cycle_completed", "", current)
        self._persist_state(event="autonomy_cycle_completed")

    def _maybe_pause_automation(self, *, reason: str, runtime_flags: dict[str, Any]) -> None:
        hard_signals = {
            "kill_switch_active",
            "automation_daily_loss_cap_hit",
            "recent_losses_threshold_hit",
            "performance_degrading_escalation",
            "daily_loss_cap_hit",
        }
        soft_signals = {
            "blocker_severity_too_high",
            "regime_noise_too_high",
            "stability_too_low",
            "overreaction_too_high",
        }
        reason_text = str(reason)
        hard_pause = any(sig in reason_text for sig in hard_signals)
        soft_pause = any(sig in reason_text for sig in soft_signals)
        if soft_pause and not hard_pause and self._automation_explicit_resume_active:
            self._audit.append(
                "automation_pause_suppressed_by_explicit_resume",
                "",
                {"reason": reason_text, "runtime": dict(runtime_flags or {}), "policy_id": self._automation_explicit_resume_policy_id},
            )
            return
        if hard_pause or soft_pause:
            self._automation_paused = True
            self._automation_pause_reason = reason_text
            if hard_pause:
                self._automation_explicit_resume_active = False
                self._automation_explicit_resume_at = 0.0
                self._automation_explicit_resume_policy_id = ""
            self._audit.append("automation_paused", "", {"reason": reason_text, "runtime": dict(runtime_flags or {})})
            self._persist_state(event="automation_paused")

    @classmethod
    def _is_live_stage(cls, stage: str) -> bool:
        return stage in {"limited_live_pilot", "restricted_live_operations", "scaled_live_reviewed"}

    @staticmethod
    def _is_live_like_environment(environment: str) -> bool:
        env = str(environment or "").strip().lower()
        return env in {"live", "demo"}

    def _resolve_request_environment(self, environment: str) -> str:
        env = str(environment or "").strip().lower()
        if env in {"paper", "demo"}:
            return env
        if env == "live":
            return "demo"
        if env in {"live_real", "real_live", "real"}:
            return "live_real"
        return self._execution_environment if self._execution_environment in {"paper", "demo"} else "paper"

    def _auto_trading_gate_reason(
        self,
        *,
        runtime_flags: dict[str, Any],
        environment: str,
        instrument: str,
        setup_family: str,
        size: float,
        session: str,
        policy: AutomationPolicy | None,
        require_freshness: bool = True,
    ) -> str:
        env = str(environment or "").strip().lower()
        if env == "live":
            env = "demo"
        if env in {"live_real", "real_live", "real"}:
            return "real_live_auto_trading_disabled"
        if env != "demo":
            return "auto_trading_demo_only"
        if self._execution_environment != "demo":
            return "execution_environment_not_demo"
        if self._sleep_mode_active:
            return "sleep_mode_active_automation_blocked"
        if not self._operator_present:
            return "operator_not_present"
        if self._kill_switch.status().active:
            return "kill_switch_active"
        if not self._gate_enabled:
            return "execution_gate_disabled"
        if policy is None:
            return "no_active_automation_policy"
        if not policy.enabled:
            return "automation_policy_disabled"
        if policy.approval_required_for_activation and not policy.approved:
            return "automation_policy_not_approved"
        if bool(self._order_status_snapshot().get("reconciliation_required", False)):
            return "reconciliation_required"
        if int(self._order_status_snapshot().get("orphan_order_count", 0) or 0) > 0:
            return "orphan_orders_pending_review"
        session_decision = self._session_control.check(runtime_flags=runtime_flags, opening_new_trade=True)
        if not session_decision.allowed:
            if (
                self._session_override_active
                and session_decision.reason in self._SESSION_OVERRIDE_ELIGIBLE_REASONS
            ):
                pass
            else:
                return str(session_decision.reason or "outside_active_session")
        if bool(session_decision.forced_flat_required):
            return "forced_flat_review_required"
        if session_decision.minutes_to_close >= 0 and session_decision.minutes_to_close <= 15:
            return "forced_flat_urgency"
        cfg = self._tiny_live_pilot
        if not bool(cfg.configured):
            return "tiny_live_pilot_not_configured"
        stage = str(self._rollout_stage)
        if stage != str(cfg.rollout_stage_required):
            return "tiny_live_pilot_stage_mismatch"
        if not cfg.enabled:
            return "tiny_live_pilot_disabled"
        if not cfg.approved:
            return "tiny_live_pilot_not_approved"
        if cfg.paused:
            return f"tiny_live_pilot_paused:{cfg.pause_reason or 'paused'}"
        if str(instrument or "").upper() != cfg.allowed_instrument:
            return "tiny_live_pilot_instrument_restricted"
        if str(setup_family or "").lower() != cfg.allowed_setup_family:
            return "tiny_live_pilot_setup_restricted"
        if float(size or 0.0) > float(cfg.size_cap):
            return "tiny_live_pilot_size_cap_exceeded"
        trade_count = int(runtime_flags.get("day_trade_count", self._tiny_live_pilot_trade_count) or self._tiny_live_pilot_trade_count)
        if trade_count >= int(cfg.max_trades_per_day):
            return "tiny_live_pilot_max_trades_reached"
        daily_loss = float(runtime_flags.get("daily_loss_pct", 0.0) or 0.0)
        if daily_loss >= float(cfg.daily_loss_cap):
            return "tiny_live_pilot_daily_loss_cap_hit"
        con_losses = int(runtime_flags.get("consecutive_losses", self._tiny_live_pilot_consecutive_losses) or self._tiny_live_pilot_consecutive_losses)
        if con_losses >= int(cfg.max_consecutive_losses):
            return "tiny_live_pilot_consecutive_loss_cap_hit"
        if require_freshness:
            freshness_check = self._check_runtime_freshness(runtime_flags, strict=True)
            if not freshness_check["ok"]:
                return str(freshness_check.get("reason", "freshness_stale"))
        return ""

    @classmethod
    def _should_record_auto_signal_result(cls, out: dict[str, Any]) -> bool:
        if bool(out.get("ok", False)):
            return True
        reason = str(out.get("reason", "") or "").strip().lower()
        if not reason:
            return False
        if reason in cls._AUTO_SIGNAL_NON_RECORD_REASONS:
            return False
        if reason.startswith("tiny_live_pilot_paused:"):
            return False
        return True

    def _auto_signal_dedupe_ttl_seconds(self) -> float:
        raw = os.getenv("ATLAS_AUTO_SIGNAL_DEDUPE_TTL_SECONDS", "")
        try:
            ttl = float(raw) if str(raw).strip() else self._AUTO_SIGNAL_DEDUPE_TTL_SECONDS
        except Exception:
            ttl = self._AUTO_SIGNAL_DEDUPE_TTL_SECONDS
        return max(5.0, min(600.0, ttl))

    def _prune_auto_signal_history(self, *, now: float | None = None) -> None:
        ts_now = float(now if now is not None else time.time())
        ttl = self._auto_signal_dedupe_ttl_seconds()
        cutoff = ts_now - ttl
        kept: dict[str, float] = {}
        for sig, seen in sorted(self._auto_signal_history.items(), key=lambda kv: kv[1], reverse=True):
            if len(kept) >= self._AUTO_SIGNAL_DEDUPE_MAX:
                break
            if float(seen) < cutoff:
                continue
            kept[str(sig)] = float(seen)
        self._auto_signal_history = kept

    def _is_duplicate_auto_signal(self, signature: str) -> dict[str, Any]:
        sig = str(signature or "").strip()
        if not sig:
            return {"duplicate": False}
        now_ts = time.time()
        self._prune_auto_signal_history(now=now_ts)
        prev = self._auto_signal_history.get(sig)
        if prev is None:
            return {"duplicate": False}
        age = max(0.0, now_ts - float(prev))
        if age <= self._auto_signal_dedupe_ttl_seconds():
            return {"duplicate": True, "age_seconds": age}
        return {"duplicate": False}

    def _record_auto_signal_signature(self, signature: str) -> None:
        sig = str(signature or "").strip()
        if not sig:
            return
        now_ts = time.time()
        self._last_auto_signal_signature = sig
        self._auto_signal_history[sig] = now_ts
        self._prune_auto_signal_history(now=now_ts)

    @staticmethod
    def _safe_float(raw: Any) -> float | None:
        try:
            return float(raw)
        except Exception:
            return None

    @staticmethod
    def _safe_int(raw: Any) -> int | None:
        try:
            return int(raw)
        except Exception:
            return None

    @staticmethod
    def _safe_bool(raw: Any, *, default: bool = False) -> bool:
        if isinstance(raw, bool):
            return raw
        if raw is None:
            return bool(default)
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        if text in {"1", "true", "yes", "on", "y"}:
            return True
        if text in {"0", "false", "no", "off", "n", ""}:
            return False
        return bool(default)

    @staticmethod
    def _reason_code(reason: str) -> str:
        return str(reason or "").split(":", 1)[0].strip() or "unknown"

    @staticmethod
    def _checked_at() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _structured_reason_code(cls, reason: str, *, gate: str = "") -> str:
        legacy = cls._reason_code(reason).lower()
        if legacy.startswith("tiny_live_pilot_"):
            return "TINY_LIVE_PILOT_LIMIT"
        if legacy in {"outside_active_session", "near_session_open_no_new_trades", "near_session_close_no_new_trades", "early_close_no_new_trades", "insufficient_time_to_close", "market_closed_holiday"}:
            return "SESSION_CLOSED"
        if legacy.startswith("live_blocked_in_stage") or legacy.endswith("_restricted_in_stage") or legacy.endswith("_exceeded_in_stage"):
            return "ROLLOUT_STAGE_BLOCKED"
        if "reconciliation" in legacy or "mismatch" in legacy:
            return "BROKER_RECONCILE_MISMATCH"
        if "orphan" in legacy:
            return "ORPHAN_ORDER_DETECTED"
        if "spread" in legacy:
            return "SPREAD_TOO_WIDE"
        if "freshness" in legacy or "stale" in legacy:
            return "STALE_RUNTIME"
        if "validation" in str(gate or "").lower() or legacy.startswith("invalid_"):
            return "VALIDATION_FAILED"
        return _REJECTION_REASON_ALIASES.get(legacy, legacy.upper() if legacy else "UNKNOWN")

    def _gate_checks_snapshot(self) -> dict[str, Any]:
        return {
            "sleep_mode": bool(self._sleep_mode_active),
            "kill_switch": bool(self._kill_switch.status().active),
            "execution_gate_enabled": bool(self._gate_enabled),
            "automation_paused": bool(self._automation_paused),
            "reconciliation_status": str(self._reconciliation_status or "unknown"),
            "rollout_stage": str(self._rollout_stage or ""),
            "operator_present": bool(self._operator_present),
        }

    def _near_miss_checks(self) -> list[dict[str, Any]]:
        misses: list[dict[str, Any]] = []
        try:
            sc = dict(self._session_control.to_dict(runtime_flags={}) or {})
            mins = int(sc.get("minutes_to_session_close", -1) or -1)
            if 1 < mins <= 20:
                misses.append({"check": "session_close", "detail": f"{mins} minutes to session close"})
        except Exception:
            pass
        return misses

    def _append_execution_decision_journal(self, record: dict[str, Any]) -> None:
        payload = {
            "journal_schema": "atlas_execution_decision_journal_v1",
            "recorded_at": self._checked_at(),
            "source": "execution_gate",
            "trace": dict(record or {}),
        }
        try:
            self._decision_journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self._decision_journal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            self._audit.append("execution_decision_journal_write_failed", self._current_gate_request_id, {"reason": f"{exc.__class__.__name__}:{exc}"})

    def get_last_rejection_reason(self) -> dict[str, Any]:
        if self._last_rejection_reason:
            return dict(self._last_rejection_reason)
        last_gate = dict(self._last_gate_decision or {})
        if last_gate.get("allowed") is False:
            return {
                "allowed": False,
                "reason_code": last_gate.get("reason_code"),
                "detail": str(last_gate.get("reason", "") or ""),
                "checked_at": str(last_gate.get("checked_at", "") or ""),
                "gate": str(last_gate.get("blocked_by", "") or last_gate.get("gate", "") or "unknown"),
                "request_id": "",
                "legacy_reason": str(last_gate.get("reason", "") or ""),
                "legacy_code": str(last_gate.get("code", "") or ""),
                "context": dict(last_gate.get("detail", {}) or {}),
                "checks": dict(last_gate.get("checks", {}) or {}),
            }
        return {}

    def _gate_block(self, reason: str, *, gate: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
        reason_text = str(reason or "unknown")
        checked_at = self._checked_at()
        legacy_code = self._reason_code(reason_text)
        reason_code = self._structured_reason_code(reason_text, gate=gate)
        structured_result = {
            "allowed": False,
            "reason_code": reason_code,
            "detail": reason_text,
            "checked_at": checked_at,
            "gate": str(gate or "unknown"),
            "request_id": self._current_gate_request_id,
            "legacy_reason": reason_text,
            "legacy_code": legacy_code,
            "context": dict(detail or {}),
            "checks": self._gate_checks_snapshot(),
        }
        payload = {
            "ok": False,
            "reason": reason_text,
            "code": legacy_code,
            "allowed": False,
            "reason_code": reason_code,
            "checked_at": checked_at,
            "detail": dict(detail or {}),
            "structured_result": dict(structured_result),
        }
        self._last_rejection_reason = dict(structured_result)
        self._last_gate_decision = {
            "allowed": False,
            "blocked_by": str(gate or "unknown"),
            "reason": reason_text,
            "code": payload["code"],
            "reason_code": reason_code,
            "detail": dict(detail or {}),
            "checks": dict(structured_result["checks"]),
            "ts": time.time(),
            "checked_at": checked_at,
        }
        self._audit.append("execution_gate_structured_decision", self._current_gate_request_id, dict(structured_result))
        self._append_execution_decision_journal(structured_result)
        return payload

    def _gate_allow(self, *, gate: str = "preflight") -> dict[str, Any]:
        checked_at = self._checked_at()
        checks = self._gate_checks_snapshot()
        near_misses = self._near_miss_checks()
        structured_result = {
            "allowed": True,
            "reason_code": None,
            "detail": "approved",
            "checked_at": checked_at,
            "gate": str(gate or "preflight"),
            "request_id": self._current_gate_request_id,
            "checks": checks,
            "passed_checks": [key for key, value in checks.items() if value not in {False, "", "unknown"}],
            "near_miss_checks": near_misses,
        }
        self._last_gate_decision = {
            "allowed": True,
            "blocked_by": "",
            "reason": "",
            "code": "",
            "reason_code": None,
            "detail": {},
            "checks": checks,
            "ts": time.time(),
            "checked_at": checked_at,
            "gate": str(gate or "preflight"),
            "near_miss_checks": near_misses,
        }
        self._audit.append("execution_gate_structured_decision", self._current_gate_request_id, dict(structured_result))
        self._audit.append("execution_gate_approved", self._current_gate_request_id, {"passed_checks": structured_result["passed_checks"], "near_miss_checks": near_misses})
        self._append_execution_decision_journal(structured_result)
        return {"ok": True, "allowed": True, "reason_code": None, "checked_at": checked_at, "structured_result": dict(structured_result)}

    def _validate_required_runtime_flags(self, flags: dict[str, Any], *, auto_execution: bool) -> dict[str, Any]:
        if not isinstance(flags, dict):
            return {"ok": False, "reason": "invalid_runtime_flags"}
        if not auto_execution:
            return {"ok": True}
        # Freshness is strict for auto-execution; risk counters may fallback to internal bounded counters
        # when omitted, but malformed provided values are blocked.
        if "day_trade_count" in flags and self._safe_int(flags.get("day_trade_count")) is None:
            return {"ok": False, "reason": "invalid_day_trade_count"}
        if "daily_loss_pct" in flags and self._safe_float(flags.get("daily_loss_pct")) is None:
            return {"ok": False, "reason": "invalid_daily_loss_pct"}
        if "consecutive_losses" in flags and self._safe_int(flags.get("consecutive_losses")) is None:
            return {"ok": False, "reason": "invalid_consecutive_losses"}
        return {"ok": True}

    def _check_runtime_freshness(self, flags: dict[str, Any], *, strict: bool) -> dict[str, Any]:
        freshness = flags.get("freshness_runtime", None)
        if freshness is None:
            return {"ok": (not strict), "reason": ("missing_runtime_freshness" if strict else "")}
        if not isinstance(freshness, dict):
            return {"ok": False, "reason": "invalid_runtime_freshness"}
        stale_raw = freshness.get("is_stale", None)
        age_raw = freshness.get("status_age_ms", None)
        if stale_raw is None or age_raw is None:
            return {"ok": False, "reason": "missing_runtime_freshness"}
        age = self._safe_int(age_raw)
        if age is None:
            return {"ok": False, "reason": "invalid_runtime_freshness"}
        stale = self._safe_bool(stale_raw)
        max_age_raw = flags.get("max_freshness_age_ms", 15000)
        max_age = self._safe_int(max_age_raw)
        if max_age is None:
            return {"ok": False, "reason": "invalid_runtime_freshness"}
        if stale or age > int(max_age):
            return {"ok": False, "reason": "freshness_stale"}
        return {"ok": True}

    @classmethod
    def _is_live_allowed_for_stage(cls, stage: str) -> bool:
        return cls._is_live_stage(stage)

    @classmethod
    def _previous_stage(cls, stage: str) -> str:
        if stage not in cls._STAGE_ORDER:
            return "paper_only"
        idx = cls._STAGE_ORDER.index(stage)
        if idx <= 0:
            return cls._STAGE_ORDER[0]
        return cls._STAGE_ORDER[idx - 1]

    @classmethod
    def _stage_restrictions(cls, stage: str) -> dict[str, Any]:
        if stage == "paper_only":
            return {
                "live_allowed": False,
                "allowed_instruments": [],
                "allowed_setup_families": [],
                "max_size": 0.0,
                "daily_loss_cap_pct": 0.0,
                "approval_required": True,
            }
        if stage == "approved_paper_execution":
            return {
                "live_allowed": False,
                "allowed_instruments": [],
                "allowed_setup_families": [],
                "max_size": 0.0,
                "daily_loss_cap_pct": 0.0,
                "approval_required": True,
            }
        if stage == "limited_live_pilot":
            return {
                "live_allowed": True,
                "allowed_instruments": ["XAUUSD"],
                "allowed_setup_families": ["continuation"],
                "max_size": 0.02,
                "daily_loss_cap_pct": 1.0,
                "approval_required": True,
            }
        if stage == "restricted_live_operations":
            return {
                "live_allowed": True,
                "allowed_instruments": ["XAUUSD", "EURUSD"],
                "allowed_setup_families": ["continuation", "mean_reversion"],
                "max_size": 0.05,
                "daily_loss_cap_pct": 1.5,
                "approval_required": True,
            }
        # scaled_live_reviewed
        return {
            "live_allowed": True,
            "allowed_instruments": ["XAUUSD", "EURUSD", "GBPUSD"],
            "allowed_setup_families": ["continuation", "mean_reversion", "breakout"],
            "max_size": 0.10,
            "daily_loss_cap_pct": 2.0,
            "approval_required": True,
        }

    @staticmethod
    def _extract_field(payload: str, key: str, default: str = "") -> str:
        text = str(payload or "")
        match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(key))}=([^\s,;]+)", text)
        if match is None:
            return default
        token = str(match.group(1)).strip().strip(",;")
        return token or default

    @staticmethod
    def _extract_float(payload: str, key: str, default: float = 0.0) -> float:
        raw = ExecutionGate._extract_field(payload, key, default="")
        try:
            return float(raw)
        except Exception:
            return float(default)

    def _trim(self) -> None:
        max_items = 800
        if len(self._order) <= max_items:
            return
        keep = self._order[-max_items:]
        self._requests = {k: self._requests[k] for k in keep if k in self._requests}
        self._order = keep
