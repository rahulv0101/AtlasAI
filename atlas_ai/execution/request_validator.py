from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ALLOWED_ACTION_TYPES = {
    "place_trade",
    "cancel_trade",
    "close_position",
    "sync_position",
}


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str


def validate_execution_request(
    *,
    action_type: str,
    environment: str,
    payload_summary: str,
    runtime_flags: dict[str, Any] | None = None,
) -> ValidationResult:
    action = str(action_type or "").strip().lower()
    env = str(environment or "").strip().lower()
    payload = str(payload_summary or "").strip()
    flags = dict(runtime_flags or {})

    if env not in {"paper", "live", "demo", "live_real"}:
        return ValidationResult(False, "invalid_environment")
    if action not in ALLOWED_ACTION_TYPES:
        return ValidationResult(False, "unsupported_action_type")
    if not payload:
        return ValidationResult(False, "empty_payload")

    read_only = bool(flags.get("read_only_ui", False))
    if read_only:
        return ValidationResult(False, "read_only_environment")

    return ValidationResult(True, "validated")
