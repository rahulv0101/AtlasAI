from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class KillSwitchState:
    active: bool
    reason: str
    changed_at: float


class KillSwitch:
    def __init__(self) -> None:
        self._active = False
        self._reason = "not_set"
        self._changed_at = time.time()

    def activate(self, reason: str = "manual_activation") -> KillSwitchState:
        self._active = True
        self._reason = str(reason or "manual_activation")
        self._changed_at = time.time()
        return self.status()

    def deactivate(self, reason: str = "manual_deactivation") -> KillSwitchState:
        self._active = False
        self._reason = str(reason or "manual_deactivation")
        self._changed_at = time.time()
        return self.status()

    def status(self) -> KillSwitchState:
        return KillSwitchState(active=self._active, reason=self._reason, changed_at=self._changed_at)

    def restore(self, *, active: bool, reason: str, changed_at: float | None = None) -> KillSwitchState:
        self._active = bool(active)
        self._reason = str(reason or "restored")
        self._changed_at = float(changed_at or time.time())
        return self.status()
