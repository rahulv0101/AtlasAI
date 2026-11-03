from __future__ import annotations

import re
from typing import Any

from atlas_ai.commands.system import close_system, start_system


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower()).strip()


def route_command(text: str) -> dict[str, Any]:
    q = _norm(text)
    if q in {"start system", "atlas wake", "wake atlas", "atlas wake up", "wake up atlas"}:
        return start_system()
    if q in {"close system"}:
        return close_system()
    if q in {"atlas sleep", "sleep atlas", "go to sleep atlas"}:
        return close_system(keep_listener=True)
    return {"ok": False, "command": "", "reason": "not_a_system_command"}
