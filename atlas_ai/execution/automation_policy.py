from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class AutomationPolicy:
    id: str
    enabled: bool
    environment: Literal["paper", "live", "both"]
    allowed_setup_families: list[str] = field(default_factory=list)
    allowed_instruments: list[str] = field(default_factory=list)
    allowed_sessions: list[str] = field(default_factory=list)
    size_cap: float = 0.0
    daily_loss_cap: float = 0.0
    max_trades_per_session: int = 0
    approval_required_for_activation: bool = True
    approved: bool = False
    active_rollout_stage: str = "paper_only"
    pause_conditions: dict[str, float] = field(default_factory=dict)
    escalation_conditions: dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

