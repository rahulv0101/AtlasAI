from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModeResolution:
    env_mode: bool
    runtime_mode: bool | None
    cli_override: str
    resolved_mode: bool
    source: str
    reason: str
    mismatch_detected: bool
    mismatch_reasons: list[str]
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "env_mode": "live" if self.env_mode else "paper",
            "runtime_mode": "live" if self.runtime_mode else ("paper" if self.runtime_mode is False else "unset"),
            "cli_override": self.cli_override,
            "resolved_mode": "live" if self.resolved_mode else "paper",
            "source": self.source,
            "reason": self.reason,
            "mismatch_detected": self.mismatch_detected,
            "mismatch_reasons": list(self.mismatch_reasons),
            "error": self.error,
        }


def _mode_label(enabled: bool | None) -> str:
    if enabled is None:
        return "unset"
    return "live" if bool(enabled) else "paper"


def resolve_mode(
    *,
    env_execute_trades: bool,
    runtime_flags: dict[str, Any],
    mode_override: str = "",
) -> ModeResolution:
    """Resolve effective execution mode without raising exceptions.

    Returns a structured ModeResolution that callers can use to either
    proceed safely or surface an error to the UI / logs.
    """
    flags = dict(runtime_flags or {})
    env_mode = bool(env_execute_trades)
    runtime_mode = bool(flags["execute_trades"]) if "execute_trades" in flags else None
    override = str(mode_override or "").strip().lower()

    mismatch_reasons: list[str] = []

    if override:
        if override == "env":
            resolved = env_mode
            source = "explicit_env_override"
            reason = "CLI --mode-override env"
        elif override == "paper":
            resolved = False
            source = "explicit_paper_override"
            reason = "CLI --mode-override paper"
        elif override == "live":
            resolved = True
            source = "explicit_live_override"
            reason = "CLI --mode-override live"
        else:
            return ModeResolution(
                env_mode=env_mode,
                runtime_mode=runtime_mode,
                cli_override=override,
                resolved_mode=env_mode,
                source="error",
                reason="invalid_override",
                mismatch_detected=True,
                mismatch_reasons=[f"unsupported_override:{override}"],
                error=f"Unsupported mode override: {mode_override}. Use env|paper|live.",
            )
        return ModeResolution(
            env_mode=env_mode,
            runtime_mode=runtime_mode,
            cli_override=override,
            resolved_mode=resolved,
            source=source,
            reason=reason,
            mismatch_detected=False,
            mismatch_reasons=[],
        )

    if runtime_mode is None:
        return ModeResolution(
            env_mode=env_mode,
            runtime_mode=None,
            cli_override="",
            resolved_mode=env_mode,
            source="runtime_missing",
            reason="runtime_missing_initialized_from_env",
            mismatch_detected=False,
            mismatch_reasons=[],
        )

    if runtime_mode == env_mode:
        return ModeResolution(
            env_mode=env_mode,
            runtime_mode=runtime_mode,
            cli_override="",
            resolved_mode=env_mode,
            source="env_runtime_match",
            reason="env_runtime_match",
            mismatch_detected=False,
            mismatch_reasons=[],
        )

    # Mismatch detected — block and surface structured error
    mismatch_reasons.append(f"env={_mode_label(env_mode)} vs runtime={_mode_label(runtime_mode)}")
    return ModeResolution(
        env_mode=env_mode,
        runtime_mode=runtime_mode,
        cli_override="",
        resolved_mode=runtime_mode,
        source="blocked",
        reason="env_runtime_mismatch_blocked",
        mismatch_detected=True,
        mismatch_reasons=mismatch_reasons,
        error=(
            "Startup blocked: EXECUTE_TRADES in .env does not match runtime_flags.json. "
            "Pass --mode-override env|paper|live to make the launch intent explicit."
        ),
    )

