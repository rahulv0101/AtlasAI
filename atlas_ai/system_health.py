from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _availability_state(raw: Any) -> str:
    state = str(raw or "").strip().lower()
    if state in {"stale", "aging", "delayed"}:
        return "stale"
    if state in {"inferred", "derived", "estimated", "proxy"}:
        return "inferred"
    if state in {"", "unknown", "none", "n/a", "unavailable"}:
        return "unavailable"
    return "available"


def _normalize_macro_state(raw: Any) -> str:
    state = str(raw or "").strip().lower()
    if state in {"", "unknown", "none", "n/a"}:
        return "unavailable"
    return state


def _macro_slot(raw_state: Any, *, inferred_fallback: str = "unavailable") -> dict[str, str]:
    avail_raw = _availability_state(raw_state)
    if avail_raw == "unavailable":
        return {"state": inferred_fallback, "availability": "unavailable", "source": "price_only_inferred"}
    return {
        "state": _normalize_macro_state(raw_state),
        "availability": avail_raw,
        "source": "macro_feed",
    }


def _parse_dt(value: str, fallback_tz=None) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None and fallback_tz is not None:
        parsed = parsed.replace(tzinfo=fallback_tz)
    return parsed


def _age_seconds(updated_at: str, recent_bar: str) -> float:
    try:
        now = _parse_dt(updated_at)
        recent = _parse_dt(recent_bar, fallback_tz=now.tzinfo if now else None)
        if now is None or recent is None:
            return 0.0
        return max(0.0, (now - recent).total_seconds())
    except Exception:
        return 0.0


def build_freshness_runtime(
    status: dict[str, Any],
    settings: Any,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_dt = now or datetime.now().astimezone()
    updated_at = str(status.get("updated_at", "") or "")
    market = dict(status.get("market", {}) or {})
    recent_times = list(market.get("recent_time", []) or [])
    recent_bar = str(recent_times[-1]) if recent_times else ""
    updated_dt = _parse_dt(updated_at, fallback_tz=now_dt.tzinfo)
    recent_dt = _parse_dt(recent_bar, fallback_tz=(updated_dt.tzinfo if updated_dt else now_dt.tzinfo))
    status_age_ms = None
    if updated_dt is not None:
        status_age_ms = round(max(0.0, (now_dt - updated_dt).total_seconds()) * 1000.0, 2)
    market_data_age_ms = None
    if updated_dt is not None and recent_dt is not None:
        market_data_age_ms = round(max(0.0, (updated_dt - recent_dt).total_seconds()) * 1000.0, 2)
    stale_threshold_ms = max(180000.0, float(getattr(settings, "timeframe_minutes", 1) or 1) * 8.0 * 60.0 * 1000.0)
    aging_threshold_ms = round(stale_threshold_ms * 0.5, 2)
    primary_age_ms = market_data_age_ms if market_data_age_ms is not None else status_age_ms
    if primary_age_ms is None:
        state = "unknown"
    elif primary_age_ms > stale_threshold_ms:
        state = "stale"
    elif primary_age_ms > aging_threshold_ms:
        state = "aging"
    else:
        state = "fresh"
    return {
        "status_age_ms": status_age_ms,
        "market_data_age_ms": market_data_age_ms,
        "primary_age_ms": primary_age_ms,
        "backend_cycle_age_ms": dict(status.get("freshness_timing", {}) or {}).get("backend_cycle_age_ms"),
        "is_status_stale": bool(status_age_ms is not None and float(status_age_ms) > stale_threshold_ms),
        "is_stale": state == "stale",
        "state": state,
        "max_freshness_age_ms": stale_threshold_ms,
        "stale_threshold_ms": stale_threshold_ms,
        "aging_threshold_ms": aging_threshold_ms,
        "source": "status_freshness_contract",
    }


def build_system_health(
    status: dict[str, Any],
    settings: Any,
    runtime_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_flags = dict(runtime_flags or {})
    health = dict(status.get("health", {}) or {})
    broker = dict(health.get("broker", {}) or {})
    llm = dict(health.get("llm", {}) or {})
    overlay_state = dict(status.get("overlay_state", {}) or {})
    conservative = dict(status.get("conservative_pretrade_filter", {}) or {})
    freshness_runtime = build_freshness_runtime(status, settings)
    stale_data = bool(freshness_runtime.get("is_stale", False))
    now_utc = datetime.now(timezone.utc)

    effective_mode = "live" if bool(status.get("execute_trades", False)) else "paper"
    runtime_execute = bool(runtime_flags.get("execute_trades", getattr(settings, "execute_trades", False)))
    runtime_gemini = bool(runtime_flags.get("gemini_influence_enabled", getattr(settings, "gemini_influence_enabled", False)))
    config_mismatch = runtime_execute != bool(getattr(settings, "execute_trades", False))
    startup_verification = dict(runtime_flags.get("startup_verification", {}) or {})
    self_improvement_state = dict(status.get("self_improvement_state", {}) or {})
    safety_layer = dict(status.get("safety_layer", {}) or {})
    startup_execute_raw = startup_verification.get("execute_trades", None)
    startup_execute = bool(startup_execute_raw) if startup_execute_raw is not None else None
    status_execute = bool(status.get("execute_trades", False))
    if startup_execute is not None and startup_execute != status_execute:
        config_mismatch = True
    routing_risk = bool(status.get("execute_trades", False)) and not bool(broker.get("connected", False))
    startup_profile = str(startup_verification.get("profile", "") or "").strip().lower()
    status_mode_label = str(status.get("mode_label", "") or "").strip()
    if status_mode_label:
        effective_mode_label = status_mode_label
    elif startup_profile == "guarded_demo":
        effective_mode_label = "GUARDED_DEMO"
    else:
        effective_mode_label = "LIVE" if effective_mode == "live" else "PAPER"

    if not config_mismatch:
        mode_explanation = "Status, runtime, startup verification, and settings are aligned."
    elif startup_execute is not None and runtime_execute == startup_execute and status_execute == startup_execute:
        mode_explanation = (
            "Startup verification, runtime flags, and live status all agree, but the persisted settings/env value "
            "for EXECUTE_TRADES does not match. Update the stored config to remove the warning."
        )
    elif startup_execute is not None and runtime_execute != startup_execute:
        mode_explanation = (
            "Runtime flags disagree with startup verification. Atlas may have been started with one execution mode "
            "and later inherited stale runtime state."
        )
    elif startup_execute is not None and status_execute != startup_execute:
        mode_explanation = (
            "Trader status disagrees with startup verification. The running loop is reporting a different execution "
            "mode than Atlas verified at startup."
        )
    else:
        mode_explanation = (
            "Mode mismatch detected across settings, runtime flags, or trader status. Inspect execute_trades values "
            "for env/settings, runtime flags, and the latest status payload."
        )

    # Canonical AI influence safety mode.
    ai_influence_mode = "observer_only"
    if bool(runtime_flags.get("gemini_influence_enabled", False)):
        ai_influence_mode = "proposal_only"
    ai_authority = "execution_authority_forbidden"

    warnings: list[str] = []
    protective_locks: list[str] = []
    if stale_data:
        warnings.append("stale_data_detected")
    if config_mismatch:
        warnings.append("config_mismatch_detected")
    if routing_risk:
        warnings.append("routing_risk_detected")
    if runtime_gemini and ai_influence_mode not in {"observer_only", "proposal_only"}:
        warnings.append("gemini_influence_runtime_enabled")
    provider_warnings: list[str] = []
    for provider in list(llm.get("providers", []) or []):
        if not isinstance(provider, dict):
            continue
        name = str(provider.get("name", "provider") or "provider").strip().lower()
        kind = str(provider.get("last_error_kind", "") or "").strip().lower()
        in_cooldown = bool(provider.get("in_cooldown", False))
        if in_cooldown and kind in {"quota_exhausted", "rate_limited"}:
            provider_warnings.append(f"llm_{name}_{kind}")
    warnings.extend(provider_warnings)
    if bool(safety_layer.get("safety_lock_active", False)):
        protective_locks.append("safety_lock_active")
    if bool(safety_layer.get("cooldown_active", False)):
        protective_locks.append("adaptive_cooldown_active")
    if str(safety_layer.get("volatility_state", "") or "") == "extreme":
        protective_locks.append("extreme_volatility_protection_active")

    # Safety lock source/metadata for stale historical vs current-session clarity.
    safety_lock_reason = str(safety_layer.get("safety_lock_reason", "") or "")
    safety_lock_started_at = str(safety_layer.get("safety_lock_started_at", "") or "")
    safety_lock_source = "manual" if "manual" in safety_lock_reason else (
        "expectancy" if "expectancy" in safety_lock_reason or "win_rate" in safety_lock_reason else "historical"
    )
    if any(x in safety_lock_reason for x in ("drawdown", "consecutive_loss", "cooldown")):
        safety_lock_source = "current_session"
    if "drift" in safety_lock_reason:
        safety_lock_source = "drift"
    loss_streak = int(_safe_float(safety_layer.get("loss_streak", 0)))
    current_session_relevant = bool(loss_streak > 0 or _safe_float(safety_layer.get("rolling_drawdown_pct", 0.0)) > 0.0)
    cooldown_level = str(safety_layer.get("cooldown_level", "none") or "none")
    cooldown_active = bool(safety_layer.get("cooldown_active", False))
    cooldown_status = (
        "not_applicable_due_to_safety_lock"
        if bool(safety_layer.get("safety_lock_active", False)) and not cooldown_active
        else ("active" if cooldown_active else "inactive")
    )

    # Recovery path UX object.
    recovery_ready = bool(safety_layer.get("recovery_ready", False))
    recovery_blockers = list(safety_layer.get("recovery_blockers", []) or [])
    required_actions = list(safety_layer.get("recovery_requirements", []) or [])
    if not required_actions and not recovery_ready:
        required_actions = [
            "manual_operator_unlock_required",
            "confirm_no_active_hard_blocker",
            "wait_for_acceptable_volatility",
            "next_setup_must_meet_high_confluence",
        ]

    # Macro/session availability and regime inference clarity.
    context = dict(status.get("context_snapshot", {}) or {})
    macro_data = {
        "dxy_alignment": _macro_slot(context.get("dxy_alignment", "")),
        "us10y_alignment": _macro_slot(context.get("us10y_alignment", "")),
        "vix_regime": _macro_slot(context.get("vix_regime", context.get("vix_state", ""))),
        "oil_context": _macro_slot(context.get("oil_backdrop", context.get("oil_regime", ""))),
    }
    session_state = str(context.get("session", "") or "").strip().lower()
    if session_state in {"", "unknown"}:
        h = now_utc.hour
        if 13 <= h < 17:
            session_state = "overlap"
        elif 7 <= h < 12:
            session_state = "london"
        elif 12 <= h < 20:
            session_state = "new_york"
        elif 0 <= h < 7:
            session_state = "asia"
        else:
            session_state = "off_hours"
    regime = dict(status.get("regime", {}) or {})
    regime_name = str(regime.get("name", "") or "")
    regime_conf = int(_safe_float(regime.get("confidence", regime.get("regime_confidence", 0))))
    regime_source = "model"
    regime_warning = ""
    if not regime_name or regime_name == "unknown":
        regime_name = "range"
        regime_conf = max(0, min(100, regime_conf or 28))
        regime_source = "price_only_inferred"
        regime_warning = "low_confidence_regime"

    # Spread clarity surface.
    spread_points = _safe_float(context.get("spread_points", status.get("spread_points", 0.0)))
    spread_pips = round(spread_points / 10.0, 2) if spread_points > 0 else 0.0
    spread_status = "normal"
    if spread_points >= _safe_float(runtime_flags.get("max_spread_points", 80.0)):
        spread_status = "blocked"
    elif spread_points >= 60:
        spread_status = "high"
    elif spread_points >= 40:
        spread_status = "elevated"
    liquidity_state = str(context.get("liquidity_state", "") or "unavailable")
    spread_liquidity_score = int(_safe_float(context.get("spread_liquidity_score", 10 if liquidity_state in {"unknown", "unavailable", "poor"} else 70)))
    spread_explanation = (
        "weak_due_to_missing_liquidity_context"
        if spread_status == "normal" and spread_liquidity_score <= 20
        else "score_consistent_with_spread_and_liquidity"
    )

    # Evolution and optional filter status normalization.
    learning_snapshot = dict(status.get("learning_status_snapshot", {}) or {})
    evolved_pool_count = int(_safe_float(learning_snapshot.get("evolved_pool_count", 0)))
    evolution_enabled = bool(getattr(settings, "evolution_enabled", False))
    if evolution_enabled and evolved_pool_count <= 0:
        evolution_status = "fallback_builtin"
        evolution_action = "seed_population_recommended"
    elif not evolution_enabled:
        evolution_status = "disabled"
        evolution_action = "none"
    else:
        evolution_status = "active"
        evolution_action = "none"

    conservative_enabled = bool(conservative.get("enabled", False))
    research_enabled = bool(getattr(settings, "research_regime_filter_enabled", False))
    filter_status = {
        "conservative_filter": {
            "status": "active" if conservative_enabled else "inactive_by_config",
            "severity": "INFO" if not conservative_enabled else "PASS",
        },
        "research_filter": {
            "status": "active" if research_enabled else "inactive_by_config",
            "severity": "INFO" if not research_enabled else "PASS",
        },
    }

    classification = {
        "PASS": [
            "broker_connectivity_ok" if bool(broker.get("connected", False)) else "",
            "execution_gate_enforced",
            "status_pipeline_alive",
        ],
        "PROTECTIVE_LOCK": protective_locks,
        "WARNING": warnings,
        "FAIL": [],
    }
    if routing_risk:
        classification["FAIL"].append("routing_risk_detected")
    if config_mismatch:
        classification["FAIL"].append("config_mismatch_detected")
    classification["PASS"] = [x for x in classification["PASS"] if str(x).strip()]

    return {
        "effective_mode": effective_mode,
        "effective_mode_label": effective_mode_label,
        "paper_mode_confirmed": effective_mode == "paper",
        "filter_state": {
            "conservative_filter_enabled": bool(conservative.get("enabled", False)),
            "research_regime_filter_enabled": bool(getattr(settings, "research_regime_filter_enabled", False)),
        },
        "ml_state": {
            "meta_model_enabled": bool(getattr(settings, "meta_model_enabled", True)),
            "decision_authority": "disabled" if not bool(getattr(settings, "meta_model_enabled", True)) else "observer_only",
        },
        "gemini_state": {
            "enabled": bool(status.get("gemini_enabled", False)),
            "influence_enabled": bool(status.get("gemini_influence_enabled", False)),
            "provider_health": llm,
        },
        "ai_influence_mode": ai_influence_mode,
        "ai_execution_authority": ai_authority,
        "stale_data_detected": stale_data,
        "freshness_runtime": freshness_runtime,
        "config_mismatch_detected": config_mismatch,
        "mode_truth": {
            "effective_mode": effective_mode,
            "effective_mode_label": effective_mode_label,
            "effective_mode_source": "status",
            "runtime_execute_trades": runtime_execute,
            "settings_execute_trades": bool(getattr(settings, "execute_trades", False)),
            "status_execute_trades": status_execute,
            "startup_execute_trades": startup_execute,
            "startup_message": str(startup_verification.get("startup_message", "") or ""),
            "explanation": mode_explanation,
        },
        "routing_risk_detected": routing_risk,
        "self_improvement": {
            "policy_version": str(self_improvement_state.get("policy_version", "") or ""),
            "proposal_count": int(self_improvement_state.get("proposal_count", 0) or 0),
            "validated_count": int(self_improvement_state.get("validated_count", 0) or 0),
            "approval_pending_count": int(self_improvement_state.get("approval_pending_count", 0) or 0),
            "rollout_active": bool(self_improvement_state.get("rollout_active", False)),
            "rollback_available": bool(self_improvement_state.get("rollback_available", False)),
            "active_rule_version": str(self_improvement_state.get("active_rule_version", "") or ""),
            "latest_proposal_id": str(self_improvement_state.get("latest_proposal_id", "") or ""),
            "pending_proposal_id": str(self_improvement_state.get("pending_proposal_id", "") or ""),
            "pending_proposal_category": str(self_improvement_state.get("pending_proposal_category", "") or ""),
            "pending_proposal_target_blocker": str(self_improvement_state.get("pending_proposal_target_blocker", "") or ""),
            "latest_rollout_id": str(self_improvement_state.get("latest_rollout_id", "") or ""),
        },
        "safety_layer": {
            "safety_lock_active": bool(safety_layer.get("safety_lock_active", False)),
            "safety_lock_reason": safety_lock_reason,
            "cooldown_active": bool(safety_layer.get("cooldown_active", False)),
            "cooldown_level": cooldown_level,
            "cooldown_reason": str(safety_layer.get("cooldown_reason", "") or ""),
            "cooldown_status": cooldown_status,
            "safety_lock_supersedes_cooldown": bool(safety_layer.get("safety_lock_active", False)) and not cooldown_active,
            "volatility_state": str(safety_layer.get("volatility_state", "normal") or "normal"),
            "volatility_action": str(safety_layer.get("volatility_action", "allow") or "allow"),
            "recovery_ready": recovery_ready,
            "recovery_mode": str(safety_layer.get("recovery_mode", "normal") or "normal"),
            "recovery_blockers": recovery_blockers,
            "safety_lock_source": safety_lock_source,
            "safety_lock_triggered_at": safety_lock_started_at,
            "safety_lock_triggering_window": "recent_outcomes_12" if safety_lock_source in {"historical", "expectancy"} else "current_session",
            "safety_lock_current_session_relevant": current_session_relevant,
            "safety_lock_requires_manual_unlock": True,
            "last_loss_at": str(safety_layer.get("last_loss_at", "") or ""),
            "trades_since_lock": int(_safe_float(safety_layer.get("trades_since_lock", 0))),
            "recovery": {
                "recovery_ready": recovery_ready,
                "recovery_blockers": recovery_blockers,
                "required_actions": required_actions,
                "can_auto_recover": False,
                "manual_unlock_endpoint": "/api/safety/unlock",
                "unlock_requires_confirmation": True,
            },
        },
        "broker_connectivity_status": {
            "connected": bool(broker.get("connected", False)),
            "symbol_selected": bool(broker.get("symbol_selected", False)),
        },
        "warnings": warnings,
        "protective_locks": protective_locks,
        "classification": classification,
        "macro_data": macro_data,
        "session_state": session_state,
        "regime_state": {
            "regime": regime_name,
            "regime_confidence": regime_conf,
            "regime_source": regime_source,
            "warning": regime_warning,
        },
        "spread_liquidity": {
            "spread_points": spread_points,
            "spread_pips": spread_pips,
            "spread_status": spread_status,
            "spread_liquidity_score": spread_liquidity_score,
            "spread_explanation": spread_explanation,
        },
        "evolution_state": {
            "evolution_enabled": evolution_enabled,
            "evolution_status": evolution_status,
            "evolved_strategy_count": evolved_pool_count,
            "action": evolution_action,
        },
        "filter_status": filter_status,
    }
