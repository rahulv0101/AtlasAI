from __future__ import annotations

import re
from typing import Any


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    return cleaned.strip("_")


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _candidate_id(status: dict[str, Any]) -> str:
    chosen = dict(status.get("chosen", {}) or {})
    timestamp = str(status.get("updated_at", ""))
    strategy = str(chosen.get("strategy", "none"))
    side = str(chosen.get("side", "flat"))
    return f"{timestamp}|{strategy}|{side}"


def _active_guards(status: dict[str, Any]) -> list[dict[str, Any]]:
    guard_keys = [
        "risk_guard",
        "directional_structure_guard",
        "loss_learning",
        "adaptation",
        "side_switch_training",
        "ema_fresh_cross_guard",
        "side_balance",
        "execution_cluster",
        "smart_context",
        "m1_quality",
        "research_regime_overlay",
        "conservative_pretrade_filter",
        "setup_quality_filter",
        "win_memory",
        "meta_model",
    ]
    guards: list[dict[str, Any]] = []
    message = str(status.get("message", "") or "")
    msg = message.lower()
    if str(status.get("action", "") or "").lower() == "blocked" and "risk guard active" in msg:
        guards.append(
            {
                "name": "risk_guard",
                "enabled": True,
                "applied": True,
                "hard_blocked": True,
                "reason": message,
            }
        )
    elif str(status.get("action", "") or "").lower() == "blocked" and "directional-structure guard" in msg:
        guards.append(
            {
                "name": "directional_structure_guard",
                "enabled": True,
                "applied": True,
                "hard_blocked": True,
                "reason": message,
            }
        )
    for key in guard_keys:
        if any(str(existing.get("name", "")) == key for existing in guards):
            continue
        payload = dict(status.get(key, {}) or {})
        if not payload:
            continue
        enabled = bool(payload.get("enabled", True))
        applied = bool(payload.get("applied", False) or payload.get("hard_blocked", False))
        if not enabled and not applied and "reason" not in payload:
            continue
        guards.append(
            {
                "name": key,
                "enabled": enabled,
                "applied": applied,
                "hard_blocked": bool(payload.get("hard_blocked", False)),
                "reason": str(payload.get("reason", "")),
            }
        )
    return guards


def _bias_breakdown(status: dict[str, Any], context_snapshot: dict[str, Any]) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    side = str(chosen.get("side", "flat") or "flat").lower()
    signal_bias = side if side in {"buy", "sell"} else "neutral"
    structure = str(
        dict(status.get("chart_features", {}) or {}).get(
            "trend_structure",
            context_snapshot.get("trend_structure", ""),
        )
        or ""
    ).upper()
    trend_state = str(
        dict(status.get("chart_features", {}) or {}).get(
            "trend_state",
            context_snapshot.get("trend_state", ""),
        )
        or ""
    ).lower()
    if structure == "HH/HL" or trend_state == "uptrend":
        chart_structure_bias = "buy"
    elif structure == "LH/LL" or trend_state == "downtrend":
        chart_structure_bias = "sell"
    else:
        chart_structure_bias = "neutral"
    htf = dict(status.get("higher_timeframe", {}) or {})
    higher_timeframe_bias = str(status.get("higher_tf_bias", htf.get("bias", "unknown")) or "unknown").lower()
    action = str(status.get("action", "") or "").lower()
    if action in {"buy", "guarded_buy", "execute_buy"}:
        final_trade_bias = "buy"
    elif action in {"sell", "guarded_sell", "execute_sell"}:
        final_trade_bias = "sell"
    else:
        final_trade_bias = "neutral"
    conflict = bool(
        signal_bias in {"buy", "sell"}
        and chart_structure_bias in {"buy", "sell"}
        and signal_bias != chart_structure_bias
    )
    return {
        "signal_bias": signal_bias,
        "chart_structure_bias": chart_structure_bias,
        "higher_timeframe_bias": higher_timeframe_bias,
        "final_trade_bias": final_trade_bias,
        "conflict": conflict,
        "explanation": (
            "Signal direction conflicts with chart structure; final trade bias remains neutral while blocked."
            if conflict and final_trade_bias == "neutral"
            else "Signal, chart structure, higher-timeframe, and final trade bias are tracked separately."
        ),
    }


def build_evidence_separation(status: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    """Expose decision lanes separately so answers do not collapse bias into execution permission."""
    runtime = dict(runtime or {})
    context_snapshot = dict(status.get("context_snapshot", {}) or {})
    chosen = dict(status.get("chosen", {}) or {})
    blocker = dict(status.get("blocker_classification", {}) or {})
    decision_analysis = dict(status.get("decision_analysis", {}) or {})
    setup_radar = dict(decision_analysis.get("setup_radar", {}) or {})
    top_watch = dict(setup_radar.get("top_watch", {}) or {})
    execution_state = dict(runtime.get("execution", {}) or {})
    action = str(status.get("action", "") or "").lower()
    confidence = _safe_float(chosen.get("confidence", 0.0))
    threshold = _safe_float(status.get("min_confidence_threshold", 0.0))
    side = str(chosen.get("side", "flat") or "flat").lower()
    trace_bias = dict(dict(status.get("decision_trace", {}) or {}).get("bias_breakdown", {}) or {})
    bias = trace_bias or _bias_breakdown(status, context_snapshot)
    gate_allowed = bool(execution_state.get("auto_trade_allowed", False) or execution_state.get("automation_live_allowed", False))
    has_direction = side in {"buy", "sell"}
    confidence_ready = bool(has_direction and (threshold <= 0.0 or confidence >= threshold))
    setup_state = str(
        top_watch.get("trigger_state")
        or top_watch.get("readiness")
        or context_snapshot.get("setup_type_hint")
        or status.get("setup_family")
        or "unknown"
    )
    blocker_text = str(blocker.get("primary") or status.get("primary_blocker") or status.get("message") or "")
    return {
        "market_bias": str(bias.get("chart_structure_bias", "neutral") or "neutral"),
        "signal_direction": side if has_direction else "neutral",
        "higher_timeframe_bias": str(bias.get("higher_timeframe_bias", "unknown") or "unknown"),
        "final_trade_bias": str(bias.get("final_trade_bias", "neutral") or "neutral"),
        "setup_readiness": setup_state,
        "confidence": round(confidence, 4),
        "threshold": round(threshold, 4),
        "confidence_ready": confidence_ready,
        "blocker_class": str(blocker.get("class") or status.get("blocker_class") or "unknown"),
        "primary_blocker": blocker_text,
        "execution_permission": "allowed" if action in {"paper", "execute", "buy", "sell"} or gate_allowed else "not_allowed",
        "execution_reason": str(execution_state.get("auto_trade_reason") or blocker_text or ""),
        "policy_note": "Evidence lanes are explanatory only; they do not change strategy, risk, or execution policy.",
    }


def _penalties(status: dict[str, Any]) -> list[dict[str, Any]]:
    penalties: list[dict[str, Any]] = []
    loss_learning = dict(status.get("loss_learning", {}) or {})
    if loss_learning:
        penalties.append(
            {
                "source": "loss_learning",
                "value": round(_safe_float(loss_learning.get("penalty", 0.0)), 4),
                "reason": str(loss_learning.get("reason", "")),
            }
        )
    adaptation = dict(status.get("adaptation", {}) or {})
    if adaptation:
        penalties.append(
            {
                "source": "adaptation",
                "value": round(1.0 - _safe_float(adaptation.get("confidence_multiplier", 1.0)), 4),
                "reason": str(adaptation.get("reason", "")),
            }
        )
    side_balance = dict(status.get("side_balance", {}) or {})
    if side_balance and bool(side_balance.get("applied", False)):
        penalties.append(
            {
                "source": "side_balance",
                "value": round(_safe_float(side_balance.get("penalty", 0.0)), 4),
                "reason": str(side_balance.get("reason", "")),
            }
        )
    return penalties


def _compact_signal_rows(rows: Any, limit: int = 8) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in list(rows or [])[: max(1, int(limit))]:
        row = dict(item or {})
        out.append(
            {
                "strategy": str(row.get("strategy", row.get("name", "")) or ""),
                "side": str(row.get("side", "flat") or "flat"),
                "confidence": round(_safe_float(row.get("confidence", row.get("score", 0.0))), 4),
                "score": round(_safe_float(row.get("score", row.get("confidence", 0.0))), 4),
            }
        )
    return out


def _decision_blocker_class(status: dict[str, Any], context_snapshot: dict[str, Any]) -> dict[str, Any]:
    message = str(status.get("message", "") or "").strip()
    msg = message.lower()
    chosen = dict(status.get("chosen", {}) or {})
    side = str(chosen.get("side", "flat") or "flat").lower()
    confidence = _safe_float(chosen.get("confidence", 0.0))
    threshold = _safe_float(status.get("min_confidence_threshold", 0.0))
    risk = dict(status.get("risk", {}) or {})
    reason_codes = [
        str(x)
        for x in list(dict(status.get("decision_explanation", {}) or {}).get("reason_codes", []) or [])
        if str(x).strip()
    ]
    hard_blockers = [g for g in _guard_trace(status) if bool(g.get("hard_blocked", False))]

    blocker_class = "unknown"
    primary = message or "unknown"
    recoverability = "inspect"

    if any(term in msg for term in ("heartbeat", "failed to fetch mt5", "ipc send failed", "broker startup failed")):
        blocker_class = "runtime_or_data"
        recoverability = "fix_runtime"
    elif "online learning reduced confidence" in msg:
        blocker_class = "quality_gate"
        recoverability = "wait_for_cleaner_setup_or_review_learning_evidence"
    elif "adaptive guard" in msg or "adaptive_guard" in reason_codes:
        blocker_class = "safety_gate"
        recoverability = "wait_or_review_recent_loss_adaptation"
    elif any(term in msg for term in ("risk guard active", "drawdown", "safety lock", "cooldown", "max open trades", "spread too high")):
        blocker_class = "safety_gate"
        recoverability = "wait_or_operator_review"
    elif "no directional trade signal" in msg:
        blocker_class = "signal_quality"
        primary = "No directional trade signal selected."
        recoverability = "wait_for_buy_or_sell_signal_confirmation"
    elif side == "flat" or "no high-confidence trade" in msg or "flat_signal" in reason_codes:
        blocker_class = "signal_quality"
        primary = "flat_or_low_confidence_signal" if not message else message
        recoverability = "wait_for_setup_or_review_signal_fit"
    elif any(
        term in msg
        for term in (
            "ema guard",
            "consensus guard",
            "loss-learning guard",
            "side-balance",
            "setup-quality filter",
            "directional-structure guard",
            "higher-timeframe alignment required",
            "filter blocked",
        )
    ):
        blocker_class = "quality_gate"
        recoverability = "review_gate_or_wait_for_cleaner_setup"
    elif hard_blockers:
        blocker_class = "safety_gate"
        primary = str(hard_blockers[0].get("reason") or hard_blockers[0].get("name") or primary)
        recoverability = "wait_or_operator_review"
    elif any(term in msg for term in ("stale",)):
        blocker_class = "runtime_or_data"
        recoverability = "fix_runtime"
    elif threshold and confidence < threshold:
        blocker_class = "signal_quality"
        primary = "confidence_below_threshold"
        recoverability = "wait_for_stronger_signal"
    elif str(status.get("action", "")).lower() in {"paper", "execute", "execution"}:
        blocker_class = "not_blocked"
        primary = "execution_allowed"
        recoverability = "none"

    return {
        "class": blocker_class,
        "primary": primary,
        "recoverability": recoverability,
        "signal": {
            "side": side,
            "confidence": round(confidence, 4),
            "threshold": round(threshold, 4),
            "confidence_gap": round(confidence - threshold, 4) if threshold else 0.0,
        },
        "risk": {
            "daily_loss_pct": risk.get("daily_loss_pct"),
            "weekly_loss_pct": risk.get("weekly_loss_pct"),
            "open_positions": risk.get("open_positions"),
            "spread_points": risk.get("spread_points"),
        },
        "context": {
            "session": str(context_snapshot.get("session", "unknown") or "unknown"),
            "regime": str(context_snapshot.get("regime", "unknown") or "unknown"),
            "volatility_state": str(context_snapshot.get("volatility_state", "unknown") or "unknown"),
        },
    }


def _guard_trace(status: dict[str, Any]) -> list[dict[str, Any]]:
    guards = _active_guards(status)
    trace: list[dict[str, Any]] = []
    for guard in guards:
        trace.append(
            {
                "name": str(guard.get("name", "")),
                "enabled": bool(guard.get("enabled", False)),
                "applied": bool(guard.get("applied", False)),
                "hard_blocked": bool(guard.get("hard_blocked", False)),
                "reason": str(guard.get("reason", "")),
            }
        )
    return trace


def build_decision_trace(
    status: dict[str, Any],
    context_snapshot: dict[str, Any],
    runtime_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a read-only trace of the trading decision pipeline.

    This records the current status fields after the fact. It must not compute
    new trading decisions, mutate thresholds, or create execution authority.
    """
    runtime_flags = dict(runtime_flags or {})
    chosen = dict(status.get("chosen", {}) or {})
    ranking = dict(status.get("contextual_ranking", {}) or {})
    risk = dict(status.get("risk", {}) or {})
    execution = dict(status.get("execution", {}) or {})
    decision_explanation = dict(status.get("decision_explanation", {}) or {})
    action = str(status.get("action", "unknown") or "unknown").lower()
    hard_blockers = [g for g in _guard_trace(status) if bool(g.get("hard_blocked", False))]
    blocker = _decision_blocker_class(status, context_snapshot)
    final_action = action or "unknown"
    if action in {"paper", "execute", "failed"}:
        final_decision = "allow"
    else:
        final_decision = "block"
    return {
        "schema": "atlas_decision_trace_v1",
        "symbol": str(status.get("symbol", "")),
        "timeframe": str(status.get("timeframe", "")),
        "final_action": final_action,
        "final_decision": final_decision,
        "message": str(status.get("message", "")),
        "blocker_class": blocker["class"],
        "primary_blocker": blocker["primary"],
        "blocker_recoverability": blocker["recoverability"],
        "blocker_summary": blocker,
        "bias_breakdown": _bias_breakdown(status, context_snapshot),
        "signal": {
            "raw": _compact_signal_rows(status.get("signals", []), limit=8),
            "active": _compact_signal_rows(status.get("signals_active", []), limit=8),
            "summary": dict(status.get("signal_summary", {}) or {}),
        },
        "ranking": {
            "chosen_score": ranking.get("chosen_score"),
            "top": list(ranking.get("top", []) or [])[:5],
            "best_actionable": dict(status.get("best_actionable_candidate", {}) or ranking.get("best_actionable", {}) or {}),
        },
        "selected_candidate": {
            "strategy": str(chosen.get("strategy", "none") or "none"),
            "side": str(chosen.get("side", "flat") or "flat"),
            "confidence": round(_safe_float(chosen.get("confidence", 0.0)), 4),
            "signal_confidence_seed": round(_safe_float(chosen.get("signal_confidence_seed", chosen.get("confidence", 0.0))), 4),
            "stop_loss_points": round(_safe_float(chosen.get("stop_loss_points", 0.0)), 4),
            "take_profit_points": round(_safe_float(chosen.get("take_profit_points", 0.0)), 4),
        },
        "context": {
            "session": str(context_snapshot.get("session", status.get("session", "unknown")) or "unknown"),
            "regime": str(context_snapshot.get("regime", context_snapshot.get("regime_name", "unknown")) or "unknown"),
            "trend_strength": round(_safe_float(context_snapshot.get("trend_strength", 0.0)), 6),
            "spread_state": str(context_snapshot.get("spread_state", "unknown") or "unknown"),
            "liquidity_state": str(context_snapshot.get("liquidity_state", "unknown") or "unknown"),
            "volatility_state": str(context_snapshot.get("volatility_state", "unknown") or "unknown"),
            "higher_timeframe_bias": str(status.get("higher_tf_bias", dict(status.get("higher_timeframe", {}) or {}).get("bias", "unknown")) or "unknown"),
            "chart_structure_bias": _bias_breakdown(status, context_snapshot)["chart_structure_bias"],
        },
        "filters": {
            "guards": _guard_trace(status),
            "hard_blockers": hard_blockers,
            "reason_codes": list(decision_explanation.get("reason_codes", []) or []),
        },
        "risk": {
            "daily_loss_pct": risk.get("daily_loss_pct"),
            "weekly_loss_pct": risk.get("weekly_loss_pct"),
            "open_positions": risk.get("open_positions"),
            "spread_points": risk.get("spread_points"),
            "trades_today": risk.get("trades_today"),
            "consecutive_losses": risk.get("consecutive_losses"),
            "max_open_trades": runtime_flags.get("max_open_trades"),
        },
        "execution_gate": {
            "execute_trades": bool(status.get("execute_trades", runtime_flags.get("execute_trades", False))),
            "environment": str(dict(runtime_flags.get("execution", {}) or {}).get("execution_environment", "")),
            "result": dict(execution),
            "request_id": str(execution.get("request_id", "") or ""),
            "decision_id": str(execution.get("decision_id", "") or ""),
        },
        "evidence_separation": build_evidence_separation(status, runtime_flags),
        "trace_notes": [
            "read_only_observability",
            "does_not_modify_strategy_risk_or_execution",
        ],
    }


def build_decision_explanation(
    status: dict[str, Any],
    context_snapshot: dict[str, Any],
    runtime_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_flags = dict(runtime_flags or {})
    chosen = dict(status.get("chosen", {}) or {})
    rollout = dict(runtime_flags.get("paper_rollout", {}) or {})
    final_decision = "allow" if str(status.get("action", "")).lower() in {"paper", "execute", "failed"} else "block"
    best_actionable = dict(status.get("best_actionable_candidate", {}) or {})
    setup_reasons = list(best_actionable.get("reasons", []) or [])
    guards = _active_guards(status)
    penalties = _penalties(status)
    blocker = _decision_blocker_class(status, context_snapshot)
    reason_codes = [_slug(code) for code in setup_reasons]
    reason_codes.extend(_slug(guard.get("name", "")) for guard in guards if guard.get("applied"))
    message_code = _slug(str(status.get("message", "")))
    if message_code:
        reason_codes.append(message_code)
    deduped_reason_codes = [code for idx, code in enumerate(reason_codes) if code and code not in reason_codes[:idx]]

    return {
        "candidate_id": _candidate_id(status),
        "trade_id": str(dict(status.get("execution", {}) or {}).get("decision_id", "")),
        "lane": str(status.get("lane", rollout.get("lane", "")) or "default"),
        "mode": "live" if bool(status.get("execute_trades", False)) else "paper",
        "symbol": str(status.get("symbol", "")),
        "timestamp": str(status.get("updated_at", "")),
        "signal_source": str(chosen.get("strategy", "none")),
        "regime": str(context_snapshot.get("regime", "unknown")),
        "session": str(context_snapshot.get("session", "unknown")),
        "trend_strength": round(_safe_float(context_snapshot.get("trend_strength", 0.0)), 6),
        "setup_quality_reasons": setup_reasons,
        "active_guards": guards,
        "penalties": penalties,
        "final_decision": final_decision,
        "blocker_class": blocker["class"],
        "primary_blocker": blocker["primary"],
        "blocker_recoverability": blocker["recoverability"],
        "bias_breakdown": _bias_breakdown(status, context_snapshot),
        "reason_codes": deduped_reason_codes,
        "explanation_text": str(status.get("message", "")),
    }
