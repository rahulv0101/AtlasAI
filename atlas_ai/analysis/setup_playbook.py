from __future__ import annotations

from typing import Any


_TEACHING_NOTES: dict[str, dict[str, str]] = {
    "pullback_continuation": {
        "good_example": "Clear HH/HL or LH/LL structure, pullback into structure, momentum turns back with the trend, spread acceptable.",
        "bad_example": "Trend label exists but price is mid-range, momentum is weak, or macro/execution context is fighting the idea.",
        "operator_label_hint": "correct_hold if Atlas avoids mid-range or weak momentum; too_cautious/missed_trade only if pullback and confirmation were clean.",
    },
    "breakout_retest": {
        "good_example": "Breakout confirms, retest holds, follow-through returns, and liquidity remains usable.",
        "bad_example": "First break is chased without retest or follow-through, especially in thin liquidity.",
        "operator_label_hint": "bad_setup if Atlas chases; missed_trade only if clean retest confirmation was present.",
    },
    "liquidity_sweep_reversal": {
        "good_example": "Sweep, fast rejection, structure reclaim, and reversal momentum confirmation all appear together.",
        "bad_example": "A wick appears without reclaim or confirmation.",
        "operator_label_hint": "correct_hold when reversal confirmation is missing; bad_setup if Atlas treats sweep alone as enough.",
    },
    "range_edge_reversal": {
        "good_example": "Defined range, price at edge, rejection appears, trend strength stays weak, and execution friction is acceptable.",
        "bad_example": "Price is in the middle of the range or the range edge is not defined.",
        "operator_label_hint": "correct_hold for mid-range avoidance; missed_trade only after clean edge rejection.",
    },
}


def _norm(raw: Any) -> str:
    return str(raw or "").strip().lower()


def _num(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _side_from_structure(trend_state: str, trend_structure: str, chosen_side: str) -> str:
    if chosen_side in {"buy", "sell"}:
        return chosen_side
    if trend_structure.upper() == "HH/HL" or trend_state == "uptrend":
        return "buy"
    if trend_structure.upper() == "LH/LL" or trend_state == "downtrend":
        return "sell"
    return "neutral"


def _readiness(score: int, missing: list[str]) -> str:
    if missing:
        return "developing" if score >= 55 else "not_ready"
    if score >= 78:
        return "ready_for_existing_gates"
    if score >= 62:
        return "developing"
    return "not_ready"


def _candidate(
    *,
    name: str,
    side: str,
    base: int,
    present: list[str],
    missing: list[str],
    supporting_evidence: list[str],
    risk_notes: list[str],
    execution_focus: str,
) -> dict[str, Any]:
    score = int(max(0, min(100, round(base))))
    return {
        "name": name,
        "side": side,
        "score": score,
        "readiness": _readiness(score, missing),
        "present": list(dict.fromkeys(present))[:8],
        "missing": list(dict.fromkeys(missing))[:8],
        "supporting_evidence": list(dict.fromkeys(supporting_evidence))[:8],
        "risk_notes": list(dict.fromkeys(risk_notes))[:8],
        "execution_focus": execution_focus,
        "teaching": dict(_TEACHING_NOTES.get(name, {}) or {}),
    }


def _setup_phase(candidate: dict[str, Any]) -> str:
    readiness = str(candidate.get("readiness", "") or "")
    score = int(candidate.get("score", 0) or 0)
    missing = list(candidate.get("missing", []) or [])
    risks = list(candidate.get("risk_notes", []) or [])
    if readiness == "ready_for_existing_gates":
        return "entry_ready"
    if score >= 62:
        return "forming_confirmations_needed"
    if score >= 48 and missing:
        return "early_watch"
    if risks:
        return "conditions_blocking"
    return "not_forming"


def _confirmation_label(code: str) -> str:
    labels = {
        "pullback_entry_location": "price must pull back into a valid structure/reaction zone",
        "momentum_confirmation": "M1 momentum must turn back in the setup direction",
        "breakout_or_retest_location": "breakout must retest or hold a clean reaction level",
        "breakout_follow_through": "breakout needs follow-through after the retest",
        "range_context": "range context must be clearly defined",
        "range_edge_location": "price must be at a range edge, not the middle",
        "avoid_mid_range_entry": "entry must move away from mid-range",
        "clear_hh_hl_or_lh_ll_structure": "trend structure must be clear HH/HL or LH/LL",
        "directional_trend_context": "directional trend context must be clear",
        "new_local_high": "liquidity sweep needs a new local high or low",
        "new_local_low": "liquidity sweep needs a new local high or low",
        "exhaustion_wick": "sweep needs rejection or exhaustion wick evidence",
        "structure_break": "reversal needs structure reclaim/break confirmation",
    }
    return labels.get(str(code or ""), str(code or "").replace("_", " "))


def _risk_label(code: str) -> str:
    labels = {
        "execution_friction_elevated": "spread/liquidity must stay acceptable",
        "macro_alignment_conflict": "macro conflict must ease or price action must strongly override it",
        "event_risk_elevated": "event/news risk must clear",
        "counter_trend_requires_confirmation": "counter-trend idea needs stronger confirmation",
    }
    return labels.get(str(code or ""), str(code or "").replace("_", " "))


def _trigger_state(candidate: dict[str, Any], blockers: list[str]) -> str:
    phase = _setup_phase(candidate)
    missing = [str(x) for x in list(candidate.get("missing", []) or [])]
    risks = [str(x) for x in list(candidate.get("risk_notes", []) or [])]
    hard_blocking = [x for x in blockers if x not in {"no_clear_setup"}]
    if phase == "entry_ready" and not risks and not hard_blocking:
        return "ready_for_existing_gates"
    if phase == "entry_ready":
        return "armed_but_context_blocked"
    if phase in {"forming_confirmations_needed", "early_watch"}:
        if len(missing) <= 1 and not hard_blocking:
            return "armed_waiting_for_final_confirmation"
        return "forming_waiting_for_confirmations"
    if hard_blocking or risks:
        return "context_blocked"
    return "not_armed"


def _trigger_checklist(candidate: dict[str, Any], blockers: list[str]) -> list[dict[str, Any]]:
    missing = [str(x) for x in list(candidate.get("missing", []) or [])]
    present = {str(x) for x in list(candidate.get("present", []) or [])}
    risks = [str(x) for x in list(candidate.get("risk_notes", []) or [])]
    checks: list[dict[str, Any]] = []
    for code in present:
        checks.append({"code": code, "status": "present", "label": code.replace("_", " ")})
    for code in missing:
        checks.append({"code": code, "status": "missing", "label": _confirmation_label(code)})
    for code in risks:
        checks.append({"code": code, "status": "risk", "label": _risk_label(code)})
    for code in [x for x in blockers if x not in {"no_clear_setup"}]:
        checks.append({"code": code, "status": "blocked", "label": code.replace("_", " ")})
    order = {"blocked": 0, "risk": 1, "missing": 2, "present": 3}
    return sorted(checks, key=lambda row: order.get(str(row.get("status", "")), 9))[:10]


def build_setup_radar(
    *,
    playbook: dict[str, Any],
    context: dict[str, Any],
    hard_blockers: list[str] | None = None,
) -> dict[str, Any]:
    """Expose forming setups without changing execution decisions."""
    ranked = [dict(x or {}) for x in list(dict(playbook or {}).get("ranked_candidates", []) or [])]
    blockers = [str(x) for x in list(hard_blockers or [])]
    radar_momentum = _norm(context.get("momentum"))
    watchlist: list[dict[str, Any]] = []
    for row in ranked[:4]:
        missing = [str(x) for x in list(row.get("missing", []) or [])]
        risks = [str(x) for x in list(row.get("risk_notes", []) or [])]
        display_row = dict(row)
        display_missing = list(missing)
        if (
            str(row.get("name", "") or "") in {"pullback_continuation", "breakout_retest"}
            and radar_momentum == "mixed"
            and "momentum_confirmation" not in display_missing
        ):
            display_missing.append("momentum_confirmation")
        display_row["missing"] = display_missing
        phase = _setup_phase(display_row)
        trigger_state = _trigger_state(display_row, blockers)
        checklist = _trigger_checklist(display_row, blockers)
        next_missing = [x for x in checklist if str(x.get("status", "")) in {"blocked", "risk", "missing"}]
        watchlist.append(
            {
                "name": str(row.get("name", "unknown") or "unknown"),
                "side": str(row.get("side", "neutral") or "neutral"),
                "score": int(row.get("score", 0) or 0),
                "readiness": str(row.get("readiness", "not_ready") or "not_ready"),
                "phase": phase,
                "trigger_state": trigger_state,
                "trigger_checklist": checklist,
                "missing_to_entry": display_missing[:4],
                "risks": risks[:4],
                "next_confirmation": (
                    str(next_missing[0].get("label", "")) if next_missing else "existing execution gates"
                ),
                "execution_focus": str(row.get("execution_focus", "") or ""),
            }
        )

    top = watchlist[0] if watchlist else {}
    spread_state = _norm(context.get("spread_state"))
    spread_points = _num(context.get("spread_points"), 0.0)
    session = _norm(context.get("session"))
    momentum = _norm(context.get("momentum"))
    location = _norm(context.get("location"))
    trend_strength = _num(context.get("trend_strength"), 0.0)
    conditions: list[str] = []
    if "spread_liquidity_too_poor" in blockers or spread_state in {"wide", "poor", "elevated"} or spread_points >= 40:
        conditions.append("spread/liquidity is slowing valid entries")
    if "no_clear_setup" in blockers:
        conditions.append("directional edge is not clear enough")
    if momentum in {"mixed", "weak"}:
        conditions.append(f"momentum is {momentum}")
    if location == "mid_range":
        conditions.append("price is mid-range")
    if session in {"late_us", "off", "off_hours"}:
        conditions.append(f"session is {session}")
    if trend_strength < 0.08:
        conditions.append("trend strength is weak")

    timing_state = "wait"
    if top:
        if top.get("phase") == "entry_ready" and not blockers:
            timing_state = "entry_ready_for_existing_gates"
        elif top.get("phase") in {"forming_confirmations_needed", "early_watch"}:
            timing_state = "setup_forming"
        elif conditions:
            timing_state = "conditions_unfavorable"

    return {
        "mode": "observability_only",
        "timing_state": timing_state,
        "top_watch": top,
        "watchlist": watchlist,
        "conditions_slowing_setups": conditions[:6],
        "operator_summary": (
            f"{top.get('name', 'No setup')} is {top.get('trigger_state', top.get('phase', 'not_forming'))}."
            if top
            else "No recognizable setup is forming."
        ),
        "policy_note": "Setup radar is advisory only; it does not lower thresholds or bypass gates.",
    }


def build_setup_quality_observation(playbook: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compact advisory setup-quality view for dashboards and learning.

    This intentionally does not feed back into execution gates. It turns the
    playbook score into a visible operator/learning signal so Atlas can explain
    what is forming instead of reporting 0.0 setup quality when the hard setup
    filter is disabled.
    """
    pb = dict(playbook or {})
    top = dict(pb.get("top_candidate", {}) or {})
    ranked = [dict(x or {}) for x in list(pb.get("ranked_candidates", []) or [])]
    ctx = dict(context or {})
    score = int(top.get("score", 0) or 0)
    missing = [str(x) for x in list(top.get("missing", []) or [])]
    risks = [str(x) for x in list(top.get("risk_notes", []) or [])]
    readiness = str(top.get("readiness", "not_ready") or "not_ready")
    location = _norm(ctx.get("location"))
    momentum = _norm(ctx.get("momentum"))
    trend_structure = str(ctx.get("trend_structure", "") or "")
    chosen_side = _norm(ctx.get("chosen_side"))
    structure_side = "buy" if trend_structure.upper() == "HH/HL" else ("sell" if trend_structure.upper() == "LH/LL" else "neutral")
    bias_entry_conflict = bool(chosen_side in {"buy", "sell"} and structure_side in {"buy", "sell"} and chosen_side != structure_side)

    lessons: list[str] = []
    if location == "mid_range":
        lessons.append("mid-range is observation, not entry")
    if momentum in {"weak", "mixed"}:
        lessons.append("wait for momentum confirmation")
    if bias_entry_conflict:
        lessons.append("bias and entry direction conflict")
    if "pullback_entry_location" in missing:
        lessons.append("wait for pullback location")
    if "breakout_follow_through" in missing:
        lessons.append("wait for breakout follow-through/retest")
    if not lessons and readiness == "ready_for_existing_gates":
        lessons.append("setup quality is ready for existing gates")
    if not lessons:
        lessons.append("collect more setup evidence")

    return {
        "mode": "advisory_only",
        "score": score,
        "score_scale": "0_to_100",
        "top_setup": str(top.get("name", "unknown") or "unknown"),
        "side": str(top.get("side", "neutral") or "neutral"),
        "readiness": readiness,
        "missing": missing[:6],
        "risks": risks[:6],
        "bias_entry_conflict": bias_entry_conflict,
        "lessons": list(dict.fromkeys(lessons))[:6],
        "ranked": [
            {
                "name": str(row.get("name", "unknown") or "unknown"),
                "score": int(row.get("score", 0) or 0),
                "readiness": str(row.get("readiness", "not_ready") or "not_ready"),
                "missing": list(row.get("missing", []) or [])[:4],
            }
            for row in ranked[:4]
        ],
        "policy_note": "Advisory setup-quality observation only; it does not change strategy, risk, thresholds, or execution policy.",
    }


def score_setup_playbook(
    *,
    status: dict[str, Any] | None,
    runtime: dict[str, Any] | None = None,
    chart_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank recognizable setup families without changing execution policy."""
    st = dict(status or {})
    rt = dict(runtime or {})
    context = dict(st.get("context_snapshot", {}) or {})
    chart = dict(chart_input or (st.get("chart_features") if isinstance(st.get("chart_features"), dict) else {}) or {})
    chosen = dict(st.get("chosen", {}) or {})

    trend_state = _norm(context.get("trend_state", chart.get("trend_state", "unclear")))
    trend_structure = str(context.get("trend_structure", chart.get("trend_structure", "mixed")) or "mixed")
    momentum = _norm(context.get("momentum", chart.get("momentum_state", "mixed")))
    location = _norm(context.get("location", chart.get("location_state", "unclear")))
    setup_hint = _norm(st.get("setup_family", chart.get("setup_type_hint", "no_trade")))
    chosen_side = _norm(chosen.get("side"))
    side = _side_from_structure(trend_state, trend_structure, chosen_side)

    spread_state = _norm(context.get("spread_state"))
    liquidity_state = _norm(context.get("liquidity_state"))
    spread_points = _num(context.get("spread_points", rt.get("spread_points", 0.0)))
    max_spread = _num(rt.get("max_spread_points", 35.0), 35.0)
    macro_states = [
        _norm(context.get("dxy_alignment")),
        _norm(context.get("us10y_alignment")),
        _norm(context.get("vix_alignment")),
    ]
    macro_conflicts = sum(1 for x in macro_states if x in {"conflict", "conflicting", "opposed"})
    event_state = _norm(context.get("event_risk_state", context.get("news_risk_state", "")))
    high_news = bool(context.get("in_high_impact_news_window") or context.get("high_impact_news_window"))
    regime = _norm(st.get("market_regime", {}).get("regime") if isinstance(st.get("market_regime"), dict) else "")
    if not regime:
        regime = _norm(context.get("market_regime", context.get("regime", "")))

    clear_up = trend_structure.upper() == "HH/HL" or trend_state == "uptrend"
    clear_down = trend_structure.upper() == "LH/LL" or trend_state == "downtrend"
    clear_trend = clear_up or clear_down
    range_like = trend_state in {"range", "ranging"} or regime == "ranging"
    strong_momentum = momentum == "strong"
    acceptable_momentum = momentum in {"strong", "mixed"}
    good_spread = spread_state in {"tight", "normal", "acceptable"} or (spread_points > 0 and spread_points < max_spread)
    weak_spread = spread_state == "wide" or liquidity_state in {"poor", "below_normal", "illiquid"} or (spread_points > 0 and spread_points >= max_spread)

    common_risk: list[str] = []
    if weak_spread:
        common_risk.append("execution_friction_elevated")
    if macro_conflicts >= 2:
        common_risk.append("macro_alignment_conflict")
    if event_state in {"high", "elevated", "locked", "imminent", "extreme"} or high_news:
        common_risk.append("event_risk_elevated")

    candidates: list[dict[str, Any]] = []

    present = []
    missing = []
    score = 36
    if clear_trend:
        score += 24
        present.append("clear_trend_structure")
    else:
        missing.append("clear_hh_hl_or_lh_ll_structure")
    if location == "pullback":
        score += 18
        present.append("pullback_location")
    else:
        missing.append("pullback_entry_location")
    if acceptable_momentum:
        score += 10 if strong_momentum else 4
        present.append("momentum_not_against_structure")
    else:
        missing.append("momentum_confirmation")
    if good_spread:
        score += 6
        present.append("execution_friction_acceptable")
    candidates.append(_candidate(
        name="pullback_continuation",
        side=side,
        base=score - macro_conflicts * 7 - len(common_risk) * 3,
        present=present,
        missing=missing,
        supporting_evidence=[f"structure={trend_structure}", f"momentum={momentum}", f"location={location}"],
        risk_notes=common_risk,
        execution_focus="wait for M1 confirmation in the pullback direction before existing gates can approve.",
    ))

    present = []
    missing = []
    score = 32
    if clear_trend:
        score += 18
        present.append("trend_context")
    else:
        missing.append("directional_trend_context")
    if location == "breakout" or setup_hint == "breakout":
        score += 22
        present.append("breakout_location")
    else:
        missing.append("breakout_or_retest_location")
    if strong_momentum:
        score += 14
        present.append("strong_breakout_momentum")
    else:
        missing.append("breakout_follow_through")
    if good_spread:
        score += 6
        present.append("execution_friction_acceptable")
    candidates.append(_candidate(
        name="breakout_retest",
        side=side,
        base=score - macro_conflicts * 8 - len(common_risk) * 3,
        present=present,
        missing=missing,
        supporting_evidence=[f"setup_hint={setup_hint}", f"momentum={momentum}", f"location={location}"],
        risk_notes=common_risk,
        execution_focus="prefer confirmed retest/follow-through instead of chasing the first break.",
    ))

    reversal_flags = {
        "new_local_high": bool(context.get("new_local_high") or chart.get("new_local_high")),
        "new_local_low": bool(context.get("new_local_low") or chart.get("new_local_low")),
        "exhaustion_wick": bool(context.get("exhaustion_wick") or chart.get("exhaustion_wick")),
        "structure_break": bool(context.get("reversal_structure_break") or chart.get("reversal_structure_break")),
        "momentum_confirmation": bool(context.get("reversal_momentum_confirmation") or chart.get("reversal_momentum_confirmation")),
    }
    present = [k for k, v in reversal_flags.items() if v]
    missing = [k for k, v in reversal_flags.items() if not v]
    score = 28 + len(present) * 11
    candidates.append(_candidate(
        name="liquidity_sweep_reversal",
        side="sell" if reversal_flags["new_local_high"] else ("buy" if reversal_flags["new_local_low"] else "neutral"),
        base=score - len(common_risk) * 3,
        present=present,
        missing=missing,
        supporting_evidence=[f"{k}={v}" for k, v in reversal_flags.items()],
        risk_notes=common_risk + (["counter_trend_requires_confirmation"] if clear_trend else []),
        execution_focus="only treat as actionable after structure break and reversal momentum confirmation.",
    ))

    present = []
    missing = []
    score = 34
    if range_like:
        score += 24
        present.append("range_regime")
    else:
        missing.append("range_context")
    if location in {"pullback", "consolidation"} or setup_hint in {"mean_reversion", "consolidation"}:
        score += 16
        present.append("range_edge_or_reaction_zone_candidate")
    else:
        missing.append("range_edge_location")
    if location != "mid_range":
        score += 8
        present.append("not_mid_range")
    else:
        missing.append("avoid_mid_range_entry")
    candidates.append(_candidate(
        name="range_edge_reversal",
        side=side if side in {"buy", "sell"} else "neutral",
        base=score - len(common_risk) * 4,
        present=present,
        missing=missing,
        supporting_evidence=[f"regime={regime or trend_state}", f"location={location}", f"setup_hint={setup_hint}"],
        risk_notes=common_risk,
        execution_focus="wait for range-edge rejection; avoid mid-range execution.",
    ))

    ranked = sorted(candidates, key=lambda item: int(item.get("score", 0)), reverse=True)
    top = ranked[0] if ranked else {}
    return {
        "mode": "advisory_only",
        "policy_note": "Ranks setup quality only; existing risk and execution gates remain authoritative.",
        "top_candidate": top,
        "ranked_candidates": ranked,
        "current_missing": list(top.get("missing", []) or [])[:6],
        "current_risks": list(top.get("risk_notes", []) or [])[:6],
        "summary": f"{top.get('name', 'no_setup')} is the closest playbook at {top.get('score', 0)}/100.",
    }
