from __future__ import annotations

from typing import Any


def _num(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _text(raw: Any, default: str = "") -> str:
    value = str(raw or "").strip()
    return value or default


def _list(raw: Any, limit: int = 8) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item or "").strip()][:limit]


def _confidence_gap(confidence: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return round(confidence - threshold, 4)


def _phase_from_radar(row: dict[str, Any], blockers: list[str]) -> str:
    phase = _text(row.get("phase"), "")
    trigger = _text(row.get("trigger_state"), "")
    if trigger == "ready_for_existing_gates" or phase == "entry_ready":
        return "candidate_near_entry" if not blockers else "candidate_context_blocked"
    if phase in {"forming_confirmations_needed", "early_watch"}:
        return "forming"
    if blockers:
        return "context_blocked"
    return "scanning"


def _formation_tasks(row: dict[str, Any]) -> list[str]:
    missing = _list(row.get("missing_to_entry") or row.get("missing"), 5)
    tasks: list[str] = []
    for code in missing:
        label = code.replace("_", " ")
        if "momentum" in code:
            tasks.append("watch for momentum confirmation in the setup direction")
        elif "pullback" in code or "location" in code:
            tasks.append("track price moving into a valid entry location")
        elif "retest" in code or "breakout" in code:
            tasks.append("track breakout retest and follow-through")
        elif "range" in code:
            tasks.append("track range edge rejection instead of mid-range")
        else:
            tasks.append(f"watch for {label}")
    if not tasks:
        tasks.append("monitor existing execution gates and final confirmation")
    return list(dict.fromkeys(tasks))[:4]


def build_setup_hunter(
    *,
    status: dict[str, Any],
    decision_analysis: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank forming setup candidates without changing strategy or execution policy.

    The hunter turns passive setup-radar output and raw strategy signals into a
    clear watchlist. It is intentionally advisory: it can look for and describe
    setup formation, but it cannot invent a market setup, lower thresholds, or
    bypass risk/execution gates.
    """
    st = dict(status or {})
    analysis = dict(decision_analysis or st.get("decision_analysis", {}) or {})
    rt = dict(runtime or {})
    setup_radar = dict(analysis.get("setup_radar", {}) or {})
    chosen = dict(st.get("chosen", {}) or {})
    thresholds = dict(st.get("thresholds", {}) or {})
    threshold = _num(thresholds.get("confidence", st.get("min_confidence_threshold", rt.get("min_confidence_threshold", 0.0))), 0.0)
    hard_blockers = _list(analysis.get("hard_blockers", analysis.get("blockers", [])), 10)

    candidates: list[dict[str, Any]] = []
    for row in list(setup_radar.get("watchlist", []) or [])[:5]:
        item = dict(row or {})
        score = _num(item.get("score"), 0.0)
        confidence = round(score / 100.0, 4) if score > 1 else round(score, 4)
        candidates.append(
            {
                "source": "setup_radar",
                "name": _text(item.get("name"), "setup"),
                "side": _text(item.get("side"), "neutral"),
                "score": int(max(0, min(100, round(score)))),
                "confidence": confidence,
                "confidence_gap": _confidence_gap(confidence, threshold),
                "readiness": _text(item.get("readiness"), "not_ready"),
                "phase": _phase_from_radar(item, hard_blockers),
                "trigger_state": _text(item.get("trigger_state"), ""),
                "missing_to_entry": _list(item.get("missing_to_entry"), 6),
                "risks": _list(item.get("risks"), 6),
                "next_confirmation": _text(item.get("next_confirmation"), "existing execution gates"),
                "formation_tasks": _formation_tasks(item),
            }
        )

    seen_signal_keys = {(_text(row.get("name")), _text(row.get("side"))) for row in candidates}
    for signal in list(st.get("signals", []) or [])[:12]:
        sig = dict(signal or {})
        side = _text(sig.get("side"), "flat").lower()
        if side not in {"buy", "sell"}:
            continue
        confidence = _num(sig.get("confidence"), 0.0)
        strategy = _text(sig.get("strategy"), "signal_candidate")
        key = (strategy, side)
        if key in seen_signal_keys:
            continue
        seen_signal_keys.add(key)
        phase = "candidate_near_entry" if threshold > 0 and confidence >= threshold else "forming"
        candidates.append(
            {
                "source": "strategy_signal",
                "name": strategy,
                "side": side,
                "score": int(max(0, min(100, round(confidence * 100)))),
                "confidence": round(confidence, 4),
                "confidence_gap": _confidence_gap(confidence, threshold),
                "readiness": "signal_candidate",
                "phase": phase,
                "trigger_state": "raw_signal_tracking",
                "missing_to_entry": ["setup_confirmation"] if phase == "forming" else [],
                "risks": hard_blockers[:4],
                "next_confirmation": "confirm signal quality and setup fit",
                "formation_tasks": ["track whether the raw signal gains setup confirmation"],
            }
        )

    candidates.sort(
        key=lambda row: (
            1 if str(row.get("phase")) == "candidate_near_entry" else 0,
            _num(row.get("score"), 0.0),
            _num(row.get("confidence"), 0.0),
        ),
        reverse=True,
    )
    top = dict(candidates[0]) if candidates else {}
    chosen_side = _text(chosen.get("side"), "flat").lower()
    chosen_conf = _num(chosen.get("confidence"), 0.0)
    state = "scanning"
    if top:
        state = _text(top.get("phase"), "forming")
    if chosen_side in {"buy", "sell"} and (threshold <= 0 or chosen_conf >= threshold):
        state = "execution_candidate_active"
    if hard_blockers and state == "candidate_near_entry":
        state = "candidate_context_blocked"

    next_action = "scan all known playbooks and raw strategy signals"
    if top:
        tasks = _list(top.get("formation_tasks"), 4)
        next_action = tasks[0] if tasks else _text(top.get("next_confirmation"), next_action)

    return {
        "mode": "proactive_scan",
        "state": state,
        "candidate_count": len(candidates),
        "closest_setup": top,
        "top_candidates": candidates[:5],
        "hard_blockers": hard_blockers[:8],
        "next_action": next_action,
        "operator_summary": (
            f"Actively tracking {top.get('name')} {top.get('side')} setup; next: {next_action}."
            if top
            else "Actively scanning known setup playbooks and raw strategy signals."
        ),
        "policy_note": (
            "Setup hunter is advisory only; it looks for forming setups but does not create fake setups, "
            "lower thresholds, bypass guards, or change execution policy."
        ),
    }
