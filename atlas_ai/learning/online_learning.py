from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from atlas_ai.learning.promotion_thresholds import online_path_thresholds


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _as_list(raw: Any) -> list[Any]:
    return list(raw or []) if isinstance(raw, list) else []


def _learning_mode(evidence_strength: float, overreaction_risk: float, samples: int, thresholds: dict[str, Any] | None = None) -> str:
    cfg = dict(thresholds or online_path_thresholds())
    fast = dict(cfg.get("fast_path", {}) or {})
    medium = dict(cfg.get("medium_path", {}) or {})
    if (
        samples >= int(fast["min_samples"])
        and evidence_strength >= float(fast["min_evidence_strength"])
        and overreaction_risk <= float(fast["max_overreaction_risk"])
    ):
        return "fast_path_observe"
    if (
        samples >= int(medium["min_samples"])
        and evidence_strength >= float(medium["min_evidence_strength"])
        and overreaction_risk <= float(medium["max_overreaction_risk"])
    ):
        return "medium_path_observe"
    return "slow_path_observe"


def _lesson_digest(
    focus: list[str],
    *,
    pattern_count: int,
    evidence_strength: float,
    overreaction_risk: float,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    digest_thresholds = dict(dict(thresholds or online_path_thresholds()).get("lesson_digest", {}) or {})
    focus = [str(x).strip() for x in list(focus or []) if str(x).strip()]
    primary = focus[0] if focus else "continue collecting evidence"
    if (
        evidence_strength >= float(digest_thresholds["strong_min_evidence_strength"])
        and overreaction_risk <= float(digest_thresholds["strong_max_overreaction_risk"])
        and pattern_count >= int(digest_thresholds["strong_min_samples"])
    ):
        confidence = "strong"
        action = "promote_to_operator_lesson"
    elif (
        evidence_strength >= float(digest_thresholds["moderate_min_evidence_strength"])
        and pattern_count >= int(digest_thresholds["moderate_min_samples"])
    ):
        confidence = "moderate"
        action = "keep_as_watch_lesson"
    else:
        confidence = "weak"
        action = "observe_more"
    return {
        "primary_lesson": primary,
        "supporting_lessons": list(dict.fromkeys(focus[1:]))[:5],
        "confidence": confidence,
        "recommended_use": action,
        "pattern_count": int(pattern_count),
        "policy_note": "Lesson digest is explanatory memory only; it cannot rewrite strategies, thresholds, risk, or execution authority.",
    }


def build_online_learning_observation(
    *,
    status: dict[str, Any],
    near_setup_sample: dict[str, Any] | None = None,
    near_setup_learning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize live evidence without changing trading policy.

    The output is intentionally report-only. It can teach the Brain what Atlas
    keeps seeing, but it must not mutate strategy, risk, or execution settings.
    """
    st = dict(status or {})
    sample = dict(near_setup_sample or {})
    learning = dict(near_setup_learning or {})
    decision = dict(st.get("decision_analysis", {}) or {})
    context = dict(st.get("context_snapshot", {}) or {})
    risk = dict(st.get("risk", {}) or {})
    chosen = dict(st.get("chosen", {}) or {})
    top_patterns = _as_list(learning.get("top_patterns"))
    top_pattern = dict(top_patterns[0] or {}) if top_patterns else {}

    hard_blockers = [str(x) for x in _as_list(decision.get("hard_blockers") or decision.get("blockers"))]
    confluence = _safe_float(decision.get("confluence_score", 0.0))
    threshold = _safe_float(decision.get("confluence_threshold", 50.0), 50.0)
    confluence_gap = round(threshold - confluence, 2)
    samples = int(learning.get("total_samples", 0) or 0)
    pattern_count = int(top_pattern.get("count", 0) or 0)
    spread_points = _safe_float(context.get("spread_points", risk.get("spread_points", 0.0)))
    session = str(context.get("session") or st.get("session") or "unknown")
    momentum = str(context.get("momentum") or dict(st.get("chart_features", {}) or {}).get("momentum_state") or "unknown")
    location = str(context.get("location") or dict(st.get("chart_features", {}) or {}).get("location_state") or "unknown")
    promotion_thresholds = online_path_thresholds()

    evidence_components = 0.0
    evidence_components += min(0.3, pattern_count / 30.0)
    evidence_components += 0.2 if hard_blockers else 0.0
    evidence_components += 0.18 if 0 <= confluence_gap <= 12 else 0.0
    evidence_components += 0.14 if sample else 0.0
    evidence_components += 0.1 if momentum in {"mixed", "weak", "strong"} else 0.0
    evidence_strength = round(max(0.0, min(1.0, evidence_components)), 4)

    overreaction = 0.15
    if samples < 5:
        overreaction += 0.35
    if session in {"late_us", "off_hours", "asia"}:
        overreaction += 0.18
    if spread_points >= 40:
        overreaction += 0.16
    if location == "mid_range":
        overreaction += 0.12
    overreaction_risk = round(max(0.0, min(1.0, overreaction)), 4)
    mode = _learning_mode(evidence_strength, overreaction_risk, samples, promotion_thresholds)

    focus = []
    if sample:
        focus.extend(str(x) for x in _as_list(sample.get("teaching_focus")))
    elif top_pattern:
        focus.extend(str(x) for x in _as_list(top_pattern.get("teaching_focus")))
    if not focus and hard_blockers:
        focus.append("inspect repeated blockers before proposing any parameter change")
    if not focus:
        focus.append("continue collecting evidence")

    allow_haircut = bool(
        samples >= 8
        and evidence_strength >= 0.65
        and (
            location == "mid_range"
            or spread_points >= 40
            or "spread_liquidity_too_poor" in hard_blockers
            or "no_clear_setup" in hard_blockers
        )
    )
    confidence_haircut_cap = 0.0
    if allow_haircut:
        confidence_haircut_cap = round(min(0.08, 0.02 + evidence_strength * 0.045), 4)

    lesson_digest = _lesson_digest(
        list(dict.fromkeys(focus))[:6],
        pattern_count=pattern_count,
        evidence_strength=evidence_strength,
        overreaction_risk=overreaction_risk,
        thresholds=promotion_thresholds,
    )

    return {
        "version": "bounded_online_learning_v1",
        "mode": mode,
        "allowed_influence": "bounded_confidence_haircut" if allow_haircut else "report_only",
        "confidence_haircut_cap": confidence_haircut_cap,
        "evidence_strength": evidence_strength,
        "overreaction_risk": overreaction_risk,
        "samples": samples,
        "top_pattern_count": pattern_count,
        "top_fingerprint": str(sample.get("fingerprint") or top_pattern.get("fingerprint") or ""),
        "current_action": str(st.get("action") or "unknown"),
        "current_blockers": hard_blockers,
        "chosen": {
            "strategy": str(chosen.get("strategy", "") or ""),
            "side": str(chosen.get("side", "") or "flat"),
            "confidence": round(_safe_float(chosen.get("confidence", 0.0)), 4),
        },
        "context": {
            "session": session,
            "momentum": momentum,
            "location": location,
            "spread_points": spread_points,
            "confluence_gap": confluence_gap,
        },
        "teaching_focus": list(dict.fromkeys(focus))[:6],
        "lesson_digest": lesson_digest,
        "policy_note": "Online learning is bounded; it cannot rewrite strategies, change risk, alter execution policy, bypass gates, or boost confidence.",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def update_online_learning_state(current_state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    state = dict(current_state or {})
    obs = dict(observation or {})
    if not obs:
        return state
    online = dict(state.get("online_learning", {}) or {})
    recent = _as_list(online.get("recent_observations"))
    recent.append(obs)
    recent = recent[-100:]
    fingerprints: dict[str, int] = {}
    for row in recent:
        key = str(dict(row or {}).get("top_fingerprint", "") or "unknown")
        fingerprints[key] = fingerprints.get(key, 0) + 1
    top = sorted(fingerprints.items(), key=lambda item: item[1], reverse=True)[:10]
    online = {
        "version": "bounded_online_learning_v1",
        "allowed_influence": str(obs.get("allowed_influence", "report_only") or "report_only"),
        "latest": obs,
        "recent_observations": recent,
        "top_fingerprints": [{"fingerprint": key, "count": count} for key, count in top],
        "lesson_digest": dict(obs.get("lesson_digest", {}) or {}),
        "policy_note": "Online learning may apply only bounded confidence haircuts when explicitly eligible; live trading behavior remains controlled by existing strategies, risk policy, and ExecutionGate.",
        "updated_at": obs.get("updated_at") or datetime.now(timezone.utc).isoformat(),
    }
    state["online_learning"] = online
    return state
