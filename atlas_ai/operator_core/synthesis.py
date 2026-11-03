from __future__ import annotations

from typing import Any


def _safe_str(v: Any, default: str = "") -> str:
    text = str(v or "").strip()
    return text or default


def _runtime_mode(runtime: dict[str, Any]) -> str:
    exec_state = dict(runtime.get("execution", {}) or {})
    env_label = str(exec_state.get("execution_environment_label", "") or "").strip()
    env_key = str(exec_state.get("execution_environment", "") or "").strip().lower()
    if env_label:
        return env_label
    if env_key == "demo":
        return "Live Demo"
    if env_key == "paper":
        return "Paper"
    if env_key in {"live_real", "real_live", "real"}:
        return "Real Live (disabled)"
    return "Live Demo" if bool(runtime.get("execute_trades", False)) else "Paper"


def _comparison_active(status: dict[str, Any]) -> bool:
    if bool(status.get("comparison_active", False)):
        return True
    comparison = dict(status.get("comparison", {}) or {})
    return bool(comparison.get("active", False))


def _comparison_meta(status: dict[str, Any]) -> tuple[str, str]:
    run = _safe_str(status.get("comparison_run_id"))
    preset = _safe_str(status.get("comparison_preset"))
    comparison = dict(status.get("comparison", {}) or {})
    if not run:
        run = _safe_str(comparison.get("run_id"))
    if not preset:
        preset = _safe_str(comparison.get("preset"))
    return run, preset


def _quick(status: dict[str, Any], runtime: dict[str, Any]) -> str:
    action = _safe_str(status.get("action"), "unknown").upper()
    reason = _safe_str(status.get("message"), "No reason available.")
    mode = _runtime_mode(runtime)
    run, _ = _comparison_meta(status)
    return f"STATUS: {action} | REASON: {reason} | MODE: {mode} | RUN: {run or '-'}"


def _build_self_model(status: dict[str, Any], runtime: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    context = dict(status.get("context_snapshot", {}) or {})
    health = dict(status.get("system_health", {}) or {})
    decision = dict(status.get("decision_explanation", {}) or {})
    conf = float(dict(status.get("chosen", {}) or {}).get("confidence", 0.0) or 0.0)
    threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
    drift_active = bool(dict(status.get("drift", {}) or {}).get("active", False))
    comparison_active = bool(comparison.get("active", _comparison_active(status)))

    readiness = "Not yet"
    readiness_reason = "Atlas remains in paper validation and has no validated live-ready signal."
    if _runtime_mode(runtime) == "LIVE" and conf >= threshold and not drift_active:
        readiness = "Closer, but still operator-gated"
        readiness_reason = "Signal health is stronger, but live readiness still requires controlled validation."

    capabilities = [
        "Interpret current market/trading state from stored status and events",
        "Explain blockers, confidence gaps, and readiness in operator language",
        "Summarize control-vs-experiment comparison outcomes when available",
        "Recommend bounded next-safe experiments in paper mode",
    ]
    limitations = [
        "Cannot claim knowledge beyond available Atlas state and stored evidence",
        "Does not execute live trading actions from Ask Atlas responses",
        "Cannot guarantee live profitability or future market direction",
    ]
    knowledge_domains = [
        "Market context (regime/session/liquidity/spread)",
        "Signal confidence and threshold gating",
        "Comparison run diagnostics and checklist classifications",
        "Runtime safety posture (paper-first validation)",
    ]
    role = "Operator support for grounded analysis, blocker diagnosis, and safe decision support."
    if context:
        role = f"Operator support focused on {str(context.get('regime', 'current'))} conditions and safe paper-mode evaluation."

    return {
        "knowledge_level": "Strong operational understanding within Atlas runtime context; not a general-world oracle.",
        "understands": knowledge_domains,
        "capabilities": capabilities,
        "limitations": limitations,
        "role": role,
        "readiness": readiness,
        "readiness_reason": readiness_reason,
        "confidence": conf,
        "threshold": threshold,
        "confidence_gap": conf - threshold,
        "health_note": "Drift is currently active." if drift_active else "No severe drift flag is active right now.",
        "comparison_note": "Active comparison run is under evaluation." if comparison_active else "No active comparison run is currently flagged.",
        "decision_note": _safe_str(decision.get("explanation_text"), "No current decision explanation is stored."),
        "message": _safe_str(status.get("message"), "No current status message is available."),
        "mode": _runtime_mode(runtime),
        "run_id": _comparison_meta(status)[0],
    }


def _main_blocker(status: dict[str, Any], runtime: dict[str, Any]) -> tuple[str, list[str]]:
    message = _safe_str(status.get("message"), "No high-confidence trade")
    reasons: list[str] = []
    if "no high-confidence" in message.lower():
        blocker = "no setup currently clears confidence thresholds."
        reasons.append("No validated high-confidence signal is active")
    else:
        blocker = message.rstrip(".") + "."
    if _runtime_mode(runtime) == "PAPER":
        reasons.append("Runtime remains paper-only")
    if _comparison_active(status):
        reasons.append("Active comparison run is still under evaluation")
    checklist = dict(dict(status.get("comparison", {}) or {}).get("decision_checklist", {}) or {})
    classification = _safe_str(checklist.get("classification"))
    if classification:
        reasons.append(f"Comparison checklist class is {classification}")
    return blocker, reasons


def synthesize_answer(
    *,
    intent: str,
    mode: str,
    query: str,
    status: dict[str, Any],
    runtime: dict[str, Any],
    events: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    action = _safe_str(status.get("action"), "unknown").upper()
    reason = _safe_str(status.get("message"), "No reason available.")
    run_id, preset = _comparison_meta(status)
    mode_label = _runtime_mode(runtime)
    self_model = _build_self_model(status, runtime, comparison)

    if mode == "quick":
        return _quick(status, runtime)

    if intent == "status":
        return (
            f"Status: {action}\n"
            f"Reason: {reason}\n"
            f"Mode: {mode_label}\n"
            f"Run: {run_id or '-'}"
        )

    if intent == "blocker":
        blocker, reasons = _main_blocker(status, runtime)
        lines = [f"Main blocker: {blocker}", "", "Supporting reasons:"]
        lines.extend(f"- {item}" for item in reasons[:4])
        return "\n".join(lines)

    if intent == "readiness":
        return (
            f"Readiness: {self_model['readiness']}\n"
            f"Why: {self_model['readiness_reason']}\n"
            f"Confidence gate: {self_model['confidence']:.3f} vs {self_model['threshold']:.3f} ({self_model['confidence_gap']:+.3f})"
        )

    if intent == "confidence":
        chosen = dict(status.get("chosen", {}) or {})
        conf = float(chosen.get("confidence", 0.0) or 0.0)
        threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
        gap = conf - threshold
        return (
            f"Confidence: {conf:.3f}\n"
            f"Threshold: {threshold:.3f}\n"
            f"Gap: {gap:+.3f}\n"
            f"Interpretation: {'above threshold' if gap >= 0 else 'below threshold'}"
        )

    if intent in {"self_knowledge", "capabilities", "limitations"}:
        if intent == "capabilities":
            lines = [
                "Knowledge level: Grounded operator intelligence focused on Atlas runtime evidence.",
                "",
                "I understand:",
            ]
            lines.extend(f"- {item}" for item in self_model["understands"])
            lines.append("")
            lines.append("Capabilities:")
            lines.extend(f"- {item}" for item in self_model["capabilities"])
            lines.append("")
            lines.append(f"Current role: {self_model['role']}")
            return "\n".join(lines)

        if intent == "limitations":
            lines = [
                "Knowledge level: Strong inside Atlas state boundaries, limited outside them.",
                "",
                "Limitations:",
            ]
            lines.extend(f"- {item}" for item in self_model["limitations"])
            lines.append("")
            lines.append("Current role:")
            lines.append(f"- {self_model['role']}")
            return "\n".join(lines)

        lines = [
            f"Knowledge level: {self_model['knowledge_level']}",
            "",
            "I understand:",
        ]
        lines.extend(f"- {item}" for item in self_model["understands"])
        lines.append("")
        lines.append("Limitations:")
        lines.extend(f"- {item}" for item in self_model["limitations"])
        lines.append("")
        lines.append("Current role:")
        lines.append(f"- {self_model['role']}")
        lines.append("")
        lines.append(f"Readiness: {self_model['readiness']} ({self_model['readiness_reason']})")
        return "\n".join(lines)

    if intent == "comparison":
        cmp_payload = comparison or dict(status.get("comparison", {}) or {})
        chk = dict(cmp_payload.get("decision_checklist", {}) or {})
        cls = _safe_str(chk.get("classification"), "unknown")
        active = bool(cmp_payload.get("active", _comparison_active(status)))
        return (
            f"Comparison: {'active' if active else 'inactive'}\n"
            f"Run: {run_id or '-'}\n"
            f"Preset: {preset or '-'}\n"
            f"Safest conclusion: {cls}"
        )

    if intent == "next_safe_experiment":
        return (
            "Next safe step: keep paper mode and run one bounded filter review.\n"
            "Scope: review latest blocked candidates in current comparison run.\n"
            "Safety: no live execution, no threshold mutation without explicit operator approval."
        )

    if intent == "summary":
        tail = events[-1] if events else {}
        latest = _safe_str(tail.get("message"), reason)
        return (
            f"Summary: {action} in {mode_label} mode.\n"
            f"Latest signal: {latest}\n"
            f"Comparison run: {run_id or '-'} ({'active' if _comparison_active(status) else 'inactive'})"
        )

    if intent == "explanation":
        blocker, reasons = _main_blocker(status, runtime)
        support = reasons[0] if reasons else reason
        return f"Answer: {blocker}\nWhy: {support}\nCurrent state: {action} in {mode_label} mode."

    return (
        f"Best grounded answer: {action} with reason '{reason}'.\n"
        f"Mode: {mode_label}\n"
        f"Run: {run_id or '-'}"
    )
