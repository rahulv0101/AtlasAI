from __future__ import annotations

import ast
import re
from typing import Any

from atlas_ai.brain.types import AnswerPlan, EvidenceItem


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(text or "").lower()) if len(t) >= 3}


def _best_named_item(
    query: str,
    items: list[dict[str, Any]],
    *,
    name_field: str,
    fields: tuple[str, ...],
) -> dict[str, Any] | None:
    q_tokens = _tokens(query)
    if not q_tokens:
        return None
    best: dict[str, Any] | None = None
    best_score = 0
    for item in items:
        name = str(item.get(name_field, "")).lower()
        if name and name in str(query or "").lower():
            return item
        name_tokens = _tokens(name)
        field_text: list[str] = []
        for field in fields:
            value = item.get(field)
            if isinstance(value, list):
                field_text.extend(str(x) for x in value)
            elif isinstance(value, str):
                field_text.append(value)
        score = len(q_tokens.intersection(name_tokens)) * 4 + len(q_tokens.intersection(_tokens(" ".join(field_text))))
        if score > best_score:
            best = item
            best_score = score
    return best if best_score > 0 else None


def _why_trace_from_evidence(evidence: list[EvidenceItem]) -> dict[str, Any]:
    for item in evidence:
        if item.source == "why.execution_trace":
            try:
                parsed = ast.literal_eval(str(item.content))
            except Exception:
                parsed = {}
            if isinstance(parsed, dict):
                return parsed
    return {}


def _blocked_stage_from_why_trace(trace: dict[str, Any]) -> dict[str, Any]:
    stages = list(dict(trace or {}).get("stages", []) or [])
    first = str(dict(trace or {}).get("first_blocked_stage", "") or "")
    for row in stages:
        data = dict(row or {})
        if str(data.get("stage", "") or "") == first and str(data.get("status", "") or "") == "blocked":
            return data
    for row in stages:
        data = dict(row or {})
        if str(data.get("status", "") or "") == "blocked":
            return data
    return {}


def _format_structured_block(stage: dict[str, Any]) -> tuple[str, str, str]:
    stage_name = str(stage.get("stage", "unknown") or "unknown")
    reason = str(stage.get("reason", "") or "No structured reason recorded.")
    data = dict(stage.get("data", {}) or {})
    gate_decision = dict(data.get("gate_decision", {}) or {})
    timing_discipline = dict(data.get("timing_discipline", {}) or {})
    source = gate_decision or timing_discipline
    code = str(source.get("reason_code", "") or source.get("code", "") or "")
    detail = str(source.get("reason", "") or source.get("detail", "") or reason)
    values = dict(source.get("condition_values", {}) or {})
    observed = dict(values.get("observed", {}) or {})
    bits: list[str] = []
    for key in ("location", "momentum", "session"):
        value = str(observed.get(key, "") or "").strip()
        if value:
            bits.append(f"{key}={value}")
    if bits and not all(bit in detail for bit in bits):
        detail = f"{detail} ({', '.join(bits)})"
    return stage_name, code, detail


def _is_structured_execution_block(stage: dict[str, Any]) -> bool:
    if str(stage.get("stage", "") or "") != "execution":
        return False
    gate_decision = dict(dict(stage.get("data", {}) or {}).get("gate_decision", {}) or {})
    return bool(gate_decision.get("reason_code") or gate_decision.get("code") or gate_decision.get("reason"))


def _is_structured_pipeline_block(stage: dict[str, Any]) -> bool:
    data = dict(stage.get("data", {}) or {})
    if _is_structured_execution_block(stage):
        return True
    timing_discipline = dict(data.get("timing_discipline", {}) or {})
    if str(stage.get("stage", "") or "") == "conservative_filter" and timing_discipline:
        return bool(
            timing_discipline.get("passed") is False
            or timing_discipline.get("reason_code")
            or timing_discipline.get("condition_values")
        )
    return False


def _display_mode_label(status: dict[str, Any], runtime: dict[str, Any]) -> str:
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
    effective_mode = str(dict(status.get("system_health", {}) or {}).get("effective_mode", "") or "").strip().lower()
    if effective_mode == "live":
        return "Live Demo"
    if effective_mode == "paper":
        return "Paper"
    return "Live Demo" if bool(runtime.get("execute_trades", False)) else "Paper"


def _is_strategy_inventory_query(query: str) -> bool:
    q = str(query or "").lower()
    if "strateg" not in q:
        return False
    inventory_terms = (
        "what strategies",
        "which strategies",
        "list strategies",
        "all strategies",
        "strategies do you know",
        "strategies do you use",
        "only strategies",
        "only strategy",
    )
    return any(term in q for term in inventory_terms)


def _strategy_names_from_knowledge(strategies: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for item in strategies:
        name = str(item.get("name", "") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _strategy_inventory_response(
    *,
    strategies: list[dict[str, Any]],
    status: dict[str, Any],
    runtime: dict[str, Any],
    voice_mode: bool,
) -> str:
    """Summarize known strategy layers without altering executable strategy config."""
    enabled = list(runtime.get("enabled_strategies", []) or [])
    if not enabled:
        enabled = list(dict(runtime.get("settings", {}) or {}).get("enabled_strategies", []) or [])
    if not enabled:
        enabled = [
            "momentum_continuation",
            "gold_micro_impulse",
            "gold_liquidity_sweep",
        ]

    executable_available = [
        "ema_crossover",
        "rsi_mean_reversion",
        "breakout",
        "candlestick_patterns",
        "macd_trend",
        "bollinger_mean_reversion",
        "stochastic_reversal",
        "vwap_proxy_reversion",
        "pullback_trend",
        "rsi_trend_hybrid",
        "gold_micro_structure_reversion",
    ]
    concept_names = _strategy_names_from_knowledge(strategies)
    chosen = dict(status.get("chosen", {}) or {})
    chosen_strategy = str(chosen.get("strategy", "") or "").strip()
    evolved_note = "evolved strategies generated from chromosomes, shown at runtime as evolved_* or evolved_rule(...)"

    if voice_mode:
        return (
            "No. Atlas knows multiple strategy layers: active executable strategies, "
            "additional available strategy classes, human-readable setup concepts, and evolved variants."
        )

    lines = [
        "No. Atlas knows multiple strategy layers.",
        "",
        "Active executable strategies:",
        *[f"- {name}" for name in enabled],
        "",
        "Other executable strategy classes available in code:",
        *[f"- {name}" for name in executable_available if name not in enabled],
        "",
        "Strategy concepts Atlas can explain:",
        *[f"- {name}" for name in concept_names],
        "",
        "Evolved strategy family:",
        f"- {evolved_note}",
    ]
    if chosen_strategy:
        lines.extend(["", f"Current selected strategy, if any: {chosen_strategy}"])
    lines.extend(
        [
            "",
            "Important: knowing a strategy does not mean it is currently enabled, promoted, or allowed to execute. Execution still goes through filters, risk policy, and operator gates.",
        ]
    )
    return "\n".join(lines)


def synthesize(
    *,
    plan: AnswerPlan,
    intent: str,
    query: str,
    mode: str,
    status: dict[str, Any],
    runtime: dict[str, Any],
    comparison: dict[str, Any],
    self_model: dict[str, Any],
    evidence: list[EvidenceItem],
    knowledge_payload: dict[str, dict[str, Any]],
) -> str:
    action = str(status.get("action", "unknown")).upper()
    reason = str(status.get("message", "No reason available."))
    run_id = str(status.get("comparison_run_id", "") or str(dict(status.get("comparison", {}) or {}).get("run_id", "")))
    mode_label = _display_mode_label(status, runtime)
    voice_mode = str(mode or "").strip().lower() == "voice"
    external_context = dict(knowledge_payload.get("external_context", {}) or {})
    ext_freshness = dict(external_context.get("freshness", {}) or {})
    ext_items = list(external_context.get("items", []) or [])
    ext_stale = bool(ext_freshness.get("is_stale", True))

    if mode == "quick" and intent == "status":
        return f"STATUS: {action} | REASON: {reason} | MODE: {mode_label} | RUN: {run_id or '-'}"
    if voice_mode and intent == "status":
        return f"{action.title()}. {reason}. Mode: {mode_label}."

    if plan.name == "status_plan":
        why_trace = _why_trace_from_evidence(evidence)
        blocked_stage = _blocked_stage_from_why_trace(why_trace)
        if blocked_stage and _is_structured_pipeline_block(blocked_stage):
            stage_name, code, detail = _format_structured_block(blocked_stage)
            detail = detail.rstrip(".")
            code_text = f" ({code})" if code else ""
            if voice_mode:
                return f"Blocked at {stage_name.replace('_', ' ')}{code_text}: {detail}."
            return f"Blocked at {stage_name.replace('_', ' ')}{code_text}: {detail}."
        freshness = dict(status.get("freshness_runtime", {}) or {})
        age_ms = freshness.get("status_age_ms")
        freshness_line = ""
        if age_ms is not None:
            freshness_line = f"\nFreshness: {float(age_ms):.0f}ms{' (STALE)' if bool(freshness.get('is_status_stale', False)) else ''}"
        if voice_mode:
            return f"{action.title()}. {reason}. Mode: {mode_label}."
        return f"Status: {action}\nReason: {reason}\nMode: {mode_label}\nRun: {run_id or '-'}{freshness_line}"
    if plan.name == "blocker_plan":
        why_trace = _why_trace_from_evidence(evidence)
        blocked_stage = _blocked_stage_from_why_trace(why_trace)
        if blocked_stage and _is_structured_pipeline_block(blocked_stage):
            stage_name, code, detail = _format_structured_block(blocked_stage)
            detail = detail.rstrip(".")
            code_text = f" ({code})" if code else ""
            if voice_mode:
                return f"Blocked at {stage_name.replace('_', ' ')}{code_text}: {detail}."
            return (
                f"Blocked at {stage_name.replace('_', ' ')}{code_text}: {detail}.\n\n"
                "Supporting reasons:\n"
                f"- Source: structured why trace / {stage_name.replace('_', ' ')} decision.\n"
                f"- Runtime remains {mode_label.lower()}."
            )
        blockers = list(dict(knowledge_payload.get("blockers_catalog", {}) or {}).get("entries", []) or [])
        blocker = blockers[0] if blockers else {}
        blocker_name = str(blocker.get("name", "No high-confidence setup"))
        blocker_meaning = str(blocker.get("meaning", reason))
        safe_response = str(blocker.get("safe_response", "hold"))
        if voice_mode:
            return f"Main blocker: {blocker_name}."
        return (
            f"Main blocker: {blocker_name}.\n\n"
            "Supporting reasons:\n"
            f"- {blocker_meaning}\n"
                f"- Runtime remains {mode_label.lower()}.\n"
            f"- Safe response: {safe_response}."
        )
    if plan.name == "readiness_plan":
        ready = "NOT YET" if mode_label.lower() == "paper" else "OPERATOR-GATED"
        conf = float(dict(status.get("chosen", {}) or {}).get("confidence", 0.0) or 0.0)
        threshold = float(status.get("min_confidence_threshold", 0.0) or 0.0)
        if voice_mode:
            return f"Not yet. Readiness is still below the live threshold by {conf - threshold:+.3f}."
        return (
            f"Readiness: {ready}\n"
            f"Why: {reason}\n"
            f"Confidence gate: {conf:.3f} vs {threshold:.3f} ({conf - threshold:+.3f})"
        )
    if plan.name == "comparison_plan":
        cls = str(dict(comparison.get("decision_checklist", {}) or {}).get("classification", "unknown"))
        if voice_mode:
            return f"Latest comparison is active. Safest reading: {cls}."
        return (
            "Comparison: active\n"
            f"What changed: latest comparison remains active.\n"
            f"Biggest difference: checklist class is {cls}.\n"
            "Safest interpretation: continue controlled validation."
        )
    if plan.name in {"recent_vs_prior_plan", "baseline_comparison_plan", "blocker_trend_plan", "adaptation_trend_plan"}:
        ci = dict(knowledge_payload.get("comparison_intelligence", {}) or {})
        direction = str(ci.get("direction", "no clear comparison yet"))
        compared_to = str(ci.get("compared_to", "recent baseline"))
        deltas = dict(ci.get("deltas", {}) or {})
        improved = list(ci.get("improved", []) or [])
        worsened = list(ci.get("worsened", []) or [])
        safe_interp = str(ci.get("safe_interpretation", "Keep bounded paper observation."))
        if plan.name == "baseline_comparison_plan":
            if voice_mode:
                return f"Current behavior versus baseline is {direction}."
            return (
                f"Current behavior vs baseline: {direction}\n\n"
                "Improvement:\n"
                + ("\n".join(f"- {x}" for x in improved[:3]) if improved else "- none clearly improved")
                + "\n\nConcern:\n"
                + ("\n".join(f"- {x}" for x in worsened[:3]) if worsened else "- no material degradation detected")
                + "\n\nSafe interpretation:\n"
                f"- {safe_interp}\n"
                f"- Compared to: {compared_to}"
            )
        if plan.name == "blocker_trend_plan":
            blocker_delta = float(deltas.get("blocker_pressure_delta", 0.0) or 0.0)
            label = "improving" if blocker_delta > 0.03 else ("worsening" if blocker_delta < -0.03 else "flat")
            if voice_mode:
                return f"Blocker trend is {label}."
            return (
                f"Blocker trend: {label}\n\n"
                f"Reason:\n- blocker_pressure_delta={blocker_delta:+.3f}\n\n"
                f"Safe interpretation:\n- {safe_interp}"
            )
        if plan.name == "adaptation_trend_plan":
            ad_delta = float(deltas.get("adaptation_speed_delta", 0.0) or 0.0)
            st_delta = float(deltas.get("stability_delta", 0.0) or 0.0)
            label = "faster but slightly less stable" if ad_delta > 0.03 and st_delta < -0.03 else direction
            if voice_mode:
                return f"Adaptation trend is {label}."
            return (
                f"Adaptation trend: {label}\n\n"
                f"Deltas:\n- adaptation_speed_delta={ad_delta:+.3f}\n- stability_delta={st_delta:+.3f}\n\n"
                f"Safe interpretation:\n- {safe_interp}"
            )
        if voice_mode:
            return f"Recent versus prior is {direction}."
        return (
            f"Recent vs prior: {direction}\n\n"
            "What improved:\n"
            + ("\n".join(f"- {x}" for x in improved[:3]) if improved else "- no clear improvement yet")
            + "\n\nWhat remains weak:\n"
            + ("\n".join(f"- {x}" for x in worsened[:3]) if worsened else "- no major weakness shift")
            + "\n\nSafe interpretation:\n"
            f"- {safe_interp}\n"
            f"- Compared to: {compared_to}"
        )
    if plan.name == "self_knowledge_plan":
        if voice_mode:
            return "I know market and strategy concepts well within Atlas limits, but I stay grounded to configured evidence."
        return (
            f"Knowledge level: {self_model['purpose']}\n\n"
            "I understand:\n"
            + "\n".join(f"- {x}" for x in self_model["knowledge_domains"])
            + "\n\nLimitations:\n"
            + "\n".join(f"- {x}" for x in self_model["limitations"])
            + "\n\nCurrent role:\n"
            + f"- {self_model['current_role']}"
        )
    if plan.name == "self_assessment_plan":
        current_blocker = str(status.get("message", "No high-confidence trade found."))
        abstract_limits = [
            "No autonomous internet learning or uncontrolled self-modification",
            "Bounded adaptation with confidence thresholds and safety guardrails",
            "Approval-gated changes only; no hidden execution or policy bypass",
        ]
        improvement_path = [
            "Expand validated signal quality through more confirmed examples",
            "Use paper-mode bounded tuning and compare baseline vs tuned behavior",
            "Use feedback + comparison loops to improve without weakening safety constraints",
        ]
        if voice_mode:
            return (
                f"Current blocker is {current_blocker}. "
                "Main limitation is bounded, approval-gated learning. "
                "Safest improvement path is validated signals plus controlled paper tuning."
            )
        ext_line = (
            "- External context is stale or missing, so Atlas stays conservative."
            if ext_stale
            else "- External context is read-only and can only add caution/context, not mutate strategy."
        )
        return (
            f"Current blocker:\n- {current_blocker}\n\n"
            "System limitations:\n"
            + "\n".join(f"- {x}" for x in abstract_limits)
            + "\n\nLearning constraints:\n"
            + "\n".join(f"- {x}" for x in list(self_model.get("limitations", []))[:3])
            + "\n\nCurrent improvement path:\n"
            + "\n".join(f"- {x}" for x in improvement_path)
            + "\n"
            + ext_line
            + "\n\nCurrent role:\n"
            + f"- {self_model.get('current_role', 'Read-only operator assistant.')}"
        )
    if plan.name == "market_knowledge_plan":
        mk = knowledge_payload.get("market_knowledge", {})
        concepts = list(mk.get("concepts", []) or []) if isinstance(mk, dict) else []
        target = _best_named_item(
            query,
            concepts,
            name_field="name",
            fields=("definition", "summary", "signals", "implications"),
        )
        rows: list[str] = []
        if target:
            explanation = target.get("definition", "") or target.get("summary", "")
            rows.append(f"- {target.get('name', 'concept')}: {explanation}")
            if target.get("signals"):
                rows.append("- signals: " + ", ".join(str(x) for x in list(target.get("signals", []))[:3]))
            if target.get("implications"):
                rows.append("- implications: " + ", ".join(str(x) for x in list(target.get("implications", []))[:3]))
        else:
            rows = [f"- {c.get('name', 'concept')}: {c.get('definition', '')}" for c in concepts[:3]]
        if not rows:
            rows = ["- Trend, volatility, liquidity, momentum, and risk management are understood operationally."]
        if voice_mode:
            return "Yes. I understand trend, volatility, liquidity, momentum, and risk basics."
        return "Market knowledge summary:\n" + "\n".join(rows)
    if plan.name == "strategy_knowledge_plan":
        sk = dict(knowledge_payload.get("strategy_knowledge", {}) or {})
        strategies = list(sk.get("strategies", []) or [])
        if _is_strategy_inventory_query(query):
            return _strategy_inventory_response(
                strategies=strategies,
                status=status,
                runtime=runtime,
                voice_mode=voice_mode,
            )
        target = _best_named_item(
            query,
            strategies,
            name_field="name",
            fields=("description", "notes", "conditions", "when_it_works", "when_to_avoid", "risks"),
        )
        target = target or (strategies[0] if strategies else {})
        if voice_mode:
            return f"Primary strategy type: {target.get('name', 'continuation')} with confirmation and noise controls."
        description = target.get("description") or target.get("notes") or "No strategy description available."
        conditions = list(target.get("conditions", []) or target.get("when_it_works", []) or [])
        risks = list(target.get("risks", []) or target.get("when_to_avoid", []) or [])
        good_example = str(target.get("good_example", "") or "")
        bad_example = str(target.get("bad_example", "") or "")
        teaching_prompt = str(target.get("operator_teaching_prompt", "") or "")
        return (
            f"Strategy: {target.get('name', 'unknown')}\n"
            f"Description: {description}\n"
            "Conditions:\n"
            + "\n".join(f"- {x}" for x in conditions[:4])
            + "\nRisks:\n"
            + "\n".join(f"- {x}" for x in risks[:4])
            + (f"\nGood example:\n- {good_example}" if good_example else "")
            + (f"\nBad example:\n- {bad_example}" if bad_example else "")
            + (f"\nTeaching note:\n- {teaching_prompt}" if teaching_prompt else "")
        )
    if plan.name == "risk_policy_plan":
        rules = list(dict(knowledge_payload.get("risk_policy", {}) or {}).get("rules", []) or [])
        if voice_mode:
            return "Risk rules prioritize high-confidence setups, low-noise conditions, and hold over forced entries."
        return "Risk rules:\n" + "\n".join(f"- {x}" for x in rules[:6])
    if plan.name == "glossary_plan":
        terms = list(dict(knowledge_payload.get("glossary", {}) or {}).get("terms", []) or [])
        match = _best_named_item(query, terms, name_field="term", fields=("definition",)) or {}
        if voice_mode:
            return f"{match.get('term', 'term')} means {match.get('definition', 'definition unavailable')}."
        return f"{match.get('term', 'term')}: {match.get('definition', 'definition unavailable')}."
    if plan.name == "capabilities_plan":
        if intent == "limitations":
            if voice_mode:
                return "I do not invent facts, I do not place trades from Ask Atlas, and I cannot guarantee outcomes."
            return "Limitations:\n" + "\n".join(f"- {x}" for x in self_model["limitations"])
        if voice_mode:
            return "I can explain status, blockers, readiness, comparison, and safe next steps using grounded evidence."
        return (
            "Capabilities:\n"
            + "\n".join(f"- {x}" for x in self_model["capabilities"])
            + f"\n\nCurrent role:\n- {self_model['current_role']}"
        )
    if plan.name == "worker_neurons_plan":
        wn = dict(knowledge_payload.get("worker_neurons", {}) or {})
        workers = list(wn.get("workers", []) or [])
        contract = list(wn.get("brain_contract", []) or [])
        boundary = list(wn.get("safety_boundary", []) or [])
        if voice_mode:
            return "Yes. Workers should behave like evidence neurons: they sense and report, while the Brain interprets and execution gates decide."
        return (
            "Yes. The better model is workers as evidence neurons, not separate decision-makers.\n\n"
            "Worker roles:\n"
            + ("\n".join(f"- {x.get('name', 'worker')}: {x.get('role', '')}" for x in workers[:6]) if workers else "- Workers report status, signals, monitoring, and execution evidence.")
            + "\n\nBrain contract:\n"
            + ("\n".join(f"- {x}" for x in contract[:6]) if contract else "- Workers feed structured evidence into Brain v2.\n- Brain interprets evidence.\n- ExecutionGate remains the only execution authority.")
            + "\n\nSafety boundary:\n"
            + ("\n".join(f"- {x}" for x in boundary[:5]) if boundary else "- No worker can trade independently.\n- No hidden strategy, risk, or execution-policy drift.")
        )
    if plan.name == "cognitive_core_plan":
        core = dict(knowledge_payload.get("cognitive_core", {}) or {})
        world = dict(core.get("world_model", {}) or {})
        decision = dict(core.get("decision_understanding", {}) or {})
        memory = dict(core.get("trade_memory", {}) or {})
        pattern = dict(core.get("pattern_memory", {}) or {})
        learning = dict(core.get("learning_state", {}) or {})
        critic = dict(core.get("decision_critic", {}) or {})
        safety = dict(core.get("safety_contract", {}) or {})
        invention = dict(core.get("strategy_invention", {}) or {})
        ai_trace = dict(knowledge_payload.get("ai_decision_trace", {}) or {})
        ai_rec = dict(ai_trace.get("recommendation", {}) or {})
        ai_blockers = dict(ai_trace.get("blocker_analysis", {}) or {})
        if voice_mode:
            return (
                "Yes. Atlas has one coordinated cognitive core, with specialist modules underneath. "
                "It can understand, learn, and propose, but execution still stays gate-controlled."
            )
        flags = list(critic.get("flags", []) or [])
        lessons = list(pattern.get("latest_lessons", []) or [])
        invented = list(invention.get("candidates", []) or [])
        invented_lines = []
        for row in invented[:3]:
            invented_lines.append(f"- {row.get('archetype', 'unknown')}: {row.get('thesis', 'proposal awaiting validation')}")
        return (
            "Yes. Atlas now presents as one coordinated mind, but it remains modular underneath for safety. "
            "The cognitive core joins market state, decision lanes, trade memory, lessons, strategy invention, and safety boundaries.\n\n"
            "Brain model:\n"
            "- One coordinated cognitive core at the top\n"
            "- Specialist modules underneath for observation, classification, memory, explanation, caution, and proposals\n"
            "- Risk and execution gates remain separate authority\n\n"
            "World model:\n"
            f"- Session: {world.get('session', 'unknown')}\n"
            f"- Regime: {world.get('regime', 'unknown')}\n"
            f"- Structure: {world.get('structure', 'unknown')}\n"
            f"- Momentum: {world.get('momentum', 'unknown')}\n"
            f"- Location: {world.get('location', 'unknown')}\n"
            f"- Current blocker: {world.get('blocker', 'none')}\n\n"
            "Decision understanding:\n"
            f"- Market bias: {decision.get('market_bias', 'unknown')}\n"
            f"- Signal direction: {decision.get('signal_direction', 'neutral')}\n"
            f"- Higher-timeframe bias: {decision.get('higher_timeframe_bias', 'unknown')}\n"
            f"- Final trade bias: {decision.get('final_trade_bias', 'neutral')}\n"
            f"- Setup readiness: {decision.get('setup_readiness', 'unknown')}\n"
            f"- Execution permission: {decision.get('execution_permission', 'not_allowed')}\n\n"
            "Learning memory:\n"
            f"- Recent sample size: {int(memory.get('recent_sample_size', 0) or 0)}\n"
            f"- Closed sample size: {int(memory.get('closed_sample_size', 0) or 0)}\n"
            f"- Recent win rate: {float(memory.get('win_rate', 0.0) or 0.0) * 100:.1f}%\n"
            f"- Latest lessons stored: {int(pattern.get('lesson_count', 0) or 0)}\n\n"
            "Decision critic:\n"
            + ("\n".join(f"- {x}" for x in flags[:5]) if flags else "- no major critic flags")
            + "\n\nLearning state:\n"
            f"- Mode: {str(learning.get('active_learning_mode', 'unknown')).replace('_', '-')}\n"
            f"- Evidence strength: {float(learning.get('evidence_strength_score', 0.0) or 0.0):.3f}\n"
            f"- Stability score: {float(learning.get('stability_score', 0.0) or 0.0):.3f}\n"
            f"- Overreaction score: {float(learning.get('overreaction_score', 0.0) or 0.0):.3f}\n\n"
            "AI decision core:\n"
            f"- Action: {ai_rec.get('action', 'hold_and_observe')}\n"
            f"- Confidence: {float(ai_rec.get('confidence', 0.0) or 0.0):.3f}\n"
            f"- Reason: {ai_rec.get('reason', 'No decision trace available.')}\n"
            f"- Primary blocker: {ai_blockers.get('primary_blocker', 'none')}\n\n"
            "Current lessons it should remember:\n"
            + ("\n".join(f"- {x}" for x in lessons[:3]) if lessons else "- no structured lessons loaded")
            + "\n\nStrategy invention:\n"
            f"- Mode: {invention.get('mode', 'proposal_only')}\n"
            f"- Current AI-designed candidates: {int(invention.get('candidate_count', 0) or 0)}\n"
            + ("\n".join(invented_lines) if invented_lines else "- no current proposals generated")
            + "\n"
            f"- Promotion path: {', '.join(str(x) for x in list(invention.get('promotion_path', []) or [])[:6]) or 'validation, backtest, shadow, demo, approval, execution gate'}"
            + "\n\nSafety contract:\n"
            f"- Can explain: {bool(safety.get('can_explain', True))}\n"
            f"- Can learn from evidence: {bool(safety.get('can_learn_from_evidence', True))}\n"
            f"- Can propose changes: {bool(safety.get('can_propose_changes', True))}\n"
            f"- Can design strategy proposals: {bool(safety.get('can_design_strategy_proposals', False))}\n"
            f"- Can bypass ExecutionGate: {bool(safety.get('can_bypass_execution_gate', False))}\n"
            f"- {safety.get('policy_note', 'No hidden strategy, risk, or execution-policy drift.')}"
        )
    if plan.name == "ai_decision_review_plan":
        trace = dict(knowledge_payload.get("ai_decision_trace", {}) or {})
        rec = dict(trace.get("recommendation", {}) or {})
        blockers = dict(trace.get("blocker_analysis", {}) or {})
        invention = dict(trace.get("strategy_invention", {}) or {})
        recent = list(knowledge_payload.get("ai_decision_journal_recent", []) or [])
        reviews = list(knowledge_payload.get("ai_decision_reviews_recent", []) or [])
        preferences = dict(knowledge_payload.get("ai_strategy_preferences", {}) or {})
        promotion = dict(knowledge_payload.get("ai_promotion_intelligence", {}) or {})
        latest = dict(recent[-1] if recent else {})
        if latest:
            trace = dict(latest.get("trace", {}) or trace)
            rec = dict(trace.get("recommendation", {}) or rec)
            blockers = dict(trace.get("blocker_analysis", {}) or blockers)
            invention = dict(trace.get("strategy_invention", {}) or invention)
        if voice_mode:
            return str(rec.get("reason", "Latest AI decision is not available."))
        archetypes = list(invention.get("candidate_archetypes", []) or [])
        latest_review = dict(reviews[-1] if reviews else {})
        ranking = list(preferences.get("ranking", []) or [])
        top_promo = dict(promotion.get("top_recommendation", {}) or {})
        pref_lines = [
            f"- {dict(row).get('archetype', 'unknown')}: {float(dict(row).get('score', 0.0) or 0.0):.2f}"
            for row in ranking[:3]
        ]
        return (
            f"Latest AI decision: {rec.get('action', 'unknown')}.\n\n"
            f"Reason:\n- {rec.get('reason', 'No recorded reason.')}\n\n"
            f"Confidence:\n- {float(rec.get('confidence', 0.0) or 0.0):.3f}\n\n"
            f"Primary blocker:\n- {blockers.get('primary_blocker', 'none')}\n\n"
            "Outcome review:\n"
            f"- Outcome: {latest_review.get('outcome', 'pending')}\n"
            f"- Blocker quality: {latest_review.get('blocker_quality', 'pending')}\n"
            f"- Lesson: {latest_review.get('lesson', 'No review recorded yet.')}\n\n"
            "Strategy invention:\n"
            f"- Candidates: {int(invention.get('candidate_count', 0) or 0)}\n"
            + ("\n".join(f"- {x}" for x in archetypes[:4]) if archetypes else "- no archetypes recorded")
            + "\n\nStrategy preference memory:\n"
            + ("\n".join(pref_lines) if pref_lines else "- no preference scores recorded yet")
            + "\n\nPromotion intelligence:\n"
            f"- Top UID: {top_promo.get('uid', 'none')}\n"
            f"- Recommendation: {top_promo.get('recommendation', 'none recorded')}\n"
            f"- Confidence: {float(top_promo.get('confidence', 0.0) or 0.0):.3f}"
            + "\n\nSafety:\n- This is journaled reasoning only. It cannot execute trades or bypass ExecutionGate."
        )
    if plan.name == "help_plan":
        if voice_mode:
            return "I can answer status, blockers, readiness, confidence, comparison, self-model, and safe experiment questions."
        return (
            "I can answer: status, blockers, readiness, comparison, confidence, next safe experiment, "
            "self-knowledge, capabilities, limitations, and market/strategy knowledge.\n"
            "Example: 'What is holding you back?'"
        )
    if plan.name == "next_safe_experiment_plan":
        if voice_mode:
            return "Next safe step: review blocked candidates in paper mode without changing live execution."
        return (
            "Next safe step: keep paper mode and run one bounded comparison review.\n"
            "Scope: inspect blocked candidates and confidence-gap reasons in the current run.\n"
            "Safety: no live execution and no threshold mutation without explicit operator approval."
        )
    if plan.name in {"safe_next_step_plan", "market_recommendation_plan", "blocker_resolution_plan", "experiment_suggestion_plan"}:
        rec = dict(knowledge_payload.get("recommendation_result", {}) or {})
        items = list(rec.get("items", []) or [])
        top = items[0] if items else {}
        top_rec = str(top.get("recommendation", "Hold and wait for confirmation."))
        top_reason = str(top.get("reason", "Conservative recommendation due to current uncertainty."))
        watch = list(top.get("supporting_evidence", []) or [])
        constraints = str(top.get("constraint", "Recommendation only. No execution changes."))
        if plan.name == "blocker_resolution_plan":
            br = next((x for x in items if str(x.get("category")) == "blocker_resolution"), top)
            if voice_mode:
                return str(br.get("recommendation", top_rec))
            return (
                f"Main recommendation: {br.get('recommendation', top_rec)}\n\n"
                f"Why this matters:\n- {br.get('reason', top_reason)}\n\n"
                "Safe follow-up:\n"
                f"- {br.get('constraint', constraints)}"
            )
        if plan.name == "experiment_suggestion_plan":
            ex = next((x for x in items if str(x.get("category")) == "experiment_suggestion"), top)
            if voice_mode:
                return str(ex.get("recommendation", "Paper-test a bounded adjustment only."))
            return (
                f"Safe experiment: {ex.get('recommendation', 'Paper-test a bounded adjustment only.')}\n\n"
                "Constraint:\n"
                f"- {ex.get('constraint', 'Paper mode only. No live logic change.')}"
            )
        if plan.name == "market_recommendation_plan":
            if "why are you recommending caution" in str(query or "").lower():
                explain = next((x for x in items if str(x.get("category")) == "explanation_assist"), top)
                if voice_mode:
                    return str(explain.get("recommendation", top_rec))
                return (
                    f"Recommendation: {explain.get('recommendation', top_rec)}\n\n"
                    "Why:\n"
                    f"- {explain.get('reason', top_reason)}\n\n"
                    "Next focus:\n"
                    + "\n".join(f"- {x}" for x in list(explain.get("supporting_evidence", []) or [])[:3])
                )
            mon = next((x for x in items if str(x.get("category")) in {"monitoring_priority", "action_posture"}), top)
            if voice_mode:
                return str(mon.get("recommendation", top_rec))
            return (
                f"Recommendation: {mon.get('recommendation', top_rec)}\n\n"
                "Why:\n"
                f"- {mon.get('reason', top_reason)}\n\n"
                "Next focus:\n"
                + "\n".join(f"- {x}" for x in list(mon.get("supporting_evidence", []) or [])[:3])
            )
        if "why are you recommending caution" in str(query or "").lower():
            explain = next((x for x in items if str(x.get("category")) == "explanation_assist"), top)
            if voice_mode:
                return str(explain.get("recommendation", top_rec))
            return (
                f"Recommendation: {explain.get('recommendation', top_rec)}\n\n"
                "Why:\n"
                f"- {explain.get('reason', top_reason)}\n\n"
                "Next focus:\n"
                + "\n".join(f"- {x}" for x in list(explain.get("supporting_evidence", []) or [])[:3])
            )
        if voice_mode:
            return top_rec
        no_high_conf = "no high-confidence trade found" in str(status.get("message", "")).strip().lower()
        ext_hint = ""
        if ext_items:
            tag = str(dict(ext_items[0]).get("tag", "") or "").strip()
            if tag:
                ext_hint = f"\n- External context tag: {tag} (read-only)"
        elif ext_stale:
            ext_hint = "\n- External context unavailable/stale; no external caution adjustment applied."
        if str(top.get("category", "")) == "operator_review_gate":
            return (
                f"Recommended next step: {top_rec}\n\n"
                "Reason:\n"
                f"- {top_reason}\n\n"
                "What to review:\n"
                + "\n".join(f"- {x}" for x in watch[:10])
                + ext_hint
                + "\n\nSafety:\n"
                + f"- {constraints}"
            )
        if no_high_conf:
            conf_diag = dict(status.get("confidence_diagnostics", {}) or {})
            drag = list(conf_diag.get("dominant_drag_reasons", []) or [])
            diag_line = (
                f"\n- Main drag factors: {', '.join(drag[:3])}"
                if drag
                else ""
            )
            if str(top.get("category", "")) == "operator_review_gate":
                return (
                    f"Recommended next step: {top_rec}\n\n"
                    "Reason:\n"
                    f"- {top_reason}\n\n"
                    "What to review:\n"
                    + "\n".join(f"- {x}" for x in watch[:8])
                    + ext_hint
                    + "\n\nSafety:\n"
                    + f"- {constraints}"
                )
            if str(top.get("category", "")) == "missed_entry_repair":
                return (
                    f"Recommended next step: {top_rec}\n\n"
                    "Reason:\n"
                    f"- {top_reason}\n\n"
                    "What to review:\n"
                    + "\n".join(f"- {x}" for x in watch[:4])
                    + ext_hint
                    + "\n\nSafety:\n"
                    + f"- {constraints}"
                )
            return (
                "No high-confidence trade found. Recommendation: hold.\n\n"
                "Reason:\n"
                f"- {top_reason}\n\n"
                "What to watch:\n"
                + "\n".join(f"- {x}" for x in watch[:3])
                + diag_line
                + ext_hint
                + "\n\nSafety:\n"
                + f"- {constraints}"
            )
        return (
            f"Recommended next step: {top_rec}\n\n"
            "Reason:\n"
            f"- {top_reason}\n\n"
            "What to watch:\n"
            + "\n".join(f"- {x}" for x in watch[:3])
            + "\n\nSafety:\n"
            + f"- {constraints}"
        )
    if plan.name in {
        "performance_review_plan",
        "adaptation_review_plan",
        "recommendation_review_plan",
        "stability_review_plan",
        "improvement_assessment_plan",
        "weakness_analysis_plan",
    }:
        pf = dict(knowledge_payload.get("performance_feedback", {}) or {})
        scores = dict(pf.get("scores", {}) or {})
        items = list(pf.get("items", []) or [])
        by_cat = {str(i.get("category")): i for i in items}
        adaptation_item = by_cat.get("adaptation_review", {})
        recommendation_item = by_cat.get("recommendation_review", {})
        stability_item = by_cat.get("stability_review", {})
        direction_item = by_cat.get("recent_performance_direction", {})
        biggest_weakness = str(pf.get("biggest_weakness", "no dominant weakness detected"))
        next_focus = str(pf.get("next_safe_focus", "Continue cautious monitoring and paper validation."))
        if plan.name == "adaptation_review_plan":
            regime_item = by_cat.get("regime_review", {})
            if voice_mode:
                return f"Adaptation review: {adaptation_item.get('label', 'partially effective')}."
            return (
                f"Adaptation review: {adaptation_item.get('label', 'partially effective')}\n\n"
                "What went well:\n"
                f"- {adaptation_item.get('summary', 'Adaptation remained bounded.')}\n\n"
                "Concern:\n"
                f"- confidence update quality score={float(scores.get('confidence_update_quality_score', 0.0)):.3f}\n\n"
                f"Recommendation review: {recommendation_item.get('label', 'appropriate')}\n"
                f"Stability review: {stability_item.get('label', 'stable')}\n"
                f"Regime review: {regime_item.get('label', 'stable')}\n"
                f"Recent direction: {pf.get('direction', 'no clear change')}\n"
                f"Main weakness: {biggest_weakness}\n\n"
                "Next safe focus:\n"
                f"- {next_focus}"
            )
        if plan.name == "recommendation_review_plan":
            if voice_mode:
                return f"Recommendation review: {recommendation_item.get('label', 'appropriate')}."
            return (
                f"Recommendation review: {recommendation_item.get('label', 'appropriate')}\n\n"
                "Why:\n"
                f"- {recommendation_item.get('summary', 'Recommendation posture was bounded and safety-oriented.')}"
            )
        if plan.name == "stability_review_plan":
            if voice_mode:
                return f"Stability review: {stability_item.get('label', 'stable')}."
            return (
                f"Stability review: {stability_item.get('label', 'stable')}\n\n"
                "Evidence:\n"
                f"- {stability_item.get('summary', 'Stability remained acceptable.')}\n"
                f"- stability_quality_score={float(scores.get('stability_quality_score', 0.0)):.3f}\n"
                f"- overreaction_score={float(dict(knowledge_payload.get('adaptation_summary', {}) or {}).get('overreaction_score', 0.0)):.3f}"
            )
        if plan.name == "improvement_assessment_plan":
            if voice_mode:
                return f"Recent direction: {pf.get('direction', 'no clear change')}."
            ci = dict(knowledge_payload.get("comparison_intelligence", {}) or {})
            return (
                f"Recent direction: {pf.get('direction', 'no clear change')}\n\n"
                "Why:\n"
                f"- {direction_item.get('summary', 'Recent behavior shows no major directional shift.')}\n"
                f"- recent_improvement_score={float(scores.get('recent_improvement_score', 0.0)):.3f}\n"
                f"- Compared to: {ci.get('compared_to', 'recent baseline')}"
            )
        if plan.name == "weakness_analysis_plan":
            if voice_mode:
                return f"Main weakness: {biggest_weakness}."
            return (
                f"Main weakness: {biggest_weakness}\n\n"
                "Why this matters:\n"
                f"- {adaptation_item.get('summary', 'Current performance has an identifiable weakness that may reduce decision quality.')}\n\n"
                "Safe next step:\n"
                f"- {next_focus}"
            )
        if voice_mode:
            return f"Performance review: {pf.get('direction', 'no clear change')}."
        return (
            f"Performance review: {pf.get('direction', 'no clear change')}\n\n"
            "Scores:\n"
            f"- adaptation_effectiveness={float(scores.get('adaptation_effectiveness_score', 0.0)):.3f}\n"
            f"- recommendation_quality={float(scores.get('recommendation_quality_score', 0.0)):.3f}\n"
            f"- regime_interpretation={float(scores.get('regime_interpretation_score', 0.0)):.3f}\n"
            f"- confidence_update_quality={float(scores.get('confidence_update_quality_score', 0.0)):.3f}\n"
            f"- stability_quality={float(scores.get('stability_quality_score', 0.0)):.3f}\n"
            f"- recent_improvement={float(scores.get('recent_improvement_score', 0.0)):.3f}\n\n"
            f"Biggest weakness: {biggest_weakness}\n"
            f"Next safe focus: {next_focus}"
        )
    if plan.name in {"tuning_recommendation_plan", "paper_experiment_plan", "tuning_comparison_plan"}:
        pt = dict(knowledge_payload.get("paper_tuning", {}) or {})
        candidates = list(pt.get("candidates", []) or [])
        comparisons = list(pt.get("comparisons", []) or [])
        top = candidates[0] if candidates else {}
        top_target = str(top.get("target", "no_target"))
        top_cat = str(top.get("category", "comparison_only_experiment"))
        top_delta = float(top.get("proposed_delta", 0.0) or 0.0)
        top_rationale = str(top.get("rationale", "No safe tuning candidate currently."))
        top_effect = str(top.get("expected_effect", "Collect more paper evidence first."))
        top_constraint = ", ".join(str(x) for x in list(top.get("safety_constraints", []) or [])[:3]) or "paper-mode only, no live promotion"
        if plan.name == "paper_experiment_plan":
            if voice_mode:
                return f"Safe paper experiment: test {top_target} with bounded delta {top_delta:+.3f}."
            return (
                f"Safe paper experiment: compare baseline vs {top_target} adjustment ({top_delta:+.3f})\n\n"
                "Success criteria:\n"
                "- improved setup acceptance or blocker reduction\n"
                "- no material stability degradation\n"
                "- no overreaction increase"
            )
        if plan.name == "tuning_comparison_plan":
            cmp = comparisons[0] if comparisons else {}
            if voice_mode:
                return f"Paper tuning result: {cmp.get('outcome', 'inconclusive')}."
            return (
                f"Paper tuning result: {cmp.get('outcome', 'inconclusive')}\n\n"
                "Observed effect:\n"
                f"- baseline: {cmp.get('baseline_summary', 'n/a')}\n"
                f"- tuned: {cmp.get('tuned_summary', 'n/a')}\n\n"
                "Recommendation:\n"
                f"- {cmp.get('recommendation', 'Do not promote; gather more paper evidence.')}"
            )
        if voice_mode:
            return f"Recommended paper-mode tuning: {top_cat} on {top_target}."
        return (
            f"Recommended paper-mode tuning: {top_cat} on {top_target} ({top_delta:+.3f})\n\n"
            "Why:\n"
            f"- {top_rationale}\n\n"
            "Constraint:\n"
            f"- {top_constraint}\n"
            "- paper-mode only\n"
            "- no live promotion\n\n"
            "Expected effect:\n"
            f"- {top_effect}"
        )
    if plan.name == "adaptation_plan":
        ad = dict(knowledge_payload.get("adaptation_summary", {}) or {})
        changes = list(ad.get("observed_changes", []) or ["no strong chart-condition change detected"])
        mode_name = str(ad.get("active_learning_mode", "slow_path")).replace("_", "-")
        reasons = list(ad.get("learning_mode_reasons", []) or ["evidence profile is mixed"])
        guardrails = list(ad.get("guardrails_triggered", []) or [])
        ql = str(query or "").lower()
        if voice_mode and ("how fast are you learning" in ql or "learning too slowly" in ql):
            return (
                f"Learning mode is {mode_name}. "
                f"Evidence strength is {float(ad.get('evidence_strength_score', 0.0)):.2f}."
            )
        if voice_mode and ("regime change" in ql or "detect this regime change" in ql):
            return f"Regime-change detection took about {float(ad.get('regime_shift_detection_delay', 0.0)):.1f} cycles."
        if voice_mode and ("adjust confidence" in ql or "confidence" in ql):
            return (
                "Confidence adjustment latency is about "
                f"{float(ad.get('detection_to_confidence_update_latency', 0.0)):.1f} cycles."
            )
        if voice_mode:
            return f"Adaptation is {ad.get('speed_label', 'moderate')} and {ad.get('stability', 'stable')}. Main change: {changes[0]}."
        return (
            f"Learning mode: {mode_name}\n\n"
            "Reason:\n"
            + "\n".join(f"- {r}" for r in reasons[:3])
            + "\n\nEffect:\n"
            + f"- Confidence update speed (effective): {float(ad.get('confidence_adjustment_speed_effective', 0.0)):.3f}\n"
            + f"- Bounded confidence delta per update: {float(ad.get('bounded_confidence_delta', 0.0)):+.3f}\n"
            + f"- Hold-bias adjustment speed: {float(ad.get('hold_bias_adjustment_speed', 0.0)):.3f}\n"
            + "- No core strategy rule changed automatically.\n\n"
            + f"Adaptation status: {ad.get('adaptation_status', 'adjusting cautiously')}\n\n"
            "Observed changes:\n"
            + "\n".join(f"- {c}" for c in changes[:3])
            + "\n\nEffect on Atlas:\n"
            + f"- Confidence adjustment speed: {float(ad.get('confidence_adjustment_speed', 0.0)):.3f}\n"
            + f"- Setup quality change: {float(ad.get('setup_quality_change', 0.0)):+.3f}\n"
            + f"- Filter rejection rate change: {float(ad.get('filter_rejection_rate_change', 0.0)):.3f}\n"
            + "\nLearning speed:\n"
            + f"- Evidence strength score: {float(ad.get('evidence_strength_score', 0.0)):.3f}\n"
            + f"- Evidence to detection latency: {float(ad.get('evidence_to_detection_latency', 0.0)):.2f}\n"
            + f"- Detection to confidence update latency: {float(ad.get('detection_to_confidence_update_latency', 0.0)):.2f}\n"
            + f"- Detection to behavior change latency: {float(ad.get('detection_to_behavior_change_latency', 0.0)):.2f}\n"
            + f"- Trades to stable adjustment: {float(ad.get('trades_to_stable_adjustment', 0.0)):.2f}\n"
            + f"- Regime learning speed: {float(ad.get('regime_learning_speed', 0.0)):.3f}\n"
            + f"- Setup decay detection speed: {float(ad.get('setup_decay_detection_speed', 0.0)):.3f}\n"
            + f"- Recovery learning speed: {float(ad.get('recovery_learning_speed', 0.0)):.3f}\n"
            + "\nAssessment:\n"
            + f"- adaptation speed: {ad.get('speed_label', 'moderate')}\n"
            + f"- confidence in regime shift: {float(ad.get('regime_shift_confidence', 0.0)):.3f}\n"
            + f"- stability: {ad.get('stability', 'stable')}\n"
            + f"- stability score: {float(ad.get('stability_score', 0.0)):.3f}\n"
            + f"- overreaction risk: {float(ad.get('overreaction_score', 0.0)):.3f}\n"
            + ("\nGuardrails:\n" + "\n".join(f"- {g}" for g in guardrails) if guardrails else "\nGuardrails:\n- none triggered")
            + f"\nSafety:\n- {ad.get('safety_note', 'Observation only.')}"
            + f"\n- {ad.get('learning_speed_note', 'Learning speed is measured, not self-modified.')}"
        )

    top = evidence[0].content if evidence else reason
    top_text = str(top or "").strip()
    if voice_mode:
        if not top_text or top_text.startswith("{") or len(top_text) > 220:
            return f"{action.title()}. {reason}"
        return top_text.splitlines()[0][:220]
    if not top_text or top_text.startswith("{"):
        top_text = reason
    return f"Closest likely interpretation answered.\n{top_text}"
