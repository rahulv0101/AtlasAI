from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return int(default)


def _first_sentence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    for marker in (". ", "! ", "? "):
        if marker in raw:
            return raw.split(marker, 1)[0].strip() + marker.strip()
    return raw


@dataclass
class BrainSnapshot:
    status: dict[str, Any]
    events: list[dict[str, Any]]
    runtime: dict[str, Any]
    learning_state: dict[str, Any]
    comparison: dict[str, Any]


@dataclass
class BrainMemory:
    recent_questions: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_conclusions: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    recent_turns: deque["BrainTurn"] = field(default_factory=lambda: deque(maxlen=10))
    repeated_blockers: Counter[str] = field(default_factory=Counter)

    def remember_question(self, query: str) -> None:
        text = " ".join(str(query or "").split())
        if text:
            self.recent_questions.append(text)

    def remember_conclusion(self, conclusion: str) -> None:
        text = " ".join(str(conclusion or "").split())
        if text:
            self.recent_conclusions.append(text)

    def remember_turn(self, turn: "BrainTurn") -> None:
        self.recent_turns.append(turn)

    def recent_relevant_turns(self, limit: int = 5) -> list["BrainTurn"]:
        relevant = [turn for turn in self.recent_turns if bool(getattr(turn, "follow_up_relevance", True))]
        if not relevant:
            relevant = list(self.recent_turns)
        return relevant[-limit:]


@dataclass(frozen=True)
class BrainTurn:
    query: str
    intent: str
    reasoning_mode: str
    result: str
    subject: str
    main_answer: str = ""
    follow_up_relevance: bool = True


@dataclass(frozen=True)
class BrainSelfModel:
    identity: str
    current_state: str
    capabilities: list[str]
    limitations: list[str]
    readiness: str
    confidence_in_self_assessment: int


@dataclass(frozen=True)
class BrainOutput:
    kind: str
    summary: str
    details: dict[str, Any]
    evidence: list[str]


class AtlasBrain:
    """Read-only reasoning layer for Atlas operator analysis."""

    def __init__(self, snapshot: BrainSnapshot, memory: BrainMemory | None = None) -> None:
        self.snapshot = snapshot
        self.memory = memory or BrainMemory()
        self.status = dict(snapshot.status or {})
        self.events = list(snapshot.events or [])
        self.runtime = dict(snapshot.runtime or {})
        self.learning_state = dict(snapshot.learning_state or {})
        self.comparison = dict(snapshot.comparison or {})

    def remember_question(self, query: str) -> None:
        self.memory.remember_question(query)

    def remember_conclusion(self, conclusion: str) -> None:
        self.memory.remember_conclusion(conclusion)

    def outcome_intelligence(self) -> dict[str, Any]:
        return dict(dict(self.learning_state.get("state", {}) or {}).get("trade_outcome_intelligence", {}) or {})

    def top_block_reasons(self) -> list[tuple[str, int]]:
        reasons: Counter[str] = Counter()
        for event in self.events:
            event_type = str(event.get("type", "") or "").lower()
            if event_type not in {"risk", "blocked", "paper_filtered"}:
                continue
            message = str(event.get("message", "") or "").strip()
            if not message:
                continue
            reason = _first_sentence(message)
            reasons[reason] += 1
            self.memory.repeated_blockers[reason] += 1
        return reasons.most_common(5)

    def favorable_unfavorable(self) -> tuple[list[str], list[str]]:
        context = dict(self.status.get("context_snapshot", {}) or {})
        regime = dict(self.status.get("regime", {}) or {})
        chosen = dict(self.status.get("chosen", {}) or {})
        higher_tf_bias = str(self.status.get("higher_tf_bias", "unknown") or "unknown")
        favorable: list[str] = []
        unfavorable: list[str] = []
        if str(context.get("spread_state", "")) == "tight":
            favorable.append("spread is tight")
        if str(context.get("liquidity_state", "")) == "normal":
            favorable.append("liquidity looks normal")
        if str(regime.get("name", "")) == "trend":
            favorable.append("the market still has directional structure")
        if higher_tf_bias in {"buy", "sell"}:
            favorable.append(f"higher timeframe bias is {higher_tf_bias}")
        if str(context.get("spread_state", "")) == "wide":
            unfavorable.append("spread is wide")
        if str(context.get("liquidity_state", "")) in {"poor", "below_normal"}:
            unfavorable.append("liquidity is weak")
        threshold = self.status.get("min_confidence_threshold")
        confidence = _safe_float(chosen.get("confidence", 0.0))
        if threshold is not None and confidence < _safe_float(threshold):
            unfavorable.append("confidence is below the current execution threshold")
        top_blocks = self.top_block_reasons()
        if top_blocks:
            unfavorable.append(f"recent blocks are led by {top_blocks[0][0].rstrip('.')}")
        return favorable, unfavorable

    def condition_direction(self) -> str:
        context = dict(self.status.get("context_snapshot", {}) or {})
        favorable, unfavorable = self.favorable_unfavorable()
        if favorable and not unfavorable:
            return "conditions look favorable overall."
        if unfavorable and not favorable:
            return "conditions look worse than favorable right now."
        if str(context.get("spread_state", "")) == "tight" and str(context.get("liquidity_state", "")) == "normal":
            return "conditions are improving but not clean enough yet."
        return "conditions are mixed or still unclear."

    def blocker_rank(self) -> BrainOutput:
        ranking: Counter[str] = Counter()
        message = _first_sentence(str(self.status.get("message", "") or "").strip())
        if message:
            ranking[message] += 2
        for reason, count in self.top_block_reasons():
            ranking[reason] += count
        ranked = ranking.most_common(3)
        dominant = bool(ranked) and (len(ranked) == 1 or ranked[0][1] >= max(1, ranked[1][1] * 2))
        summary = ranked[0][0] if ranked else "No dominant blocker identified yet."
        return BrainOutput(
            kind="blocker_rank",
            summary=summary,
            details={
                "ranking": ranked,
                "dominant": dominant,
                "stacked": bool(ranked) and not dominant,
            },
            evidence=["current status", "recent blocking events", "session-local repeated blocker memory"],
        )

    def setup_assessment(self) -> BrainOutput:
        context = dict(self.status.get("context_snapshot", {}) or {})
        regime = dict(self.status.get("regime", {}) or {})
        chosen = dict(self.status.get("chosen", {}) or {})
        best_actionable = dict(self.status.get("best_actionable_candidate", {}) or {})
        decision = dict(self.status.get("decision_explanation", {}) or {})
        shadow = dict(self.status.get("shadow_advice", {}) or {})
        candidate = chosen or best_actionable
        side = str(candidate.get("side", "flat") or "flat")
        higher_tf_bias = str(self.status.get("higher_tf_bias", candidate.get("higher_tf_bias", "unknown")) or "unknown")
        threshold = self.status.get("min_confidence_threshold")
        confidence = _safe_float(candidate.get("confidence", 0.0))
        issues: list[str] = []
        positives: list[str] = []
        missing: list[str] = []
        issue_types: list[str] = []
        score = 0

        spread_state = str(context.get("spread_state", "unknown") or "unknown")
        liquidity_state = str(context.get("liquidity_state", "unknown") or "unknown")
        volatility_state = str(context.get("volatility_state", "unknown") or "unknown")
        regime_name = str(regime.get("name", context.get("regime", "unknown")) or "unknown")

        if regime_name == "trend":
            positives.append("market structure still has direction")
            score += 1
        elif regime_name == "range":
            issues.append("the market is ranging rather than trending cleanly")
            missing.append("clearer directional structure")
        if spread_state == "tight":
            positives.append("spread is tight")
            score += 1
        elif spread_state == "wide":
            issues.append("spread is too wide")
            missing.append("tighter spread")
            issue_types.append("spread")
            score -= 1
        if liquidity_state == "normal":
            positives.append("liquidity is stable")
            score += 1
        elif liquidity_state in {"poor", "below_normal"}:
            issues.append("liquidity is weak")
            missing.append("better liquidity")
            issue_types.append("liquidity")
            score -= 1
        if threshold is not None and confidence:
            if confidence >= _safe_float(threshold):
                positives.append("confidence is above the current threshold")
                score += 1
            else:
                issues.append("confidence is below the current threshold")
                missing.append("stronger confidence")
                issue_types.append("confidence")
                score -= 1
        if higher_tf_bias in {"buy", "sell"} and side in {"buy", "sell"}:
            if higher_tf_bias == side:
                positives.append("higher-timeframe bias is aligned")
                score += 1
            else:
                issues.append("higher-timeframe bias is not aligned with the current side")
                missing.append("better higher-timeframe alignment")
                issue_types.append("higher-timeframe alignment")
                score -= 1

        reason_text = " ".join(
            [
                str(self.status.get("message", "") or ""),
                " ".join(str(x) for x in list(decision.get("reason_codes", []) or [])),
                " ".join(str(x) for x in list(decision.get("setup_quality_reasons", []) or [])),
                " ".join(str(x) for x in list(decision.get("penalties", []) or [])),
                " ".join(str(x) for x in list(shadow.get("anomaly_flags", []) or [])),
            ]
        ).lower()
        if any(term in reason_text for term in ("consensus", "weak_consensus")):
            issues.append("consensus is not strong enough")
            missing.append("stronger strategy consensus")
            issue_types.append("consensus")
            score -= 1
        if any(term in reason_text for term in ("loss_pattern", "loss-pattern", "memory", "drift")):
            issues.append("loss-pattern protection is still warning against this setup")
            missing.append("cleaner conditions against the recent loss pattern")
            issue_types.append("loss-pattern protection")
            score -= 1
        if volatility_state == "high":
            issues.append("volatility is elevated")
        elif volatility_state == "normal":
            positives.append("volatility is manageable")
        if str(self.status.get("action", "") or "") == "blocked":
            score -= 1
        blocks = self.top_block_reasons()
        if blocks:
            issues.append(f"recent blocked conditions are still led by {blocks[0][0].rstrip('.')}")
        quality = "strong" if score >= 3 else ("moderate" if score >= 1 else "weak")
        return BrainOutput(
            kind="setup_assessment",
            summary=quality,
            details={
                "candidate": candidate,
                "quality": quality,
                "positives": positives,
                "issues": issues,
                "missing": list(dict.fromkeys(x for x in missing if x)),
                "issue_types": list(dict.fromkeys(x for x in issue_types if x)),
                "higher_tf_bias": higher_tf_bias,
                "confidence": confidence,
                "threshold": threshold,
            },
            evidence=["current status", "context snapshot", "decision explanation", "shadow advice", "recent events"],
        )

    def confidence_assessment(self) -> BrainOutput:
        setup = self.setup_assessment()
        details = dict(setup.details)
        confidence = _safe_float(details.get("confidence", 0.0))
        threshold = details.get("threshold")
        score = max(0, min(100, round(confidence * 100)))
        if threshold is not None:
            threshold_pct = round(_safe_float(threshold) * 100)
            summary = f"Current setup confidence is about {score}%, against a threshold of {threshold_pct}%."
        else:
            summary = f"Current setup confidence is about {score}%."
        return BrainOutput(
            kind="confidence_assessment",
            summary=summary,
            details={"confidence_pct": score, "threshold": threshold, "quality": details.get("quality", "unknown")},
            evidence=["chosen candidate", "confidence threshold", "setup assessment"],
        )

    def market_read(self) -> BrainOutput:
        context = dict(self.status.get("context_snapshot", {}) or {})
        regime = dict(self.status.get("regime", {}) or {})
        favorable, unfavorable = self.favorable_unfavorable()
        return BrainOutput(
            kind="market_read",
            summary=f"{str(regime.get('name', context.get('regime', 'unknown')))} conditions with {self.condition_direction()}",
            details={
                "regime": str(regime.get("name", context.get("regime", "unknown")) or "unknown"),
                "session": str(context.get("session", "unknown") or "unknown"),
                "trend_strength": _safe_float(regime.get("trend_strength", context.get("trend_strength", 0.0))),
                "volatility_state": str(context.get("volatility_state", "unknown") or "unknown"),
                "spread_state": str(context.get("spread_state", "unknown") or "unknown"),
                "liquidity_state": str(context.get("liquidity_state", "unknown") or "unknown"),
                "action": str(self.status.get("action", "unknown") or "unknown"),
                "message": str(self.status.get("message", "") or ""),
                "favorable": favorable,
                "unfavorable": unfavorable,
                "direction": self.condition_direction(),
            },
            evidence=["current status", "market snapshot", "context snapshot", "recent events"],
        )

    def experiment_brief(self) -> BrainOutput:
        comparison = self.comparison
        if not comparison:
            return BrainOutput(
                kind="experiment_brief",
                summary="No active comparison run is bound to the current session.",
                details={"active": False},
                evidence=["comparison artifacts when present"],
            )
        control = dict(comparison.get("control_metrics", {}) or {})
        experiment = dict(comparison.get("experiment_metrics", {}) or {})
        checklist = dict(comparison.get("decision_checklist", {}) or {})
        summary = (
            f"Control vs experiment closed trades: "
            f"{_safe_int(control.get('trades_closed', control.get('total_trades', 0)))} vs "
            f"{_safe_int(experiment.get('trades_closed', experiment.get('total_trades', 0)))}."
        )
        return BrainOutput(
            kind="experiment_brief",
            summary=summary,
            details={
                "active": True,
                "run_id": str(comparison.get("run_id", "") or ""),
                "classification": str(checklist.get("classification", "") or ""),
                "control_status": dict(comparison.get("control_status", {}) or {}),
                "experiment_status": dict(comparison.get("experiment_status", {}) or {}),
            },
            evidence=["comparison_summary.json", "comparison_report.json", "comparison_tracker.json when present"],
        )

    def research_plan(self) -> BrainOutput:
        blocker = self.blocker_rank()
        setup = self.setup_assessment()
        comparison = self.experiment_brief()
        learning = dict(self.learning_state.get("state", {}) or {})
        setup_pool = dict(learning.get("evolved_strategy_pool", {}) or {})
        steps = [
            "analyze current market structure and chart context",
            "review repeated block conditions and recent no-trade patterns",
            "check volatility, spread, liquidity, and higher-timeframe alignment",
            "propose one safe offline experiment with a single changed variable",
        ]
        priorities: list[str] = []
        ranked = list(blocker.details.get("ranking", []) or [])
        if ranked:
            priorities.append(f"Primary research target: {ranked[0][0].rstrip('.')}.")
        issue_types = list(setup.details.get("issue_types", []) or [])
        if issue_types:
            priorities.append(f"Biggest likely performance drag: {issue_types[0]}.")
        if comparison.details.get("active"):
            priorities.append("Keep the active comparison isolated until the checklist outcome is clearer.")
        if setup_pool.get("count", 0):
            priorities.append(f"Review {int(setup_pool.get('count', 0))} evolved candidates offline before any promotion decision.")
        return BrainOutput(
            kind="research_plan",
            summary="Next safe research plan is ready.",
            details={"steps": steps, "priorities": priorities},
            evidence=["recent events", "setup assessment", "comparison evidence", "learning state"],
        )

    def self_model(self) -> BrainOutput:
        assessment = self.self_assessment().details
        capability_limits = list(dict.fromkeys(str(item) for item in list(assessment.get("limits", []) or []) if item))
        capabilities = list(dict.fromkeys(str(item) for item in list(assessment.get("strengths", []) or []) if item))
        action = str(self.status.get("action", "unknown") or "unknown")
        message = str(self.status.get("message", "") or "No current status message is available.").strip()
        health = dict(self.status.get("system_health", {}) or {})
        mode = str(health.get("effective_mode", "paper") or "paper")
        blocker = self.blocker_rank().summary
        readiness = "not ready for live demo"
        if int(assessment.get("score", 0) or 0) >= 70 and not bool(dict(self.status.get("drift", {}) or {}).get("active", False)):
            readiness = "closer to readiness, but still not validated for live demo"
        current_state = f"{action} in {mode} mode"
        if message:
            current_state += f", with current message: {message.rstrip('.')}"
        if blocker and blocker != "No dominant blocker identified yet.":
            current_state += f". Main blocker: {blocker.rstrip('.')}"
        model = BrainSelfModel(
            identity="Atlas AI, a read-only trading and operator assistant built around paper evaluation and research support",
            current_state=current_state,
            capabilities=capabilities or ["summarize current market state", "rank blockers", "evaluate setup quality", "propose safe offline research steps"],
            limitations=capability_limits or ["validated live readiness is not established yet"],
            readiness=readiness,
            confidence_in_self_assessment=int(assessment.get("score", 0) or 0),
        )
        return BrainOutput(
            kind="self_model",
            summary=model.current_state,
            details={
                "identity": model.identity,
                "current_state": model.current_state,
                "capabilities": model.capabilities,
                "limitations": model.limitations,
                "readiness": model.readiness,
                "confidence_in_self_assessment": model.confidence_in_self_assessment,
            },
            evidence=["current status", "system health", "blocker ranking", "self assessment", "paper-validation state"],
        )

    def self_assessment(self) -> BrainOutput:
        score = 25
        strengths: list[str] = []
        limits: list[str] = []
        context = dict(self.status.get("context_snapshot", {}) or {})
        regime = dict(self.status.get("regime", {}) or {})
        drift = dict(self.status.get("drift", {}) or {})
        win_rate_focus = dict(self.status.get("win_rate_focus", {}) or {})
        decision = dict(self.status.get("decision_explanation", {}) or {})
        health = dict(self.status.get("system_health", {}) or {})
        chosen = dict(self.status.get("chosen", {}) or {})
        blocks = self.top_block_reasons()

        if regime or context:
            score += 15
            strengths.append("identify trend and regime")
        if any(context.get(key) not in {None, "", "unknown"} for key in ("volatility_state", "spread_state", "liquidity_state")):
            score += 15
            strengths.append("evaluate volatility, spread, and liquidity")
        if decision.get("final_decision") or blocks:
            score += 15
            strengths.append("detect blocking conditions")
        if chosen and _safe_float(chosen.get("confidence", 0.0)) > 0:
            score += 5
            strengths.append("evaluate current setup confidence")
        if self.comparison and (self.comparison.get("control_metrics") or self.comparison.get("experiment_metrics")):
            score += 10
            strengths.append("compare current evidence against paper experiments")

        samples = _safe_int(win_rate_focus.get("samples", 0))
        recent_wr = _safe_float(win_rate_focus.get("recent_win_rate", 0.0))
        if samples >= 30:
            score += 10
        elif samples >= 10:
            score += 4
        else:
            limits.append("limited confirmed trade outcomes are available right now")
            score -= 10
        if samples and recent_wr >= 0.5:
            score += 5
        elif samples:
            limits.append("recent validated edge is still below the target win-rate goal")
            score -= 5
        else:
            limits.append("there is no strong validated live edge yet")
        if drift.get("active"):
            limits.append("recent drift is active, so current signal quality is not fully stable")
            score -= 5
        if decision.get("final_decision") == "block" and decision.get("reason_codes"):
            limits.append("blocked or low-quality conditions are still more common than confirmed opportunities")
        if bool(health.get("paper_mode_confirmed", False)):
            limits.append("the system is still in paper evaluation rather than validated live deployment")
        else:
            limits.append("paper-mode safety confirmation is incomplete")

        score = max(10, min(85, score))
        rating = "fairly strong" if score >= 70 else ("structured but still developing" if score >= 50 else "limited and still early-stage")
        reliability = "moderate" if score >= 60 else ("limited" if score < 45 else "developing")
        readiness = "not ready for live demo"
        if score >= 70 and not bool(drift.get("active", False)) and samples >= 30 and recent_wr >= 0.5:
            readiness = "closer to live readiness, but still needs operator-controlled validation"
        return BrainOutput(
            kind="self_assessment",
            summary=f"I would rate current chart and market understanding at approximately {score}%.",
            details={
                "score": score,
                "rating": rating,
                "reliability": reliability,
                "readiness": readiness,
                "strengths": list(dict.fromkeys(strengths)),
                "limits": list(dict.fromkeys(limits)),
            },
            evidence=["current status", "win-rate focus", "drift state", "decision explanation", "comparison evidence when active"],
        )

    def action_guidance(self, component: str) -> BrainOutput:
        component_key = str(component or "general").strip().lower()
        blocker = self.blocker_rank()
        setup = self.setup_assessment()
        market = self.market_read()
        plan = self.research_plan()
        comparison = self.experiment_brief()
        missed_learning = dict(self.status.get("missed_entry_learning_summary", {}) or {})
        repair_candidate = dict(missed_learning.get("top_repair_candidate", {}) or {})
        repair_blocker = str(repair_candidate.get("primary_blocker", "") or "").strip()
        repair_priority = str(repair_candidate.get("repair_priority", "") or "").strip().lower()
        repair_action = str(repair_candidate.get("suggested_action", "") or "").strip()
        repair_samples = _safe_int(repair_candidate.get("samples", 0))
        repair_missed = _safe_int(repair_candidate.get("missed_opportunity", 0))
        repair_correct = _safe_int(repair_candidate.get("correct_hold_or_block", 0))
        has_repair_candidate = bool(missed_learning.get("available")) and bool(repair_blocker)
        top_ranking = list(blocker.details.get("ranking", []) or [])
        primary_blocker = top_ranking[0][0] if top_ranking else str(self.status.get("message", "") or "no dominant blocker identified")
        setup_issue_types = [str(x) for x in list(setup.details.get("issue_types", []) or [])]
        setup_issues = [str(x) for x in list(setup.details.get("issues", []) or [])]
        setup_missing = [str(x) for x in list(setup.details.get("missing", []) or [])]
        market_details = dict(market.details or {})

        if component_key == "consensus":
            diagnosis = (
                "too strict only if it is blocking clean setups with acceptable expectancy, "
                "but currently it still looks more like a quality screen than the main raw-edge problem"
            )
            recommendation = "improve upstream"
            reasoning = (
                f"consensus is the current primary blocker: {primary_blocker.rstrip('.')}. "
                f"The setup assessment still shows {', '.join(setup_issue_types[:3]) or 'multiple quality gaps'}, "
                "which means weakening consensus directly would likely admit more weak setups before the raw signal improves."
            )
        elif component_key in {"loss_pattern", "loss-pattern", "loss_pattern_protection", "loss-pattern protection"}:
            diagnosis = (
                "selective rather than dominant right now, and it looks more protective than excessive unless isolated replay proves otherwise"
            )
            recommendation = "run experiment"
            reasoning = (
                f"loss-pattern protection appears in the current setup issues as {', '.join([i for i in setup_issue_types if 'loss-pattern' in i][:2]) or 'a secondary warning'}, "
                "not as the leading blocker. That makes it a better candidate for isolated offline testing than a direct adjustment."
            )
        elif component_key in {"signal", "signals", "signal_quality", "raw_signal_quality"}:
            diagnosis = "currently too weak before the guard stack finishes filtering them"
            recommendation = "improve upstream"
            reasoning = (
                f"the market read is {market.summary.rstrip('.')}, and the setup is {setup.details.get('quality', 'unclear')} "
                f"with missing conditions such as {', '.join(setup_missing[:3]) or 'cleaner alignment'}. "
                "That points to a raw signal-quality problem before consensus and other blockers."
            )
        elif component_key in {"strategy", "strategies", "momentum", "momentum_continuation"}:
            diagnosis = "mixed and condition-dependent rather than broadly reliable"
            recommendation = "improve upstream"
            reasoning = (
                f"the current setup issues are {', '.join(setup_issue_types[:3]) or 'still unresolved'}, "
                "and the latest market read does not show a uniformly favorable environment. "
                "The better move is to improve regime-to-strategy fit rather than forcing more strategy output."
            )
        elif component_key in {"guard", "guards", "blockers", "filter", "filters"}:
            diagnosis = "stacked, but with one dominant blocker rather than random noise"
            recommendation = "run experiment"
            reasoning = (
                f"the blocker ranking is led by {primary_blocker.rstrip('.')}, "
                f"while the setup still shows {', '.join(setup_issue_types[:3]) or 'other quality issues'}. "
                "That means guard behavior should be tested one variable at a time offline, not changed directly."
            )
        else:
            diagnosis = "still constrained by the current blocker and setup-quality gaps"
            recommendation = "focus on the primary blocker first"
            reasoning = (
                f"the current blocker ranking is led by {primary_blocker.rstrip('.')}, and the setup is {setup.details.get('quality', 'unclear')}. "
                "That makes the safest path targeted research before any system change is considered."
            )

        if has_repair_candidate and component_key in {"general", "guard", "guards", "blockers", "filter", "filters", "consensus"}:
            recommendation = "run experiment"
            diagnosis = (
                f"forward validation marks {repair_blocker.rstrip('.')} as a {repair_priority or 'watch'} repair candidate"
            )
            reasoning = (
                f"missed-entry reviews found {repair_missed} missed opportunities versus {repair_correct} correct blocks "
                f"across {repair_samples} reviewed samples for this blocker. "
                "That is enough to inspect blocker selectivity offline, but not enough to silently loosen execution rules."
            )

        current_role = (
            f"{component_key.replace('_', ' ')} is currently acting as part of the decision-quality layer"
            if component_key not in {"signal", "signals", "signal_quality", "raw_signal_quality"}
            else "raw signal quality is currently shaping how many setups reach the decision layer with real conviction"
        )
        if component_key == "consensus":
            current_role = f"consensus is currently acting as the main decision-quality gate, and it is led by {primary_blocker.rstrip('.')}"
        elif component_key in {"loss_pattern", "loss-pattern", "loss_pattern_protection", "loss-pattern protection"}:
            current_role = "loss-pattern protection is currently acting as a memory-based veto that warns against repeating recent bad setup conditions"
        elif component_key in {"strategy", "strategies", "momentum", "momentum_continuation"}:
            current_role = "strategy behavior is currently deciding which raw ideas become directional candidates before the guard stack evaluates them"

        next_step = ""
        priorities = [str(x) for x in list(plan.details.get("priorities", []) or [])]
        if recommendation == "run experiment" and priorities:
            next_step = priorities[0]
        elif recommendation == "improve upstream":
            next_step = "improve raw signal quality and regime fit so more setups naturally satisfy the current decision layer"
        elif recommendation == "keep":
            next_step = "keep the component unchanged and continue monitoring it against current blocker and setup data"
        elif recommendation == "adjust":
            next_step = "prepare a narrow offline calibration experiment before considering any broader change"
        else:
            next_step = "focus on the primary blocker and validate one offline change at a time"
        if has_repair_candidate and component_key in {"general", "guard", "guards", "blockers", "filter", "filters", "consensus"}:
            next_step = "run an offline blocker-selectivity review for the top missed-entry repair candidate before changing any threshold"

        return BrainOutput(
            kind="action_guidance",
            summary=f"{component_key.replace('_', ' ')} guidance is ready.",
            details={
                "component": component_key,
                "current_role": current_role,
                "diagnosis": diagnosis,
                "recommendation": recommendation,
                "reasoning": reasoning,
                "primary_blocker": primary_blocker,
                "setup_quality": str(setup.details.get("quality", "unknown")),
                "market_read": market_details,
                "comparison_active": bool(comparison.details.get("active")),
                "next_step": next_step,
                "missed_entry_repair_candidate": repair_candidate if has_repair_candidate else {},
                "missed_entry_repair_action": repair_action,
                "setup_issue_types": setup_issue_types,
                "setup_issues": setup_issues,
                "setup_missing": setup_missing,
            },
            evidence=[
                "blocker ranking",
                "setup assessment",
                "market read",
                "research plan",
                "comparison artifacts when active",
                "missed-entry learning summary",
            ],
        )
