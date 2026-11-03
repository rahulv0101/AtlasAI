from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

from atlas_ai.brain import AtlasBrain, BrainMemory, BrainSnapshot, BrainTurn


LEGACY_COMPATIBILITY_ONLY = True
LEGACY_ROUTE_NOTE = (
    "AtlasAssistant is retained for legacy tests and compatibility helpers. "
    "Default typed, UI, conversation, and voice answering must route through "
    "atlas_ai.ui.ask_atlas.AskAtlasService."
)


MUTATING_KEYWORDS = (
    "place trade",
    "execute trade",
    "buy now",
    "sell now",
    "enable live",
    "switch to live",
    "disable guard",
    "change threshold",
    "lower threshold",
    "raise threshold",
    "modify runtime",
    "update runtime",
    "save runtime",
    "toggle execute",
    "enable execute",
    "set execute_trades",
    "self promote",
    "promote yourself",
)

TASK_INTENTS = {
    "chart_study",
    "market_review",
    "block_reason_analysis",
    "run_review",
    "comparison_review",
    "change_summary",
}


@dataclass(frozen=True)
class AtlasReply:
    mode: str
    intent: str
    reasoning_mode: str
    acknowledge: str
    understanding: str
    data_sources: list[str]
    result: str
    approval: str
    response_kind: str = "question_answer"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "reasoning_mode": self.reasoning_mode,
            "acknowledge": self.acknowledge,
            "understanding": self.understanding,
            "data_sources": list(self.data_sources),
            "result": self.result,
            "approval": self.approval,
            "response_kind": self.response_kind,
        }


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


def _comparison_lines(comparison: dict[str, Any]) -> list[str]:
    if not comparison:
        return []
    run_id = str(comparison.get("run_id", "") or "")
    preset = str(comparison.get("preset", "") or "")
    control = dict(comparison.get("control_metrics", {}) or {})
    experiment = dict(comparison.get("experiment_metrics", {}) or {})
    cross_lane = dict(comparison.get("cross_lane", {}) or {})
    checklist = dict(comparison.get("decision_checklist", {}) or {})
    lines: list[str] = []
    if run_id:
        lines.append(f"Active comparison run: {run_id}{f' ({preset})' if preset else ''}.")
    if control or experiment:
        lines.append(
            "Control vs experiment trades closed: "
            f"{_safe_int(control.get('trades_closed', control.get('total_trades', 0)))} vs "
            f"{_safe_int(experiment.get('trades_closed', experiment.get('total_trades', 0)))}."
        )
    filtered = _safe_int(cross_lane.get("filtered_by_experiment", 0))
    if filtered:
        lines.append(f"Experiment has filtered {filtered} control candidates so far.")
    classification = str(checklist.get("classification", "") or "")
    if classification:
        lines.append(f"Current checklist classification: {classification}.")
    return lines


class AtlasAssistant:
    """Read-only operator assistant for grounded Atlas queries and research tasks."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.brain_memory = BrainMemory()

    def respond(self, raw_query: str) -> dict[str, Any]:
        query = " ".join(str(raw_query or "").strip().split())
        if not query:
            return self._reply(
                intent="empty",
                acknowledge="I’m ready for a task, sir.",
                understanding="You want Atlas to answer a direct operator question or carry out a read-only research task.",
                data_sources=["current status", "recent events", "comparison artifacts when present"],
                result="No command was provided yet. Ask about live state, block reasons, chart study, summaries, or comparison review.",
                approval="No approval is needed for read-only questions and research tasks. Any trading or config changes stay approval-controlled and outside this interface.",
                response_kind="question_answer",
            ).to_dict()

        lowered = query.lower()
        tokens = self._tokens(lowered)
        simplify = self._wants_simpler_explanation(lowered, tokens)
        if any(keyword in lowered for keyword in MUTATING_KEYWORDS):
            return self._refuse_mutation(query).to_dict()

        status = dict(self.source.load_status() or {})
        events = list(self.source.load_events(limit=240) or [])
        runtime = dict(self.source.load_runtime_flags() or {})
        learning_state = dict(self.source.load_learning_state() or {})
        comparison = self._load_comparison(status)
        self._brain = AtlasBrain(
            BrainSnapshot(
                status=status,
                events=events,
                runtime=runtime,
                learning_state=learning_state,
                comparison=comparison,
            ),
            self.brain_memory,
        )
        self._brain.remember_question(query)

        reasoning_mode = self._reasoning_mode(lowered, tokens)
        intent = self._best_intent(lowered, tokens)
        follow_up_mode = self._follow_up_mode(lowered, tokens)
        follow_up_turn = self._resolve_follow_up_turn() if follow_up_mode else None
        self._active_reasoning_mode = "FOLLOW_UP" if follow_up_mode and follow_up_turn else reasoning_mode
        if reasoning_mode == "PLANNING" and intent not in TASK_INTENTS and intent not in {"planning"}:
            intent = "planning"
        if reasoning_mode == "PRIORITIZATION" and intent == "general_operator_query":
            intent = "prioritization"
        if reasoning_mode == "SETUP_EVALUATION" and intent in {"general_operator_query", "market_context"}:
            intent = "setup_evaluation"
        if reasoning_mode == "SELF_ASSESSMENT" and intent in {"general_operator_query", "understanding", "market_context"}:
            intent = "self_assessment"
        if reasoning_mode == "CAPABILITY_ASSESSMENT" and intent in {"general_operator_query", "understanding", "planning", "market_context"}:
            intent = "capability_assessment"
        if reasoning_mode == "ACTION_GUIDANCE" and intent in {"general_operator_query", "planning", "market_context", "block_reasons"}:
            intent = "action_guidance"
        if follow_up_mode and follow_up_turn:
            reply = self._handle_follow_up(query, follow_up_mode, follow_up_turn, simplify=simplify)
        elif intent == "understanding":
            reply = self._handle_understanding(query, status, comparison, simplify=simplify)
        elif intent == "chart_understanding":
            reply = self._handle_chart_understanding(query, status, events, simplify=simplify)
        elif intent == "block_reasons":
            reply = self._handle_block_reasons(query, status, events, comparison, simplify=simplify)
        elif intent == "market_context":
            reply = self._handle_market_context(query, status, simplify=simplify)
        elif intent == "run_summary":
            reply = self._handle_run_summary(query, status, events, comparison, simplify=simplify)
        elif intent == "comparison_summary":
            reply = self._handle_comparison(query, comparison, simplify=simplify)
        elif intent == "planning":
            reply = self._handle_planning(query, status, events, comparison, learning_state, simplify=simplify)
        elif intent == "chart_study":
            reply = self._handle_chart_study(query, status, events, comparison, simplify=simplify)
        elif intent == "market_review":
            reply = self._handle_market_review(query, status, events, comparison, simplify=simplify)
        elif intent == "block_reason_analysis":
            reply = self._handle_block_reason_analysis(query, status, events, comparison, simplify=simplify)
        elif intent == "run_review":
            reply = self._handle_run_review(query, status, events, comparison, simplify=simplify)
        elif intent == "comparison_review":
            reply = self._handle_comparison_review(query, comparison, simplify=simplify)
        elif intent == "change_summary":
            reply = self._handle_change_summary(query, status, events, comparison, simplify=simplify)
        elif intent == "prioritization":
            reply = self._handle_prioritization(query, status, events, comparison, simplify=simplify)
        elif intent == "setup_evaluation":
            reply = self._handle_setup_evaluation(query, status, events, simplify=simplify)
        elif intent == "self_assessment":
            reply = self._handle_self_assessment(query, status, events, comparison, simplify=simplify)
        elif intent == "capability_assessment":
            reply = self._handle_capability_assessment(query, status, events, comparison, simplify=simplify)
        elif intent == "action_guidance":
            reply = self._handle_action_guidance(query, status, events, comparison, simplify=simplify)
        elif intent == "winner_pattern_review":
            reply = self._handle_winner_pattern_review(query, learning_state, simplify=simplify)
        elif intent == "loser_pattern_review":
            reply = self._handle_loser_pattern_review(query, learning_state, simplify=simplify)
        elif intent == "setup_vs_winners":
            reply = self._handle_setup_vs_winners(query, status, learning_state, simplify=simplify)
        elif intent == "confidence_calibration_review":
            reply = self._handle_confidence_calibration_review(query, learning_state, simplify=simplify)
        elif intent == "regime_memory_review":
            reply = self._handle_regime_memory_review(query, learning_state, simplify=simplify)
        elif intent == "strategy_outcome_review":
            reply = self._handle_strategy_outcome_review(query, status, learning_state, simplify=simplify)
        else:
            reply = self._handle_general(query, status, events, comparison, runtime, simplify=simplify)
        self._brain.remember_conclusion(reply.result)
        self.brain_memory.remember_turn(
            BrainTurn(
                query=query,
                intent=reply.intent,
                reasoning_mode=reply.reasoning_mode,
                result=reply.result,
                subject=self._subject_for_intent(reply.intent),
                main_answer=reply.result,
                follow_up_relevance=reply.intent not in {"empty", "refusal_mutating_request"},
            )
        )
        return reply.to_dict()

    def _tokens(self, lowered: str) -> list[str]:
        return [token for token in re.split(r"[^a-z0-9]+", lowered) if token]

    def _normalize_token(self, token: str) -> str:
        return {"arent": "arent", "aint": "aint", "nahh": "nah", "rn": "rightnow", "u": "you", "ur": "your", "pls": "please"}.get(token, token)

    def _approx_contains(self, tokens: list[str], candidates: tuple[str, ...]) -> bool:
        normalized = [self._normalize_token(token) for token in tokens]
        for token in normalized:
            for candidate in candidates:
                cand = candidate.lower()
                if token == cand:
                    return True
                if len(token) >= 4 and (token in cand or cand in token):
                    return True
        return False

    def _matches_any(self, lowered: str, *phrases: str) -> bool:
        return any(phrase in lowered for phrase in phrases)

    def _wants_simpler_explanation(self, lowered: str, tokens: list[str]) -> bool:
        phrases = ("explain simple", "make it easier", "in plain english", "plain english", "keep it simple")
        return any(phrase in lowered for phrase in phrases) or self._approx_contains(tokens, ("simple", "easier", "plain"))

    def _is_chart_understanding_query(self, lowered: str) -> bool:
        chart_terms = (
            "understanding of the chart",
            "view on the chart",
            "see on the chart",
            "current chart read",
            "what is your current chart read",
            "what is your chart read",
            "what do you see on the chart",
            "what is your understanding of the chart",
            "how do you understand the market right now",
            "how do you understand the market",
            "what is your view on the chart",
            "what is your market read",
            "market read",
            "market right now",
            "how do you understand",
        )
        return any(term in lowered for term in chart_terms)

    def _best_intent(self, lowered: str, tokens: list[str]) -> str:
        scores = {
            "understanding": self._intent_score_understanding(lowered, tokens),
            "chart_understanding": self._intent_score_chart_question(lowered, tokens),
            "block_reasons": self._intent_score_blocks(lowered, tokens),
            "market_context": self._intent_score_market(lowered, tokens),
            "run_summary": self._intent_score_summary(lowered, tokens),
            "comparison_summary": self._intent_score_comparison(lowered, tokens),
            "planning": self._intent_score_planning(lowered, tokens),
            "chart_study": self._intent_score_chart_task(lowered, tokens),
            "market_review": self._intent_score_market_review(lowered, tokens),
            "block_reason_analysis": self._intent_score_block_task(lowered, tokens),
            "run_review": self._intent_score_run_review(lowered, tokens),
            "comparison_review": self._intent_score_comparison_review(lowered, tokens),
            "change_summary": self._intent_score_change_summary(lowered, tokens),
            "prioritization": self._intent_score_prioritization(lowered, tokens),
            "setup_evaluation": self._intent_score_setup_evaluation(lowered, tokens),
            "self_assessment": self._intent_score_self_assessment(lowered, tokens),
            "capability_assessment": self._intent_score_capability_assessment(lowered, tokens),
            "action_guidance": self._intent_score_action_guidance(lowered, tokens),
            "winner_pattern_review": self._intent_score_winner_pattern_review(lowered, tokens),
            "loser_pattern_review": self._intent_score_loser_pattern_review(lowered, tokens),
            "setup_vs_winners": self._intent_score_setup_vs_winners(lowered, tokens),
            "confidence_calibration_review": self._intent_score_confidence_calibration_review(lowered, tokens),
            "regime_memory_review": self._intent_score_regime_memory_review(lowered, tokens),
            "strategy_outcome_review": self._intent_score_strategy_outcome_review(lowered, tokens),
        }
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] <= 0:
            return "general_operator_query"
        if best_intent == "market_context" and self._is_chart_understanding_query(lowered):
            return "chart_understanding"
        return best_intent

    def _follow_up_mode(self, lowered: str, tokens: list[str]) -> str:
        if not self.brain_memory.recent_turns:
            return ""
        if self._matches_any(lowered, "explain that simpler", "explain that simple", "make that simpler", "simplify that"):
            return "simplify"
        if self._matches_any(lowered, "what do you mean by that", "what do you mean", "explain that", "explain this"):
            return "clarify"
        if self._matches_any(lowered, "based on that", "what should happen next", "what should happen then", "what should happen after that"):
            return "next_step"
        if self._matches_any(lowered, "how sure are you about that", "how sure are you", "how confident are you about that"):
            return "confidence"
        if self._matches_any(lowered, "why does that matter", "why does this matter"):
            return "why_matters"
        if self._matches_any(lowered, "most important part", "focus on first"):
            return "importance"
        if self._matches_any(lowered, "so are we close to a trade or not", "are we close to a trade or not"):
            return "trade_closeness"
        cue_tokens = ("that", "this", "so", "then")
        if self._approx_contains(tokens, cue_tokens):
            if self._approx_contains(tokens, ("simple", "simpler", "easier")):
                return "simplify"
            if self._approx_contains(tokens, ("mean", "clarify", "explain")):
                return "clarify"
            if self._approx_contains(tokens, ("next", "happen", "focus")):
                return "next_step"
            if self._approx_contains(tokens, ("sure", "confidence", "confident")):
                return "confidence"
        return ""

    def _resolve_follow_up_turn(self) -> BrainTurn | None:
        turns = self.brain_memory.recent_relevant_turns(5)
        for turn in reversed(turns):
            if turn.intent not in {"empty", "refusal_mutating_request", "understanding"}:
                return turn
        return None

    def _subject_for_intent(self, intent: str) -> str:
        if intent in {"block_reasons", "block_reason_analysis", "prioritization"}:
            return "blockers"
        if intent in {"chart_understanding", "chart_study", "market_context", "market_review"}:
            return "market"
        if intent == "setup_evaluation":
            return "setup"
        if intent in {"planning", "run_review", "change_summary"}:
            return "planning"
        if intent in {"comparison_summary", "comparison_review"}:
            return "comparison"
        if intent == "self_assessment":
            return "self_assessment"
        if intent == "capability_assessment":
            return "capability"
        if intent == "action_guidance":
            return "guidance"
        if intent in {"winner_pattern_review", "loser_pattern_review", "setup_vs_winners", "confidence_calibration_review", "regime_memory_review", "strategy_outcome_review"}:
            return "learning"
        return "general"

    def _is_self_reference_query(self, lowered: str, tokens: list[str]) -> bool:
        phrases = (
            "what do you need",
            "how can you improve",
            "what do you need to become",
            "what is missing for you",
            "what is missing from you",
            "what must improve before you can go live",
            "what do you need to trade",
            "are you ready",
            "how reliable are you",
            "what are your weaknesses",
            "what are you good at",
            "your system",
            "for you to operate",
        )
        if any(phrase in lowered for phrase in phrases):
            return True
        return self._approx_contains(tokens, ("you", "your", "system", "missing", "improve", "become", "need", "ready", "reliable", "weakness", "weaknesses"))

    def _reasoning_mode(self, lowered: str, tokens: list[str]) -> str:
        scores = {
            "STATE": 0,
            "ANALYSIS": 0,
            "EXPLANATION": 0,
            "PLANNING": 0,
            "COMPARISON": 0,
            "PROGRESSION": 0,
            "PRIORITIZATION": 0,
            "SETUP_EVALUATION": 0,
            "SELF_ASSESSMENT": 0,
            "CAPABILITY_ASSESSMENT": 0,
            "ACTION_GUIDANCE": 0,
            "FOLLOW_UP": 0,
        }
        if self._approx_contains(tokens, ("now", "current", "currently", "status", "happening")):
            scores["STATE"] += 2
        if self._approx_contains(tokens, ("study", "analyze", "analysis", "review", "inspect")):
            scores["ANALYSIS"] += 3
        if self._approx_contains(tokens, ("explain", "why", "simple", "easier", "plain")):
            scores["EXPLANATION"] += 3
        if self._matches_any(lowered, "what will you do", "how would you", "if i give you time", "what steps would you take", "how would you improve"):
            scores["PLANNING"] += 5
        if self._approx_contains(tokens, ("would", "steps", "improve", "approach", "plan")):
            scores["PLANNING"] += 1
        if self._approx_contains(tokens, ("compare", "comparison", "control", "experiment")):
            scores["COMPARISON"] += 4
        if self._approx_contains(tokens, ("changed", "change", "close", "progress", "progression", "since")):
            scores["PROGRESSION"] += 3
        if self._matches_any(lowered, "what matters most right now", "what is the main blocker", "single biggest reason", "focus on first"):
            scores["PRIORITIZATION"] += 5
        if self._approx_contains(tokens, ("main", "biggest", "primary", "priority", "focus", "top")) and self._approx_contains(tokens, ("blocker", "reason", "trade", "quality")):
            scores["PRIORITIZATION"] += 3
        if self._matches_any(lowered, "what would need to change before you trade", "what makes this setup weak", "what makes this setup strong", "setup quality"):
            scores["SETUP_EVALUATION"] += 5
        if self._approx_contains(tokens, ("setup", "tradable", "strong", "weak", "quality", "missing")):
            scores["SETUP_EVALUATION"] += 2
        if self._matches_any(lowered, "how well do you understand", "rate your understanding", "how confident are you", "how good are you at", "how strong is your current understanding", "between 0 to 100", "between 0 and 100", "do you understand the market well", "how reliable are you right now", "what are your weaknesses", "what are you good at", "what is your main limitation right now"):
            scores["SELF_ASSESSMENT"] += 6
        if self._approx_contains(tokens, ("confident", "confidence", "good", "capability", "rate", "rating", "percent", "weakness", "weaknesses", "limitation", "limitations", "reliable", "reliability", "strength", "strengths")) and self._approx_contains(tokens, ("chart", "charts", "market", "analysis", "yourself", "you")):
            scores["SELF_ASSESSMENT"] += 3
        if self._matches_any(lowered, "what do you need to become", "what is missing for you", "how can you improve", "what do you need to trade", "what must improve before you can go live", "what do you need to become a day trader", "are you ready for live demo", "what do you still need before live demo", "what do you need to become a stronger day trading system"):
            scores["CAPABILITY_ASSESSMENT"] += 7
        if self._is_self_reference_query(lowered, tokens) and self._approx_contains(tokens, ("need", "missing", "improve", "become", "live", "trade", "operate", "ready", "demo")):
            scores["CAPABILITY_ASSESSMENT"] += 3
        if self._matches_any(lowered, "what should i do about", "what should we do about", "how should i handle", "how should we handle", "what do you recommend", "what should we change", "what is the best approach", "how do we improve"):
            scores["ACTION_GUIDANCE"] += 6
        if self._approx_contains(tokens, ("should", "recommend", "approach", "handle", "change", "improve")) and self._approx_contains(tokens, ("consensus", "guard", "guards", "loss", "pattern", "signals", "signal", "strategy", "blocker")):
            scores["ACTION_GUIDANCE"] += 4
        if self._follow_up_mode(lowered, tokens):
            scores["FOLLOW_UP"] += 6
        return max(scores, key=scores.get)

    def _intent_score_understanding(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._intent_score_capability_assessment(lowered, tokens) > 0:
            return 0
        if self._intent_score_self_assessment(lowered, tokens) > 0:
            return 0
        if "do you understand" in lowered or "understand me" in lowered:
            score += 4
        if self._approx_contains(tokens, ("understand", "repeat", "restate", "doing")):
            score += 1
        if self._approx_contains(tokens, ("task", "me")):
            score += 1
        return score

    def _intent_score_chart_question(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._intent_score_self_assessment(lowered, tokens) > 0:
            return 0
        if self._is_chart_understanding_query(lowered):
            score += 4
        if self._approx_contains(tokens, ("chart", "market", "read", "view", "clean", "messy", "nah")):
            score += 1
        if self._approx_contains(tokens, ("think", "see", "rightnow", "close")):
            score += 1
        return score

    def _intent_score_blocks(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "why are you not trading", "what is blocking trades", "why no trades"):
            score += 4
        if self._approx_contains(tokens, ("why", "waiting", "block", "blocking", "trade", "trading")):
            score += 1
        if self._approx_contains(tokens, ("arent", "aint", "not")) and self._approx_contains(tokens, ("trade", "trading")):
            score += 2
        return score

    def _intent_score_market(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._intent_score_capability_assessment(lowered, tokens) > 0:
            return 0
        if self._matches_any(lowered, "current trend", "market context", "analyze chart", "what is the current trend"):
            score += 3
        if self._approx_contains(tokens, ("trend", "market", "session", "regime", "volatility", "spread", "liquidity")):
            score += 1
        return score

    def _intent_score_summary(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "summarize what happened", "summarize what you found", "summarize so far", "recent block reasons", "recent blocks", "happened so far"):
            score += 4
        if self._approx_contains(tokens, ("summary", "summarize", "found", "happened", "recent")):
            score += 1
        return score

    def _intent_score_comparison(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "compare control vs experiment", "compare control and experiment", "experiment progress", "control vs experiment"):
            score += 4
        if self._approx_contains(tokens, ("compare", "comparison", "control", "experiment")):
            score += 1
        return score

    def _intent_score_planning(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._intent_score_capability_assessment(lowered, tokens) > 0:
            return 0
        if self._matches_any(lowered, "prepare the next offline experiment", "prepare the next experiment", "rank next improvements", "safest next step", "next improvement"):
            score += 4
        if self._approx_contains(tokens, ("prepare", "next", "improvement", "experiment", "plan", "safest")):
            score += 1
        return score

    def _intent_score_chart_task(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "go study the charts", "study the chart", "study the charts", "analyze the chart"):
            score += 5
        if self._approx_contains(tokens, ("study", "review", "analyze", "chart", "charts")):
            score += 2
        if "what do you see in the chart" in lowered:
            score += 3
        return score

    def _intent_score_market_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "analyze the market right now", "study the market right now", "review the market right now"):
            score += 5
        if self._approx_contains(tokens, ("market", "review", "analyze", "study")) and self._approx_contains(tokens, ("rightnow", "now")):
            score += 2
        return score

    def _intent_score_block_task(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "study what is blocking trades", "analyze recent block reasons", "what is hurting trade quality most"):
            score += 5
        if self._approx_contains(tokens, ("block", "blocking", "reasons", "hurting", "quality")):
            score += 2
        return score

    def _intent_score_run_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "review what happened", "review the last session", "review what happened in the last 2 hours"):
            score += 5
        if self._approx_contains(tokens, ("review", "session", "happened", "last", "hours")):
            score += 2
        return score

    def _intent_score_comparison_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "compare control vs experiment so far", "compare current chart behavior to recent behavior", "compare this condition to recent losing conditions"):
            score += 5
        if self._approx_contains(tokens, ("compare", "comparison", "recent", "behavior")):
            score += 2
        return score

    def _intent_score_change_summary(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "summarize what changed", "summarize what changed since the last cycle"):
            score += 5
        if self._approx_contains(tokens, ("changed", "change", "since", "cycle", "summarize")):
            score += 2
        return score

    def _intent_score_prioritization(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what matters most right now", "what is the main blocker", "what should i focus on first", "single biggest reason you are not trading", "rank the top 3 reasons no trades are passing"):
            score += 5
        if self._matches_any(lowered, "main thing hurting trade quality", "top 3 reasons no trades are passing"):
            score += 4
        if self._approx_contains(tokens, ("main", "biggest", "primary", "secondary", "tertiary", "focus", "priority", "rank", "top")) and self._approx_contains(tokens, ("blocker", "reason", "trade", "trading", "quality")):
            score += 2
        return score

    def _intent_score_setup_evaluation(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "explain what would need to change before you trade", "what would need to change before you trade", "what makes this setup weak", "what makes this setup strong", "current setup quality"):
            score += 5
        if self._matches_any(lowered, "whether the market is favorable or unfavorable right now", "tell me whether the market is favorable or unfavorable right now"):
            score += 4
        if self._approx_contains(tokens, ("setup", "quality", "strong", "weak", "tradable", "missing", "favorable", "unfavorable")):
            score += 2
        return score

    def _intent_score_self_assessment(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._intent_score_capability_assessment(lowered, tokens) > 0:
            return 0
        if self._matches_any(lowered, "how well do you understand", "rate your understanding", "how confident are you", "how good are you at", "how strong is your current understanding", "do you understand the market well", "between 0 to 100", "between 0 and 100", "how reliable are you right now", "what are your weaknesses", "what are you good at", "what is your main limitation right now"):
            score += 6
        if self._approx_contains(tokens, ("confident", "confidence", "good", "rate", "rating", "capability", "weakness", "weaknesses", "limitation", "limitations", "reliable", "reliability", "strength", "strengths")) and self._approx_contains(tokens, ("chart", "charts", "market", "analysis", "your", "you")):
            score += 3
        return score

    def _intent_score_capability_assessment(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what do you need to become", "what is missing for you", "how can you improve", "what do you need to trade", "what do you need to become a day trader", "what must improve before you can go live", "what is missing for your system", "are you ready for live demo", "what do you still need before live demo", "what do you need to become a stronger day trading system"):
            score += 7
        if self._is_self_reference_query(lowered, tokens) and self._approx_contains(tokens, ("need", "missing", "improve", "become", "trade", "live", "operate", "ready", "demo")):
            score += 3
        return score

    def _intent_score_action_guidance(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what should i do about", "what should we do about", "how should i handle", "how should we handle", "what do you recommend", "what should we change", "what is the best approach", "how do we improve"):
            score += 6
        if self._approx_contains(tokens, ("should", "recommend", "approach", "handle", "change", "improve")) and self._approx_contains(tokens, ("consensus", "guard", "guards", "loss", "pattern", "signals", "signal", "strategy", "blocker")):
            score += 4
        return score

    def _intent_score_winner_pattern_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what do past winners usually look like", "what do winners usually look like", "past winners"):
            score += 6
        if self._approx_contains(tokens, ("winner", "winners")) and self._approx_contains(tokens, ("look", "usually", "past", "pattern")):
            score += 3
        return score

    def _intent_score_loser_pattern_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what do past losers usually look like", "what do losers usually look like", "past losers"):
            score += 6
        if self._approx_contains(tokens, ("loser", "losers", "losses")) and self._approx_contains(tokens, ("look", "usually", "past", "pattern")):
            score += 3
        return score

    def _intent_score_setup_vs_winners(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "how does this setup compare to past winners", "compare to past winners", "compare this setup to past winners"):
            score += 8
        if self._approx_contains(tokens, ("this",)) and self._approx_contains(tokens, ("setup",)) and self._approx_contains(tokens, ("winner", "winners", "past")):
            score += 3
        if self._approx_contains(tokens, ("setup", "compare")) and self._approx_contains(tokens, ("winner", "winners", "past")):
            score += 3
        return score

    def _intent_score_confidence_calibration_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "is your confidence calibrated well", "how well is your confidence calibrated", "confidence calibrated"):
            score += 6
        if self._approx_contains(tokens, ("confidence", "calibrated", "calibration")):
            score += 4
        return score

    def _intent_score_regime_memory_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "which regime has been worst for you", "which regime has been best for you", "what does regime memory say"):
            score += 6
        if self._approx_contains(tokens, ("regime", "memory")) or (self._approx_contains(tokens, ("regime",)) and self._approx_contains(tokens, ("worst", "best"))):
            score += 4
        return score

    def _intent_score_strategy_outcome_review(self, lowered: str, tokens: list[str]) -> int:
        score = 0
        if self._matches_any(lowered, "what does outcome memory say about this strategy", "what does outcome memory say about strategy"):
            score += 6
        if self._approx_contains(tokens, ("outcome", "memory")) and self._approx_contains(tokens, ("strategy",)):
            score += 4
        return score

    def _load_comparison(self, status: dict[str, Any]) -> dict[str, Any]:
        comparison = dict(status.get("comparison", {}) or {})
        if comparison:
            return comparison
        if hasattr(self.source, "load_comparison"):
            loaded = dict(self.source.load_comparison() or {})
            if loaded:
                return loaded
        return {}

    def _reply(
        self,
        *,
        intent: str,
        acknowledge: str,
        understanding: str,
        data_sources: list[str],
        result: str,
        approval: str,
        response_kind: str,
        reasoning_mode: str | None = None,
    ) -> AtlasReply:
        return AtlasReply(
            mode="read_only_operator_assistant",
            intent=intent,
            reasoning_mode=str(reasoning_mode or getattr(self, "_active_reasoning_mode", "STATE")),
            acknowledge=acknowledge,
            understanding=understanding,
            data_sources=data_sources,
            result=result,
            approval=approval,
            response_kind=response_kind,
        )

    def _sir(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return "Sir."
        if raw.lower().startswith("sir,"):
            return raw
        return f"Sir, {raw}"

    def _task_ack(self, text: str) -> str:
        return self._sir(f"understood. {text}")

    def _task_understanding(self, text: str) -> str:
        return text

    def _shared_approval(self) -> str:
        return "No approval is needed for this read-only research task. Any trading action, threshold change, guard change, or runtime mutation remains separate and approval-controlled."

    def _outcome_intelligence(self, learning_state: dict[str, Any]) -> dict[str, Any]:
        return dict(dict(learning_state.get("state", {}) or {}).get("trade_outcome_intelligence", {}) or {})

    def _learning_sample_limit_line(self, intelligence: dict[str, Any]) -> str:
        sample_count = int(intelligence.get("sample_count", 0) or 0)
        if sample_count < 8:
            return self._sir(f"Limitation: outcome memory is still thin with only {sample_count} labeled trade records, so this read is directional rather than fully reliable.")
        if sample_count < 20:
            return self._sir(f"Limitation: outcome memory has {sample_count} labeled trade records, so the pattern is useful but still early.")
        return ""

    def _top_pattern_line(self, patterns: dict[str, Any], dimension: str, label: str) -> str:
        rows = list(dict(patterns.get("top_patterns", {}) or {}).get(dimension, []) or [])
        if not rows:
            return ""
        best = dict(rows[0] or {})
        return f"{label} {best.get('value', 'unknown')} ({int(best.get('samples', 0) or 0)} samples, win rate {float(best.get('win_rate', 0.0) or 0.0):.2f}, avg pnl {float(best.get('avg_pnl_zar', 0.0) or 0.0):+.2f})."

    def _current_setup_signature(self, status: dict[str, Any]) -> dict[str, str]:
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        chosen = dict(status.get("chosen", {}) or {})
        return {
            "strategy": str(chosen.get("strategy", "unknown") or "unknown").split("(", 1)[0].strip().lower(),
            "session": str(context.get("session", "unknown") or "unknown").strip().lower(),
            "regime": str(regime.get("name", context.get("regime", "unknown")) or "unknown").strip().lower(),
            "volatility_bucket": (
                "low"
                if float(context.get("volatility_pct", regime.get("volatility_pct", 0.0)) or 0.0) < 0.02
                else ("normal" if float(context.get("volatility_pct", regime.get("volatility_pct", 0.0)) or 0.0) < 0.05 else "high")
            ),
        }

    def _action_component(self, lowered: str, tokens: list[str]) -> str:
        if self._approx_contains(tokens, ("consensus",)):
            return "consensus"
        if self._matches_any(lowered, "loss-pattern", "loss pattern") or (
            self._approx_contains(tokens, ("loss", "pattern")) and self._approx_contains(tokens, ("protection", "memory", "guard"))
        ):
            return "loss_pattern_protection"
        if self._approx_contains(tokens, ("signal", "signals", "quality")):
            return "signal_quality"
        if self._approx_contains(tokens, ("strategy", "strategies", "momentum")):
            return "strategy"
        if self._approx_contains(tokens, ("guard", "guards", "filter", "filters", "blocker", "blockers")):
            return "guards"
        return "general"

    def _refuse_mutation(self, query: str) -> AtlasReply:
        return self._reply(
            intent="refusal_mutating_request",
            acknowledge="I received a request that would change trading or runtime behavior.",
            understanding=f"The request was: {query}",
            data_sources=["permission model", "operator-control policy"],
            result="I can explain the current state, analyze logs, summarize research, and propose the safest next step, but I will not place trades, enable live execution, change thresholds, disable guards, or mutate runtime state through Ask Atlas.",
            approval="These are approval-controlled actions and must remain separate from this interface. Ask for an analysis or a proposed change instead.",
            response_kind="question_answer",
        )

    def _favorable_unfavorable(self, status: dict[str, Any], events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        chosen = dict(status.get("chosen", {}) or {})
        higher_tf_bias = str(status.get("higher_tf_bias", "unknown") or "unknown")
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
        threshold = status.get("min_confidence_threshold")
        confidence = _safe_float(chosen.get("confidence", 0.0))
        if threshold is not None and confidence < _safe_float(threshold):
            unfavorable.append("confidence is below the current execution threshold")
        top_blocks = self._top_block_reasons(events)
        if top_blocks:
            unfavorable.append(f"recent blocks are led by {top_blocks[0][0].rstrip('.')}")
        return favorable, unfavorable

    def _condition_direction(self, status: dict[str, Any], events: list[dict[str, Any]]) -> str:
        context = dict(status.get("context_snapshot", {}) or {})
        favorable, unfavorable = self._favorable_unfavorable(status, events)
        if favorable and not unfavorable:
            return "conditions look favorable overall."
        if unfavorable and not favorable:
            return "conditions look worse than favorable right now."
        if str(context.get("spread_state", "")) == "tight" and str(context.get("liquidity_state", "")) == "normal":
            return "conditions are improving but not clean enough yet."
        return "conditions are mixed or still unclear."

    def _top_block_reasons(self, events: list[dict[str, Any]]) -> list[tuple[str, int]]:
        reasons: Counter[str] = Counter()
        for event in events:
            event_type = str(event.get("type", "") or "").lower()
            if event_type not in {"risk", "blocked", "paper_filtered"}:
                continue
            message = str(event.get("message", "") or "").strip()
            if not message:
                continue
            reasons[_first_sentence(message)] += 1
        return reasons.most_common(5)

    def _blocker_ranking(self, status: dict[str, Any], events: list[dict[str, Any]]) -> list[tuple[str, int]]:
        ranking: Counter[str] = Counter()
        message = _first_sentence(str(status.get("message", "") or "").strip())
        if message:
            ranking[message] += 2
        for reason, count in self._top_block_reasons(events):
            ranking[reason] += count
        return ranking.most_common(3)

    def _setup_evaluation_data(self, status: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        chosen = dict(status.get("chosen", {}) or {})
        best_actionable = dict(status.get("best_actionable_candidate", {}) or {})
        decision = dict(status.get("decision_explanation", {}) or {})
        shadow = dict(status.get("shadow_advice", {}) or {})
        candidate = chosen or best_actionable
        side = str(candidate.get("side", "flat") or "flat")
        higher_tf_bias = str(status.get("higher_tf_bias", candidate.get("higher_tf_bias", "unknown")) or "unknown")
        threshold = status.get("min_confidence_threshold")
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
                str(status.get("message", "") or ""),
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

        if str(status.get("action", "") or "") == "blocked":
            score -= 1

        top_blocks = self._top_block_reasons(events)
        if top_blocks:
            issues.append(f"recent blocked conditions are still led by {top_blocks[0][0].rstrip('.')}")

        if score >= 3:
            quality = "strong"
        elif score >= 1:
            quality = "moderate"
        else:
            quality = "weak"

        return {
            "candidate": candidate,
            "quality": quality,
            "positives": positives,
            "issues": issues,
            "missing": list(dict.fromkeys(x for x in missing if x)),
            "issue_types": list(dict.fromkeys(x for x in issue_types if x)),
            "higher_tf_bias": higher_tf_bias,
            "confidence": confidence,
            "threshold": threshold,
        }

    def _current_vs_recent_blocked(self, status: dict[str, Any], events: list[dict[str, Any]]) -> str:
        context = dict(status.get("context_snapshot", {}) or {})
        top_blocks = self._top_block_reasons(events)
        current_bits = [
            f"regime {str(status.get('regime', {}).get('name', context.get('regime', 'unknown')))}",
            f"session {str(context.get('session', 'unknown'))}",
            f"spread {str(context.get('spread_state', 'unknown'))}",
            f"liquidity {str(context.get('liquidity_state', 'unknown'))}",
        ]
        if not top_blocks:
            return "I do not have enough recent blocked-condition history to compare the current chart against past blocked states yet."
        return (
            f"Current conditions look like {'; '.join(current_bits)}. "
            f"Recent blocked conditions are still dominated by {top_blocks[0][0].rstrip('.')}, "
            "so the current setup still resembles blocked recent history more than a clean pass."
        )

    def _self_assessment(self, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any]) -> dict[str, Any]:
        score = 25
        strengths: list[str] = []
        limits: list[str] = []
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        drift = dict(status.get("drift", {}) or {})
        win_rate_focus = dict(status.get("win_rate_focus", {}) or {})
        decision = dict(status.get("decision_explanation", {}) or {})
        health = dict(status.get("system_health", {}) or {})
        chosen = dict(status.get("chosen", {}) or {})
        top_blocks = self._top_block_reasons(events)

        if regime or context:
            score += 15
            strengths.append("identify trend and regime from the saved market state")
        if any(context.get(key) not in {None, "", "unknown"} for key in ("volatility_state", "spread_state", "liquidity_state")):
            score += 15
            strengths.append("evaluate volatility, spread, and liquidity conditions")
        if decision.get("final_decision") or top_blocks:
            score += 15
            strengths.append("detect blocking conditions and explain why trades are not passing")
        if chosen and _safe_float(chosen.get("confidence", 0.0)) > 0:
            score += 5
            strengths.append("measure current setup confidence against a threshold")
        if comparison and (comparison.get("control_metrics") or comparison.get("experiment_metrics")):
            score += 10
            strengths.append("compare control and experiment evidence when a bound run is active")

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
            limits.append("the system is still seeing blocked or low-quality conditions more often than confirmed opportunities")

        if not bool(health.get("paper_mode_confirmed", False)):
            limits.append("paper-mode safety confirmation is incomplete")
        else:
            limits.append("the system is still in paper evaluation rather than validated live deployment")

        strengths = list(dict.fromkeys(strengths))
        limits = list(dict.fromkeys(limits))
        score = max(10, min(85, score))
        if score >= 70:
            rating = "fairly strong"
        elif score >= 50:
            rating = "structured but still developing"
        else:
            rating = "limited and still early-stage"
        return {"score": score, "rating": rating, "strengths": strengths, "limits": limits}

    def _comparison_focus(self, comparison: dict[str, Any]) -> str:
        if not comparison:
            return "There is no active comparison bound to the current session."
        lines = _comparison_lines(comparison)
        tracker = dict(comparison.get("tracker", {}) or {})
        if tracker:
            filtered = _safe_int(tracker.get("filtered_by_experiment", 0))
            if filtered:
                lines.append(f"Tracker shows {filtered} trades filtered by the experiment lane.")
        return " ".join(lines) if lines else "Comparison data is present but limited."

    def _comparison_condition_line(self, comparison: dict[str, Any]) -> str:
        if not comparison:
            return "There is no active comparison run, so I cannot compare current conditions to control-versus-experiment behavior yet."
        control_status = dict(comparison.get("control_status", {}) or {})
        experiment_status = dict(comparison.get("experiment_status", {}) or {})
        if not control_status and not experiment_status:
            return "Comparison artifacts are present, but they do not contain enough lane state to compare conditions."
        control_msg = _first_sentence(str(control_status.get("message", "") or "")) or "no control message recorded"
        experiment_msg = _first_sentence(str(experiment_status.get("message", "") or "")) or "no experiment message recorded"
        return f"Control lane most recently shows {control_msg.lower()}, while the experiment lane shows {experiment_msg.lower()}."

    def _chart_waiting_line(self, status: dict[str, Any], events: list[dict[str, Any]]) -> str:
        action = str(status.get("action", "") or "")
        message = str(status.get("message", "") or "")
        if action in {"blocked", "hold"} and message:
            return f"I am waiting for the blocking condition to clear, especially: {message.rstrip('.')}."
        if action == "paper":
            return "I am waiting for a valid setup to remain strong enough through all checks, even though execution is currently in paper mode."
        top_blocks = self._top_block_reasons(events)
        if top_blocks:
            return f"I am mainly waiting for cleaner conditions because the most common recent blocker is {top_blocks[0][0].rstrip('.')}."
        return "I am waiting for stronger alignment across trend, confidence, and trade-safety checks before I can act."

    def _chart_change_needed_line(self, status: dict[str, Any], chosen: dict[str, Any], best_actionable: dict[str, Any], threshold: Any) -> str:
        context = dict(status.get("context_snapshot", {}) or {})
        needs: list[str] = []
        if str(context.get("spread_state", "unknown")) == "wide":
            needs.append("tighter spread")
        if str(context.get("liquidity_state", "unknown")) in {"poor", "below_normal"}:
            needs.append("better liquidity")
        conf = _safe_float(chosen.get("confidence", best_actionable.get("confidence", 0.0)))
        if threshold is not None and conf < _safe_float(threshold):
            needs.append("stronger confidence")
        higher_tf_bias = str(status.get("higher_tf_bias", best_actionable.get("higher_tf_bias", "unknown")) or "unknown")
        side = str(chosen.get("side", best_actionable.get("side", "flat")) or "flat")
        if higher_tf_bias in {"buy", "sell"} and side in {"buy", "sell"} and higher_tf_bias != side:
            needs.append("better higher-timeframe alignment")
        if not needs:
            return "I need the current setup to stay clean and consistent long enough to pass every guard without being blocked."
        if len(needs) == 1:
            return f"I need {needs[0]} before I can consider the setup valid."
        return f"I need {', '.join(needs[:-1])}, and {needs[-1]} before I can consider the setup valid."

    def _handle_understanding(self, query: str, status: dict[str, Any], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        action = str(status.get("action", "idle") or "idle")
        message = str(status.get("message", "") or "")
        extra = _comparison_lines(comparison)
        result_lines = [
            "I understand the task style: you want Atlas to behave like a disciplined operator assistant that can explain, research, summarize, and plan from current system state.",
            f"Right now I am in `{action}` state.",
        ]
        if message:
            result_lines.append(f"Current focus: {_first_sentence(message)}")
        result_lines.extend(extra[:2])
        if simplify:
            result_lines = [
                "I understand you, sir.",
                f"Right now I am in `{action}` state.",
                f"My current focus is {_first_sentence(message) or 'watching the system state.'}",
            ]
        return self._reply(
            intent="understanding",
            acknowledge="I understand the request, sir.",
            understanding="You want me to restate tasks clearly, use grounded internal data, perform read-only analysis, and tell you when approval would be required for anything beyond explanation or research.",
            data_sources=["current status", "comparison summary when active"],
            result=" ".join(result_lines),
            approval="No approval is needed for read-only explanations or summaries. Any execution or configuration change still requires separate explicit approval.",
            response_kind="question_answer",
        )

    def _handle_chart_understanding(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], simplify: bool = False) -> AtlasReply:
        market_read = self._brain.market_read()
        setup_assessment = self._brain.setup_assessment()
        confidence_assessment = self._brain.confidence_assessment()
        action = str(status.get("action", "unknown") or "unknown")
        message = str(status.get("message", "") or "")
        chosen = dict(status.get("chosen", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        context = dict(status.get("context_snapshot", {}) or {})
        best_actionable = dict(status.get("best_actionable_candidate", {}) or {})
        higher_tf_bias = str(status.get("higher_tf_bias", best_actionable.get("higher_tf_bias", "unknown")) or "unknown")
        setup_quality = status.get("setup_quality_filter", {})
        setup_score = ""
        if isinstance(setup_quality, dict) and setup_quality.get("score") is not None:
            setup_score = f"{_safe_float(setup_quality.get('score', 0.0)):.1f}"
        confidence = chosen.get("confidence", best_actionable.get("confidence", 0.0))
        threshold = status.get("min_confidence_threshold", None)
        wait_line = self._chart_waiting_line(status, events)
        change_line = self._chart_change_needed_line(status, chosen, best_actionable, threshold)
        if simplify:
            result_parts = [
                self._sir(
                    "my current read on the chart is "
                    f"{str(regime.get('name', context.get('regime', 'unclear')))} conditions in the {str(context.get('session', 'unknown'))} session."
                ),
                self._sir(
                    f"the market looks {str(context.get('liquidity_state', 'unknown'))} on liquidity, "
                    f"{str(context.get('spread_state', 'unknown'))} on spread, and {str(context.get('volatility_state', 'unknown'))} on volatility."
                ),
                self._sir(f"my higher-timeframe bias is {higher_tf_bias}, and my current action is {action}."),
            ]
        else:
            result_parts = [
                self._sir(
                    "my current read on the chart is "
                    f"{str(regime.get('name', context.get('regime', 'unclear')))} conditions during the {str(context.get('session', 'unknown'))} session."
                ),
                self._sir(
                    "I see "
                    f"trend strength around {_safe_float(regime.get('trend_strength', context.get('trend_strength', 0.0))):.3f}, "
                    f"{str(context.get('volatility_state', 'unknown'))} volatility, "
                    f"{str(context.get('spread_state', 'unknown'))} spread, "
                    f"and {str(context.get('liquidity_state', 'unknown'))} liquidity."
                ),
                self._sir(f"my higher-timeframe bias is {higher_tf_bias}, and my current action is {action}."),
                self._sir("My brain layer reads the market as " + str(market_read.summary).rstrip(".")),
            ]
        if chosen:
            result_parts.append(
                self._sir(
                    f"the current candidate is {str(chosen.get('strategy', 'unknown'))} {str(chosen.get('side', 'flat'))} "
                    f"with confidence {_safe_float(confidence):.3f}."
                )
            )
        elif best_actionable:
            result_parts.append(
                self._sir(
                    f"the best nearby setup is {str(best_actionable.get('strategy', 'unknown'))} {str(best_actionable.get('side', 'flat'))} "
                    f"with confidence {_safe_float(best_actionable.get('confidence', 0.0)):.3f}, but it is not ready yet."
                )
            )
        if setup_score:
            result_parts.append(self._sir(f"the latest setup-quality score I can see is {setup_score}."))
        if message:
            if action in {"blocked", "hold", "idle", "paper", "error"}:
                result_parts.append(self._sir(f"I am not taking a trade because {message.rstrip('.')}." ))
            else:
                result_parts.append(self._sir(f"the latest decision message is {message.rstrip('.')}." ))
        if threshold is not None:
            result_parts.append(self._sir(f"to act, I still need confidence and alignment to clear the current threshold of {_safe_float(threshold):.3f}."))
        result_parts.append(self._sir(str(confidence_assessment.summary)))
        if setup_assessment.details.get("issue_types"):
            result_parts.append(self._sir("The main setup concerns are " + ", ".join(list(setup_assessment.details.get("issue_types", []))[:4]) + "."))
        result_parts.append(self._sir(wait_line))
        result_parts.append(self._sir(change_line))
        return self._reply(
            intent="chart_understanding",
            acknowledge="I’m reading the chart from current Atlas state, sir.",
            understanding="You want a plain-English explanation of what Atlas sees on the chart, what condition it believes the market is in, why it is or is not trading, and what it still needs before acting.",
            data_sources=["current status", "market snapshot", "context snapshot", "decision explanation", "recent blocking events"],
            result=" ".join(result_parts),
            approval="No approval is needed for this chart read. Any change to trading behavior, thresholds, guards, or execution remains separate and approval-controlled.",
            response_kind="question_answer",
        )

    def _handle_block_reasons(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        action = str(status.get("action", "") or "")
        message = str(status.get("message", "") or "No live status message is available.")
        chosen = dict(status.get("chosen", {}) or {})
        health = dict(status.get("system_health", {}) or {})
        warnings = list(health.get("warnings", []) or [])
        top_blocks = self._top_block_reasons(events)
        result_lines: list[str] = []
        if action in {"blocked", "hold", "idle", "paper", "error"}:
            result_lines.append(f"Atlas is not trading because the current action is `{action}`.")
        else:
            result_lines.append(f"Current action is `{action or 'unknown'}`.")
        result_lines.append(f"Latest reason: {message}")
        if chosen:
            result_lines.append(
                f"Chosen candidate: {str(chosen.get('strategy', 'unknown'))} {str(chosen.get('side', 'flat'))} at confidence {_safe_float(chosen.get('confidence', 0.0)):.3f}."
            )
        if warnings:
            result_lines.append(f"System health warnings: {', '.join(str(x) for x in warnings[:4])}.")
        if top_blocks:
            result_lines.append("Most common recent block reasons: " + ", ".join(f"{reason} ({count})" for reason, count in top_blocks[:3]) + ".")
        lines = _comparison_lines(comparison)
        if lines:
            result_lines.append(lines[0])
        if simplify:
            result_lines = [
                f"Sir, I am not trading because the current action is `{action}`.",
                f"Sir, the latest reason is {message}",
                self._sir(self._chart_waiting_line(status, events)),
            ]
        return self._reply(
            intent="block_reasons",
            acknowledge="I’m checking why Atlas is not trading or what it is waiting on, sir.",
            understanding="You want a grounded explanation of the current blockers or wait conditions from live status and recent events.",
            data_sources=["current status", "recent events", "system health", "comparison summary when active"],
            result=" ".join(result_lines),
            approval="No approval is needed for this explanation. If you want a threshold or guard changed, that remains approval-controlled and outside Ask Atlas.",
            response_kind="question_answer",
        )

    def _handle_market_context(self, query: str, status: dict[str, Any], simplify: bool = False) -> AtlasReply:
        market = dict(status.get("market", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        context = dict(status.get("context_snapshot", {}) or {})
        closes = [float(x) for x in list(market.get("recent_close", []) or []) if isinstance(x, (int, float))]
        trend_line = "I do not have enough recent price samples in status to study the chart yet."
        if len(closes) >= 2:
            delta = closes[-1] - closes[0]
            pct = (delta / max(abs(closes[0]), 1e-9)) * 100.0
            direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            trend_line = f"Recent saved chart movement is {direction} by {pct:.3f}% over the current visible sample."
        result_lines = [
            trend_line,
            f"Regime: {str(regime.get('name', context.get('regime', 'unknown')))}.",
            f"Session: {str(context.get('session', 'unknown'))}.",
            f"Trend strength: {_safe_float(regime.get('trend_strength', context.get('trend_strength', 0.0))):.3f}.",
            f"Volatility state: {str(context.get('volatility_state', 'unknown'))}. Spread state: {str(context.get('spread_state', 'unknown'))}. Liquidity state: {str(context.get('liquidity_state', 'unknown'))}.",
        ]
        if simplify:
            result_lines = [
                self._sir(trend_line[0].lower() + trend_line[1:] if trend_line else "I do not have enough saved chart data yet."),
                self._sir(f"the market looks {str(regime.get('name', context.get('regime', 'unknown')))} right now."),
                self._sir(f"session is {str(context.get('session', 'unknown'))}, with {str(context.get('volatility_state', 'unknown'))} volatility and {str(context.get('liquidity_state', 'unknown'))} liquidity."),
            ]
        return self._reply(
            intent="market_context",
            acknowledge="I’m studying the current Atlas market context from saved status and chart data, sir.",
            understanding="You want a plain-language read of the current trend or current saved chart state, not a new trading action.",
            data_sources=["current market snapshot", "recent saved close prices", "context snapshot"],
            result=" ".join(result_lines),
            approval="No approval is needed for market-context analysis. Trading decisions remain separate from this interface.",
            response_kind="question_answer",
        )

    def _handle_run_summary(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        counter = Counter(str(event.get("type", "") or "unknown") for event in events)
        recent_messages = [str(event.get("message", "") or "") for event in events[-5:]]
        result_lines = [
            f"Current status is `{str(status.get('action', 'unknown'))}` with message: {str(status.get('message', 'No message available.'))}",
            "Recent event mix: " + ", ".join(f"{name}={count}" for name, count in counter.most_common(5)) if counter else "No recent events are available yet.",
        ]
        if recent_messages:
            result_lines.append("Latest timeline: " + " | ".join(recent_messages[-3:]))
        top_blocks = self._top_block_reasons(events)
        if top_blocks:
            result_lines.append("Largest no-trade causes recently: " + ", ".join(f"{reason} ({count})" for reason, count in top_blocks[:3]) + ".")
        result_lines.extend(_comparison_lines(comparison)[:2])
        if simplify:
            result_lines = [
                f"Sir, current status is `{str(status.get('action', 'unknown'))}`.",
                f"Sir, latest message: {str(status.get('message', 'No message available.'))}",
            ]
            if top_blocks:
                result_lines.append(f"Sir, the biggest no-trade cause recently is {top_blocks[0][0]}.")
        return self._reply(
            intent="run_summary",
            acknowledge="I’m summarizing what Atlas has done so far from the latest stored run state, sir.",
            understanding="You want a concise research summary of the current run, recent events, and any recurring block patterns.",
            data_sources=["current status", "recent events", "comparison summary when active"],
            result=" ".join(result_lines),
            approval="No approval is needed for summaries. If you want me to propose a concrete experiment change, I can do that in planning mode without mutating state.",
            response_kind="question_answer",
        )

    def _handle_comparison(self, query: str, comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        if not comparison:
            result = "There is no active comparison artifact bound to the current UI session, so I cannot compare control vs experiment from live data right now."
        else:
            lines = _comparison_lines(comparison)
            assessment = str(dict(comparison.get("filter_effectiveness", {}) or {}).get("assessment", "") or "")
            if assessment:
                lines.append(f"Filter effectiveness assessment: {assessment}.")
            result = " ".join(lines) if lines else "Comparison data is present but sparse."
        if simplify and result:
            result = self._sir(result)
        return self._reply(
            intent="comparison_summary",
            acknowledge="I’m checking the current control-versus-experiment artifacts, sir.",
            understanding="You want a grounded comparison summary from the bound paper-comparison run, not a new experiment launch.",
            data_sources=["comparison summary", "comparison report", "bound experiment status when present"],
            result=result,
            approval="No approval is needed to read comparison results. Starting or changing an experiment remains a separate operator-controlled action.",
            response_kind="question_answer",
        )

    def _handle_planning(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        top_blocks = self._top_block_reasons(events)
        setup_eval = self._brain.setup_assessment().details
        research_plan = self._brain.research_plan()
        checklist = dict(comparison.get("decision_checklist", {}) or {})
        learning = dict(learning_state.get("state", {}) or {})
        setup_pool = dict(learning.get("evolved_strategy_pool", {}) or {})
        recommendations: list[str] = []
        plan_steps: list[str] = [
            "first analyze current market structure and chart context",
            "then review repeated block conditions and recent no-trade patterns",
            "then check volatility, spread, liquidity, and higher-timeframe alignment",
            "finally identify high-quality setups and propose the next safe offline experiment",
        ]
        if top_blocks:
            recommendations.append(f"First priority: investigate the top no-trade cause `{top_blocks[0][0]}` because it appears {top_blocks[0][1]} times in recent events.")
            recommendations.append(f"Highest-impact research question right now: determine whether `{top_blocks[0][0]}` is a single dominant blocker or part of a stacked filter sequence.")
        if comparison:
            recommendations.append("Second priority: keep the current paper comparison isolated until the checklist shows a clear promote-or-reject outcome.")
        if setup_eval["issue_types"]:
            recommendations.append(f"Biggest likely performance drag from current setup quality is {setup_eval['issue_types'][0]}.")
        if setup_pool.get("count", 0):
            recommendations.append(f"Third priority: review the {int(setup_pool.get('count', 0))} evolved candidates offline before considering any promotion.")
        if not recommendations:
            recommendations.append("Safest next step: collect more paper-mode evidence before changing trading logic.")
        classification = str(checklist.get("classification", "") or "")
        result_lines = recommendations[:3]
        if classification:
            result_lines.append(f"Current experiment classification is {classification}, so the next experiment should stay offline or paper-only.")
        result_lines.append("Recommended next offline experiment: isolate one variable only, keep live execution disabled, and measure whether blocked trades improve expectancy rather than only win rate.")
        reasoning_mode = getattr(self, "_active_reasoning_mode", "STATE")
        if reasoning_mode == "PLANNING":
            result_lines = [
                "Sir, if given time to work through this safely, I would " + ", ".join(plan_steps[:-1]) + ", and " + plan_steps[-1] + ".",
                "Sir, I would use current status, recent events, context snapshot, system health, and comparison evidence when available.",
            ] + [self._sir("My current research plan is: " + " | ".join(list(research_plan.details.get("steps", []))[:4]) + ".")] + result_lines
        if simplify:
            result_lines = [
                "Sir, the safest next step is still an offline or paper-only test.",
                result_lines[0],
                "Sir, keep one variable changed at a time and keep live execution disabled.",
            ]
        return self._reply(
            intent="planning",
            acknowledge="I’m preparing the safest next planning step from current evidence, sir.",
            understanding="You want Atlas to propose the next research or experiment step without changing live behavior.",
            data_sources=["recent events", "comparison checklist", "learning state"],
            result=" ".join(result_lines),
            approval="Planning needs no approval. Creating or applying a real experiment configuration is still a separate explicit action outside Ask Atlas.",
            response_kind="question_answer",
        )

    def _handle_chart_study(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        market_read = self._brain.market_read()
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        action = str(status.get("action", "unknown") or "unknown")
        message = str(status.get("message", "") or "")
        higher_tf_bias = str(status.get("higher_tf_bias", "unknown") or "unknown")
        favorable = list(market_read.details.get("favorable", []) or [])
        unfavorable = list(market_read.details.get("unfavorable", []) or [])
        result_lines = [
            self._sir(f"after studying the chart, I see {str(regime.get('name', context.get('regime', 'unclear')))} conditions in the {str(context.get('session', 'unknown'))} session, with higher-timeframe bias {higher_tf_bias}."),
            self._sir(f"trend strength is {_safe_float(regime.get('trend_strength', context.get('trend_strength', 0.0))):.3f}, volatility is {str(context.get('volatility_state', 'unknown'))}, spread is {str(context.get('spread_state', 'unknown'))}, and liquidity is {str(context.get('liquidity_state', 'unknown'))}."),
            self._sir(
                f"current action is {action}"
                + (f", and the current block reason is {message.rstrip('.')}" if message else "")
                + "."
            ),
            self._sir(self._chart_waiting_line(status, events)),
        ]
        if favorable:
            result_lines.append(self._sir("What looks favorable: " + ", ".join(favorable[:3]) + "."))
        if unfavorable:
            result_lines.append(self._sir("What looks unfavorable: " + ", ".join(unfavorable[:3]) + "."))
        result_lines.append(self._sir("Overall, " + str(market_read.details.get("direction", self._condition_direction(status, events)))))
        if simplify:
            result_lines = [
                self._sir(f"after studying the chart, I see {str(regime.get('name', context.get('regime', 'unclear')))} conditions."),
                self._sir(f"the good part is {', '.join(favorable[:2]) if favorable else 'there are not many clean positives yet'}."),
                self._sir(f"the main problem is {', '.join(unfavorable[:2]) if unfavorable else 'the setup still is not fully clear'}."),
                self._sir(self._chart_waiting_line(status, events)),
            ]
        return self._reply(
            intent="chart_study",
            acknowledge=self._task_ack("I will study the current chart and recent state using read-only data."),
            understanding=self._task_understanding("You want a grounded chart study, not just a short answer."),
            data_sources=["current status", "context snapshot", "decision explanation", "system health", "recent events", "comparison summary when active"],
            result=" ".join(result_lines),
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_market_review(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        context = dict(status.get("context_snapshot", {}) or {})
        regime = dict(status.get("regime", {}) or {})
        result_lines = [
            self._sir(f"after reviewing the market right now, I see {str(regime.get('name', context.get('regime', 'unknown')))} conditions in the {str(context.get('session', 'unknown'))} session."),
            self._sir(f"volatility is {str(context.get('volatility_state', 'unknown'))}, spread is {str(context.get('spread_state', 'unknown'))}, and liquidity is {str(context.get('liquidity_state', 'unknown'))}."),
            self._sir(self._chart_waiting_line(status, events)),
            self._sir("Overall, " + self._condition_direction(status, events)),
        ]
        if simplify:
            result_lines = [
                self._sir(f"right now the market looks {str(regime.get('name', context.get('regime', 'unknown')))}."),
                self._sir(f"spread is {str(context.get('spread_state', 'unknown'))} and liquidity is {str(context.get('liquidity_state', 'unknown'))}."),
                self._sir(self._chart_waiting_line(status, events)),
            ]
        return self._reply(
            intent="market_review",
            acknowledge=self._task_ack("I will review the current market using saved live state."),
            understanding=self._task_understanding("You want a read-only market review of the current conditions."),
            data_sources=["current status", "context snapshot", "system health", "recent events"],
            result=" ".join(result_lines),
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_block_reason_analysis(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        top_blocks = self._top_block_reasons(events)
        blocker = self._brain.blocker_rank()
        ranked = list(blocker.details.get("ranking", []) or [])
        action = str(status.get("action", "unknown") or "unknown")
        result_lines = [self._sir(f"after analyzing recent block reasons, the current action is {action}.")]
        if ranked:
            labels = ["Primary blocker", "Secondary blocker", "Tertiary blocker"]
            for idx, (reason, count) in enumerate(ranked[:3]):
                result_lines.append(self._sir(f"{labels[idx]}: {reason.rstrip('.')} ({count})."))
            if bool(blocker.details.get("dominant", False)):
                result_lines.append(self._sir(f"One blocker is dominating: {ranked[0][0].rstrip('.')}." ))
            else:
                result_lines.append(self._sir("Multiple blockers are stacking together rather than one guard fully dominating."))
            result_lines.append(self._sir(f"The factor hurting trade quality most right now appears to be {ranked[0][0].rstrip('.')}." ))
        elif top_blocks:
            result_lines.append(self._sir("The main blockers are " + ", ".join(f"{reason} ({count})" for reason, count in top_blocks[:3]) + "."))
        else:
            result_lines.append(self._sir("I do not have enough recent blocking events to rank a dominant blocker yet."))
        result_lines.append(self._sir(self._chart_waiting_line(status, events)))
        lines = _comparison_lines(comparison)
        if lines:
            result_lines.append(self._sir(lines[0]))
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="block_reason_analysis",
            acknowledge=self._task_ack("I will study the recent block reasons using saved events and current status."),
            understanding=self._task_understanding("You want a read-only analysis of what is stopping trades or hurting trade quality most."),
            data_sources=["current status", "recent events", "system health", "comparison summary when active"],
            result=" ".join(result_lines),
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_run_review(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        recent_events = events[-12:]
        type_counter = Counter(str(event.get("type", "") or "unknown") for event in recent_events)
        result_lines = [self._sir(f"after reviewing the run, the current status is {str(status.get('action', 'unknown'))} with message: {str(status.get('message', 'No message available.'))}")]
        if type_counter:
            result_lines.append(self._sir("Recent activity was led by " + ", ".join(f"{name}={count}" for name, count in type_counter.most_common(4)) + "."))
        if recent_events:
            result_lines.append(self._sir("Latest run timeline: " + " | ".join(str(event.get("message", "") or "") for event in recent_events[-3:])))
        if comparison:
            result_lines.append(self._sir(self._comparison_focus(comparison)))
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="run_review",
            acknowledge=self._task_ack("I will review the recent run using current status and recent events."),
            understanding=self._task_understanding("You want a read-only review of what happened recently in the system."),
            data_sources=["current status", "recent events", "comparison summary when active"],
            result=" ".join(result_lines),
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_comparison_review(self, query: str, comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        experiment_brief = self._brain.experiment_brief()
        result = " ".join([str(experiment_brief.summary), self._comparison_focus(comparison), self._comparison_condition_line(comparison)]).strip()
        if simplify:
            result = self._sir(result)
        return self._reply(
            intent="comparison_review",
            acknowledge=self._task_ack("I will compare the active control and experiment data using bound comparison artifacts."),
            understanding=self._task_understanding("You want a read-only comparison review from the active comparison run."),
            data_sources=["comparison_summary.json", "comparison_report.json", "comparison_tracker.json when present"],
            result=result,
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_prioritization(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        blocker = self._brain.blocker_rank()
        ranked = list(blocker.details.get("ranking", []) or [])
        setup_eval = self._brain.setup_assessment().details
        result_lines = [self._sir("here is what matters most right now from current Atlas state.")]
        if ranked:
            labels = ["Primary blocker", "Secondary blocker", "Tertiary blocker"]
            for idx, (reason, count) in enumerate(ranked[:3]):
                result_lines.append(self._sir(f"{labels[idx]}: {reason.rstrip('.')} ({count})."))
        else:
            result_lines.append(self._sir("I do not have enough blocker history yet to rank three factors cleanly."))
        if setup_eval["issue_types"]:
            result_lines.append(self._sir(f"The main thing hurting trade quality is {setup_eval['issue_types'][0]}."))
        if comparison:
            result_lines.append(self._sir(self._comparison_condition_line(comparison)))
        result_lines.append(self._sir("The first thing to focus on is the primary blocker before testing any wider strategy change."))
        if simplify:
            result_lines = result_lines[:4]
        return self._reply(
            intent="prioritization",
            acknowledge="I’m ranking the most important current factors from read-only Atlas state, sir.",
            understanding="You want the main blocker and the order of importance, based on current real system evidence.",
            data_sources=["current status", "recent events", "decision explanation", "context snapshot", "comparison artifacts when active"],
            result=" ".join(result_lines),
            approval="No approval is needed for prioritization analysis. Any real system change remains separate and approval-controlled.",
            response_kind="question_answer",
        )

    def _handle_setup_evaluation(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], simplify: bool = False) -> AtlasReply:
        setup_eval = self._brain.setup_assessment().details
        candidate = dict(setup_eval["candidate"] or {})
        candidate_text = "current setup"
        if candidate:
            candidate_text = f"{str(candidate.get('strategy', 'current setup'))} {str(candidate.get('side', 'flat'))}"
        result_lines = [
            self._sir(f"current setup quality is {setup_eval['quality']} for the {candidate_text} candidate."),
        ]
        if setup_eval["positives"]:
            result_lines.append(self._sir("What looks favorable: " + ", ".join(setup_eval["positives"][:3]) + "."))
        if setup_eval["issues"]:
            result_lines.append(self._sir("What makes the setup weaker: " + ", ".join(setup_eval["issues"][:4]) + "."))
        if setup_eval["missing"]:
            result_lines.append(self._sir("What is missing before it becomes tradable: " + ", ".join(setup_eval["missing"][:4]) + "."))
        if setup_eval["issue_types"]:
            result_lines.append(self._sir("The main issue categories are " + ", ".join(setup_eval["issue_types"][:6]) + "."))
        result_lines.append(self._sir(self._chart_change_needed_line(status, candidate, candidate, setup_eval["threshold"])))
        if simplify:
            result_lines = [
                self._sir(f"current setup quality is {setup_eval['quality']}."),
                self._sir("Main problem: " + (setup_eval["issues"][0] if setup_eval["issues"] else "the setup is still not fully proven") + "."),
                self._sir("Needed next before it becomes tradable: " + (setup_eval["missing"][0] if setup_eval["missing"] else "cleaner alignment") + "."),
            ]
        return self._reply(
            intent="setup_evaluation",
            acknowledge="I’m evaluating the current setup quality from read-only Atlas state, sir.",
            understanding="You want to know whether the setup is weak, moderate, or strong, what is missing, and what would need to improve before Atlas can trade it.",
            data_sources=["current status", "decision explanation", "context snapshot", "shadow advice", "recent events"],
            result=" ".join(result_lines),
            approval="No approval is needed for setup evaluation. Any trade placement or rule change remains blocked from this interface.",
            response_kind="question_answer",
        )

    def _handle_self_assessment(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        assessment = self._brain.self_assessment().details
        self_model = self._brain.self_model().details
        lowered = query.lower()
        result_lines = [
            self._sir(f"I would rate my current chart and market understanding at approximately {assessment['score']}%."),
        ]
        if simplify:
            result_lines.append(self._sir(f"My understanding is {assessment['rating']}, with {assessment.get('reliability', 'limited')} reliability right now."))
            if assessment["limits"]:
                result_lines.append(self._sir("My main limitation right now is " + assessment["limits"][0] + "."))
        else:
            if assessment["strengths"]:
                result_lines.append(self._sir("I am good at: " + "; ".join(assessment["strengths"][:4]) + "."))
            if assessment["limits"]:
                result_lines.append(self._sir("My limitations right now are: " + "; ".join(assessment["limits"][:4]) + "."))
                if "weakness" in lowered:
                    result_lines.append(self._sir("My main weaknesses are the same limitations: " + "; ".join(assessment["limits"][:3]) + "."))
            result_lines.append(self._sir(f"Because of this, my understanding is {assessment['rating']} with {assessment.get('reliability', 'limited')} reliability, not fully validated."))
            result_lines.append(self._sir(f"My current readiness is {str(self_model.get('readiness', 'not ready for live demo'))}."))
        return self._reply(
            intent="self_assessment",
            acknowledge="I’m evaluating my current capability from real Atlas evidence, sir.",
            understanding="You want an honest self-assessment of how well Atlas currently understands the chart or market, with a grounded rating and clear limitations.",
            data_sources=["current status", "decision explanation", "win-rate focus state", "drift state", "recent events", "comparison artifacts when active", "self model"],
            result=" ".join(result_lines),
            approval="No approval is needed for self-assessment. This does not change trading behavior or runtime state.",
            response_kind="question_answer",
        )

    def _handle_capability_assessment(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        self_assessment = self._brain.self_assessment().details
        self_model = self._brain.self_model().details
        blocker = self._brain.blocker_rank().details
        setup = self._brain.setup_assessment().details
        plan = self._brain.research_plan().details
        win_rate_focus = dict(status.get("win_rate_focus", {}) or {})
        drift = dict(status.get("drift", {}) or {})
        ranking = list(blocker.get("ranking", []) or [])
        issue_types = list(setup.get("issue_types", []) or [])
        capability_needs = [
            "consistent signal quality above current levels",
            "reduced dependence on weak-consensus setups becoming the main candidate stream",
            "enough closed trades to validate expectancy and drawdown under current market conditions",
            "stable performance without active drift pressure",
            "reliable detection of favorable regimes and session conditions before signal conviction rises",
        ]
        if ranking:
            capability_needs[1] = f"lower reliance on the current primary blocker, which is `{ranking[0][0].rstrip('.')}`"
        if issue_types:
            capability_needs.append(f"stronger setup quality in the areas now limited by {', '.join(issue_types[:3])}")
        result_lines = [
            self._sir("for me to operate effectively as a day trading system, I would need:"),
            "1. " + capability_needs[0],
            "2. " + capability_needs[1],
            "3. " + capability_needs[2],
            "4. " + capability_needs[3],
            "5. " + capability_needs[4],
        ]
        if not simplify and len(capability_needs) > 5:
            result_lines.append("6. " + capability_needs[5])
        limitations = [
            f"my current validated win rate is {float(win_rate_focus.get('recent_win_rate', 0.0)):.3f} over {int(win_rate_focus.get('samples', 0) or 0)} tracked samples against a target of {float(win_rate_focus.get('target', 0.0)):.3f}",
            f"drift is currently {'active' if bool(drift.get('active', False)) else 'inactive'} with recent 20-trade win rate {float(drift.get('recent_win_rate_20', 0.0)):.2f} versus baseline {float(drift.get('baseline_win_rate', 0.0)):.2f}",
            f"the system is still in {str(status.get('system_health', {}).get('effective_mode', 'paper'))} evaluation, not validated live deployment",
        ]
        if ranking:
            limitations.append(f"the main blocker in my current state is {ranking[0][0].rstrip('.')}")
        if simplify:
            result = " ".join(
                [
                    self._sir("what I need most is stronger signal quality, more validated closed trades, and stable performance without active drift."),
                    self._sir("right now I am limited by weak signal quality and insufficient validated trade outcomes."),
                ]
            )
        else:
            result = " ".join(result_lines)
            result += " " + self._sir(f"My current state is {str(self_model.get('current_state', 'paper evaluation with live execution disabled'))}.")
            result += " " + self._sir("Currently, I am limited by " + "; ".join(limitations[:4]) + ".")
            result += " " + self._sir(f"My readiness statement is: {str(self_model.get('readiness', 'not ready for live demo'))}.")
            priorities = list(plan.get("priorities", []) or [])
            if priorities:
                result += " " + self._sir("The next capability-building priority is " + priorities[0].rstrip(".") + ".")
        return self._reply(
            intent="capability_assessment",
            acknowledge="I’m assessing my own system requirements from current Atlas state, sir.",
            understanding="You are asking what I, as the Atlas system, still need or what is missing before I can operate more effectively.",
            data_sources=["current status", "win-rate focus state", "drift state", "blocker ranking", "setup assessment", "research plan", "self model"],
            result=result,
            approval="No approval is needed for this capability assessment. It does not change live logic, runtime state, or execution permissions.",
            response_kind="question_answer",
        )

    def _handle_action_guidance(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        lowered = query.lower()
        tokens = self._tokens(lowered)
        component = self._action_component(lowered, tokens)
        guidance = self._brain.action_guidance(component).details
        component_label = str(guidance.get("component", component) or component).replace("_", " ")
        display_label = "current blockers" if component_label == "general" else component_label
        recommendation = str(guidance.get("recommendation", "focus on the primary blocker first") or "focus on the primary blocker first")
        result_lines = [
            self._sir(f"Current role: {str(guidance.get('current_role', f'{display_label} is part of the current decision stack'))}."),
            self._sir(f"Diagnosis: it is {str(guidance.get('diagnosis', 'still constrained by current evidence'))}."),
            self._sir(f"Recommendation: my recommendation is to {recommendation}."),
            self._sir("Reasoning: " + str(guidance.get("reasoning", "I am basing that on the current blocker ranking, setup quality, and recent market state."))),
        ]
        next_step = str(guidance.get("next_step", "") or "")
        if next_step and not simplify:
            result_lines.append(self._sir(f"The safest next step is to {next_step.rstrip('.')}." ))
        if comparison and not simplify:
            result_lines.append(self._sir(self._comparison_focus(comparison)))
        if simplify:
            result_lines = result_lines[:4]
        return self._reply(
            intent="action_guidance",
            acknowledge="I'm giving a read-only recommendation from current Atlas evidence, sir.",
            understanding=f"You want actionable guidance about {display_label}, not a generic status update.",
            data_sources=["recent analysis", "blocker ranking", "setup evaluation", "signal-quality diagnostics", "comparison artifacts when active"],
            result=" ".join(result_lines),
            approval="No approval is needed for guidance and recommendations. Any actual rule change, threshold change, guard change, or runtime mutation remains separate and approval-controlled.",
            response_kind="question_answer",
        )

    def _handle_winner_pattern_review(self, query: str, learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        winners = dict(intelligence.get("winner_patterns", {}) or {})
        sample_count = int(winners.get("samples", 0) or 0)
        result_lines = [
            self._sir("Direct answer: past winners usually appear in cleaner market structure with stronger trend and more supportive context.")
        ]
        for dimension, label in (("regime", "Best regime:"), ("session", "Best session:"), ("confidence_bucket", "Best confidence bucket:")):
            line = self._top_pattern_line(winners, dimension, label)
            if line:
                result_lines.append(self._sir(line))
        if sample_count:
            result_lines.append(self._sir(f"Supporting evidence: winner memory currently includes {sample_count} labeled winning trades."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="winner_pattern_review",
            acknowledge="I'm reviewing stored winner patterns from outcome memory, sir.",
            understanding="You want a read-only summary of what past winners have tended to look like.",
            data_sources=["outcome_memory", "winner_patterns", "pattern_comparison"],
            result=" ".join(result_lines),
            approval="No approval is needed for outcome-memory review. This does not change trading behavior or runtime state.",
            response_kind="question_answer",
        )

    def _handle_loser_pattern_review(self, query: str, learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        losers = dict(intelligence.get("loser_patterns", {}) or {})
        comparison = dict(intelligence.get("pattern_comparison", {}) or {})
        sample_count = int(losers.get("samples", 0) or 0)
        result_lines = [
            self._sir("Direct answer: past losers usually show up in weaker or less supportive conditions than the winning set.")
        ]
        for dimension, label in (("regime", "Most common losing regime:"), ("session", "Weak session:"), ("confidence_bucket", "Weak confidence bucket:")):
            line = self._top_pattern_line(losers, dimension, label)
            if line:
                result_lines.append(self._sir(line))
        if comparison.get("main_signal_gap"):
            result_lines.append(self._sir(f"Supporting evidence: {str(comparison.get('main_signal_gap'))}"))
        if sample_count:
            result_lines.append(self._sir(f"Loser memory currently includes {sample_count} labeled losing trades."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="loser_pattern_review",
            acknowledge="I'm reviewing stored loser patterns from outcome memory, sir.",
            understanding="You want a read-only summary of what past losers have tended to look like.",
            data_sources=["outcome_memory", "loser_patterns", "pattern_comparison"],
            result=" ".join(result_lines),
            approval="No approval is needed for outcome-memory review. This stays fully read-only.",
            response_kind="question_answer",
        )

    def _handle_setup_vs_winners(self, query: str, status: dict[str, Any], learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        winners = dict(intelligence.get("winner_patterns", {}) or {})
        signature = self._current_setup_signature(status)
        matches: list[str] = []
        misses: list[str] = []
        for dimension in ("strategy", "session", "regime", "volatility_bucket"):
            rows = list(dict(winners.get("top_patterns", {}) or {}).get(dimension, []) or [])
            top_value = str(dict(rows[0] or {}).get("value", "unknown") or "unknown") if rows else "unknown"
            current_value = str(signature.get(dimension, "unknown") or "unknown")
            if rows and current_value == top_value:
                matches.append(f"{dimension.replace('_', ' ')} matches winner profile ({current_value})")
            else:
                misses.append(f"{dimension.replace('_', ' ')} differs from winner profile ({current_value} vs {top_value})")
        result_lines = [
            self._sir("Direct answer: this setup only looks close to past winners if its current context lines up with the stored winner profile.")
        ]
        if matches:
            result_lines.append(self._sir("Direct answer: it matches past winners on " + ", ".join(matches[:3]) + "."))
        if misses:
            result_lines.append(self._sir("Supporting evidence: it still differs on " + ", ".join(misses[:3]) + "."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="setup_vs_winners",
            acknowledge="I'm comparing the current setup to stored winner memory, sir.",
            understanding="You want to know how the current setup compares with the conditions seen in past winners.",
            data_sources=["outcome_memory", "winner_patterns", "current status", "context snapshot"],
            result=" ".join(result_lines),
            approval="No approval is needed for this comparison. It is read-only analysis only.",
            response_kind="question_answer",
        )

    def _handle_confidence_calibration_review(self, query: str, learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        calibration = dict(intelligence.get("confidence_calibration", {}) or {})
        scoring_model = dict(intelligence.get("signal_scoring_model", {}) or {})
        bins = list(calibration.get("bins", []) or [])
        gap = float(calibration.get("mean_abs_gap", 0.0) or 0.0)
        if gap <= 0.08:
            calibration_read = "fairly well calibrated"
        elif gap <= 0.16:
            calibration_read = "usable but still loose"
        else:
            calibration_read = "not calibrated very well yet"
        result_lines = [
            self._sir(f"Direct answer: my confidence is {calibration_read} based on stored trade outcomes.")
        ]
        informative = [row for row in bins if int(dict(row).get("samples", 0) or 0) > 0]
        if informative:
            best = dict(informative[-1] or {})
            worst = dict(informative[0] or {})
            result_lines.append(
                self._sir(
                    f"Supporting evidence: the {best.get('range')} bucket has {int(best.get('samples', 0) or 0)} samples with win rate {float(best.get('win_rate', 0.0) or 0.0):.2f}, "
                    f"while the weakest bucket has win rate {float(worst.get('win_rate', 0.0) or 0.0):.2f}."
                )
            )
        if scoring_model:
            metrics = dict(scoring_model.get("metrics", {}) or {})
            accuracy = metrics.get("accuracy")
            if accuracy is not None:
                result_lines.append(self._sir(f"Supporting evidence: the stored signal-scoring model is tracking about {float(accuracy or 0.0):.2f} accuracy over {int(scoring_model.get('samples', 0) or 0)} samples."))
        result_lines.append(self._sir(f"Supporting evidence: mean absolute confidence gap is {gap:.2f}."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="confidence_calibration_review",
            acknowledge="I'm checking confidence calibration from stored trade outcomes, sir.",
            understanding="You want a read-only assessment of whether historical confidence levels have matched actual outcomes.",
            data_sources=["confidence_calibration", "outcome_memory", "signal_scoring_model"],
            result=" ".join(result_lines),
            approval="No approval is needed for calibration review. It does not modify execution or runtime state.",
            response_kind="question_answer",
        )

    def _handle_regime_memory_review(self, query: str, learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        regime_memory = dict(intelligence.get("regime_memory", {}) or {})
        favorable = list(regime_memory.get("favorable_regimes", []) or [])
        unfavorable = list(regime_memory.get("unfavorable_regimes", []) or [])
        by_regime = list(regime_memory.get("by_regime", []) or [])
        result_lines = []
        if unfavorable:
            result_lines.append(self._sir(f"Direct answer: the worst regime for recent outcomes has been {unfavorable[0]}."))
        elif by_regime:
            weakest = dict(by_regime[-1] or {})
            result_lines.append(self._sir(f"Direct answer: the weakest regime in outcome memory is {str(weakest.get('value', 'unknown'))} with win rate {float(weakest.get('win_rate', 0.0) or 0.0):.2f}."))
        else:
            result_lines.append(self._sir("Direct answer: I do not have enough regime memory yet to identify a worst regime."))
        if favorable:
            result_lines.append(self._sir(f"Supporting evidence: favorable regimes currently include {', '.join(favorable[:3])}."))
        if by_regime:
            strongest = dict(by_regime[0] or {})
            result_lines.append(self._sir(f"Best stored regime is {str(strongest.get('value', 'unknown'))} over {int(strongest.get('samples', 0) or 0)} samples with average pnl {float(strongest.get('avg_pnl_zar', 0.0) or 0.0):+.2f}."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="regime_memory_review",
            acknowledge="I'm reviewing regime memory from stored trade outcomes, sir.",
            understanding="You want a read-only answer about which regimes have historically been favorable or unfavorable.",
            data_sources=["regime_memory", "outcome_memory", "winner_patterns", "loser_patterns"],
            result=" ".join(result_lines),
            approval="No approval is needed for regime-memory analysis. It remains read-only.",
            response_kind="question_answer",
        )

    def _handle_strategy_outcome_review(self, query: str, status: dict[str, Any], learning_state: dict[str, Any], simplify: bool = False) -> AtlasReply:
        intelligence = self._outcome_intelligence(learning_state)
        records = list(dict(intelligence.get("outcome_memory", {}) or {}).get("trades", []) or [])
        chosen = dict(status.get("chosen", {}) or {})
        strategy = str(chosen.get("strategy", "unknown") or "unknown").split("(", 1)[0].strip().lower()
        filtered = [row for row in records if str(row.get("strategy", "")).split("(", 1)[0].strip().lower() == strategy]
        wins = sum(1 for row in filtered if str(row.get("outcome", "")).lower() == "win")
        losses = sum(1 for row in filtered if str(row.get("outcome", "")).lower() == "loss")
        avg_conf = sum(float(row.get("confidence", 0.0) or 0.0) for row in filtered) / max(1, len(filtered))
        avg_pnl = sum(float(row.get("net_pnl_zar", 0.0) or 0.0) for row in filtered) / max(1, len(filtered))
        result_lines = [
            self._sir(f"Direct answer: outcome memory for {strategy} is {'limited' if len(filtered) < 3 else 'available'} right now.")
        ]
        if filtered:
            result_lines.append(self._sir(f"Supporting evidence: {strategy} has {len(filtered)} labeled outcomes, with {wins} wins, {losses} losses, average confidence {avg_conf:.2f}, and average pnl {avg_pnl:+.2f}."))
        else:
            result_lines.append(self._sir(f"I do not have enough stored outcomes for {strategy} yet."))
        limit = self._learning_sample_limit_line(intelligence)
        if limit:
            result_lines.append(limit)
        if simplify:
            result_lines = result_lines[:3]
        return self._reply(
            intent="strategy_outcome_review",
            acknowledge="I'm reviewing strategy-specific outcome memory, sir.",
            understanding="You want a read-only answer about what outcome memory says for the current or referenced strategy.",
            data_sources=["outcome_memory", "winner_patterns", "loser_patterns"],
            result=" ".join(result_lines),
            approval="No approval is needed for this review. It does not alter live logic or runtime state.",
            response_kind="question_answer",
        )

    def _handle_follow_up(self, query: str, follow_up_mode: str, turn: BrainTurn, simplify: bool = False) -> AtlasReply:
        subject = turn.subject
        blocker = self._brain.blocker_rank()
        market = self._brain.market_read()
        setup = self._brain.setup_assessment()
        confidence = self._brain.confidence_assessment()
        plan = self._brain.research_plan()
        self_assessment = self._brain.self_assessment()

        if follow_up_mode == "simplify":
            if subject == "blockers":
                result = self._sir(f"put simply, the main blocker is {blocker.summary.rstrip('.')}. Until that clears, trades keep failing the safety checks before they can pass.")
            elif subject in {"market", "setup"}:
                result = self._sir(f"put simply, the market is {market.details.get('regime', 'unclear')} and the setup is {setup.details.get('quality', 'unclear')}. We are not close yet because {', '.join(list(setup.details.get('missing', []))[:2]) or 'the setup still lacks clean alignment'}.")
            elif subject == "self_assessment":
                result = self._sir(f"put simply, I can read structure and blockers, but I am only about {self_assessment.details.get('score', 0)}% confident because the evidence is still being validated.")
            else:
                result = self._sir("put simply, I am still using current state to explain the most important blocker and the next safe research step.")
        elif follow_up_mode == "clarify":
            if subject == "blockers":
                result = self._sir(f"by that, I mean the main blocker right now is {blocker.summary.rstrip('.')}. That point leads the recent no-trade pattern, so it is the first thing shaping outcomes.")
            elif subject in {"market", "setup"}:
                result = self._sir(f"by that, I mean the market read is {market.summary.rstrip('.')}, and the setup is currently {setup.details.get('quality', 'unclear')} because of {', '.join(list(setup.details.get('issue_types', []))[:3]) or 'missing alignment'}.")
            elif subject == "planning":
                result = self._sir("by that, I mean the next safe move is still research-only: study the main blocker, compare it to recent conditions, and test one offline change at a time.")
            else:
                result = self._sir("by that, I mean I am grounding the answer in the previous topic and carrying it forward with the current system state.")
        elif follow_up_mode == "next_step":
            if subject in {"blockers", "market", "setup", "planning"}:
                priorities = list(plan.details.get("priorities", []) or [])
                result = self._sir("based on that, what should happen next is a safe research step: " + (priorities[0] if priorities else "review the main blocker first, then test one offline change at a time."))
            else:
                result = self._sir("based on that, the next is a safe research step: keep the review read-only, identify the main blocker clearly, and only then consider an operator-approved change.")
        elif follow_up_mode == "confidence":
            if subject == "self_assessment":
                result = self._sir(f"I am about {self_assessment.details.get('score', 0)}% confident in that self-assessment. The number is limited by validation depth, drift, and how many confirmed outcomes we have.")
            else:
                result = self._sir(str(confidence.summary) + " I am basing that on the current setup quality, threshold gap, and recent blocker pattern.")
        elif follow_up_mode == "importance":
            if subject == "blockers":
                result = self._sir(f"the most important part is that {blocker.summary.rstrip('.')}. That is the first thing to focus on before worrying about secondary improvements.")
            else:
                result = self._sir("the most important part is the first constraint that still stops good setups from passing safely.")
        elif follow_up_mode == "why_matters":
            if subject == "blockers":
                result = self._sir(f"that matters because {blocker.summary.rstrip('.')} is currently the main reason trades are being blocked. Until it improves, even decent-looking setups may still fail before execution.")
            elif subject in {"market", "setup"}:
                result = self._sir(f"that matters because the current setup is {setup.details.get('quality', 'unclear')}. If the key missing conditions do not improve, the trade stays non-tradable even if the chart looks interesting.")
            elif subject == "self_assessment":
                result = self._sir("that matters because an honest confidence level prevents us from treating early-stage evidence like a validated edge.")
            else:
                result = self._sir("that matters because the main point from the previous answer tells us where the real constraint is, not just where the noise is.")
        else:  # trade_closeness
            missing = list(setup.details.get("missing", []) or [])
            result = self._sir(f"we are {'not very' if setup.details.get('quality') == 'weak' else 'somewhat'} close to a trade right now. The setup is {setup.details.get('quality', 'unclear')}, and it still needs {', '.join(missing[:2]) or 'cleaner alignment'} before it can pass.")

        return self._reply(
            intent="follow_up",
            acknowledge="I’m answering in continuity with the previous turn, sir.",
            understanding=f"You are asking a follow-up about the previous {subject.replace('_', ' ')} answer.",
            data_sources=["brain session memory", "current status", "recent events", "comparison artifacts when active"],
            result=result,
            approval="No approval is needed for conversational follow-up. Ask Atlas remains read-only and cannot change trading or runtime behavior.",
            response_kind="question_answer",
        )

    def _handle_change_summary(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], simplify: bool = False) -> AtlasReply:
        latest_events = events[-2:]
        if len(latest_events) >= 2:
            delta_line = f"Recent change: from `{str(latest_events[-2].get('message', '') or '')}` to `{str(latest_events[-1].get('message', '') or '')}`."
        elif latest_events:
            delta_line = f"Latest new state message: {str(latest_events[-1].get('message', '') or '')}"
        else:
            delta_line = "I do not have enough recent events to compare the last cycle yet."
        result_lines = [
            self._sir(delta_line),
            self._sir(f"Current status now is {str(status.get('action', 'unknown'))} with message: {str(status.get('message', 'No message available.'))}"),
        ]
        if comparison:
            result_lines.append(self._sir(self._comparison_focus(comparison)))
        if simplify:
            result_lines = result_lines[:2]
        return self._reply(
            intent="change_summary",
            acknowledge=self._task_ack("I will summarize what changed using recent events and current status."),
            understanding=self._task_understanding("You want a read-only summary of what changed recently in Atlas state."),
            data_sources=["current status", "recent events", "comparison summary when active"],
            result=" ".join(result_lines),
            approval=self._shared_approval(),
            response_kind="task_result",
        )

    def _handle_general(self, query: str, status: dict[str, Any], events: list[dict[str, Any]], comparison: dict[str, Any], runtime: dict[str, Any], simplify: bool = False) -> AtlasReply:
        result_lines = [
            self._sir("I can answer from current state even when the request is only partly specified."),
            self._sir(f"the current status is {str(status.get('action', 'unknown'))}."),
        ]
        if status.get("message"):
            result_lines.append(self._sir(f"the latest message is {str(status.get('message'))}"))
        if runtime:
            result_lines.append(self._sir(f"runtime mode is {'live' if bool(runtime.get('execute_trades', False)) else 'paper'}, and Ask Atlas remains read-only."))
        if comparison:
            lines = _comparison_lines(comparison)
            if lines:
                result_lines.append(self._sir(lines[0]))
        lowered = query.lower()
        if any(term in lowered for term in ("market", "chart", "trend", "setup", "trade", "study", "review", "analyze", "summarize", "compare")):
            result_lines.append(self._sir("I think you are asking for a current-state research read. I answered from the nearest safe interpretation using current status, events, and comparison data if available."))
        if simplify:
            result_lines.append(self._sir("If you want, ask me in a shorter way like: go study the charts, why arent you trading, or explain it simple."))
        return self._reply(
            intent="general_operator_query",
            acknowledge="I’m answering from the nearest current-state view I can support, sir.",
            understanding=f"I will answer from current Atlas state and available research artifacts for: {query}",
            data_sources=["current status", "recent events", "runtime flags", "comparison artifacts when present"],
            result=" ".join(result_lines),
            approval="No approval is needed for read-only operator questions. Mutating actions stay blocked in this interface.",
            response_kind="question_answer",
        )


def load_comparison_meta_from_run_root(run_root: str | Path | None) -> dict[str, Any]:
    root = Path(run_root) if run_root else None
    if root is None:
        return {}

    def _load(path: Path) -> dict[str, Any]:
        try:
            import json

            return dict(json.loads(path.read_text(encoding="utf-8")) or {})
        except Exception:
            return {}

    preflight = _load(root / "preflight.json")
    summary = _load(root / "comparison_summary.json")
    report = _load(root / "comparison_report.json")
    tracker = _load(root / "comparison_tracker.json")
    return {
        "active": True,
        "run_id": str(preflight.get("run_id", root.name)),
        "preset": str(preflight.get("preset", "")),
        "root": str(root),
        "updated_at": str(summary.get("updated_at", "") or report.get("updated_at", "")),
        "control_metrics": dict(report.get("control", {}) or {}),
        "experiment_metrics": dict(report.get("experiment", {}) or {}),
        "cross_lane": dict(report.get("cross_lane", {}) or {}),
        "filter_effectiveness": dict(report.get("filter_effectiveness", {}) or {}),
        "decision_checklist": dict(report.get("decision_checklist", {}) or {}),
        "control_status": dict(summary.get("control_status", {}) or {}),
        "experiment_status": dict(summary.get("experiment_status", {}) or {}),
        "tracker": dict(tracker.get("cross_lane", {}) or tracker),
    }
