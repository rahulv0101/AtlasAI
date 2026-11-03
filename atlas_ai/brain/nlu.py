from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NLUResult:
    intent: str
    confidence: float
    entities: dict[str, Any] = field(default_factory=dict)
    normalized_text: str = ""
    ambiguity_detected: bool = False
    clarification_question: str = ""
    route: str = "general"
    safety_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9:\-_/\. ]+", " ", str(text or "").lower())).strip()


def _has_any(text: str, words: set[str]) -> bool:
    return any(w in text for w in words)


_EXAMPLE_CACHE: list[dict[str, str]] | None = None


def _load_examples() -> list[dict[str, str]]:
    global _EXAMPLE_CACHE
    if _EXAMPLE_CACHE is not None:
        return _EXAMPLE_CACHE
    path = Path("knowledge/nlu_examples.json")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        _EXAMPLE_CACHE = [r for r in rows if isinstance(r, dict) and "text" in r and "intent" in r]
    except Exception:
        _EXAMPLE_CACHE = []
    return _EXAMPLE_CACHE


def _token_set(text: str) -> set[str]:
    return {t for t in _norm(text).split(" ") if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _example_intent(text: str) -> tuple[str | None, float]:
    tokens = _token_set(text)
    best_intent = None
    best_score = 0.0
    for row in _load_examples():
        score = _jaccard(tokens, _token_set(str(row.get("text", ""))))
        if score > best_score:
            best_score = score
            best_intent = str(row.get("intent", "") or "")
    if best_intent and best_score >= 0.58:
        return best_intent, min(0.95, 0.62 + (best_score * 0.33))
    return None, 0.0


def parse_user_language(text: str) -> NLUResult:
    n = _norm(text)
    entities: dict[str, Any] = {}
    if not n:
        return NLUResult(
            intent="ambiguous",
            confidence=0.0,
            normalized_text=n,
            ambiguity_detected=True,
            clarification_question="Can you clarify what you want Atlas to do?",
            route="clarify",
            safety_required=False,
        )

    if "xauusd" in n or "gold" in n:
        entities["symbol"] = "XAUUSD"
    tf_match = re.search(r"\b(m1|m5|m15|m30|h1|h4)\b", n)
    if tf_match:
        entities["timeframe"] = tf_match.group(1).upper()
    if _has_any(n, {" buy", " long"}):
        entities["direction"] = "buy"
    elif _has_any(n, {" sell", " short"}):
        entities["direction"] = "sell"
    elif _has_any(n, {" no trade", " hold"}):
        entities["direction"] = "hold"
    if _has_any(n, {"demo learning", "learning"}):
        entities["mode"] = "demo_learning"
    elif "demo" in n:
        entities["mode"] = "demo"
    elif "paper" in n:
        entities["mode"] = "paper"
    elif "live" in n:
        entities["mode"] = "live"
    for comp in ("backend", "frontend", "ui-server", "dashboard", "hfm", "mt5", "listener", "voice"):
        if comp in n:
            entities["component"] = comp
            break
    for r in ("confidence", "spread", "news", "session", "regime", "blocker", "data", "broker"):
        if r in n:
            entities["reason_target"] = r
            break
    if "if " in n:
        entities["condition"] = n[n.find("if ") :].strip()

    safety_phrases = {"ignore the guard", "bypass safety", "force the buy", "trade anyway"}
    if _has_any(n, safety_phrases):
        return NLUResult(
            intent="trading_command",
            confidence=0.92,
            entities=entities,
            normalized_text=n,
            ambiguity_detected=False,
            clarification_question="",
            route="safe_trading_command",
            safety_required=True,
        )

    ambiguous_set = {"do it", "go", "what about now", "is it ready"}
    if n in ambiguous_set:
        return NLUResult(
            intent="ambiguous",
            confidence=0.35,
            entities=entities,
            normalized_text=n,
            ambiguity_detected=True,
            clarification_question="Do you want trading analysis, system status, or a system command?",
            route="clarify",
            safety_required=False,
        )

    # Example-driven intent fallback to improve slang/typo comprehension.
    example_intent, example_conf = _example_intent(n)

    system_cmd_words = {"wake up", "start the full system", "open the dashboard", "shut down", "sleep", "restart the ui"}
    trading_cmd_words = {"take the buy", "enter only if", "sell if confidence", "trade only if the gate allows"}
    analysis_words = {"should i buy", "can we sell", "what s the bias", "is gold bullish", "is gold bearish", "buy or sell"}
    explain_words = {"why no trade", "why are you blocked", "what are you waiting for", "why is confidence low", "what needs to happen before you trade"}
    status_words = {"is hfm connected", "is the backend running", "is the frontend alive", "what mode are you in", "are you in demo"}
    learning_words = {"what did you learn", "why are sell trades weak", "what data do you need", "how are you improving", "last 50 trades"}

    intent = "general_question"
    route = "general"
    conf = 0.55
    safety_required = False

    question_like = _has_any(n, {"why ", "what ", "how ", "is ", "are ", "status", "mode"})
    if _has_any(n, system_cmd_words) and not question_like:
        intent = "system_command"
        route = "system_command"
        conf = 0.9
        safety_required = _has_any(n, {"shut down", "close system", "stop everything", "shutdown"})
        entities["action_type"] = "start" if _has_any(n, {"wake", "start", "open"}) else "stop"
    elif _has_any(n, trading_cmd_words):
        intent = "trading_command"
        route = "safe_trading_command"
        conf = 0.85
        safety_required = True
        entities["action_type"] = "trade"
    elif _has_any(n, analysis_words):
        intent = "trading_analysis"
        route = "trading_analysis"
        conf = 0.88
        entities["action_type"] = "analyze"
    elif _has_any(n, explain_words) or ("why" in n and "trading" in n and ("not" in n or "blocked" in n)):
        intent = "trade_explanation"
        route = "trade_explanation"
        conf = 0.9
        entities["action_type"] = "explain"
    elif _has_any(n, status_words) or ("hfm" in n and ("connect" in n or "conect" in n)) or ("backend" in n and ("running" in n or "alive" in n)):
        intent = "system_status"
        route = "system_status"
        conf = 0.9
        entities["action_type"] = "status"
    elif _has_any(n, learning_words) or ("learn" in n and _has_any(n, {"what", "why", "how", "did", "wht", "u"})):
        intent = "learning_question"
        route = "learning_question"
        conf = 0.88
        entities["action_type"] = "learn"
    elif example_intent:
        intent = example_intent
        conf = max(conf, example_conf)
        route = {
            "trading_analysis": "trading_analysis",
            "trade_explanation": "trade_explanation",
            "system_status": "system_status",
            "system_command": "system_command",
            "trading_command": "safe_trading_command",
            "learning_question": "learning_question",
            "general_question": "general",
            "ambiguous": "clarify",
        }.get(example_intent, "general")
        if example_intent == "system_command":
            safety_required = _has_any(n, {"shut down", "sleep", "stop", "restart"})
            entities["action_type"] = "start" if _has_any(n, {"wake", "start", "open"}) else "stop"
        elif example_intent == "trading_command":
            safety_required = True
            entities["action_type"] = "trade"
        elif example_intent == "ambiguous":
            return NLUResult(
                intent="ambiguous",
                confidence=max(0.35, example_conf),
                entities=entities,
                normalized_text=n,
                ambiguity_detected=True,
                clarification_question="Do you want trading analysis, system status, or a system command?",
                route="clarify",
                safety_required=False,
            )

    if intent == "general_question":
        entities.setdefault("action_type", "general")
    return NLUResult(
        intent=intent,
        confidence=conf,
        entities=entities,
        normalized_text=n,
        ambiguity_detected=False,
        clarification_question="",
        route=route,
        safety_required=safety_required,
    )
