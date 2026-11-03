from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from atlas_ai.strategies.strategy_inventor import StrategyInventor


@dataclass(frozen=True)
class AIDecisionTrace:
    schema: str
    created_at: str
    market_read: dict[str, Any]
    recommendation: dict[str, Any]
    blocker_analysis: dict[str, Any]
    strategy_invention: dict[str, Any]
    safety: dict[str, Any]
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at": self.created_at,
            "market_read": dict(self.market_read),
            "recommendation": dict(self.recommendation),
            "blocker_analysis": dict(self.blocker_analysis),
            "strategy_invention": dict(self.strategy_invention),
            "safety": dict(self.safety),
            "audit": dict(self.audit),
        }


class AIDecisionCore:
    """Central AI coordination layer for Atlas decisions.

    The core produces a structured decision trace from state, memory, cognitive
    context, and strategy invention. It does not execute, promote live, mutate
    risk policy, or bypass deterministic gates.
    """

    def __init__(self, inventor: StrategyInventor | None = None) -> None:
        self.inventor = inventor or StrategyInventor()

    def decide(
        self,
        *,
        status: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        events: list[dict[str, Any]] | None = None,
        learning_state: dict[str, Any] | None = None,
        cognitive_core: dict[str, Any] | None = None,
        strategy_preferences: dict[str, Any] | None = None,
        mode: str = "analysis",
    ) -> AIDecisionTrace:
        status = dict(status or {})
        runtime = dict(runtime or {})
        events = list(events or [])
        learning_state = dict(learning_state or {})
        cognitive_core = dict(cognitive_core or {})
        world = dict(cognitive_core.get("world_model", {}) or {})
        lanes = dict(cognitive_core.get("decision_understanding", {}) or {})
        learning = dict(cognitive_core.get("learning_state", {}) or {})
        safety_contract = dict(cognitive_core.get("safety_contract", {}) or {})

        side = str(lanes.get("signal_direction") or dict(status.get("chosen", {}) or {}).get("side") or "neutral").lower()
        final_bias = str(lanes.get("final_trade_bias") or "neutral").lower()
        permission = str(lanes.get("execution_permission") or "not_allowed")
        readiness = str(lanes.get("setup_readiness") or "unknown")
        confidence = _safe_float(lanes.get("confidence"), _safe_float(dict(status.get("chosen", {}) or {}).get("confidence"), 0.0))
        threshold = _safe_float(lanes.get("threshold"), _safe_float(status.get("min_confidence_threshold"), 0.0))
        blocker = str(world.get("blocker") or status.get("primary_blocker") or status.get("message") or "").strip()
        stability = _safe_float(learning.get("stability_score"), 0.0)
        overreaction = _safe_float(learning.get("overreaction_score"), 0.0)

        blockers = self._blockers(
            side=side,
            final_bias=final_bias,
            permission=permission,
            readiness=readiness,
            confidence=confidence,
            threshold=threshold,
            blocker=blocker,
            overreaction=overreaction,
        )
        action = self._action(permission=permission, blockers=blockers, confidence=confidence, threshold=threshold)
        decision_confidence = self._decision_confidence(confidence, threshold, stability, blockers)
        invented = self.inventor.invent(
            status=status,
            runtime=runtime,
            learning_state=learning_state,
            cognitive_core=cognitive_core,
            strategy_preferences=strategy_preferences,
            limit=3,
        )

        return AIDecisionTrace(
            schema="atlas_ai_decision_trace_v1",
            created_at=datetime.now(timezone.utc).isoformat(),
            market_read={
                "symbol": str(status.get("symbol") or world.get("symbol") or "XAUUSD"),
                "timeframe": str(status.get("timeframe") or world.get("timeframe") or "unknown"),
                "session": str(world.get("session") or "unknown"),
                "regime": str(world.get("regime") or "unknown"),
                "structure": str(world.get("structure") or "unknown"),
                "momentum": str(world.get("momentum") or "unknown"),
                "location": str(world.get("location") or "unknown"),
                "market_bias": str(lanes.get("market_bias") or "unknown"),
                "signal_direction": side,
                "final_trade_bias": final_bias,
            },
            recommendation={
                "mode": mode,
                "action": action,
                "confidence": round(decision_confidence, 4),
                "trade_bias": final_bias,
                "setup_readiness": readiness,
                "reason": self._reason(action, blockers, final_bias),
            },
            blocker_analysis={
                "primary_blocker": blocker or (blockers[0] if blockers else "none"),
                "blockers": blockers,
                "overreaction_score": round(overreaction, 4),
                "stability_score": round(stability, 4),
            },
            strategy_invention={
                "should_invent": bool(invented),
                "candidate_count": len(invented),
                "candidate_archetypes": [p.archetype for p in invented],
                "proposal_uids": [p.chromosome.uid for p in invented],
                "preference_top": _top_preference(strategy_preferences),
                "promotion_required": ["validation", "backtest", "shadow", "demo", "operator_approval", "execution_gate"],
            },
            safety={
                "can_execute": False,
                "can_promote_live": False,
                "can_mutate_risk_policy": False,
                "can_bypass_execution_gate": False,
                "execution_permission_from_state": permission,
                "cognitive_contract": safety_contract,
            },
            audit={
                "evidence_used": [
                    "status",
                    "runtime",
                    "learning_state",
                    "cognitive_core",
                    "strategy_inventor",
                ],
                "events_seen": len(events),
                "fact_vs_inference": "structured_inference",
            },
        )

    @staticmethod
    def _blockers(
        *,
        side: str,
        final_bias: str,
        permission: str,
        readiness: str,
        confidence: float,
        threshold: float,
        blocker: str,
        overreaction: float,
    ) -> list[str]:
        blockers: list[str] = []
        if blocker:
            blockers.append(blocker)
        if side not in {"buy", "sell"}:
            blockers.append("no_directional_signal")
        if final_bias in {"neutral", "flat", "unknown"}:
            blockers.append("final_trade_bias_not_directional")
        if permission != "allowed":
            blockers.append("execution_permission_not_allowed")
        if threshold > 0 and confidence < threshold:
            blockers.append("confidence_below_threshold")
        if "blocked" in readiness:
            blockers.append(f"setup_readiness:{readiness}")
        if overreaction >= 0.5:
            blockers.append("overreaction_guard_active")
        return _dedupe(blockers)

    @staticmethod
    def _action(*, permission: str, blockers: list[str], confidence: float, threshold: float) -> str:
        if permission == "allowed" and not blockers and confidence >= threshold:
            return "prepare_trade_candidate"
        if any("overreaction" in b or "spread" in b.lower() for b in blockers):
            return "wait_and_reduce_frequency"
        if any("confidence" in b for b in blockers):
            return "wait_for_confirmation"
        return "hold_and_observe"

    @staticmethod
    def _decision_confidence(confidence: float, threshold: float, stability: float, blockers: list[str]) -> float:
        base = confidence if confidence > 0 else 0.35
        if threshold > 0:
            base = (base + min(1.0, confidence / max(threshold, 1e-9))) / 2.0
        base = (base + max(0.0, min(1.0, stability))) / 2.0
        base -= min(0.35, 0.05 * len(blockers))
        return max(0.05, min(0.99, base))

    @staticmethod
    def _reason(action: str, blockers: list[str], final_bias: str) -> str:
        if action == "prepare_trade_candidate":
            return f"State is directionally aligned for {final_bias}, pending deterministic gate checks."
        if blockers:
            return f"AI core recommends waiting because {blockers[0]}."
        return "AI core has insufficient directional evidence, so observation is preferred."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _top_preference(preferences: dict[str, Any] | None) -> str:
    ranking = list(dict(preferences or {}).get("ranking", []) or [])
    if ranking:
        return str(dict(ranking[0]).get("archetype", "") or "")
    return ""
