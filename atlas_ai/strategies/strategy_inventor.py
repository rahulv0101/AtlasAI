"""AI-centered strategy invention for Atlas.

The inventor turns Atlas' cognitive context into strategy chromosomes. It is
deliberately proposal-only: generated strategies must still pass validation,
backtesting, shadow/demo promotion, human approval, and ExecutionGate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from atlas_ai.strategies.dna import StrategyChromosome, make_standard_gene_pool
from atlas_ai.strategies.gold_validator import GoldValidator


@dataclass(frozen=True)
class InventedStrategyPlan:
    archetype: str
    thesis: str
    evidence_used: list[str]
    safety_constraints: list[str]
    chromosome: StrategyChromosome

    def to_dict(self) -> dict[str, Any]:
        payload = self.chromosome.to_dict()
        return {
            "archetype": self.archetype,
            "thesis": self.thesis,
            "evidence_used": list(self.evidence_used),
            "safety_constraints": list(self.safety_constraints),
            "chromosome": payload,
        }


class StrategyInventor:
    """Design strategy candidates from structured Atlas context.

    This is not an LLM free-write path. It is an AI orchestration layer over a
    bounded gene library so invention stays inspectable and testable.
    """

    def __init__(self, validator: GoldValidator | None = None) -> None:
        self.validator = validator or GoldValidator()

    def invent(
        self,
        *,
        status: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        learning_state: dict[str, Any] | None = None,
        cognitive_core: dict[str, Any] | None = None,
        strategy_preferences: dict[str, Any] | None = None,
        limit: int = 3,
    ) -> list[InventedStrategyPlan]:
        status = dict(status or {})
        runtime = dict(runtime or {})
        learning_state = dict(learning_state or {})
        cognitive_core = dict(cognitive_core or {})
        context = dict(status.get("context_snapshot", {}) or {})
        world = dict(cognitive_core.get("world_model", {}) or {})
        decision = dict(cognitive_core.get("decision_understanding", {}) or {})
        learning = dict(cognitive_core.get("learning_state", {}) or {})

        regime = str(context.get("regime") or world.get("regime") or "unknown").lower()
        blocker = str(status.get("primary_blocker") or world.get("blocker") or "").lower()
        bias = str(decision.get("final_trade_bias") or decision.get("market_bias") or "neutral").lower()
        stability = _safe_float(learning.get("stability_score"), 0.5)
        overreaction = _safe_float(learning.get("overreaction_score"), 0.0)

        archetypes = self._choose_archetypes(regime=regime, blocker=blocker, bias=bias, stability=stability)
        archetypes = self._rank_with_preferences(archetypes, dict(strategy_preferences or {}))
        plans: list[InventedStrategyPlan] = []
        for archetype in archetypes[: max(1, int(limit))]:
            chromosome = self._chromosome_for(
                archetype=archetype,
                regime=regime,
                blocker=blocker,
                bias=bias,
                stability=stability,
                overreaction=overreaction,
                seed_payload={"status": status, "runtime": runtime, "learning_state": learning_state},
            )
            ok, reason = self.validator.validate_silent(chromosome)
            if not ok:
                chromosome.reject_reason = f"inventor_validation_failed:{reason}"
                continue
            plans.append(
                InventedStrategyPlan(
                    archetype=archetype,
                    thesis=self._thesis(archetype, regime, blocker, bias),
                    evidence_used=[
                        f"regime:{regime or 'unknown'}",
                        f"blocker:{blocker or 'none'}",
                        f"bias:{bias or 'neutral'}",
                        f"stability:{stability:.2f}",
                        f"overreaction:{overreaction:.2f}",
                    ],
                    safety_constraints=[
                        "proposal_only",
                        "gold_validator_required",
                        "offline_backtest_required",
                        "shadow_demo_promotion_required",
                        "human_approval_required_for_live",
                        "execution_gate_cannot_be_bypassed",
                    ],
                    chromosome=chromosome,
                )
            )
        return plans

    def _choose_archetypes(self, *, regime: str, blocker: str, bias: str, stability: float) -> list[str]:
        chosen: list[str] = []
        if "trend" in regime or bias in {"buy", "sell", "bullish", "bearish"}:
            chosen.append("trend_pullback_continuation")
        if "range" in regime or "confidence" in blocker:
            chosen.append("range_reversion_filter")
        if "spread" in blocker or "volatility" in regime or stability < 0.45:
            chosen.append("low_frequency_volatility_guard")
        if not chosen:
            chosen.append("balanced_confirmation")
        return chosen

    @staticmethod
    def _rank_with_preferences(archetypes: list[str], preferences: dict[str, Any]) -> list[str]:
        scores = dict(preferences.get("scores", {}) or {})
        indexed = list(enumerate(archetypes))
        indexed.sort(key=lambda item: (float(scores.get(item[1], 0.0) or 0.0), -item[0]), reverse=True)
        return [name for _, name in indexed]

    def _chromosome_for(
        self,
        *,
        archetype: str,
        regime: str,
        blocker: str,
        bias: str,
        stability: float,
        overreaction: float,
        seed_payload: dict[str, Any],
    ) -> StrategyChromosome:
        genes = make_standard_gene_pool()
        values = {g.name: g for g in genes}

        if archetype == "trend_pullback_continuation":
            self._set(values, "entry_ema_fast", 14)
            self._set(values, "entry_ema_slow", 55)
            self._set(values, "rsi_buy_threshold", 47)
            self._set(values, "rsi_sell_threshold", 53)
            self._set(values, "confirm_min_trend_strength", 0.28)
            self._set(values, "confirm_min_body_ratio", 0.32)
            self._set(values, "confirm_volume_min_ratio", 0.78)
            self._set(values, "confirm_htf_alignment", 1)
            self._set(values, "regime_prefer_trend", 1)
            self._set(values, "risk_min_confidence", 0.46)
            self._set(values, "exit_sl_atr_multiple", 1.7)
            self._set(values, "exit_tp_atr_multiple", 3.1)
        elif archetype == "range_reversion_filter":
            self._set(values, "entry_ema_fast", 18)
            self._set(values, "entry_ema_slow", 48)
            self._set(values, "rsi_buy_threshold", 41)
            self._set(values, "rsi_sell_threshold", 59)
            self._set(values, "confirm_min_body_ratio", 0.30)
            self._set(values, "confirm_volume_min_ratio", 0.72)
            self._set(values, "regime_prefer_trend", 0)
            self._set(values, "risk_min_confidence", 0.42)
            self._set(values, "exit_sl_atr_multiple", 1.4)
            self._set(values, "exit_tp_atr_multiple", 2.4)
        elif archetype == "low_frequency_volatility_guard":
            self._set(values, "entry_ema_fast", 24)
            self._set(values, "entry_ema_slow", 78)
            self._set(values, "rsi_buy_threshold", 40)
            self._set(values, "rsi_sell_threshold", 60)
            self._set(values, "confirm_min_trend_strength", 0.32)
            self._set(values, "confirm_min_body_ratio", 0.48)
            self._set(values, "confirm_volume_min_ratio", 0.85)
            self._set(values, "risk_min_confidence", 0.64)
            self._set(values, "risk_max_trades_per_day", 4)
            self._set(values, "exit_sl_atr_multiple", 1.9)
            self._set(values, "exit_tp_atr_multiple", 3.4)
        else:
            self._set(values, "entry_ema_fast", 20)
            self._set(values, "entry_ema_slow", 60)
            self._set(values, "rsi_buy_threshold", 41)
            self._set(values, "rsi_sell_threshold", 59)
            self._set(values, "confirm_min_body_ratio", 0.34)
            self._set(values, "confirm_volume_min_ratio", 0.78)
            self._set(values, "risk_min_confidence", 0.45)

        self._set(values, "regime_allowed_sessions", 3)

        if "spread" in blocker or overreaction > 0.45:
            self._set(values, "risk_min_confidence", max(values["risk_min_confidence"].value, 0.62))
            self._set(values, "risk_max_trades_per_day", min(values["risk_max_trades_per_day"].value, 5))
        if stability >= 0.7 and archetype == "trend_pullback_continuation":
            self._set(values, "risk_confidence_multiplier", 1.08)
        if bias in {"neutral", "flat"}:
            self._set(values, "confirm_htf_alignment", 1)

        uid = "ai_" + hashlib.sha1(repr(seed_payload).encode("utf-8")).hexdigest()[:12]
        return StrategyChromosome(
            uid=uid,
            generation=0,
            genes=genes,
            origin=f"ai_inventor:{archetype}",
            created_at=datetime.now(timezone.utc).isoformat(),
            promotion_stage="untested",
            promotion_reason="ai_invented_proposal_awaiting_offline_backtest",
        )

    @staticmethod
    def _set(genes: dict[str, Any], name: str, value: float) -> None:
        gene = genes[name]
        gene.value = gene.allowed_range.clamp(float(value))

    @staticmethod
    def _thesis(archetype: str, regime: str, blocker: str, bias: str) -> str:
        if archetype == "trend_pullback_continuation":
            return "Exploit directional gold moves only after pullback and higher-timeframe confirmation."
        if archetype == "range_reversion_filter":
            return "Trade only deeper RSI extremes while filtering weak range signals with candle and volume confirmation."
        if archetype == "low_frequency_volatility_guard":
            return "Reduce trade frequency and require stronger confirmation when volatility, spread, or stability risk is elevated."
        return f"Balanced XAUUSD candidate for regime={regime or 'unknown'}, bias={bias or 'neutral'}, blocker={blocker or 'none'}."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default
