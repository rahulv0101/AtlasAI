from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import defaultdict

import pandas as pd

from atlas_ai.config.settings import Settings
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.brain.ai_shadow_review_runner import AIShadowReviewRunner


class AIShadowRiskAuditor:
    """Explain why a shadow candidate is blocked by drawdown risk."""

    def __init__(self, settings: Settings, store: KnowledgeStore) -> None:
        self.settings = settings
        self.store = store
        self.base = Path(store.base)
        self.population_path = self.base / "evolution_population.json"
        self.output_path = self.base / "ai_shadow_risk_audits.jsonl"

    def audit(self, uid: str, df: pd.DataFrame, *, bars: int | None = None) -> dict[str, Any]:
        chromosome = self._load_candidate(uid)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}
        if df.empty:
            return {"ok": False, "uid": uid, "reason": "empty_market_data", "executed": False}

        review_bars = int(bars or getattr(self.settings, "ai_shadow_review_bars", 1200) or 1200)
        df_slice = df.tail(max(60, review_bars)).reset_index(drop=True)
        outcomes = AIShadowReviewRunner._run_shadow(chromosome, df_slice)
        metrics = self._metrics(outcomes)
        guard_limit = float(getattr(self.settings, "evolution_max_shadow_drawdown_pct", 0.05) or 0.05)
        payload = {
            "schema": "atlas_ai_shadow_risk_audit_v1",
            "uid": uid,
            "origin": chromosome.origin,
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "bars": int(len(df_slice)),
            "metrics": metrics,
            "guard": {
                "shadow_drawdown_limit": guard_limit,
                "raw_peak_drawdown_passed": metrics["raw_peak_drawdown_pct"] <= guard_limit,
                "gross_normalized_drawdown_passed": metrics["gross_normalized_drawdown_pct"] <= guard_limit,
            },
            "diagnosis": self._diagnosis(metrics, guard_limit),
            "safety": {
                "audit_only": True,
                "does_not_execute": True,
                "does_not_promote": True,
                "does_not_mutate_strategy": True,
            },
            "executed": False,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self.store.log_event("evolution", "AI shadow risk audit completed.", payload)
        return {"ok": True, **payload}

    @staticmethod
    def _metrics(outcomes: list[Any]) -> dict[str, Any]:
        running = 0.0
        peak = 0.0
        max_drawdown = 0.0
        gross = 0.0
        wins = 0
        losses = 0
        current_loss_streak = 0
        worst_loss_streak = 0
        current_loss_sum = 0.0
        worst_loss_streak_pnl = 0.0
        worst_trade = 0.0
        drawdown_start_trade = 0
        drawdown_end_trade = 0
        peak_trade = 0
        equity_curve: list[float] = []
        side_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "wins": 0.0, "losses": 0.0, "pnl": 0.0})
        session_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "wins": 0.0, "losses": 0.0, "pnl": 0.0})
        hour_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"trades": 0.0, "wins": 0.0, "losses": 0.0, "pnl": 0.0})

        for idx, outcome in enumerate(outcomes, start=1):
            pnl = float(getattr(outcome, "pnl", 0.0) or 0.0)
            side = str(getattr(outcome, "side", "unknown") or "unknown")
            hour = _outcome_hour(getattr(outcome, "timestamp", "") or "")
            session = _session_from_hour(hour)
            for bucket in (side_stats[side], session_stats[session], hour_stats[str(hour) if hour >= 0 else "unknown"]):
                bucket["trades"] += 1.0
                bucket["pnl"] += pnl
                if pnl > 0:
                    bucket["wins"] += 1.0
                else:
                    bucket["losses"] += 1.0
            gross += abs(pnl)
            running += pnl
            equity_curve.append(running)
            if pnl > 0:
                wins += 1
                current_loss_streak = 0
                current_loss_sum = 0.0
            else:
                losses += 1
                current_loss_streak += 1
                current_loss_sum += pnl
                if current_loss_streak > worst_loss_streak or current_loss_sum < worst_loss_streak_pnl:
                    worst_loss_streak = current_loss_streak
                    worst_loss_streak_pnl = current_loss_sum
            if pnl < worst_trade:
                worst_trade = pnl
            if running > peak:
                peak = running
                peak_trade = idx
            drawdown = peak - running
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                drawdown_start_trade = peak_trade
                drawdown_end_trade = idx

        trades = len(outcomes)
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades if trades else 0.0,
            "net_pnl": running,
            "gross_pnl_movement": gross,
            "peak_pnl": peak,
            "max_drawdown": max_drawdown,
            "raw_peak_drawdown_pct": max_drawdown / max(1e-9, peak) if peak > 0 else 0.0,
            "gross_normalized_drawdown_pct": max_drawdown / max(1e-9, gross) if gross > 0 else 0.0,
            "worst_loss_streak": worst_loss_streak,
            "worst_loss_streak_pnl": worst_loss_streak_pnl,
            "worst_trade_pnl": worst_trade,
            "drawdown_start_trade": drawdown_start_trade,
            "drawdown_end_trade": drawdown_end_trade,
            "drawdown_span_trades": max(0, drawdown_end_trade - drawdown_start_trade),
            "average_abs_trade": gross / trades if trades else 0.0,
            "side_breakdown": _finalize_buckets(side_stats),
            "session_breakdown": _finalize_buckets(session_stats),
            "hour_breakdown": _top_negative_buckets(hour_stats, limit=6),
        }

    @staticmethod
    def _diagnosis(metrics: dict[str, Any], guard_limit: float) -> dict[str, Any]:
        gross_dd = float(metrics.get("gross_normalized_drawdown_pct", 0.0) or 0.0)
        raw_dd = float(metrics.get("raw_peak_drawdown_pct", 0.0) or 0.0)
        worst_streak = int(metrics.get("worst_loss_streak", 0) or 0)
        if gross_dd > guard_limit:
            primary = "drawdown_exceeds_gross_normalized_limit"
        elif raw_dd > guard_limit:
            primary = "raw_peak_drawdown_metric_over_sensitive"
        else:
            primary = "drawdown_within_limit"
        return {
            "primary": primary,
            "suggested_repair_focus": (
                "reduce_loss_clusters"
                if worst_streak >= 3
                else "reduce_single_trade_loss_or_peak_sensitivity"
            ),
            "notes": [
                "Audit is paper-only and does not change promotion status.",
                "Raw peak drawdown can be inflated when early running PnL peak is small.",
                "Gross-normalized drawdown gives an additional stability view without overriding the governor.",
            ],
        }

    def _load_candidate(self, uid: str) -> StrategyChromosome | None:
        if not self.population_path.exists():
            return None
        try:
            payload = json.loads(self.population_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        for row in list(payload.get("chromosomes", []) or []):
            if str(row.get("uid", "")) == str(uid):
                return StrategyChromosome.from_dict(row)
        return None


def _outcome_hour(timestamp: str) -> int:
    try:
        return int(pd.Timestamp(timestamp).hour)
    except Exception:
        return -1


def _session_from_hour(hour: int) -> str:
    # Match atlas_ai.strategies.strategy_factory._EvolvedGoldStrategy._session_label.
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 20:
        return "new_york"
    if 20 <= hour <= 23:
        return "late_us"
    return "unknown"


def _finalize_buckets(raw: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for key, bucket in raw.items():
        trades = int(bucket.get("trades", 0.0) or 0.0)
        wins = int(bucket.get("wins", 0.0) or 0.0)
        losses = int(bucket.get("losses", 0.0) or 0.0)
        rows.append({
            "bucket": key,
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / trades if trades else 0.0,
            "pnl": float(bucket.get("pnl", 0.0) or 0.0),
        })
    rows.sort(key=lambda row: (float(row["pnl"]), -int(row["trades"])))
    return rows


def _top_negative_buckets(raw: dict[str, dict[str, float]], *, limit: int) -> list[dict[str, Any]]:
    rows = _finalize_buckets(raw)
    negative = [row for row in rows if float(row.get("pnl", 0.0) or 0.0) < 0]
    return (negative or rows)[:limit]
