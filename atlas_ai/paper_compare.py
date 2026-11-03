from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from atlas_ai.config.settings import Settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.atlas_runtime_control import clear_stop_request, load_stop_request, stop_requested
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.runtime_heartbeat import start_runtime_heartbeat
from atlas_ai.trader import AtlasTrader

PAPER_PRESET_CONSERVATIVE_FILTER_V1 = "conservative_filter_rollout_v1"
PAPER_PRESET_SETUP_QUALITY_FILTER_V1 = "setup_quality_rollout_v1"
PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A = "ema_crossover_closed_confirm_mid_exit_stage_a"
PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A = "pullback_trend_m15_mid_exit_stage_a"


@dataclass
class ShadowPaperTrade:
    instance: str
    candidate_key: str
    strategy: str
    side: str
    confidence: float
    entry_time: str
    entry_price: float
    stop_loss_points: float
    take_profit_points: float
    stop_price: float
    take_price: float
    max_hold_bars: int
    bars_open: int = 0
    exit_time: str = ""
    exit_price: float = 0.0
    pnl_return: float = 0.0
    outcome: str = "open"
    exit_reason: str = ""
    regime: str = "unknown"
    session: str = "unknown"
    trend_strength: float = 0.0
    filter_reason: str = ""
    decision_explanation: dict[str, Any] | None = None
    shadow_advice: dict[str, Any] | None = None
    context_snapshot: dict[str, Any] | None = None


class PaperRolloutConfigError(RuntimeError):
    """Raised when a paper rollout preset is unsafe or inconsistent."""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _point_size(settings: Settings) -> float:
    info = mt5.symbol_info(settings.symbol)
    point = float(getattr(info, "point", 0.0) or 0.0) if info is not None else 0.0
    if point > 0:
        return point
    return max(1e-6, float(getattr(settings, "backtest_point_size", 0.01) or 0.01))


def _entry_price(settings: Settings, side: str, fallback: float) -> float:
    tick = mt5.symbol_info_tick(settings.symbol)
    if tick is None:
        return float(fallback)
    if str(side).lower() == "buy":
        return float(getattr(tick, "ask", fallback) or fallback)
    return float(getattr(tick, "bid", fallback) or fallback)


def _market_bar_time(status: dict[str, Any]) -> str:
    market = dict(status.get("market", {}) or {})
    recent_time = list(market.get("recent_time", []) or [])
    if recent_time:
        return str(recent_time[-1])
    return str(status.get("updated_at", ""))


def _candidate_key_from_status(status: dict[str, Any], settings: Settings) -> str:
    chosen = dict(status.get("chosen", {}) or {})
    parts = [
        str(settings.symbol),
        str(settings.timeframe),
        _market_bar_time(status),
        str(chosen.get("strategy", "")),
        str(chosen.get("side", "")),
        str(round(float(chosen.get("stop_loss_points", 0.0) or 0.0), 4)),
        str(round(float(chosen.get("take_profit_points", 0.0) or 0.0), 4)),
    ]
    return "|".join(parts)


def _status_trade_context(status: dict[str, Any], settings: Settings, lane: str) -> dict[str, Any]:
    chosen = dict(status.get("chosen", {}) or {})
    regime = dict(status.get("regime", {}) or {})
    return {
        "lane": lane,
        "candidate_key": _candidate_key_from_status(status, settings),
        "action": str(status.get("action", "")),
        "message": str(status.get("message", "")),
        "strategy": str(chosen.get("strategy", "")),
        "side": str(chosen.get("side", "")),
        "confidence": round(float(chosen.get("confidence", 0.0) or 0.0), 6),
        "session": str(dict(status.get("research_regime_overlay", {}) or {}).get("session", "")) or str(status.get("session", "")),
        "regime": str(regime.get("name", "unknown")),
        "trend_strength": round(float(regime.get("trend_strength", 0.0) or 0.0), 6),
        "filter_reason": str(status.get("message", "")),
        "decision_explanation": dict(status.get("decision_explanation", {}) or {}),
        "shadow_advice": dict(status.get("shadow_advice", {}) or {}),
        "context_snapshot": dict(status.get("context_snapshot", {}) or {}),
        "system_health": dict(status.get("system_health", {}) or {}),
        "updated_at": str(status.get("updated_at", "")),
    }


class ShadowPaperPortfolio:
    def __init__(self, settings: Settings, store: KnowledgeStore, name: str):
        self.settings = settings
        self.store = store
        self.name = name
        self.state_file = Path(settings.storage_dir) / "paper_compare_state.json"
        self.open_trades: list[ShadowPaperTrade] = []
        self.closed_trades: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.filtered_trades = 0
        self.missed_trades = 0
        self._load()

    def _load(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        self.open_trades = [ShadowPaperTrade(**item) for item in list(raw.get("open_trades", []) or [])]
        self.closed_trades = [dict(item) for item in list(raw.get("closed_trades", []) or [])][-500:]
        self.filtered_trades = int(raw.get("filtered_trades", 0) or 0)
        self.missed_trades = int(raw.get("missed_trades", 0) or 0)
        self.metrics = dict(raw.get("metrics", {}) or {})

    def _save(self) -> None:
        payload = {
            "open_trades": [asdict(item) for item in self.open_trades],
            "closed_trades": self.closed_trades[-500:],
            "filtered_trades": int(self.filtered_trades),
            "missed_trades": int(self.missed_trades),
            "metrics": dict(self.metrics or {}),
            "updated_at": _now_utc().isoformat(),
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def note_filtered(self, status: dict[str, Any] | None = None) -> None:
        self.filtered_trades += 1
        meta = {
            "instance": self.name,
            "filtered_trades": int(self.filtered_trades),
        }
        if isinstance(status, dict):
            meta.update(
                {
                    "action": status.get("action"),
                    "message": status.get("message"),
                    "chosen": dict(status.get("chosen", {}) or {}),
                    "decision_explanation": dict(status.get("decision_explanation", {}) or {}),
                    "shadow_advice": dict(status.get("shadow_advice", {}) or {}),
                    "context_snapshot": dict(status.get("context_snapshot", {}) or {}),
                }
            )
        self.store.log_event("paper_filtered", f"{self.name} filtered trade candidate.", meta)
        self._save()

    def maybe_open(self, status: dict[str, Any]) -> bool:
        if str(status.get("action", "")) != "paper":
            return False
        chosen = dict(status.get("chosen", {}) or {})
        side = str(chosen.get("side", "flat"))
        if side not in {"buy", "sell"}:
            return False
        if len(self.open_trades) >= int(self.settings.max_open_trades):
            self.missed_trades += 1
            self.store.log_event(
                "paper_skipped",
                f"{self.name} skipped paper trade due to max_open_trades limit.",
                {
                    "instance": self.name,
                    "missed_trades": int(self.missed_trades),
                    "max_open_trades": int(self.settings.max_open_trades),
                    "chosen": dict(status.get("chosen", {}) or {}),
                },
            )
            self._save()
            return False
        market = dict(status.get("market", {}) or {})
        fallback_entry = float(market.get("last_close", 0.0) or 0.0)
        entry = _entry_price(self.settings, side, fallback_entry)
        point = _point_size(self.settings)
        stop_points = max(1.0, float(chosen.get("stop_loss_points", 0.0) or 0.0))
        take_points = max(stop_points * 0.8, float(chosen.get("take_profit_points", 0.0) or 0.0))
        stop_price = entry - stop_points * point if side == "buy" else entry + stop_points * point
        take_price = entry + take_points * point if side == "buy" else entry - take_points * point
        trade = ShadowPaperTrade(
            instance=self.name,
            candidate_key=_candidate_key_from_status(status, self.settings),
            strategy=str(chosen.get("strategy", "")),
            side=side,
            confidence=float(chosen.get("confidence", 0.0) or 0.0),
            entry_time=str(status.get("updated_at", _now_utc().isoformat())),
            entry_price=float(entry),
            stop_loss_points=stop_points,
            take_profit_points=take_points,
            stop_price=float(stop_price),
            take_price=float(take_price),
            max_hold_bars=max(1, int(getattr(self.settings, "backtest_max_hold_bars", 12))),
        )
        regime = dict(status.get("regime", {}) or {})
        overlay = dict(status.get("research_regime_overlay", {}) or {})
        trade_dict = asdict(trade)
        trade_dict["regime"] = str(regime.get("name", "unknown"))
        trade_dict["session"] = str(overlay.get("session", "unknown") or "unknown")
        trade_dict["trend_strength"] = float(regime.get("trend_strength", 0.0) or 0.0)
        trade_dict["decision_explanation"] = dict(status.get("decision_explanation", {}) or {})
        trade_dict["shadow_advice"] = dict(status.get("shadow_advice", {}) or {})
        trade_dict["context_snapshot"] = dict(status.get("context_snapshot", {}) or {})
        self.open_trades.append(ShadowPaperTrade(**{k: v for k, v in trade_dict.items() if k in ShadowPaperTrade.__dataclass_fields__}))
        self.store.log_event(
            "paper_open",
            f"{self.name} opened shadow paper trade.",
            {
                "instance": self.name,
                "strategy": trade.strategy,
                "side": trade.side,
                "confidence": round(float(trade.confidence), 4),
                "entry_time": trade.entry_time,
                "decision_explanation": dict(status.get("decision_explanation", {}) or {}),
                "shadow_advice": dict(status.get("shadow_advice", {}) or {}),
            },
        )
        self._save()
        return True

    def update_from_df(self, df) -> None:
        if not self.open_trades or "time" not in df.columns:
            return
        remaining: list[ShadowPaperTrade] = []
        point = _point_size(self.settings)
        for trade in self.open_trades:
            entry_dt = datetime.fromisoformat(trade.entry_time.replace("Z", "+00:00"))
            bars = df[df["time"] > entry_dt]
            closed = False
            for row in bars.itertuples(index=False):
                trade.bars_open += 1
                high = float(row.high)
                low = float(row.low)
                close = float(row.close)
                ts = row.time.to_pydatetime() if hasattr(row.time, "to_pydatetime") else row.time
                if trade.side == "buy":
                    hit_stop = low <= trade.stop_price
                    hit_take = high >= trade.take_price
                    if hit_stop or hit_take:
                        trade.exit_price = float(trade.take_price if (hit_take and not hit_stop) else trade.stop_price)
                        trade.exit_reason = "take_profit" if (hit_take and not hit_stop) else "stop_loss"
                    elif trade.bars_open >= trade.max_hold_bars:
                        trade.exit_price = close
                        trade.exit_reason = "time_exit"
                    else:
                        continue
                    move = trade.exit_price - trade.entry_price
                else:
                    hit_stop = high >= trade.stop_price
                    hit_take = low <= trade.take_price
                    if hit_stop or hit_take:
                        trade.exit_price = float(trade.take_price if (hit_take and not hit_stop) else trade.stop_price)
                        trade.exit_reason = "take_profit" if (hit_take and not hit_stop) else "stop_loss"
                    elif trade.bars_open >= trade.max_hold_bars:
                        trade.exit_price = close
                        trade.exit_reason = "time_exit"
                    else:
                        continue
                    move = trade.entry_price - trade.exit_price
                trade.exit_time = ts.astimezone(timezone.utc).isoformat() if getattr(ts, "tzinfo", None) else str(ts)
                risk_distance = max(point, trade.stop_loss_points * point)
                r_multiple = move / max(risk_distance, 1e-9)
                trade.pnl_return = (r_multiple * float(self.settings.max_risk_per_trade)) - (
                    float(getattr(self.settings, "backtest_spread_cost_bps", 0.0)) / 10000.0
                    + float(getattr(self.settings, "backtest_slippage_cost_bps", 0.0)) / 10000.0
                )
                trade.outcome = "win" if trade.pnl_return > 0 else "loss"
                self.closed_trades.append(asdict(trade))
                self.store.log_event(
                    "paper_closed",
                    f"{self.name} closed shadow paper trade.",
                    {
                        "instance": self.name,
                        "strategy": trade.strategy,
                        "side": trade.side,
                        "outcome": trade.outcome,
                        "pnl_return": round(float(trade.pnl_return), 6),
                        "exit_reason": trade.exit_reason,
                    },
                )
                closed = True
                break
            if not closed:
                remaining.append(trade)
        self.open_trades = remaining
        self.metrics = self._compute_metrics()
        self._save()

    def _compute_metrics(self) -> dict[str, Any]:
        trades = list(self.closed_trades)
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "drawdown": 0.0,
                "trades_per_day": 0.0,
                "missed_trades": int(self.missed_trades),
                "filtered_trades": int(self.filtered_trades),
                "regime_distribution": {},
            }
        returns = [float(x.get("pnl_return", 0.0) or 0.0) for x in trades]
        wins = [x for x in returns if x > 0]
        losses = [x for x in returns if x < 0]
        equity = 10000.0
        peak = equity
        max_dd = 0.0
        for r in returns:
            equity *= max(0.0, 1.0 + r)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / max(peak, 1e-9))
        start = datetime.fromisoformat(str(trades[0]["entry_time"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(trades[-1]["exit_time"] or trades[-1]["entry_time"]).replace("Z", "+00:00"))
        days = max(1e-9, (end - start).total_seconds() / 86400.0)
        regime_distribution: dict[str, int] = {}
        for item in trades:
            regime = str(item.get("regime", "unknown"))
            regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / max(1, len(trades)),
            "expectancy": sum(returns) / max(1, len(trades)),
            "profit_factor": (sum(wins) / max(abs(sum(losses)), 1e-9)) if losses else 0.0,
            "average_win": (sum(wins) / len(wins)) if wins else 0.0,
            "average_loss": (sum(losses) / len(losses)) if losses else 0.0,
            "drawdown": max_dd,
            "trades_per_day": len(trades) / days,
            "missed_trades": int(self.missed_trades),
            "filtered_trades": int(self.filtered_trades),
            "regime_distribution": regime_distribution,
        }


class ComparisonTracker:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.tracker_file = self.root / "comparison_tracker.json"
        self.report_file = self.root / "comparison_report.json"
        self.state: dict[str, Any] = {
            "paired_candidates": {},
            "filtered_by_experiment": {},
            "taken_by_both": {},
            "control_only_taken": {},
            "experiment_only_taken": {},
            "last_report_at": "",
        }
        self._load()

    def _load(self) -> None:
        if not self.tracker_file.exists():
            return
        try:
            self.state = json.loads(self.tracker_file.read_text(encoding="utf-8"))
        except Exception:
            return

    def _save(self) -> None:
        self.tracker_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _check(status: str, value: Any, *, reason: str = "", threshold: Any | None = None) -> dict[str, Any]:
        payload = {
            "status": status,
            "value": value,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if reason:
            payload["reason"] = reason
        return payload

    @staticmethod
    def _distribution_metrics(trades: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
        buckets: dict[str, list[float]] = {}
        for trade in trades:
            name = str(trade.get(field, "unknown") or "unknown")
            pnl = ComparisonTracker._safe_float(trade.get("pnl_return", 0.0))
            buckets.setdefault(name, []).append(pnl)
        summary: dict[str, dict[str, Any]] = {}
        for name, pnls in buckets.items():
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            summary[name] = {
                "trades": len(pnls),
                "win_rate": len(wins) / max(1, len(pnls)),
                "expectancy": sum(pnls) / max(1, len(pnls)),
                "profit_factor": (sum(wins) / max(abs(sum(losses)), 1e-9)) if losses else (999.0 if wins else 0.0),
            }
        return summary

    @classmethod
    def _stability_check(
        cls,
        control_trades: list[dict[str, Any]],
        experiment_trades: list[dict[str, Any]],
        field: str,
        label: str,
    ) -> dict[str, Any]:
        control = cls._distribution_metrics(control_trades, field)
        experiment = cls._distribution_metrics(experiment_trades, field)
        shared = sorted(set(control) & set(experiment))
        eligible = [name for name in shared if int(control[name].get("trades", 0)) >= 10 and int(experiment[name].get("trades", 0)) >= 8]
        if not eligible:
            return cls._check(
                "unknown",
                {
                    "shared_buckets": shared,
                    "eligible_buckets": eligible,
                    "control": control,
                    "experiment": experiment,
                },
                reason=f"Not enough {label} data to assess stability.",
            )
        degraded = []
        for name in eligible:
            control_expectancy = cls._safe_float(control[name].get("expectancy", 0.0))
            experiment_expectancy = cls._safe_float(experiment[name].get("expectancy", 0.0))
            if experiment_expectancy < control_expectancy - max(0.0005, abs(control_expectancy) * 0.10):
                degraded.append(name)
        status = "pass" if not degraded else "fail"
        reason = (
            f"{label.title()} stability looks acceptable across eligible buckets."
            if not degraded
            else f"{label.title()} stability degraded in: {', '.join(degraded)}."
        )
        return cls._check(
            status,
            {
                "shared_buckets": shared,
                "eligible_buckets": eligible,
                "degraded_buckets": degraded,
                "control": control,
                "experiment": experiment,
            },
            reason=reason,
        )

    @classmethod
    def _sample_stage(
        cls,
        control_closed: int,
        experiment_closed: int,
        filtered_count: int,
        removed_outcomes_known: int,
    ) -> dict[str, Any]:
        stages = [
            {
                "name": "stronger_confidence",
                "thresholds": {
                    "control_closed": 250,
                    "experiment_closed": 175,
                    "filtered_observations": 50,
                    "removed_outcomes_known": 40,
                },
            },
            {
                "name": "minimum_comparison",
                "thresholds": {
                    "control_closed": 100,
                    "experiment_closed": 70,
                    "filtered_observations": 25,
                    "removed_outcomes_known": 20,
                },
            },
            {
                "name": "early_sanity",
                "thresholds": {
                    "control_closed": 30,
                    "experiment_closed": 20,
                    "filtered_observations": 10,
                    "removed_outcomes_known": 0,
                },
            },
        ]
        current = {
            "control_closed": int(control_closed),
            "experiment_closed": int(experiment_closed),
            "filtered_observations": int(filtered_count),
            "removed_outcomes_known": int(removed_outcomes_known),
        }
        matched = "insufficient"
        matched_thresholds: dict[str, int] = {}
        for stage in stages:
            thresholds = dict(stage["thresholds"])
            if all(int(current[key]) >= int(thresholds[key]) for key in thresholds):
                matched = str(stage["name"])
                matched_thresholds = thresholds
                break
        minimum_thresholds = dict(stages[1]["thresholds"])
        return {
            "current_stage": matched,
            "current_counts": current,
            "matched_thresholds": matched_thresholds,
            "minimum_comparison_thresholds": minimum_thresholds,
            "minimum_sample_reached": matched in {"minimum_comparison", "stronger_confidence"},
            "early_sanity_reached": matched in {"early_sanity", "minimum_comparison", "stronger_confidence"},
        }

    @classmethod
    def _decision_checklist(
        cls,
        control_summary: dict[str, Any],
        experiment_summary: dict[str, Any],
        filtered_rows: list[dict[str, Any]],
        control_trades: list[dict[str, Any]],
        experiment_trades: list[dict[str, Any]],
    ) -> dict[str, Any]:
        control_closed = int(control_summary.get("trades_closed", 0) or 0)
        experiment_closed = int(experiment_summary.get("trades_closed", 0) or 0)
        filtered_count = len(filtered_rows)
        removed_outcomes_known = sum(
            1 for row in filtered_rows if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() in {"win", "loss"}
        )
        stage = cls._sample_stage(control_closed, experiment_closed, filtered_count, removed_outcomes_known)
        trade_ratio = experiment_closed / max(1, control_closed)
        control_win_rate = cls._safe_float(control_summary.get("win_rate", 0.0))
        experiment_win_rate = cls._safe_float(experiment_summary.get("win_rate", 0.0))
        control_expectancy = cls._safe_float(control_summary.get("expectancy", 0.0))
        experiment_expectancy = cls._safe_float(experiment_summary.get("expectancy", 0.0))
        control_pf = cls._safe_float(control_summary.get("profit_factor", 0.0))
        experiment_pf = cls._safe_float(experiment_summary.get("profit_factor", 0.0))
        control_dd = cls._safe_float(control_summary.get("drawdown", 0.0))
        experiment_dd = cls._safe_float(experiment_summary.get("drawdown", 0.0))
        losers_avoided = sum(
            1 for row in filtered_rows if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() == "loss"
        )
        winners_removed = sum(
            1 for row in filtered_rows if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() == "win"
        )
        net_removal_edge = int(losers_avoided - winners_removed)
        win_rate_delta = experiment_win_rate - control_win_rate
        expectancy_delta = experiment_expectancy - control_expectancy
        profit_factor_delta = experiment_pf - control_pf
        drawdown_delta = experiment_dd - control_dd
        raw = {
            "sample_stage": stage["current_stage"],
            "control_closed_trades": control_closed,
            "experiment_closed_trades": experiment_closed,
            "experiment_trade_ratio": trade_ratio,
            "filtered_trade_observations": filtered_count,
            "removed_outcomes_known": removed_outcomes_known,
            "control_win_rate": control_win_rate,
            "experiment_win_rate": experiment_win_rate,
            "win_rate_delta": win_rate_delta,
            "control_expectancy": control_expectancy,
            "experiment_expectancy": experiment_expectancy,
            "expectancy_delta": expectancy_delta,
            "control_profit_factor": control_pf,
            "experiment_profit_factor": experiment_pf,
            "profit_factor_delta": profit_factor_delta,
            "control_drawdown": control_dd,
            "experiment_drawdown": experiment_dd,
            "drawdown_delta": drawdown_delta,
            "losers_avoided": int(losers_avoided),
            "winners_removed": int(winners_removed),
            "net_removal_edge": net_removal_edge,
        }
        checks: dict[str, dict[str, Any]] = {}
        checks["minimum_sample_reached"] = cls._check(
            "pass" if stage["minimum_sample_reached"] else ("fail" if not stage["early_sanity_reached"] else "unknown"),
            stage,
            reason=(
                "Minimum comparison sample reached."
                if stage["minimum_sample_reached"]
                else (
                    "Only early sanity-check sample reached."
                    if stage["early_sanity_reached"]
                    else "Sample is still below early sanity-check thresholds."
                )
            ),
        )
        if trade_ratio < 0.35:
            meaningful_status = "fail"
            meaningful_reason = "Experiment trade count collapsed below 35% of control."
        elif trade_ratio < 0.50:
            meaningful_status = "fail"
            meaningful_reason = "Experiment trade count is below 50% of control."
        elif trade_ratio < 0.65:
            meaningful_status = "unknown"
            meaningful_reason = "Experiment trade count is below the normal 65% acceptance floor."
        else:
            meaningful_status = "pass"
            meaningful_reason = "Experiment trade count remains meaningful versus control."
        checks["trade_count_meaningful"] = cls._check(
            meaningful_status,
            trade_ratio,
            threshold={"warn_below": 0.65, "fail_below": 0.50, "hard_fail_below": 0.35},
            reason=meaningful_reason,
        )
        checks["win_rate_change"] = cls._check(
            "pass" if win_rate_delta > 0 else ("unknown" if abs(win_rate_delta) < 1e-12 else "fail"),
            win_rate_delta,
            reason=(
                "Experiment win rate improved."
                if win_rate_delta > 0
                else ("No meaningful win-rate change yet." if abs(win_rate_delta) < 1e-12 else "Experiment win rate worsened.")
            ),
        )
        if experiment_expectancy >= control_expectancy:
            expectancy_status = "pass"
            expectancy_reason = "Experiment expectancy improved or stayed flat."
        elif control_expectancy > 0 and experiment_expectancy >= control_expectancy * 0.95:
            expectancy_status = "unknown"
            expectancy_reason = "Experiment expectancy is slightly lower but still within the 5% stability band."
        else:
            expectancy_status = "fail"
            expectancy_reason = "Experiment expectancy worsened beyond the stability band."
        checks["expectancy_change"] = cls._check(expectancy_status, expectancy_delta, reason=expectancy_reason)
        checks["profit_factor_change"] = cls._check(
            "pass" if profit_factor_delta > 0 else ("unknown" if abs(profit_factor_delta) < 1e-12 else "fail"),
            profit_factor_delta,
            reason=(
                "Experiment profit factor improved."
                if profit_factor_delta > 0
                else ("No meaningful profit-factor change yet." if abs(profit_factor_delta) < 1e-12 else "Experiment profit factor worsened.")
            ),
        )
        drawdown_limit = control_dd * 1.10 if control_dd > 0 else 0.0
        if experiment_dd <= drawdown_limit:
            drawdown_status = "pass"
            drawdown_reason = "Experiment drawdown did not materially worsen."
        else:
            drawdown_status = "fail"
            drawdown_reason = "Experiment drawdown worsened beyond the 10% relative tolerance."
        checks["drawdown_change"] = cls._check(
            drawdown_status,
            drawdown_delta,
            threshold={"max_relative_worsening": 0.10},
            reason=drawdown_reason,
        )
        if removed_outcomes_known <= 0:
            removal_status = "unknown"
            removal_reason = "Not enough removed-trade outcomes are known yet."
        elif losers_avoided > winners_removed:
            removal_status = "pass"
            removal_reason = "Filter is removing more losers than winners."
        elif losers_avoided == winners_removed:
            removal_status = "unknown"
            removal_reason = "Filter is removing winners and losers at the same rate so far."
        else:
            removal_status = "fail"
            removal_reason = "Filter is removing too many winners relative to avoided losers."
        checks["losers_avoided_vs_winners_removed"] = cls._check(
            removal_status,
            {"losers_avoided": losers_avoided, "winners_removed": winners_removed, "net_removal_edge": net_removal_edge},
            reason=removal_reason,
        )
        shrinkage_fail = trade_ratio < 0.65 and expectancy_delta <= 0 and profit_factor_delta <= 0
        shrinkage_unknown = trade_ratio < 0.65 and (expectancy_delta > 0 or profit_factor_delta > 0)
        checks["sample_shrinkage_illusion"] = cls._check(
            "fail" if shrinkage_fail else ("unknown" if shrinkage_unknown else "pass"),
            {"trade_ratio": trade_ratio, "expectancy_delta": expectancy_delta, "profit_factor_delta": profit_factor_delta},
            reason=(
                "Results look vulnerable to a sample-shrinkage illusion."
                if shrinkage_fail
                else (
                    "Trade count shrank enough that more evidence is needed to rule out sample-shrinkage effects."
                    if shrinkage_unknown
                    else "Improvement is not obviously explained only by sample shrinkage."
                )
            ),
        )
        checks["session_stability"] = cls._stability_check(control_trades, experiment_trades, "session", "session")
        checks["regime_stability"] = cls._stability_check(control_trades, experiment_trades, "regime", "regime")

        fail_keys = {name for name, payload in checks.items() if str(payload.get("status", "")).lower() == "fail"}
        pass_keys = {name for name, payload in checks.items() if str(payload.get("status", "")).lower() == "pass"}
        accept_ready = (
            stage["minimum_sample_reached"]
            and "trade_count_meaningful" in pass_keys
            and "expectancy_change" in pass_keys
            and "drawdown_change" in pass_keys
            and "losers_avoided_vs_winners_removed" in pass_keys
            and "sample_shrinkage_illusion" in pass_keys
            and ("win_rate_change" in pass_keys or "profit_factor_change" in pass_keys)
            and "session_stability" not in fail_keys
            and "regime_stability" not in fail_keys
        )
        early_reject = (
            stage["early_sanity_reached"]
            and (
                "expectancy_change" in fail_keys
                or "drawdown_change" in fail_keys
                or "losers_avoided_vs_winners_removed" in fail_keys
                or "sample_shrinkage_illusion" in fail_keys
                or ("trade_count_meaningful" in fail_keys and trade_ratio < 0.35)
            )
        )
        minimum_sample_reject = (
            stage["minimum_sample_reached"]
            and (
                "expectancy_change" in fail_keys
                or "trade_count_meaningful" in fail_keys
                or "drawdown_change" in fail_keys
                or "losers_avoided_vs_winners_removed" in fail_keys
                or "sample_shrinkage_illusion" in fail_keys
            )
        )
        if accept_ready:
            classification = "ACCEPT"
            explanation = "Minimum sample is met and the experiment shows real trade-quality improvement without material drawdown or sample-shrinkage concerns."
        elif minimum_sample_reject or early_reject:
            classification = "REJECT"
            explanation = "The experiment fails one or more hard safety checks such as expectancy, trade-count retention, drawdown, or removal quality."
        else:
            classification = "CONTINUE_TESTING"
            explanation = "Evidence is still mixed or the sample is not yet strong enough for a conservative decision."
        return {
            "sample": stage,
            "raw": raw,
            "checks": checks,
            "classification": classification,
            "explanation": explanation,
        }

    def record_cycle(self, control_status: dict[str, Any], experiment_status: dict[str, Any], settings: Settings) -> None:
        control_ctx = _status_trade_context(control_status, settings, "control")
        experiment_ctx = _status_trade_context(experiment_status, settings, "experiment")
        candidate_key = str(control_ctx.get("candidate_key") or experiment_ctx.get("candidate_key") or "")
        if not candidate_key:
            return
        paired = dict(self.state.get("paired_candidates", {}) or {})
        row = dict(paired.get(candidate_key, {}) or {})
        row["candidate_key"] = candidate_key
        row["control"] = control_ctx
        row["experiment"] = experiment_ctx
        row["updated_at"] = _now_utc().isoformat()
        paired[candidate_key] = row
        self.state["paired_candidates"] = paired

        control_action = str(control_ctx.get("action", ""))
        experiment_action = str(experiment_ctx.get("action", ""))
        if control_action == "paper" and experiment_action == "paper":
            self.state.setdefault("taken_by_both", {})[candidate_key] = row
        elif control_action == "paper" and experiment_action == "blocked":
            filtered = dict(self.state.get("filtered_by_experiment", {}) or {})
            filtered[candidate_key] = {
                "candidate_key": candidate_key,
                "filter_reason": experiment_ctx.get("filter_reason", ""),
                "regime": experiment_ctx.get("regime", ""),
                "session": experiment_ctx.get("session", ""),
                "trend_strength": experiment_ctx.get("trend_strength", 0.0),
                "confidence": control_ctx.get("confidence", 0.0),
                "strategy": control_ctx.get("strategy", ""),
                "side": control_ctx.get("side", ""),
                "control_decision_explanation": dict(control_ctx.get("decision_explanation", {}) or {}),
                "experiment_decision_explanation": dict(experiment_ctx.get("decision_explanation", {}) or {}),
                "control_shadow_advice": dict(control_ctx.get("shadow_advice", {}) or {}),
                "experiment_shadow_advice": dict(experiment_ctx.get("shadow_advice", {}) or {}),
                "control_context_snapshot": dict(control_ctx.get("context_snapshot", {}) or {}),
                "experiment_context_snapshot": dict(experiment_ctx.get("context_snapshot", {}) or {}),
                "control_system_health": dict(control_ctx.get("system_health", {}) or {}),
                "experiment_system_health": dict(experiment_ctx.get("system_health", {}) or {}),
                "decision_explanation": dict(experiment_ctx.get("decision_explanation", {}) or {}),
                "shadow_advice": dict(experiment_ctx.get("shadow_advice", {}) or {}),
                "context_snapshot": dict(experiment_ctx.get("context_snapshot", {}) or {}),
                "control_outcome": {},
                "experiment_outcome": {},
                "updated_at": _now_utc().isoformat(),
            }
            self.state["filtered_by_experiment"] = filtered
            self.state.setdefault("control_only_taken", {})[candidate_key] = row
        elif experiment_action == "paper" and control_action != "paper":
            self.state.setdefault("experiment_only_taken", {})[candidate_key] = row
        self._save()

    def attach_closed_outcomes(self, control_portfolio: ShadowPaperPortfolio, experiment_portfolio: ShadowPaperPortfolio) -> None:
        control_closed = {str(row.get("candidate_key", "")): dict(row) for row in list(control_portfolio.closed_trades or []) if str(row.get("candidate_key", ""))}
        experiment_closed = {str(row.get("candidate_key", "")): dict(row) for row in list(experiment_portfolio.closed_trades or []) if str(row.get("candidate_key", ""))}
        updated = False
        for bucket_name in ("filtered_by_experiment", "taken_by_both", "control_only_taken", "experiment_only_taken"):
            bucket = dict(self.state.get(bucket_name, {}) or {})
            for key, row in bucket.items():
                if key in control_closed:
                    row["control_outcome"] = {
                        "outcome": control_closed[key].get("outcome", ""),
                        "pnl_return": control_closed[key].get("pnl_return", 0.0),
                        "exit_reason": control_closed[key].get("exit_reason", ""),
                    }
                    updated = True
                if key in experiment_closed:
                    row["experiment_outcome"] = {
                        "outcome": experiment_closed[key].get("outcome", ""),
                        "pnl_return": experiment_closed[key].get("pnl_return", 0.0),
                        "exit_reason": experiment_closed[key].get("exit_reason", ""),
                    }
                    updated = True
                bucket[key] = row
            self.state[bucket_name] = bucket
        if updated:
            self._save()

    @staticmethod
    def _lane_summary(portfolio: ShadowPaperPortfolio) -> dict[str, Any]:
        metrics = dict(portfolio.metrics or {})
        return {
            "trades_taken": int(len(portfolio.closed_trades) + len(portfolio.open_trades)),
            "trades_skipped_filtered": int(portfolio.filtered_trades),
            "trades_closed": int(len(portfolio.closed_trades)),
            "win_rate": float(metrics.get("win_rate", 0.0) or 0.0),
            "expectancy": float(metrics.get("expectancy", 0.0) or 0.0),
            "profit_factor": float(metrics.get("profit_factor", 0.0) or 0.0),
            "drawdown": float(metrics.get("drawdown", 0.0) or 0.0),
            "average_win": float(metrics.get("average_win", 0.0) or 0.0),
            "average_loss": float(metrics.get("average_loss", 0.0) or 0.0),
        }

    def build_report(self, control_portfolio: ShadowPaperPortfolio, experiment_portfolio: ShadowPaperPortfolio) -> dict[str, Any]:
        filtered = dict(self.state.get("filtered_by_experiment", {}) or {})
        filtered_rows = list(filtered.values())
        removed_returns = [
            self._safe_float(dict(row.get("control_outcome", {}) or {}).get("pnl_return", 0.0))
            for row in filtered_rows
            if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() in {"win", "loss"}
        ]
        losers_avoided = sum(
            1 for row in filtered_rows
            if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() == "loss"
        )
        winners_lost = sum(
            1 for row in filtered_rows
            if str(dict(row.get("control_outcome", {}) or {}).get("outcome", "")).lower() == "win"
        )
        control_summary = self._lane_summary(control_portfolio)
        experiment_summary = self._lane_summary(experiment_portfolio)
        paired_candidates = dict(self.state.get("paired_candidates", {}) or {})
        latest_pair = {}
        if paired_candidates:
            latest_pair = max(
                paired_candidates.values(),
                key=lambda row: str(dict(row).get("updated_at", "")),
            )
        decision_checklist = self._decision_checklist(
            control_summary,
            experiment_summary,
            filtered_rows,
            list(control_portfolio.closed_trades or []),
            list(experiment_portfolio.closed_trades or []),
        )
        report = {
            "updated_at": _now_utc().isoformat(),
            "control": control_summary,
            "experiment": experiment_summary,
            "cross_lane": {
                "taken_by_both": len(dict(self.state.get("taken_by_both", {}) or {})),
                "control_only_taken": len(dict(self.state.get("control_only_taken", {}) or {})),
                "experiment_only_taken": len(dict(self.state.get("experiment_only_taken", {}) or {})),
                "filtered_by_experiment": len(filtered),
            },
            "lane_diagnostics": {
                "latest_control": dict(latest_pair.get("control", {}) or {}),
                "latest_experiment": dict(latest_pair.get("experiment", {}) or {}),
            },
            "trades_removed_breakdown": filtered_rows[-200:],
            "winners_vs_losers_removed": {
                "losers_avoided": int(losers_avoided),
                "winners_lost": int(winners_lost),
                "net_trade_quality_impact": int(losers_avoided - winners_lost),
                "removed_trade_expectancy": (
                    sum(removed_returns) / max(1, len(removed_returns))
                    if removed_returns
                    else 0.0
                ),
                "removed_trade_pnl_sum": sum(removed_returns) if removed_returns else 0.0,
                "removed_trade_outcomes_known": int(len(removed_returns)),
            },
            "filter_effectiveness": {
                "assessment": (
                    "positive"
                    if losers_avoided > winners_lost
                    else ("neutral" if losers_avoided == winners_lost else "negative")
                ),
            },
            "decision_checklist": decision_checklist,
        }
        self.state["last_report_at"] = report["updated_at"]
        self.report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        self._save()
        return report


def _clone_learning_state(source_store: KnowledgeStore, target_store: KnowledgeStore) -> None:
    source = Path(source_store.base) / "learning_state.json"
    target = Path(target_store.base) / "learning_state.json"
    if target.exists() or not source.exists():
        return
    shutil.copy2(source, target)


def _metrics_from_store(store: KnowledgeStore) -> dict[str, Any]:
    state_file = Path(store.base) / "paper_compare_state.json"
    if not state_file.exists():
        return {}
    try:
        return dict(json.loads(state_file.read_text(encoding="utf-8")).get("metrics", {}) or {})
    except Exception:
        return {}


def _settings_fingerprint(settings: Settings) -> dict[str, Any]:
    return {
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "execute_trades": bool(settings.execute_trades),
        "meta_model_enabled": bool(getattr(settings, "meta_model_enabled", True)),
        "gemini_influence_enabled": bool(getattr(settings, "gemini_influence_enabled", False)),
        "research_confidence_threshold": float(settings.research_confidence_threshold),
        "research_regime_filter_enabled": bool(getattr(settings, "research_regime_filter_enabled", False)),
        "conservative_filter_enabled": bool(getattr(settings, "conservative_filter_enabled", False)),
        "setup_quality_enabled": bool(getattr(settings, "setup_quality_enabled", False)),
        "setup_quality_filter_enabled": bool(getattr(settings, "setup_quality_filter_enabled", False)),
        "setup_quality_filter_min_score": float(getattr(settings, "setup_quality_filter_min_score", 76.0)),
        "conservative_filter_block_late_us_trend": bool(getattr(settings, "conservative_filter_block_late_us_trend", True)),
        "conservative_filter_block_post_spike": bool(getattr(settings, "conservative_filter_block_post_spike", True)),
        "conservative_filter_block_poor_liquidity": bool(getattr(settings, "conservative_filter_block_poor_liquidity", True)),
        "conservative_filter_poor_liquidity_volume_ratio": float(getattr(settings, "conservative_filter_poor_liquidity_volume_ratio", 0.80)),
        "conservative_filter_post_spike_atr_multiple": float(getattr(settings, "conservative_filter_post_spike_atr_multiple", 1.6)),
        "enabled_strategies": list(settings.enabled_strategies),
        "disabled_strategies": list(settings.disabled_strategies),
        "max_risk_per_trade": float(settings.max_risk_per_trade),
        "max_open_trades": int(settings.max_open_trades),
        "max_spread_points": float(settings.max_spread_points),
        "min_same_side_signals": int(settings.min_same_side_signals),
        "backtest_max_hold_bars": int(settings.backtest_max_hold_bars),
        "backtest_include_evolved": bool(settings.backtest_include_evolved),
    }


def comparison_version_hash(settings: Settings) -> str:
    root = Path(__file__).resolve().parents[1]
    tracked = [
        root / "backtest.py",
        root / "trader.py",
        root / "regime_filters.py",
        root / "learning" / "edge_building.py",
        root / "paper_compare.py",
        root / "config" / "settings.py",
        root / "strategies" / "built_in.py",
        root / "strategies" / "minute_pack.py",
        root / "strategies" / "strategy_lab.py",
    ]
    digest = hashlib.sha256()
    digest.update(json.dumps(_settings_fingerprint(settings), sort_keys=True).encode("utf-8"))
    for path in tracked:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _paper_preset_settings(settings: Settings, lane: str, storage_dir: Path, preset: str) -> Settings:
    preset_key = str(preset or PAPER_PRESET_CONSERVATIVE_FILTER_V1).strip().lower()
    lane_key = str(lane or "").strip().lower()
    if preset_key not in {
        PAPER_PRESET_CONSERVATIVE_FILTER_V1,
        PAPER_PRESET_SETUP_QUALITY_FILTER_V1,
        PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A,
        PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A,
    }:
        raise PaperRolloutConfigError(f"Unsupported paper rollout preset: {preset}")
    if lane_key not in {"control", "experiment"}:
        raise PaperRolloutConfigError(f"Unsupported paper rollout lane: {lane}")
    updates = {
        "storage_dir": str(storage_dir),
        "execute_trades": False,
        "meta_model_enabled": False,
        "gemini_influence_enabled": False,
        "research_regime_filter_enabled": False,
        "setup_quality_enabled": False,
        "setup_quality_filter_enabled": False,
        "setup_quality_filter_min_score": 76.0,
        "conservative_filter_enabled": False,
        "conservative_filter_block_late_us_trend": True,
        "conservative_filter_block_post_spike": True,
        "conservative_filter_block_poor_liquidity": True,
        "conservative_filter_poor_liquidity_volume_ratio": 0.80,
        "conservative_filter_post_spike_atr_multiple": 1.6,
    }
    if preset_key == PAPER_PRESET_CONSERVATIVE_FILTER_V1:
        updates["conservative_filter_enabled"] = lane_key == "experiment"
    elif preset_key == PAPER_PRESET_SETUP_QUALITY_FILTER_V1:
        updates["setup_quality_filter_enabled"] = lane_key == "experiment"
        updates["setup_quality_filter_min_score"] = 76.0
    elif preset_key == PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A:
        updates["timeframe"] = "M15"
        updates["backtest_include_evolved"] = False
        updates["enabled_strategies"] = (
            ("ema_crossover_closed_confirm_mid_exit",)
            if lane_key == "experiment"
            else tuple(getattr(settings, "enabled_strategies", ()) or ())
        )
    elif preset_key == PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A:
        updates["timeframe"] = "M15"
        updates["backtest_include_evolved"] = False
        updates["enabled_strategies"] = (
            ("pullback_trend_m15_mid_exit",)
            if lane_key == "experiment"
            else tuple(getattr(settings, "enabled_strategies", ()) or ())
        )
    try:
        return replace(settings, **updates)
    except TypeError:
        values = dict(getattr(settings, "__dict__", {}) or {})
        values.update(updates)
        return settings.__class__(**values)


def _validate_rollout_pair(control_settings: Settings, experiment_settings: Settings, preset: str) -> None:
    if bool(control_settings.execute_trades) or bool(experiment_settings.execute_trades):
        raise PaperRolloutConfigError("Paper rollout safety failed: execute_trades must be false for all lanes.")
    if bool(getattr(control_settings, "meta_model_enabled", True)) or bool(getattr(experiment_settings, "meta_model_enabled", True)):
        raise PaperRolloutConfigError("Paper rollout safety failed: meta_model_enabled must be false for all lanes.")
    if bool(getattr(control_settings, "gemini_influence_enabled", False)) or bool(getattr(experiment_settings, "gemini_influence_enabled", False)):
        raise PaperRolloutConfigError("Paper rollout safety failed: gemini_influence_enabled must be false for all lanes.")
    if bool(getattr(control_settings, "research_regime_filter_enabled", False)) != bool(getattr(experiment_settings, "research_regime_filter_enabled", False)):
        raise PaperRolloutConfigError("Paper rollout parity failed: research regime overlay must match across lanes.")

    control_fp = _settings_fingerprint(control_settings)
    experiment_fp = _settings_fingerprint(experiment_settings)
    allowed_diff = {"conservative_filter_enabled", "setup_quality_filter_enabled"}
    if str(preset or "").strip().lower() in {
        PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A,
        PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A,
    }:
        allowed_diff.update({"enabled_strategies", "timeframe", "backtest_include_evolved"})
    mismatches = [
        key for key in control_fp.keys()
        if key not in allowed_diff and control_fp[key] != experiment_fp[key]
    ]
    if mismatches:
        raise PaperRolloutConfigError(
            "Paper rollout parity failed; lanes differ outside the conservative filter: "
            + ", ".join(sorted(mismatches))
        )
    if str(preset or "").strip().lower() == PAPER_PRESET_CONSERVATIVE_FILTER_V1:
        if bool(control_fp["conservative_filter_enabled"]):
            raise PaperRolloutConfigError("Paper rollout parity failed: control lane must have conservative filter disabled.")
        if not bool(experiment_fp["conservative_filter_enabled"]):
            raise PaperRolloutConfigError("Paper rollout parity failed: experiment lane must have conservative filter enabled.")
    elif str(preset or "").strip().lower() == PAPER_PRESET_SETUP_QUALITY_FILTER_V1:
        if bool(control_fp["setup_quality_filter_enabled"]):
            raise PaperRolloutConfigError("Paper rollout parity failed: control lane must have setup-quality filter disabled.")
        if not bool(experiment_fp["setup_quality_filter_enabled"]):
            raise PaperRolloutConfigError("Paper rollout parity failed: experiment lane must have setup-quality filter enabled.")
    elif str(preset or "").strip().lower() == PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A:
        if bool(control_settings.execute_trades) or bool(experiment_settings.execute_trades):
            raise PaperRolloutConfigError("EMA Stage A paper preset failed: execute_trades must remain false.")
        if tuple(getattr(experiment_settings, "enabled_strategies", ()) or ()) != ("ema_crossover_closed_confirm_mid_exit",):
            raise PaperRolloutConfigError("EMA Stage A paper preset failed: experiment lane must isolate the recalibrated EMA variant.")
    elif str(preset or "").strip().lower() == PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A:
        if bool(control_settings.execute_trades) or bool(experiment_settings.execute_trades):
            raise PaperRolloutConfigError("Pullback M15 Stage A paper preset failed: execute_trades must remain false.")
        if tuple(getattr(experiment_settings, "enabled_strategies", ()) or ()) != ("pullback_trend_m15_mid_exit",):
            raise PaperRolloutConfigError("Pullback M15 Stage A paper preset failed: experiment lane must isolate the recalibrated pullback variant.")


def _enforce_paper_runtime_flags(store: KnowledgeStore, lane: str, run_id: str, preset: str) -> dict[str, Any]:
    current = store.load_runtime_flags()
    if bool(current.get("execute_trades", False)):
        raise PaperRolloutConfigError(
            f"Paper rollout safety failed for lane={lane}: runtime execute_trades must be false."
        )
    if bool(current.get("gemini_influence_enabled", False)):
        raise PaperRolloutConfigError(
            f"Paper rollout safety failed for lane={lane}: runtime gemini_influence_enabled must be false."
        )
    flags = {
        **current,
        "execute_trades": False,
        "gemini_influence_enabled": False,
        "paper_rollout": {
            "lane": lane,
            "run_id": run_id,
            "preset": preset,
        },
    }
    store.save_runtime_flags(flags)
    return flags


def _paper_rollout_status_payload(settings: Settings, lane: str, run_id: str, preset: str) -> dict[str, Any]:
    return {
        "action": "paper_ready",
        "message": f"Paper rollout preset ready for lane={lane}.",
        "lane": lane,
        "run_id": run_id,
        "preset": preset,
        "effective_mode": "paper",
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "enabled_strategies": list(getattr(settings, "enabled_strategies", ()) or ()),
        "strategy_candidate": (
            "ema_crossover_closed_confirm_mid_exit"
            if str(preset or "").strip().lower() == PAPER_PRESET_EMA_CLOSED_CONFIRM_MID_EXIT_STAGE_A and lane == "experiment"
            else "pullback_trend_m15_mid_exit"
            if str(preset or "").strip().lower() == PAPER_PRESET_PULLBACK_M15_MID_EXIT_STAGE_A and lane == "experiment"
            else ""
        ),
        "conservative_filter": {
            "enabled": bool(getattr(settings, "conservative_filter_enabled", False)),
            "block_late_us_trend": bool(getattr(settings, "conservative_filter_block_late_us_trend", True)),
            "block_post_spike": bool(getattr(settings, "conservative_filter_block_post_spike", True)),
            "block_poor_liquidity": bool(getattr(settings, "conservative_filter_block_poor_liquidity", True)),
            "poor_liquidity_volume_ratio": float(getattr(settings, "conservative_filter_poor_liquidity_volume_ratio", 0.80)),
            "post_spike_atr_multiple": float(getattr(settings, "conservative_filter_post_spike_atr_multiple", 1.6)),
        },
        "setup_quality_filter": {
            "enabled": bool(getattr(settings, "setup_quality_filter_enabled", False)),
            "min_score": float(getattr(settings, "setup_quality_filter_min_score", 76.0)),
        },
        "meta_model": {
            "enabled": bool(getattr(settings, "meta_model_enabled", True)),
        },
        "overlay_state": {
            "research_regime_filter_enabled": bool(getattr(settings, "research_regime_filter_enabled", False)),
            "gemini_influence_enabled": bool(getattr(settings, "gemini_influence_enabled", False)),
        },
        "risk": {
            "max_risk_per_trade": float(settings.max_risk_per_trade),
            "max_open_trades": int(settings.max_open_trades),
            "max_spread_points": float(settings.max_spread_points),
            "max_trades_per_day": int(settings.max_trades_per_day),
            "max_consecutive_losses": int(settings.max_consecutive_losses),
        },
    }


def _log_paper_rollout_startup(store: KnowledgeStore, settings: Settings, lane: str, run_id: str, preset: str) -> None:
    payload = _paper_rollout_status_payload(settings, lane, run_id, preset)
    store.save_status(payload)
    store.log_event("paper_rollout", "Paper rollout lane initialized.", payload)


def prepare_comparison_run(
    settings: Settings,
    comparison_root: str | None = None,
    run_id: str | None = None,
    resume: bool = False,
    require_market_open: bool = True,
    preset: str = PAPER_PRESET_CONSERVATIVE_FILTER_V1,
) -> dict[str, Any]:
    settings = replace(settings, execute_trades=False)
    base_root = Path(comparison_root) if comparison_root else (Path(settings.storage_dir).parent / "storage_compare_runs")
    base_root.mkdir(parents=True, exist_ok=True)
    resolved_run_id = str(run_id or datetime.now(timezone.utc).strftime("compare_%Y%m%dT%H%M%SZ"))
    run_root = base_root / resolved_run_id
    if run_root.exists() and not resume:
        raise RuntimeError(f"Comparison run already exists: {run_root}. Use --resume or a new run id.")
    run_root.mkdir(parents=True, exist_ok=True)
    control_dir = run_root / "control"
    experiment_dir = run_root / "experiment"
    control_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    feed = MarketDataFeed(settings)
    now_utc = _now_utc()
    max_age_seconds = max(180, int(getattr(settings, "timeframe_minutes", 1)) * 8 * 60)
    latest_dt: datetime | None = None
    latest_df = None
    market_open = False
    market_diag: dict[str, Any] = {}
    attempts = 4 if require_market_open else 1
    for attempt in range(1, attempts + 1):
        df = feed.get_rates(50)
        diag = dict(feed.diagnostics(50) or {})
        if not df.empty:
            latest_bar = df.iloc[-1]["time"]
            latest_dt = latest_bar.to_pydatetime() if hasattr(latest_bar, "to_pydatetime") else latest_bar
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            latest_df = df
            age_seconds = (now_utc - latest_dt.astimezone(timezone.utc)).total_seconds()
            market_open = age_seconds <= max_age_seconds
            diag["freshness_threshold_seconds"] = int(max_age_seconds)
            diag["latest_bar_age_seconds"] = round(float(age_seconds), 3)
            diag["freshness_passed"] = bool(market_open)
            diag["attempt"] = int(attempt)
            market_diag = diag
            if market_open or not require_market_open:
                break
        else:
            diag["freshness_threshold_seconds"] = int(max_age_seconds)
            diag["latest_bar_age_seconds"] = None
            diag["freshness_passed"] = False
            diag["attempt"] = int(attempt)
            market_diag = diag
        if attempt < attempts:
            time.sleep(2.0)

    diagnostics_file = run_root / "preflight_diagnostics.json"
    diagnostics_file.write_text(
        json.dumps(
            {
                "run_id": resolved_run_id,
                "checked_at": now_utc.isoformat(),
                "require_market_open": bool(require_market_open),
                "market_data": market_diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if latest_df is None or latest_df.empty or latest_dt is None:
        raise RuntimeError("Paper comparison preflight failed: no market data available.")
    if require_market_open and not market_open:
        latest_bar_text = str(market_diag.get("latest_bar_utc", "") or latest_dt.astimezone(timezone.utc).isoformat())
        raise RuntimeError(
            "Paper comparison preflight failed: market appears closed or data is stale. "
            f"Latest bar {latest_bar_text}. "
            f"Threshold={int(max_age_seconds)}s. "
            f"Tick={str(market_diag.get('latest_tick_utc', '') or 'unknown')}. "
            f"Source={str(market_diag.get('timestamp_source', 'unknown'))}."
        )

    control_settings = _paper_preset_settings(settings, "control", control_dir, preset)
    experiment_settings = _paper_preset_settings(settings, "experiment", experiment_dir, preset)
    _validate_rollout_pair(control_settings, experiment_settings, preset)
    parity_control = _settings_fingerprint(control_settings)
    parity_experiment = _settings_fingerprint(experiment_settings)
    version_hash = comparison_version_hash(settings)
    manifest = {
        "run_id": resolved_run_id,
        "created_at": now_utc.isoformat(),
        "mode": "paper_compare",
        "preset": str(preset),
        "version_hash": version_hash,
        "market_open": bool(market_open),
        "latest_bar_utc": latest_dt.astimezone(timezone.utc).isoformat(),
        "preflight_market_data": market_diag,
        "comparison_root": str(run_root),
        "control": {
            **parity_control,
            "lane": "control",
            "storage_dir": str(control_dir),
        },
        "experiment": {
            **parity_experiment,
            "lane": "experiment",
            "storage_dir": str(experiment_dir),
        },
    }
    (run_root / "preflight.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "run_root": run_root,
        "run_id": resolved_run_id,
        "control_settings": control_settings,
        "experiment_settings": experiment_settings,
        "manifest": manifest,
    }


def run_shadow_paper_comparison(
    settings: Settings,
    poll_seconds: int = 60,
    comparison_root: str | None = None,
    run_id: str | None = None,
    resume: bool = False,
    require_market_open: bool = True,
    preset: str = PAPER_PRESET_CONSERVATIVE_FILTER_V1,
) -> None:
    prepared = prepare_comparison_run(
        settings=settings,
        comparison_root=comparison_root,
        run_id=run_id,
        resume=resume,
        require_market_open=require_market_open,
        preset=preset,
    )
    root = Path(prepared["run_root"])
    clear_stop_request(str(root.parent), str(prepared["run_id"]))
    control_settings: Settings = prepared["control_settings"]
    experiment_settings: Settings = prepared["experiment_settings"]
    control_settings = replace(
        control_settings,
    )
    experiment_settings = replace(
        experiment_settings,
    )
    control_store = KnowledgeStore(str(root / "control"))
    experiment_store = KnowledgeStore(str(root / "experiment"))
    _clone_learning_state(KnowledgeStore(settings.storage_dir), control_store)
    _clone_learning_state(KnowledgeStore(settings.storage_dir), experiment_store)
    _enforce_paper_runtime_flags(control_store, "control", str(prepared["run_id"]), str(preset))
    _enforce_paper_runtime_flags(experiment_store, "experiment", str(prepared["run_id"]), str(preset))
    _log_paper_rollout_startup(control_store, control_settings, "control", str(prepared["run_id"]), str(preset))
    _log_paper_rollout_startup(experiment_store, experiment_settings, "experiment", str(prepared["run_id"]), str(preset))

    control = AtlasTrader(control_settings, store=control_store)
    experiment = AtlasTrader(experiment_settings, store=experiment_store)
    # FIX: keep trader heartbeat fresh in paper-compare lanes so heartbeat guard
    # classifies workers as healthy instead of stale-blocking every cycle.
    heartbeat_interval = float(120)
    heartbeat_stale_after = float(180)
    control_heartbeat_stop = start_runtime_heartbeat(
        control_store,
        component="backend_trader",
        interval_seconds=heartbeat_interval,
        stale_after_seconds=heartbeat_stale_after,
    )
    experiment_heartbeat_stop = start_runtime_heartbeat(
        experiment_store,
        component="backend_trader",
        interval_seconds=heartbeat_interval,
        stale_after_seconds=heartbeat_stale_after,
    )
    control_portfolio = ShadowPaperPortfolio(control_settings, control_store, "control")
    experiment_portfolio = ShadowPaperPortfolio(experiment_settings, experiment_store, "experiment")
    tracker = ComparisonTracker(root)
    summary_file = root / "comparison_summary.json"
    root.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            shared_df = control.feed.get_rates(1200)
            control_portfolio.update_from_df(shared_df)
            experiment_portfolio.update_from_df(shared_df)

            control_status = control.step()
            experiment_status = experiment.step()
            tracker.record_cycle(control_status, experiment_status, control_settings)

            if str(control_status.get("action", "")) == "paper":
                control_portfolio.maybe_open(control_status)
            elif str(control_status.get("action", "")) == "blocked":
                control_portfolio.note_filtered(control_status)

            if str(experiment_status.get("action", "")) == "paper":
                experiment_portfolio.maybe_open(experiment_status)
            elif str(experiment_status.get("action", "")) == "blocked":
                experiment_portfolio.note_filtered(experiment_status)
            tracker.attach_closed_outcomes(control_portfolio, experiment_portfolio)
            tracker_report = tracker.build_report(control_portfolio, experiment_portfolio)

            comparison = {
            "updated_at": _now_utc().isoformat(),
            "run_id": str(prepared["run_id"]),
            "preset": str(preset),
            "version_hash": str(prepared["manifest"].get("version_hash", "")),
            "control": _metrics_from_store(control_store),
            "experiment": _metrics_from_store(experiment_store),
            "control_status": {
                "action": control_status.get("action"),
                "message": control_status.get("message"),
            },
            "experiment_status": {
                "action": experiment_status.get("action"),
                "message": experiment_status.get("message"),
            },
            "comparison_tracker": {
                "report_path": str(tracker.report_file),
                "tracker_path": str(tracker.tracker_file),
                "cross_lane": dict(tracker_report.get("cross_lane", {}) or {}),
                "winners_vs_losers_removed": dict(tracker_report.get("winners_vs_losers_removed", {}) or {}),
                "filter_effectiveness": dict(tracker_report.get("filter_effectiveness", {}) or {}),
            },
            }
            summary_file.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            if stop_requested(str(root.parent), str(prepared["run_id"])):
                stop_meta = load_stop_request(str(root.parent), str(prepared["run_id"]))
                control_store.log_event("system", "Atlas sleep requested; paper comparison exiting cleanly after cycle.", stop_meta)
                experiment_store.log_event("system", "Atlas sleep requested; paper comparison exiting cleanly after cycle.", stop_meta)
                break
            time.sleep(max(5, int(poll_seconds)))
    finally:
        control_heartbeat_stop.set()
        experiment_heartbeat_stop.set()
