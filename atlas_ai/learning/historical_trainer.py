from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from types import SimpleNamespace

from atlas_ai.backtest import Backtester, BacktestResult
from atlas_ai.config.settings import Settings
from atlas_ai.data.market_data import MarketDataFeed
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.strategies.strategy_lab import StrategyLab


@dataclass
class HistoricalTrainingResult:
    bars: int
    total_return: float
    trades: int
    wins: int
    losses: int


class HistoricalTrainer:
    def __init__(self, settings: Settings, store: KnowledgeStore):
        self.settings = settings
        self.store = store
        self.feed = MarketDataFeed(settings)

    def _fetch_history(self):
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30 * self.settings.historical_train_months)
        try:
            df = self.feed.get_rates_range(start, end)
        except Exception:
            # Some MT5 terminals reject wide ranges; fallback to deep recent bars.
            df = self.feed.get_rates(self.settings.historical_train_max_bars)
        if len(df) > self.settings.historical_train_max_bars:
            df = df.tail(self.settings.historical_train_max_bars)
        return df

    def _log_training_event(self, message: str, meta: dict[str, object] | None = None) -> None:
        self.store.log_event("training", message, meta or {})

    def _candidate_score_adjuster(self):
        trend_weight = float(getattr(self.settings, "training_trend_robustness_weight", 0.0) or 0.0)
        range_penalty = float(getattr(self.settings, "training_range_dependency_penalty", 0.0) or 0.0)
        trend_loss_penalty = float(getattr(self.settings, "training_trend_loss_penalty_weight", 0.0) or 0.0)
        profit_alignment_weight = float(getattr(self.settings, "training_profit_alignment_weight", 0.0) or 0.0)
        profit_regression_penalty = float(getattr(self.settings, "training_profit_regression_penalty_weight", 0.0) or 0.0)
        min_win_rate_alignment = float(getattr(self.settings, "training_min_win_rate_alignment", 0.0) or 0.0)
        pullback_quality_weight = float(getattr(self.settings, "training_pullback_quality_weight", 0.0) or 0.0)
        wick_pressure_penalty = float(getattr(self.settings, "training_wick_pressure_penalty_weight", 0.0) or 0.0)
        trend_maturity_weight = float(getattr(self.settings, "training_trend_maturity_weight", 0.0) or 0.0)
        target_candidate_win_rate = float(getattr(self.settings, "training_target_candidate_win_rate", 0.0) or 0.0)
        overtrade_penalty_weight = float(getattr(self.settings, "training_overtrade_penalty_weight", 0.0) or 0.0)
        continuation_quality_weight = float(getattr(self.settings, "training_continuation_quality_weight", 0.0) or 0.0)
        validation_weight = float(getattr(self.settings, "training_quality_validation_weight", 0.0) or 0.0)
        if (
            trend_weight <= 0.0
            and range_penalty <= 0.0
            and trend_loss_penalty <= 0.0
            and profit_alignment_weight <= 0.0
            and profit_regression_penalty <= 0.0
            and min_win_rate_alignment <= 0.0
            and pullback_quality_weight <= 0.0
            and wick_pressure_penalty <= 0.0
            and trend_maturity_weight <= 0.0
            and target_candidate_win_rate <= 0.0
            and overtrade_penalty_weight <= 0.0
            and continuation_quality_weight <= 0.0
            and validation_weight <= 0.0
        ):
            return None

        def _adjust(metrics: dict[str, float]) -> float:
            base = float(metrics.get("score", -999.0) or -999.0)
            profit_component = float(metrics.get("profit_component", 0.0) or 0.0)
            win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
            trades = float(metrics.get("trades", 0.0) or 0.0)
            trend_profit = float(metrics.get("trend_profit_component", 0.0) or 0.0)
            range_profit = float(metrics.get("range_profit_component", 0.0) or 0.0)
            trend_win_rate = float(metrics.get("trend_win_rate", 0.0) or 0.0)
            trend_losses = float(metrics.get("trend_losses", 0.0) or 0.0)
            trend_wins = float(metrics.get("trend_wins", 0.0) or 0.0)
            pullback_quality = float(metrics.get("pullback_quality_score", 0.0) or 0.0)
            wick_pressure = float(metrics.get("wick_pressure_score", 0.0) or 0.0)
            trend_maturity = float(metrics.get("trend_maturity_score", 0.0) or 0.0)
            continuation_quality = float(metrics.get("continuation_quality_score", 0.0) or 0.0)
            quality_validation_score = float(metrics.get("quality_validation_score", 0.0) or 0.0)
            adjusted = base
            adjusted += profit_alignment_weight * max(0.0, profit_component) * (0.65 + 0.35 * win_rate)
            adjusted -= profit_regression_penalty * max(0.0, -profit_component)
            adjusted += min_win_rate_alignment * max(0.0, win_rate - 0.5)
            adjusted -= min_win_rate_alignment * max(0.0, 0.5 - win_rate) * 1.5
            adjusted += trend_weight * max(0.0, trend_profit) * (0.7 + 0.3 * trend_win_rate)
            adjusted -= range_penalty * max(0.0, range_profit - trend_profit)
            adjusted -= trend_loss_penalty * max(0.0, trend_losses - trend_wins) * 0.001
            adjusted += pullback_quality_weight * pullback_quality
            adjusted -= wick_pressure_penalty * wick_pressure
            adjusted += trend_maturity_weight * trend_maturity
            if target_candidate_win_rate > 0.0:
                adjusted += max(0.0, win_rate - target_candidate_win_rate) * 0.8
                adjusted -= max(0.0, target_candidate_win_rate - win_rate) * 0.6
            adjusted -= overtrade_penalty_weight * max(0.0, trades - 260.0) * 0.0002
            adjusted += continuation_quality_weight * continuation_quality
            adjusted += validation_weight * quality_validation_score
            return float(adjusted)

        return _adjust

    @staticmethod
    def _weak_point_summary(best_strategy: dict[str, object], promoted_pool: dict[str, object]) -> dict[str, object]:
        candidates = list(promoted_pool.get("candidates", []) or [])
        champion = candidates[0] if candidates else dict(best_strategy or {})
        trend_losses = int(champion.get("trend_losses", 0) or 0)
        trend_wins = int(champion.get("trend_wins", 0) or 0)
        range_dependency = float(champion.get("range_dependency_score", 0.0) or 0.0)
        stop_hunt_vulnerability = float(champion.get("stop_hunt_vulnerability", 0.0) or 0.0)
        trend_edge_score = float(champion.get("trend_edge_score", 0.0) or 0.0)

        weaknesses: list[str] = []
        strengths: list[str] = []
        if trend_losses > trend_wins:
            weaknesses.append("trend conditions still lose too often")
        else:
            strengths.append("trend conditions are no longer loss-dominant")
        if range_dependency > 0.0:
            weaknesses.append("returns still lean on range behavior")
        else:
            strengths.append("returns are not leaning on range behavior")
        if stop_hunt_vulnerability >= 0.08:
            weaknesses.append("early stop-out pressure is elevated")
        elif stop_hunt_vulnerability <= 0.03:
            strengths.append("early stop-out pressure is controlled")
        if trend_edge_score > 0.0:
            strengths.append("trend edge is measurable")
        else:
            weaknesses.append("trend edge is still weak")

        return {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "explainability_summary": str(champion.get("explainability_summary", "")),
            "trend_edge_score": trend_edge_score,
            "range_dependency_score": range_dependency,
            "stop_hunt_vulnerability": stop_hunt_vulnerability,
        }

    def _promotion_thresholds(self) -> dict[str, float]:
        return {
            "min_trades": float(getattr(self.settings, "evolved_candidate_min_trades", 200)),
            "min_win_rate": float(getattr(self.settings, "evolved_candidate_min_win_rate", 0.52)),
            "min_profit_component": float(getattr(self.settings, "evolved_candidate_min_profit_component", 0.0)),
        }

    @staticmethod
    def _setup_family_from_candidate(candidate_like: dict[str, object]) -> str:
        fast = int(float(candidate_like.get("fast", 0) or 0))
        slow = int(float(candidate_like.get("slow", 0) or 0))
        rsi_buy = float(candidate_like.get("rsi_buy", 50.0) or 50.0)
        rsi_sell = float(candidate_like.get("rsi_sell", 50.0) or 50.0)
        gap = slow - fast
        if gap >= 55 and rsi_sell >= 60.0:
            return "continuation"
        if gap >= 45 and rsi_buy >= 45.0 and rsi_sell <= 55.0:
            return "pullback"
        if gap <= 35 and rsi_buy < 42.0:
            return "breakout"
        if rsi_buy <= 45.0 and rsi_sell >= 55.0:
            return "mean_reversion"
        return "hybrid"

    def _setup_family_summary(self, promoted_pool: dict[str, object]) -> dict[str, object]:
        candidates = list(promoted_pool.get("candidates", []) or [])
        if not candidates:
            return {"families": [], "best_family": "none"}
        grouped: dict[str, dict[str, float]] = {}
        for item in candidates:
            family = self._setup_family_from_candidate(dict(item or {}))
            bucket = grouped.setdefault(
                family,
                {"family": family, "count": 0.0, "score_sum": 0.0, "win_rate_sum": 0.0, "trend_edge_sum": 0.0},
            )
            bucket["count"] += 1.0
            bucket["score_sum"] += float(item.get("score", 0.0) or 0.0)
            bucket["win_rate_sum"] += float(item.get("win_rate", 0.0) or 0.0)
            bucket["trend_edge_sum"] += float(item.get("trend_edge_score", 0.0) or 0.0)
        families = []
        for bucket in grouped.values():
            count = max(1.0, bucket["count"])
            families.append(
                {
                    "family": str(bucket["family"]),
                    "count": int(bucket["count"]),
                    "avg_score": round(bucket["score_sum"] / count, 6),
                    "avg_win_rate": round(bucket["win_rate_sum"] / count, 4),
                    "avg_trend_edge": round(bucket["trend_edge_sum"] / count, 6),
                }
            )
        families.sort(key=lambda row: (float(row["avg_score"]), float(row["avg_win_rate"])), reverse=True)
        return {"families": families, "best_family": str(families[0]["family"]) if families else "none"}

    def _candidate_rejection_reasons(self, candidate, thresholds: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if float(candidate.trades) < thresholds["min_trades"]:
            reasons.append("low_trades")
        if float(candidate.win_rate) < thresholds["min_win_rate"]:
            reasons.append("low_win_rate")
        if float(candidate.profit_component) < thresholds["min_profit_component"]:
            reasons.append("negative_profit_component")
        if float(getattr(candidate, "trend_edge_score", 0.0) or 0.0) <= 0.0:
            reasons.append("weak_trend_edge")
        return reasons

    def _candidate_strategy(self, candidate) -> object:
        if hasattr(candidate, "to_strategy"):
            return candidate.to_strategy()
        return SimpleNamespace(
            fast=int(getattr(candidate, "fast", 0) or 0),
            slow=int(getattr(candidate, "slow", 0) or 0),
            rsi_buy=float(getattr(candidate, "rsi_buy", 50.0) or 50.0),
            rsi_sell=float(getattr(candidate, "rsi_sell", 50.0) or 50.0),
        )

    def _recent_slice_quality(self, candidate, df, lab: StrategyLab) -> dict[str, object]:
        slices = max(1, int(getattr(self.settings, "training_quality_validation_slices", 3) or 3))
        min_slice_win_rate = float(getattr(self.settings, "training_quality_min_slice_win_rate", 0.48) or 0.48)
        min_slice_profit_component = float(
            getattr(self.settings, "training_quality_min_slice_profit_component", -0.002) or -0.002
        )
        if len(df) < 600 or slices <= 1:
            return {"passed": True, "quality_score": 0.0, "validated_slices": []}
        if not hasattr(lab, "evaluate_strategy"):
            return {"passed": True, "quality_score": 0.0, "validated_slices": []}

        strategy = self._candidate_strategy(candidate)
        windows = []
        available = max(1, len(df) - 300)
        slice_size = max(300, min(len(df), available // slices))
        end = len(df)
        for _ in range(slices):
            start = max(0, end - slice_size)
            chunk = df.iloc[start:end].copy()
            if len(chunk) < 300:
                break
            windows.append(chunk)
            end = start
            if end <= 300:
                break
        windows = list(reversed(windows))
        if not windows:
            return {"passed": True, "quality_score": 0.0, "validated_slices": []}

        validated = []
        weak_count = 0
        quality_sum = 0.0
        for chunk in windows:
            metrics = dict(lab.evaluate_strategy(strategy, chunk) or {})
            win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
            profit_component = float(metrics.get("profit_component", 0.0) or 0.0)
            trades = int(metrics.get("trades", 0) or 0)
            passed = trades > 0 and win_rate >= min_slice_win_rate and profit_component >= min_slice_profit_component
            if not passed:
                weak_count += 1
            quality_sum += max(0.0, win_rate - min_slice_win_rate) + max(0.0, profit_component)
            validated.append(
                {
                    "bars": int(len(chunk)),
                    "trades": trades,
                    "win_rate": round(win_rate, 4),
                    "profit_component": round(profit_component, 6),
                    "passed": bool(passed),
                }
            )
        passed = weak_count == 0
        quality_score = quality_sum / max(1, len(validated))
        return {
            "passed": passed,
            "quality_score": round(quality_score, 6),
            "validated_slices": validated,
        }

    def _filter_candidates_by_recent_slice_quality(self, candidates: list, df, lab: StrategyLab) -> tuple[list, list[dict[str, object]]]:
        vetted = []
        rejected = []
        for candidate in candidates:
            quality = self._recent_slice_quality(candidate, df, lab)
            setattr(candidate, "quality_validation_score", float(quality.get("quality_score", 0.0) or 0.0))
            setattr(candidate, "validated_slices", list(quality.get("validated_slices", []) or []))
            if quality.get("passed", True):
                vetted.append(candidate)
            else:
                rejected.append(
                    {
                        "signature": candidate.signature(),
                        "reason": "fragile_recent_slices",
                        "quality_validation_score": float(quality.get("quality_score", 0.0) or 0.0),
                        "validated_slices": list(quality.get("validated_slices", []) or []),
                    }
                )
        return vetted, rejected

    def _promote_candidates(self, candidates: list) -> tuple[list, dict[str, object]]:
        thresholds = self._promotion_thresholds()
        promoted = []
        rejected = []
        for candidate in candidates:
            reasons = self._candidate_rejection_reasons(candidate, thresholds)
            if not reasons:
                promoted.append(candidate)
            else:
                rejected.append(
                    {
                        "signature": candidate.signature(),
                        "trades": int(candidate.trades),
                        "win_rate": float(candidate.win_rate),
                        "profit_component": float(candidate.profit_component),
                        "trend_edge_score": float(getattr(candidate, "trend_edge_score", 0.0) or 0.0),
                        "reasons": reasons,
                    }
                )

        if promoted:
            return promoted, {"mode": "strict", "rejected": rejected, "thresholds": thresholds}

        near_miss = [
            candidate
            for candidate in candidates
            if float(candidate.trades) >= thresholds["min_trades"] * 0.85
            and float(candidate.win_rate) >= thresholds["min_win_rate"] - 0.02
            and float(candidate.profit_component) >= max(-0.005, thresholds["min_profit_component"] - 0.005)
            and float(getattr(candidate, "trend_edge_score", 0.0) or 0.0) > 0.0
            and float(getattr(candidate, "range_dependency_score", 0.0) or 0.0) <= 0.02
            and float(getattr(candidate, "stop_hunt_vulnerability", 0.0) or 0.0) <= 0.05
        ]
        if near_miss:
            return near_miss, {"mode": "near_miss", "rejected": rejected, "thresholds": thresholds}

        return [], {"mode": "fallback_only", "rejected": rejected, "thresholds": thresholds}

    def train(self) -> HistoricalTrainingResult:
        started = time.monotonic()
        self._log_training_event(
            "Historical training started",
            {
                "symbol": str(self.settings.symbol),
                "timeframe": str(self.settings.timeframe),
                "months": int(self.settings.historical_train_months),
                "max_bars": int(self.settings.historical_train_max_bars),
            },
        )
        try:
            df = self._fetch_history()
            self._log_training_event(
                "Historical training fetched history",
                {
                    "bars": int(len(df)),
                    "fetch_meta": dict(self.feed.last_fetch_meta or {}),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            if len(df) < 400:
                raise RuntimeError(f"Not enough historical bars for training: {len(df)}")

            backtest_bars = max(600, int(getattr(self.settings, "historical_backtest_max_bars", len(df))))
            backtest_df = df.tail(min(len(df), backtest_bars)).copy()
            result: BacktestResult = Backtester(store=self.store, settings=self.settings).run(backtest_df)
            self._log_training_event(
                "Historical training backtest completed",
                {
                    "bars": int(len(backtest_df)),
                    "trades": int(result.trades),
                    "total_return": float(result.total_return),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            lab = StrategyLab(seed=17)
            evolved_train_bars = max(600, int(getattr(self.settings, "evolved_strategy_train_max_bars", 5000)))
            evolved_df = df.tail(min(len(df), evolved_train_bars)).copy()
            stage_timeout_seconds = max(
                5.0,
                float(getattr(self.settings, "historical_training_stage_timeout_seconds", 45.0) or 45.0),
            )
            score_adjuster = self._candidate_score_adjuster()
            if score_adjuster is not None:
                self._log_training_event(
                    "Historical training trend robustness scoring active",
                    {
                        "trend_weight": float(getattr(self.settings, "training_trend_robustness_weight", 0.0) or 0.0),
                        "range_penalty": float(getattr(self.settings, "training_range_dependency_penalty", 0.0) or 0.0),
                        "trend_loss_penalty": float(getattr(self.settings, "training_trend_loss_penalty_weight", 0.0) or 0.0),
                        "profit_alignment_weight": float(getattr(self.settings, "training_profit_alignment_weight", 0.0) or 0.0),
                        "profit_regression_penalty": float(getattr(self.settings, "training_profit_regression_penalty_weight", 0.0) or 0.0),
                        "min_win_rate_alignment": float(getattr(self.settings, "training_min_win_rate_alignment", 0.0) or 0.0),
                        "pullback_quality_weight": float(getattr(self.settings, "training_pullback_quality_weight", 0.0) or 0.0),
                        "wick_pressure_penalty": float(getattr(self.settings, "training_wick_pressure_penalty_weight", 0.0) or 0.0),
                        "trend_maturity_weight": float(getattr(self.settings, "training_trend_maturity_weight", 0.0) or 0.0),
                        "target_candidate_win_rate": float(getattr(self.settings, "training_target_candidate_win_rate", 0.0) or 0.0),
                        "overtrade_penalty_weight": float(getattr(self.settings, "training_overtrade_penalty_weight", 0.0) or 0.0),
                        "continuation_quality_weight": float(getattr(self.settings, "training_continuation_quality_weight", 0.0) or 0.0),
                    },
                )

            def _progress(meta: dict[str, object]) -> None:
                generation = int(meta.get("generation", 0) or 0)
                generations = int(meta.get("generations", 0) or 0)
                if generation <= 2 or generation == generations or generation % 5 == 0:
                    self._log_training_event("Historical training progress", meta)

            discover_kwargs = {
                "generations": int(getattr(self.settings, "evolved_strategy_generations", 24)),
                "population": int(getattr(self.settings, "evolved_strategy_population", 36)),
                "survivors": int(getattr(self.settings, "evolved_strategy_survivors", 4)),
                "max_seconds": stage_timeout_seconds,
                "progress_callback": _progress,
            }
            if score_adjuster is not None:
                discover_kwargs["score_adjuster"] = score_adjuster
            candidates = lab.discover_candidates(
                evolved_df,
                **discover_kwargs,
            )
            candidates, quality_rejections = self._filter_candidates_by_recent_slice_quality(candidates, evolved_df, lab)
            if quality_rejections:
                self._log_training_event(
                    "Historical training rejected fragile candidates on recent slices",
                    {
                        "rejected_count": len(quality_rejections),
                        "rejections": quality_rejections[:10],
                    },
                )
            candidates, promotion_meta = self._promote_candidates(candidates)
            if quality_rejections:
                promotion_meta = {
                    **dict(promotion_meta or {}),
                    "quality_rejections": quality_rejections,
                }
            if candidates:
                if promotion_meta.get("mode") == "near_miss":
                    self._log_training_event(
                        "Historical training promoted near-miss candidates",
                        {
                            "thresholds": dict(promotion_meta.get("thresholds", {})),
                            "rejected_count": len(list(promotion_meta.get("rejected", []) or [])),
                        },
                    )
                champion = candidates[0]
                best_strategy = {
                    "fast": champion.fast,
                    "slow": champion.slow,
                    "rsi_buy": champion.rsi_buy,
                    "rsi_sell": champion.rsi_sell,
                    "profit_score": champion.score,
                    "trained_bars": len(backtest_df),
                    "trades": champion.trades,
                    "wins": champion.wins,
                    "losses": champion.losses,
                    "win_rate": champion.win_rate,
                    "signature": champion.signature(),
                    "source": champion.source,
                    "mutation_train_bars": len(evolved_df),
                }
                evolved_pool = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "trained_bars": len(evolved_df),
                    "backtest_bars": len(backtest_df),
                    "count": len(candidates),
                    "promotion_mode": str(promotion_meta.get("mode", "strict")),
                    "promotion_rejections": list(promotion_meta.get("rejected", []) or []),
                    "champion_signature": champion.signature(),
                    "candidates": [candidate.to_dict() for candidate in candidates],
                }
            else:
                self._log_training_event(
                    "Historical training candidate search yielded no promoted pool; using fallback",
                    {
                        "timeout_seconds": float(stage_timeout_seconds),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "promotion_meta": promotion_meta,
                    },
                )
                best, best_score = lab.optimize_for_profit(
                    evolved_df,
                    generations=min(6, int(getattr(self.settings, "evolved_strategy_generations", 24))),
                    population=min(10, int(getattr(self.settings, "evolved_strategy_population", 36))),
                    max_seconds=max(5.0, stage_timeout_seconds / 2.0),
                    progress_callback=_progress,
                )
                best_strategy = {
                    "fast": best.fast,
                    "slow": best.slow,
                    "rsi_buy": best.rsi_buy,
                    "rsi_sell": best.rsi_sell,
                    "profit_score": best_score,
                    "trained_bars": len(backtest_df),
                    "source": "optimize_for_profit_fallback",
                    "mutation_train_bars": len(evolved_df),
                }
                evolved_pool = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "trained_bars": len(evolved_df),
                    "backtest_bars": len(backtest_df),
                    "count": 0,
                    "candidates": [],
                }
        except Exception as exc:
            self._log_training_event(
                "Historical training failed",
                {
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "fetch_meta": dict(self.feed.last_fetch_meta or {}),
                },
            )
            raise
        self.store.update_learning_state(
            lambda existing: {
                **existing,
                "best_evolved_strategy": best_strategy,
                "evolved_strategy_pool": evolved_pool,
            }
        )
        existing = self.store.load_learning_state().get("state", {})

        loss_memory = existing.get("loss_memory", {})
        recent_outcomes = list(loss_memory.get("recent_outcomes", []))
        session_summary: dict[str, dict[str, int]] = {}
        for x in recent_outcomes[-200:]:
            ses = str(x.get("session", "unknown"))
            out = str(x.get("outcome", "draw"))
            bucket = session_summary.setdefault(ses, {"win": 0, "loss": 0, "draw": 0})
            if out in bucket:
                bucket[out] += 1

        payload = {
            "bars": len(backtest_df),
            "total_return": result.total_return,
            "trades": result.trades,
            "wins": result.wins,
            "losses": result.losses,
            "expectancy": result.expectancy,
            "profit_factor": result.profit_factor,
            "max_drawdown": result.max_drawdown,
            "sharpe": result.sharpe,
            "sortino": result.sortino,
            "average_win": result.average_win,
            "average_loss": result.average_loss,
            "risk_of_ruin": result.risk_of_ruin,
            "average_hold_bars": result.average_hold_bars,
            "regime_performance": result.regime_performance,
            "months": self.settings.historical_train_months,
            "max_bars": self.settings.historical_train_max_bars,
            "backtest_bars": len(backtest_df),
            "mutation_train_bars": len(evolved_df),
            "best_strategy": best_strategy,
            "promoted_candidates": evolved_pool,
            "setup_family_summary": self._setup_family_summary(evolved_pool),
            "weak_point_summary": self._weak_point_summary(best_strategy, evolved_pool),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "labels_summary": {
                "recent_outcomes_count": len(recent_outcomes),
                "session_outcomes": session_summary,
            },
        }
        self.store.log_event("training", "Historical training completed", payload)
        return HistoricalTrainingResult(
            bars=len(backtest_df),
            total_return=result.total_return,
            trades=result.trades,
            wins=result.wins,
            losses=result.losses,
        )
