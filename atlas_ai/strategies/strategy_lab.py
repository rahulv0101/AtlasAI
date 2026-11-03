from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import pandas as pd

from atlas_ai.models import TradeSignal
from atlas_ai.strategies.base import Strategy
from atlas_ai.strategies.indicators import atr, ema, rsi


@dataclass
class EvolvedRuleStrategy(Strategy):
    fast: int
    slow: int
    rsi_buy: float
    rsi_sell: float
    name: str = "evolved_rule"

    def generate_signal(self, df: pd.DataFrame) -> TradeSignal:
        close = df["close"]
        e_fast = ema(close, self.fast)
        e_slow = ema(close, self.slow)
        r = rsi(close).iloc[-1]
        vol = atr(df).iloc[-1]

        side = "flat"
        if e_fast.iloc[-1] > e_slow.iloc[-1] and r < self.rsi_buy:
            side = "buy"
        elif e_fast.iloc[-1] < e_slow.iloc[-1] and r > self.rsi_sell:
            side = "sell"

        confidence = min(abs(e_fast.iloc[-1] - e_slow.iloc[-1]) / max(close.iloc[-1], 1) * 1000, 1.0)
        if side == "flat":
            confidence = 0.0
        return TradeSignal(self.name, side, confidence, max(vol * 1.9, 100), max(vol * 2.8, 160))


@dataclass
class EvolvedStrategyCandidate:
    fast: int
    slow: int
    rsi_buy: float
    rsi_sell: float
    score: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_component: float
    confidence_floor: float
    generations: int
    population: int
    trend_trades: int = 0
    trend_wins: int = 0
    trend_losses: int = 0
    trend_win_rate: float = 0.0
    trend_profit_component: float = 0.0
    range_trades: int = 0
    range_wins: int = 0
    range_losses: int = 0
    range_win_rate: float = 0.0
    range_profit_component: float = 0.0
    trend_edge_score: float = 0.0
    range_dependency_score: float = 0.0
    stop_hunt_vulnerability: float = 0.0
    pullback_quality_score: float = 0.0
    wick_pressure_score: float = 0.0
    trend_maturity_score: float = 0.0
    continuation_quality_score: float = 0.0
    explainability_summary: str = ""
    source: str = "mutation_tournament"

    def to_strategy(self) -> EvolvedRuleStrategy:
        return EvolvedRuleStrategy(
            fast=self.fast,
            slow=self.slow,
            rsi_buy=self.rsi_buy,
            rsi_sell=self.rsi_sell,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["signature"] = self.signature()
        return payload

    def signature(self) -> str:
        return (
            f"f{int(self.fast)}_s{int(self.slow)}_"
            f"rb{float(self.rsi_buy):.1f}_rs{float(self.rsi_sell):.1f}"
        )

    @staticmethod
    def explainability_from_metrics(metrics: dict[str, float]) -> dict[str, Any]:
        trend_profit = float(metrics.get("trend_profit_component", 0.0) or 0.0)
        range_profit = float(metrics.get("range_profit_component", 0.0) or 0.0)
        trend_win_rate = float(metrics.get("trend_win_rate", 0.0) or 0.0)
        range_win_rate = float(metrics.get("range_win_rate", 0.0) or 0.0)
        trend_losses = float(metrics.get("trend_losses", 0.0) or 0.0)
        trend_wins = float(metrics.get("trend_wins", 0.0) or 0.0)
        trades = float(metrics.get("trades", 0.0) or 0.0)
        trend_trades = float(metrics.get("trend_trades", 0.0) or 0.0)
        trend_share = trend_trades / max(1.0, trades)
        trend_edge_score = trend_profit * (0.65 + 0.35 * trend_win_rate)
        range_dependency_score = max(0.0, range_profit - trend_profit)
        stop_hunt_vulnerability = max(0.0, trend_losses - trend_wins) / max(1.0, trades)

        parts: list[str] = []
        if trend_edge_score <= 0.0:
            parts.append("trend edge is weak")
        elif trend_win_rate < 0.5:
            parts.append("trend entries are directionally fragile")
        else:
            parts.append("trend edge is usable")

        if range_dependency_score > 0.0:
            parts.append("performance leans on range behavior")
        else:
            parts.append("performance is not range-dependent")

        if stop_hunt_vulnerability >= 0.08:
            parts.append("stop-hunt vulnerability is elevated")
        elif stop_hunt_vulnerability >= 0.03:
            parts.append("stop-hunt vulnerability is moderate")
        else:
            parts.append("stop-hunt vulnerability is controlled")

        if trend_share >= 0.65:
            parts.append("candidate engages trend conditions often")
        elif trend_share <= 0.35:
            parts.append("candidate sees relatively few trend conditions")

        return {
            "trend_edge_score": float(trend_edge_score),
            "range_dependency_score": float(range_dependency_score),
            "stop_hunt_vulnerability": float(stop_hunt_vulnerability),
            "explainability_summary": "; ".join(parts),
        }


class StrategyLab:
    """Generates and evolves simple rule-based strategies from market history."""

    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)

    def random_strategy(self) -> EvolvedRuleStrategy:
        fast = self.random.randint(5, 35)
        slow = self.random.randint(max(fast + 5, 25), 90)
        rsi_buy = self.random.uniform(30, 55)
        rsi_sell = self.random.uniform(45, 70)
        return EvolvedRuleStrategy(fast=fast, slow=slow, rsi_buy=rsi_buy, rsi_sell=rsi_sell)

    def mutate(self, strategy: EvolvedRuleStrategy) -> EvolvedRuleStrategy:
        return EvolvedRuleStrategy(
            fast=max(5, strategy.fast + self.random.randint(-3, 3)),
            slow=max(strategy.fast + 5, strategy.slow + self.random.randint(-4, 4)),
            rsi_buy=min(60, max(20, strategy.rsi_buy + self.random.uniform(-2, 2))),
            rsi_sell=min(80, max(40, strategy.rsi_sell + self.random.uniform(-2, 2))),
            name=strategy.name,
        )

    def score_strategy(self, strategy: EvolvedRuleStrategy, df: pd.DataFrame) -> float:
        return float(self.evaluate_strategy(strategy, df)["score"])

    @staticmethod
    def _signature(strategy: EvolvedRuleStrategy) -> str:
        return (
            f"f{int(strategy.fast)}_s{int(strategy.slow)}_"
            f"rb{float(strategy.rsi_buy):.3f}_rs{float(strategy.rsi_sell):.3f}"
        )

    @staticmethod
    def _regime_name(closes: pd.Series) -> str:
        if len(closes) < 30:
            return "unknown"
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if abs(first) < 1e-9:
            return "unknown"
        drift_pct = abs((last - first) / first) * 100.0
        abs_moves = closes.diff().abs().dropna()
        avg_abs = float(abs_moves.mean()) if not abs_moves.empty else 0.0
        vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
        if drift_pct >= max(0.12, vol_pct * 2.2):
            return "trend"
        if vol_pct >= 0.06:
            return "high_volatility_range"
        return "range"

    def evaluate_strategy(self, strategy: EvolvedRuleStrategy, df: pd.DataFrame) -> dict[str, float]:
        if len(df) < 300:
            return {
                "score": -999.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_component": -999.0,
                "confidence_floor": 0.0,
            }

        # Precompute indicators once for the full dataset to avoid O(n^2)
        # recalculation inside the walk-forward loop.
        close = df["close"]
        e_fast = ema(close, strategy.fast)
        e_slow = ema(close, strategy.slow)
        r = rsi(close)

        equity = 0.0
        trades = 0
        wins = 0
        losses = 0
        confidence_sum = 0.0
        pullback_quality_sum = 0.0
        wick_pressure_sum = 0.0
        trend_maturity_sum = 0.0
        continuation_quality_sum = 0.0
        regime_stats: dict[str, dict[str, float]] = {
            "trend": {"trades": 0.0, "wins": 0.0, "losses": 0.0, "profit_component": 0.0},
            "range": {"trades": 0.0, "wins": 0.0, "losses": 0.0, "profit_component": 0.0},
            "high_volatility_range": {"trades": 0.0, "wins": 0.0, "losses": 0.0, "profit_component": 0.0},
        }

        for i in range(220, len(df) - 2):
            current = float(close.iat[i])
            nxt = float(close.iat[i + 1])
            change = (nxt - current) / max(current, 1.0)

            fast_val = float(e_fast.iat[i])
            slow_val = float(e_slow.iat[i])
            rsi_val = float(r.iat[i])

            side = "flat"
            if fast_val > slow_val and rsi_val < strategy.rsi_buy:
                side = "buy"
            elif fast_val < slow_val and rsi_val > strategy.rsi_sell:
                side = "sell"

            confidence = min(abs(fast_val - slow_val) / max(current, 1.0) * 1000, 1.0)
            if side == "flat" or confidence < 0.1:
                continue

            pnl = change if side == "buy" else -change
            pnl *= (1 + confidence)
            regime = self._regime_name(close.iloc[max(0, i - 59) : i + 1])
            bar = df.iloc[i]
            open_px = float(bar["open"])
            high_px = float(bar["high"])
            low_px = float(bar["low"])
            close_px = float(bar["close"])
            bar_range = max(1e-9, high_px - low_px)
            body = abs(close_px - open_px)
            upper_wick = max(0.0, high_px - max(open_px, close_px))
            lower_wick = max(0.0, min(open_px, close_px) - low_px)
            if side == "buy":
                pullback_quality = max(0.0, min(1.0, (e_fast.iat[i] - current) / max(bar_range, current * 0.0002)))
                wick_pressure = min(1.0, upper_wick / bar_range)
            else:
                pullback_quality = max(0.0, min(1.0, (current - e_fast.iat[i]) / max(bar_range, current * 0.0002)))
                wick_pressure = min(1.0, lower_wick / bar_range)
            ema_gap = abs(fast_val - slow_val) / max(current, 1.0)
            trend_maturity = min(1.0, ema_gap * 1800.0) * min(1.0, body / bar_range)
            continuation_quality = max(0.0, min(1.0, (1.0 - wick_pressure) * (0.55 + 0.45 * confidence) * (0.5 + 0.5 * min(1.0, body / bar_range))))
            equity += pnl
            trades += 1
            confidence_sum += confidence
            pullback_quality_sum += pullback_quality
            wick_pressure_sum += wick_pressure
            trend_maturity_sum += trend_maturity
            continuation_quality_sum += continuation_quality
            bucket = regime_stats.setdefault(
                regime,
                {"trades": 0.0, "wins": 0.0, "losses": 0.0, "profit_component": 0.0},
            )
            bucket["trades"] += 1.0
            bucket["profit_component"] += float(pnl)
            if pnl >= 0:
                wins += 1
                bucket["wins"] += 1.0
            else:
                losses += 1
                bucket["losses"] += 1.0

        if trades == 0:
            return {
                "score": -999.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "profit_component": -999.0,
                "confidence_floor": 0.0,
            }

        win_rate = wins / max(1, (wins + losses))
        profit_score = equity * (0.6 + 0.4 * win_rate)
        confidence_floor = confidence_sum / max(1, trades)
        trend_stats = dict(regime_stats.get("trend", {}))
        range_stats = dict(regime_stats.get("range", {}))
        trend_trades = int(trend_stats.get("trades", 0.0) or 0.0)
        range_trades = int(range_stats.get("trades", 0.0) or 0.0)
        trend_wins = int(trend_stats.get("wins", 0.0) or 0.0)
        trend_losses = int(trend_stats.get("losses", 0.0) or 0.0)
        range_wins = int(range_stats.get("wins", 0.0) or 0.0)
        range_losses = int(range_stats.get("losses", 0.0) or 0.0)
        pullback_quality_score = pullback_quality_sum / max(1, trades)
        wick_pressure_score = wick_pressure_sum / max(1, trades)
        trend_maturity_score = trend_maturity_sum / max(1, trades)
        continuation_quality_score = continuation_quality_sum / max(1, trades)
        return {
            "score": float(profit_score),
            "trades": int(trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "profit_component": float(equity),
            "confidence_floor": float(confidence_floor),
            "trend_trades": trend_trades,
            "trend_wins": trend_wins,
            "trend_losses": trend_losses,
            "trend_win_rate": float(trend_wins / max(1, trend_wins + trend_losses)),
            "trend_profit_component": float(trend_stats.get("profit_component", 0.0) or 0.0),
            "range_trades": range_trades,
            "range_wins": range_wins,
            "range_losses": range_losses,
            "range_win_rate": float(range_wins / max(1, range_wins + range_losses)),
            "range_profit_component": float(range_stats.get("profit_component", 0.0) or 0.0),
            "pullback_quality_score": float(pullback_quality_score),
            "wick_pressure_score": float(wick_pressure_score),
            "trend_maturity_score": float(trend_maturity_score),
            "continuation_quality_score": float(continuation_quality_score),
            **EvolvedStrategyCandidate.explainability_from_metrics(
                {
                    "trades": trades,
                    "trend_trades": trend_trades,
                    "trend_wins": trend_wins,
                    "trend_losses": trend_losses,
                    "trend_win_rate": float(trend_wins / max(1, trend_wins + trend_losses)),
                    "trend_profit_component": float(trend_stats.get("profit_component", 0.0) or 0.0),
                    "range_win_rate": float(range_wins / max(1, range_wins + range_losses)),
                    "range_profit_component": float(range_stats.get("profit_component", 0.0) or 0.0),
                }
            ),
        }

    def backtest_profit_score(self, strategy: EvolvedRuleStrategy, df: pd.DataFrame) -> float:
        return float(self.evaluate_strategy(strategy, df)["score"])

    def evolve(
        self,
        df: pd.DataFrame,
        generations: int = 15,
        population: int = 20,
        max_seconds: float | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> EvolvedRuleStrategy:
        pool = [self.random_strategy() for _ in range(population)]
        score_cache: dict[str, float] = {}
        started = time.monotonic()

        def _score(strategy: EvolvedRuleStrategy) -> float:
            signature = self._signature(strategy)
            cached = score_cache.get(signature)
            if cached is not None:
                return cached
            value = self.score_strategy(strategy, df)
            score_cache[signature] = value
            return value

        for idx in range(generations):
            ranked = sorted(pool, key=_score, reverse=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "evolve",
                        "generation": int(idx + 1),
                        "generations": int(generations),
                        "best_score": float(_score(ranked[0])) if ranked else -999.0,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
            if max_seconds is not None and (time.monotonic() - started) >= max(0.1, float(max_seconds)):
                break
            elites = ranked[: max(2, population // 4)]
            next_gen = list(elites)
            while len(next_gen) < population:
                parent = self.random.choice(elites)
                child = self.mutate(parent)
                next_gen.append(child)
            pool = next_gen

        return max(pool, key=_score)

    def optimize_for_profit(
        self,
        df: pd.DataFrame,
        generations: int = 20,
        population: int = 30,
        max_seconds: float | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[EvolvedRuleStrategy, float]:
        best = self.evolve(
            df,
            generations=generations,
            population=population,
            max_seconds=max_seconds,
            progress_callback=progress_callback,
        )
        score = self.backtest_profit_score(best, df)
        return best, score

    def discover_candidates(
        self,
        df: pd.DataFrame,
        generations: int = 20,
        population: int = 30,
        survivors: int = 4,
        max_seconds: float | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        score_adjuster: Callable[[dict[str, float]], float] | None = None,
    ) -> list[EvolvedStrategyCandidate]:
        if len(df) < 300:
            return []

        pool = [self.random_strategy() for _ in range(max(population, survivors * 3))]
        score_cache: dict[str, float] = {}
        metrics_cache: dict[str, dict[str, float]] = {}
        started = time.monotonic()

        def _metrics(strategy: EvolvedRuleStrategy) -> dict[str, float]:
            signature = self._signature(strategy)
            cached = metrics_cache.get(signature)
            if cached is not None:
                return cached
            value = self.evaluate_strategy(strategy, df)
            if score_adjuster is not None:
                value["score"] = float(score_adjuster(dict(value)))
            metrics_cache[signature] = value
            score_cache[signature] = float(value["score"])
            return value

        def _score(strategy: EvolvedRuleStrategy) -> float:
            signature = self._signature(strategy)
            cached = score_cache.get(signature)
            if cached is not None:
                return cached
            return float(_metrics(strategy)["score"])

        total_generations = max(1, generations)
        for idx in range(total_generations):
            ranked = sorted(pool, key=_score, reverse=True)
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "discover_candidates",
                        "generation": int(idx + 1),
                        "generations": int(total_generations),
                        "best_score": float(_score(ranked[0])) if ranked else -999.0,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                    }
                )
            if max_seconds is not None and (time.monotonic() - started) >= max(0.1, float(max_seconds)):
                break
            elites = ranked[: max(3, population // 4)]
            next_gen = list(elites)
            while len(next_gen) < max(population, survivors * 3):
                parent = self.random.choice(elites)
                child = self.mutate(parent)
                next_gen.append(child)
            pool = next_gen

        ranked = sorted(pool, key=_score, reverse=True)
        unique: list[EvolvedStrategyCandidate] = []
        seen: set[str] = set()
        for strategy in ranked:
            metrics = _metrics(strategy)
            if float(metrics["score"]) <= -999.0:
                continue
            explainability = EvolvedStrategyCandidate.explainability_from_metrics(metrics)
            candidate = EvolvedStrategyCandidate(
                fast=int(strategy.fast),
                slow=int(strategy.slow),
                rsi_buy=float(strategy.rsi_buy),
                rsi_sell=float(strategy.rsi_sell),
                score=float(metrics["score"]),
                trades=int(metrics["trades"]),
                wins=int(metrics["wins"]),
                losses=int(metrics["losses"]),
                win_rate=float(metrics["win_rate"]),
                profit_component=float(metrics["profit_component"]),
                confidence_floor=float(metrics["confidence_floor"]),
                generations=int(generations),
                population=int(population),
                trend_trades=int(metrics.get("trend_trades", 0)),
                trend_wins=int(metrics.get("trend_wins", 0)),
                trend_losses=int(metrics.get("trend_losses", 0)),
                trend_win_rate=float(metrics.get("trend_win_rate", 0.0)),
                trend_profit_component=float(metrics.get("trend_profit_component", 0.0)),
                range_trades=int(metrics.get("range_trades", 0)),
                range_wins=int(metrics.get("range_wins", 0)),
                range_losses=int(metrics.get("range_losses", 0)),
                range_win_rate=float(metrics.get("range_win_rate", 0.0)),
                range_profit_component=float(metrics.get("range_profit_component", 0.0)),
                trend_edge_score=float(metrics.get("trend_edge_score", explainability["trend_edge_score"])),
                range_dependency_score=float(metrics.get("range_dependency_score", explainability["range_dependency_score"])),
                stop_hunt_vulnerability=float(metrics.get("stop_hunt_vulnerability", explainability["stop_hunt_vulnerability"])),
                pullback_quality_score=float(metrics.get("pullback_quality_score", 0.0)),
                wick_pressure_score=float(metrics.get("wick_pressure_score", 0.0)),
                trend_maturity_score=float(metrics.get("trend_maturity_score", 0.0)),
                continuation_quality_score=float(metrics.get("continuation_quality_score", 0.0)),
                explainability_summary=str(metrics.get("explainability_summary", explainability["explainability_summary"])),
            )
            signature = candidate.signature()
            if signature in seen:
                continue
            seen.add(signature)
            unique.append(candidate)
            if len(unique) >= max(1, survivors):
                break
        return unique
