from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from collections import Counter

import pandas as pd

from atlas_ai.config.settings import Settings
from atlas_ai.learning.online_allocator import OnlineAllocator
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.regime_filters import (
    build_research_regime_overlay_context,
    conservative_pretrade_filter_reason,
    research_regime_filter_reason,
)
from atlas_ai.strategies.built_in import BreakoutStrategy, CandlestickPatternStrategy, EMACrossoverStrategy, RSIMeanReversionStrategy
from atlas_ai.strategies.minute_pack import (
    BollingerMeanReversionStrategy,
    GoldBreakoutRetestStrategy,
    MACDTrendStrategy,
    MomentumContinuationStrategy,
    PullbackTrendStrategy,
    RSITrendHybridStrategy,
    StochasticReversalStrategy,
    VWAPProxyReversionStrategy,
)
from atlas_ai.strategies.strategy_lab import StrategyLab


@dataclass
class BacktestResult:
    total_return: float
    trades: int
    wins: int
    losses: int
    win_rate: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    risk_of_ruin: float = 1.0
    average_hold_bars: float = 0.0
    regime_performance: dict[str, dict[str, float]] | None = None
    trade_list: list[dict[str, object]] | None = None


class Backtester:
    def __init__(self, store: KnowledgeStore | None = None, settings: Settings | None = None):
        self.store = store
        self.settings = settings
        persist_learning = bool(getattr(settings, "backtest_persist_learning_state", False)) if settings is not None else False
        # Backtests must never mutate live allocator state. Persisting allocator
        # learning is allowed only for explicitly isolated training stores.
        self.allocator = OnlineAllocator(store=store if persist_learning else None)
        if settings is not None:
            self.allocator.set_exploration(float(getattr(settings, "exploration_epsilon", 0.0)))
        self.base = [
            EMACrossoverStrategy(),
            RSIMeanReversionStrategy(),
            BreakoutStrategy(),
            CandlestickPatternStrategy(),
            MACDTrendStrategy(),
            BollingerMeanReversionStrategy(),
            StochasticReversalStrategy(),
            VWAPProxyReversionStrategy(),
            GoldBreakoutRetestStrategy(),
            MomentumContinuationStrategy(),
            PullbackTrendStrategy(),
            RSITrendHybridStrategy(),
        ]
        self.lab = StrategyLab(seed=7)
        self._last_simulated_trade_details: dict[str, object] = {}

    def _point_size(self) -> float:
        if self.settings is None:
            return 0.01
        return max(1e-6, float(getattr(self.settings, "backtest_point_size", 0.01)))

    def _exit_params(self) -> dict[str, float]:
        if self.settings is None:
            return {
                "break_even_r_trigger": 0.9,
                "break_even_lock_r": 0.05,
                "trailing_r_trigger": 1.4,
                "trailing_buffer_r": 0.9,
                "stale_exit_bars": 4.0,
                "stale_min_progress_r": 0.15,
            }
        return {
            "break_even_r_trigger": max(0.1, float(getattr(self.settings, "backtest_break_even_r_trigger", 0.9))),
            "break_even_lock_r": max(0.0, float(getattr(self.settings, "backtest_break_even_lock_r", 0.05))),
            "trailing_r_trigger": max(0.2, float(getattr(self.settings, "backtest_trailing_r_trigger", 1.4))),
            "trailing_buffer_r": max(0.1, float(getattr(self.settings, "backtest_trailing_buffer_r", 0.9))),
            "stale_exit_bars": float(max(1, int(getattr(self.settings, "backtest_stale_exit_bars", 4)))),
            "stale_min_progress_r": max(0.0, float(getattr(self.settings, "backtest_stale_min_progress_r", 0.15))),
        }

    def _strategy_enabled(self, strategy_name: str) -> bool:
        enabled = {str(x).strip() for x in getattr(self.settings, "enabled_strategies", ()) if str(x).strip()}
        disabled = {str(x).strip() for x in getattr(self.settings, "disabled_strategies", ())}
        name = str(strategy_name or "").strip()
        if enabled and name not in enabled:
            return False
        return name not in disabled

    def _regime_name(self, closes: list[float]) -> str:
        if len(closes) < 30:
            return "unknown"
        first = float(closes[0])
        last = float(closes[-1])
        if abs(first) < 1e-9:
            return "unknown"
        drift_pct = abs((last - first) / first) * 100.0
        abs_moves = [abs(float(closes[i]) - float(closes[i - 1])) for i in range(1, len(closes))]
        avg_abs = sum(abs_moves) / max(1, len(abs_moves))
        vol_pct = (avg_abs / max(1e-9, abs(last))) * 100.0
        if drift_pct >= max(0.12, vol_pct * 2.2):
            return "trend"
        if vol_pct >= 0.06:
            return "high_volatility_range"
        return "range"

    def _session_label(self, ts) -> str:
        try:
            hour = int(pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")).hour
        except Exception:
            hour = 0
        if 7 <= hour < 12:
            return "london"
        if 12 <= hour < 17:
            return "new_york"
        if 17 <= hour < 21:
            return "late_us"
        return "other"

    def _classify_backtest_loss(
        self,
        *,
        outcome: str,
        exit_reason: str,
        duration_minutes: float,
        session: str,
        regime: str,
        best_mfe_r: float = 0.0,
        setup_family: str = "unknown",
    ) -> str:
        if outcome != "loss":
            return "none"
        if setup_family == "continuation" and best_mfe_r < 0.25:
            return "false_continuation"
        if setup_family == "pullback" and best_mfe_r < 0.35:
            return "weak_pullback"
        if best_mfe_r >= 0.45 and exit_reason == "stop_loss":
            return "late_entry"
        if exit_reason == "stop_loss" and duration_minutes <= 5:
            return "stop_hunt"
        if regime in {"range", "high_volatility_range"}:
            return "weak_structure"
        if session in {"late_us", "other"}:
            return "poor_session"
        return "momentum_failure"

    @staticmethod
    def _setup_family_from_strategy(strategy_name: str) -> str:
        name = str(strategy_name or "").split("(", 1)[0].strip().lower()
        if name in {"momentum_continuation", "macd_trend", "gold_micro_impulse"}:
            return "continuation"
        if name in {"pullback_trend", "ema_crossover", "rsi_trend_hybrid"}:
            return "pullback"
        if name in {"breakout"}:
            return "breakout"
        if name in {"rsi_mean_reversion", "bollinger_mean_reversion", "vwap_proxy_reversion", "stochastic_reversal"}:
            return "mean_reversion"
        return "unknown"

    def _persist_backtest_outcomes(
        self,
        outcomes: list[dict[str, object]],
        strategy_stats: dict[str, dict[str, float]],
    ) -> None:
        if self.store is None or not outcomes:
            return
        existing = self.store.load_learning_state().get("state", {})
        memory = dict(existing.get("loss_memory", {}) or {})
        recent_outcomes = list(memory.get("recent_outcomes", []) or [])
        recent_outcomes.extend(outcomes)
        recent_outcomes = recent_outcomes[-200:]
        side_hour_stats = dict(memory.get("side_hour_stats", {}) or {})
        for key, row in strategy_stats.items():
            current = dict(side_hour_stats.get(key, {"wins": 0, "losses": 0, "pnl": 0.0}))
            current["wins"] = int(current.get("wins", 0)) + int(row.get("wins", 0) or 0)
            current["losses"] = int(current.get("losses", 0)) + int(row.get("losses", 0) or 0)
            current["pnl"] = float(current.get("pnl", 0.0)) + float(row.get("pnl", 0.0) or 0.0)
            side_hour_stats[key] = current
        memory["recent_outcomes"] = recent_outcomes
        memory["side_hour_stats"] = side_hour_stats
        self.store.update_learning_state(lambda state: {**state, "loss_memory": memory})

    @staticmethod
    def _top_counts(rows: list[dict[str, object]], field: str, limit: int = 8) -> list[dict[str, object]]:
        counts = Counter(str(row.get(field, "unknown") or "unknown") for row in rows)
        return [{"value": value, "count": int(count)} for value, count in counts.most_common(limit)]

    def _persist_historical_loss_analysis(
        self,
        *,
        losses: list[dict[str, object]],
        result_summary: dict[str, object],
        initial_equity: float,
        df: pd.DataFrame,
    ) -> None:
        if self.store is None:
            return
        if self.settings is not None and not bool(getattr(self.settings, "backtest_record_loss_analysis", True)):
            return

        max_samples = 200
        if self.settings is not None:
            max_samples = max(10, int(getattr(self.settings, "backtest_loss_analysis_max_samples", 200) or 200))
        sampled_losses = losses[-max_samples:]
        total_losses = int(result_summary.get("losses", len(losses)) or len(losses))
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "historical_backtest",
            "symbol": str(getattr(self.settings, "symbol", "unknown") if self.settings is not None else "unknown"),
            "timeframe": str(getattr(self.settings, "timeframe", "unknown") if self.settings is not None else "unknown"),
            "training_balance_zar": float(initial_equity),
            "bars": int(len(df)),
            "first_bar_utc": str(pd.Timestamp(df.iloc[0]["time"]).isoformat()) if len(df) and "time" in df.columns else "",
            "last_bar_utc": str(pd.Timestamp(df.iloc[-1]["time"]).isoformat()) if len(df) and "time" in df.columns else "",
            "total_trades": int(result_summary.get("total_trades", 0) or 0),
            "wins": int(result_summary.get("wins", 0) or 0),
            "losses": total_losses,
            "win_rate": float(result_summary.get("win_rate", 0.0) or 0.0),
            "total_return": float(result_summary.get("total_return", 0.0) or 0.0),
            "sampled_loss_count": int(len(sampled_losses)),
            "patterns": {
                "classification": self._top_counts(sampled_losses, "loss_classification"),
                "strategy": self._top_counts(sampled_losses, "strategy"),
                "session": self._top_counts(sampled_losses, "session"),
                "regime": self._top_counts(sampled_losses, "regime"),
                "side": self._top_counts(sampled_losses, "side"),
                "exit_reason": self._top_counts(sampled_losses, "exit_reason"),
            },
            "teaching_notes": [
                "Use sampled historical losses as training evidence, not hidden execution policy.",
                "Focus on why entries failed: false continuation, weak pullback, stop hunt, late entry, weak structure, or poor session.",
                "Compare loss classes against winners before changing live behavior.",
            ],
            "loss_samples": sampled_losses,
        }
        self.store.update_learning_state(lambda state: {**state, "historical_loss_analysis": payload})
        self.store.log_event("training", "Historical backtest loss analysis recorded", payload)

    def _persist_historical_winner_analysis(
        self,
        *,
        wins: list[dict[str, object]],
        result_summary: dict[str, object],
        initial_equity: float,
        df: pd.DataFrame,
    ) -> None:
        if self.store is None:
            return
        if self.settings is not None and not bool(getattr(self.settings, "backtest_record_loss_analysis", True)):
            return

        max_samples = 200
        if self.settings is not None:
            max_samples = max(10, int(getattr(self.settings, "backtest_loss_analysis_max_samples", 200) or 200))
        sampled_wins = wins[-max_samples:]
        total_wins = int(result_summary.get("wins", len(wins)) or len(wins))
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "historical_backtest",
            "symbol": str(getattr(self.settings, "symbol", "unknown") if self.settings is not None else "unknown"),
            "timeframe": str(getattr(self.settings, "timeframe", "unknown") if self.settings is not None else "unknown"),
            "training_balance_zar": float(initial_equity),
            "bars": int(len(df)),
            "first_bar_utc": str(pd.Timestamp(df.iloc[0]["time"]).isoformat()) if len(df) and "time" in df.columns else "",
            "last_bar_utc": str(pd.Timestamp(df.iloc[-1]["time"]).isoformat()) if len(df) and "time" in df.columns else "",
            "total_trades": int(result_summary.get("total_trades", 0) or 0),
            "wins": total_wins,
            "losses": int(result_summary.get("losses", 0) or 0),
            "win_rate": float(result_summary.get("win_rate", 0.0) or 0.0),
            "total_return": float(result_summary.get("total_return", 0.0) or 0.0),
            "sampled_win_count": int(len(sampled_wins)),
            "patterns": {
                "strategy": self._top_counts(sampled_wins, "strategy"),
                "session": self._top_counts(sampled_wins, "session"),
                "regime": self._top_counts(sampled_wins, "regime"),
                "side": self._top_counts(sampled_wins, "side"),
                "exit_reason": self._top_counts(sampled_wins, "exit_reason"),
                "setup_family": self._top_counts(sampled_wins, "setup_family"),
            },
            "teaching_notes": [
                "Use sampled historical winners as positive setup evidence, not hidden execution policy.",
                "Compare winning contexts against loss-to-win corrections before changing live behavior.",
                "Winning patterns can confirm what to wait for; they cannot force trades or bypass gates.",
            ],
            "win_samples": sampled_wins,
        }
        self.store.update_learning_state(lambda state: {**state, "historical_winner_analysis": payload})
        self.store.log_event("training", "Historical backtest winner analysis recorded", payload)

    @staticmethod
    def _normalize_simulated_trade(result) -> tuple[float, int, float]:
        if isinstance(result, tuple):
            if len(result) == 4:
                return float(result[0]), int(result[1]), float(result[2])
            if len(result) == 3:
                return float(result[0]), int(result[1]), float(result[2])
            if len(result) == 2:
                return float(result[0]), int(result[1]), 0.0
        raise ValueError("unexpected simulated trade result shape")

    @staticmethod
    def _normalize_simulated_trade_details(result) -> tuple[float, int, float, dict[str, object]]:
        if isinstance(result, tuple):
            if len(result) == 4:
                return float(result[0]), int(result[1]), float(result[2]), dict(result[3] or {})
            if len(result) == 3:
                return float(result[0]), int(result[1]), float(result[2]), {}
            if len(result) == 2:
                return float(result[0]), int(result[1]), 0.0, {}
        raise ValueError("unexpected simulated trade result shape")

    def _simulate_trade(self, df: pd.DataFrame, signal, signal_idx: int) -> tuple[float, int, float]:
        entry_idx = signal_idx + 1
        if entry_idx >= len(df):
            self._last_simulated_trade_details = {"exit_reason": "no_entry_bar"}
            return 0.0, signal_idx, 0.0
        point_size = self._point_size()
        entry_price = float(df.iloc[entry_idx]["open"])
        stop_points = max(1.0, float(getattr(signal, "stop_loss_points", 0.0) or 0.0))
        take_points = max(stop_points * 0.8, float(getattr(signal, "take_profit_points", 0.0) or 0.0))
        stop_distance = stop_points * point_size
        take_distance = take_points * point_size
        max_hold_bars = max(1, int(getattr(self.settings, "backtest_max_hold_bars", 12))) if self.settings is not None else 12
        exit_params = self._exit_params()
        best_mfe_r = 0.0

        if str(signal.side) == "buy":
            stop_price = entry_price - stop_distance
            take_price = entry_price + take_distance
        else:
            stop_price = entry_price + stop_distance
            take_price = entry_price - take_distance

        exit_idx = min(len(df) - 1, entry_idx + max_hold_bars - 1)
        exit_price = float(df.iloc[exit_idx]["close"])
        r_multiple = 0.0
        exit_reason = "max_hold"
        for j in range(entry_idx, min(len(df), entry_idx + max_hold_bars)):
            row = df.iloc[j]
            open_px = float(row["open"])
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            if j > entry_idx:
                if best_mfe_r >= float(exit_params["break_even_r_trigger"]):
                    lock_r = float(exit_params["break_even_lock_r"])
                    if str(signal.side) == "buy":
                        stop_price = max(stop_price, entry_price + (lock_r * stop_distance))
                    else:
                        stop_price = min(stop_price, entry_price - (lock_r * stop_distance))
                if best_mfe_r >= float(exit_params["trailing_r_trigger"]):
                    locked_r = max(
                        float(exit_params["break_even_lock_r"]),
                        best_mfe_r - float(exit_params["trailing_buffer_r"]),
                    )
                    if str(signal.side) == "buy":
                        stop_price = max(stop_price, entry_price + (locked_r * stop_distance))
                    else:
                        stop_price = min(stop_price, entry_price - (locked_r * stop_distance))
            if str(signal.side) == "buy":
                stop_r = (stop_price - entry_price) / max(stop_distance, 1e-9)
                take_r = (take_price - entry_price) / max(stop_distance, 1e-9)
                hit_stop = low <= stop_price
                hit_take = high >= take_price
                if hit_stop and hit_take:
                    bullish_bar = close >= open_px
                    exit_price = take_price if bullish_bar else stop_price
                    exit_idx = j
                    r_multiple = take_r if bullish_bar else stop_r
                    exit_reason = "take_profit" if bullish_bar else "stop_loss"
                    break
                if hit_stop:
                    exit_price = stop_price
                    exit_idx = j
                    r_multiple = stop_r
                    exit_reason = "stop_loss"
                    break
                if hit_take:
                    exit_price = take_price
                    exit_idx = j
                    r_multiple = take_r
                    exit_reason = "take_profit"
                    break
                if j == min(len(df), entry_idx + max_hold_bars) - 1:
                    exit_price = close
                    exit_idx = j
                favorable_r = max(0.0, (high - entry_price) / max(stop_distance, 1e-9))
            else:
                stop_r = (entry_price - stop_price) / max(stop_distance, 1e-9)
                take_r = (entry_price - take_price) / max(stop_distance, 1e-9)
                hit_stop = high >= stop_price
                hit_take = low <= take_price
                if hit_stop and hit_take:
                    bearish_bar = close <= open_px
                    exit_price = take_price if bearish_bar else stop_price
                    exit_idx = j
                    r_multiple = take_r if bearish_bar else stop_r
                    exit_reason = "take_profit" if bearish_bar else "stop_loss"
                    break
                if hit_stop:
                    exit_price = stop_price
                    exit_idx = j
                    r_multiple = stop_r
                    exit_reason = "stop_loss"
                    break
                if hit_take:
                    exit_price = take_price
                    exit_idx = j
                    r_multiple = take_r
                    exit_reason = "take_profit"
                    break
                if j == min(len(df), entry_idx + max_hold_bars) - 1:
                    exit_price = close
                    exit_idx = j
                favorable_r = max(0.0, (entry_price - low) / max(stop_distance, 1e-9))
            if j == exit_idx and abs(r_multiple) < 1e-12:
                move = (exit_price - entry_price) if str(signal.side) == "buy" else (entry_price - exit_price)
                r_multiple = move / max(stop_distance, 1e-9)
            best_mfe_r = max(best_mfe_r, favorable_r)
            bars_open = (j - entry_idx) + 1
            if abs(r_multiple) < 1e-12 and bars_open >= int(exit_params["stale_exit_bars"]) and best_mfe_r < float(exit_params["stale_min_progress_r"]):
                exit_price = close
                exit_idx = j
                move = (exit_price - entry_price) if str(signal.side) == "buy" else (entry_price - exit_price)
                r_multiple = move / max(stop_distance, 1e-9)
                exit_reason = "stale_exit"
                break

        if abs(r_multiple) < 1e-12:
            move = (exit_price - entry_price) if str(signal.side) == "buy" else (entry_price - exit_price)
            r_multiple = move / max(stop_distance, 1e-9)

        details = {
            "entry_idx": int(entry_idx),
            "exit_idx": int(exit_idx),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "stop_points": float(stop_points),
            "take_profit_points": float(take_points),
            "exit_reason": str(exit_reason),
        }
        self._last_simulated_trade_details = details
        return float(r_multiple), int(exit_idx), float(best_mfe_r)

    def _risk_of_ruin(self, average_win: float, average_loss: float, win_rate: float, trades: int) -> float:
        if trades <= 0 or average_win <= 0.0 or average_loss >= 0.0:
            return 1.0
        loss_size = abs(average_loss)
        payoff = average_win / max(loss_size, 1e-9)
        edge = (win_rate * payoff) - (1.0 - win_rate)
        if edge <= 0:
            return 1.0
        ruin = ((1.0 - edge) / max(1.0 + edge, 1e-9)) ** max(1, trades // 10)
        return max(0.0, min(1.0, ruin))

    @staticmethod
    def _closed_trade_counts(trade_returns: list[float]) -> tuple[int, int, int, float]:
        total_trades = len(trade_returns)
        wins = sum(1 for value in trade_returns if value > 0.0)
        losses = sum(1 for value in trade_returns if value < 0.0)
        classified = wins + losses
        win_rate = (wins / classified) if classified > 0 else 0.0
        return total_trades, wins, losses, win_rate

    def _summarize(self, equity_curve: list[float], trade_returns: list[float], holds: list[int], regime_stats: dict[str, dict[str, float]]) -> dict[str, float | dict[str, dict[str, float]]]:
        if not trade_returns:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "sortino": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "risk_of_ruin": 1.0,
                "average_hold_bars": 0.0,
                "regime_performance": regime_stats,
            }
        total_trades, closed_wins, closed_losses, win_rate = self._closed_trade_counts(trade_returns)
        wins = [x for x in trade_returns if x > 0]
        losses = [x for x in trade_returns if x < 0]
        average_win = sum(wins) / len(wins) if wins else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        expectancy = sum(trade_returns) / len(trade_returns)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / max(gross_loss, 1e-9) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)
        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            drawdown = (peak - value) / max(peak, 1e-9)
            max_drawdown = max(max_drawdown, drawdown)
        mean = expectancy
        variance = sum((x - mean) ** 2 for x in trade_returns) / max(1, len(trade_returns))
        downside = [min(0.0, x - mean) for x in trade_returns]
        downside_var = sum(x * x for x in downside) / max(1, len(downside))
        sharpe = mean / math.sqrt(max(variance, 1e-12))
        sortino = mean / math.sqrt(max(downside_var, 1e-12))
        risk_of_ruin = self._risk_of_ruin(average_win, average_loss, win_rate, total_trades)
        average_hold_bars = sum(holds) / max(1, len(holds))
        for bucket in regime_stats.values():
            trades = max(1.0, float(bucket.get("trades", 0.0)))
            bucket["avg_trade_return"] = round(float(bucket.get("sum_return", 0.0)) / trades, 6)
        return {
            "total_trades": total_trades,
            "wins": closed_wins,
            "losses": closed_losses,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "sortino": sortino,
            "average_win": average_win,
            "average_loss": average_loss,
            "risk_of_ruin": risk_of_ruin,
            "average_hold_bars": average_hold_bars,
            "regime_performance": regime_stats,
        }

    def _evolved_strategy(self, df: pd.DataFrame):
        calibration_bars = 1200
        generations = 4
        population = 8
        if self.settings is not None:
            calibration_bars = max(300, int(getattr(self.settings, "backtest_evolved_lookback_bars", 1200)))
            generations = min(4, max(1, int(getattr(self.settings, "evolved_strategy_generations", 4))))
            population = min(8, max(4, int(getattr(self.settings, "evolved_strategy_population", 8))))
        window = df.tail(min(len(df), calibration_bars))
        return self.lab.evolve(window, generations=generations, population=population)

    def run(self, df: pd.DataFrame) -> BacktestResult:
        initial_equity = 10000.0
        if self.settings is not None:
            initial_equity = max(1.0, float(getattr(self.settings, "backtest_initial_balance", 10000.0) or 10000.0))
        equity = float(initial_equity)
        evolved = None
        equity_curve = [equity]
        trade_returns: list[float] = []
        hold_bars: list[int] = []
        trade_list: list[dict[str, object]] = []
        regime_stats: dict[str, dict[str, float]] = {}
        learning_outcomes: list[dict[str, object]] = []
        historical_losses: list[dict[str, object]] = []
        historical_wins: list[dict[str, object]] = []
        strategy_stats: dict[str, dict[str, float]] = {}
        refresh_bars = 500
        if self.settings is not None:
            refresh_bars = max(0, int(getattr(self.settings, "backtest_evolved_refresh_bars", 500)))
        include_evolved = bool(getattr(self.settings, "backtest_include_evolved", False))

        i = 250
        while i < len(df) - 2:
            window = df.iloc[: i + 1]

            if include_evolved and (evolved is None or (refresh_bars and (i - 250) % refresh_bars == 0)):
                evolved = self._evolved_strategy(window)

            strategies = [s for s in self.base if self._strategy_enabled(getattr(s, "name", ""))]
            if include_evolved and evolved is not None:
                strategies.append(evolved)
            signals = [s.generate_signal(window) for s in strategies]
            best = self.allocator.choose(signals)
            min_conf = float(getattr(self.settings, "research_confidence_threshold", 0.1)) if self.settings is not None else 0.1
            if best.side == "flat" or best.confidence < min_conf:
                i += 1
                continue
            overlay_context = build_research_regime_overlay_context(
                window,
                window.iloc[-1]["time"] if "time" in window.columns else None,
                spread_points=0.0,
            )
            if self.settings is not None and research_regime_filter_reason(self.settings, overlay_context):
                i += 1
                continue
            if self.settings is not None and conservative_pretrade_filter_reason(self.settings, overlay_context):
                i += 1
                continue

            r_multiple, exit_idx, best_mfe_r = self._normalize_simulated_trade(self._simulate_trade(df, best, i))
            trade_details = dict(self._last_simulated_trade_details or {})
            risk_fraction = float(getattr(self.settings, "max_risk_per_trade", 0.005)) if self.settings is not None else 0.005
            trade_return = r_multiple * risk_fraction
            if self.settings is not None:
                # Keep transaction costs in the same unit as trade_return: account-return
                # fraction. The strategy PnL path is already normalized into R-multiples,
                # so subtracting notional-price bps inside _simulate_trade massively
                # over-penalizes trades on high-priced instruments like XAUUSD.
                spread_cost = float(getattr(self.settings, "backtest_spread_cost_bps", 0.0)) / 10000.0
                slippage_cost = float(getattr(self.settings, "backtest_slippage_cost_bps", 0.0)) / 10000.0
                trade_return -= (spread_cost + slippage_cost)
            equity *= max(0.0, 1.0 + trade_return)
            equity_curve.append(equity)
            trade_returns.append(trade_return)
            held_bars = max(1, int(exit_idx - i))
            hold_bars.append(held_bars)
            regime_name = self._regime_name(list(window["close"].tail(60)))
            bucket = regime_stats.setdefault(regime_name, {"trades": 0.0, "wins": 0.0, "losses": 0.0, "sum_return": 0.0})
            bucket["trades"] += 1.0
            bucket["sum_return"] += float(trade_return)
            self.allocator.update(best.strategy, trade_return)
            if trade_return > 0:
                bucket["wins"] += 1.0
            elif trade_return < 0:
                bucket["losses"] += 1.0
            entry_idx = int(trade_details.get("entry_idx", i + 1) or i + 1)
            entry_time = df.iloc[entry_idx]["time"] if "time" in df.columns and entry_idx < len(df) else None
            exit_time = df.iloc[exit_idx]["time"] if "time" in df.columns and exit_idx < len(df) else None
            trade_list.append(
                {
                    "trade_id": f"bt_{i}_{exit_idx}",
                    "signal_bar": int(i),
                    "entry_bar": int(entry_idx),
                    "exit_bar": int(exit_idx),
                    "entry_time": pd.Timestamp(entry_time).isoformat() if entry_time is not None else "",
                    "exit_time": pd.Timestamp(exit_time).isoformat() if exit_time is not None else "",
                    "strategy": str(best.strategy),
                    "side": str(best.side),
                    "confidence": round(float(getattr(best, "confidence", 0.0) or 0.0), 6),
                    "entry_price": round(float(trade_details.get("entry_price", 0.0) or 0.0), 6),
                    "exit_price": round(float(trade_details.get("exit_price", 0.0) or 0.0), 6),
                    "exit_reason": str(trade_details.get("exit_reason", "")),
                    "r_multiple": round(float(r_multiple), 6),
                    "return_fraction": round(float(trade_return), 10),
                    "hold_bars": int(held_bars),
                    "best_mfe_r": round(float(best_mfe_r), 6),
                    "regime": str(regime_name),
                }
            )
            if self.store is not None and bool(getattr(self.settings, "backtest_persist_learning_state", False)):
                close_time = (
                    pd.Timestamp(df.iloc[exit_idx]["time"]).to_pydatetime()
                    if "time" in df.columns
                    else pd.Timestamp.now("UTC").to_pydatetime()
                )
                if getattr(close_time, "tzinfo", None) is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                session_label = self._session_label(close_time)
                side = str(best.side)
                outcome = "win" if trade_return > 0 else ("loss" if trade_return < 0 else "draw")
                exit_reason = "take_profit" if trade_return > 0 else ("stop_loss" if trade_return < 0 else "other")
                duration_minutes = round(float(held_bars * int(getattr(self.settings, "timeframe_minutes", 1))), 2)
                setup_family = self._setup_family_from_strategy(str(best.strategy))
                loss_classification = self._classify_backtest_loss(
                    outcome=outcome,
                    exit_reason=exit_reason,
                    duration_minutes=duration_minutes,
                    session=session_label,
                    regime=regime_name,
                    best_mfe_r=float(best_mfe_r),
                    setup_family=setup_family,
                )
                outcome_row = {
                    "ticket": f"bt_{i}_{exit_idx}",
                    "position_id": f"bt_{i}",
                    "decision_id": f"bt_{i}_{best.strategy}",
                    "time": close_time.isoformat(),
                    "side": side,
                    "hour_utc": int(close_time.hour),
                    "net_pnl_zar": round(float(trade_return * initial_equity), 2),
                    "outcome": outcome,
                    "session": session_label,
                    "exit_reason": exit_reason,
                    "duration_minutes": duration_minutes,
                    "loss_classification": loss_classification,
                    "setup_family": setup_family,
                    "best_mfe_r": round(float(best_mfe_r), 4),
                    "post_trade_review": {
                        "classification": loss_classification,
                        "duration_minutes": duration_minutes,
                        "exit_reason": exit_reason,
                        "bounded_learning_only": True,
                        "validation_required": True,
                        "reversible": True,
                        "source": "historical_backtest",
                    },
                    "strategy": str(best.strategy),
                    "regime": regime_name,
                }
                learning_outcomes.append(outcome_row)
                stat_key = f"{str(best.strategy)}|{side}|{int(close_time.hour)}"
                stat_row = strategy_stats.setdefault(stat_key, {"wins": 0, "losses": 0, "pnl": 0.0})
                stat_row["pnl"] += float(trade_return * initial_equity)
                if trade_return > 0:
                    stat_row["wins"] += 1
                elif trade_return < 0:
                    stat_row["losses"] += 1
            elif self.store is not None and bool(getattr(self.settings, "backtest_record_loss_analysis", True)) and trade_return != 0:
                close_time = (
                    pd.Timestamp(df.iloc[exit_idx]["time"]).to_pydatetime()
                    if "time" in df.columns
                    else pd.Timestamp.now("UTC").to_pydatetime()
                )
                if getattr(close_time, "tzinfo", None) is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                session_label = self._session_label(close_time)
                side = str(best.side)
                outcome = "win" if trade_return > 0 else ("loss" if trade_return < 0 else "draw")
                exit_reason = "take_profit" if trade_return > 0 else ("stop_loss" if trade_return < 0 else "other")
                duration_minutes = round(float(held_bars * int(getattr(self.settings, "timeframe_minutes", 1))), 2)
                setup_family = self._setup_family_from_strategy(str(best.strategy))
                loss_classification = self._classify_backtest_loss(
                    outcome=outcome,
                    exit_reason=exit_reason,
                    duration_minutes=duration_minutes,
                    session=session_label,
                    regime=regime_name,
                    best_mfe_r=float(best_mfe_r),
                    setup_family=setup_family,
                )
                outcome_row = {
                    "ticket": f"bt_{i}_{exit_idx}",
                    "position_id": f"bt_{i}",
                    "decision_id": f"bt_{i}_{best.strategy}",
                    "time": close_time.isoformat(),
                    "side": side,
                    "hour_utc": int(close_time.hour),
                    "net_pnl_zar": round(float(trade_return * initial_equity), 2),
                    "outcome": outcome,
                    "session": session_label,
                    "exit_reason": exit_reason,
                    "duration_minutes": duration_minutes,
                    "loss_classification": loss_classification,
                    "setup_family": setup_family,
                    "best_mfe_r": round(float(best_mfe_r), 4),
                    "post_trade_review": {
                        "classification": loss_classification,
                        "duration_minutes": duration_minutes,
                        "exit_reason": exit_reason,
                        "bounded_learning_only": True,
                        "validation_required": True,
                        "reversible": True,
                        "source": "historical_backtest",
                    },
                    "strategy": str(best.strategy),
                    "regime": regime_name,
                }
                if trade_return > 0:
                    historical_wins.append(outcome_row)
                else:
                    historical_losses.append(outcome_row)
            i = max(i + 1, int(exit_idx))

        if self.store is not None and bool(getattr(self.settings, "backtest_persist_learning_state", False)):
            self._persist_backtest_outcomes(learning_outcomes, strategy_stats)

        stats = self._summarize(equity_curve, trade_returns, hold_bars, regime_stats)
        stats["total_return"] = (equity - initial_equity) / max(initial_equity, 1e-9)
        if self.store is not None:
            if historical_losses:
                self._persist_historical_loss_analysis(
                    losses=historical_losses,
                    result_summary=stats,
                    initial_equity=initial_equity,
                    df=df,
                )
            elif bool(getattr(self.settings, "backtest_persist_learning_state", False)):
                self._persist_historical_loss_analysis(
                    losses=[row for row in learning_outcomes if str(row.get("outcome", "")) == "loss"],
                    result_summary=stats,
                    initial_equity=initial_equity,
                    df=df,
                )
            if historical_wins:
                self._persist_historical_winner_analysis(
                    wins=historical_wins,
                    result_summary=stats,
                    initial_equity=initial_equity,
                    df=df,
                )
            elif bool(getattr(self.settings, "backtest_persist_learning_state", False)):
                self._persist_historical_winner_analysis(
                    wins=[row for row in learning_outcomes if str(row.get("outcome", "")) == "win"],
                    result_summary=stats,
                    initial_equity=initial_equity,
                    df=df,
                )
        return BacktestResult(
            total_return=(equity - initial_equity) / max(initial_equity, 1e-9),
            trades=int(stats["total_trades"]),
            wins=int(stats["wins"]),
            losses=int(stats["losses"]),
            win_rate=float(stats["win_rate"]),
            expectancy=float(stats["expectancy"]),
            profit_factor=float(stats["profit_factor"]),
            max_drawdown=float(stats["max_drawdown"]),
            sharpe=float(stats["sharpe"]),
            sortino=float(stats["sortino"]),
            average_win=float(stats["average_win"]),
            average_loss=float(stats["average_loss"]),
            risk_of_ruin=float(stats["risk_of_ruin"]),
            average_hold_bars=float(stats["average_hold_bars"]),
            regime_performance=dict(stats["regime_performance"] or {}),
            trade_list=trade_list,
        )
