from __future__ import annotations

"""Continuous market scanner for XAUUSD (scan-only, execution-gate routed)."""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from atlas_ai.config.settings import get_active_thresholds, load_settings
from atlas_ai.decision_framework import DecisionFramework, DecisionInput, DecisionOutput
from atlas_ai.execution.session_control import DayTradingSessionControl
from atlas_ai.volatility_intelligence import (
    VolatilityAssessment,
    VolatilityFeatures,
    VolatilityOutcomeRecord,
    classify_volatility,
    summarize_performance_by_volatility,
)

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    # purpose: store one full scanner cycle for observability.
    # what it does: captures reads, scores, blockers, and request state.
    timestamp: str
    session: str
    structure_read: str
    momentum_read: str
    location_read: str
    confirmation_read: str
    buy_score: float
    sell_score: float
    no_trade_score: float
    decision: str
    confidence: float
    blockers: list[str]
    reason: str
    can_create_execution_request: bool
    volatility_class: str = "normal_volatility"
    volatility_confidence: float = 0.0
    atr_ratio: float = 0.0
    spread_status: str = "unknown"
    news_window: bool = False
    response_rule: str = "normal_analysis"
    final_candidate: str = "NO_TRADE"
    tags: list[str] = None
    execution_request_created: bool = False
    forced_demo_entry: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class MarketScanner:
    # purpose: continuously scan market conditions without direct execution.
    # what it does: evaluates confluence and creates gate requests when eligible.
    SCAN_LOG_PATH = "logs/scan_results.jsonl"
    REJECTED_LOG_PATH = "logs/rejected_candidates.jsonl"

    def __init__(self, execution_gate, broker_adapter, settings):
        self.execution_gate = execution_gate
        self.broker = broker_adapter
        self.settings = settings
        self.framework = DecisionFramework()
        self._running = False
        self._thread = None
        self._last_decision = None
        self._last_decision_time = None
        self._cooldown_until = None
        self._rejected_debounce_until = None
        self._last_rejected_decision = None
        self._scan_count = 0
        self._session_trades: dict[str, int] = {}
        self._last_probe_reason = "none"
        self._last_probe_block_reason = ""
        self._last_probe_confidence_floor = 0.0
        self._last_probe_edge_gap = 0.0
        self._last_probe_used_at = ""
        self._last_probe_session = ""
        self._last_probe_decision = "NO_TRADE"
        self._last_volatility_recommendation_sample_size = 0
        self._latest_volatility: VolatilityAssessment | None = None
        self._latest_volatility_features: VolatilityFeatures | None = None
        self._volatility_outcomes: list[VolatilityOutcomeRecord] = []
        self._storage_dir = Path(str(getattr(settings, "storage_dir", "storage") or "storage"))
        self._probe_state_path = self._storage_dir / "scanner_probe_state.json"
        self._session_control = DayTradingSessionControl()
        os.makedirs("logs", exist_ok=True)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_probe_state()

    def _session_key(self) -> str:
        # purpose: key trade counts by UTC date/session bucket.
        # what it does: provides deterministic forced-entry session scope.
        now = datetime.now(timezone.utc)
        try:
            decision = self._session_control.check(runtime_flags={}, now=now, opening_new_trade=True)
            session = str(decision.current_session or "none")
        except Exception:
            session = "unknown"
        return f"{now.date()}_{session}"

    def _trades_this_session(self) -> int:
        # purpose: read current session trade count.
        # what it does: returns the number of scanner-requested trades this session.
        return int(self._session_trades.get(self._session_key(), 0) or 0)

    def _record_session_trade(self):
        # purpose: increment current session trade count.
        # what it does: tracks forced-entry eligibility.
        key = self._session_key()
        self._session_trades[key] = self._session_trades.get(key, 0) + 1
        self._save_probe_state()

    def _load_probe_state(self):
        # purpose: persist probe session usage across process restarts.
        # what it does: restores session trade counters and latest probe metadata.
        try:
            if not self._probe_state_path.exists():
                return
            raw = json.loads(self._probe_state_path.read_text(encoding="utf-8"))
            self._session_trades = {
                str(k): int(v) for k, v in dict(raw.get("session_trades", {}) or {}).items()
            }
            self._last_probe_reason = str(raw.get("last_probe_reason", "none") or "none")
            self._last_probe_block_reason = str(raw.get("last_probe_block_reason", "") or "")
            self._last_probe_confidence_floor = float(raw.get("last_probe_confidence_floor", 0.0) or 0.0)
            self._last_probe_edge_gap = float(raw.get("last_probe_edge_gap", 0.0) or 0.0)
            self._last_probe_used_at = str(raw.get("last_probe_used_at", "") or "")
            self._last_probe_session = str(raw.get("last_probe_session", "") or "")
            self._last_probe_decision = str(raw.get("last_probe_decision", "NO_TRADE") or "NO_TRADE")
            self._last_volatility_recommendation_sample_size = int(
                raw.get("last_volatility_recommendation_sample_size", 0) or 0
            )
        except Exception:
            return

    def _save_probe_state(self):
        # purpose: persist probe state to avoid duplicate probes after restarts.
        # what it does: writes scanner probe counters and metadata to storage.
        payload = {
            "session_trades": dict(self._session_trades or {}),
            "last_probe_reason": str(self._last_probe_reason or "none"),
            "last_probe_block_reason": str(self._last_probe_block_reason or ""),
            "last_probe_confidence_floor": float(self._last_probe_confidence_floor or 0.0),
            "last_probe_edge_gap": float(self._last_probe_edge_gap or 0.0),
            "last_probe_used_at": str(self._last_probe_used_at or ""),
            "last_probe_session": str(self._last_probe_session or ""),
            "last_probe_decision": str(self._last_probe_decision or "NO_TRADE"),
            "last_volatility_recommendation_sample_size": int(
                self._last_volatility_recommendation_sample_size or 0
            ),
        }
        try:
            self._probe_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return

    def start(self):
        # purpose: launch scanner loop in a daemon thread.
        # what it does: starts non-blocking background scanning.
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._scan_loop, name="AtlasMarketScanner", daemon=True)
        self._thread.start()

    def stop(self):
        # purpose: stop scanner loop gracefully.
        # what it does: flips running flag; loop exits naturally.
        self._running = False

    def _scan_loop(self):
        # purpose: run periodic scanner cycles.
        # what it does: executes one scan per configured interval.
        while self._running:
            interval = int(getattr(load_settings(), "scan_interval_seconds", 30))
            try:
                self._run_single_scan()
            except Exception as exc:
                logger.error("Scanner cycle error: %s", exc, exc_info=True)
            time.sleep(max(1, interval))

    def _run_single_scan(self):
        # purpose: perform one full scan/evaluate/log cycle.
        # what it does: builds input, evaluates, and optionally creates gate request.
        self._scan_count += 1
        now = datetime.now(timezone.utc)
        inp = self._build_decision_input(now)
        out = self.framework.evaluate(inp)
        settings = load_settings()
        vol = self._latest_volatility or VolatilityAssessment()
        vol_features = self._latest_volatility_features or VolatilityFeatures()
        result = ScanResult(
            timestamp=now.isoformat(),
            session=out.session,
            structure_read=out.structure_read,
            momentum_read=out.momentum_read,
            location_read=out.location_read,
            confirmation_read=out.confirmation_read,
            buy_score=out.buy_score,
            sell_score=out.sell_score,
            no_trade_score=out.no_trade_score,
            decision=out.trade_decision,
            confidence=out.confidence,
            blockers=out.blockers,
            reason=out.reason,
            can_create_execution_request=out.can_create_execution_request,
            volatility_class=str(vol.volatility_class),
            volatility_confidence=float(vol.volatility_confidence),
            atr_ratio=float(vol_features.atr_ratio),
            spread_status=str(vol_features.spread_status),
            news_window=bool(vol_features.news_window_flag),
            response_rule=str(vol.response_rule),
            final_candidate=str(out.trade_decision),
            tags=list(vol.tags or []),
        )
        if not inp.session_tradeable or self._in_cooldown(now) or self._is_duplicate(out, now) or self._in_rejected_debounce(out, now):
            self._log_scan_result(result)
            return
        closed_trade_count = int((self._read_status() or {}).get("closed_trades_count", 0) or 0)
        if self._should_apply_adaptive_probe(settings=settings, out=out, inp=inp, vol=vol, closed_trade_count=closed_trade_count):
            logger.warning(
                "LEARNING MODE ADAPTIVE PROBE ENTRY: session=%s confidence=%.3f decision_was=%s trades_this_session=0",
                inp.session,
                float(out.confidence),
                out.trade_decision,
            )
            out.can_create_execution_request = True
            if out.trade_decision == "NO_TRADE":
                out.trade_decision = "BUY_CANDIDATE" if float(out.buy_score) >= float(out.sell_score) else "SELL_CANDIDATE"
            result.forced_demo_entry = True
            result.final_candidate = str(out.trade_decision)
            result.tags = list(set(list(result.tags or []) + ["adaptive_probe_entry"]))
            self._last_probe_used_at = now.isoformat()
            self._last_probe_session = self._session_key()
            self._last_probe_decision = str(out.trade_decision)
            self._record_session_trade()
            self._create_execution_request(out, inp, vol=vol)
            result.execution_request_created = True
            self._start_cooldown(now)
            self._last_decision = out.trade_decision
            self._last_decision_time = now
            self._log_scan_result(result)
            self._publish_volatility_state(status=self._read_status(), vol=vol)
            self._publish_learning_status(settings)
            return
        if out.trade_decision == "NO_TRADE":
            self._log_rejected_candidate(out)
            self._log_scan_result(result)
            return
        if out.can_create_execution_request:
            self._create_execution_request(out, inp, vol=vol)
            result.execution_request_created = True
            self._start_cooldown(now)
            self._last_decision = out.trade_decision
            self._last_decision_time = now
            self._record_session_trade()
            self._last_probe_reason = "normal_candidate"
            self._last_probe_block_reason = ""
            self._last_probe_used_at = now.isoformat()
            self._last_probe_session = self._session_key()
            self._last_probe_decision = str(out.trade_decision)
            self._save_probe_state()
        else:
            self._log_rejected_candidate(out)
            self._start_rejected_debounce(now, out)
        self._log_scan_result(result)
        self._publish_learning_status(settings)
        self._publish_volatility_state(status=self._read_status(), vol=vol)

    def _build_decision_input(self, now: datetime) -> DecisionInput:
        # purpose: map runtime/market data into DecisionInput.
        # what it does: reads MT5 + status with safe defaults.
        inp = DecisionInput(timestamp=now)
        try:
            import MetaTrader5 as mt5
            session_decision = self._session_control.check(runtime_flags={}, now=now, opening_new_trade=True)
            inp.session = str(session_decision.current_session or "none")
            inp.session_tradeable = bool(session_decision.allowed)
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick:
                inp.spread_pips = round((tick.ask - tick.bid) / 0.1, 2)
            bars_h1 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_H1, 0, 60)
            bars_m15 = mt5.copy_rates_from_pos("XAUUSD", mt5.TIMEFRAME_M15, 0, 30)
            if bars_h1 is not None and len(bars_h1) >= 20:
                inp = self._analyse_structure(inp, bars_h1)
                inp = self._analyse_location(inp, bars_h1)
                inp.adx = self._calculate_adx(bars_h1)
            if bars_m15 is not None and len(bars_m15) >= 6:
                inp = self._analyse_momentum(inp, bars_m15)
            status = self._read_status()
            inp.dxy_direction = str(status.get("dxy_direction", "flat"))
            inp.us10y_direction = str(status.get("us10y_direction", "flat"))
            inp.regime = str(status.get("regime", "unknown"))
            inp.in_news_window = bool(status.get("in_news_window", False))
            inp.open_trades_count = int(status.get("open_trades_count", 0) or 0)
            inp.daily_trades_count = int(status.get("trades_today", 0) or 0)
            inp.daily_drawdown_pct = float(status.get("daily_drawdown_pct", 0.0) or 0.0)
            features = self._build_volatility_features(
                now=now,
                status=status,
                spread_pips=float(inp.spread_pips),
                bars_h1=bars_h1,
                bars_m15=bars_m15,
            )
            vol = classify_volatility(features)
            self._latest_volatility = vol
            self._latest_volatility_features = features
            inp.atr_pips = float(features.atr_14)
            inp.volatility_class = str(vol.volatility_class)
            inp.volatility_confidence = float(vol.volatility_confidence)
            inp.volatility_response_rule = str(vol.response_rule)
            inp.volatility_tags = list(vol.tags or [])
            inp.high_impact_news_hard_block = bool(features.high_impact_news_hard_block)
            inp.spread_status = str(features.spread_status)
            if vol.blockers:
                # Mirror protective volatility blockers in decision input.
                inp.in_news_window = bool(inp.in_news_window or ("high_impact_news_window" in vol.blockers))
        except Exception as exc:
            logger.error("DecisionInput build error: %s", exc, exc_info=True)
        return inp

    def _create_execution_request(self, out: DecisionOutput, inp: DecisionInput, *, vol: VolatilityAssessment | None = None):
        # purpose: send candidate to ExecutionGate as a request only.
        # what it does: creates normalized request payload without broker execution.
        direction = "BUY" if out.trade_decision == "BUY_CANDIDATE" else "SELL"
        sl, tp = self._calculate_sl_tp(price=0.0, direction=direction, atr=float(inp.atr_pips or 15.0), settings=self.settings)
        environment = "demo" if bool(getattr(self.settings, "execute_trades", False)) else "paper"
        vol_class = str((vol.volatility_class if vol else inp.volatility_class) or "normal_volatility")
        vol_rule = str((vol.response_rule if vol else inp.volatility_response_rule) or "normal_analysis")
        vol_conf = float(vol.volatility_confidence if vol else inp.volatility_confidence)
        payload_summary = (
            f"instrument=XAUUSD setup=scanner size=0.01 data_source=broker_runtime environment_mode={environment} "
            f"direction={direction} confidence={float(out.confidence):.3f} session={inp.session} "
            f"volatility_class={vol_class} volatility_confidence={vol_conf:.3f} volatility_rule={vol_rule} "
            f"news_window={bool(inp.in_news_window)} spread_status={str(inp.spread_status or 'unknown')} "
            f"sl={sl:.2f} tp={tp:.2f}"
        )
        self.execution_gate.create_request(
            environment=environment,
            action_type="place_trade",
            payload_summary=payload_summary,
            requested_by="market_scanner",
            dry_run=not bool(getattr(self.settings, "execute_trades", False)),
            auto_generated=True,
            request_origin="scanner",
        )

    def _calculate_sl_tp(self, price: float, direction: str, atr: float, settings) -> tuple[float, float]:
        # purpose: compute SL/TP distances using active mode profile.
        # what it does: applies wider LEARNING-mode SL and modest TP ratio.
        thresholds = get_active_thresholds(settings)
        sl_pips = max(15.0, float(atr or 0.0) * float(thresholds["sl_atr_mult"]))
        tp_pips = sl_pips * float(thresholds["tp_rr"])
        if str(direction).upper() == "BUY":
            sl = price - sl_pips * 0.1
            tp = price + tp_pips * 0.1
        else:
            sl = price + sl_pips * 0.1
            tp = price - tp_pips * 0.1
        if abs(float(price or 0.0)) <= 1e-9:
            logger.info(
                "SL/TP distance calc (%s): SL=%0.1fpips TP=%0.1fpips ATR=%.2f mult=%.2f",
                str(thresholds["mode_label"]),
                sl_pips,
                tp_pips,
                float(atr or 0.0),
                float(thresholds["sl_atr_mult"]),
            )
        else:
            logger.info(
                "SL/TP calc (%s): SL=%.2f (%0.1fpips) TP=%.2f (%0.1fpips) ATR=%.2f mult=%.2f",
                str(thresholds["mode_label"]),
                sl,
                sl_pips,
                tp,
                tp_pips,
                float(atr or 0.0),
                float(thresholds["sl_atr_mult"]),
            )
        return sl, tp

    def _should_force_entry(self, settings, closed_trade_count: int) -> bool:
        # purpose: decide whether one forced learning entry is required this session.
        # what it does: allows one forced entry per session until graduation trade count.
        if not bool(getattr(settings, "learning_mode", False)):
            return False
        if not bool(getattr(settings, "learning_force_entry_per_session", True)):
            return False
        if int(closed_trade_count) >= int(getattr(settings, "learning_graduation_min_trades", 100)):
            return False
        return self._trades_this_session() == 0

    def _should_apply_adaptive_probe(
        self,
        *,
        settings,
        out: DecisionOutput,
        inp: DecisionInput,
        vol: VolatilityAssessment,
        closed_trade_count: int,
    ) -> bool:
        # purpose: allow one safe probe trade per session in learning mode.
        # what it does: expands probe eligibility beyond strict NO_TRADE while preserving hard blocks.
        self._last_probe_reason = "none"
        self._last_probe_block_reason = ""
        self._last_probe_edge_gap = abs(float(out.buy_score or 0.0) - float(out.sell_score or 0.0))
        if not self._should_force_entry(settings, closed_trade_count):
            self._last_probe_block_reason = "session_probe_already_used_or_learning_disabled"
            return False
        if out.blockers:
            self._last_probe_block_reason = "decision_blockers_active"
            return False
        if bool(inp.in_news_window) or bool(inp.high_impact_news_hard_block):
            self._last_probe_block_reason = "news_window_active"
            return False
        if str(vol.response_rule) in {"volatility_hard_block", "news_hard_block", "spread_slippage_block"}:
            self._last_probe_block_reason = f"volatility_rule_{str(vol.response_rule)}"
            return False
        if str(inp.spread_status).strip().lower() == "blocked":
            self._last_probe_block_reason = "spread_status_blocked"
            return False
        min_probe_conf = self._effective_probe_confidence_min(settings=settings, vol=vol)
        self._last_probe_confidence_floor = float(min_probe_conf)
        if float(out.confidence) < min_probe_conf:
            self._last_probe_block_reason = "probe_confidence_below_dynamic_floor"
            return False
        min_edge_gap = float(getattr(settings, "learning_probe_min_edge_gap", 0.07))
        if self._last_probe_edge_gap < min_edge_gap:
            self._last_probe_block_reason = "probe_edge_gap_too_small"
            return False
        if out.trade_decision in {"BUY_CANDIDATE", "SELL_CANDIDATE"} and not out.can_create_execution_request:
            self._last_probe_reason = "candidate_below_threshold_probe"
            self._save_probe_state()
            return True
        allowed = out.trade_decision == "NO_TRADE"
        if allowed:
            self._last_probe_reason = "no_trade_probe_override"
            self._save_probe_state()
        else:
            self._last_probe_block_reason = "probe_not_required_for_current_decision"
        return allowed

    def _effective_probe_confidence_min(self, *, settings, vol: VolatilityAssessment) -> float:
        # purpose: make probe confidence floor volatility-aware.
        # what it does: increases minimum confidence in noisier/riskier regimes.
        base = float(getattr(settings, "learning_force_min_confidence", 0.40))
        klass = str(vol.volatility_class or "normal_volatility")
        bump = 0.0
        if klass in {"elevated_volatility", "reversal_exhaustion_volatility", "trend_expansion_volatility"}:
            bump += float(getattr(settings, "learning_probe_confidence_bump_elevated", 0.04))
        if klass in {"chop_volatility", "session_transition_volatility"}:
            bump += float(getattr(settings, "learning_probe_confidence_bump_chop", 0.06))
        if klass == "session_transition_volatility":
            bump += float(getattr(settings, "learning_probe_confidence_bump_transition", 0.03))
        return max(base, min(0.80, base + bump))

    def _analyse_structure(self, inp: DecisionInput, bars) -> DecisionInput:
        # purpose: derive structure flags for decision framework.
        # what it does: compares recent H1 swing highs/lows.
        highs = [float(b["high"]) for b in bars[-10:]]
        lows = [float(b["low"]) for b in bars[-10:]]
        recent_high, prev_high = max(highs[-5:]), max(highs[-10:-5])
        recent_low, prev_low = min(lows[-5:]), min(lows[-10:-5])
        inp.higher_highs = recent_high > prev_high
        inp.higher_lows = recent_low > prev_low
        inp.lower_highs = recent_high < prev_high
        inp.lower_lows = recent_low < prev_low
        return inp

    def _analyse_momentum(self, inp: DecisionInput, bars) -> DecisionInput:
        # purpose: quantify short-horizon candle momentum.
        # what it does: converts latest candle body-to-range ratio into strength.
        last = bars[-1]
        body = abs(float(last["close"]) - float(last["open"]))
        total = max(1e-9, float(last["high"]) - float(last["low"]))
        ratio = max(0.0, min(1.0, body / total))
        if float(last["close"]) > float(last["open"]):
            inp.bullish_candle_strength = ratio
            inp.momentum_direction = "bullish"
        elif float(last["close"]) < float(last["open"]):
            inp.bearish_candle_strength = ratio
            inp.momentum_direction = "bearish"
        else:
            inp.momentum_direction = "neutral"
        return inp

    def _analyse_location(self, inp: DecisionInput, bars) -> DecisionInput:
        # purpose: detect support/resistance location context.
        # what it does: uses 20-bar percentile position in range.
        window = bars[-20:]
        current = float(window[-1]["close"])
        high_20 = max(float(b["high"]) for b in window)
        low_20 = min(float(b["low"]) for b in window)
        span = max(1e-9, high_20 - low_20)
        position = (current - low_20) / span
        inp.near_support = position <= 0.25
        inp.near_resistance = position >= 0.75
        inp.in_middle_of_range = 0.35 <= position <= 0.65
        return inp

    def _calculate_adx(self, bars) -> float:
        # purpose: approximate trend strength without external indicator dependency.
        # what it does: returns normalized true-range proxy.
        try:
            closes = [float(b["close"]) for b in bars[-15:]]
            highs = [float(b["high"]) for b in bars[-15:]]
            lows = [float(b["low"]) for b in bars[-15:]]
            trs = []
            for i in range(1, len(closes)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                trs.append(tr)
            return float((sum(trs[-14:]) / max(1, len(trs[-14:]))) / max(1e-9, closes[-1]) * 1000.0)
        except Exception:
            return 0.0

    def _build_volatility_features(self, *, now: datetime, status: dict, spread_pips: float, bars_h1, bars_m15) -> VolatilityFeatures:
        # purpose: compute volatility taxonomy features each scan cycle.
        # what it does: derives ATR/range/wick/spread/session/news/macro feature set.
        session = str(status.get("session", "") or "").strip().lower()
        # Normalize missing session labels so scanner always derives a usable session.
        if session in {"", "unknown", "unavailable", "-"}:
            h = now.hour
            if 13 <= h < 17:
                session = "overlap"
            elif 7 <= h < 12:
                session = "london"
            elif 12 <= h < 20:
                session = "new_york"
            elif 0 <= h < 7:
                session = "asian"
            else:
                session = "off_hours"
        transition = now.minute < 20 and now.hour in {7, 12, 13, 21}
        atr_14 = 0.0
        atr_baseline = 0.0
        candle_body_ratio = 0.0
        wick_ratio = 0.0
        average_range = 0.0
        range_expansion_ratio = 1.0
        structure_state = "range_or_mixed"
        momentum_state = "mixed"
        if bars_h1 is not None and len(bars_h1) >= 20:
            ranges = [max(0.0, float(b["high"]) - float(b["low"])) for b in bars_h1[-30:]]
            average_range = float(sum(ranges) / max(1, len(ranges)))
            atr_14 = float(sum(ranges[-14:]) / max(1, len(ranges[-14:])))
            atr_baseline = float(sum(ranges[:-14]) / max(1, len(ranges[:-14]) or 1)) if len(ranges) > 14 else average_range
            atr_baseline = max(1e-9, atr_baseline)
            range_expansion_ratio = float(ranges[-1] / max(1e-9, average_range))
            recent_high = max(float(b["high"]) for b in bars_h1[-5:])
            prev_high = max(float(b["high"]) for b in bars_h1[-10:-5])
            recent_low = min(float(b["low"]) for b in bars_h1[-5:])
            prev_low = min(float(b["low"]) for b in bars_h1[-10:-5])
            # Build a more stable structure read so "trend unclear" appears only
            # when price action is genuinely mixed.
            last_close = float(bars_h1[-1]["close"])
            prev_close = float(bars_h1[-6]["close"])
            close_drift = (last_close - prev_close) / max(1e-9, abs(prev_close))
            if recent_high > prev_high and recent_low > prev_low and close_drift > 0.0006:
                structure_state = "trend_up"
            elif recent_high < prev_high and recent_low < prev_low and close_drift < -0.0006:
                structure_state = "trend_down"
            else:
                # If highs/lows are mixed but drift is directional, keep a soft trend label.
                if close_drift > 0.0012:
                    structure_state = "soft_trend_up"
                elif close_drift < -0.0012:
                    structure_state = "soft_trend_down"
                else:
                    structure_state = "range_or_mixed"
        if bars_m15 is not None and len(bars_m15) >= 4:
            b = bars_m15[-1]
            high = float(b["high"])
            low = float(b["low"])
            opn = float(b["open"])
            close = float(b["close"])
            total = max(1e-9, high - low)
            body = abs(close - opn)
            upper_wick = max(0.0, high - max(opn, close))
            lower_wick = max(0.0, min(opn, close) - low)
            candle_body_ratio = body / total
            wick_ratio = (upper_wick + lower_wick) / total
            if close > opn and candle_body_ratio >= 0.55:
                momentum_state = "bullish"
            elif close < opn and candle_body_ratio >= 0.55:
                momentum_state = "bearish"
            else:
                momentum_state = "mixed"
        spread_status = "normal"
        if spread_pips >= 6.0:
            spread_status = "blocked"
        elif spread_pips >= 4.5:
            spread_status = "high"
        elif spread_pips >= 3.2:
            spread_status = "elevated"
        macro_available = any(str(status.get(k, "") or "").lower() not in {"", "unknown", "unavailable"} for k in ("dxy_direction", "us10y_direction", "vix_state"))
        news_window = bool(status.get("in_news_window", False))
        return VolatilityFeatures(
            atr_current=float(atr_14),
            atr_14=float(atr_14),
            atr_baseline=float(atr_baseline),
            atr_ratio=float(atr_14 / max(1e-9, atr_baseline)) if atr_baseline > 0 else 0.0,
            candle_body_ratio=float(candle_body_ratio),
            wick_ratio=float(wick_ratio),
            average_range=float(average_range),
            range_expansion_ratio=float(range_expansion_ratio),
            spread_points=float(spread_pips * 10.0),
            spread_pips=float(spread_pips),
            spread_status=spread_status,
            session=session,
            session_transition_flag=bool(transition),
            news_window_flag=bool(news_window),
            high_impact_news_hard_block=bool(news_window),
            dxy_alignment_state=str(status.get("dxy_alignment_state", "unavailable")),
            us10y_alignment_state=str(status.get("us10y_alignment_state", "unavailable")),
            vix_state=str(status.get("vix_state", "unavailable")),
            price_structure_state=structure_state,
            momentum_state=momentum_state,
            macro_available=bool(macro_available),
        )

    def _publish_learning_status(self, settings):
        # purpose: expose learning mode and graduation progress in status payload.
        # what it does: writes lightweight status file fields used by dashboard.
        status = self._read_status()
        if not status:
            return
        thresholds = get_active_thresholds(settings)
        closed = int(status.get("closed_trades_count", 0) or 0)
        wins = int(status.get("wins_today", 0) or 0)
        avg_rr = float(status.get("avg_rr", 0.0) or 0.0)
        win_rate = float(wins / max(1, closed)) if closed > 0 else 0.0
        status["learning_mode"] = bool(getattr(settings, "learning_mode", False))
        status["mode_label"] = str(thresholds["mode_label"])
        status["aggressive_demo_profile_active"] = bool(
            getattr(settings, "learning_mode", False) and getattr(settings, "demo_mode", False)
        )
        status["thresholds"] = {
            "confidence": float(thresholds["confidence"]),
            "structure": float(thresholds["structure"]),
            "spread_max": float(thresholds["spread_max"]),
        }
        status["graduation_progress"] = {
            "trades_completed": closed,
            "trades_needed": int(getattr(settings, "learning_graduation_min_trades", 100)),
            "win_rate": win_rate,
            "win_rate_needed": float(getattr(settings, "learning_graduation_min_winrate", 0.52)),
            "avg_rr": avg_rr,
            "avg_rr_needed": float(getattr(settings, "learning_graduation_min_rr", 1.2)),
        }
        next_probe_eligible_at = ""
        if bool(self._cooldown_until):
            try:
                next_probe_eligible_at = datetime.fromtimestamp(float(self._cooldown_until), tz=timezone.utc).isoformat()
            except Exception:
                next_probe_eligible_at = ""
        status["probe_mode"] = {
            "probe_used_this_session": bool(self._trades_this_session() > 0),
            "probe_reason": str(self._last_probe_reason or "none"),
            "probe_block_reason": str(self._last_probe_block_reason or ""),
            "probe_confidence_floor": round(float(self._last_probe_confidence_floor or 0.0), 4),
            "probe_quality_edge_gap": round(float(self._last_probe_edge_gap or 0.0), 4),
            "next_probe_eligible_at": next_probe_eligible_at,
            "last_probe_used_at": str(self._last_probe_used_at or ""),
            "last_probe_session": str(self._last_probe_session or ""),
            "last_probe_decision": str(self._last_probe_decision or "NO_TRADE"),
        }
        for path in ["status.json", "atlas_ai/status.json"]:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2, default=str)
                break
            except Exception:
                continue

    def _publish_volatility_state(self, *, status: dict, vol: VolatilityAssessment):
        # purpose: expose volatility intelligence to dashboard/API/Ask Atlas.
        # what it does: persists current class/confidence/response/tags and learning summary.
        if not status:
            return
        status["volatility_intelligence"] = {
            "class": str(vol.volatility_class),
            "confidence": float(vol.volatility_confidence),
            "response_rule": str(vol.response_rule),
            "tags": list(vol.tags or []),
            "blockers": list(vol.blockers or []),
            "explanation": str(vol.explanation or ""),
            "features": dict(vol.feature_snapshot or {}),
        }
        status["volatility_state"] = "extreme" if str(vol.volatility_class) in {"extreme_volatility", "news_spike_volatility"} else (
            "elevated" if str(vol.volatility_class) in {"elevated_volatility", "liquidity_gap_volatility"} else "normal"
        )
        status["volatility_action"] = "block_new_entries" if vol.response_rule in {"volatility_hard_block", "news_hard_block", "spread_slippage_block"} else "allow"
        status["volatility_learning_performance"] = summarize_performance_by_volatility(self._volatility_outcomes) if self._volatility_outcomes else {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_size": 0,
            "classes": {},
            "recommendations": [],
        }
        sample_size = int(status["volatility_learning_performance"].get("sample_size", 0) or 0)
        # purpose: checkpoint recommendations every 50 completed demo outcomes.
        # what it does: keeps recommendations stable and cadence-based, no auto-policy mutation.
        if sample_size >= 50 and sample_size % 50 == 0 and sample_size != int(self._last_volatility_recommendation_sample_size or 0):
            status["volatility_learning_performance"]["recommendation_checkpoint"] = {
                "sample_size": sample_size,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "note": "recommendations refreshed at 50-trade checkpoint; operator review required",
            }
            self._last_volatility_recommendation_sample_size = sample_size
            self._save_probe_state()
        for path in ["status.json", "atlas_ai/status.json"]:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2, default=str)
                break
            except Exception:
                continue

    def _is_duplicate(self, out: DecisionOutput, now: datetime) -> bool:
        # purpose: suppress duplicate rapid request-created signals.
        # what it does: checks last created request decision within configured duplicate window.
        if not self._last_decision or not self._last_decision_time:
            return False
        window = int(getattr(self.settings, "duplicate_signal_window_seconds", 60))
        return self._last_decision == out.trade_decision and (now - self._last_decision_time).total_seconds() < window

    def _in_cooldown(self, now: datetime) -> bool:
        # purpose: throttle request frequency.
        # what it does: blocks scans from creating requests until cooldown expiry.
        return bool(self._cooldown_until and now.timestamp() < self._cooldown_until)

    def _in_rejected_debounce(self, out: DecisionOutput, now: datetime) -> bool:
        # purpose: avoid spam from identical non-executable candidates only.
        # what it does: never blocks a candidate that is currently execution-eligible.
        if bool(getattr(out, "can_create_execution_request", False)):
            return False
        return bool(
            self._rejected_debounce_until
            and now.timestamp() < self._rejected_debounce_until
            and self._last_rejected_decision == out.trade_decision
        )

    def _start_cooldown(self, now: datetime):
        # purpose: start cooldown timer.
        # what it does: sets cooldown deadline using configured seconds.
        self._cooldown_until = now.timestamp() + int(getattr(self.settings, "scanner_cooldown_seconds", 300))

    def _start_rejected_debounce(self, now: datetime, out: DecisionOutput):
        # purpose: briefly debounce repeated rejected candidates without blocking future requests.
        # what it does: uses a short same-decision window instead of the full request cooldown.
        seconds = int(getattr(self.settings, "rejected_candidate_debounce_seconds", 5))
        self._last_rejected_decision = out.trade_decision
        self._rejected_debounce_until = now.timestamp() + max(0, seconds)

    def _log_scan_result(self, result: ScanResult):
        # purpose: append scanner cycle output for traceability.
        # what it does: writes one json line to scan results log.
        try:
            path = Path(self.SCAN_LOG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result), default=str) + "\n")
        except OSError as exc:
            logger.warning("Scanner scan-result log write skipped: %s", exc)

    def _log_rejected_candidate(self, out: DecisionOutput):
        # purpose: record rejected candidates for later learning analysis.
        # what it does: appends rejected signal payload to dedicated jsonl log.
        vol = self._latest_volatility or VolatilityAssessment()
        vol_features = self._latest_volatility_features or VolatilityFeatures()
        payload = {
            "timestamp": out.timestamp.isoformat(),
            "decision": out.trade_decision,
            "confidence": out.confidence,
            "blockers": out.blockers,
            "rejection_reason": out.reason,
            "volatility_class_at_entry": str(vol.volatility_class),
            "atr_ratio_at_entry": float(vol_features.atr_ratio),
            "spread_status_at_entry": str(vol_features.spread_status),
            "session": str(out.session),
            "response_rule": str(vol.response_rule),
        }
        with open(self.REJECTED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def record_volatility_outcome(self, payload: dict) -> None:
        # purpose: allow trader/learning layers to feed completed outcome samples.
        # what it does: stores normalized volatility-at-entry record for analysis.
        try:
            rec = VolatilityOutcomeRecord(
                timestamp=str(payload.get("timestamp", datetime.now(timezone.utc).isoformat())),
                volatility_class_at_entry=str(payload.get("volatility_class_at_entry", "normal_volatility")),
                atr_ratio_at_entry=float(payload.get("atr_ratio_at_entry", 0.0) or 0.0),
                spread_status_at_entry=str(payload.get("spread_status_at_entry", "unknown")),
                session=str(payload.get("session", "unknown")),
                outcome=str(payload.get("outcome", "unknown")),
                r_multiple=float(payload.get("r_multiple", 0.0) or 0.0),
                mfe=float(payload.get("mfe", 0.0) or 0.0),
                mae=float(payload.get("mae", 0.0) or 0.0),
                response_was_correct=bool(payload.get("response_was_correct", False)),
                blocker_prevented_loss=bool(payload.get("blocker_prevented_loss", False)),
                blocker_missed_profit=bool(payload.get("blocker_missed_profit", False)),
            )
            self._volatility_outcomes.append(rec)
            if len(self._volatility_outcomes) > 500:
                self._volatility_outcomes = self._volatility_outcomes[-500:]
        except Exception:
            return

    def _read_status(self) -> dict:
        # purpose: read latest status from common status file locations.
        # what it does: returns parsed json or empty dict on failure.
        for path in ["status.json", "atlas_ai/status.json", "../status.json"]:
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
        return {}


def check_learning_graduation(trade_log: list[dict], settings) -> dict:
    # purpose: evaluate whether learning-mode graduation criteria are met.
    # what it does: reports recommendation only; does not auto-toggle live behavior.
    if not bool(getattr(settings, "learning_mode", False)):
        return {"graduated": False, "reason": "already past learning mode"}
    closed = [t for t in list(trade_log or []) if str(dict(t).get("status", "")).lower() == "closed"]
    total = len(closed)
    min_trades = int(getattr(settings, "learning_graduation_min_trades", 100))
    if total < min_trades:
        remaining = min_trades - total
        logger.info("Graduation check: %s/%s trades -- %s more needed", total, min_trades, remaining)
        return {"graduated": False, "trades_remaining": remaining}
    wins = len([t for t in closed if float(dict(t).get("pnl", 0.0) or 0.0) > 0.0])
    win_rate = (wins / total) if total > 0 else 0.0
    avg_rr = (
        sum(float(dict(t).get("rr_achieved", 0.0) or 0.0) for t in closed) / total
        if total > 0
        else 0.0
    )
    need_wr = float(getattr(settings, "learning_graduation_min_winrate", 0.52))
    need_rr = float(getattr(settings, "learning_graduation_min_rr", 1.2))
    meets_winrate = win_rate >= need_wr
    meets_rr = avg_rr >= need_rr
    logger.info(
        "Graduation check: %s trades | Win rate: %.1f%% (need %.1f%%) %s | Avg R:R: %.2f (need %.2f) %s",
        total,
        win_rate * 100.0,
        need_wr * 100.0,
        "PASS" if meets_winrate else "FAIL",
        avg_rr,
        need_rr,
        "PASS" if meets_rr else "FAIL",
    )
    if meets_winrate and meets_rr:
        logger.warning(
            "GRADUATION THRESHOLD MET: ready to move from LEARNING MODE to DEMO MODE. Set learning_mode=False after review."
        )
        try:
            from atlas_ai.voice.output import SpeechOutput
            SpeechOutput(enabled=True).speak(
                "Atlas completed learning mode. Review trade log and set learning mode to false when ready."
            )
        except Exception:
            pass
        return {
            "graduated": True,
            "total_trades": total,
            "win_rate": win_rate,
            "avg_rr": avg_rr,
            "action_required": "Set learning_mode=False in settings to graduate",
        }
    return {
        "graduated": False,
        "total_trades": total,
        "win_rate": win_rate,
        "avg_rr": avg_rr,
        "what_is_failing": ([] + (["win_rate"] if not meets_winrate else []) + (["avg_rr"] if not meets_rr else [])),
    }
