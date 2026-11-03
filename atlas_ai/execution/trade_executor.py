from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

import MetaTrader5 as mt5

from atlas_ai.config.settings import Settings
from atlas_ai.execution.comment_codec import build_broker_comment
from atlas_ai.models import TradeSignal


@dataclass
class ExecutionResult:
    success: bool
    retcode: int
    comment: str
    order_id: int = 0
    deal_id: int = 0


class TradeExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _stop_buffer_points(symbol_info, spread_points: float = 0.0, extra_points: int = 0) -> int:
        stops_level = int(getattr(symbol_info, "trade_stops_level", 0) or 0)
        freeze_level = int(getattr(symbol_info, "trade_freeze_level", 0) or 0)
        spread_buffer = int(max(2, round(float(spread_points or 0.0) * 0.35)))
        return max(stops_level, freeze_level, 0) + spread_buffer + int(max(0, extra_points))

    @staticmethod
    def _normalize_volume(volume: float, symbol_info) -> float:
        vol = float(volume or 0.0)
        vol_min = float(getattr(symbol_info, "volume_min", 0.0) or 0.0)
        vol_max = float(getattr(symbol_info, "volume_max", 0.0) or 0.0)
        vol_step = float(getattr(symbol_info, "volume_step", 0.0) or 0.0)

        if vol_step > 0:
            steps = round(vol / vol_step)
            vol = steps * vol_step
            # Keep decimals stable for common broker steps like 0.01 / 0.1.
            step_text = f"{vol_step:.10f}".rstrip("0")
            decimals = len(step_text.split(".", 1)[1]) if "." in step_text else 0
            vol = round(vol, decimals)
        if vol_max > 0:
            vol = min(vol_max, vol)
        return vol

    @staticmethod
    def _safe_comment(raw: str, max_len: int = 31) -> str:
        # Many brokers enforce short ASCII-only comments. Keep it deterministic and compact.
        text = (raw or "AtlasAI").encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^A-Za-z0-9_-]", "_", text).strip("_")
        if not text:
            text = "AtlasAI"
        return text[:max_len]

    @staticmethod
    def _is_atlas_comment(raw: str) -> bool:
        text = str(raw or "").strip()
        compact = "".join(ch for ch in text.lower() if ch.isalnum())
        return compact.startswith("aai") or compact.startswith("atlasai")

    @staticmethod
    def _strategy_comment_tag(
        strategy_name: str,
        decision_id: str = "",
        lane_tag: str = "",
        overlay_tag: str = "",
        regime_tag: str = "",
    ) -> str:
        return build_broker_comment(
            strategy_name=strategy_name,
            decision_id=decision_id,
            lane_tag=lane_tag,
            overlay_tag=overlay_tag,
            regime_tag=regime_tag,
        )

    @staticmethod
    def _round_price(price: float, symbol_info) -> float:
        digits = int(getattr(symbol_info, "digits", 2) or 2)
        return round(float(price or 0.0), max(0, digits))

    @classmethod
    def _normalize_stops(
        cls,
        side: str,
        price: float,
        sl: float,
        tp: float,
        symbol_info,
        spread_points: float = 0.0,
        extra_points: int = 0,
    ) -> tuple[float, float]:
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        if point <= 0:
            return sl, tp
        min_stop_points = cls._stop_buffer_points(symbol_info, spread_points=spread_points, extra_points=extra_points)
        if min_stop_points <= 0:
            return cls._round_price(sl, symbol_info), cls._round_price(tp, symbol_info)

        min_distance = min_stop_points * point
        px = float(price or 0.0)
        sl_v = float(sl or 0.0)
        tp_v = float(tp or 0.0)
        if str(side).lower() == "buy":
            if sl_v > 0 and (px - sl_v) < min_distance:
                sl_v = px - min_distance
            if tp_v > 0 and (tp_v - px) < min_distance:
                tp_v = px + min_distance
        else:
            if sl_v > 0 and (sl_v - px) < min_distance:
                sl_v = px + min_distance
            if tp_v > 0 and (px - tp_v) < min_distance:
                tp_v = px - min_distance
        sl_v = cls._round_price(sl_v, symbol_info)
        tp_v = cls._round_price(tp_v, symbol_info)

        # After price rounding, push once more if needed so the final sent price
        # still clears the broker minimum distance.
        if str(side).lower() == "buy":
            if sl_v > 0 and (px - sl_v) / point < min_stop_points:
                sl_v = cls._round_price(px - ((min_stop_points + 1) * point), symbol_info)
            if tp_v > 0 and (tp_v - px) / point < min_stop_points:
                tp_v = cls._round_price(px + ((min_stop_points + 1) * point), symbol_info)
        else:
            if sl_v > 0 and (sl_v - px) / point < min_stop_points:
                sl_v = cls._round_price(px + ((min_stop_points + 1) * point), symbol_info)
            if tp_v > 0 and (px - tp_v) / point < min_stop_points:
                tp_v = cls._round_price(px - ((min_stop_points + 1) * point), symbol_info)
        return sl_v, tp_v

    def _build_request(self, signal: TradeSignal, price: float, volume: float, symbol_info, decision_id: str = "", extra_stop_points: int = 0) -> dict:
        tick = mt5.symbol_info_tick(self.settings.symbol)
        spread_points = 0.0
        if tick is not None:
            point = float(getattr(symbol_info, "point", 0.0) or 0.0)
            if point > 0:
                spread_points = abs(float(tick.ask) - float(tick.bid)) / point
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        sl = price - signal.stop_loss_points * point if signal.side == "buy" else price + signal.stop_loss_points * point
        tp = price + signal.take_profit_points * point if signal.side == "buy" else price - signal.take_profit_points * point
        sl, tp = self._normalize_stops(
            signal.side,
            sl=sl,
            tp=tp,
            price=price,
            symbol_info=symbol_info,
            spread_points=spread_points,
            extra_points=extra_stop_points,
        )
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.settings.symbol,
            "volume": self._normalize_volume(volume, symbol_info),
            "type": mt5.ORDER_TYPE_BUY if signal.side == "buy" else mt5.ORDER_TYPE_SELL,
            "price": self._round_price(price, symbol_info),
            "sl": sl,
            "tp": tp,
            "deviation": max(1, int(self.settings.max_slippage_points)),
            "magic": self.settings.magic_number,
            "comment": self._safe_comment(self._strategy_comment_tag(signal.strategy, decision_id=decision_id)),
            "type_time": mt5.ORDER_TIME_GTC,
        }

    def execute(self, signal: TradeSignal, volume: float, decision_id: str = "") -> ExecutionResult:
        # Legacy direct-entry path for non-guarded internals/manual harnesses.
        # AtlasTrader demo/live-like entry routing should use ExecutionGate, not this method.
        if signal.side == "flat":
            return ExecutionResult(True, 0, "No trade")

        tick = mt5.symbol_info_tick(self.settings.symbol)
        if tick is None:
            return ExecutionResult(False, -1, "No symbol tick")

        symbol_info = mt5.symbol_info(self.settings.symbol)
        if symbol_info is None:
            return ExecutionResult(False, -3, "No symbol info")

        price = tick.ask if signal.side == "buy" else tick.bid
        request_base = self._build_request(signal, price=price, volume=volume, symbol_info=symbol_info, decision_id=decision_id)

        preflight_ok, preflight_note = self._preflight_order_request(request_base, symbol_info)
        if not preflight_ok:
            return ExecutionResult(False, -4, f"preflight failed: {preflight_note}")

        # Build filling mode candidates from symbol capabilities first.
        fill_modes: list[int] = []
        filling_flags = int(getattr(symbol_info, "filling_mode", 0) or 0)
        symbol_fok = int(getattr(mt5, "SYMBOL_FILLING_FOK", 1))
        symbol_ioc = int(getattr(mt5, "SYMBOL_FILLING_IOC", 2))
        symbol_return = int(getattr(mt5, "SYMBOL_FILLING_RETURN", 4))
        if filling_flags & symbol_fok:
            fill_modes.append(mt5.ORDER_FILLING_FOK)
        if filling_flags & symbol_ioc:
            fill_modes.append(mt5.ORDER_FILLING_IOC)
        if filling_flags & symbol_return:
            fill_modes.append(mt5.ORDER_FILLING_RETURN)
        # Fallback order if broker flags are missing/unexpected.
        if not fill_modes:
            fill_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]

        last_result = None
        last_check_error = ""
        last_send_error = ""
        stop_retry_points = [0, 12]
        for extra_stop_points in stop_retry_points:
            if extra_stop_points > 0:
                latest_tick = mt5.symbol_info_tick(self.settings.symbol)
                if latest_tick is not None:
                    price = latest_tick.ask if signal.side == "buy" else latest_tick.bid
                request_base = self._build_request(
                    signal,
                    price=price,
                    volume=volume,
                    symbol_info=symbol_info,
                    decision_id=decision_id,
                    extra_stop_points=extra_stop_points,
                )
                preflight_ok, preflight_note = self._preflight_order_request(request_base, symbol_info)
                if not preflight_ok:
                    last_check_error = f"preflight retry failed: {preflight_note}"
                    continue
            for fill in fill_modes:
                request = dict(request_base)
                request["type_filling"] = fill

                check = mt5.order_check(request)
                if check is None:
                    last_check_error = f"order_check none: {mt5.last_error()}"
                    continue
                check_retcode = getattr(check, "retcode", 0)
                check_comment = str(getattr(check, "comment", "") or "")
                if check_retcode not in (0, mt5.TRADE_RETCODE_DONE):
                    last_check_error = f"order_check retcode={check_retcode} comment={check_comment}"
                    if extra_stop_points == 0 and "invalid stops" in check_comment.lower():
                        break
                    continue

                result = mt5.order_send(request)
                if result is None:
                    last_send_error = f"order_send none: {mt5.last_error()}"
                    continue
                last_result = result
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    return ExecutionResult(
                        True,
                        result.retcode,
                        getattr(result, "comment", ""),
                        order_id=int(getattr(result, "order", 0) or 0),
                        deal_id=int(getattr(result, "deal", 0) or 0),
                    )
                if extra_stop_points == 0 and str(getattr(result, "comment", "") or "").lower() == "invalid stops":
                    break

        if last_result is None:
            detail = last_send_error or last_check_error or "order_send failed"
            return ExecutionResult(False, -2, detail)
        return ExecutionResult(
            False,
            last_result.retcode,
            getattr(last_result, "comment", ""),
            order_id=int(getattr(last_result, "order", 0) or 0),
            deal_id=int(getattr(last_result, "deal", 0) or 0),
        )

    def _preflight_order_request(self, request: dict, symbol_info) -> tuple[bool, str]:
        tick = mt5.symbol_info_tick(self.settings.symbol)
        if tick is None:
            return False, "No market tick"
        account = mt5.account_info()
        if account is None:
            return False, "No account info"
        if not getattr(account, "trade_allowed", True):
            return False, "Account trade not allowed"
        if not getattr(symbol_info, "visible", True):
            return False, "Symbol not visible"
        if getattr(symbol_info, "trade_mode", 0) == 0:
            return False, "Symbol trade mode disabled"
        magic = int(request.get("magic", 0) or 0)
        if magic <= 0:
            return False, "Missing Atlas magic number"
        comment = str(request.get("comment", "") or "")
        if not self._is_atlas_comment(comment):
            return False, "Missing Atlas broker comment"

        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        if point <= 0:
            return False, "Invalid symbol point size"

        spread_points = abs(float(tick.ask) - float(tick.bid)) / point
        if spread_points > float(self.settings.max_spread_points):
            return False, f"Spread too high ({spread_points:.1f} pts)"

        vol = float(request.get("volume", 0.0) or 0.0)
        vol_min = float(getattr(symbol_info, "volume_min", 0.0) or 0.0)
        vol_max = float(getattr(symbol_info, "volume_max", 0.0) or 0.0)
        vol_step = float(getattr(symbol_info, "volume_step", 0.0) or 0.0)
        if vol_min and vol < vol_min:
            return False, f"Volume below min ({vol} < {vol_min})"
        if vol_max and vol > vol_max:
            return False, f"Volume above max ({vol} > {vol_max})"
        if vol_step:
            steps = round(vol / vol_step)
            aligned = abs((steps * vol_step) - vol) <= max(1e-6, vol_step * 0.01)
            if not aligned:
                return False, f"Volume not aligned to step ({vol_step})"

        min_stop = self._stop_buffer_points(symbol_info, spread_points=spread_points)
        if min_stop > 0:
            price = float(request.get("price", 0.0) or 0.0)
            sl = float(request.get("sl", 0.0) or 0.0)
            tp = float(request.get("tp", 0.0) or 0.0)
            if sl > 0 and abs(price - sl) / point < min_stop:
                return False, f"SL too close (min {min_stop} pts)"
            if tp > 0 and abs(tp - price) / point < min_stop:
                return False, f"TP too close (min {min_stop} pts)"

        try:
            margin = mt5.order_calc_margin(
                request["type"],
                request["symbol"],
                request["volume"],
                request["price"],
            )
            if margin is None:
                return False, f"order_calc_margin failed: {mt5.last_error()}"
            if float(margin) > float(getattr(account, "margin_free", 0.0) or 0.0):
                return False, f"Insufficient free margin (need {margin:.2f})"
        except Exception as exc:
            return False, f"Margin precheck failed: {exc}"

        return True, "ok"

    def close_stale_positions(self, *, allow_close: bool = False) -> list[ExecutionResult]:
        # Operational housekeeping exception: close pre-existing stale positions directly.
        # This is not an entry-order path and remains isolated from guarded entry execution.
        # Fail closed by default; caller must explicitly allow housekeeping close after
        # upstream guarded safety checks (kill switch/sleep/gate sync).
        if not bool(allow_close):
            return []
        positions = mt5.positions_get(symbol=self.settings.symbol) or []
        today_utc = datetime.now(timezone.utc).date()
        closed: list[ExecutionResult] = []

        for pos in positions:
            opened = datetime.fromtimestamp(pos.time, tz=timezone.utc).date()
            if opened >= today_utc:
                continue

            tick = mt5.symbol_info_tick(pos.symbol)
            if tick is None:
                closed.append(ExecutionResult(False, -1, f"No tick for close {pos.ticket}"))
                continue

            close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "position": pos.ticket,
                "volume": pos.volume,
                "type": close_type,
                "price": price,
                "deviation": 20,
                "magic": self.settings.magic_number,
                "comment": self._safe_comment("AtlasAI"),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            result = mt5.order_send(request)
            if result is None:
                closed.append(ExecutionResult(False, -2, f"Close failed {pos.ticket}"))
                continue
            ok = result.retcode == mt5.TRADE_RETCODE_DONE
            closed.append(
                ExecutionResult(
                    ok,
                    result.retcode,
                    getattr(result, "comment", ""),
                    order_id=int(getattr(result, "order", 0) or 0),
                    deal_id=int(getattr(result, "deal", 0) or 0),
                )
            )

        return closed
