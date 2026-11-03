from __future__ import annotations

import json
import time
import ctypes
import ctypes.wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_PATTERN_TYPES = {
    "pullback_continuation",
    "breakout_retest",
    "range_reaction",
    "liquidity_sweep",
    "messy_range",
    "unclear",
}


def _text(value: Any, default: str = "") -> str:
    raw = str(value if value is not None else "").strip()
    return raw or default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _image_pixels(image: Any) -> list[Any]:
    flattened = getattr(image, "get_flattened_data", None)
    if callable(flattened):
        return list(flattened())
    return list(image.getdata())


def _image_probe(image_path: str | Path | None) -> dict[str, Any]:
    if not image_path:
        return {"available": False, "status": "no_screenshot_supplied"}
    path = Path(image_path)
    if not path.exists():
        return {"available": False, "status": "screenshot_missing", "path": str(path)}
    try:
        from PIL import Image

        with Image.open(path) as img:
            width, height = img.size
            sample = img.convert("RGB").resize((64, 36))
            pixels = _image_pixels(sample)
        if not pixels:
            raise ValueError("empty image")
        brightness = [sum(px) / 3.0 for px in pixels]
        bright_ratio = sum(1 for b in brightness if b > 90.0) / max(1, len(brightness))
        contrast = (max(brightness) - min(brightness)) / 255.0
        return {
            "available": True,
            "status": "loaded",
            "path": str(path),
            "width": int(width),
            "height": int(height),
            "brightness_ratio": round(bright_ratio, 4),
            "contrast": round(contrast, 4),
        }
    except Exception as exc:
        return {"available": False, "status": "image_probe_failed", "path": str(path), "error": str(exc)}


def _parse_visual_price_action(image_path: str | Path | None) -> dict[str, Any]:
    """Extract simple visual structure from the captured chart image.

    This is intentionally local and deterministic. It does not attempt to be a
    full computer-vision trader; it gives Atlas inspectable "eyes" such as
    visible direction, chart noise, and compression so the brain can compare
    visual evidence with candle-derived features.
    """

    if not image_path:
        return {"available": False, "status": "no_screenshot_supplied"}
    path = Path(image_path)
    if not path.exists():
        return {"available": False, "status": "screenshot_missing", "path": str(path)}
    try:
        from PIL import Image, ImageFilter

        with Image.open(path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            # Focus on the chart body. Toolbars/title bars tend to sit at the
            # top; the right edge often contains price scale and may include
            # cursor labels.
            left = int(width * 0.05)
            top = int(height * 0.18)
            right = int(width * 0.94)
            bottom = int(height * 0.94)
            crop = rgb.crop((left, top, right, bottom))
            cw, ch = crop.size
            small = crop.resize((160, 90))
            pixels = _image_pixels(small)

        def _is_foreground(px: tuple[int, int, int]) -> bool:
            r, g, b = px
            mx = max(r, g, b)
            mn = min(r, g, b)
            sat = mx - mn
            # Candles, chart line, grid labels, and price marks are usually
            # brighter or more saturated than the dark chart background.
            return (mx >= 145 and sat >= 20) or (sat >= 70 and mx >= 90)

        columns: list[list[int]] = [[] for _ in range(160)]
        foreground = 0
        for idx, px in enumerate(pixels):
            if not _is_foreground(px):
                continue
            x = idx % 160
            y = idx // 160
            columns[x].append(y)
            foreground += 1

        populated = [(x, sum(ys) / len(ys)) for x, ys in enumerate(columns) if ys]
        density = foreground / max(1, len(pixels))
        if len(populated) < 16 or density < 0.004:
            return {
                "available": True,
                "status": "insufficient_visual_price_pixels",
                "width": width,
                "height": height,
                "chart_crop": {"left": left, "top": top, "right": right, "bottom": bottom},
                "foreground_density": round(density, 5),
                "visual_direction": "unknown",
                "visual_structure": "mixed",
                "visual_momentum": "unknown",
                "visual_location": "unknown",
                "visual_noise": "unknown",
                "confidence": 0.18,
            }

        xs = [float(x) for x, _ in populated]
        ys = [float(y) for _, y in populated]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs) or 1.0
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        # In image coordinates, lower y means higher price.
        direction = "up" if slope < -0.035 else ("down" if slope > 0.035 else "sideways")
        structure = "HH/HL" if direction == "up" else ("LH/LL" if direction == "down" else "mixed")

        first = [y for x, y in populated if x < 42]
        last = [y for x, y in populated if x > 117]
        visual_move = 0.0
        if first and last:
            visual_move = (sum(first) / len(first)) - (sum(last) / len(last))

        edge = small.convert("L").filter(ImageFilter.FIND_EDGES)
        edge_pixels = _image_pixels(edge)
        edge_density = sum(1 for p in edge_pixels if p > 32) / max(1, len(edge_pixels))
        y_span = (max(ys) - min(ys)) / 90.0
        noise = "high" if edge_density > 0.28 or density > 0.17 else ("medium" if edge_density > 0.16 else "low")
        momentum = "strong" if abs(visual_move) > 8.0 and noise != "high" else ("weak" if abs(visual_move) < 3.0 or noise == "high" else "mixed")
        location = "mid_range"
        if last:
            last_avg = sum(last) / len(last)
            pos = last_avg / 90.0
            if direction == "up" and pos >= 0.54:
                location = "pullback"
            elif direction == "down" and pos <= 0.46:
                location = "pullback"
            elif pos <= 0.18 or pos >= 0.82:
                location = "edge"

        confidence = 0.35
        confidence += min(0.22, len(populated) / 160.0 * 0.22)
        confidence += 0.12 if abs(slope) >= 0.06 else 0.02
        confidence += 0.10 if noise == "low" else (-0.08 if noise == "high" else 0.0)
        confidence += 0.08 if y_span >= 0.18 else -0.04
        confidence = _clamp(confidence)
        return {
            "available": True,
            "status": "parsed",
            "width": width,
            "height": height,
            "chart_crop": {"left": left, "top": top, "right": right, "bottom": bottom},
            "foreground_density": round(density, 5),
            "edge_density": round(edge_density, 5),
            "slope": round(slope, 5),
            "visual_move": round(visual_move, 4),
            "visual_direction": direction,
            "visual_structure": structure,
            "visual_momentum": momentum,
            "visual_location": location,
            "visual_noise": noise,
            "visual_range_span": round(y_span, 4),
            "confidence": round(confidence, 4),
        }
    except Exception as exc:
        return {"available": False, "status": "visual_parse_failed", "path": str(path), "error": str(exc)}


def _find_chart_window_bbox() -> dict[str, Any]:
    """Find a visible MT5/HFM chart window on Windows.

    This stays deliberately narrow: Atlas should capture the trading terminal
    when available, and report a clear unavailable state when it cannot find it.
    """

    if not hasattr(ctypes, "windll"):
        return {"found": False, "reason": "windows_window_api_unavailable"}
    try:
        user32 = ctypes.windll.user32
        matches: list[dict[str, Any]] = []
        keywords = ("metatrader", "hfmarketssa", "hf markets", "demo account", "xauusd")

        def _add_match(hwnd: int, title: str, rect_values: tuple[int, int, int, int]) -> None:
            title = str(title or "").strip()
            if not title:
                return
            lowered = title.lower()
            if not any(k in lowered for k in keywords):
                return
            score = _terminal_window_score(title)
            if score <= 0:
                return
            left, top, right, bottom = [int(value) for value in rect_values]
            width = int(right - left)
            height = int(bottom - top)
            if width < 320 or height < 240:
                try:
                    user32.ShowWindow(int(hwnd), 9)
                    time.sleep(0.1)
                    refreshed = ctypes.wintypes.RECT()
                    if user32.GetWindowRect(int(hwnd), ctypes.byref(refreshed)):
                        left, top, right, bottom = (
                            int(refreshed.left),
                            int(refreshed.top),
                            int(refreshed.right),
                            int(refreshed.bottom),
                        )
                        width = int(right - left)
                        height = int(bottom - top)
                except Exception:
                    pass
            if width < 320 or height < 240:
                return
            matches.append(
                {
                    "hwnd": int(hwnd),
                    "title": title,
                    "bbox": (left, top, right, bottom),
                    "width": width,
                    "height": height,
                    "terminal_score": score,
                }
            )

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def _callback(hwnd: int, _lparam: int) -> bool:
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = int(user32.GetWindowTextLengthW(hwnd))
                if length <= 0:
                    return True
                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = str(buffer.value or "").strip()
                if not title:
                    return True
                rect = ctypes.wintypes.RECT()
                if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                    return True
                _add_match(int(hwnd), title, (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)))
            except Exception:
                return True
            return True

        user32.EnumWindows(EnumWindowsProc(_callback), 0)
        if not matches:
            try:
                import win32gui

                def _win32_callback(hwnd: int, _extra: object) -> bool:
                    try:
                        if not win32gui.IsWindowVisible(hwnd):
                            return True
                        title = str(win32gui.GetWindowText(hwnd) or "").strip()
                        if not title:
                            return True
                        rect = win32gui.GetWindowRect(hwnd)
                        _add_match(int(hwnd), title, tuple(int(value) for value in rect))
                    except Exception:
                        return True
                    return True

                win32gui.EnumWindows(_win32_callback, None)
            except Exception:
                pass
        if not matches:
            return {"found": False, "reason": "mt5_or_hfm_window_not_found"}
        matches.sort(
            key=lambda row: (
                -int(row.get("terminal_score", 0) or 0),
                -int(row.get("width", 0)) * int(row.get("height", 0)),
            )
        )
        best = matches[0]
        best["found"] = True
        best["match_count"] = len(matches)
        return best
    except Exception as exc:
        return {"found": False, "reason": "window_scan_failed", "error": str(exc)}


def _terminal_window_score(title: str) -> int:
    lowered = str(title or "").strip().lower()
    if not lowered:
        return 0
    negative = (
        "claude",
        "chatgpt",
        "browser",
        "chrome",
        "microsoft edge",
        "firefox",
        "https://",
        "http://",
        "new chat",
        "artifact",
    )
    if any(token in lowered for token in negative):
        return 0
    score = 0
    if "metatrader" in lowered:
        score += 80
    if "hfmarketssa" in lowered or "hf markets" in lowered:
        score += 55
    if "demo account" in lowered:
        score += 35
    if "hedge" in lowered:
        score += 15
    if "[xauusd" in lowered or "xauusd,m" in lowered:
        score += 35
    elif "xauusd" in lowered or "xau/usd" in lowered:
        score += 8
    return score


def _focus_chart_window(hwnd: int | None) -> dict[str, Any]:
    if not hwnd or not hasattr(ctypes, "windll"):
        return {"attempted": False, "ok": False, "reason": "window_focus_unavailable"}
    try:
        user32 = ctypes.windll.user32
        # Restore if minimized, then request foreground so ImageGrab captures
        # the terminal pixels instead of whatever overlaps it.
        user32.ShowWindow(int(hwnd), 9)
        ok = bool(user32.SetForegroundWindow(int(hwnd)))
        time.sleep(0.25)
        return {"attempted": True, "ok": ok, "hwnd": int(hwnd)}
    except Exception as exc:
        return {"attempted": True, "ok": False, "hwnd": int(hwnd), "error": str(exc)}


def _capture_window_content(hwnd: int, width: int, height: int):
    try:
        import win32con
        import win32gui
        import win32ui
        from PIL import Image

        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, int(width), int(height))
        save_dc.SelectObject(bitmap)
        result = ctypes.windll.user32.PrintWindow(int(hwnd), save_dc.GetSafeHdc(), 2)
        bmp_info = bitmap.GetInfo()
        bmp_bytes = bitmap.GetBitmapBits(True)
        image = Image.frombuffer(
            "RGB",
            (int(bmp_info["bmWidth"]), int(bmp_info["bmHeight"])),
            bmp_bytes,
            "raw",
            "BGRX",
            0,
            1,
        )
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        if not result:
            return None, {"ok": False, "method": "printwindow", "reason": "printwindow_returned_false"}
        return image, {"ok": True, "method": "printwindow"}
    except Exception as exc:
        return None, {"ok": False, "method": "printwindow", "error": str(exc)}


def _pattern_from_chart_features(chart_features: dict[str, Any]) -> tuple[str, str, list[str]]:
    trend = _text(chart_features.get("trend_state"), "unclear").lower()
    structure = _text(chart_features.get("trend_structure"), "mixed").upper()
    momentum = _text(chart_features.get("momentum_state"), "mixed").lower()
    location = _text(chart_features.get("location_state"), "unclear").lower()
    setup_hint = _text(chart_features.get("setup_type_hint"), "no_trade").lower()
    wickiness = _num(chart_features.get("wickiness"), 0.0)
    compression = _num(chart_features.get("compression_score"), 0.0)
    expansion = _num(chart_features.get("candle_expansion"), 0.0)
    reasons: list[str] = []

    clear_structure = structure in {"HH/HL", "LH/LL"} or trend in {"uptrend", "downtrend"}
    if clear_structure:
        reasons.append("directional structure is visible")
    else:
        reasons.append("structure is mixed or unclear")

    if location == "pullback" and clear_structure:
        reasons.append("price is in a pullback location")
        return "pullback_continuation", "clean" if momentum in {"strong", "mixed"} else "developing", reasons
    if location == "breakout" and clear_structure:
        reasons.append("price is pressing a breakout area")
        return "breakout_retest", "developing" if momentum == "strong" else "unclear", reasons
    if trend == "range" and location in {"pullback", "consolidation"}:
        reasons.append("range-edge or consolidation behavior is present")
        return "range_reaction", "developing", reasons
    if wickiness >= 0.68 and expansion >= 0.25:
        reasons.append("large wick/rejection behavior is visible in recent candles")
        return "liquidity_sweep", "developing", reasons
    if compression >= 0.55 or setup_hint == "consolidation":
        reasons.append("compressed or overlapping candles suggest a messy range")
        return "messy_range", "messy", reasons
    return "unclear", "unclear", reasons


def analyze_chart_visual_context(
    *,
    status: dict[str, Any] | None = None,
    chart_features: dict[str, Any] | None = None,
    image_path: str | Path | None = None,
    timeframe: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Build a bounded visual chart read.

    This is observability and confirmation evidence only. It never creates an
    order, never bypasses gates, and never changes strategy/risk/execution
    policy. When no screenshot is supplied it produces a visual-style read from
    Atlas' existing chart-feature state and marks the source accordingly.
    """

    st = dict(status or {})
    features = dict(chart_features or st.get("chart_features") or {})
    image = _image_probe(image_path)
    visual_features = _parse_visual_price_action(image_path) if image.get("available") else {"available": False, "status": image.get("status", "no_screenshot_supplied")}
    pattern, cleanliness, reasons = _pattern_from_chart_features(features)
    trend = _text(features.get("trend_state"), "unclear").lower()
    structure = _text(features.get("trend_structure"), "mixed")
    momentum = _text(features.get("momentum_state"), "mixed").lower()
    location = _text(features.get("location_state"), "unclear").lower()
    reliability = _text(features.get("reliability"), "low").lower()

    visual_bias = "neutral"
    if structure.upper() == "HH/HL" or trend == "uptrend":
        visual_bias = "buy"
    elif structure.upper() == "LH/LL" or trend == "downtrend":
        visual_bias = "sell"

    parsed_structure = _text(visual_features.get("visual_structure"), "").upper()
    parsed_direction = _text(visual_features.get("visual_direction"), "").lower()
    visual_parser_bias = "neutral"
    if parsed_structure == "HH/HL" or parsed_direction == "up":
        visual_parser_bias = "buy"
    elif parsed_structure == "LH/LL" or parsed_direction == "down":
        visual_parser_bias = "sell"
    visual_conflict = bool(
        image.get("available")
        and visual_parser_bias in {"buy", "sell"}
        and visual_bias in {"buy", "sell"}
        and visual_parser_bias != visual_bias
    )
    if visual_parser_bias in {"buy", "sell"} and visual_bias == "neutral":
        visual_bias = visual_parser_bias
    if visual_conflict:
        reasons.append("screenshot structure conflicts with candle-derived structure")

    quality = 0.25
    if visual_bias in {"buy", "sell"}:
        quality += 0.18
    if location in {"pullback", "breakout"}:
        quality += 0.18
    if momentum == "strong":
        quality += 0.16
    elif momentum == "mixed":
        quality += 0.06
    if cleanliness == "messy" or location == "mid_range":
        quality -= 0.18
    if reliability == "high":
        quality += 0.08
    elif reliability == "low":
        quality -= 0.08
    if image.get("available"):
        quality += 0.04
        quality += min(0.08, max(0.0, _num(visual_features.get("confidence"), 0.0) - 0.45) * 0.18)
        if visual_conflict:
            quality -= 0.16
        if _text(visual_features.get("visual_noise"), "").lower() == "high":
            quality -= 0.08
        if _num(image.get("contrast"), 0.0) < 0.12:
            quality -= 0.12
            reasons.append("screenshot contrast is too low for strong visual confidence")
    quality = _clamp(quality)

    visual_confirmation = "caution"
    if quality >= 0.72 and pattern not in {"messy_range", "unclear"}:
        visual_confirmation = "supports_setup"
    elif quality <= 0.42 or pattern in {"messy_range", "unclear"}:
        visual_confirmation = "does_not_confirm"

    confidence_effect = "advisory_no_change"
    if visual_confirmation == "supports_setup":
        confidence_effect = "confirmation_only_no_bypass"
    elif visual_confirmation == "does_not_confirm":
        confidence_effect = "caution_or_confidence_haircut_only"

    source = "screenshot_plus_chart_features" if image.get("available") else "chart_features_visual_proxy"
    if pattern not in VALID_PATTERN_TYPES:
        pattern = "unclear"

    return {
        "enabled": True,
        "advisory_only": True,
        "source": source,
        "symbol": _text(symbol or st.get("symbol"), "XAUUSD"),
        "timeframe": _text(timeframe or st.get("timeframe") or features.get("timeframe"), "unknown"),
        "captured_at": _utc_now(),
        "visual_bias": visual_bias,
        "visual_structure": structure or "mixed",
        "pattern_seen": pattern,
        "setup_cleanliness": cleanliness,
        "setup_quality": round(quality, 4),
        "visual_confirmation": visual_confirmation,
        "confidence_effect": confidence_effect,
        "reasons": reasons[:6],
        "image": image,
        "visual_features": visual_features,
        "visual_parser_bias": visual_parser_bias,
        "visual_conflict": visual_conflict,
        "evidence_required": [
            "screenshot from the active chart",
            "matching M1/M15 candle-derived structure",
            "clean location away from mid-range",
        ],
        "policy_note": "Visual chart reading is advisory evidence only; it cannot place trades, bypass guards, or rewrite strategy/risk/execution policy.",
    }


def capture_desktop_screenshot(
    output_dir: str | Path = "storage/chart_vision",
    *,
    target_window: str = "mt5",
    fallback_full_desktop: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "latest_desktop_chart.png"
    window = _find_chart_window_bbox() if str(target_window or "").lower() in {"mt5", "hfm", "terminal", "chart"} else {"found": False, "reason": "window_target_disabled"}
    try:
        bbox = window.get("bbox") if bool(window.get("found", False)) else None
        if not bbox and not fallback_full_desktop:
            return {
                "ok": False,
                "path": str(target),
                "captured_at": _utc_now(),
                "target_window": str(target_window or "mt5"),
                "window": window,
                "error": str(window.get("reason", "target_window_not_found") or "target_window_not_found"),
                "note": "MT5/HFM chart window was not found; visual read can still use chart-feature proxy.",
            }

        from PIL import ImageGrab

        if bbox:
            hwnd = int(window.get("hwnd", 0) or 0)
            image, capture_method = _capture_window_content(
                hwnd,
                int(window.get("width", 0) or (int(bbox[2]) - int(bbox[0]))),
                int(window.get("height", 0) or (int(bbox[3]) - int(bbox[1]))),
            )
            if image is None:
                focus = _focus_chart_window(hwnd)
                image = ImageGrab.grab(bbox=tuple(bbox))
                capture_method = {
                    "ok": True,
                    "method": "imagegrab_visible_pixels",
                    "fallback_from": "printwindow",
                    "printwindow": capture_method,
                }
            else:
                focus = {"attempted": False, "ok": False, "reason": "printwindow_content_capture_used"}
            capture_source = "mt5_window"
        else:
            focus = {"attempted": False, "ok": False, "reason": "full_desktop_fallback"}
            image = ImageGrab.grab()
            capture_source = "full_desktop_fallback"
            capture_method = {"ok": True, "method": "imagegrab_full_desktop"}
        image.save(target)
        return {
            "ok": True,
            "path": str(target),
            "captured_at": _utc_now(),
            "target_window": str(target_window or "mt5"),
            "capture_source": capture_source,
            "capture_method": capture_method,
            "window": window,
            "focus": focus,
            "note": "Captured MT5/HFM chart screenshot for advisory visual chart analysis.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "path": str(target),
            "captured_at": _utc_now(),
            "target_window": str(target_window or "mt5"),
            "window": window,
            "error": str(exc),
            "note": "Screenshot capture unavailable; visual read can still use chart-feature proxy.",
        }


def save_visual_read(read: dict[str, Any], output_dir: str | Path = "storage/chart_vision") -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(read or {})
    payload.setdefault("saved_at", _utc_now())
    target = out_dir / "latest_visual_read.json"
    should_write_latest = True
    if str(payload.get("source", "") or "") == "chart_features_visual_proxy" and target.exists():
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
            current_source = str(current.get("source", "") or "")
            current_ts = _text(current.get("captured_at") or current.get("saved_at"), "")
            if current_source == "screenshot_plus_chart_features" and current_ts:
                ts = datetime.fromisoformat(current_ts.replace("Z", "+00:00"))
                should_write_latest = (datetime.now(timezone.utc) - ts).total_seconds() > 900
        except Exception:
            should_write_latest = True
    if should_write_latest:
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history = out_dir / "visual_reads.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")
    return payload


def load_latest_visual_read(output_dir: str | Path = "storage/chart_vision", max_age_seconds: int = 600) -> dict[str, Any]:
    target = Path(output_dir) / "latest_visual_read.json"
    if not target.exists():
        return {"enabled": True, "available": False, "status": "no_visual_read_yet", "advisory_only": True}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"enabled": True, "available": False, "status": "visual_read_load_failed", "error": str(exc), "advisory_only": True}
    captured = _text(payload.get("captured_at") or payload.get("saved_at"), "")
    stale = False
    if captured:
        try:
            ts = datetime.fromisoformat(captured.replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - ts).total_seconds() > max(1, int(max_age_seconds))
        except Exception:
            stale = False
    payload["available"] = True
    payload["stale"] = bool(stale)
    payload["loaded_at_epoch"] = time.time()
    return payload
