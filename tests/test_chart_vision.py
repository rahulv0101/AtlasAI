from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from atlas_ai.vision import chart_vision
from atlas_ai.vision.chart_vision import analyze_chart_visual_context, capture_desktop_screenshot, load_latest_visual_read, save_visual_read


def test_image_pixels_prefers_pillow_flattened_data_when_available() -> None:
    class _ImageLike:
        def get_flattened_data(self):
            return [(1, 2, 3), (4, 5, 6)]

        def getdata(self):
            raise AssertionError("deprecated fallback should not be used")

    assert chart_vision._image_pixels(_ImageLike()) == [(1, 2, 3), (4, 5, 6)]


def test_visual_chart_read_supports_clean_pullback_without_execution_authority() -> None:
    read = analyze_chart_visual_context(
        status={"symbol": "XAUUSD", "timeframe": "M1"},
        chart_features={
            "trend_state": "uptrend",
            "trend_structure": "HH/HL",
            "momentum_state": "strong",
            "location_state": "pullback",
            "setup_type_hint": "pullback",
            "reliability": "medium",
        },
    )

    assert read["advisory_only"] is True
    assert read["source"] == "chart_features_visual_proxy"
    assert read["visual_bias"] == "buy"
    assert read["pattern_seen"] == "pullback_continuation"
    assert read["visual_confirmation"] == "supports_setup"
    assert "cannot place trades" in read["policy_note"]


def test_visual_chart_read_flags_messy_mid_range_as_caution() -> None:
    read = analyze_chart_visual_context(
        chart_features={
            "trend_state": "range",
            "trend_structure": "mixed",
            "momentum_state": "weak",
            "location_state": "mid_range",
            "setup_type_hint": "consolidation",
            "compression_score": 0.7,
            "reliability": "low",
        },
    )

    assert read["pattern_seen"] == "messy_range"
    assert read["visual_confirmation"] == "does_not_confirm"
    assert read["confidence_effect"] == "caution_or_confidence_haircut_only"


def test_fresh_screenshot_read_is_not_overwritten_by_proxy(tmp_path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    save_visual_read(
        {
            "source": "screenshot_plus_chart_features",
            "captured_at": now,
            "pattern_seen": "breakout_retest",
            "advisory_only": True,
        },
        tmp_path,
    )
    save_visual_read(
        {
            "source": "chart_features_visual_proxy",
            "captured_at": now,
            "pattern_seen": "pullback_continuation",
            "advisory_only": True,
        },
        tmp_path,
    )

    latest = load_latest_visual_read(tmp_path, max_age_seconds=999999999)

    assert latest["source"] == "screenshot_plus_chart_features"
    assert latest["pattern_seen"] == "breakout_retest"


def test_capture_refuses_to_fake_mt5_window_when_not_found(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(chart_vision, "_find_chart_window_bbox", lambda: {"found": False, "reason": "mt5_or_hfm_window_not_found"})

    capture = capture_desktop_screenshot(tmp_path, fallback_full_desktop=False)

    assert capture["ok"] is False
    assert capture["error"] == "mt5_or_hfm_window_not_found"
    assert "MT5/HFM chart window was not found" in capture["note"]


def test_capture_marks_mt5_window_source_when_found(tmp_path, monkeypatch) -> None:
    class _FakeImage:
        def save(self, path):
            path.write_bytes(b"fake-png")

    class _Grabber:
        @staticmethod
        def grab(bbox=None):
            assert bbox == (10, 20, 410, 320)
            return _FakeImage()

    pil = types.ModuleType("PIL")
    image_grab = types.ModuleType("PIL.ImageGrab")
    image_grab.grab = _Grabber.grab
    pil.ImageGrab = image_grab
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(sys.modules, "PIL.ImageGrab", image_grab)

    monkeypatch.setattr(
        chart_vision,
        "_find_chart_window_bbox",
        lambda: {
            "found": True,
            "title": "MetaTrader 5 - XAUUSD,M1",
            "bbox": (10, 20, 410, 320),
            "width": 400,
            "height": 300,
        },
    )
    monkeypatch.setattr(chart_vision, "_focus_chart_window", lambda hwnd: {"attempted": True, "ok": True, "hwnd": hwnd})
    monkeypatch.setattr(
        chart_vision,
        "_capture_window_content",
        lambda hwnd, width, height: (None, {"ok": False, "method": "printwindow", "reason": "test_fallback"}),
    )
    capture = capture_desktop_screenshot(tmp_path, fallback_full_desktop=False)

    assert capture["ok"] is True
    assert capture["capture_source"] == "mt5_window"
    assert capture["capture_method"]["method"] == "imagegrab_visible_pixels"
    assert capture["focus"]["attempted"] is True
    assert capture["focus"]["hwnd"] == 0
    assert capture["window"]["title"] == "MetaTrader 5 - XAUUSD,M1"
    assert (tmp_path / "latest_desktop_chart.png").exists()


def test_terminal_window_score_rejects_browser_chat_titles() -> None:
    browser_title = "HFM Gold Signal Bot · XAU/USD - Claude"
    terminal_title = "57399598 - HFMarketsSA-Demo2: Demo Account - Hedge - HF Markets SA (Pty) Ltd - [XAUUSD,M1]"

    assert chart_vision._terminal_window_score(browser_title) == 0
    assert chart_vision._terminal_window_score(terminal_title) > chart_vision._terminal_window_score("HFM Gold Signal Bot · XAU/USD")


def test_find_chart_window_restores_minimized_hfm_terminal(monkeypatch) -> None:
    title = "57399598 - HFMarketsSA-Demo2: Demo Account - Hedge - HF Markets SA (Pty) Ltd - [XAUUSD,M1]"

    class _User32:
        def __init__(self):
            self.restored = False

        def EnumWindows(self, callback, lparam):
            callback(66764, lparam)
            return True

        def IsWindowVisible(self, hwnd):
            return True

        def GetWindowTextLengthW(self, hwnd):
            return len(title)

        def GetWindowTextW(self, hwnd, buffer, length):
            buffer.value = title
            return len(title)

        def GetWindowRect(self, hwnd, rect_ptr):
            rect = rect_ptr._obj
            if self.restored:
                rect.left, rect.top, rect.right, rect.bottom = 0, 0, 1536, 816
            else:
                rect.left, rect.top, rect.right, rect.bottom = -25600, -25600, -25441, -25573
            return True

        def ShowWindow(self, hwnd, flag):
            self.restored = True
            return True

    monkeypatch.setattr(chart_vision.ctypes, "windll", types.SimpleNamespace(user32=_User32()), raising=False)
    monkeypatch.setattr(chart_vision.time, "sleep", lambda *_args, **_kwargs: None)

    window = chart_vision._find_chart_window_bbox()

    assert window["found"] is True
    assert window["title"] == title
    assert window["bbox"] == (0, 0, 1536, 816)
    assert window["terminal_score"] >= 100


def _draw_chart(path, points):
    img = Image.new("RGB", (800, 500), color=(8, 18, 26))
    draw = ImageDraw.Draw(img)
    for y in range(100, 470, 60):
        draw.line((40, y, 760, y), fill=(35, 55, 68), width=1)
    draw.line(points, fill=(40, 210, 255), width=6)
    img.save(path)


def test_visual_parser_reads_upward_chart_line(tmp_path) -> None:
    image_path = tmp_path / "up.png"
    _draw_chart(image_path, [(80, 390), (220, 330), (360, 295), (520, 230), (720, 160)])

    read = analyze_chart_visual_context(
        chart_features={"trend_state": "uptrend", "trend_structure": "HH/HL", "momentum_state": "strong", "location_state": "pullback"},
        image_path=image_path,
    )

    assert read["source"] == "screenshot_plus_chart_features"
    assert read["visual_features"]["visual_direction"] == "up"
    assert read["visual_parser_bias"] == "buy"
    assert read["visual_conflict"] is False


def test_visual_parser_flags_conflict_against_candle_structure(tmp_path) -> None:
    image_path = tmp_path / "down.png"
    _draw_chart(image_path, [(80, 160), (220, 215), (360, 260), (520, 330), (720, 390)])

    read = analyze_chart_visual_context(
        chart_features={"trend_state": "uptrend", "trend_structure": "HH/HL", "momentum_state": "mixed", "location_state": "mid_range"},
        image_path=image_path,
    )

    assert read["visual_features"]["visual_direction"] == "down"
    assert read["visual_parser_bias"] == "sell"
    assert read["visual_conflict"] is True
    assert "screenshot structure conflicts with candle-derived structure" in read["reasons"]
