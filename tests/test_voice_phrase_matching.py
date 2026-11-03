from __future__ import annotations

from atlas_ai.commands.router import route_command
from atlas_ai.voice.wake_loop import _recent_voice_window, _sleep_phrase, _wake_phrase


def test_wake_phrase_variants_match() -> None:
    assert _wake_phrase("atlas wake")
    assert _wake_phrase("atlas wake up")
    assert _wake_phrase("wake up atlas")
    assert _wake_phrase("wake atlas")
    assert _wake_phrase("please atlas wake up now")
    assert _wake_phrase("atlas wake up and start everything")


def test_split_wake_phrase_window_recombines_audio_chunks() -> None:
    window = _recent_voice_window([(1.0, "atlas"), (1.8, "wake up")], now=1.8)
    assert window == "atlas wake up"
    assert _wake_phrase(window, confidence=1.0, min_confidence=0.0)


def test_sleep_phrase_variants_match() -> None:
    assert _sleep_phrase("atlas sleep")
    assert _sleep_phrase("sleep atlas")
    assert _sleep_phrase("go to sleep atlas")
    assert _sleep_phrase("atlas sleep now please")
    assert _sleep_phrase("please atlas sleep and close all")
    assert _sleep_phrase("at least leap")
    assert not _sleep_phrase("leap")
    assert not _sleep_phrase("at least")


def test_router_routes_sleep_alias_to_listener_preserving_shutdown(monkeypatch) -> None:
    called = {"keep_listener": None}

    def _fake_close_system(*, keep_listener: bool = False):
        called["keep_listener"] = keep_listener
        return {"ok": True, "command": "close_system", "listener_preserved": keep_listener}

    monkeypatch.setattr("atlas_ai.commands.router.close_system", _fake_close_system)
    out = route_command("sleep atlas")
    assert out["ok"] is True
    assert called["keep_listener"] is True
