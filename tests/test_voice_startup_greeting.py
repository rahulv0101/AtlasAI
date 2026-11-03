from __future__ import annotations

from atlas_ai.voice import startup as startup_voice


def test_startup_announcement_prefers_clone_backend_and_speaks_once(monkeypatch) -> None:
    spoken: list[str] = []
    backends_seen: list[str] = []

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):  # noqa: ANN001
            self._target = target

        def start(self) -> None:
            if self._target:
                self._target()

    monkeypatch.setattr(startup_voice.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(startup_voice, "_has_announced_startup", False)

    class _Speaker:
        def __init__(self) -> None:
            backend = str(__import__("os").getenv("ATLAS_TTS_BACKEND", "")).strip().lower()
            self._backend = "xtts_clone" if backend == "clone" else "edge"
            backends_seen.append(self._backend)

        def speak(self, text: str) -> None:
            spoken.append(text)

    monkeypatch.setattr(startup_voice, "SpeechOutput", _Speaker)

    assert startup_voice.announce_ui_ready_once() is True
    assert startup_voice.announce_ui_ready_once() is False
    assert len(spoken) == 1
    assert spoken[0].startswith("Hello sir, the time is ")
    assert ", and let's make the market respect us." in spoken[0]
    assert "xtts_clone" in backends_seen


def test_startup_announcement_falls_back_when_clone_backend_unavailable(monkeypatch) -> None:
    spoken: list[str] = []
    backends_seen: list[str] = []

    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):  # noqa: ANN001
            self._target = target

        def start(self) -> None:
            if self._target:
                self._target()

    monkeypatch.setattr(startup_voice.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(startup_voice, "_has_announced_startup", False)

    class _Speaker:
        def __init__(self) -> None:
            backend = str(__import__("os").getenv("ATLAS_TTS_BACKEND", "")).strip().lower()
            self._backend = "edge" if backend == "clone" else "sapi"
            backends_seen.append(self._backend)

        def speak(self, text: str) -> None:
            spoken.append(text)

    monkeypatch.setattr(startup_voice, "SpeechOutput", _Speaker)

    assert startup_voice.announce_ui_ready_once() is True
    assert len(spoken) == 1
    assert spoken[0].startswith("Hello sir, the time is ")
    assert len(backends_seen) >= 2


def test_startup_announcement_failure_is_non_fatal(monkeypatch) -> None:
    class _ImmediateThread:
        def __init__(self, target=None, daemon=None):  # noqa: ANN001
            self._target = target

        def start(self) -> None:
            if self._target:
                self._target()

    monkeypatch.setattr(startup_voice.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(startup_voice, "_has_announced_startup", False)

    class _Speaker:
        def __init__(self) -> None:
            self._backend = "xtts_clone"

        def speak(self, text: str) -> None:
            raise RuntimeError("audio backend failure")

    monkeypatch.setattr(startup_voice, "SpeechOutput", _Speaker)
    assert startup_voice.announce_ui_ready_once() is True

