from __future__ import annotations

from atlas_ai.commands import router
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.voice import startup as startup_voice
from atlas_ai.voice.output import SpeechOutput
from atlas_ai.voice.wake_loop import _is_unclear_input, _should_execute_sleep_shutdown, _sleep_phrase, _spoken_summary, _voice_session_turn_limit, _wake_phrase


class _Source:
    def __init__(self) -> None:
        self.execute_called = False

    def load_status(self) -> dict:
        return {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"confidence": 0.25},
            "min_confidence_threshold": 0.58,
            "comparison": {"active": True, "run_id": "atlas_wake_latest"},
            "regime": {"name": "chop"},
        }

    def load_events(self, limit: int = 160) -> list[dict]:
        return [{"type": "risk", "message": "Conservative filter blocked trade due to noise."}]

    def load_runtime_flags(self) -> dict:
        return {"execute_trades": False}

    def load_comparison(self) -> dict:
        return {"active": True, "run_id": "atlas_wake_latest"}

    def execute_trade(self) -> None:
        self.execute_called = True


def test_voice_mode_shorter_than_standard_and_answer_first() -> None:
    svc = AskAtlasService(_Source())
    standard = svc.answer("What should you do next?", mode="standard")
    voice = svc.answer("What should you do next?", mode="voice")
    assert len(voice["result"]) <= len(standard["result"])
    first_line = voice["result"].splitlines()[0].strip().lower()
    assert first_line
    assert not first_line.startswith("reason:")


def test_voice_and_ui_use_same_reasoning_path() -> None:
    svc = AskAtlasService(_Source())
    standard = svc.answer("What is holding you back?", mode="standard")
    voice = svc.answer("What is holding you back?", mode="voice")
    assert standard["intent"] == voice["intent"]
    assert standard["selected_domain"] == voice["selected_domain"]


def test_startup_announcement_triggers_once(monkeypatch, tmp_path) -> None:
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

    store = KnowledgeStore(str(tmp_path / "storage"))
    assert startup_voice.announce_ui_ready_once(store) is True
    assert startup_voice.announce_ui_ready_once(store) is False
    assert len(spoken) == 1
    assert spoken[0].startswith("Hello sir, the time is ")
    assert ", and let's make the market respect us." in spoken[0]
    assert "xtts_clone" in backends_seen
    flags = store.load_runtime_flags()
    assert flags["voice_startup_greeting"]["status"] in {"success", "fallback_success", "skipped"}


def test_startup_announcement_falls_back_if_clone_backend_unavailable(monkeypatch, tmp_path) -> None:
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
            # Simulate clone path unavailable at runtime.
            self._backend = "edge" if backend == "clone" else "sapi"
            backends_seen.append(self._backend)

        def speak(self, text: str) -> None:
            spoken.append(text)

    monkeypatch.setattr(startup_voice, "SpeechOutput", _Speaker)

    store = KnowledgeStore(str(tmp_path / "storage"))
    assert startup_voice.announce_ui_ready_once(store) is True
    assert len(spoken) == 1
    assert spoken[0].startswith("Hello sir, the time is ")
    # Clone attempt then fallback speaker construction.
    assert len(backends_seen) >= 2
    flags = store.load_runtime_flags()
    assert flags["voice_startup_greeting"]["status"] == "fallback_success"


def test_startup_announcement_failure_is_non_fatal(monkeypatch, tmp_path) -> None:
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

    # Must not raise even if cloned playback fails.
    store = KnowledgeStore(str(tmp_path / "storage"))
    assert startup_voice.announce_ui_ready_once(store) is True
    flags = store.load_runtime_flags()
    assert flags["voice_startup_greeting"]["status"] == "failed"


def test_wake_and_sleep_route_to_system_commands(monkeypatch) -> None:
    monkeypatch.setattr(router, "start_system", lambda: {"ok": True, "command": "start_system"})
    close_calls: list[bool] = []

    def _close_system(*, keep_listener: bool = False):
        close_calls.append(bool(keep_listener))
        return {"ok": True, "command": "close_system", "listener_preserved": bool(keep_listener)}

    monkeypatch.setattr(router, "close_system", _close_system)
    wake = router.route_command("Atlas wake")
    sleep = router.route_command("Atlas sleep")
    close = router.route_command("close system")
    assert wake["ok"] is True and wake["command"] == "start_system"
    assert sleep["ok"] is True and sleep["command"] == "close_system"
    assert sleep["listener_preserved"] is True
    assert close["ok"] is True and close["command"] == "close_system"
    assert close_calls == [True, False]


def test_voice_queries_do_not_execute_trades_side_effect() -> None:
    source = _Source()
    svc = AskAtlasService(source)
    _ = svc.answer("What state are you in?", mode="voice")
    assert source.execute_called is False


def test_spoken_summary_avoids_raw_state_dump() -> None:
    reply = {"result": "{'execute_trades': False, 'action': 'hold'}\nextra"}
    spoken = _spoken_summary(reply)
    assert "{" not in spoken
    assert "}" not in spoken


def test_wake_sleep_phrase_variants_supported() -> None:
    assert _wake_phrase("atlas wake")
    assert _wake_phrase("atlas wake up")
    assert _wake_phrase("atlas wake up daddy's home")
    assert _wake_phrase("please atlas wake up now")
    assert _wake_phrase("wake atlas")
    assert _wake_phrase("wake up atlas")
    assert _sleep_phrase("atlas sleep")
    assert _sleep_phrase("go to sleep atlas")
    assert not _sleep_phrase("sleep")
    assert _sleep_phrase("sleep", allow_short_form=True)


def test_sleep_shutdown_executes_immediately() -> None:
    execute, arm = _should_execute_sleep_shutdown("atlas sleep", awake=True, pending_since=None, now=100.0)
    assert execute is True
    assert arm is False
    execute, arm = _should_execute_sleep_shutdown("atlas sleep", awake=True, pending_since=None, confidence=0.0, now=100.0)
    assert execute is True
    assert arm is False
    execute, arm = _should_execute_sleep_shutdown("atlas sleep", awake=True, pending_since=100.0, now=102.0)
    assert execute is True
    assert arm is False
    execute, arm = _should_execute_sleep_shutdown("at least leap", awake=True, pending_since=None, confidence=0.0, now=102.0)
    assert execute is True
    assert arm is False
    execute, arm = _should_execute_sleep_shutdown("confirm sleep", awake=True, pending_since=None, now=103.0)
    assert execute is False
    assert arm is False


def test_voice_session_turn_limit_defaults_to_single_turn(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_VOICE_SESSION_TURN_LIMIT", raising=False)
    assert _voice_session_turn_limit() == 1


def test_voice_session_turn_limit_clamps_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("ATLAS_VOICE_SESSION_TURN_LIMIT", "not-a-number")
    assert _voice_session_turn_limit() == 1
    monkeypatch.setenv("ATLAS_VOICE_SESSION_TURN_LIMIT", "99")
    assert _voice_session_turn_limit() == 10


def test_unclear_input_detection_for_repeat_prompt() -> None:
    assert _is_unclear_input("um")
    assert _is_unclear_input("hmm")
    assert _is_unclear_input("x")
    assert not _is_unclear_input("what state are you in")


def test_tts_normalizer_short_and_unlabeled() -> None:
    line = "Status: HOLD | Reason: No high-confidence trade | Run: atlas_wake_latest"
    normalized = SpeechOutput._normalize_for_speech(line)
    assert "Status:" not in normalized
    assert "Reason:" not in normalized
    assert "|" not in normalized
    assert len(normalized) <= 240
