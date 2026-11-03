from __future__ import annotations

from types import SimpleNamespace

from atlas_ai import atlas_wake_listener
from atlas_ai.commands import system as system_commands
from atlas_ai.voice import wake_loop


def test_wake_listener_accepts_wake_up_alias() -> None:
    assert atlas_wake_listener._is_wake_trigger("atlas wake up", "Atlas wake")
    assert atlas_wake_listener._is_wake_trigger("wake up atlas", "Atlas wake")
    assert atlas_wake_listener._is_wake_trigger("atlas wake", "Atlas wake")
    assert atlas_wake_listener._is_wake_trigger("hey atlas wake up please", "Atlas wake")


def test_wake_listener_accepts_sleep_aliases() -> None:
    assert atlas_wake_listener._is_sleep_trigger("atlas sleep", "Atlas sleep")
    assert atlas_wake_listener._is_sleep_trigger("sleep atlas", "Atlas sleep")
    assert atlas_wake_listener._is_sleep_trigger("go to sleep atlas", "Atlas sleep")
    assert atlas_wake_listener._is_sleep_trigger("please atlas sleep now", "Atlas sleep")


def test_voice_wake_phrase_requires_confidence() -> None:
    assert wake_loop._wake_phrase("atlas wake", confidence=0.9, min_confidence=0.72) is True
    assert wake_loop._wake_phrase("atlas wake", confidence=0.0, min_confidence=0.72) is False


def test_close_system_forces_shutdown_even_when_startup_in_progress(monkeypatch) -> None:
    class _FakeStore:
        def __init__(self) -> None:
            self.flags = {"startup_orchestration": {"startup_in_progress": True}}
            self.events: list[tuple[str, str, dict]] = []

        def load_runtime_flags(self) -> dict:
            return dict(self.flags)

        def update_runtime_flags(self, mutator):
            self.flags = dict(mutator(dict(self.flags)) or {})
            return dict(self.flags)

        def log_event(self, event_type: str, message: str, meta: dict | None = None) -> None:
            self.events.append((event_type, message, dict(meta or {})))

    fake_store = _FakeStore()

    monkeypatch.setattr(
        system_commands,
        "load_atlas_wake_config",
        lambda: SimpleNamespace(project_root="C:\\Users\\User\\Desktop\\Atlas AI"),
    )
    monkeypatch.setattr(system_commands, "_state_store", lambda _config: fake_store)
    monkeypatch.setattr(
        system_commands,
        "run_full_shutdown",
        lambda _config, keep_listener=False: {
            "overall_result": "FULL SYSTEM STOPPED CLEANLY",
            "listener_preserved": bool(keep_listener),
        },
    )

    out = system_commands.close_system(keep_listener=True)
    assert out["ok"] is True
    assert out["listener_preserved"] is True
    assert out["overall_result"] == "FULL SYSTEM STOPPED CLEANLY"
    assert bool(dict(fake_store.flags.get("startup_orchestration", {})).get("startup_in_progress", True)) is False
