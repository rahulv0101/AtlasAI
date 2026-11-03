from __future__ import annotations

import threading
from pathlib import Path

from atlas_ai.commands import system


class _Cfg:
    def __init__(self, project_root: Path):
        self.project_root = str(project_root)


def test_multiple_wake_triggers_are_debounced(monkeypatch, tmp_path):
    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(system, "load_atlas_wake_config", lambda: cfg)
    monkeypatch.setenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "30")
    launches = {"n": 0}

    def _launch(_cfg):
        launches["n"] += 1
        return {"health": {"overall_result": "SUCCESS"}}

    monkeypatch.setattr(system, "run_full_launch", _launch)

    first = system.start_system()
    second = system.start_system()

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["reason"] == "wake_debounced"
    assert launches["n"] == 1


def test_start_system_returns_not_ok_when_launch_failed(monkeypatch, tmp_path):
    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(system, "load_atlas_wake_config", lambda: cfg)
    monkeypatch.setenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "1")
    monkeypatch.setattr(
        system,
        "run_full_launch",
        lambda _cfg: {"health": {"overall_result": "FAILED", "overall_reason": "backend_unhealthy"}},
    )

    out = system.start_system()
    assert out["ok"] is False
    assert out["overall_result"] == "FAILED"


def test_close_command_forces_shutdown_while_startup_in_progress(monkeypatch, tmp_path):
    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(system, "load_atlas_wake_config", lambda: cfg)
    monkeypatch.setenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "1")
    gate = threading.Event()
    started = threading.Event()
    releases = {"count": 0}
    shutdown_calls = {"n": 0}

    def _launch(_cfg):
        started.set()
        gate.wait(timeout=5.0)
        releases["count"] += 1
        return {"health": {"overall_result": "SUCCESS"}}

    def _shutdown(_cfg, *, keep_listener: bool = False):
        shutdown_calls["n"] += 1
        return {"overall_result": "FULL SYSTEM STOPPED CLEANLY", "listener_preserved": bool(keep_listener)}

    monkeypatch.setattr(system, "run_full_launch", _launch)
    monkeypatch.setattr(system, "run_full_shutdown", _shutdown)

    worker = threading.Thread(target=system.start_system, daemon=True)
    worker.start()
    assert started.wait(timeout=2.0)

    stopped = system.close_system()
    assert stopped["ok"] is True
    assert stopped["overall_result"] == "FULL SYSTEM STOPPED CLEANLY"
    assert shutdown_calls["n"] == 1

    gate.set()
    worker.join(timeout=2.0)
    assert releases["count"] == 1


def test_start_system_recovers_stale_startup_in_progress(monkeypatch, tmp_path):
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(system, "load_atlas_wake_config", lambda: cfg)
    monkeypatch.setenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "1")
    monkeypatch.setenv("ATLAS_STARTUP_STALE_AFTER_SECONDS", "60")
    launches = {"n": 0}

    def _launch(_cfg):
        launches["n"] += 1
        return {"health": {"overall_result": "SUCCESS"}}

    monkeypatch.setattr(system, "run_full_launch", _launch)
    store = KnowledgeStore(str(tmp_path / "storage"))

    def _seed(flags: dict) -> dict:
        payload = dict(flags or {})
        payload["startup_orchestration"] = {
            "startup_in_progress": True,
            "startup_started_at": "2000-01-01T00:00:00+00:00",
            "startup_source": "stale_test",
            "reason": "launch_completed",
        }
        return payload

    store.update_runtime_flags(_seed)

    out = system.start_system()
    orchestration = dict(store.load_runtime_flags().get("startup_orchestration", {}) or {})

    assert out["ok"] is True
    assert launches["n"] == 1
    assert orchestration["startup_in_progress"] is False
    assert orchestration["reason"] == "launch_completed"


def test_atlas_sleep_preserves_listener_while_close_system_does_not(monkeypatch, tmp_path):
    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(system, "load_atlas_wake_config", lambda: cfg)
    calls: list[bool] = []

    def _shutdown(_cfg, *, keep_listener: bool = False):
        calls.append(bool(keep_listener))
        return {
            "overall_result": "FULL SYSTEM STOPPED CLEANLY",
            "listener_preserved": bool(keep_listener),
        }

    monkeypatch.setattr(system, "run_full_shutdown", _shutdown)

    sleep_result = system.close_system(keep_listener=True)
    close_result = system.close_system()

    assert sleep_result["ok"] is True
    assert sleep_result["listener_preserved"] is True
    assert close_result["ok"] is True
    assert close_result["listener_preserved"] is False
    assert calls == [True, False]


def test_wake_listener_uses_orchestration_and_prevents_duplicate_launch(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener

    cfg = _Cfg(tmp_path)
    monkeypatch.setattr(
        listener,
        "load_atlas_wake_config",
        lambda: type("Cfg", (), {"trigger_phrase": "Atlas wake", "sleep_phrase": "Atlas sleep", "project_root": str(tmp_path)})(),
    )
    popen_calls = {"n": 0}

    class _Proc:
        pass

    def _popen(*_args, **_kwargs):
        popen_calls["n"] += 1
        return _Proc()

    monkeypatch.setattr(listener.subprocess, "Popen", _popen)

    first = listener.handle_trigger("Atlas wake")
    second = listener.handle_trigger("Atlas wake")

    assert first["matched"] is True and first["launched"] is True
    assert second["matched"] is True and second["launched"] is False
    assert second["reason"] in {"startup_already_in_progress", "startup_dispatch_in_progress"}
    assert popen_calls["n"] == 1


def test_wake_listener_persists_voice_startup_diagnostics(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    class _Cfg:
        trigger_phrase = "Atlas wake"
        sleep_phrase = "Atlas sleep"
        project_root = str(tmp_path)

    monkeypatch.setattr(listener, "load_atlas_wake_config", lambda: _Cfg())
    monkeypatch.setattr(listener.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        listener,
        "start_system",
        lambda: {
            "ok": True,
            "overall_result": "SUCCESS",
            "summary": {
                "health": {
                    "overall_result": "SUCCESS",
                    "overall_reason": "backend_healthy_desktop_ready",
                    "desktop_shell_health": {
                        "fallback_active": False,
                        "failure_reason": "",
                    },
                }
            },
        },
    )

    result = listener.handle_trigger_async("Atlas wake")
    listener._complete_async_start()
    flags = KnowledgeStore(str(tmp_path / "storage")).load_runtime_flags()
    state = dict(flags.get("voice_listener_orchestration", {}) or {})

    assert result["matched"] is True
    assert state["listener_path"] == "atlas_wake_listener"
    assert state["matched_command"] == "start_system"
    assert state["final_startup_verdict"] == "SUCCESS"
    assert state["browser_fallback_used"] is False


def test_wake_listener_persists_shutdown_diagnostics(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    class _Cfg:
        trigger_phrase = "Atlas wake"
        sleep_phrase = "Atlas sleep"
        project_root = str(tmp_path)

    monkeypatch.setattr(listener, "load_atlas_wake_config", lambda: _Cfg())
    monkeypatch.setattr(
        listener,
        "close_system",
        lambda keep_listener=False: {
            "ok": True,
            "overall_result": "FULL SYSTEM STOPPED CLEANLY",
            "listener_preserved": bool(keep_listener),
            "summary": {},
        },
    )

    result = listener.handle_trigger("Atlas sleep")
    flags = KnowledgeStore(str(tmp_path / "storage")).load_runtime_flags()
    state = dict(flags.get("voice_listener_orchestration", {}) or {})

    assert result["matched"] is True
    assert state["matched_command"] == "close_system"
    assert state["confirmation_satisfied"] is True
    assert state["shutdown_result"] == "FULL SYSTEM STOPPED CLEANLY"
    assert state["listener_preserved"] is True


def test_wake_listener_async_dispatch_returns_immediately(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    class _Cfg:
        trigger_phrase = "Atlas wake"
        sleep_phrase = "Atlas sleep"
        project_root = str(tmp_path)

    monkeypatch.setattr(listener, "load_atlas_wake_config", lambda: _Cfg())
    monkeypatch.setattr(listener.subprocess, "Popen", lambda *args, **kwargs: object())

    result = listener.handle_trigger_async("Atlas wake")
    flags = KnowledgeStore(str(tmp_path / "storage")).load_runtime_flags()
    state = dict(flags.get("voice_listener_orchestration", {}) or {})

    assert result["matched"] is True
    assert result["pending"] is True
    assert result["dispatch_state"] == "started_async"
    assert state["launch_dispatch_state"] == "started_async"


def test_wake_listener_clears_startup_in_progress_after_async_completion(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    class _Cfg:
        trigger_phrase = "Atlas wake"
        sleep_phrase = "Atlas sleep"
        project_root = str(tmp_path)

    monkeypatch.setattr(listener, "load_atlas_wake_config", lambda: _Cfg())
    monkeypatch.setattr(listener.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        listener,
        "start_system",
        lambda: {
            "ok": True,
            "overall_result": "SUCCESS",
            "summary": {"health": {"overall_result": "SUCCESS", "overall_reason": "backend_healthy_desktop_ready"}},
        },
    )

    result = listener.handle_trigger_async("Atlas wake")
    assert result["matched"] is True
    listener._complete_async_start()

    flags = KnowledgeStore(str(tmp_path / "storage")).load_runtime_flags()
    orchestration = dict(flags.get("startup_orchestration", {}) or {})
    assert orchestration["startup_in_progress"] is False
    assert orchestration["reason"] == "launch_completed"
    assert orchestration.get("startup_finished_at")


def test_wake_listener_recovers_stale_startup_in_progress_before_dispatch(monkeypatch, tmp_path):
    from atlas_ai import atlas_wake_listener as listener
    from atlas_ai.memory.knowledge_store import KnowledgeStore

    class _Cfg:
        trigger_phrase = "Atlas wake"
        sleep_phrase = "Atlas sleep"
        project_root = str(tmp_path)

    monkeypatch.setattr(listener, "load_atlas_wake_config", lambda: _Cfg())
    popen_calls = {"n": 0}

    def _popen(*_args, **_kwargs):
        popen_calls["n"] += 1
        return object()

    monkeypatch.setattr(listener.subprocess, "Popen", _popen)
    store = KnowledgeStore(str(tmp_path / "storage"))

    def _seed(flags: dict) -> dict:
        payload = dict(flags or {})
        payload["startup_orchestration"] = {
            "startup_in_progress": True,
            "startup_started_at": "2000-01-01T00:00:00+00:00",
            "startup_source": "stale_test",
        }
        return payload

    store.update_runtime_flags(_seed)
    result = listener.handle_trigger_async("Atlas wake")
    flags = store.load_runtime_flags()
    orchestration = dict(flags.get("startup_orchestration", {}) or {})

    assert result["matched"] is True
    assert result["launched"] is True
    assert popen_calls["n"] == 1
    assert orchestration["startup_in_progress"] is False
    assert orchestration["reason"] == "stale_startup_state_recovered"
