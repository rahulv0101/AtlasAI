from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import atlas_ai.desktop as desktop


class _StoreStub:
    def __init__(self, base: Path):
        self.base = Path(base)
        self._flags: dict = {}
        self.saved_flags: list[dict] = []
        self.events: list[dict] = []

    def load_runtime_flags(self) -> dict:
        return dict(self._flags)

    def save_runtime_flags(self, flags: dict) -> None:
        self._flags = dict(flags)
        self.saved_flags.append(dict(flags))

    def log_event(self, event_type: str, message: str, meta: dict | None = None) -> None:
        self.events.append({"type": event_type, "message": message, "meta": dict(meta or {})})


class _ThreadStub:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_window_control_api_keeps_native_window_private():
    api = desktop.WindowControlAPI()
    marker = object()

    api.attach(marker)

    assert getattr(api, "_window", None) is marker
    assert "window" not in vars(api)


def test_desktop_startup_failure_persists_diagnostics(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("WebView2 initialization failed")),
        start=lambda _cb: None,
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)

    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert payload.get("ok") is False
    assert payload.get("stage") in {"window_create", "corewebview2_init_failure", "start_loop"}
    assert "WebView2 initialization failed" in str(payload.get("error", ""))
    assert payload.get("exception_type") == "RuntimeError"
    assert str(payload.get("traceback", "")) != ""
    assert dict(payload.get("classification", {}) or {}).get("code") == "corewebview2_init_failure"
    assert int(payload.get("failure_count", 0)) == 1
    diag_log = store.base / "logs" / "ui_startup_diagnostics.log"
    assert diag_log.exists()
    text = diag_log.read_text(encoding="utf-8")
    assert "WebView2 initialization failed" in text
    assert "class=corewebview2_init_failure" in text
    assert "traceback=" in text
    native = dict((store.load_runtime_flags() or {}).get("ui_desktop_native_lifecycle", {}) or {})
    assert native.get("last_stage") == "wrapper_exit_reason"
    assert str(native.get("wrapper_exit_reason", "")) == "native_exception"
    history = list(native.get("history", []) or [])
    assert any(str(item.get("stage", "")) in {"start_loop", "corewebview2_init_failure"} for item in history)
    assert any(str(item.get("stage", "")) == "entered_create_window" for item in history)


def test_desktop_startup_success_records_ok_true(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    store.save_runtime_flags(
        {
            "desktop_host_stability": {
                "accessibility_recursion_seen": True,
                "force_browser_fallback": True,
                "last_reason": "desktop_host_accessibility_recursion",
                "last_detected_at": "2026-04-22T20:52:40+00:00",
                "recommended_frontend_mode": "browser_fallback",
            }
        }
    )
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_window = SimpleNamespace(maximize=lambda: None)
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: fake_window,
        start=lambda cb, **kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)

    snapshots = [dict(item.get("ui_desktop_window", {}) or {}) for item in store.saved_flags]
    assert any(bool(item.get("ok", False)) is True and str(item.get("stage", "")) == "started" for item in snapshots)
    final_payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert int(final_payload.get("failure_count", 0)) == 0
    stability = dict((store.load_runtime_flags() or {}).get("desktop_host_stability", {}) or {})
    assert stability.get("force_browser_fallback") is False
    assert stability.get("recommended_frontend_mode") == "desktop"
    native = dict((store.load_runtime_flags() or {}).get("ui_desktop_native_lifecycle", {}) or {})
    history = list(native.get("history", []) or [])
    assert any(str(item.get("stage", "")) == "create_window_returned" for item in history)
    assert any(str(item.get("stage", "")) == "first_native_callback" for item in history)


def test_desktop_startup_failure_is_fail_closed(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("catastrophic startup failure")),
        start=lambda _cb: None,
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)

    final_payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert final_payload.get("ok") is False
    assert bool(final_payload.get("ready", False)) is False


def test_webview2_failure_classification_is_actionable():
    out = desktop.classify_ui_startup_failure(
        stage="window_create",
        error="WebView2 initialization failed with CoreWebView2 E_UNEXPECTED 0x8000FFFF",
        exception_type="RuntimeError",
    )
    assert out["code"] == "corewebview2_init_failure"
    assert "CoreWebView2" in out["operator_cause"]
    assert "WebView2 Runtime" in out["operator_action"]


def test_import_failure_classification_is_actionable():
    out = desktop.classify_ui_startup_failure(
        stage="import_webview",
        error="No module named webview",
        exception_type="ModuleNotFoundError",
    )
    assert out["code"] == "pywebview_import_failure"


def test_startup_timing_failure_classification_is_actionable():
    out = desktop.classify_ui_startup_failure(
        stage="ui_server_wait",
        error="Unable to connect to the remote server (timed out)",
        exception_type="URLError",
    )
    assert out["code"] == "startup_timing_failure"


def test_generic_failure_classification_uses_required_bucket():
    out = desktop.classify_ui_startup_failure(
        stage="start_loop",
        error="unmapped error text",
        exception_type="RuntimeError",
    )
    assert out["code"] == "generic_desktop_init_failure"


def test_startup_progress_without_error_is_not_classified_as_failure():
    out = desktop.classify_ui_startup_failure(
        stage="start_loop",
        error="",
        exception_type="",
    )
    assert out["code"] == ""
    assert out["category"] == "startup_in_progress"


def test_accessibility_recursion_failure_classification_is_actionable():
    out = desktop.classify_ui_startup_failure(
        stage="start_loop",
        error="window.native.AccessibilityObject.Bounds ... maximum recursion depth exceeded while calling a Python object",
        exception_type="RuntimeError",
    )
    assert out["code"] == "desktop_host_accessibility_recursion"
    assert "accessibility recursion" in out["operator_cause"].lower()


def test_repeated_failures_increment_and_trigger_browser_only_bypass(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setenv("ATLAS_UI_DESKTOP_FAILURE_THRESHOLD", "2")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *_args, **_kwargs: opened.append(str(url)))
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    calls = {"create_window": 0}

    def _raise_webview(*args, **kwargs):
        calls["create_window"] += 1
        raise RuntimeError("WebView2 initialization failed")

    fake_webview = SimpleNamespace(
        create_window=_raise_webview,
        start=lambda _cb: None,
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)
    payload_1 = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert int(payload_1.get("failure_count", 0)) == 1
    assert bool(payload_1.get("disabled_due_to_repeated_failures", False)) is False

    desktop.launch_desktop_ui(store)
    payload_2 = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert int(payload_2.get("failure_count", 0)) >= 2
    assert bool(payload_2.get("disabled_due_to_repeated_failures", False)) is True

    calls_before_bypass = int(calls["create_window"])
    desktop.launch_desktop_ui(store)
    payload_3 = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert payload_3.get("stage") == "desktop_bypassed_repeated_failures"
    assert bool(payload_3.get("fallback_active", False)) is True
    assert calls["create_window"] == calls_before_bypass
    assert len(opened) >= 1


def test_desktop_preflight_import_failure(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "webview":
            raise ModuleNotFoundError("No module named webview")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)
    out = desktop.desktop_startup_preflight(preferred_gui="edgechromium")
    assert out["ok"] is False
    assert out["classification"]["code"] == "pywebview_import_failure"


def test_launch_import_failure_persists_actionable_diagnostics(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *_args, **_kwargs: opened.append(str(url)))
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        desktop,
        "desktop_startup_preflight",
        lambda preferred_gui="": {
            "ok": False,
            "stage": "import_webview",
            "error": "No module named webview",
            "exception_type": "ModuleNotFoundError",
            "minimal_init_possible": False,
            "classification": desktop.classify_ui_startup_failure(
                stage="import_webview",
                error="No module named webview",
                exception_type="ModuleNotFoundError",
            ),
        },
    )

    desktop.launch_desktop_ui(store)

    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert payload.get("ok") is False
    assert payload.get("stage") == "import_webview"
    assert dict(payload.get("classification", {}) or {}).get("code") == "pywebview_import_failure"
    assert bool(payload.get("fallback_active", False)) is True
    assert opened and opened[0].startswith("http://127.0.0.1:")


def test_backend_readiness_wait_happens_before_window_create(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    poll_count = {"n": 0}

    def _urlopen(*args, **kwargs):
        poll_count["n"] += 1
        if poll_count["n"] < 3:
            raise TimeoutError("timed out")
        return _Resp()

    monkeypatch.setattr(desktop.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    calls = {"create_window": 0}
    fake_window = SimpleNamespace(maximize=lambda: None)

    def _create_window(*args, **kwargs):
        calls["create_window"] += 1
        return fake_window

    fake_webview = SimpleNamespace(
        create_window=_create_window,
        start=lambda cb, **kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)
    monkeypatch.setattr(
        desktop,
        "desktop_startup_preflight",
        lambda preferred_gui="": {
            "ok": True,
            "stage": "preflight",
            "error": "",
            "exception_type": "",
            "minimal_init_possible": True,
            "webview2_runtime_available": True,
            "classification": {},
        },
    )

    desktop.launch_desktop_ui(store)
    assert poll_count["n"] >= 3
    assert calls["create_window"] == 1


def test_desktop_wrapper_attach_to_existing_server_skips_local_server_creation(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    create_app_calls = {"n": 0}
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: create_app_calls.__setitem__("n", create_app_calls["n"] + 1),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_window = SimpleNamespace(maximize=lambda: None)
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: fake_window,
        start=lambda cb, **kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store, server_already_running=True)

    assert create_app_calls["n"] == 0
    native = dict((store.load_runtime_flags() or {}).get("ui_desktop_native_lifecycle", {}) or {})
    assert native.get("server_already_running") is True


def test_successful_desktop_startup_resets_failure_counter(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setenv("ATLAS_UI_DESKTOP_FAILURE_THRESHOLD", "3")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.webbrowser, "open", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_webview_fail = SimpleNamespace(
        create_window=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("init failure")),
        start=lambda _cb: None,
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview_fail)
    desktop.launch_desktop_ui(store)
    assert int(dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {}).get("failure_count", 0)) == 1

    fake_window = SimpleNamespace(maximize=lambda: None)
    fake_webview_ok = SimpleNamespace(
        create_window=lambda *args, **kwargs: fake_window,
        start=lambda cb, **kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview_ok)
    desktop.launch_desktop_ui(store)
    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert int(payload.get("failure_count", 0)) == 0
    assert bool(payload.get("disabled_due_to_repeated_failures", False)) is False


def test_accessibility_recursion_logger_event_activates_browser_fallback(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    opened: list[str] = []
    monkeypatch.setattr(desktop.webbrowser, "open", lambda url, *_args, **_kwargs: opened.append(str(url)))
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    fake_window = SimpleNamespace(maximize=lambda: None)

    def _start(cb, **kwargs):
        cb()
        desktop.logging.getLogger("pywebview").error(
            "window.native.AccessibilityObject.Bounds maximum recursion depth exceeded while calling a Python object"
        )

    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: fake_window,
        start=_start,
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)
    desktop.launch_desktop_ui(store)
    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    assert payload.get("ok") is False
    assert bool(payload.get("fallback_active", False)) is True
    assert str(payload.get("failure_reason", "")) == "desktop_host_accessibility_recursion"
    evidence = dict(payload.get("failure_evidence", {}) or {})
    assert str(evidence.get("matched_pattern", "")) != ""
    assert len(opened) >= 1


def test_known_accessibility_recursion_prefers_safe_native_attempt(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    store._flags = {
        "ui_desktop_window": {
            "failure_reason": "desktop_host_accessibility_recursion",
        },
        "desktop_host_stability": {
            "accessibility_recursion_seen": True,
            "force_browser_fallback": True,
            "last_reason": "desktop_host_accessibility_recursion",
            "last_detected_at": "2026-04-22T17:16:36+00:00",
        },
    }
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    calls = {"create_window": 0, "kwargs": None}
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: (
            calls.__setitem__("create_window", calls["create_window"] + 1),
            calls.__setitem__("kwargs", dict(kwargs)),
            SimpleNamespace(maximize=lambda: None),
        )[-1],
        start=lambda cb, **_kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)

    payload = dict((store.load_runtime_flags() or {}).get("ui_desktop_window", {}) or {})
    native = dict((store.load_runtime_flags() or {}).get("ui_desktop_native_lifecycle", {}) or {})
    assert calls["create_window"] == 1
    assert bool(calls["kwargs"]["frameless"]) is False
    assert bool(calls["kwargs"]["easy_drag"]) is False
    assert native.get("safe_native_window") is True
    assert any(str(item.get("stage", "")) == "known_instability_detected" for item in list(native.get("history", []) or []))
    assert payload.get("frontend_mode") in {"desktop", "desktop_unavailable"}

def test_recent_diagnostics_accessibility_recursion_prefers_safe_native_attempt(monkeypatch, tmp_path):
    store = _StoreStub(tmp_path / "storage")
    log_dir = store.base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "ui_startup_diagnostics.log").write_text(
        "2026-04-22T17:16:32.220862+00:00 | ok=False | stage=desktop_bypassed_recent_instability | "
        "type= | class=desktop_host_accessibility_recursion | error=recent desktop host instability detected\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        desktop,
        "create_ui_app",
        lambda *args, **kwargs: SimpleNamespace(run=lambda **kw: None),
    )
    monkeypatch.setattr(desktop.threading, "Thread", _ThreadStub)
    monkeypatch.setattr(desktop.urllib.request, "urlopen", lambda *args, **kwargs: _Resp())
    monkeypatch.setattr(desktop.time, "sleep", lambda *_args, **_kwargs: None)
    calls = {"create_window": 0}
    fake_webview = SimpleNamespace(
        create_window=lambda *args, **kwargs: (
            calls.__setitem__("create_window", calls["create_window"] + 1),
            SimpleNamespace(maximize=lambda: None),
        )[-1],
        start=lambda cb, **_kwargs: cb(),
        windows=[],
    )
    monkeypatch.setitem(__import__("sys").modules, "webview", fake_webview)

    desktop.launch_desktop_ui(store)

    native = dict((store.load_runtime_flags() or {}).get("ui_desktop_native_lifecycle", {}) or {})
    assert calls["create_window"] == 1
    assert any(str(item.get("stage", "")) == "recent_instability_detected" for item in list(native.get("history", []) or []))
