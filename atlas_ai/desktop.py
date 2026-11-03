from __future__ import annotations

import contextlib
import os
import threading
import time
import traceback
import urllib.request
import webbrowser
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

from atlas_ai.memory.jsonl_tail import tail_text_lines
from atlas_ai.ui_server import create_ui_app


class WindowControlAPI:
    def __init__(self) -> None:
        self._window = None
        self._maximized = False

    def attach(self, window) -> None:
        self._window = window

    def minimize(self) -> bool:
        try:
            import webview

            win = self._window or (webview.windows[0] if webview.windows else None)
            if win is None:
                return False
            if hasattr(win, "minimize"):
                win.minimize()
                return True
        except Exception:
            return False
        return False

    def toggle_restore(self) -> bool:
        try:
            import webview

            win = self._window or (webview.windows[0] if webview.windows else None)
            if win is None:
                return False
            if self._maximized and hasattr(win, "restore"):
                win.restore()
                self._maximized = False
                return True
            if hasattr(win, "maximize"):
                win.maximize()
                self._maximized = True
                return True
            if hasattr(win, "toggle_fullscreen"):
                win.toggle_fullscreen()
                self._maximized = not self._maximized
                return True
        except Exception:
            return False
        return False

    def close(self) -> bool:
        try:
            import webview

            win = self._window or (webview.windows[0] if webview.windows else None)
            if win is None:
                return False
            if hasattr(win, "destroy"):
                win.destroy()
                return True
        except Exception:
            return False
        return False


def classify_ui_startup_failure(
    *,
    stage: str,
    error: str,
    exception_type: str,
    traceback_text: str = "",
) -> dict[str, str]:
    stage_text = str(stage or "").strip().lower()
    err = str(error or "").strip()
    exc_type = str(exception_type or "").strip()
    combined = " ".join([stage_text, err, exc_type, str(traceback_text or "")]).lower()

    startup_progress_stages = {
        "bootstrap",
        "ui_server_ready",
        "import_webview",
        "window_create",
        "start_loop",
    }
    if stage_text in startup_progress_stages and not err and not exc_type and not str(traceback_text or "").strip():
        return {
            "code": "",
            "category": "startup_in_progress",
            "operator_cause": "",
            "operator_action": "",
        }

    if stage_text == "import_webview" and (
        exc_type in {"ImportError", "ModuleNotFoundError"} or "no module named webview" in combined
    ):
        return {
            "code": "pywebview_import_failure",
            "category": "import_failure",
            "operator_cause": "pywebview package import failed for desktop host.",
            "operator_action": "Install/repair pywebview in the Atlas virtual environment.",
        }
    if any(token in combined for token in ["webview2loader.dll", "edge webview2 runtime", "0x8007007e", "class not registered"]):
        return {
            "code": "webview2_runtime_missing",
            "category": "runtime_missing",
            "operator_cause": "Microsoft Edge WebView2 runtime appears missing or not registered.",
            "operator_action": "Install or repair Microsoft Edge WebView2 Runtime, then restart Atlas.",
        }
    if any(token in combined for token in ["corewebview2", "webview2 initialization failed", "0x8000ffff", "e_unexpected"]):
        return {
            "code": "corewebview2_init_failure",
            "category": "runtime_init_failure",
            "operator_cause": "CoreWebView2 failed to initialize in this Windows session.",
            "operator_action": "Repair WebView2 Runtime, reboot Windows session, and check endpoint security/policy blocks.",
        }
    if ("accessibilityobject" in combined and "maximum recursion depth exceeded" in combined) or (
        "window.native.accessibilityobject.bounds" in combined and "recursion" in combined
    ):
        return {
            "code": "desktop_host_accessibility_recursion",
            "category": "desktop_host_runtime",
            "operator_cause": "Desktop host hit a pywebview/native accessibility recursion failure.",
            "operator_action": "Use browser fallback mode on this machine and disable desktop shell until environment is repaired.",
        }
    if any(token in combined for token in ["backend", "gtk", "qt", "cef", "edgechromium", "gui"]) and any(
        token in combined for token in ["unsupported", "invalid", "unavailable", "missing", "not found"]
    ):
        return {
            "code": "backend_selection_failure",
            "category": "backend_selection",
            "operator_cause": "pywebview backend selection is incompatible with this environment.",
            "operator_action": "Use a supported backend and verify backend dependencies are installed.",
        }
    if any(token in combined for token in ["no display", "headless", "x11", "wayland", "opengl"]):
        return {
            "code": "unsupported_environment",
            "category": "environment",
            "operator_cause": "Desktop UI backend is unavailable in the current environment/session.",
            "operator_action": "Run Atlas from an interactive desktop session with graphics/display support.",
        }
    if stage_text in {"ui_server_wait"} or any(
        token in combined for token in ["timed out", "timeout", "connection refused", "unable to connect", "create_connection"]
    ):
        return {
            "code": "startup_timing_failure",
            "category": "timing",
            "operator_cause": "UI backend endpoint was not ready in time for desktop host startup.",
            "operator_action": "Keep backend running and use browser fallback. Check local port readiness and startup latency.",
        }
    return {
        "code": "generic_desktop_init_failure",
        "category": "unknown",
        "operator_cause": "Desktop UI failed to initialize for an unknown reason.",
        "operator_action": "Inspect ui_startup_diagnostics.log and wake_spawn_logs frontend output.",
    }


def desktop_startup_preflight(preferred_gui: str = "") -> dict[str, object]:
    stage = "preflight"
    gui = str(preferred_gui or "").strip().lower()
    try:
        import webview  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "stage": "import_webview",
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "minimal_init_possible": False,
            "classification": classify_ui_startup_failure(
                stage="import_webview",
                error=str(exc),
                exception_type=type(exc).__name__,
            ),
        }

    allowed_guis = {"edgechromium", "cef", "qt", "gtk", "mshtml", "winforms"}
    if gui and gui not in allowed_guis:
        error = f"Unsupported explicit gui backend: {gui}"
        return {
            "ok": False,
            "stage": "backend_selection",
            "error": error,
            "exception_type": "ValueError",
            "minimal_init_possible": False,
            "classification": classify_ui_startup_failure(
                stage="backend_selection",
                error=error,
                exception_type="ValueError",
            ),
        }

    webview2_runtime_available = True
    webview2_detail = "not_checked"
    if os.name == "nt":
        webview2_runtime_available = False
        webview2_detail = "registry_check_failed"
        try:
            import winreg  # type: ignore

            runtime_paths = [
                (winreg.HKEY_CURRENT_USER, r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "pv"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "pv"),
                (winreg.HKEY_LOCAL_MACHINE, r"Software\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}", "pv"),
            ]
            for root, key, value_name in runtime_paths:
                try:
                    with winreg.OpenKey(root, key) as handle:
                        value, _ = winreg.QueryValueEx(handle, value_name)
                        if str(value or "").strip():
                            webview2_runtime_available = True
                            webview2_detail = f"detected:{value}"
                            break
                except Exception:
                    continue
        except Exception as exc:
            webview2_detail = f"registry_unavailable:{exc}"
    if os.name == "nt" and not webview2_runtime_available:
        error = f"WebView2 runtime not detected ({webview2_detail})"
        return {
            "ok": False,
            "stage": stage,
            "error": error,
            "exception_type": "RuntimeError",
            "minimal_init_possible": False,
            "webview2_runtime_available": False,
            "classification": classify_ui_startup_failure(
                stage=stage,
                error=error,
                exception_type="RuntimeError",
            ),
        }
    return {
        "ok": True,
        "stage": stage,
        "error": "",
        "exception_type": "",
        "minimal_init_possible": bool(hasattr(webview, "create_window") and hasattr(webview, "start")),
        "webview2_runtime_available": bool(webview2_runtime_available),
        "classification": {},
    }


def launch_desktop_ui(
    store,
    host: str = "127.0.0.1",
    port: int = 8765,
    comparison_root: str | None = None,
    comparison_run_id: str | None = None,
    server_already_running: bool = False,
) -> None:
    fallback_url = f"http://{host}:{port}"
    force_browser_only = str(os.getenv("ATLAS_UI_FORCE_BROWSER_ONLY", "")).strip().lower() in {"1", "true", "yes", "on"}
    preferred_gui = str(os.getenv("ATLAS_UI_WEBVIEW_GUI", "edgechromium" if os.name == "nt" else "")).strip().lower()
    failure_threshold_raw = str(os.getenv("ATLAS_UI_DESKTOP_FAILURE_THRESHOLD", "2")).strip()
    try:
        failure_threshold = max(1, int(failure_threshold_raw))
    except Exception:
        failure_threshold = 2
    stability_timeout_raw = str(os.getenv("ATLAS_UI_DESKTOP_STABILITY_TIMEOUT_SECONDS", "4")).strip()
    try:
        stability_timeout_seconds = max(2.0, min(15.0, float(stability_timeout_raw)))
    except Exception:
        stability_timeout_seconds = 4.0
    safe_native_window_default = os.name == "nt"
    safe_native_window = str(
        os.getenv("ATLAS_UI_DESKTOP_SAFE_NATIVE_WINDOW", "true" if safe_native_window_default else "false")
    ).strip().lower() in {"1", "true", "yes", "on"}

    def _truncate(value: str, limit: int = 4000) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
    fallback_opened = {"value": False}
    pywebview_error_times: list[float] = []
    native_lifecycle_entries: list[dict] = []
    wrapper_exit_reason = {"value": ""}

    def _detect_recent_desktop_instability() -> dict | None:
        try:
            base = Path(getattr(store, "base", Path.cwd() / "storage"))
            spawn_dir = base / "wake_spawn_logs"
            if spawn_dir.exists():
                candidates = sorted(spawn_dir.glob("frontend_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
                for path in candidates:
                    try:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    low = text.lower()
                    if "accessibilityobject" in low and "maximum recursion depth exceeded" in low:
                        return {
                            "classified_reason": "desktop_host_accessibility_recursion",
                            "source": str(path),
                            "matched_pattern": "accessibilityobject + maximum recursion depth exceeded",
                            "detected_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        }
            diag_log = base / "logs" / "ui_startup_diagnostics.log"
            if diag_log.exists():
                try:
                    lines = tail_text_lines(diag_log, 200)
                except Exception:
                    lines = []
                for line in reversed(lines):
                    low = line.lower()
                    if "class=desktop_host_accessibility_recursion" in low:
                        ts = line.split(" | ", 1)[0].strip() if " | " in line else ""
                        return {
                            "classified_reason": "desktop_host_accessibility_recursion",
                            "source": str(diag_log),
                            "matched_pattern": "ui_startup_diagnostics class=desktop_host_accessibility_recursion",
                            "detected_at": ts or datetime.fromtimestamp(diag_log.stat().st_mtime, timezone.utc).isoformat(),
                        }
        except Exception:
            return None
        return None

    def _append_startup_diag_log(payload: dict) -> None:
        try:
            base = Path(getattr(store, "base", Path.cwd() / "storage"))
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            line = (
                f"{payload.get('ts', '')} | ok={payload.get('ok')} | stage={payload.get('stage', '')} | "
                f"type={payload.get('exception_type', '')} | class={dict(payload.get('classification', {}) or {}).get('code', '')} | "
                f"error={payload.get('error', '')}"
            )
            with (log_dir / "ui_startup_diagnostics.log").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                trace_line = str(payload.get("traceback", "") or "").strip()
                if trace_line:
                    fh.write(f"traceback={trace_line}\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass

    def _record_native_lifecycle(
        stage: str,
        *,
        status: str = "",
        error: str = "",
        exc: BaseException | None = None,
        meta: dict | None = None,
        set_first_error: bool = False,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        exception_type = type(exc).__name__ if exc is not None else ""
        traceback_text = _truncate(traceback.format_exc() if exc is not None else "")
        payload = {
            "ts": ts,
            "stage": str(stage or ""),
            "status": str(status or ""),
            "error": _truncate(error or (str(exc) if exc is not None else "")),
            "exception_type": str(exception_type or ""),
            "traceback": traceback_text,
            "meta": dict(meta or {}),
        }
        native_lifecycle_entries.append(payload)
        try:
            flags = dict(store.load_runtime_flags() or {})
        except Exception:
            flags = {}
        existing = dict(flags.get("ui_desktop_native_lifecycle", {}) or {})
        history = list(existing.get("history", []) or [])
        history.append(payload)
        history = history[-40:]
        first_error = dict(existing.get("first_error", {}) or {})
        if set_first_error and not first_error:
            first_error = payload
        existing.update(
            {
                "preferred_gui": str(preferred_gui),
                "safe_native_window": bool(safe_native_window),
                "server_already_running": bool(server_already_running),
                "fallback_url": str(fallback_url),
                "wrapper_exit_reason": str(wrapper_exit_reason["value"] or ""),
                "last_stage": str(stage or ""),
                "last_status": str(status or ""),
                "last_updated_at": ts,
                "first_error": first_error,
                "history": history,
            }
        )
        try:
            flags["ui_desktop_native_lifecycle"] = existing
            store.save_runtime_flags(flags)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            store.log_event(
                "error" if payload["error"] else "system",
                "Desktop native lifecycle updated.",
                {"ui_desktop_native_lifecycle": payload},
            )
        logging.getLogger(__name__).info(
            "Desktop native lifecycle: stage=%s status=%s error=%s",
            payload["stage"],
            payload["status"],
            payload["error"],
        )

    def _record_ui_startup_state(
        *,
        ok: bool,
        stage: str,
        error: str = "",
        backend: str = "webview",
        exc: BaseException | None = None,
        fallback_active: bool = False,
        count_failure: bool = False,
        reset_failure_count: bool = False,
        evidence: dict | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        exception_type = type(exc).__name__ if exc is not None else ""
        traceback_text = _truncate(traceback.format_exc() if exc is not None else "")
        resolved_error = _truncate(error or (str(exc) if exc is not None else ""))
        classification = classify_ui_startup_failure(
            stage=stage,
            error=resolved_error,
            exception_type=exception_type,
            traceback_text=traceback_text,
        )
        evidence_payload = dict(evidence or {})
        override_reason = str(evidence_payload.get("classified_reason", "") or "").strip()
        if override_reason == "desktop_host_accessibility_recursion":
            classification = {
                "code": "desktop_host_accessibility_recursion",
                "category": "desktop_host_runtime",
                "operator_cause": "Desktop host hit a pywebview/native accessibility recursion failure.",
                "operator_action": "Use browser fallback mode on this machine and disable desktop shell until environment is repaired.",
            }
        try:
            flags = dict(store.load_runtime_flags() or {})
        except Exception:
            flags = {}
        previous = dict(flags.get("ui_desktop_window", {}) or {})
        previous_count = int(previous.get("failure_count", 0) or 0)
        if reset_failure_count:
            failure_count = 0
        elif count_failure:
            failure_count = previous_count + 1
        else:
            failure_count = previous_count
        disabled_due_to_repeated_failures = bool(failure_count >= failure_threshold)
        degraded_now = (not bool(ok)) or bool(fallback_active)
        previous_degraded_since = str(previous.get("degraded_since", "") or "")
        if degraded_now:
            degraded_since = previous_degraded_since or ts
        else:
            degraded_since = ""
        if bool(ok):
            classification = {
                "code": "",
                "category": "ok",
                "operator_cause": "",
                "operator_action": "",
            }
        payload = {
            "ok": bool(ok),
            "ready": bool(ok),  # backward-compatible health signal
            "stage": str(stage or ""),
            "backend": str(backend or ""),
            "error": resolved_error,
            "exception_type": str(exception_type or ""),
            "traceback": traceback_text,
            "classification": classification,
            "failure_reason": str(classification.get("code", "") or ""),
            "frontend_mode": "desktop" if bool(ok) else ("browser_fallback" if bool(fallback_active) else "desktop_unavailable"),
            "desktop_window_ready": bool(ok),
            "desktop_shell_healthy": bool(ok),
            "desktop_ui_available": bool(ok),
            "fallback_active": bool(fallback_active),
            "fallback_url": str(fallback_url),
            "degraded_since": degraded_since,
            "failure_evidence": evidence_payload,
            "preferred_gui": str(preferred_gui),
            "failure_count": int(failure_count),
            "failure_threshold": int(failure_threshold),
            "disabled_due_to_repeated_failures": bool(disabled_due_to_repeated_failures),
            "wrapper_pid": int(os.getpid()),
            "ts": ts,
            "updated_at": ts,
        }
        try:
            flags["ui_desktop_window"] = payload
            if str(payload.get("failure_reason", "") or "") == "desktop_host_accessibility_recursion":
                flags["desktop_host_stability"] = {
                    "accessibility_recursion_seen": True,
                    "force_browser_fallback": True,
                    "last_reason": "desktop_host_accessibility_recursion",
                    "last_detected_at": ts,
                    "recommended_frontend_mode": "browser_fallback",
                    "updated_at": ts,
                }
            elif bool(ok):
                prior_stability = dict(flags.get("desktop_host_stability", {}) or {})
                flags["desktop_host_stability"] = {
                    "accessibility_recursion_seen": bool(prior_stability.get("accessibility_recursion_seen", False)),
                    "force_browser_fallback": False,
                    "last_reason": str(prior_stability.get("last_reason", "") or ""),
                    "last_detected_at": str(prior_stability.get("last_detected_at", "") or ""),
                    "recommended_frontend_mode": "desktop",
                    "last_success_at": ts,
                    "updated_at": ts,
                }
            store.save_runtime_flags(flags)
            store.log_event(
                "system" if bool(ok) else "error",
                "Desktop UI startup state updated.",
                {"ui_desktop_window": payload},
            )
        except Exception:
            pass
        logging.getLogger(__name__).info(
            "UI startup diagnostic: ok=%s stage=%s exception_type=%s error=%s",
            bool(ok),
            str(stage or ""),
            str(exception_type or ""),
            str(payload.get("error", "")),
        )
        if payload["disabled_due_to_repeated_failures"]:
            logging.getLogger(__name__).warning(
                "Desktop UI host disabled due to repeated failures (count=%s, threshold=%s). Browser fallback active at %s",
                payload["failure_count"],
                payload["failure_threshold"],
                payload["fallback_url"],
            )
        _append_startup_diag_log(payload)

    def _activate_browser_fallback(*, stage: str, error: str, count_failure: bool, exc: BaseException | None = None, evidence: dict | None = None) -> None:
        evidence_payload = dict(evidence or {})
        evidence_payload.setdefault("fallback_triggered_reason", stage)
        evidence_payload.setdefault("fallback_activated_at", datetime.now(timezone.utc).isoformat())
        _record_ui_startup_state(
            ok=False,
            stage=stage,
            error=error,
            exc=exc,
            fallback_active=True,
            count_failure=count_failure,
            evidence=evidence_payload,
        )
        if not fallback_opened["value"]:
            fallback_opened["value"] = True
            webbrowser.open(url)

    url = f"http://{host}:{port}"
    _record_ui_startup_state(ok=False, stage="bootstrap")
    _record_native_lifecycle(
        "bootstrap",
        status="entered",
        meta={
            "host": str(host),
            "port": int(port),
            "preferred_gui": str(preferred_gui),
            "force_browser_only": bool(force_browser_only),
            "server_already_running": bool(server_already_running),
            "safe_native_window": bool(safe_native_window),
        },
    )
    thread: threading.Thread | None = None
    if not server_already_running:
        try:
            _record_native_lifecycle("create_ui_app", status="entered")
            app = create_ui_app(store, comparison_root=comparison_root, comparison_run_id=comparison_run_id)
            _record_native_lifecycle("create_ui_app", status="returned")
        except BaseException as exc:
            _record_ui_startup_state(ok=False, stage="create_ui_app", exc=exc)
            wrapper_exit_reason["value"] = "create_ui_app_exception"
            _record_native_lifecycle("create_ui_app", status="failed", exc=exc, set_first_error=True)
            raise

        def _serve() -> None:
            app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

        # Legacy combined mode: keep the Flask host on a non-daemon thread so
        # detached wake startups do not silently lose the dashboard API if the
        # desktop window backend exits.
        thread = threading.Thread(target=_serve, daemon=False)
        thread.start()
        _record_native_lifecycle("ui_server_thread_start", status="started")

    def _wait_for_server_owner() -> None:
        if thread is None:
            return
        while thread.is_alive():
            time.sleep(1)

    # Wait until the local UI server is actually ready before opening the window.
    for _ in range(40):
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    _record_ui_startup_state(ok=False, stage="ui_server_ready")
                    _record_native_lifecycle("ui_server_ready", status="ok")
                    break
        except Exception:
            time.sleep(0.25)
    else:
        _activate_browser_fallback(
            stage="ui_server_wait",
            error="ui_server_not_reachable",
            count_failure=True,
        )
        wrapper_exit_reason["value"] = "ui_server_not_reachable"
        _record_native_lifecycle("ui_server_wait", status="failed", error="ui_server_not_reachable", set_first_error=True)
        _wait_for_server_owner()
        return

    try:
        flags = dict(store.load_runtime_flags() or {})
    except Exception:
        flags = {}
    prior_ui_state = dict(flags.get("ui_desktop_window", {}) or {})
    desktop_host_stability = dict(flags.get("desktop_host_stability", {}) or {})
    recent_instability = _detect_recent_desktop_instability()
    if bool(prior_ui_state.get("disabled_due_to_repeated_failures", False)):
        _activate_browser_fallback(
            stage="desktop_bypassed_repeated_failures",
            error="desktop host disabled due to repeated startup failures",
            count_failure=False,
        )
        wrapper_exit_reason["value"] = "repeated_failures_bypass"
        _record_native_lifecycle("desktop_bypass", status="repeated_failures")
        _wait_for_server_owner()
        return
    if force_browser_only:
        _activate_browser_fallback(
            stage="desktop_bypassed_forced_browser_only",
            error="browser-only frontend mode forced by launcher due to known desktop instability",
            count_failure=False,
            evidence={
                "classified_reason": "desktop_host_accessibility_recursion",
                "source": "launcher_env",
                "matched_pattern": "ATLAS_UI_FORCE_BROWSER_ONLY",
                "detected_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        wrapper_exit_reason["value"] = "forced_browser_only"
        _record_native_lifecycle("desktop_bypass", status="forced_browser_only")
        _wait_for_server_owner()
        return
    if (
        str(prior_ui_state.get("failure_reason", "") or "") == "desktop_host_accessibility_recursion"
        or bool(desktop_host_stability.get("force_browser_fallback", False))
        or str(desktop_host_stability.get("last_reason", "") or "") == "desktop_host_accessibility_recursion"
    ):
        _record_native_lifecycle(
            "known_instability_detected",
            status="continuing_native_with_safe_mode",
            meta={
                "last_detected_at": str(desktop_host_stability.get("last_detected_at", "") or ""),
                "last_reason": str(desktop_host_stability.get("last_reason", "") or ""),
                "safe_native_window": bool(safe_native_window),
            },
        )
    if recent_instability:
        _record_native_lifecycle(
            "recent_instability_detected",
            status="continuing_native_with_safe_mode",
            meta=dict(recent_instability or {}),
        )

    preflight = desktop_startup_preflight(preferred_gui=preferred_gui)
    if not bool(preflight.get("ok", False)):
        _activate_browser_fallback(
            stage=str(preflight.get("stage", "preflight") or "preflight"),
            error=str(preflight.get("error", "") or ""),
            count_failure=True,
            evidence={
                "source": "desktop_startup_preflight",
                "classification": dict(preflight.get("classification", {}) or {}),
                "detected_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        wrapper_exit_reason["value"] = "preflight_failed"
        _record_native_lifecycle(
            str(preflight.get("stage", "preflight") or "preflight"),
            status="failed",
            error=str(preflight.get("error", "") or ""),
            meta={"classification": dict(preflight.get("classification", {}) or {})},
            set_first_error=True,
        )
        _wait_for_server_owner()
        return

    pywebview_logger = logging.getLogger("pywebview")
    fallback_watchdog = {"triggered": False}

    class _PywebviewRecursionHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                message = str(record.getMessage() or "")
            except Exception:
                message = ""
            if fallback_watchdog["triggered"]:
                return
            combined = message.lower()
            if "pywebview" in str(record.name or "").lower():
                pywebview_error_times.append(time.monotonic())
                while pywebview_error_times and (time.monotonic() - pywebview_error_times[0]) > 3.0:
                    pywebview_error_times.pop(0)
            matched_accessibility = (
                ("accessibilityobject" in combined and "maximum recursion depth exceeded" in combined)
                or ("window.native.accessibilityobject.bounds" in combined and "recursion" in combined)
            )
            repeated_pywebview_errors = len(pywebview_error_times) >= 3
            if matched_accessibility or repeated_pywebview_errors:
                fallback_watchdog["triggered"] = True
                wrapper_exit_reason["value"] = (
                    "accessibility_recursion_detected"
                    if matched_accessibility
                    else "repeated_pywebview_errors"
                )
                _record_native_lifecycle(
                    "first_error",
                    status="detected",
                    error=message or "repeated pywebview errors detected",
                    meta={
                        "matched_accessibility": bool(matched_accessibility),
                        "repeated_pywebview_errors": bool(repeated_pywebview_errors),
                        "log_record_name": str(record.name or ""),
                    },
                    set_first_error=True,
                )
                _activate_browser_fallback(
                    stage=(
                        "desktop_host_accessibility_recursion"
                        if matched_accessibility
                        else "desktop_host_repeated_pywebview_errors"
                    ),
                    error=message or "repeated pywebview errors detected",
                    count_failure=True,
                    evidence={
                        "source": "pywebview_logger",
                        "matched_pattern": (
                            "accessibilityobject + maximum recursion depth exceeded"
                            if matched_accessibility
                            else "repeated pywebview errors within 3 seconds"
                        ),
                        "detected_at": datetime.now(timezone.utc).isoformat(),
                        "log_record_name": str(record.name or ""),
                    },
                )

    persistent_watcher = _PywebviewRecursionHandler()
    pywebview_logger.addHandler(persistent_watcher)

    max_desktop_attempts = 2
    for desktop_attempt in range(1, max_desktop_attempts + 1):
        try:
            _record_ui_startup_state(ok=False, stage="import_webview")
            _record_native_lifecycle("import_webview", status="entered", meta={"desktop_attempt": int(desktop_attempt)})
            import webview
            _record_native_lifecycle(
                "import_webview",
                status="returned",
                meta={
                    "desktop_attempt": int(desktop_attempt),
                    "module_file": str(getattr(webview, "__file__", "") or ""),
                    "module_name": str(getattr(webview, "__name__", "webview") or "webview"),
                },
            )

            api = WindowControlAPI()
            _record_ui_startup_state(ok=False, stage="window_create")
            native_window_kwargs = {
                "width": 1320,
                "height": 860,
                "frameless": False if safe_native_window else True,
                "easy_drag": False if safe_native_window else True,
                "js_api": api,
            }
            _record_native_lifecycle(
                "entered_create_window",
                status="entered",
                meta={"desktop_attempt": int(desktop_attempt), "window_kwargs": dict(native_window_kwargs)},
            )
            window = webview.create_window(
                "Atlas AI Command Deck",
                url,
                **native_window_kwargs,
            )
            _record_native_lifecycle(
                "create_window_returned",
                status="returned",
                meta={
                    "desktop_attempt": int(desktop_attempt),
                    "window_type": type(window).__name__,
                    "has_maximize": bool(hasattr(window, "maximize")),
                    "has_destroy": bool(hasattr(window, "destroy")),
                },
            )
            api.attach(window)
            startup_marker = {"started": False}
            native_callback_marker = {"seen": False}
            start_loop_started_at = time.monotonic()

            def _maximize_on_start() -> None:
                try:
                    if not native_callback_marker["seen"]:
                        native_callback_marker["seen"] = True
                        _record_native_lifecycle("first_native_callback", status="entered")
                    if not startup_marker["started"]:
                        startup_marker["started"] = True
                        _record_ui_startup_state(ok=True, stage="started", reset_failure_count=True)
                        _record_native_lifecycle("first_native_callback", status="started")
                    if hasattr(window, "maximize"):
                        window.maximize()
                        api._maximized = True
                except Exception:
                    pass

            def _stability_watchdog() -> None:
                while not fallback_watchdog["triggered"]:
                    if startup_marker["started"]:
                        return
                    if (time.monotonic() - start_loop_started_at) >= stability_timeout_seconds:
                        fallback_watchdog["triggered"] = True
                        _activate_browser_fallback(
                            stage="desktop_loop_stability_timeout",
                            error="desktop window loop did not stabilize before timeout",
                            count_failure=True,
                            evidence={
                                "source": "desktop_stability_watchdog",
                                "detected_at": datetime.now(timezone.utc).isoformat(),
                                "timeout_seconds": stability_timeout_seconds,
                            },
                        )
                        return
                    time.sleep(0.2)

            _record_ui_startup_state(ok=False, stage="start_loop")
            start_kwargs = {"gui": preferred_gui} if preferred_gui else {}
            threading.Thread(target=_stability_watchdog, daemon=True).start()
            try:
                _record_native_lifecycle("entered_start", status="entered", meta={"start_kwargs": dict(start_kwargs)})
                webview.start(_maximize_on_start, **start_kwargs)
            finally:
                pass
            _record_native_lifecycle("start_returned", status="returned")
            if fallback_watchdog["triggered"]:
                _record_native_lifecycle("wrapper_exit_reason", status="returning", meta={"reason": str(wrapper_exit_reason["value"] or "")})
                _wait_for_server_owner()
                return
            if not startup_marker["started"]:
                _record_ui_startup_state(ok=True, stage="started", reset_failure_count=True)
            _record_ui_startup_state(ok=False, stage="start_loop_returned", error="window_loop_returned")
            wrapper_exit_reason["value"] = "start_loop_returned"
            _record_native_lifecycle("wrapper_exit_reason", status="returning", meta={"reason": "start_loop_returned"})
            _wait_for_server_owner()
            return
        except BaseException as exc:
            stage = "start_loop"
            message = str(exc)
            if "webview2" in message.lower() or "corewebview2" in message.lower():
                stage = "corewebview2_init_failure"
            classification = classify_ui_startup_failure(
                stage=stage,
                error=message,
                exception_type=type(exc).__name__,
                traceback_text=traceback.format_exc(),
            )
            retryable = (
                desktop_attempt < max_desktop_attempts
                and str(classification.get("code", "")) in {"corewebview2_init_failure", "startup_timing_failure"}
            )
            if retryable:
                wrapper_exit_reason["value"] = "retryable_native_exception"
                _record_ui_startup_state(
                    ok=False,
                    stage=stage,
                    exc=exc,
                    fallback_active=False,
                    count_failure=False,
                )
                _record_native_lifecycle(stage, status="retryable_failure", exc=exc, set_first_error=True)
                logging.getLogger(__name__).warning(
                    "Desktop startup attempt %s/%s failed (%s). Retrying once.",
                    desktop_attempt,
                    max_desktop_attempts,
                    classification.get("code", "unknown"),
                )
                time.sleep(0.8)
                continue
            _activate_browser_fallback(
                stage=stage,
                error=str(exc),
                exc=exc,
                count_failure=True,
                evidence={
                    "source": "desktop_exception",
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            wrapper_exit_reason["value"] = "native_exception"
            _record_native_lifecycle(stage, status="failed", exc=exc, set_first_error=True)
            if "destroy" in dir(locals().get("window", None) or object()):
                with contextlib.suppress(Exception):
                    _record_native_lifecycle("destroy_called", status="attempted")
                    locals()["window"].destroy()
            logging.getLogger(__name__).warning(
                "Desktop window backend unavailable, falling back to browser mode: %s",
                str(exc),
            )
            _record_native_lifecycle("wrapper_exit_reason", status="returning", meta={"reason": "native_exception"})
            _wait_for_server_owner()
            return
    # Defensive fallback path (should not be reached due to explicit returns above).
    _activate_browser_fallback(
        stage="generic_desktop_init_failure",
        error="desktop startup exhausted without success",
        count_failure=True,
    )
    wrapper_exit_reason["value"] = "generic_desktop_init_failure"
    _record_native_lifecycle("wrapper_exit_reason", status="returning", meta={"reason": "generic_desktop_init_failure"})
    _wait_for_server_owner()
