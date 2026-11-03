from __future__ import annotations

import argparse
import ctypes
import os
import re
import time
import threading
from difflib import SequenceMatcher
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas_ai.atlas_wake_config import AtlasWakeConfig, load_atlas_wake_config
from atlas_ai.brain.nlu import parse_user_language
from atlas_ai.commands.router import route_command
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.ui.ask_atlas import AskAtlasService
from atlas_ai.voice.input import SpeechInput
from atlas_ai.voice.output import SpeechOutput


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower()).strip()


def _mutex() -> object | None:
    if os.name != "nt":
        return None
    k32 = ctypes.windll.kernel32
    for name in ("Local\\AtlasAI_VoiceLoop", "AtlasAI_VoiceLoop"):
        k32.SetLastError(0)
        handle = k32.CreateMutexW(None, False, name)
        if not handle:
            continue
        if k32.GetLastError() == 183:
            return False
        return handle
    return None


@dataclass
class VoiceSource:
    store: KnowledgeStore

    def load_status(self) -> dict[str, Any]:
        return dict(self.store.load_status() or {})

    def load_events(self, limit: int = 160) -> list[dict[str, Any]]:
        return list(self.store.tail_events(limit=limit))

    def load_runtime_flags(self) -> dict[str, Any]:
        return dict(self.store.load_runtime_flags() or {})

    def load_comparison(self) -> dict[str, Any]:
        status = self.load_status()
        return dict(status.get("comparison", {}) or {})


def _update_voice_listener_state(store: KnowledgeStore, **fields: Any) -> dict[str, Any]:
    ts = datetime.now().isoformat()

    def _mutator(current: dict[str, Any]) -> dict[str, Any]:
        flags = dict(current or {})
        state = dict(flags.get("voice_listener_orchestration", {}) or {})
        state.update({k: v for k, v in fields.items()})
        state["updated_at"] = ts
        flags["voice_listener_orchestration"] = state
        return flags

    try:
        return dict(store.update_runtime_flags(_mutator) or {})
    except Exception:
        return {}


def _fuzzy_match(text: str, targets: set[str], threshold: float = 0.72) -> bool:
    n = _norm(text)
    if not n:
        return False
    if n in targets:
        return True
    for target in targets:
        if SequenceMatcher(None, n, target).ratio() >= threshold:
            return True
    return False


def _contains_phrase_variant(text: str, targets: set[str]) -> bool:
    n = _norm(text)
    if not n:
        return False
    for target in targets:
        t = _norm(target)
        if not t:
            continue
        if n == t:
            return True
        pattern = rf"(?:^|\s){re.escape(t)}(?:\s|$)"
        if re.search(pattern, n):
            return True
    return False


def _wake_phrase(text: str, *, confidence: float = 1.0, min_confidence: float = 0.72) -> bool:
    n = _norm(text)
    if not n:
        return False
    if float(confidence or 0.0) < float(min_confidence):
        return False
    wake_targets = {"atlas wake", "atlas wake up", "wake atlas", "wake up atlas"}
    # Exact/contains phrase still require minimum confidence.
    if _contains_phrase_variant(n, wake_targets):
        return True
    if _fuzzy_match(
        n,
        wake_targets,
        threshold=0.88,
    ):
        return True
    words = set(n.split())
    atlas_like = any(
        SequenceMatcher(None, token, "atlas").ratio() >= 0.72
        for token in words
    )
    if atlas_like and ("wake" in words or "awake" in words):
        return True
    return False


def _recent_voice_window(parts: list[tuple[float, str]], *, now: float, seconds: float = 3.0) -> str:
    floor = float(now) - max(0.5, float(seconds))
    recent = [text for ts, text in parts if float(ts) >= floor and _norm(text)]
    return _norm(" ".join(recent[-4:]))


def _sleep_phrase(text: str, *, allow_short_form: bool = False, confidence: float = 1.0, min_confidence: float = 0.78) -> bool:
    n = _norm(text)
    sleep_targets = {"atlas sleep", "sleep atlas", "go to sleep atlas"}
    sleep_mishear_targets = {
        # Vosk commonly hears "Atlas sleep" as this phrase on this setup.
        "at least leap",
    }
    if float(confidence or 0.0) < float(min_confidence):
        return False
    # Exact/contains phrase still require minimum confidence.
    if _contains_phrase_variant(n, sleep_targets):
        return True
    if n in sleep_mishear_targets:
        return True
    if allow_short_form and n == "sleep":
        return True
    words = set(n.split())
    atlas_like = any(
        SequenceMatcher(None, token, "atlas").ratio() >= 0.72
        for token in words
    )
    if atlas_like and ("sleep" in words or "offline" in words):
        return True
    # Keep sleep matching strict to avoid false-offline triggers from noisy substrings.
    return False


def _spoken_summary(reply: dict[str, Any]) -> str:
    result = str(reply.get("result", "") or "").strip()
    if not result:
        return "No grounded answer available right now."
    first = result.splitlines()[0].strip()
    if first.startswith("{") and first.endswith("}"):
        return "I have a grounded update, but please ask again more specifically."
    first = re.sub(r"^[A-Z_ ]+:\s*", "", first)
    first = first.replace("|", ", ").replace(" - ", ", ")
    first = re.sub(r"[{}]", "", first)
    return first[:220]


def _is_unclear_input(text: str) -> bool:
    n = _norm(text)
    if not n:
        return True
    unclear_tokens = {"uh", "um", "hmm", "noise", "unknown", "static"}
    parts = n.split()
    if len(parts) <= 1 and parts[0] in unclear_tokens:
        return True
    if len(parts) == 1 and len(parts[0]) <= 2:
        return True
    return False


def _voice_session_turn_limit() -> int:
    raw = str(os.getenv("ATLAS_VOICE_SESSION_TURN_LIMIT", "1")).strip()
    try:
        return max(1, min(10, int(raw)))
    except Exception:
        return 1


def _should_execute_sleep_shutdown(
    heard_text: str,
    *,
    awake: bool,
    pending_since: float | None,
    confidence: float = 1.0,
    now: float | None = None,
) -> tuple[bool, bool]:
    normalized = _norm(heard_text)
    explicit_sleep = _contains_phrase_variant(
        normalized,
        {"atlas sleep", "sleep atlas", "go to sleep atlas", "at least leap"},
    )
    # Explicit "atlas sleep" style phrases execute immediately for expected
    # operator UX; only ambiguous short-form sleep uses a confirmation arm.
    if explicit_sleep:
        return True, False
    if not _sleep_phrase(heard_text, allow_short_form=awake, confidence=confidence):
        return False, False
    # Require explicit second confirmation to avoid accidental short-form shutdown.
    if pending_since is None:
        return False, True
    now_ts = float(now if now is not None else time.monotonic())
    # Confirmation window is intentionally short and bounded.
    if (now_ts - float(pending_since)) <= 10.0:
        return True, False
    return False, True


def run_voice_loop(config: AtlasWakeConfig | None = None) -> int:
    config = config or load_atlas_wake_config()
    gate = _mutex()
    if gate is False:
        print("Atlas voice loop already running.", flush=True)
        return 0

    store = KnowledgeStore(str(Path(config.project_root) / "storage"))
    source = VoiceSource(store=store)
    ask = AskAtlasService(source)
    speaker = SpeechOutput()
    voice_responses_enabled = str(os.getenv("ATLAS_VOICE_RESPONSES_ENABLED", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    awake = False
    remaining_turns = 0
    wake_starts_stack = str(os.getenv("ATLAS_WAKE_STARTS_STACK", "true")).strip().lower() in {"1", "true", "yes", "on"}
    sleep_stops_stack = str(os.getenv("ATLAS_SLEEP_STOPS_STACK", "true")).strip().lower() in {"1", "true", "yes", "on"}
    sleep_stops_listener = str(os.getenv("ATLAS_SLEEP_STOPS_LISTENER", "false")).strip().lower() in {"1", "true", "yes", "on"}
    log_path = Path(os.getenv("ATLAS_VOICE_LOG_PATH", str(Path(config.project_root) / "storage" / "voice_listener.log")))
    startup_line = str(os.getenv("ATLAS_VOICE_STARTUP_LINE", "false")).strip().lower() in {"1", "true", "yes", "on"}
    echo_guard_seconds = float(str(os.getenv("ATLAS_VOICE_ECHO_GUARD_SECONDS", "2.8")).strip() or "2.8")
    ptt_enabled = str(os.getenv("ATLAS_PTT_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
    ptt_armed = not ptt_enabled
    suppress_until = 0.0
    wake_debounce_seconds = float(str(os.getenv("ATLAS_WAKE_DEBOUNCE_SECONDS", "1.8")).strip() or "1.8")
    sleep_debounce_seconds = float(str(os.getenv("ATLAS_SLEEP_DEBOUNCE_SECONDS", "1.8")).strip() or "1.8")
    wake_conf_threshold = float(str(os.getenv("ATLAS_VOICE_WAKE_CONFIDENCE_MIN", "0.72")).strip() or "0.72")
    sleep_conf_threshold = float(str(os.getenv("ATLAS_VOICE_SLEEP_CONFIDENCE_MIN", "0.78")).strip() or "0.78")
    last_wake_trigger_at = 0.0
    last_sleep_trigger_at = 0.0
    launch_thread_ref: dict[str, threading.Thread | None] = {"thread": None}
    recent_voice_parts: list[tuple[float, str]] = []
    stack_state = "UNKNOWN"
    def _log(line: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"{stamp} | {line}"
        try:
            print(msg, flush=True)
        except OSError:
            # Detached watchdog launches can leave stdout/stderr unusable on Windows.
            pass
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except Exception:
            pass
    def _log_stack_state(state: str, detail: str = "") -> None:
        nonlocal stack_state
        stack_state = str(state or "UNKNOWN").upper()
        suffix = f" | {detail}" if detail else ""
        _log(f"STACK: {stack_state}{suffix}")

    _log("Atlas voice loop active. Say 'Atlas wake' to activate.")
    _log(f"Voice session turn limit={_voice_session_turn_limit()}. Say 'Atlas sleep' anytime to shut everything down.")
    _update_voice_listener_state(
        store,
        listener_path="voice.wake_loop",
        listener_status="active",
        launch_dispatch_state="idle",
        launch_thread_alive=False,
        voice_session_turn_limit=_voice_session_turn_limit(),
        sleep_confirmation_window_seconds=0.0,
    )
    if ptt_enabled:
        _log("Push-to-talk mode enabled. Say 'Atlas listen' before each command.")
    if startup_line and voice_responses_enabled:
        speaker.speak("Hello sir. Voice listener online.")
        suppress_until = time.monotonic() + echo_guard_seconds
    sleep_confirmation_pending_since: float | None = None
    with SpeechInput(config) as mic:
        _log(f"Microphone ready: device={getattr(mic, '_device', 'unknown')} sample_rate={getattr(mic, '_sample_rate', 'unknown')}")
        while True:
            text, text_conf = mic.next_recognition()
            if not text:
                continue
            if time.monotonic() < suppress_until:
                continue
            heard = _norm(text)
            if not heard:
                continue
            now_ts = time.monotonic()
            recent_voice_parts.append((now_ts, heard))
            recent_voice_parts[:] = [(ts, part) for ts, part in recent_voice_parts if now_ts - ts <= 3.0]
            heard_window = _recent_voice_window(recent_voice_parts, now=now_ts)
            _log(f"Heard: {text}")
            nlu = parse_user_language(text)
            voice_state_update = {
                "last_detected_transcript": text,
                "last_detected_transcript_normalized": heard,
                "last_detected_intent": str(nlu.intent),
                "last_detected_at": datetime.now().isoformat(),
            }
            if nlu.intent != "system_command":
                voice_state_update.update(
                    {
                        "matched_command": "",
                        "confirmation_required": False,
                        "confirmation_satisfied": False,
                        "wake_phrase_matched": False,
                        "wake_confidence": float(text_conf),
                    }
                )
            _update_voice_listener_state(
                store,
                **voice_state_update,
            )

            if ptt_enabled and _norm(heard) in {"atlas listen", "listen atlas", "atlas ready"}:
                ptt_armed = True
                if voice_responses_enabled:
                    speaker.speak("Listening.")
                    suppress_until = time.monotonic() + min(1.2, echo_guard_seconds)
                continue

            wake_phrase_matched = _wake_phrase(heard, confidence=text_conf, min_confidence=wake_conf_threshold) or _wake_phrase(
                heard_window,
                confidence=text_conf,
                min_confidence=wake_conf_threshold,
            )
            if (nlu.intent == "system_command" and ("wake" in str(nlu.normalized_text) or "start the full system" in str(nlu.normalized_text))) or wake_phrase_matched:
                if (now_ts - last_wake_trigger_at) < wake_debounce_seconds:
                    continue
                last_wake_trigger_at = now_ts
                awake = True
                remaining_turns = _voice_session_turn_limit()
                _update_voice_listener_state(
                    store,
                    matched_command="start_system",
                    confirmation_required=False,
                    confirmation_satisfied=True,
                    startup_requested_at=datetime.now().isoformat(),
                    startup_source="voice.wake_loop",
                    launch_dispatch_state="queued",
                    wake_phrase_matched=True,
                    wake_confidence=float(text_conf),
                )
                if wake_starts_stack:
                    active_thread = launch_thread_ref.get("thread")
                    if active_thread is not None and active_thread.is_alive():
                        _log("Wake trigger ignored: launch already in progress from listener thread.")
                        if voice_responses_enabled:
                            speaker.speak("Startup already in progress.")
                            suppress_until = time.monotonic() + min(1.4, echo_guard_seconds)
                        continue

                    def _start_stack_async() -> None:
                        try:
                            _log("Wake command accepted. Launch command dispatch started.")
                            _update_voice_listener_state(
                                store,
                                launch_dispatch_state="running",
                                launch_dispatched_at=datetime.now().isoformat(),
                            )
                            result = route_command("start system")
                            _log(f"Start system command result: {result}")
                            summary = dict(result.get("summary", {}) or {})
                            health = dict(summary.get("health", {}) or {})
                            desktop_health = dict(health.get("desktop_shell_health", {}) or {})
                            _update_voice_listener_state(
                                store,
                                launch_dispatch_state="completed",
                                final_startup_verdict=str(result.get("overall_result", "") or health.get("overall_result", "")),
                                final_startup_reason=str(health.get("overall_reason", "") or result.get("reason", "")),
                                browser_fallback_used=bool(desktop_health.get("fallback_active", False)),
                                fallback_reason=str(
                                    desktop_health.get("failure_reason", "")
                                    or health.get("degraded_reason", "")
                                    or ""
                                ),
                                launch_completed_at=datetime.now().isoformat(),
                            )
                            verdict = str(result.get("overall_result", "") or health.get("overall_result", "")).upper()
                            if verdict == "SUCCESS":
                                _log_stack_state("RUNNING", str(health.get("overall_reason", "") or "wake_ok"))
                            else:
                                _log_stack_state("FAILED", str(health.get("overall_reason", "") or result.get("reason", "") or "wake_failed"))
                        except Exception as exc:
                            _log(f"Start system command failed: {exc}")
                            _update_voice_listener_state(
                                store,
                                launch_dispatch_state="failed",
                                final_startup_verdict="FAILED",
                                final_startup_reason=str(exc),
                                launch_completed_at=datetime.now().isoformat(),
                            )
                            _log_stack_state("FAILED", str(exc))
                        finally:
                            _update_voice_listener_state(
                                store,
                                launch_thread_alive=False,
                                launch_thread_finished_at=datetime.now().isoformat(),
                            )

                    launch_thread = threading.Thread(target=_start_stack_async, daemon=True)
                    launch_thread_ref["thread"] = launch_thread
                    launch_thread.start()
                if voice_responses_enabled:
                    speaker.speak("Atlas wake acknowledged.")
                    suppress_until = time.monotonic() + echo_guard_seconds
                continue

            execute_sleep, arm_sleep_confirmation = _should_execute_sleep_shutdown(
                heard,
                awake=awake,
                pending_since=sleep_confirmation_pending_since,
                confidence=text_conf,
                now=now_ts,
            )
            if (
                not execute_sleep
                and nlu.intent == "system_command"
                and ("sleep" in str(nlu.normalized_text) or "shut down" in str(nlu.normalized_text))
            ):
                if sleep_confirmation_pending_since is None:
                    execute_sleep = False
                    arm_sleep_confirmation = True
                elif (now_ts - float(sleep_confirmation_pending_since)) <= 10.0:
                    execute_sleep = True
                    arm_sleep_confirmation = False
            if arm_sleep_confirmation:
                _log("Sleep command heard once. Awaiting confirmation before shutdown.")
                sleep_confirmation_pending_since = now_ts
                _update_voice_listener_state(
                    store,
                    matched_command="close_system",
                    confirmation_required=True,
                    confirmation_satisfied=False,
                    shutdown_requested_at="",
                    sleep_confirmation_pending_at=datetime.now().isoformat(),
                )
                if voice_responses_enabled:
                    speaker.speak("Please confirm sleep.")
                    suppress_until = time.monotonic() + min(1.4, echo_guard_seconds)
                continue

            if execute_sleep:
                if (now_ts - last_sleep_trigger_at) < sleep_debounce_seconds:
                    continue
                last_sleep_trigger_at = now_ts
                sleep_confirmation_pending_since = None
                _update_voice_listener_state(
                    store,
                    matched_command="close_system",
                    confirmation_required=False,
                    confirmation_satisfied=True,
                    shutdown_requested_at=datetime.now().isoformat(),
                    listener_preserve_requested=True,
                    sleep_confidence=float(text_conf),
                )
                if sleep_stops_stack:
                    try:
                        result = route_command("atlas sleep")
                        if not bool(result.get("ok", False)):
                            _log(f"Close system command blocked: {result}")
                            _update_voice_listener_state(
                                store,
                                shutdown_result="blocked",
                                shutdown_reason=str(result.get("reason", "") or ""),
                                listener_preserved=bool(result.get("listener_preserved", True)),
                            )
                        else:
                            _log(f"Close system command result: {result}")
                            _update_voice_listener_state(
                                store,
                                shutdown_result=str(result.get("overall_result", "") or "ok"),
                                shutdown_reason="",
                                listener_preserved=bool(result.get("listener_preserved", True)),
                            )
                            overall = str(result.get("overall_result", "") or "").upper()
                            if "STOPPED" in overall:
                                _log_stack_state("SLEEPING", overall)
                            else:
                                _log_stack_state("FAILED", overall or "sleep_unknown")
                    except Exception as exc:
                        _log(f"Close system command failed: {exc}")
                        _update_voice_listener_state(
                            store,
                            shutdown_result="failed",
                            shutdown_reason=str(exc),
                            listener_preserved=not sleep_stops_listener,
                        )
                        _log_stack_state("FAILED", str(exc))
                if voice_responses_enabled:
                    speaker.speak("Atlas stack sleeping. Listener remains online.")
                    suppress_until = time.monotonic() + echo_guard_seconds
                awake = False
                remaining_turns = 0
                if sleep_stops_listener:
                    _log("Sleep command received. Stopping voice loop.")
                    break
                continue

            if not awake:
                continue

            if ptt_enabled and not ptt_armed:
                continue

            if _is_unclear_input(heard):
                if voice_responses_enabled:
                    speaker.speak("Please repeat.")
                    suppress_until = time.monotonic() + min(1.5, echo_guard_seconds)
                if ptt_enabled:
                    ptt_armed = False
                continue

            cmd = route_command(heard)
            if bool(cmd.get("ok", False)):
                if voice_responses_enabled:
                    if cmd.get("command") == "start_system":
                        speaker.speak("System start requested.")
                        suppress_until = time.monotonic() + echo_guard_seconds
                    elif cmd.get("command") == "close_system":
                        speaker.speak("System close requested.")
                        suppress_until = time.monotonic() + echo_guard_seconds
                remaining_turns = max(0, remaining_turns - 1)
                if remaining_turns == 0:
                    awake = False
                    _log("Voice session completed. Waiting for next wake phrase.")
                if ptt_enabled:
                    ptt_armed = False
                continue

            reply = ask.answer(text, mode="voice")
            if voice_responses_enabled:
                speaker.speak(_spoken_summary(reply))
                suppress_until = time.monotonic() + echo_guard_seconds
            if ptt_enabled:
                ptt_armed = False
            _log(f"Reply: {reply.get('result', '')}")
            remaining_turns = max(0, remaining_turns - 1)
            if remaining_turns == 0:
                awake = False
                _log("Voice session completed. Waiting for next wake phrase.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atlas voice wake/sleep loop with STT/TTS.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_args(argv)
    return run_voice_loop()


if __name__ == "__main__":
    raise SystemExit(main())
