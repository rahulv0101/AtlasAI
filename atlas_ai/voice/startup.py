from __future__ import annotations

import ctypes
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from atlas_ai.voice.output import SpeechOutput


_lock = threading.Lock()
_has_announced_startup = False
_logger = logging.getLogger(__name__)


def _default_startup_mp3_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "storage" / "buckets" / "audio-lang" / "ATLAS_Greeting.mp3"


def _play_startup_mp3() -> bool:
    mp3_path = Path(str(os.getenv("ATLAS_STARTUP_MP3_PATH", "")).strip() or _default_startup_mp3_path())
    if not mp3_path.exists() or not mp3_path.is_file():
        return False

    try:
        import time
        import pygame

        pygame.mixer.init()
        pygame.mixer.music.load(str(mp3_path))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        return True
    except Exception:
        try:
            pygame.mixer.quit()  # type: ignore[name-defined]
        except Exception:
            pass

    try:
        from playsound import playsound

        playsound(str(mp3_path), block=True)
        return True
    except Exception:
        pass

    def _mci(command: str) -> int:
        return int(ctypes.windll.winmm.mciSendStringW(command, None, 0, None))

    alias = "atlas_startup_voice"
    try:
        _mci(f"close {alias}")
        open_cmd = f'open "{mp3_path}" type mpegvideo alias {alias}'
        if _mci(open_cmd) != 0:
            return False
        if _mci(f"play {alias} wait") != 0:
            _mci(f"close {alias}")
            return False
        _mci(f"close {alias}")
        return True
    except Exception:
        try:
            _mci(f"close {alias}")
        except Exception:
            pass
        return False


def _persist_startup_greeting(
    store: Any | None,
    *,
    status: str,
    backend_requested: str = "clone",
    backend_used: str = "",
    message: str = "",
    error: str = "",
) -> None:
    if store is None:
        return
    ts = datetime.now().isoformat()
    payload = {
        "status": str(status or ""),
        "backend_requested": str(backend_requested or ""),
        "backend_used": str(backend_used or ""),
        "message": str(message or ""),
        "error": str(error or ""),
        "ts": ts,
        "updated_at": ts,
    }
    try:
        def _mutator(current: dict[str, Any]) -> dict[str, Any]:
            flags = dict(current or {})
            flags["voice_startup_greeting"] = payload
            return flags

        store.update_runtime_flags(_mutator)
    except Exception:
        pass
    try:
        event_type = "system" if status in {"triggered", "success", "fallback_success", "skipped"} else "error"
        store.log_event(event_type, "Voice startup greeting updated.", {"voice_startup_greeting": payload})
    except Exception:
        pass


def announce_ui_ready_once(store: Any | None = None) -> bool:
    """Speak the startup line once per UI process lifetime.

    Returns True when announcement was spoken, False when skipped.
    """
    global _has_announced_startup
    with _lock:
        if _has_announced_startup:
            _persist_startup_greeting(store, status="skipped", message="startup_greeting_already_announced")
            return False
        _has_announced_startup = True
    _logger.info("Startup greeting triggered.")
    time_str = datetime.now().strftime("%H:%M")
    message = f"Hello sir, the time is {time_str}, and let's make the market respect us."
    _persist_startup_greeting(store, status="triggered", message=message)

    def _new_speaker_with_backend(backend: str) -> SpeechOutput:
        key = "ATLAS_TTS_BACKEND"
        previous = os.getenv(key)
        try:
            os.environ[key] = str(backend or "").strip()
            return SpeechOutput()
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def _announce_worker() -> None:
        try:
            clone_speaker = _new_speaker_with_backend("clone")
            clone_backend = str(getattr(clone_speaker, "_backend", "") or "").strip().lower()
            if clone_backend == "xtts_clone":
                _logger.info("Using cloned voice for startup greeting.")
                clone_speaker.speak(message)
                _logger.info("Startup greeting playback success.")
                _persist_startup_greeting(
                    store,
                    status="success",
                    backend_requested="clone",
                    backend_used=clone_backend,
                    message=message,
                )
                return
            _logger.warning(
                "Startup voice clone backend unavailable (backend=%s); falling back to existing speech chain.",
                clone_backend or "unknown",
            )
            _logger.info("Fallback voice used for startup greeting.")
            fallback_speaker = SpeechOutput()
            fallback_backend = str(getattr(fallback_speaker, "_backend", "") or "").strip().lower()
            fallback_speaker.speak(message)
            _logger.info("Startup greeting playback success.")
            _persist_startup_greeting(
                store,
                status="fallback_success",
                backend_requested="clone",
                backend_used=fallback_backend or clone_backend,
                message=message,
            )
        except Exception as exc:
            _logger.warning("Startup greeting playback failed; continuing startup. error=%s", exc)
            _persist_startup_greeting(
                store,
                status="failed",
                backend_requested="clone",
                backend_used="",
                message=message,
                error=str(exc),
            )

    threading.Thread(target=_announce_worker, daemon=True).start()
    return True
