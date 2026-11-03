from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import re
import socket
import tempfile
import threading
import uuid
import winsound
from datetime import datetime
from pathlib import Path


class SpeechOutput:
    def __init__(self) -> None:
        self.enabled = str(os.getenv("ATLAS_TTS_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self._backend_preference = str(os.getenv("ATLAS_TTS_BACKEND", "edge")).strip().lower()
        self._engine = None
        self._sapi_voice = None
        self._xtts_engine = None
        self._xtts_speaker_wav = None
        self._xtts_language = str(os.getenv("ATLAS_TTS_CLONE_LANGUAGE", "en")).strip() or "en"
        self._backend = "none"
        if self.enabled:
            try:
                if self._backend_preference in {"clone", "xtts_clone"} and self._init_xtts_clone():
                    self._backend = "xtts_clone"
                elif self._backend_preference == "edge" and self._init_edge():
                    self._backend = "edge"
                    # Keep a local spoken fallback ready if neural playback fails.
                    self._init_sapi()
                elif self._init_sapi():
                    self._backend = "sapi"
                else:
                    import pyttsx3

                    self._engine = pyttsx3.init()
                    rate = int(os.getenv("ATLAS_TTS_RATE", "175") or "175")
                    self._engine.setProperty("rate", rate)
                    volume = float(os.getenv("ATLAS_TTS_VOLUME", "1.0") or "1.0")
                    self._engine.setProperty("volume", max(0.0, min(1.0, volume)))
                    self._configure_voice()
                    self._backend = "pyttsx3"
            except Exception:
                self.enabled = False
                self._engine = None
                self._sapi_voice = None
                self._backend = "none"
        self._bridge_enabled = str(os.getenv("ATLAS_AUDIO_BRIDGE_SEND_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self._bridge_port = int(str(os.getenv("ATLAS_AUDIO_BRIDGE_PORT", "47291")).strip() or "47291")
        self._bridge_queue = queue.Queue()
        self._bridge_thread_started = False
        self._ensure_bridge_thread()

    def _ensure_bridge_thread(self) -> None:
        if not self._bridge_enabled or self._bridge_thread_started:
            return
        self._bridge_thread_started = True
        t = threading.Thread(target=self._bridge_sender_loop, daemon=True)
        t.start()

    def _bridge_sender_loop(self) -> None:
        while True:
            try:
                text = self._bridge_queue.get(timeout=1.0)
            except Exception:
                continue
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2.0)
                    s.connect(("127.0.0.1", self._bridge_port))
                    payload = {"type": "announce", "text": str(text or "")}
                    s.sendall(json.dumps(payload).encode("utf-8"))
            except Exception:
                continue

    def _init_xtts_clone(self) -> bool:
        sample_path = Path(
            str(
                os.getenv(
                    "ATLAS_TTS_CLONE_SAMPLE_PATH",
                    Path(__file__).resolve().parents[2] / "storage" / "buckets" / "audio-lang" / "ATLAS_Greeting.mp3",
                )
            ).strip()
        )
        if not sample_path.exists() or not sample_path.is_file():
            return False
        try:
            from TTS.api import TTS
        except Exception:
            return False
        try:
            model_name = str(os.getenv("ATLAS_TTS_CLONE_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")).strip()
            tts = TTS(model_name=model_name)
            if hasattr(tts, "to"):
                use_gpu = str(os.getenv("ATLAS_TTS_CLONE_USE_GPU", "false")).strip().lower() in {"1", "true", "yes", "on"}
                tts = tts.to("cuda" if use_gpu else "cpu")
            speaker_wav = sample_path
            if sample_path.suffix.lower() != ".wav":
                speaker_wav = self._ensure_wav_sample(sample_path)
                if speaker_wav is None:
                    return False
            self._xtts_engine = tts
            self._xtts_speaker_wav = str(speaker_wav)
            return True
        except Exception:
            self._xtts_engine = None
            self._xtts_speaker_wav = None
            return False

    def _ensure_wav_sample(self, sample_path: Path) -> Path | None:
        try:
            from pydub import AudioSegment
        except Exception:
            return None
        out_wav = Path(tempfile.gettempdir()) / f"atlas_clone_ref_{uuid.uuid4().hex}.wav"
        try:
            audio = AudioSegment.from_file(str(sample_path))
            audio = audio.set_channels(1).set_frame_rate(24000)
            audio.export(str(out_wav), format="wav")
            return out_wav
        except Exception:
            try:
                if out_wav.exists():
                    out_wav.unlink()
            except Exception:
                pass
            return None

    def _init_edge(self) -> bool:
        try:
            import edge_tts  # noqa: F401
            return True
        except Exception:
            return False

    def _init_sapi(self) -> bool:
        try:
            import win32com.client  # type: ignore
        except Exception:
            return False

        preferred_voice_id = str(os.getenv("ATLAS_TTS_VOICE_ID", "")).strip().lower()
        preferred_voice_hint = str(os.getenv("ATLAS_TTS_VOICE_HINT", "zira")).strip().lower()
        prefer_female = str(os.getenv("ATLAS_TTS_PREFER_FEMALE", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            rate = int(os.getenv("ATLAS_TTS_RATE", "170") or "170")
            volume = float(os.getenv("ATLAS_TTS_VOLUME", "1.0") or "1.0")
            voice.Rate = max(-10, min(10, int((rate - 170) / 10)))
            voice.Volume = int(max(0.0, min(1.0, volume)) * 100)
            tokens = list(voice.GetVoices())
            selected = None

            def token_identity(token: object) -> str:
                try:
                    return f"{token.GetDescription()} {token.Id}".lower()  # type: ignore[attr-defined]
                except Exception:
                    return ""

            for token in tokens:
                ident = token_identity(token)
                if preferred_voice_id and preferred_voice_id in ident:
                    selected = token
                    break
            if selected is None and preferred_voice_hint:
                for token in tokens:
                    if preferred_voice_hint in token_identity(token):
                        selected = token
                        break
            if selected is None and prefer_female:
                for token in tokens:
                    ident = token_identity(token)
                    if any(name in ident for name in ("zira", "female", "hazel", "susan", "aria", "eva")):
                        selected = token
                        break

            if selected is not None:
                voice.Voice = selected
            self._sapi_voice = voice
            return True
        except Exception:
            self._sapi_voice = None
            return False

    def _configure_voice(self) -> None:
        if self._engine is None:
            return

        preferred_voice_id = str(os.getenv("ATLAS_TTS_VOICE_ID", "")).strip()
        preferred_voice_hint = str(os.getenv("ATLAS_TTS_VOICE_HINT", "zira")).strip().lower()
        prefer_female = str(os.getenv("ATLAS_TTS_PREFER_FEMALE", "true")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        try:
            voices = list(self._engine.getProperty("voices") or [])
        except Exception:
            voices = []
        if not voices:
            return

        if preferred_voice_id:
            for voice in voices:
                if str(getattr(voice, "id", "")).strip().lower() == preferred_voice_id.lower():
                    self._engine.setProperty("voice", voice.id)
                    return

        if preferred_voice_hint:
            for voice in voices:
                identity = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                if preferred_voice_hint in identity:
                    self._engine.setProperty("voice", voice.id)
                    return

        if prefer_female:
            for voice in voices:
                identity = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')} {getattr(voice, 'gender', '')}".lower()
                if any(token in identity for token in ("female", "zira", "hazel", "susan", "aria", "eva")):
                    self._engine.setProperty("voice", voice.id)
                    return

    def speak(self, text: str) -> None:
        line = self._normalize_for_speech(text)
        if not line:
            return
        if self._bridge_enabled:
            try:
                self._bridge_queue.put_nowait(line)
                return
            except Exception:
                pass
        if not self.enabled:
            return
        try:
            if self._backend == "xtts_clone":
                try:
                    self._speak_xtts_clone(line)
                    return
                except Exception:
                    # Fallback to existing TTS chain if clone path is unavailable at runtime.
                    pass
            if self._backend == "edge":
                try:
                    self._speak_edge(line)
                    return
                except Exception:
                    pass
            if self._sapi_voice is not None:
                self._sapi_voice.Speak(line)
                return
            if self._engine is not None:
                self._engine.say(line)
                self._engine.runAndWait()
        except Exception:
            pass

    @staticmethod
    def _normalize_for_speech(text: str) -> str:
        line = str(text or "").strip()
        if not line:
            return ""
        # Remove label-heavy markers that sound awkward when spoken.
        line = re.sub(
            r"(?i)\b(status|reason|mode|run|evidence|constraint|safety|facts|inference|recommendation)\s*:\s*",
            "",
            line,
        )
        line = line.replace("|", ". ")
        line = line.replace("->", " to ")
        line = re.sub(r"\s+", " ", line).strip()
        # Keep spoken output concise.
        max_len = int(str(os.getenv("ATLAS_VOICE_MAX_CHARS", "240")) or "240")
        if len(line) > max_len:
            cut = line[:max_len].rsplit(" ", 1)[0].strip()
            line = cut or line[:max_len]
        return line

    def _speak_edge(self, text: str) -> None:
        voice = str(os.getenv("ATLAS_TTS_EDGE_VOICE", "en-US-AriaNeural")).strip() or "en-US-AriaNeural"
        rate = str(os.getenv("ATLAS_TTS_EDGE_RATE", "+0%")).strip() or "+0%"
        pitch = str(os.getenv("ATLAS_TTS_EDGE_PITCH", "+0Hz")).strip() or "+0Hz"
        volume = str(os.getenv("ATLAS_TTS_EDGE_VOLUME", "+0%")).strip() or "+0%"
        out_path = os.path.join(tempfile.gettempdir(), f"atlas_tts_{uuid.uuid4().hex}.wav")

        async def _render() -> None:
            import edge_tts

            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                pitch=pitch,
                volume=volume,
                output_format="riff-24khz-16bit-mono-pcm",
            )
            await communicate.save(out_path)

        try:
            asyncio.run(_render())
            with open(out_path, "rb") as fh:
                header = fh.read(12)
            # Guard against non-WAV payloads; fallback path in speak() will use SAPI/pyttsx.
            if not (header.startswith(b"RIFF") and header[8:12] == b"WAVE"):
                raise RuntimeError("edge_tts produced non-WAV payload")
            winsound.PlaySound(out_path, winsound.SND_FILENAME)
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

    def _speak_xtts_clone(self, text: str) -> None:
        if self._xtts_engine is None or not self._xtts_speaker_wav:
            raise RuntimeError("xtts clone backend not initialized")
        out_path = os.path.join(tempfile.gettempdir(), f"atlas_tts_clone_{uuid.uuid4().hex}.wav")
        try:
            self._xtts_engine.tts_to_file(
                text=text,
                speaker_wav=self._xtts_speaker_wav,
                language=self._xtts_language,
                file_path=out_path,
            )
            with open(out_path, "rb") as fh:
                header = fh.read(12)
            if not (header.startswith(b"RIFF") and header[8:12] == b"WAVE"):
                raise RuntimeError("xtts clone produced non-WAV payload")
            winsound.PlaySound(out_path, winsound.SND_FILENAME)
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

    def announce_startup(self) -> None:
        time_str = datetime.now().strftime("%H:%M")
        endings = ["Time to make the market respect us."]
        # Optional variation for non-startup conversational lines can be enabled separately.
        if str(os.getenv("ATLAS_STARTUP_LINE_VARIATION", "false")).strip().lower() in {"1", "true", "yes", "on"}:
            endings.extend(["Let's run this session with precision.", "Operator loop is ready."])
        message = f"Hello sir. The time is {time_str}. {random.choice(endings)}"
        self.speak(message)
