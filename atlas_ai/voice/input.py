from __future__ import annotations

import json
import queue
import warnings
from dataclasses import dataclass
from pathlib import Path

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from atlas_ai.atlas_vosk_setup import ensure_vosk_model
from atlas_ai.atlas_wake_config import AtlasWakeConfig

with warnings.catch_warnings():
    # TODO(py3.13): replace audioop.ratecv with a Python 3.13-safe resampler.
    warnings.filterwarnings(
        "ignore",
        message=".*audioop.*deprecated.*",
        category=DeprecationWarning,
    )
    import audioop  # type: ignore[import-not-found]


@dataclass
class SpeechInput:
    config: AtlasWakeConfig

    def __post_init__(self) -> None:
        model_path = Path(ensure_vosk_model())
        self._target_sample_rate = int(self.config.audio_sample_rate or 16000)
        self._device, self._sample_rate = self._resolve_device()
        self._resample_state = None
        self._model = Model(str(model_path))
        self._recognizer = KaldiRecognizer(self._model, float(self._target_sample_rate))
        self._recognizer.SetWords(False)
        self._queue: queue.Queue[bytes] = queue.Queue()
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self.config.audio_block_size,
                dtype="int16",
                channels=1,
                callback=self._callback,
                device=self._device,
            )
        except Exception:
            default_device = sd.default.device[0]
            info = sd.query_devices(default_device, "input")
            self._device = default_device
            self._sample_rate = int(float(info["default_samplerate"]))
            self._recognizer = KaldiRecognizer(self._model, float(self._target_sample_rate))
            self._recognizer.SetWords(False)
            self._stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self.config.audio_block_size,
                dtype="int16",
                channels=1,
                callback=self._callback,
                device=self._device,
            )

    def _resolve_device(self) -> tuple[str | int | None, int]:
        device = self.config.audio_device_name or None
        preferred_rate = int(self.config.audio_sample_rate or 16000)
        if device is None:
            default_device = sd.default.device[0]
            try:
                sd.check_input_settings(device=default_device, samplerate=preferred_rate, channels=1, dtype="int16")
                return default_device, preferred_rate
            except Exception:
                info = sd.query_devices(default_device, "input")
                return default_device, int(float(info["default_samplerate"]))
        try:
            sd.check_input_settings(device=device, samplerate=preferred_rate, channels=1, dtype="int16")
            return device, preferred_rate
        except Exception:
            info = sd.query_devices(device, "input")
            return device, int(float(info["default_samplerate"]))

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        self._queue.put(bytes(indata))

    def __enter__(self) -> "SpeechInput":
        self._stream.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self._stream.stop()
        self._stream.close()

    def next_recognition(self) -> tuple[str, float]:
        data = self._queue.get()
        if self._sample_rate != self._target_sample_rate:
            data, self._resample_state = audioop.ratecv(
                data,
                2,
                1,
                self._sample_rate,
                self._target_sample_rate,
                self._resample_state,
            )
        if self._recognizer.AcceptWaveform(data):
            payload = json.loads(self._recognizer.Result())
            text = str(payload.get("text", "") or "").strip()
            conf = 0.0
            words = list(payload.get("result", []) or [])
            if words:
                vals = []
                for row in words:
                    try:
                        vals.append(float(row.get("conf", 0.0) or 0.0))
                    except Exception:
                        continue
                if vals:
                    conf = max(0.0, min(1.0, sum(vals) / len(vals)))
            return text, conf
        return "", 0.0

    def next_text(self) -> str:
        text, _ = self.next_recognition()
        return text
