from __future__ import annotations

import json
import logging
import os
import queue
import re
import socket
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from atlas_ai.atlas_wake_config import load_atlas_wake_config
from atlas_ai.commands.router import route_command
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.voice.input import SpeechInput
from atlas_ai.voice.output import SpeechOutput


class RuntimeAPIClient:
    def __init__(self, store: KnowledgeStore):
        self.store = store
        self.ui_base = str(os.getenv("ATLAS_UI_BASE", "http://127.0.0.1:8765")).strip()

    def read_flags(self) -> dict[str, Any]:
        return dict(self.store.load_runtime_flags() or {})

    def write_flag(self, key: str, value: Any) -> bool:
        try:
            flags = dict(self.store.load_runtime_flags() or {})
            flags[key] = value
            self.store.save_runtime_flags(flags)
            return True
        except Exception:
            return False

    def read_status(self) -> dict[str, Any]:
        return dict(self.store.load_status() or {})

    def get_ui(self, endpoint: str) -> dict[str, Any]:
        try:
            resp = requests.get(f"{self.ui_base}{endpoint}", timeout=3)
            if int(resp.status_code) == 200:
                return dict(resp.json() or {})
        except Exception:
            return {}
        return {}

    def post_ui(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = requests.post(f"{self.ui_base}{endpoint}", json=payload, timeout=3)
            if int(resp.status_code) == 200:
                return dict(resp.json() or {})
        except Exception:
            return {}
        return {}


class VoiceCommandRouter:
    COMMAND_MAP = {
        "status": ["status", "what's happening", "update", "how are we doing"],
        "sleep": ["atlas sleep", "sleep atlas", "go to sleep atlas", "close system", "shut down"],
        "pause": ["pause", "stop trading", "halt", "freeze"],
        "resume": ["resume", "start trading", "go", "unpause"],
        "kill_switch": ["kill switch", "kill", "emergency", "stop everything", "abort"],
        "close_all": ["close all", "close everything", "exit all", "flatten"],
        "report": ["report", "summary", "how did we do", "today's results"],
        "whats_open": ["what's open", "open trades", "open positions", "what do we have"],
        "help": ["help", "commands", "what can you do"],
    }

    def __init__(self, api: RuntimeAPIClient, speak_fn, log_fn) -> None:
        self.api = api
        self.speak = speak_fn
        self.log = log_fn
        self.pending_confirmation: dict[str, Any] = {}

    def _wait_for_confirmation(self, expected: str = "atlas confirm", timeout_seconds: float = 10.0) -> bool:
        self.pending_confirmation = {"expected": expected, "expires_at": time.time() + timeout_seconds}
        return False

    def consume_confirmation(self, heard: str) -> bool:
        if not self.pending_confirmation:
            return False
        if time.time() > float(self.pending_confirmation.get("expires_at", 0.0)):
            self.pending_confirmation = {}
            return False
        expected = str(self.pending_confirmation.get("expected", "atlas confirm"))
        if str(heard or "").strip().lower() == expected:
            if str(self.pending_confirmation.get("next", "")) == "activate_live" and expected == "atlas live confirmed":
                self.api.write_flag("mode", "live")
                self.api.write_flag("execute_trades", True)
                self.api.write_flag("live_mode_activated_at", datetime.now(timezone.utc).isoformat())
                self.api.write_flag("live_mode_activated_by", "voice_command")
                self.speak("LIVE mode is now active. Real money trading enabled. Stay alert.")
            self.pending_confirmation = {}
            return True
        return False

    def route(self, raw_text: str) -> None:
        text = str(raw_text or "").lower().strip()
        if self.consume_confirmation(text):
            self.speak("Confirmation received.")
            return
        if self.finalize_live_mode_if_confirmed(text):
            return
        risk_match = re.search(r"risk\s+(\d+\.?\d*)", text)
        if risk_match:
            self.cmd_set_risk(float(risk_match.group(1)))
            return
        if "mode live" in text:
            self.cmd_mode_live()
            return
        if "mode demo" in text:
            self.cmd_mode_demo()
            return
        for command, phrases in self.COMMAND_MAP.items():
            if any(phrase in text for phrase in phrases):
                getattr(self, f"cmd_{command}")()
                return
        self.speak("Command not recognised. Say atlas help for available commands.")
        self.log(logging.INFO, f"Unrecognised voice command: {raw_text}")

    def cmd_status(self) -> None:
        s = self.api.read_status()
        f = self.api.read_flags()
        risk = dict(s.get("risk", {}) or {})
        session = str(dict(s.get("context_snapshot", {}) or {}).get("session", "unknown"))
        regime = str(dict(s.get("market_regime", {}) or {}).get("regime", "unknown"))
        pnl = float(risk.get("daily_profit_zar", 0.0) or 0.0)
        pnl_pct = float(risk.get("daily_loss_pct", 0.0) or 0.0) * -1.0 if pnl < 0 else 0.0
        open_trades = int(risk.get("open_positions", 0) or 0)
        executing = bool(f.get("execute_trades", False))
        mode = "live" if executing else "paused"
        self.speak(f"Atlas status. Mode: {mode}. Session: {session}. Regime: {regime}. Open trades: {open_trades}. Today's P and L: {pnl:.2f} dollars, {pnl_pct:.1f} percent.")

    def cmd_sleep(self) -> None:
        result = route_command("atlas sleep")
        if bool(result.get("ok", False)):
            self.speak("Atlas stack sleeping. Listener remains online.")
            return
        self.speak("Sleep command was blocked.")

    def cmd_pause(self) -> None:
        self.api.write_flag("execute_trades", False)
        self.speak("Trading paused. Atlas will not open new positions.")

    def cmd_resume(self) -> None:
        self.api.write_flag("execute_trades", True)
        self.speak("Trading resumed. Atlas is scanning for opportunities.")

    def cmd_kill_switch(self) -> None:
        self.api.write_flag("execute_trades", False)
        self.api.write_flag("kill_switch_active", True)
        self.api.write_flag("kill_switch_reason", "voice_command")
        self.api.write_flag("kill_switch_time", datetime.now(timezone.utc).isoformat())
        self.api.post_ui("/api/ask", {"question": "activate kill switch"})
        self.speak("Kill switch activated. All trading halted.")

    def cmd_close_all(self) -> None:
        result = self.api.post_ui("/api/ask", {"question": "close all positions now"})
        self.speak(f"Close all sent. {result.get('result', 'request accepted')}.")

    def cmd_report(self) -> None:
        s = self.api.read_status()
        risk = dict(s.get("risk", {}) or {})
        trades_today = int(risk.get("trades_today", 0) or 0)
        pnl = float(risk.get("daily_profit_zar", 0.0) or 0.0)
        session = str(dict(s.get("context_snapshot", {}) or {}).get("session", "unknown"))
        regime = str(dict(s.get("market_regime", {}) or {}).get("regime", "unknown"))
        self.speak(f"Today's report. {trades_today} trades taken. Total P and L: {pnl:.2f} dollars. Current session: {session}. Market regime: {regime}.")

    def cmd_whats_open(self) -> None:
        s = self.api.read_status()
        risk = dict(s.get("risk", {}) or {})
        open_positions = int(risk.get("open_positions", 0) or 0)
        if open_positions <= 0:
            self.speak("No open positions.")
            return
        self.speak(f"{open_positions} open positions.")

    def cmd_set_risk(self, number: float) -> None:
        if number < 0.1 or number > 3.0:
            self.speak(f"Risk {number} percent is outside safe range.")
            return
        self.api.write_flag("risk_per_trade_pct", float(number))
        self.speak(f"Risk per trade set to {number} percent.")

    def cmd_mode_demo(self) -> None:
        self.api.write_flag("mode", "demo")
        self.api.write_flag("execute_trades", True)
        self.speak("Demo mode active.")

    def cmd_mode_live(self) -> None:
        self.speak("WARNING. You are switching to LIVE mode with real money. Say atlas confirm to continue.")
        self.pending_confirmation = {"expected": "atlas confirm", "expires_at": time.time() + 10.0, "next": "live_phrase"}

    def finalize_live_mode_if_confirmed(self, heard: str) -> bool:
        if not self.pending_confirmation:
            return False
        if str(self.pending_confirmation.get("next", "")) != "live_phrase":
            return False
        if str(heard or "").strip().lower() != "atlas confirm":
            return False
        self.speak("Final confirmation required. Say atlas live confirmed. You have 10 seconds.")
        self.pending_confirmation = {"expected": "atlas live confirmed", "expires_at": time.time() + 10.0, "next": "activate_live"}
        return True

    def cmd_help(self) -> None:
        self.speak("Available commands. Atlas status. Atlas pause. Atlas resume. Atlas kill switch. Atlas close all. Atlas report. Atlas what's open. Atlas risk number. Atlas mode demo. Atlas mode live. Atlas help.")


class AudioListenerBridge:
    def __init__(self) -> None:
        self.config = load_atlas_wake_config()
        self.store = KnowledgeStore(str(Path(self.config.project_root) / "storage"))
        self.api = RuntimeAPIClient(self.store)
        self.socket_port = 47291
        self.running = True
        self.voice_enabled = str(os.getenv("VOICE_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.wake_word = str(os.getenv("WAKE_WORD", "atlas")).strip().lower()
        self.wake_word_sensitivity = float(str(os.getenv("WAKE_WORD_SENSITIVITY", "0.5")).strip() or "0.5")
        self._speaker = SpeechOutput()
        self._announce_q: queue.Queue[str] = queue.Queue()
        self._crashes: list[float] = []
        self._last_mic_name = "default microphone"
        self._router = VoiceCommandRouter(self.api, self.speak, self._log)

    def _configure_logging(self) -> None:
        log_dir = Path(self.config.project_root) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_dir / "audio_listener.log", encoding="utf-8"), logging.StreamHandler()],
        )

    def _log(self, level: int, msg: str, **extra: Any) -> None:
        logging.log(level, msg + (f" | {json.dumps(extra, default=str)}" if extra else ""))

    def speak(self, text: str) -> None:
        if not self.voice_enabled:
            return
        try:
            self._announce_q.put_nowait(str(text or ""))
        except Exception:
            pass

    def _speak_worker(self) -> None:
        while self.running:
            try:
                text = self._announce_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._speaker.speak(text)

    def startup_announcement(self) -> None:
        status: dict[str, Any] = {}
        for i in range(6):
            status = self.api.read_status()
            if status:
                break
            self._log(logging.INFO, f"Waiting for status.json... attempt {i+1}/6")
            time.sleep(5)
        flags = self.api.read_flags()
        risk = dict(status.get("risk", {}) or {})
        account = status.get("account_number", "demo account")
        broker = "HFM"
        equity = float(risk.get("equity", 0.0) or 0.0)
        session = str(dict(status.get("context_snapshot", {}) or {}).get("session", "unknown"))
        regime = str(dict(status.get("market_regime", {}) or {}).get("regime", "unknown"))
        daily_pnl = float(risk.get("daily_profit_zar", 0.0) or 0.0)
        open_trades = int(risk.get("open_positions", 0) or 0)
        strategies = int(dict(status.get("signal_summary", {}) or {}).get("active_count", 0) or 0)
        mode = str(flags.get("mode", "demo"))
        executing = bool(flags.get("execute_trades", False))
        connected = bool(dict(status.get("health", {}).get("broker", {}) or {}).get("connected", False))
        self.speak("Atlas trading system online.")
        if connected:
            self.speak(f"Connected to {broker}. Account {account}. Equity {equity:.2f} dollars.")
        else:
            self.speak("Warning. Broker connection failed. Trading is paused.")
        self.speak(f"Mode: {mode}. Trading {'active' if executing else 'paused'}.")
        self.speak(f"{strategies} strategies loaded.")
        self.speak(f"Current session: {session}. Market regime: {regime}.")
        if open_trades > 0:
            self.speak(f"{open_trades} open positions recovered.")
        if daily_pnl != 0.0:
            self.speak(f"Today's P and L so far: {daily_pnl:.2f} dollars.")
        self.speak(f"Audio listener active. Microphone: {self._last_mic_name}.")
        self.speak("Say atlas help for available voice commands.")
        self.speak("Atlas is ready.")

    def receive_announcements(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", self.socket_port))
        sock.listen(5)
        sock.settimeout(1.0)
        while self.running:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    data = conn.recv(4096).decode("utf-8", errors="ignore")
                    msg = json.loads(data or "{}")
                    if msg.get("type") == "announce":
                        self.speak(str(msg.get("text", "")))
                except Exception as exc:
                    self._log(logging.ERROR, f"Socket receive error: {exc}")

    def _listen_once(self) -> None:
        with SpeechInput(self.config) as mic:
            self._last_mic_name = str(getattr(mic, "_device", "default microphone"))
            self._log(logging.INFO, f"Audio listener using microphone: {self._last_mic_name}")
            armed = False
            while self.running:
                text, conf = mic.next_recognition()
                if not text:
                    continue
                heard = str(text).strip().lower()
                if self.wake_word in heard and float(conf or 0.0) >= self.wake_word_sensitivity:
                    armed = True
                    self.speak("Listening.")
                    continue
                if armed and heard.startswith("atlas"):
                    self._router.route(heard)
                    armed = False

    def _listen_supervisor(self) -> None:
        while self.running:
            try:
                self._listen_once()
            except Exception:
                self._crashes.append(time.time())
                self._crashes = [x for x in self._crashes if (time.time() - x) <= 3600]
                self._log(logging.ERROR, "Audio listener crashed — restarting in 5s", traceback=traceback.format_exc())
                if len(self._crashes) > 10:
                    self._log(logging.CRITICAL, "Too many crashes in one hour; switching to TTS-only mode")
                    self.voice_enabled = False
                time.sleep(5.0)

    def run(self) -> int:
        self._configure_logging()
        threading.Thread(target=self._speak_worker, daemon=True).start()
        threading.Thread(target=self.receive_announcements, daemon=True).start()
        for i in range(1, 13):
            try:
                self._log(logging.INFO, f"Audio device initialisation attempt [{i}]/12...")
                test = SpeechInput(self.config)
                self._last_mic_name = str(getattr(test, "_device", "default microphone"))
                break
            except Exception as exc:
                self._log(logging.ERROR, f"Audio init failed: {exc}")
                time.sleep(10.0)
        self.startup_announcement()
        self._listen_supervisor()
        return 0


def main() -> int:
    return AudioListenerBridge().run()


if __name__ == "__main__":
    raise SystemExit(main())
