from __future__ import annotations

import logging
import os
import subprocess
import time
import contextlib
import ctypes
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5

from atlas_ai.config.settings import Settings

logger = logging.getLogger(__name__)


class HFMConnector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._broker_mutex_handle = None

    def _candidate_terminal_paths(self) -> list[Path]:
        candidates: list[Path] = []
        if self.settings.hfm_terminal_path:
            candidates.append(Path(self.settings.hfm_terminal_path))

        appdata = os.getenv("APPDATA", "")
        localappdata = os.getenv("LOCALAPPDATA", "")
        program_files = os.getenv("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")

        candidates.extend(
            [
                Path(program_files) / "HF Markets MT5 Terminal" / "terminal64.exe",
                Path(program_files_x86) / "HF Markets MT5 Terminal" / "terminal64.exe",
                Path(program_files) / "MetaTrader 5" / "terminal64.exe",
                Path(program_files_x86) / "MetaTrader 5" / "terminal64.exe",
                Path(appdata) / "MetaQuotes" / "Terminal" / "terminal64.exe",
                Path(localappdata) / "Programs" / "HF Markets MT5 Terminal" / "terminal64.exe",
            ]
        )
        return candidates

    def _launch_hfm_terminal(self) -> Path | None:
        for path in self._candidate_terminal_paths():
            if path and path.exists():
                subprocess.Popen([str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logger.info("Opened terminal: %s", path)
                return path
        return None

    def connect(self) -> None:
        self._acquire_broker_mutex()
        try:
            opened_path = self._launch_hfm_terminal()
            if opened_path is not None:
                time.sleep(3)
            init_ok = False
            last_error = None
            # Retry IPC initialization because MT5 can take time to expose its IPC endpoint.
            for _ in range(4):
                init_ok = mt5.initialize(path=self.settings.hfm_terminal_path) if self.settings.hfm_terminal_path else mt5.initialize()
                if init_ok:
                    break
                last_error = mt5.last_error()
                time.sleep(2)

            if not init_ok and self.settings.hfm_terminal_path:
                # Fallback without explicit path for installations where terminal location differs from runtime IPC endpoint.
                for _ in range(3):
                    init_ok = mt5.initialize()
                    if init_ok:
                        break
                    last_error = mt5.last_error()
                    time.sleep(2)

            if not init_ok:
                raise RuntimeError(
                    "MT5 initialize failed after retries. Install/open HFM terminal, verify HFM_TERMINAL_PATH, "
                    f"and keep only one MT5 instance for this account. MT5 error: {last_error or mt5.last_error()}"
                )

            authorized = mt5.login(
                login=self.settings.hfm_login,
                password=self.settings.hfm_password,
                server=self.settings.hfm_server,
            )
            if not authorized:
                raise RuntimeError(f"HFM login failed: {mt5.last_error()}")

            terminal = mt5.terminal_info()
            if terminal is None:
                raise RuntimeError("MT5 terminal info unavailable")
            if not terminal.trade_allowed:
                raise RuntimeError("MT5 AutoTrading is disabled. Enable AutoTrading in terminal.")

            selected = mt5.symbol_select(self.settings.symbol, True)
            if not selected:
                raise RuntimeError(f"Failed to select trading symbol: {self.settings.symbol}")

            logger.info("Connected to HFM account %s", self.settings.hfm_login)
        except Exception:
            self._release_broker_mutex()
            raise

    def shutdown(self) -> None:
        mt5.shutdown()
        self._release_broker_mutex()

    def account_info(self) -> Any:
        return mt5.account_info()

    def symbol_info(self, symbol: str) -> Any:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        return info

    def _acquire_broker_mutex(self) -> None:
        if os.name != "nt":
            return
        if self._broker_mutex_handle:
            return
        server = str(self.settings.hfm_server or "default")
        server = "".join(ch if ch.isalnum() else "_" for ch in server)[:40]
        mutex_name = f"Global\\AtlasAI_broker_{int(self.settings.hfm_login)}_{server}"
        k32 = ctypes.windll.kernel32
        k32.SetLastError(0)
        handle = k32.CreateMutexW(None, False, mutex_name)
        if not handle:
            raise RuntimeError("Failed to acquire broker session mutex.")
        already_exists = k32.GetLastError() == 183
        if already_exists:
            k32.CloseHandle(handle)
            raise RuntimeError("Another Atlas broker session is already active for this account.")
        self._broker_mutex_handle = handle

    def _release_broker_mutex(self) -> None:
        handle = self._broker_mutex_handle
        if handle and os.name == "nt":
            with contextlib.suppress(Exception):
                ctypes.windll.kernel32.CloseHandle(handle)
        self._broker_mutex_handle = None
