@echo off
setlocal
cd /d "%~dp0"
REM FIX: keep legacy wrapper compatible, but launch the no-console watchdog path.
if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" "atlas_audio_watchdog.py"
) else (
  start "" /b pythonw "atlas_audio_watchdog.py"
)
