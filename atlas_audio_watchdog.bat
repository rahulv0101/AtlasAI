@echo off
setlocal
cd /d "%~dp0"
REM FIX: retire legacy shell loop to prevent cmd respawn windows and duplicate listeners.
if exist ".venv\Scripts\pythonw.exe" (
  ".venv\Scripts\pythonw.exe" "atlas_audio_watchdog.py"
) else (
  start "" /b pythonw "atlas_audio_watchdog.py"
)
