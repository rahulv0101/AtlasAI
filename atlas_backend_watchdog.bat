@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" "tools\backend_heartbeat_watchdog.py"
) else if exist ".venv\Scripts\python.exe" (
  start "" ".venv\Scripts\python.exe" "tools\backend_heartbeat_watchdog.py"
) else (
  start "" python "tools\backend_heartbeat_watchdog.py"
)
endlocal
exit /b 0
