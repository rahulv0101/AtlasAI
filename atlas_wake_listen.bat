@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m atlas_ai.atlas_wake_listener --stdin --json
) else (
  python -m atlas_ai.atlas_wake_listener --stdin --json
)
