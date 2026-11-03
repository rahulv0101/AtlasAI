@echo off
setlocal
set "PROJECT_ROOT=c:\Users\User\Desktop\Atlas AI"
set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"
set "VENV_PYW=%PROJECT_ROOT%\.venv\Scripts\pythonw.exe"
set "PYVENV_CFG=%PROJECT_ROOT%\.venv\pyvenv.cfg"
cd /d "%PROJECT_ROOT%"

REM Single-instance guard: do nothing if listener or watchdog is already running.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|pythonw' -and (($_.CommandLine -match '-m\s+atlas_ai\.atlas_audio_listener(\s|$)') -or ($_.CommandLine -match 'atlas_audio_watchdog\.py')) }; if ($p) { exit 0 } else { exit 1 }"
if %errorlevel%==0 (
  echo Atlas audio path already running.
  goto :eof
)

set "BASE_PY="
set "BASE_PYW="
for /f "tokens=1,* delims==" %%A in ('findstr /B /I "executable" "%PYVENV_CFG%"') do (
  set "BASE_PY=%%B"
)
if defined BASE_PY (
  if "%BASE_PY:~0,1%"==" " set "BASE_PY=%BASE_PY:~1%"
)
if not defined BASE_PY set "BASE_PY=%VENV_PY%"
if not exist "%BASE_PY%" set "BASE_PY=%VENV_PY%"
set "BASE_PYW=%BASE_PY:python.exe=pythonw.exe%"
if not exist "%BASE_PYW%" set "BASE_PYW=%VENV_PYW%"

set "__PYVENV_LAUNCHER__=%VENV_PY%"
set "VIRTUAL_ENV=%PROJECT_ROOT%\.venv"
REM Launch pythonw directly so Atlas does not create a visible cmd shell on startup.
"%BASE_PYW%" "%PROJECT_ROOT%\atlas_audio_watchdog.py"
endlocal
