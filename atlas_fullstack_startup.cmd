@echo off
setlocal
cd /d "c:\Users\User\Desktop\Atlas AI"

if not exist "atlas_fullstack_startup.pyw" (
  echo ERROR: atlas_fullstack_startup.pyw not found.
  exit /b 1
)

REM Compatibility wrapper: delegate to the no-console pythonw launcher.
start "" ".venv\Scripts\pythonw.exe" "c:\Users\User\Desktop\Atlas AI\atlas_fullstack_startup.pyw"

endlocal
exit /b 0
