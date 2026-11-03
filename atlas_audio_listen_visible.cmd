@echo off
cd /d "c:\Users\User\Desktop\Atlas AI"
REM FIX: legacy visible wrapper now delegates to the no-console audio path and exits.
call atlas_audio_listen.bat
