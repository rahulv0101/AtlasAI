Set shell = CreateObject("WScript.Shell")
repo = "C:\Users\User\Desktop\Atlas AI"
' FIX: launch Atlas audio watchdog directly with pythonw so startup never opens cmd.
pythonw = repo & "\.venv\Scripts\pythonw.exe"
script = repo & "\atlas_audio_watchdog.py"
cmd = """" & pythonw & """ """ & script & """"
shell.Run cmd, 0, False
