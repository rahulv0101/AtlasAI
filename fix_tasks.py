import subprocess
import os
import sys

python = r"C:\Users\User\Desktop\Atlas AI\.venv\Scripts\python.exe"
base = r"C:\Users\User\Desktop\Atlas AI"
log_dir = base + r"\logs"

os.makedirs(log_dir, exist_ok=True)

pythonw = python.replace("python.exe", "pythonw.exe")
audio_script = base + r"\atlas_audio_watchdog.py"
TASK_NAME = "Atlas Audio Listener"
LEGACY_TASKS = ["AtlasGoldTrader", "AtlasWatchdog", "AtlasAudioBridge"]

print("Migrating Atlas startup tasks to audio-only mode...")
print(f"Python: {pythonw}")
print(f"Base:   {base}")
print()

all_ok = True
for name in LEGACY_TASKS:
    print(f"Removing legacy task: {name}...")
    result = subprocess.run(["schtasks", "/delete", "/tn", name, "/f"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  SUCCESS: {name} removed")
    else:
        error = result.stderr.strip() or result.stdout.strip()
        print(f"  INFO:    {name} not present ({error or 'already absent'})")
    print()

cmd = [
    "schtasks",
    "/create",
    "/tn",
    TASK_NAME,
    "/tr",
    f'"{pythonw}" "{audio_script}"',
    "/sc",
    "ONLOGON",
    "/rl",
    "HIGHEST",
    "/f",
]
print(f"Creating: {TASK_NAME}...")
print(f"Command:  {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
    print(f"  SUCCESS: {TASK_NAME} created")
else:
    error = result.stderr.strip() or result.stdout.strip()
    print(f"  FAILED:  {TASK_NAME}")
    print(f"  Error:   {error}")
    all_ok = False
print()

print("Verifying startup tasks...")
for name in [TASK_NAME]:
    result = subprocess.run(
        ["schtasks", "/query", "/tn", name, "/fo", "LIST"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        status = [l for l in result.stdout.splitlines() if "Status" in l]
        print(f"  OK:       {name} -- {status[0].strip() if status else 'found'}")
    else:
        print(f"  MISSING:  {name}")
        all_ok = False

print()
if all_ok:
    print("AUDIO TASK CREATED SUCCESSFULLY.")
    print("Atlas listener will start automatically on next boot.")
    print()
    print("To test without rebooting, run:")
    print('  schtasks /run /tn "Atlas Audio Listener"')
else:
    print("SOME TASKS FAILED. Check errors above.")
    print("Make sure you are running this from an admin CMD terminal.")

input("\nPress Enter to exit...")
