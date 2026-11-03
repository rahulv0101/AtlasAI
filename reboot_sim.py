import sys
import os
import psutil
import time
import subprocess

# Force UTF-8 output to avoid cp1252 Unicode errors on Windows CMD
sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

ATLAS_SCRIPTS = ["main.py", "audio_listener_bridge.py", "watchdog.py"]
BASE = r"C:\Users\User\Desktop\Atlas AI"
PYTHON = BASE + r"\.venv\Scripts\python.exe"


def kill_atlas_processes():
    killed = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if any(s in cmdline for s in ATLAS_SCRIPTS):
                proc.kill()
                killed.append(f"PID {proc.info['pid']}: {cmdline[:60]}")
        except Exception:
            continue
    return killed


def find_atlas_processes():
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            for s in ATLAS_SCRIPTS:
                if s in cmdline and s not in found:
                    found.append(s)
        except Exception:
            continue
    return found


print("=== ATLAS REBOOT SIMULATION ===")
print()

# Step 1 - kill all Atlas processes
print("Step 1: Killing all Atlas processes...")
killed = kill_atlas_processes()
if killed:
    for k in killed:
        print(f"  Killed: {k}")
else:
    print("  No Atlas processes were running.")
    print("  Starting them now for simulation...")
    # Start main and watchdog so we can test the kill/restart cycle
    subprocess.Popen([PYTHON, BASE + r"\watchdog.py"], creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(2)
    killed = kill_atlas_processes()

print()

# Step 2 - wait for watchdog to detect and restart
print("Step 2: Waiting 25 seconds for watchdog to detect and restart...")
time.sleep(25)

# Step 3 - check what restarted
print("Step 3: Checking for restarted processes...")
restarted = find_atlas_processes()
for s in restarted:
    print(f"  Restarted: {s}")

# Step 4 - results
print()
print("=== RESULTS ===")
watchdog_restarted = "watchdog.py" in restarted
main_restarted = "main.py" in restarted

print(f"  {'PASS' if watchdog_restarted else 'FAIL'} -- watchdog.py")
print(f"  {'PASS' if main_restarted else 'INFO'} -- main.py (only restarts if watchdog is running)")
print()

if watchdog_restarted:
    print("Watchdog is running and monitoring. System will auto-recover from crashes.")
else:
    print("Watchdog did not restart. Check watchdog.py and scheduled task AtlasWatchdog.")

input("\nPress Enter to exit...")

