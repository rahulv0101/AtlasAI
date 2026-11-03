from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def relaunch_as_admin() -> None:
    print("Administrator rights required. Requesting elevation...")
    script = os.path.abspath(__file__)
    params = " ".join([f'"{arg}"' for arg in sys.argv[1:]])
    result = ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, f'"{script}" {params}', None, 1)
    if result <= 32:
        print(f"Elevation failed with code {result}")
        print("Please right-click setup_autostart.py and select 'Run as administrator'")
        sys.exit(1)
    print("Elevated process launched. Check the new window for results.")
    sys.exit(0)


def remove_legacy_scheduled_tasks() -> dict:
    results = {}
    for task_name in ["AtlasGoldTrader", "AtlasWatchdog", "AtlasAudioBridge"]:
        result = subprocess.run(
            ["schtasks", "/delete", "/tn", task_name, "/f"],
            capture_output=True,
            text=True,
        )
        removed = result.returncode == 0
        results[task_name] = "removed" if removed else "absent_or_failed"
        if removed:
            print(f"  OK Removed legacy task: {task_name}")
    return results


def create_scheduled_tasks(python_path: str, script_dir: str) -> dict:
    results = {}
    audio_script = os.path.join(script_dir, "atlas_audio_watchdog.py")
    log_dir = os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    tasks = [
        {
            "name": "Atlas Audio Listener",
            "script": audio_script,
            "trigger": "ONLOGON",
            "user": "",
            "delay": None,
            "log": os.path.join(log_dir, "audio_listener_startup.log"),
            "pythonw": True,
        },
    ]
    for task in tasks:
        python_target = python_path.replace("python.exe", "pythonw.exe") if task.get("pythonw") else python_path
        tr = f'"{python_target}" "{task["script"]}"'
        cmd = ["schtasks", "/create", "/tn", task["name"], "/tr", tr, "/sc", task["trigger"], "/rl", "HIGHEST", "/f"]
        if task.get("delay") and task["trigger"] == "ONSTART":
            cmd += ["/delay", task["delay"]]
        if task["user"]:
            cmd += ["/ru", task["user"]]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  OK Task created: {task['name']}")
            results[task["name"]] = "created"
        else:
            error_msg = (result.stderr.strip() or result.stdout.strip())
            print(f"  FAIL Task failed: {task['name']}")
            print(f"    Error: {error_msg}")
            print(f"    Command: {' '.join(cmd)}")
            results[task["name"]] = f"failed: {error_msg}"
    return results


def create_startup_folder_entry(python_path: str, script_dir: str) -> bool:
    try:
        import winshell
        from win32com.client import Dispatch
        startup_folder = winshell.startup()
        shortcut_path = os.path.join(startup_folder, "Atlas Audio Listener.lnk")
        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = python_path.replace("python.exe", "pythonw.exe")
        shortcut.Arguments = f'"{os.path.join(script_dir, "atlas_audio_watchdog.py")}"'
        shortcut.WorkingDirectory = script_dir
        shortcut.WindowStyle = 7
        shortcut.IconLocation = shortcut.Targetpath
        shortcut.save()
        print(f"  OK Startup folder shortcut created: {shortcut_path}")
        return True
    except ImportError:
        print("  FAIL winshell/pywin32 not available - startup folder skipped")
        return False
    except Exception as e:
        print(f"  FAIL Startup folder failed: {e}")
        return False


def create_registry_entry(python_path: str, script_dir: str) -> None:
    import winreg
    key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
    value_name = "AtlasAudioListener"
    pythonw_path = python_path.replace("python.exe", "pythonw.exe")
    command = f'"{pythonw_path}" "{os.path.join(script_dir, "atlas_audio_watchdog.py")}"'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, command)
    print("  ✓ Registry Run key created")


def remove_legacy_startup_entries(script_dir: str) -> None:
    try:
        import winshell

        startup_folder = winshell.startup()
        for name in ["AtlasGoldTrader.lnk", "AtlasGoldTrader.lnk.disabled", "Atlas Audio Listener.lnk.disabled"]:
            path = os.path.join(startup_folder, name)
            if os.path.exists(path):
                os.remove(path)
                print(f"  OK Removed legacy startup shortcut: {path}")
    except Exception:
        pass

    try:
        import winreg

        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE) as key:
            for value_name in ["AtlasGoldTrader", "AtlasAudioBridge"]:
                try:
                    winreg.DeleteValue(key, value_name)
                    print(f"  OK Removed legacy Run key: {value_name}")
                except FileNotFoundError:
                    pass
    except Exception:
        pass


def run_reboot_simulation() -> bool:
    print("\n=== AUDIO RESTART SIMULATION TEST ===")
    import psutil
    atlas_scripts = ["atlas_audio_watchdog.py", "atlas_ai.atlas_audio_listener"]
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            if any(script in cmdline for script in atlas_scripts):
                proc.kill()
        except Exception:
            continue
    print("  Waiting 10 seconds for listener task/watchdog restart...")
    time.sleep(10)
    restarted = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or [])
            for script in atlas_scripts:
                if script in cmdline:
                    restarted.append(script)
        except Exception:
            continue
    for script in atlas_scripts:
        status = "PASS ✓" if script in restarted else "FAIL ✗"
        print(f"  {status} — {script}")
    return len(set(restarted)) == len(atlas_scripts)


def main():
    print("=" * 55)
    print("  Atlas Gold Trading System — Autostart Setup")
    print("=" * 55)
    python_path = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nPython: {python_path}")
    print(f"Script dir: {script_dir}")
    print(f"Running as admin: {is_admin()}")
    print("\n[1/6] Installing dependencies...")
    subprocess.run([python_path, "-m", "pip", "install", "winshell", "pywin32", "psutil", "--quiet"])
    subprocess.run([python_path, "-m", "pywin32_postinstall", "-install"], capture_output=True)
    print("  Dependencies installed")
    print("\n[2/6] Removing legacy autostarts...")
    remove_legacy_scheduled_tasks()
    remove_legacy_startup_entries(script_dir)
    print("\n[3/6] Creating scheduled tasks...")
    create_scheduled_tasks(python_path, script_dir)
    print("\n[4/6] Creating startup folder shortcut...")
    create_startup_folder_entry(python_path, script_dir)
    print("\n[5/6] Creating registry Run key...")
    create_registry_entry(python_path, script_dir)
    print("\n[6/6] Verifying scheduled tasks...")
    all_tasks_ok = True
    for task_name in ["Atlas Audio Listener"]:
        result = subprocess.run(["schtasks", "/query", "/tn", task_name, "/fo", "LIST"], capture_output=True, text=True)
        if result.returncode == 0 and "Ready" in result.stdout:
            print(f"  OK {task_name}: Ready")
        elif result.returncode == 0:
            status_line = [l for l in result.stdout.splitlines() if "Status" in l]
            print(f"  ~ {task_name}: {status_line[0] if status_line else 'created'}")
        else:
            print(f"  FAIL {task_name}: NOT FOUND")
            all_tasks_ok = False
    print("\n" + "=" * 55)
    print("  SETUP COMPLETE")
    print("=" * 55)
    print(f"  Scheduled tasks: {'OK' if all_tasks_ok else 'PARTIAL — check above'}")
    print("  Registry Run key: OK")
    print("  Atlas audio listener starts automatically on next boot.")
    if all_tasks_ok:
        print("Running audio restart simulation test...")
        run_reboot_simulation()
    print("\nSetup finished.")


if __name__ == "__main__":
    if "--uninstall" not in sys.argv and "--no-relaunch" not in sys.argv:
        if not is_admin():
            relaunch_as_admin()
            sys.exit(0)
    main()
