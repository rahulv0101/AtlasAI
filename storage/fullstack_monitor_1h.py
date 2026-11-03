import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(r'c:\Users\User\Desktop\Atlas AI')
PY = ROOT / '.venv' / 'Scripts' / 'python.exe'
LOG = ROOT / 'storage' / 'fullstack_monitor_1h.log'
END = datetime.now(timezone.utc) + timedelta(hours=1)
ALLOW_RESPAWN = str(os.getenv("ATLAS_MONITOR_ALLOW_RESPAWN", "false")).strip().lower() in {"1", "true", "yes", "on"}

ATLAS_PATTERNS = [
    'atlas_full_launcher',
    'atlas_ai.main live',
    'atlas_ai.main ui-server',
    'atlas_ai.main ui',
]


def run_ps(cmd: str) -> str:
    p = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True)
    return (p.stdout or '') + (p.stderr or '')


def snapshot():
    ps = """
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq 'python.exe' -and $_.CommandLine -and (
    $_.CommandLine -match 'atlas_full_launcher' -or
    $_.CommandLine -match 'atlas_ai\\.main live' -or
    $_.CommandLine -match 'atlas_ai\\.main ui-server' -or
    $_.CommandLine -match 'atlas_ai\\.main ui'
  )
} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 4
"""
    out = run_ps(ps).strip()
    if not out:
        return []
    try:
        data = json.loads(out)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def kill_system_duplicates(rows):
    pids = []
    for r in rows:
        cl = str(r.get('CommandLine',''))
        pid = int(r.get('ProcessId',0) or 0)
        if 'Python311\\python.exe' in cl and pid:
            pids.append(pid)
    if pids:
        run_ps('; '.join([f'Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue' for pid in pids]))
    return pids


def ensure_venv_stack(rows):
    if not ALLOW_RESPAWN:
        return False
    txt = ' || '.join(str(r.get('CommandLine','')) for r in rows)
    have_launcher = '.venv\\Scripts\\python.exe" -m atlas_ai.atlas_full_launcher --json' in txt
    if not have_launcher:
        subprocess.Popen([str(PY), '-m', 'atlas_ai.atlas_full_launcher', '--json'], cwd=str(ROOT))
        return True
    return False


LOG.parent.mkdir(parents=True, exist_ok=True)
with LOG.open('a', encoding='utf-8') as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] monitor_started allow_respawn={ALLOW_RESPAWN}\n")

while datetime.now(timezone.utc) < END:
    rows = snapshot()
    killed = kill_system_duplicates(rows)
    if killed:
        time.sleep(2)
        rows = snapshot()
    restarted = ensure_venv_stack(rows)
    ts = datetime.now(timezone.utc).isoformat()
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{ts}] count={len(rows)} killed_system={killed} restarted={restarted}\n")
    time.sleep(60)

with LOG.open('a', encoding='utf-8') as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] monitor_finished\n")
