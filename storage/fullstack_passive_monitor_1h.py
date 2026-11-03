import json
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(r'c:\Users\User\Desktop\Atlas AI')
LOG = ROOT / 'storage' / 'fullstack_passive_monitor_1h.log'
END = datetime.now(timezone.utc) + timedelta(hours=1)


def run_ps(cmd: str) -> str:
    p = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True)
    return (p.stdout or '') + (p.stderr or '')


def snapshot_rows():
    ps = """
Get-CimInstance Win32_Process | Where-Object {
 $_.Name -eq 'python.exe' -and $_.CommandLine -and (
   $_.CommandLine -match 'atlas_full_launcher' -or
   $_.CommandLine -match 'atlas_ai\\.main live' -or
   $_.CommandLine -match 'atlas_ai\\.main ui-server' -or
   $_.CommandLine -match 'atlas_ai\\.main ui'
 )
} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 4
"""
    out = run_ps(ps).strip()
    if not out:
        return []
    try:
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def classify(rows):
    sys_n = 0
    venv_n = 0
    roles = {'launcher':0,'live':0,'ui_server':0,'ui':0}
    for r in rows:
        cl = str(r.get('CommandLine',''))
        low = cl.lower()
        if 'python311\\python.exe' in low:
            sys_n += 1
        if '.venv\\scripts\\python.exe' in low:
            venv_n += 1
        if 'atlas_full_launcher' in low:
            roles['launcher'] += 1
        if 'atlas_ai.main live' in low:
            roles['live'] += 1
        if 'atlas_ai.main ui-server' in low:
            roles['ui_server'] += 1
        if 'atlas_ai.main ui --mode-override' in low:
            roles['ui'] += 1
    healthy_roles = roles['launcher'] >= 1 and roles['live'] >= 1
    return sys_n, venv_n, roles, healthy_roles

LOG.parent.mkdir(parents=True, exist_ok=True)
with LOG.open('a', encoding='utf-8') as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] passive_monitor_started\n")

while datetime.now(timezone.utc) < END:
    rows = snapshot_rows()
    sys_n, venv_n, roles, healthy = classify(rows)
    ts = datetime.now(timezone.utc).isoformat()
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{ts}] total={len(rows)} system={sys_n} venv={venv_n} roles={roles} healthy_roles={healthy}\n")
    time.sleep(60)

with LOG.open('a', encoding='utf-8') as f:
    f.write(f"[{datetime.now(timezone.utc).isoformat()}] passive_monitor_finished\n")
