from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests


def _detect_base_url(candidates: list[int]) -> str | None:
    for port in candidates:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/api/status", timeout=2.0)
            if r.status_code == 200:
                return f"http://127.0.0.1:{port}"
        except Exception:
            continue
    return None


def _role_for(cmd: str) -> str:
    c = str(cmd or "").lower()
    if "-m atlas_ai.main live" in c:
        return "live"
    if "-m atlas_ai.main ui-server" in c:
        return "ui-server"
    if "-m atlas_ai.main ui" in c and "ui-server" not in c:
        return "ui"
    if "-m atlas_ai.atlas_audio_listener" in c:
        return "audio-listener"
    return "other"


def _atlas_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "exe"]):
        try:
            name = str(proc.info.get("name") or "").lower()
            if name not in {"python.exe", "pythonw.exe"}:
                continue
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "atlas_ai." not in cmd and "atlas_audio_listener" not in cmd:
                continue
            rows.append(
                {
                    "pid": int(proc.info.get("pid") or 0),
                    "ppid": int(proc.info.get("ppid") or 0),
                    "name": str(proc.info.get("name") or ""),
                    "exe": str(proc.info.get("exe") or ""),
                    "cmd": cmd,
                    "role": _role_for(cmd),
                }
            )
        except Exception:
            continue
    return rows


def _logical_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_pid = {int(r.get("pid", 0)): r for r in rows}
    counts = {"live": 0, "ui-server": 0, "ui": 0, "audio-listener": 0, "other": 0}
    for r in rows:
        role = str(r.get("role") or "other")
        pid = int(r.get("pid") or 0)
        ppid = int(r.get("ppid") or 0)
        cmd = str(r.get("cmd") or "").lower()
        parent = by_pid.get(ppid)
        parent_cmd = str((parent or {}).get("cmd") or "").lower()

        has_same_role_child = any(
            int(c.get("ppid", -1)) == pid and str(c.get("role") or "") == role
            for c in rows
        )
        is_venv_launcher = ".venv\\scripts\\python.exe" in cmd
        if is_venv_launcher and has_same_role_child:
            continue

        if parent and str(parent.get("role") or "") == role:
            if ".venv\\scripts\\python.exe" in parent_cmd and "\\python311\\python.exe" in cmd:
                counts[role] += 1
                continue

        counts[role] += 1
    return counts


def _physical_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"live": 0, "ui-server": 0, "ui": 0, "audio-listener": 0, "other": 0}
    for r in rows:
        role = str(r.get("role") or "other")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _broker_connected(status: dict[str, Any]) -> bool:
    sh = dict(status.get("system_health", {}) or {})
    b = dict(sh.get("broker_connectivity_status", {}) or {})
    if "connected" in b:
        return bool(b.get("connected"))
    return str(status.get("action", "") or "").strip().lower() == "connected"


def run_monitor(duration_sec: int, interval_sec: int, base_url: str | None, out_path: str) -> dict[str, Any]:
    candidates = [8765, 5000, 5001]
    if base_url is None:
        base_url = _detect_base_url(candidates)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    end = start + max(10, int(duration_sec))
    samples: list[dict[str, Any]] = []

    while time.time() < end:
        ts = datetime.now(timezone.utc).isoformat()
        rec: dict[str, Any] = {
            "ts": ts,
            "base_url": base_url,
            "api_ok": False,
            "broker_connected": False,
            "action": "",
            "execute_trades": None,
            "physical_counts": {},
            "logical_counts": {},
            "errors": [],
        }

        if not base_url:
            base_url = _detect_base_url(candidates)
            rec["base_url"] = base_url

        status: dict[str, Any] = {}
        if base_url:
            try:
                r = requests.get(f"{base_url}/api/status", timeout=4)
                if r.status_code == 200:
                    status = r.json() if isinstance(r.json(), dict) else {}
                    rec["api_ok"] = True
            except Exception as exc:
                rec["errors"].append(f"status:{type(exc).__name__}")

        rec["action"] = str(status.get("action", "") or "")
        rec["execute_trades"] = status.get("execute_trades", None)
        rec["broker_connected"] = _broker_connected(status)

        rows = _atlas_processes()
        rec["physical_counts"] = _physical_counts(rows)
        rec["logical_counts"] = _logical_counts(rows)

        samples.append(rec)
        with out_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        print(
            f"[MONITOR] {ts} api_ok={rec['api_ok']} broker_connected={rec['broker_connected']} "
            f"action={rec['action'] or 'n/a'} physical={rec['physical_counts']} logical={rec['logical_counts']}",
            flush=True,
        )
        time.sleep(max(5, int(interval_sec)))

    def _ratio(key: str) -> float:
        vals = [bool(s.get(key)) for s in samples]
        return round(sum(1 for x in vals if x) / max(1, len(vals)), 4)

    summary = {
        "started_at_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": int(time.time() - start),
        "samples": len(samples),
        "base_url": base_url,
        "api_ok_ratio": _ratio("api_ok"),
        "broker_connected_ratio": _ratio("broker_connected"),
        "execute_true_ratio": round(
            sum(1 for s in samples if bool(s.get("execute_trades")) is True)
            / max(1, sum(1 for s in samples if s.get("execute_trades") is not None)),
            4,
        ),
        "last_physical_counts": dict(samples[-1].get("physical_counts", {}) if samples else {}),
        "last_logical_counts": dict(samples[-1].get("logical_counts", {}) if samples else {}),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas full-stack runtime monitor")
    parser.add_argument("--duration-sec", type=int, default=3600)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--out", type=str, default="logs/atlas_runtime_monitor.jsonl")
    parser.add_argument("--summary-out", type=str, default="logs/atlas_runtime_monitor_summary.json")
    args = parser.parse_args()

    summary = run_monitor(
        duration_sec=int(args.duration_sec),
        interval_sec=int(args.interval_sec),
        base_url=str(args.base_url or "") or None,
        out_path=str(args.out),
    )
    summary_path = Path(str(args.summary_out))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[MONITOR] COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
