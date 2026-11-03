from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "storage" / "monitoring" / "atlas_weekend_learning_digest.jsonl"
SUMMARY = ROOT / "storage" / "monitoring" / "atlas_weekend_learning_summary.json"
ONLINE = ROOT / "storage" / "monitoring" / "atlas_online_learning.json"
NEAR = ROOT / "storage" / "monitoring" / "atlas_near_setup_learning.json"
POST = ROOT / "storage" / "monitoring" / "atlas_post_trade_learning.json"
FLAGS = ROOT / "storage" / "runtime_flags.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _status() -> dict[str, Any]:
    try:
        response = requests.get("http://127.0.0.1:8765/api/status", timeout=8)
        if response.status_code == 200:
            payload = response.json()
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


def _top_patterns(report: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    patterns = report.get("top_patterns") or report.get("top_fingerprints") or []
    if isinstance(patterns, list):
        return [p for p in patterns[:limit] if isinstance(p, dict)]
    return []


def _build_record() -> dict[str, Any]:
    status = _status()
    online = _read_json(ONLINE)
    near = _read_json(NEAR)
    post = _read_json(POST)
    flags = _read_json(FLAGS)
    health = dict(status.get("system_health", {}) or {})
    automation = dict(status.get("automation_truth", {}) or {})
    learning = dict(online.get("latest", {}) or {})
    post_patterns = _top_patterns(post)
    top_post = post_patterns[0] if post_patterns else {}
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "mode": "weekend_learning_only",
        "execution_disabled": not bool(flags.get("execute_trades", False)),
        "weekend_learning_only": bool(flags.get("weekend_learning_only", False)),
        "automation_paused": bool(automation.get("automation_paused", False)),
        "automation_pause_reason": str(automation.get("automation_pause_reason", "") or ""),
        "automation_explicit_resume_active": bool(automation.get("automation_explicit_resume_active", False)),
        "broker_connected": bool(dict(health.get("broker_connectivity_status", {}) or {}).get("connected", False)),
        "status_action": str(status.get("action", "") or ""),
        "primary_blocker": str(status.get("primary_blocker", "") or ""),
        "hard_blockers": list(dict(status.get("decision_analysis", {}) or {}).get("hard_blockers", []) or []),
        "market": {
            "session": status.get("session", ""),
            "setup_family": status.get("setup_family", ""),
            "structure": dict(status.get("chart_features", {}) or {}).get("trend_structure", ""),
            "momentum": dict(status.get("chart_features", {}) or {}).get("momentum_state", ""),
            "location": dict(status.get("chart_features", {}) or {}).get("location_state", ""),
            "spread_points": dict(status.get("risk", {}) or {}).get("spread_points", None),
        },
        "online_learning": {
            "samples": learning.get("samples", 0),
            "mode": learning.get("mode", ""),
            "allowed_influence": learning.get("allowed_influence", ""),
            "evidence_strength": learning.get("evidence_strength", 0),
            "overreaction_risk": learning.get("overreaction_risk", 0),
            "top_fingerprint": learning.get("top_fingerprint", ""),
            "teaching_focus": learning.get("teaching_focus", []),
        },
        "near_setup_top_patterns": _top_patterns(near),
        "post_trade": {
            "total_closed_trades_seen": post.get("total_closed_trades_seen", 0),
            "top_pattern": top_post,
            "policy_note": post.get("policy_note", ""),
        },
        "policy_note": "Weekend learning digest is observability only; it does not change strategy, risk, guards, or execution policy.",
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Monday 08:00 local SAST is 06:00 UTC.
    now = datetime.now(timezone.utc)
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    end_at = (now + timedelta(days=days_until_monday)).replace(hour=6, minute=0, second=0, microsecond=0)
    interval_seconds = 15 * 60
    last_record: dict[str, Any] = {}
    while datetime.now(timezone.utc) < end_at:
        last_record = _build_record()
        with OUT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(last_record, separators=(",", ":")) + "\n")
        SUMMARY.write_text(json.dumps({"end_at": end_at.isoformat(), "latest": last_record}, indent=2), encoding="utf-8")
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
