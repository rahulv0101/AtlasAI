from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def stop_request_path(comparison_root: str, comparison_run_id: str) -> Path:
    return Path(comparison_root) / comparison_run_id / "atlas_stop_request.json"


def write_stop_request(comparison_root: str, comparison_run_id: str, reason: str) -> Path:
    path = stop_request_path(comparison_root, comparison_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason or "Atlas sleep requested"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_stop_request(comparison_root: str, comparison_run_id: str) -> dict[str, Any]:
    path = stop_request_path(comparison_root, comparison_run_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def clear_stop_request(comparison_root: str, comparison_run_id: str) -> None:
    path = stop_request_path(comparison_root, comparison_run_id)
    if path.exists():
        path.unlink()


def stop_requested(comparison_root: str, comparison_run_id: str) -> bool:
    return stop_request_path(comparison_root, comparison_run_id).exists()
