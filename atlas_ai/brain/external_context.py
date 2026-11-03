from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def load_external_context() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "knowledge" / "external_context.json"
    if not path.exists():
        return {
            "enabled": False,
            "source": "external_context.json",
            "status": "missing",
            "freshness": {"is_stale": True, "age_minutes": None},
            "items": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "enabled": False,
            "source": str(path),
            "status": "invalid",
            "freshness": {"is_stale": True, "age_minutes": None},
            "items": [],
        }

    generated_at_raw = str(payload.get("generated_at", "") or "")
    age_minutes = None
    is_stale = True
    if generated_at_raw:
        try:
            generated_at = datetime.fromisoformat(generated_at_raw)
            if generated_at.tzinfo is None:
                generated_at = generated_at.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - generated_at).total_seconds() / 60.0
            is_stale = age_minutes > 90.0
        except Exception:
            age_minutes = None
            is_stale = True
    return {
        "enabled": True,
        "source": str(path),
        "status": "ok",
        "freshness": {
            "is_stale": bool(is_stale),
            "age_minutes": round(float(age_minutes), 2) if age_minutes is not None else None,
            "generated_at": generated_at_raw or "",
        },
        "items": list(payload.get("items", []) or []),
    }

