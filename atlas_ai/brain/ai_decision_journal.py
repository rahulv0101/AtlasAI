from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail


class AIDecisionJournal:
    """Append-only journal for AI decision traces."""

    def __init__(self, storage_dir: str | Path = "storage", *, filename: str = "ai_decision_journal.jsonl") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / filename

    def append(self, trace: dict[str, Any], *, source: str, query: str = "") -> dict[str, Any]:
        payload = {
            "journal_schema": "atlas_ai_decision_journal_v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source or "unknown"),
            "query": str(query or ""),
            "trace": dict(trace or {}),
        }
        payload["decision_id"] = decision_record_id(payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return payload

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return read_jsonl_tail(self.path, limit)

    def latest(self) -> dict[str, Any]:
        rows = self.recent(limit=1)
        return rows[-1] if rows else {}


def summarize_decision_record(record: dict[str, Any]) -> dict[str, Any]:
    trace = dict(record.get("trace", {}) or {})
    rec = dict(trace.get("recommendation", {}) or {})
    blockers = dict(trace.get("blocker_analysis", {}) or {})
    invention = dict(trace.get("strategy_invention", {}) or {})
    market = dict(trace.get("market_read", {}) or {})
    return {
        "decision_id": decision_record_id(record),
        "source": str(record.get("source", "") or ""),
        "query": str(record.get("query", "") or ""),
        "recorded_at": str(record.get("recorded_at", "") or ""),
        "action": str(rec.get("action", "unknown") or "unknown"),
        "confidence": float(rec.get("confidence", 0.0) or 0.0),
        "reason": str(rec.get("reason", "") or ""),
        "primary_blocker": str(blockers.get("primary_blocker", "none") or "none"),
        "trade_bias": str(rec.get("trade_bias", market.get("final_trade_bias", "neutral")) or "neutral"),
        "strategy_candidate_count": int(invention.get("candidate_count", 0) or 0),
        "strategy_archetypes": list(invention.get("candidate_archetypes", []) or []),
    }


def decision_record_id(record: dict[str, Any]) -> str:
    if record.get("decision_id"):
        return str(record["decision_id"])
    identity = {key: record.get(key) for key in ("recorded_at", "source", "query", "trace")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
