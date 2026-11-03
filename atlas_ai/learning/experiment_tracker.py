from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail


class ExperimentTracker:
    def __init__(self, storage_dir: str) -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.file = self.base / "experiments.jsonl"

    def log_run(
        self,
        name: str,
        variant: str,
        metrics: dict[str, Any],
        sample_size: int,
        notes: str = "",
    ) -> dict[str, Any]:
        record = {
            "id": str(uuid.uuid4()),
            "time": datetime.now(timezone.utc).isoformat(),
            "name": str(name),
            "variant": str(variant),
            "sample_size": int(sample_size),
            "metrics": dict(metrics),
            "notes": str(notes),
        }
        with self.file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    def list_runs(self, name: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        # Preserve the old contract: select the last N records first, then
        # optionally filter those records by experiment name.
        rows = read_jsonl_tail(self.file, limit)
        if name:
            rows = [row for row in rows if str(row.get("name", "")) == str(name)]
        return rows

    def summarize(self, name: str) -> dict[str, Any]:
        runs = self.list_runs(name=name, limit=5000)
        if not runs:
            return {"name": name, "variants": {}, "total_runs": 0}
        agg: dict[str, dict[str, Any]] = {}
        for r in runs:
            variant = str(r.get("variant", "unknown"))
            bucket = agg.setdefault(
                variant,
                {
                    "runs": 0,
                    "sample_size": 0,
                    "avg_accuracy": 0.0,
                    "avg_brier": 0.0,
                },
            )
            m = dict(r.get("metrics", {}) or {})
            bucket["runs"] += 1
            bucket["sample_size"] += int(r.get("sample_size", 0) or 0)
            bucket["avg_accuracy"] += float(m.get("accuracy", 0.0) or 0.0)
            bucket["avg_brier"] += float(m.get("brier", 0.0) or 0.0)
        for b in agg.values():
            n = max(1, int(b["runs"]))
            b["avg_accuracy"] = round(float(b["avg_accuracy"]) / n, 4)
            b["avg_brier"] = round(float(b["avg_brier"]) / n, 4)
        return {"name": name, "variants": agg, "total_runs": len(runs)}
