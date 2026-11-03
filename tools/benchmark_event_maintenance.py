"""Isolated memory/output comparison for KnowledgeStore journal maintenance."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_ai.memory.knowledge_store import KnowledgeStore


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def benchmark(rows: int, payload_bytes: int, retained: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="atlas-maintenance-benchmark-") as directory:
        store = KnowledgeStore(directory)
        with store.events_file.open("w", encoding="utf-8") as stream:
            for index in range(rows):
                row = {"time": f"fixed-{index}", "type": "fixture", "message": str(index),
                       "meta": {"payload": "x" * payload_bytes}}
                stream.write(json.dumps(row) + "\n")
                if index % 101 == 0:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                if index % 203 == 0:
                    stream.write("\n{malformed\n")
        size = store.events_file.stat().st_size
        gc.collect()
        tracemalloc.start()
        started = time.perf_counter()
        try:
            result = store.maintain_events(max_lines=retained)
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        archives = sorted((store.base / "archive").glob("events_*.jsonl"))
        return {"rows": rows, "payload_bytes": payload_bytes, "retained": retained,
                "input_bytes": size, "elapsed_seconds": round(elapsed, 3),
                "peak_python_bytes": peak, "result": result,
                "active_sha256": digest(store.events_file),
                "archive_sha256": [digest(path) for path in archives]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = {"captured_at": datetime.now(timezone.utc).isoformat(),
              "scope": "synthetic temp stores only; no trading or live journal writes",
              "cases": [benchmark(500, 8192, 300), benchmark(4000, 8192, 3000)]}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        differences = []
        for old, new in zip(before["cases"], result["cases"], strict=True):
            for key in ("rows", "payload_bytes", "retained", "result", "active_sha256", "archive_sha256"):
                if old[key] != new[key]:
                    differences.append({"rows": new["rows"], "field": key, "before": old[key], "after": new[key]})
        result["output_differences"] = differences
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
