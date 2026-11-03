"""Isolated all-history collector equivalence and allocation benchmark."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_ai.learning import missed_entry_audit as audit


def run_case(records: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="atlas_collector_benchmark_") as directory:
        base = Path(directory)
        path = base / "missed_entry_outcome_reviews.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for index in range(records):
                batch = {"reviews": [
                    {"primary_blocker": "consensus", "outcome": [
                        "missed_opportunity", "correct_hold_or_block", "inconclusive", "other"
                    ][index % 4], "evidence": "x" * 8192},
                    {"primary_blocker": "other", "outcome": "missed_opportunity"},
                    {"primary_blocker": "consensus", "outcome": "reference_price_unavailable"},
                    {"primary_blocker": "consensus", "outcome": "direction_unavailable"},
                    None,
                ]}
                handle.write(json.dumps(batch) + "\n")
                if index % 100 == 0:
                    handle.write("\ninvalid json\n[]\nnull\n")
        with patch.object(audit, "_proposal_target_blocker", return_value="consensus"):
            tracemalloc.start()
            started = time.perf_counter()
            result = audit.build_blocker_selectivity_offline_collector(base)
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        result.pop("created_at")
        result.pop("path")
        return {"records": records, "input_bytes": path.stat().st_size,
                "elapsed_seconds": elapsed, "peak_allocated_bytes": peak, "result": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = {"cases": [run_case(500), run_case(4000)]}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["output_differences"] = [
            {"records": new["records"], "before": old["result"], "after": new["result"]}
            for old, new in zip(before["cases"], report["cases"], strict=True)
            if old["result"] != new["result"]
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("output_differences"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
