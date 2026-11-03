"""Read retained trader events without loading the journal into memory."""
from __future__ import annotations

import hashlib
import json
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def category(message):
    text = message.lower()
    for label, terms in [
        ("timing_discipline", ["timing discipline", "waiting for clean entry", "timing-discipline"]),
        ("consensus", ["consensus guard"]),
        ("htf_alignment", ["higher-timeframe alignment"]),
        ("confidence_or_flat", ["no high-confidence", "no directional", "below threshold", "below execution threshold"]),
        ("loss_learning", ["loss-learning", "loss learning", "loss pattern", "loss-pattern"]),
        ("session", ["session", "outside trading", "weekend"]),
        ("risk_volatility", ["drawdown", "volatility", "spread", "safety lock", "loss limit"]),
        ("execution_gate", ["executiongate", "execution gate", "approval", "freshness"]),
    ]:
        if any(term in text for term in terms):
            return label
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-09-17", help="Split retained history at this ISO date/time")
    parser.add_argument("--output", type=Path, help="New diagnostic artifact; existing files are not overwritten")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "docs" / "diagnostics" / f"atlas_pipeline_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
    if output.exists():
        parser.error(f"Refusing to overwrite existing diagnostic: {output}")
    store = root / "storage"
    files = sorted((store / "archive").glob("events_*.jsonl")) + [store / "events.jsonl"]
    seen = set()
    groups = {}
    types = Counter()
    examples = {}
    executions = []
    first = last = ""
    errors = []
    for path in files:
        with path.open("rb") as stream:
            for line in stream:
                key = hashlib.sha256(line.strip()).digest()
                if key in seen:
                    continue
                seen.add(key)
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                stamp = str(event.get("time", ""))
                first = min(first, stamp) if first else stamp
                last = max(last, stamp)
                types[event.get("type", "")] += 1
                meta = event.get("meta") or {}
                chosen = meta.get("chosen") or {}
                msg = str(event.get("message", ""))
                if event.get("type") == "error":
                    errors.append({"time": stamp, "message": msg[:700]})
                if event.get("type") in {"execution", "failed", "decision_sample"}:
                    executions.append({"time": stamp, "type": event.get("type"), "message": msg, "chosen": chosen, "execution": meta.get("execution"), "strategy": meta.get("strategy")})
                if event.get("type") not in {"decision", "risk", "paper", "execution"} or not chosen or meta.get("timeframe", "M1") != "M1":
                    continue
                directional = chosen.get("side") in {"buy", "sell"}
                reason = category(msg)
                for name in ["all_retained", "since_cutoff" if stamp >= args.since else "before_cutoff"]:
                    group = groups.setdefault(name, {"cycles": 0, "directional": 0, "actions": Counter(), "reasons": Counter(), "directional_reasons": Counter(), "directional_messages": Counter(), "flat": 0, "max_confidence": 0, "pre_gate_over_120s": 0, "timed_cycles": 0})
                    group["cycles"] += 1
                    group["directional"] += int(directional)
                    group["flat"] += int(chosen.get("side") == "flat")
                    group["actions"][meta.get("action", "unknown")] += 1
                    group["reasons"][reason] += 1
                    if directional:
                        group["directional_reasons"][reason] += 1
                        group["directional_messages"][msg] += 1
                        group["max_confidence"] = max(group["max_confidence"], float(chosen.get("confidence", 0) or 0))
                        sample = {"time": stamp, "message": msg, "chosen": chosen, "threshold": meta.get("min_confidence_threshold"), "support": meta.get("same_side_support"), "htf": meta.get("higher_timeframe"), "timing": meta.get("timing_discipline")}
                        exkey = name + ":" + reason
                        if stamp > examples.get(exkey, {}).get("time", ""):
                            examples[exkey] = sample
                    timing = meta.get("freshness_timing") or {}
                    duration = timing.get("cycle_elapsed_ms_before_gates")
                    if duration is not None:
                        group["timed_cycles"] += 1
                        group["pre_gate_over_120s"] += int(duration > 120000)
    for group in groups.values():
        group["directional_messages"] = group["directional_messages"].most_common(15)
    state = json.loads((store / "learning_state.json").read_text(encoding="utf-8"))
    learning = state.get("state", {})
    learning_summary = {"updated_at": state.get("updated_at"), "keys": list(learning)}
    for key in ["loss_memory", "win_memory", "allocator", "meta_model", "training", "online_learning"]:
        value = learning.get(key, {})
        learning_summary[key] = {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in value.items()} if isinstance(value, dict) else str(value)[:200]
    result = {"captured_at": datetime.now(timezone.utc).isoformat(), "cutoff": args.since, "files": len(files), "deduplication": "SHA256 of stripped JSONL records; cycles are logged chosen-status decision/risk/paper/execution records, not distinct market opportunities. Confidence may be raw or conviction depending on the stage reached; do not compare stages as the same score.", "first": first, "last": last, "event_types": types, "groups": groups, "examples": examples, "execution_records": sorted(executions, key=lambda x: x["time"])[-30:], "recent_errors": sorted(errors, key=lambda x: x["time"])[-15:], "learning": learning_summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in {"examples", "learning"}}, indent=2))
    print("Saved", output)


if __name__ == "__main__":
    main()
