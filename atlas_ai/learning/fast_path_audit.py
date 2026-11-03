from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PATHS = ("slow_path", "medium_path", "fast_path")
NEXT_TRANSITIONS = {
    ("slow_path", "medium_path"): "slow_to_medium",
    ("medium_path", "fast_path"): "medium_to_fast",
    ("slow_path", "fast_path"): "slow_to_fast",
}
PROMOTION_READY_STATUSES = {"promotion_candidate"}
PROMOTED_STATUSES = {"promoted"}
PROMOTION_EVENTS = {
    "promotion_completed",
    "proposal_status_changed",
    "rollout_transitioned",
    "proposal_created",
    "proposal_updated",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except Exception:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_time(row: dict[str, Any]) -> datetime | None:
    for key in ("updated_at", "time", "created_at", "recorded_at", "generated_at"):
        parsed = _parse_time(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _in_window(row: dict[str, Any], cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    parsed = _event_time(row)
    return parsed is not None and parsed >= cutoff


def _normalize_path(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.endswith("_observe"):
        text = text[: -len("_observe")]
    if text in PATHS:
        return text
    if "fast_path" in text:
        return "fast_path"
    if "medium_path" in text:
        return "medium_path"
    if "slow_path" in text:
        return "slow_path"
    return ""


def _evidence_id(row: dict[str, Any]) -> str:
    for key in ("top_fingerprint", "fingerprint", "proposal_id", "insight_id", "pattern_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    meta = dict(row.get("meta", {}) or {})
    for key in ("top_fingerprint", "fingerprint", "proposal_id", "insight_id", "pattern_id"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            return value
    return "unknown"


def _load_online_observations(store: Any) -> list[dict[str, Any]]:
    if not hasattr(store, "load_learning_state"):
        return []
    state = dict(store.load_learning_state().get("state", {}) or {})
    online = dict(state.get("online_learning", {}) or {})
    rows = []
    latest = dict(online or {})
    if latest:
        rows.append(latest)
    rows.extend(dict(item or {}) for item in list(online.get("recent_observations", []) or []))
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        path = _normalize_path(row.get("mode") or row.get("path") or row.get("learning_path"))
        if not path:
            continue
        key = (_evidence_id(row), path, str(row.get("updated_at", "") or row.get("time", "") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({**row, "path": path})
    return deduped


def _load_self_improvement_audit(store: Any) -> list[dict[str, Any]]:
    path = Path(getattr(store, "self_improvement_audit_file", Path(getattr(store, "base", ".")) / "self_improvement_audit.jsonl"))
    if not path.exists():
        if hasattr(store, "tail_self_improvement_audit"):
            return list(store.tail_self_improvement_audit(limit=1000))
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    return rows


def _classify_promotion_event(row: dict[str, Any]) -> str:
    event = str(row.get("event", "") or "").strip()
    status = str(row.get("status", "") or "").strip().lower()
    meta = dict(row.get("meta", {}) or {})
    if event == "promotion_completed" or status in PROMOTED_STATUSES:
        return "promoted"
    if status in PROMOTION_READY_STATUSES or bool(meta.get("promotion_ready", False)):
        return "promotion_candidate"
    if event in {"proposal_created", "proposal_updated"} and bool(row.get("promotion_ready", False)):
        return "promotion_candidate"
    return ""


def build_fast_path_promotion_audit_report(
    store: Any,
    *,
    window_hours: float = 168.0,
    write_file: bool = False,
) -> dict[str, Any]:
    """Summarize learning-path movement without changing promotion behavior."""
    try:
        hours = max(0.0, float(window_hours))
    except Exception:
        hours = 168.0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours) if hours > 0 else None

    observations = [row for row in _load_online_observations(store) if _in_window(row, cutoff)]
    audit_rows = [row for row in _load_self_improvement_audit(store) if _in_window(row, cutoff)]

    path_counts = {path: {"entered": 0, "transitioned_to_next": 0, "reached_promotion": 0} for path in PATHS}
    by_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        path = str(row.get("path", "") or "")
        if path not in path_counts:
            continue
        path_counts[path]["entered"] += 1
        by_evidence[_evidence_id(row)].append(row)

    transition_counts = {name: 0 for name in NEXT_TRANSITIONS.values()}
    transition_examples: list[dict[str, Any]] = []
    for evidence_id, rows in by_evidence.items():
        ordered = sorted(rows, key=lambda item: _event_time(item) or datetime.min.replace(tzinfo=timezone.utc))
        seen_pairs: set[tuple[str, str]] = set()
        for previous, current in zip(ordered, ordered[1:]):
            pair = (str(previous.get("path", "") or ""), str(current.get("path", "") or ""))
            transition_name = NEXT_TRANSITIONS.get(pair)
            if not transition_name or pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            transition_counts[transition_name] += 1
            path_counts[pair[0]]["transitioned_to_next"] += 1
            transition_examples.append(
                {
                    "evidence_id": evidence_id,
                    "from": pair[0],
                    "to": pair[1],
                    "at": str(current.get("updated_at", "") or current.get("time", "") or ""),
                }
            )

    promotion_events: list[dict[str, Any]] = []
    promotion_counts = {"promotion_candidate": 0, "promoted": 0}
    promoted_evidence: set[str] = set()
    for row in audit_rows:
        if str(row.get("event", "") or "") not in PROMOTION_EVENTS:
            continue
        kind = _classify_promotion_event(row)
        if not kind:
            continue
        promotion_counts[kind] += 1
        evidence_id = _evidence_id(row)
        if kind == "promoted":
            promoted_evidence.add(evidence_id)
        promotion_events.append(
            {
                "event": str(row.get("event", "") or ""),
                "status": str(row.get("status", "") or ""),
                "evidence_id": evidence_id,
                "proposal_id": str(row.get("proposal_id", "") or ""),
                "time": str(row.get("time", "") or row.get("updated_at", "") or ""),
                "reason": str(row.get("reason", "") or ""),
            }
        )

    for evidence_id in promoted_evidence:
        latest_path = ""
        rows = by_evidence.get(evidence_id, [])
        if rows:
            latest_path = str(sorted(rows, key=lambda item: _event_time(item) or datetime.min.replace(tzinfo=timezone.utc))[-1].get("path", "") or "")
        if latest_path in path_counts:
            path_counts[latest_path]["reached_promotion"] += 1

    report = {
        "schema": "atlas_fast_path_promotion_audit_v1",
        "generated_at": _now_iso(),
        "window_hours": hours,
        "read_only": True,
        "path_counts": path_counts,
        "transitions": transition_counts,
        "transition_examples": transition_examples[:20],
        "promotion": {
            "promotion_candidate_events": promotion_counts["promotion_candidate"],
            "promoted_events": promotion_counts["promoted"],
            "events": promotion_events[:40],
        },
        "evidence_count": len(by_evidence),
        "observation_count": len(observations),
        "audit_event_count": len(audit_rows),
        "summary": (
            f"{len(observations)} learning observations across {len(by_evidence)} evidence fingerprints; "
            f"{sum(transition_counts.values())} path transitions; "
            f"{promotion_counts['promoted']} promotion completions in the selected window."
        ),
        "policy_note": "Observability only; this report does not change promotion logic, ranking weights, or execution policy.",
    }

    if write_file:
        base = Path(getattr(store, "base", "storage"))
        base.mkdir(parents=True, exist_ok=True)
        target = base / "fast_path_promotion_audit_report.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(target)
    return report
