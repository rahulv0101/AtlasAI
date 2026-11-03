from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable


def _reverse_lines(path: str | Path, *, chunk_size: int = 64 * 1024):
    target = Path(path)
    with target.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        remainder = b""
        while position > 0:
            read_size = min(max(1024, int(chunk_size)), position)
            position -= read_size
            handle.seek(position)
            lines = (handle.read(read_size) + remainder).splitlines()
            if position > 0:
                remainder = lines.pop(0) if lines else remainder
            for line in reversed(lines):
                if line.strip():
                    yield line


def tail_text_lines(path: str | Path, limit: int) -> list[str]:
    wanted = max(1, int(limit))
    rows: list[str] = []
    try:
        for raw in _reverse_lines(path):
            rows.append(raw.decode("utf-8", errors="replace"))
            if len(rows) >= wanted:
                break
    except FileNotFoundError:
        return []
    rows.reverse()
    return rows


def read_jsonl_tail(
    filepath: str | Path,
    limit: int,
    *,
    event_type_filter: str | None = None,
    matcher: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return the last matching JSON-object records without loading the file.

    Records are returned in their original chronological order. Malformed,
    blank, and non-object JSON lines are ignored.
    """
    wanted = max(1, int(limit))
    wanted_type = str(event_type_filter or "").strip().lower()
    rows: list[dict[str, Any]] = []
    try:
        for raw in _reverse_lines(filepath):
            try:
                value = json.loads(raw.strip())
            except Exception:
                continue
            if not isinstance(value, dict):
                continue
            row = dict(value)
            if wanted_type and str(row.get("type", "")).strip().lower() != wanted_type:
                continue
            if matcher is not None and not matcher(row):
                continue
            rows.append(row)
            if len(rows) >= wanted:
                break
    except FileNotFoundError:
        return []
    rows.reverse()
    return rows


# Backward-compatible name for migrations already using the extracted helper.
tail_jsonl = read_jsonl_tail
