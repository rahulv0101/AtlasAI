from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class AIReviewLedger:
    """Cross-process idempotency for reviews; legacy JSONL remains audit history."""

    def __init__(self, base: Path) -> None:
        self.path = base / "ai_review_evidence.sqlite3"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS reviews "
                "(decision_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS preference_evidence "
                "(decision_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def recent(self, limit: int) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT payload FROM reviews ORDER BY rowid DESC LIMIT ?", (max(1, int(limit)),)
            ).fetchall()
        return [json.loads(row[0]) for row in reversed(rows)]

    def preference_evidence(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT payload FROM preference_evidence ORDER BY rowid").fetchall()
        return [json.loads(row[0]) for row in rows]
