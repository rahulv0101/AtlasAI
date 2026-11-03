from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.evolution_reporter import EvolutionReporter


class AIDemoReviewPacket:
    """Build a research-only handoff packet for a demo-review candidate."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def build(self, uid: str) -> dict[str, Any]:
        population = self._load_population()
        chromosome = next((row for row in population if row.uid == uid), None)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}

        queue_item = self._find_demo_review_item(uid)
        latest_review = self._latest_jsonl("ai_shadow_reviews.jsonl", uid)
        latest_audit = self._latest_jsonl("ai_shadow_risk_audits.jsonl", uid)
        detail = EvolutionReporter(self.base).strategy_detail(chromosome)

        packet = {
            "schema": "atlas_ai_demo_review_packet_v1",
            "ok": True,
            "uid": uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "executed": False,
            "stage": chromosome.promotion_stage,
            "recommendation": "operator_demo_review",
            "queue_item": queue_item,
            "strategy": {
                "origin": chromosome.origin,
                "promotion_stage": chromosome.promotion_stage,
                "promotion_reason": chromosome.promotion_reason,
                "shadow_passed": chromosome.shadow_passed,
                "compact_summary": detail.get("compact_summary", ""),
            },
            "backtest": detail.get("backtest", {}),
            "shadow_review": {
                "available": bool(latest_review),
                "recommendation": latest_review.get("recommendation", "") if latest_review else "",
                "summary": latest_review.get("summary", {}) if latest_review else {},
                "promotion_governor": latest_review.get("promotion_governor", {}) if latest_review else {},
            },
            "risk_audit": {
                "available": bool(latest_audit),
                "metrics": latest_audit.get("metrics", {}) if latest_audit else {},
                "diagnosis": latest_audit.get("diagnosis", {}) if latest_audit else {},
                "guard": latest_audit.get("guard", {}) if latest_audit else {},
            },
            "operator_checklist": [
                "Confirm the intended shadow drawdown policy before any demo activation.",
                "Confirm the candidate remains paper/shadow-only until explicit demo promotion.",
                "Review raw peak drawdown sensitivity versus gross-normalized drawdown.",
                "Confirm no live promotion is requested or implied by this packet.",
                "Confirm ExecutionGate remains the only execution authority.",
            ],
            "safety": {
                "research_only": True,
                "does_not_execute": True,
                "does_not_promote_demo": True,
                "does_not_promote_live": True,
                "requires_operator_review": True,
                "requires_existing_promotion_governor": True,
            },
        }
        path = self._packet_path(uid)
        path.write_text(json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8")
        return {**packet, "path": str(path)}

    def _packet_path(self, uid: str) -> Path:
        safe_uid = "".join(ch for ch in str(uid) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        return self.base / f"ai_demo_review_packet_{safe_uid or 'unknown'}.json"

    def _load_population(self) -> list[StrategyChromosome]:
        path = self.base / "evolution_population.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])]

    def _find_demo_review_item(self, uid: str) -> dict[str, Any]:
        path = self.base / "ai_research_queue.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        candidates = [
            dict(item)
            for item in list(payload.get("items", []) or [])
            if str(dict(item).get("uid", "")) == str(uid)
            and str(dict(item).get("recommendation", "")) == "eligible_for_demo_review"
        ]
        if not candidates:
            return {}
        candidates.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
        return candidates[0]

    def _latest_jsonl(self, filename: str, uid: str) -> dict[str, Any]:
        path = self.base / filename
        rows = read_jsonl_tail(path, 1, matcher=lambda row: str(row.get("uid", "")) == str(uid))
        return rows[-1] if rows else {}
