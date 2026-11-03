from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.promotion_governor import PromotionGovernor


class AIDemoActivationProposal:
    """Create a non-executing proposal for demo activation consideration."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def build(self, uid: str) -> dict[str, Any]:
        population = self._load_population()
        chromosome = next((row for row in population if row.uid == uid), None)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}

        queue_item = self._find_queue_item(uid)
        if str(queue_item.get("status", "")) != "demo_review_approved":
            return {
                "ok": False,
                "uid": uid,
                "reason": "demo_review_not_approved",
                "status": str(queue_item.get("status", "")),
                "executed": False,
            }

        packet = self._load_packet(uid)
        latest_review = self._latest_jsonl("ai_shadow_reviews.jsonl", uid)
        governor = PromotionGovernor()
        can_promote_without_shadow, reason_without_shadow = governor.can_promote_to_demo(
            chromosome,
            population,
            shadow_outcomes=None,
        )

        proposal = {
            "schema": "atlas_ai_demo_activation_proposal_v1",
            "ok": True,
            "uid": uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "executed": False,
            "recommendation": "operator_may_consider_demo_activation",
            "queue_item": queue_item,
            "packet": {
                "available": bool(packet),
                "path": str(self._packet_path(uid)),
                "safety": dict(packet.get("safety", {}) or {}) if packet else {},
            },
            "candidate": {
                "promotion_stage": chromosome.promotion_stage,
                "promotion_reason": chromosome.promotion_reason,
                "demo_active": chromosome.demo_active,
                "shadow_passed": chromosome.shadow_passed,
                "backtest_score": chromosome.backtest_score,
                "backtest_trades": chromosome.backtest_trades,
                "backtest_win_rate": chromosome.backtest_win_rate,
            },
            "latest_shadow_review": {
                "available": bool(latest_review),
                "recommendation": latest_review.get("recommendation", "") if latest_review else "",
                "promotion_governor": latest_review.get("promotion_governor", {}) if latest_review else {},
                "summary": latest_review.get("summary", {}) if latest_review else {},
            },
            "readiness": {
                "demo_review_approved": True,
                "packet_available": bool(packet),
                "candidate_shadow_passed": bool(chromosome.shadow_passed),
                "candidate_stage_allows_demo": chromosome.promotion_stage in {"backtested", "shadow"},
                "governor_basic_check_passed": bool(can_promote_without_shadow),
                "governor_basic_reason": reason_without_shadow,
            },
            "operator_boundary": {
                "requires_explicit_demo_activation_command": True,
                "does_not_call_promote_to_demo": True,
                "does_not_set_demo_active": True,
                "does_not_execute_trades": True,
                "does_not_promote_live": True,
            },
            "safety": {
                "proposal_only": True,
                "research_only": True,
                "executed": False,
                "does_not_execute": True,
                "does_not_promote_demo": True,
                "does_not_promote_live": True,
                "requires_promotion_governor": True,
                "requires_operator_command": True,
            },
        }
        path = self._proposal_path(uid)
        path.write_text(json.dumps(proposal, indent=2, sort_keys=True), encoding="utf-8")
        return {**proposal, "path": str(path)}

    def _load_population(self) -> list[StrategyChromosome]:
        path = self.base / "evolution_population.json"
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])]

    def _find_queue_item(self, uid: str) -> dict[str, Any]:
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
        candidates.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
        return candidates[0] if candidates else {}

    def _load_packet(self, uid: str) -> dict[str, Any]:
        path = self._packet_path(uid)
        if not path.exists():
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    def _latest_jsonl(self, filename: str, uid: str) -> dict[str, Any]:
        path = self.base / filename
        rows = read_jsonl_tail(path, 1, matcher=lambda row: str(row.get("uid", "")) == str(uid))
        return rows[-1] if rows else {}

    def _packet_path(self, uid: str) -> Path:
        return self.base / f"ai_demo_review_packet_{self._safe_uid(uid)}.json"

    def _proposal_path(self, uid: str) -> Path:
        return self.base / f"ai_demo_activation_proposal_{self._safe_uid(uid)}.json"

    @staticmethod
    def _safe_uid(uid: str) -> str:
        safe_uid = "".join(ch for ch in str(uid) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        return safe_uid or "unknown"
