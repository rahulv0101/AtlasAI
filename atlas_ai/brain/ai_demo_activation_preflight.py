from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail

from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.promotion_governor import PromotionGovernor


class AIDemoActivationPreflight:
    """Validate demo activation readiness without activating demo."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def run(self, uid: str) -> dict[str, Any]:
        population = self._load_population()
        chromosome = next((row for row in population if row.uid == uid), None)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "executed": False}

        queue_item = self._find_queue_item(uid)
        packet = self._load_json(self.base / f"ai_demo_review_packet_{self._safe_uid(uid)}.json")
        proposal = self._load_json(self.base / f"ai_demo_activation_proposal_{self._safe_uid(uid)}.json")
        latest_review = self._latest_jsonl("ai_shadow_reviews.jsonl", uid)

        governor = PromotionGovernor()
        governor_ok, governor_reason = governor.can_promote_to_demo(chromosome, population, shadow_outcomes=None)
        promotion_payload = dict(latest_review.get("promotion_governor", {}) or {})
        checks = {
            "queue_status_demo_activation_proposed": str(queue_item.get("status", "")) == "demo_activation_proposed",
            "review_packet_available": bool(packet),
            "activation_proposal_available": bool(proposal),
            "candidate_stage_allows_demo": chromosome.promotion_stage in {"backtested", "shadow"},
            "candidate_not_already_demo_active": not bool(chromosome.demo_active),
            "candidate_shadow_passed": bool(chromosome.shadow_passed),
            "latest_shadow_review_available": bool(latest_review),
            "latest_review_allows_demo": bool(promotion_payload.get("can_promote_to_demo", False)),
            "governor_basic_check_passed": bool(governor_ok),
            "packet_does_not_promote_demo": bool(dict(packet.get("safety", {}) or {}).get("does_not_promote_demo", False)),
            "proposal_does_not_promote_demo": bool(dict(proposal.get("safety", {}) or {}).get("does_not_promote_demo", False)),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        result = {
            "schema": "atlas_ai_demo_activation_preflight_v1",
            "ok": not blockers,
            "uid": uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "executed": False,
            "checks": checks,
            "blockers": blockers,
            "governor_basic_reason": governor_reason,
            "queue_item": {
                "id": str(queue_item.get("id", "") or ""),
                "status": str(queue_item.get("status", "") or ""),
                "operator_note": str(queue_item.get("operator_note", "") or ""),
            },
            "candidate": {
                "promotion_stage": chromosome.promotion_stage,
                "promotion_reason": chromosome.promotion_reason,
                "demo_active": chromosome.demo_active,
                "shadow_passed": chromosome.shadow_passed,
                "backtest_score": chromosome.backtest_score,
            },
            "latest_shadow_review": {
                "recommendation": str(latest_review.get("recommendation", "") or ""),
                "promotion_governor": promotion_payload,
                "summary": dict(latest_review.get("summary", {}) or {}),
            },
            "operator_boundary": {
                "preflight_only": True,
                "does_not_call_promote_to_demo": True,
                "does_not_set_demo_active": True,
                "does_not_execute_trades": True,
                "does_not_promote_live": True,
                "requires_explicit_demo_activation_command": True,
            },
            "safety": {
                "executed": False,
                "does_not_execute": True,
                "does_not_promote_demo": True,
                "does_not_promote_live": True,
                "requires_promotion_governor": True,
                "requires_operator_command": True,
            },
        }
        path = self.base / f"ai_demo_activation_preflight_{self._safe_uid(uid)}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return {**result, "path": str(path)}

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
        payload = self._load_json(self.base / "ai_research_queue.json")
        candidates = [
            dict(item)
            for item in list(payload.get("items", []) or [])
            if str(dict(item).get("uid", "")) == str(uid)
            and str(dict(item).get("recommendation", "")) == "eligible_for_demo_review"
        ]
        candidates.sort(key=lambda row: str(row.get("updated_at", "")), reverse=True)
        return candidates[0] if candidates else {}

    def _latest_jsonl(self, filename: str, uid: str) -> dict[str, Any]:
        path = self.base / filename
        rows = read_jsonl_tail(path, 1, matcher=lambda row: str(row.get("uid", "")) == str(uid))
        return rows[-1] if rows else {}

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    @staticmethod
    def _safe_uid(uid: str) -> str:
        safe_uid = "".join(ch for ch in str(uid) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        return safe_uid or "unknown"
