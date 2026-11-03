from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail

from atlas_ai.brain.ai_research_queue import AIResearchQueue
from atlas_ai.strategies.dna import StrategyChromosome
from atlas_ai.strategies.promotion_governor import PromotionError, PromotionGovernor


class AIDemoActivator:
    """Activate a reviewed candidate for demo only, with an audit record."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def activate(self, uid: str, explicit_command: str, *, operator: str = "operator") -> dict[str, Any]:
        command = str(explicit_command or "").strip()
        if not self._command_allows_activation(command, uid):
            return {
                "ok": False,
                "uid": uid,
                "reason": "explicit_demo_activation_command_required",
                "state_changed": False,
                "trade_executed": False,
            }

        preflight = self._load_json(self.base / f"ai_demo_activation_preflight_{self._safe_uid(uid)}.json")
        if not bool(preflight.get("ok", False)) or list(preflight.get("blockers", []) or []):
            return {
                "ok": False,
                "uid": uid,
                "reason": "preflight_not_passed",
                "preflight": preflight,
                "state_changed": False,
                "trade_executed": False,
            }

        payload = self._load_population_payload()
        population = [StrategyChromosome.from_dict(row) for row in list(payload.get("chromosomes", []) or [])]
        chromosome = next((row for row in population if row.uid == uid), None)
        if chromosome is None:
            return {"ok": False, "uid": uid, "reason": "candidate_not_in_population", "state_changed": False, "trade_executed": False}

        queue_item = self._find_queue_item(uid)
        latest_review = self._latest_jsonl("ai_shadow_reviews.jsonl", uid)
        promotion_payload = dict(latest_review.get("promotion_governor", {}) or {})
        checks = {
            "queue_status_demo_activation_proposed": str(queue_item.get("status", "")) == "demo_activation_proposed",
            "candidate_stage_allows_demo": chromosome.promotion_stage in {"backtested", "shadow"},
            "candidate_not_already_demo_active": not bool(chromosome.demo_active),
            "candidate_shadow_passed": bool(chromosome.shadow_passed),
            "latest_shadow_review_available": bool(latest_review),
            "latest_review_allows_demo": bool(promotion_payload.get("can_promote_to_demo", False)),
        }
        blockers = [name for name, passed in checks.items() if not passed]
        if blockers:
            record = self._build_record(
                uid,
                command,
                operator,
                ok=False,
                state_changed=False,
                reason="activation_checks_failed",
                checks=checks,
                blockers=blockers,
                chromosome=chromosome,
                queue_item=queue_item,
                latest_review=latest_review,
            )
            path = self._write_record(uid, record)
            return {**record, "path": str(path)}

        governor = PromotionGovernor()
        try:
            governor.promote_to_demo(chromosome, population, shadow_outcomes=None)
        except PromotionError as exc:
            record = self._build_record(
                uid,
                command,
                operator,
                ok=False,
                state_changed=False,
                reason=f"promotion_governor_blocked:{exc}",
                checks={**checks, "promotion_governor_passed": False},
                blockers=["promotion_governor_passed"],
                chromosome=chromosome,
                queue_item=queue_item,
                latest_review=latest_review,
            )
            path = self._write_record(uid, record)
            return {**record, "path": str(path)}

        chromosome.promotion_reason = "demo_activated_by_explicit_operator_command"
        payload["chromosomes"] = [row.to_dict() for row in population]
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._population_path().write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        item_id = str(queue_item.get("id", "") or f"{uid}:eligible_for_demo_review")
        queue_update = AIResearchQueue(self.base).update_item(
            item_id,
            "demo_activated",
            note="demo_activated_by_explicit_operator_command",
            operator=operator,
        )
        record = self._build_record(
            uid,
            command,
            operator,
            ok=True,
            state_changed=True,
            reason="demo_activated",
            checks={**checks, "promotion_governor_passed": True},
            blockers=[],
            chromosome=chromosome,
            queue_item=queue_item,
            latest_review=latest_review,
            queue_update=queue_update,
        )
        path = self._write_record(uid, record)
        return {**record, "path": str(path)}

    def _build_record(
        self,
        uid: str,
        command: str,
        operator: str,
        *,
        ok: bool,
        state_changed: bool,
        reason: str,
        checks: dict[str, bool],
        blockers: list[str],
        chromosome: StrategyChromosome,
        queue_item: dict[str, Any],
        latest_review: dict[str, Any],
        queue_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "schema": "atlas_ai_demo_activation_v1",
            "ok": bool(ok),
            "uid": uid,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "explicit_command": command,
            "operator": str(operator or "operator"),
            "state_changed": bool(state_changed),
            "demo_activated": bool(ok and state_changed and chromosome.demo_active and chromosome.promotion_stage == "demo"),
            "trade_executed": False,
            "live_promoted": False,
            "checks": checks,
            "blockers": blockers,
            "candidate": {
                "promotion_stage": chromosome.promotion_stage,
                "promotion_reason": chromosome.promotion_reason,
                "demo_active": chromosome.demo_active,
                "shadow_passed": chromosome.shadow_passed,
                "backtest_score": chromosome.backtest_score,
                "demo_trades": chromosome.demo_trades,
            },
            "queue_item": {
                "id": str(queue_item.get("id", "") or ""),
                "status": str(queue_item.get("status", "") or ""),
            },
            "queue_update": queue_update or {},
            "latest_shadow_review": {
                "recommendation": str(latest_review.get("recommendation", "") or ""),
                "promotion_governor": dict(latest_review.get("promotion_governor", {}) or {}),
                "summary": dict(latest_review.get("summary", {}) or {}),
            },
            "safety": {
                "does_not_execute_trades": True,
                "does_not_promote_live": True,
                "requires_preflight_pass": True,
                "requires_operator_command": True,
                "promotion_governor_checked": True,
                "shadow_outcomes_source": "latest_shadow_review_and_preflight_artifact",
            },
        }

    def _load_population_payload(self) -> dict[str, Any]:
        path = self._population_path()
        if not path.exists():
            return {"chromosomes": []}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {"chromosomes": []}

    def _population_path(self) -> Path:
        return self.base / "evolution_population.json"

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

    def _write_record(self, uid: str, record: dict[str, Any]) -> Path:
        path = self.base / f"ai_demo_activation_{self._safe_uid(uid)}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        with (self.base / "ai_demo_activations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return path

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return {}

    @staticmethod
    def _command_allows_activation(command: str, uid: str) -> bool:
        normalized = " ".join(command.lower().split())
        expected = f"activate demo for {str(uid).lower()}"
        return normalized == expected

    @staticmethod
    def _safe_uid(uid: str) -> str:
        safe_uid = "".join(ch for ch in str(uid) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        return safe_uid or "unknown"
