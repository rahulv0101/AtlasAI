from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas_ai.memory.jsonl_tail import read_jsonl_tail


ALLOWED_STATUSES = {
    "pending_review",
    "reviewed",
    "deferred",
    "rejected",
    "paper_test_requested",
    "demo_review_requested",
    "demo_review_approved",
    "demo_activation_proposed",
    "demo_activated",
}


class AIResearchQueue:
    """Operator-reviewed research queue for AI promotion recommendations."""

    def __init__(self, storage_dir: str | Path = "storage") -> None:
        self.base = Path(storage_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / "ai_research_queue.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_payload()
        try:
            return self._normalize(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:
            return self._default_payload()

    def sync_from_promotion_report(self, report: dict[str, Any]) -> dict[str, Any]:
        payload = self.load()
        items = list(payload.get("items", []) or [])
        by_id = {str(item.get("id", "")): dict(item) for item in items}
        for rec in list(dict(report or {}).get("recommendations", []) or []):
            uid = str(rec.get("uid", "") or "").strip()
            recommendation = str(rec.get("recommendation", "") or "").strip()
            if not uid or not recommendation:
                continue
            item_id = f"{uid}:{recommendation}"
            existing = by_id.get(item_id, {})
            if str(existing.get("status", "pending_review")) in {"rejected", "paper_test_requested"}:
                continue
            by_id[item_id] = {
                **existing,
                "id": item_id,
                "uid": uid,
                "origin": str(rec.get("origin", "") or ""),
                "stage": str(rec.get("stage", "") or ""),
                "recommendation": recommendation,
                "confidence": float(rec.get("confidence", 0.0) or 0.0),
                "reasons": list(rec.get("reasons", []) or []),
                "blockers": list(rec.get("blockers", []) or []),
                "preference_score": float(rec.get("preference_score", 0.0) or 0.0),
                "status": str(existing.get("status", "pending_review") or "pending_review"),
                "operator_note": str(existing.get("operator_note", "") or ""),
                "created_at": str(existing.get("created_at", "") or datetime.now(timezone.utc).isoformat()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "safety": {
                    "research_only": True,
                    "does_not_promote_live": True,
                    "does_not_execute": True,
                    "paper_or_shadow_only": True,
                },
            }
        payload["items"] = sorted(by_id.values(), key=lambda row: (str(row.get("status")) != "pending_review", -float(row.get("confidence", 0.0) or 0.0), str(row.get("updated_at", ""))))
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        return self.save(payload)

    def sync_from_archive(self, *, limit: int = 12) -> dict[str, Any]:
        """Surface recently archived candidates as research-only review items."""
        archive_path = self.base / "evolution_archive.jsonl"
        if not archive_path.exists():
            return self.load()
        rows = read_jsonl_tail(archive_path, limit)

        payload = self.load()
        items = list(payload.get("items", []) or [])
        by_id = {str(item.get("id", "")): dict(item) for item in items}
        now = datetime.now(timezone.utc).isoformat()
        for candidate in rows:
            uid = str(candidate.get("uid", "") or "").strip()
            reject_reason = str(candidate.get("reject_reason", "") or "").strip()
            if not uid or not reject_reason:
                continue
            recommendation, confidence, reasons = self._archive_recommendation(candidate, reject_reason)
            item_id = f"{uid}:{recommendation}"
            existing = by_id.get(item_id, {})
            if str(existing.get("status", "pending_review")) in {"rejected", "paper_test_requested"}:
                continue
            by_id[item_id] = {
                **existing,
                "id": item_id,
                "uid": uid,
                "origin": str(candidate.get("origin", "") or ""),
                "stage": "archived",
                "recommendation": recommendation,
                "confidence": confidence,
                "reasons": reasons,
                "blockers": [reject_reason],
                "preference_score": 0.0,
                "status": str(existing.get("status", "pending_review") or "pending_review"),
                "operator_note": str(existing.get("operator_note", "") or ""),
                "created_at": str(existing.get("created_at", "") or now),
                "updated_at": now,
                "safety": {
                    "research_only": True,
                    "does_not_promote_live": True,
                    "does_not_execute": True,
                    "paper_or_shadow_only": True,
                },
            }

        payload["items"] = sorted(by_id.values(), key=lambda row: (str(row.get("status")) != "pending_review", -float(row.get("confidence", 0.0) or 0.0), str(row.get("updated_at", ""))))
        payload["updated_at"] = now
        return self.save(payload)

    @staticmethod
    def _archive_recommendation(candidate: dict[str, Any], reject_reason: str) -> tuple[str, float, list[str]]:
        trades = int(float(candidate.get("backtest_trades", 0) or 0))
        win_rate = float(candidate.get("backtest_win_rate", 0.0) or 0.0)
        profit_factor = float(candidate.get("backtest_profit_factor", 0.0) or 0.0)
        max_drawdown = float(candidate.get("backtest_max_drawdown", 1.0) or 1.0)
        if (
            reject_reason.startswith("insufficient_trades:")
            and trades > 0
            and win_rate >= 0.55
            and profit_factor >= 1.2
            and max_drawdown <= 0.05
        ):
            confidence = min(0.82, 0.45 + min(0.2, trades / 300.0) + min(0.12, (profit_factor - 1.2) * 0.04))
            return (
                "extend_backtest_sample",
                round(confidence, 3),
                [
                    reject_reason,
                    f"promising_under_sampled:trades={trades},win_rate={win_rate:.2f},profit_factor={profit_factor:.2f},drawdown={max_drawdown:.2%}",
                ],
            )
        return "review_archived_candidate", 0.35, [reject_reason]

    def update_item(self, item_id: str, status: str, *, note: str = "", operator: str = "operator") -> dict[str, Any]:
        if status not in ALLOWED_STATUSES:
            return {"ok": False, "reason": "invalid_status", "item_id": item_id}
        payload = self.load()
        items = list(payload.get("items", []) or [])
        for idx, item in enumerate(items):
            if str(item.get("id", "")) != str(item_id):
                continue
            updated = dict(item)
            updated["status"] = status
            updated["operator_note"] = str(note or updated.get("operator_note", "") or "")
            updated["reviewed_by"] = str(operator or "operator")
            updated["updated_at"] = datetime.now(timezone.utc).isoformat()
            history = list(updated.get("history", []) or [])
            history.append({
                "at": updated["updated_at"],
                "status": status,
                "operator": str(operator or "operator"),
                "note": str(note or ""),
            })
            updated["history"] = history[-20:]
            items[idx] = updated
            payload["items"] = items
            payload["updated_at"] = updated["updated_at"]
            saved = self.save(payload)
            return {"ok": True, "item": updated, "queue": saved, "executed": False}
        return {"ok": False, "reason": "item_not_found", "item_id": item_id}

    def attach_demo_review_packet(self, item_id: str, packet: dict[str, Any]) -> dict[str, Any]:
        """Attach generated packet metadata to a queue item without changing strategy state."""
        payload = self.load()
        items = list(payload.get("items", []) or [])
        now = datetime.now(timezone.utc).isoformat()
        for idx, item in enumerate(items):
            if str(item.get("id", "")) != str(item_id):
                continue
            updated = dict(item)
            updated["demo_review_packet"] = {
                "path": str(packet.get("path", "") or ""),
                "created_at": str(packet.get("created_at", "") or now),
                "schema": str(packet.get("schema", "") or ""),
                "executed": bool(packet.get("executed", False)),
                "does_not_promote_demo": bool(dict(packet.get("safety", {}) or {}).get("does_not_promote_demo", True)),
                "does_not_promote_live": bool(dict(packet.get("safety", {}) or {}).get("does_not_promote_live", True)),
            }
            updated["updated_at"] = now
            history = list(updated.get("history", []) or [])
            history.append({
                "at": now,
                "status": str(updated.get("status", "pending_review") or "pending_review"),
                "operator": "system",
                "note": "demo_review_packet_generated",
            })
            updated["history"] = history[-20:]
            items[idx] = updated
            payload["items"] = items
            payload["updated_at"] = now
            saved = self.save(payload)
            return {"ok": True, "item": updated, "queue": saved, "executed": False}
        return {"ok": False, "reason": "item_not_found", "item_id": item_id, "executed": False}

    def sync_from_extended_backtests(self, *, limit: int = 20) -> dict[str, Any]:
        results_path = self.base / "ai_extended_backtests.jsonl"
        if not results_path.exists():
            return self.load()
        rows = read_jsonl_tail(results_path, limit)

        payload = self.load()
        items = list(payload.get("items", []) or [])
        by_id = {str(item.get("id", "")): dict(item) for item in items}
        now = datetime.now(timezone.utc).isoformat()
        for result in rows:
            uid = str(result.get("uid", "") or "")
            recommendation = str(result.get("recommendation", "") or "")
            if not uid or not recommendation:
                continue
            metrics = dict(result.get("result", {}) or {})
            item_id = f"{uid}:{recommendation}"
            existing = by_id.get(item_id, {})
            by_id[item_id] = {
                **existing,
                "id": item_id,
                "uid": uid,
                "origin": str(result.get("origin", "") or ""),
                "stage": "extended_backtest",
                "recommendation": recommendation,
                "confidence": 0.78 if bool(result.get("passed_extended_sample", False)) else 0.48,
                "reasons": [
                    f"extended_sample:bars={int(result.get('bars', 0) or 0)},score={float(result.get('score', 0.0) or 0.0):.2f}",
                    f"trades={int(metrics.get('trades', 0) or 0)},win_rate={float(metrics.get('win_rate', 0.0) or 0.0):.2f},profit_factor={float(metrics.get('profit_factor', 0.0) or 0.0):.2f},drawdown={float(metrics.get('max_drawdown', 0.0) or 0.0):.2%}",
                ],
                "blockers": [str(result.get("reject_reason", "") or "")] if result.get("reject_reason") else [],
                "preference_score": 0.0,
                "status": str(existing.get("status", "pending_review") or "pending_review"),
                "operator_note": str(existing.get("operator_note", "") or ""),
                "created_at": str(existing.get("created_at", "") or now),
                "updated_at": now,
                "safety": {
                    "research_only": True,
                    "does_not_promote_live": True,
                    "does_not_execute": True,
                    "paper_or_shadow_only": True,
                },
            }

        payload["items"] = sorted(by_id.values(), key=lambda row: (str(row.get("status")) != "pending_review", -float(row.get("confidence", 0.0) or 0.0), str(row.get("updated_at", ""))))
        payload["updated_at"] = now
        return self.save(payload)

    def sync_from_shadow_reviews(self, *, limit: int = 20) -> dict[str, Any]:
        results_path = self.base / "ai_shadow_reviews.jsonl"
        if not results_path.exists():
            return self.load()
        rows = read_jsonl_tail(results_path, limit)

        payload = self.load()
        items = list(payload.get("items", []) or [])
        by_id = {str(item.get("id", "")): dict(item) for item in items}
        now = datetime.now(timezone.utc).isoformat()
        for result in rows:
            uid = str(result.get("uid", "") or "")
            recommendation = str(result.get("recommendation", "") or "")
            if not uid or not recommendation:
                continue
            summary = dict(result.get("summary", {}) or {})
            governor = dict(result.get("promotion_governor", {}) or {})
            drawdown_policy = str(governor.get("shadow_drawdown_policy", "") or "")
            drawdown_limit = governor.get("max_shadow_drawdown_pct")
            item_id = f"{uid}:{recommendation}"
            existing = by_id.get(item_id, {})
            reasons = [
                f"shadow_review:bars={int(result.get('bars', 0) or 0)},trades={int(summary.get('trades', 0) or 0)}",
                f"wins={int(summary.get('wins', 0) or 0)},losses={int(summary.get('losses', 0) or 0)},win_rate={float(summary.get('win_rate', 0.0) or 0.0):.2f}",
            ]
            if drawdown_policy:
                try:
                    reasons.append(f"drawdown_policy={drawdown_policy},limit={float(drawdown_limit):.2%}")
                except Exception:
                    reasons.append(f"drawdown_policy={drawdown_policy}")
            by_id[item_id] = {
                **existing,
                "id": item_id,
                "uid": uid,
                "origin": str(existing.get("origin", "") or ""),
                "stage": "shadow_review",
                "recommendation": recommendation,
                "confidence": 0.68 if recommendation == "eligible_for_demo_review" else 0.52,
                "reasons": reasons,
                "blockers": [str(governor.get("reason", "") or "")] if governor.get("reason") else [],
                "preference_score": float(existing.get("preference_score", 0.0) or 0.0),
                "status": str(existing.get("status", "pending_review") or "pending_review"),
                "operator_note": str(existing.get("operator_note", "") or ""),
                "created_at": str(existing.get("created_at", "") or now),
                "updated_at": now,
                "safety": {
                    "research_only": True,
                    "does_not_promote_live": True,
                    "does_not_execute": True,
                    "paper_or_shadow_only": True,
                },
            }

        payload["items"] = sorted(by_id.values(), key=lambda row: (str(row.get("status")) != "pending_review", -float(row.get("confidence", 0.0) or 0.0), str(row.get("updated_at", ""))))
        payload["updated_at"] = now
        return self.save(payload)

    def sync_from_shadow_risk_audits(self, *, limit: int = 20) -> dict[str, Any]:
        results_path = self.base / "ai_shadow_risk_audits.jsonl"
        if not results_path.exists():
            return self.load()
        rows = read_jsonl_tail(results_path, limit)

        payload = self.load()
        items = list(payload.get("items", []) or [])
        by_id = {str(item.get("id", "")): dict(item) for item in items}
        now = datetime.now(timezone.utc).isoformat()
        for audit in rows:
            uid = str(audit.get("uid", "") or "")
            if not uid:
                continue
            metrics = dict(audit.get("metrics", {}) or {})
            diagnosis = dict(audit.get("diagnosis", {}) or {})
            recommendation = f"repair_{str(diagnosis.get('suggested_repair_focus', 'shadow_risk'))}"
            item_id = f"{uid}:{recommendation}"
            existing = by_id.get(item_id, {})
            by_id[item_id] = {
                **existing,
                "id": item_id,
                "uid": uid,
                "origin": str(audit.get("origin", "") or existing.get("origin", "") or ""),
                "stage": "shadow_risk_audit",
                "recommendation": recommendation,
                "confidence": 0.64,
                "reasons": [
                    f"shadow_risk_audit:trades={int(metrics.get('trades', 0) or 0)},win_rate={float(metrics.get('win_rate', 0.0) or 0.0):.2f},net_pnl={float(metrics.get('net_pnl', 0.0) or 0.0):.2f}",
                    f"gross_normalized_drawdown={float(metrics.get('gross_normalized_drawdown_pct', 0.0) or 0.0):.2%},worst_loss_streak={int(metrics.get('worst_loss_streak', 0) or 0)}",
                ],
                "blockers": [str(diagnosis.get("primary", "") or "")],
                "preference_score": float(existing.get("preference_score", 0.0) or 0.0),
                "status": str(existing.get("status", "pending_review") or "pending_review"),
                "operator_note": str(existing.get("operator_note", "") or ""),
                "created_at": str(existing.get("created_at", "") or now),
                "updated_at": now,
                "safety": {
                    "research_only": True,
                    "does_not_promote_live": True,
                    "does_not_execute": True,
                    "paper_or_shadow_only": True,
                },
            }

        payload["items"] = sorted(by_id.values(), key=lambda row: (str(row.get("status")) != "pending_review", -float(row.get("confidence", 0.0) or 0.0), str(row.get("updated_at", ""))))
        payload["updated_at"] = now
        return self.save(payload)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize(payload)
        self.path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        return normalized

    @staticmethod
    def summary(payload: dict[str, Any]) -> dict[str, Any]:
        items = list(dict(payload or {}).get("items", []) or [])
        counts: dict[str, int] = {}
        for item in items:
            status = str(dict(item or {}).get("status", "pending_review") or "pending_review")
            counts[status] = counts.get(status, 0) + 1
        return {
            "count": len(items),
            "pending_review": counts.get("pending_review", 0),
            "reviewed": counts.get("reviewed", 0),
            "deferred": counts.get("deferred", 0),
            "rejected": counts.get("rejected", 0),
            "paper_test_requested": counts.get("paper_test_requested", 0),
            "demo_review_requested": counts.get("demo_review_requested", 0),
            "demo_review_approved": counts.get("demo_review_approved", 0),
            "demo_activation_proposed": counts.get("demo_activation_proposed", 0),
            "demo_activated": counts.get("demo_activated", 0),
        }

    @staticmethod
    def _default_payload() -> dict[str, Any]:
        return {
            "schema": "atlas_ai_research_queue_v1",
            "updated_at": "",
            "items": [],
            "policy": {
                "research_only": True,
                "does_not_promote_live": True,
                "does_not_execute": True,
                "requires_existing_promotion_governor": True,
            },
        }

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        base = AIResearchQueue._default_payload()
        raw = dict(payload or {})
        items = [dict(item or {}) for item in list(raw.get("items", []) or [])]
        for item in items:
            status = str(item.get("status", "pending_review") or "pending_review")
            item["status"] = status if status in ALLOWED_STATUSES else "pending_review"
            item.setdefault("safety", {
                "research_only": True,
                "does_not_promote_live": True,
                "does_not_execute": True,
                "paper_or_shadow_only": True,
            })
        return {
            "schema": "atlas_ai_research_queue_v1",
            "updated_at": str(raw.get("updated_at", "") or ""),
            "items": items,
            "policy": {**dict(base["policy"]), **dict(raw.get("policy", {}) or {})},
        }
