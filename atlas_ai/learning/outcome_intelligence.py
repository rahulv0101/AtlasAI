from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from atlas_ai.learning.self_improvement_scoring import score_evidence_strength
from atlas_ai.learning.meta_model import SignalQualityModel, context_to_features, evaluate_binary_classifier, outcomes_to_samples
from atlas_ai.learning.post_trade_lessons import build_post_trade_lesson
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import append_or_update_insight, append_or_update_proposal, load_self_improvement_bundle


def _parse_iso(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _volatility_bucket(value: Any) -> str:
    vol = float(value or 0.0)
    if vol < 0.02:
        return "low"
    if vol < 0.05:
        return "normal"
    return "high"


def _trend_bucket(value: Any) -> str:
    strength = float(value or 0.0)
    if strength < 0.12:
        return "weak"
    if strength < 0.28:
        return "moderate"
    return "strong"


def _confidence_bucket(value: Any) -> str:
    confidence = float(value or 0.0)
    if confidence < 0.30:
        return "low"
    if confidence < 0.50:
        return "medium"
    if confidence < 0.70:
        return "good"
    return "high"


def _safe_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "unknown"


def build_trade_outcome_memory(store: KnowledgeStore, max_lookback_hours: float = 18.0) -> list[dict[str, Any]]:
    state = store.load_learning_state().get("state", {})
    outcomes = list(dict(state.get("loss_memory", {})).get("recent_outcomes", []))
    decision_events = store.tail_events(limit=5000, event_type="decision_sample")
    decision_pool: list[dict[str, Any]] = []
    decision_aliases: dict[str, str] = {}

    for event in decision_events:
        event_time = _parse_iso(event.get("time"))
        meta = dict(event.get("meta", {}) or {})
        if event_time is None:
            continue
        if str(meta.get("side", "flat")).lower() not in {"buy", "sell"}:
            continue
        if "execution_success" in meta and not bool(meta.get("execution_success", False)):
            continue
        decision_id = str(meta.get("decision_id", "") or "").strip().lower()
        if decision_id:
            decision_aliases[decision_id] = decision_id
            decision_aliases.setdefault(decision_id[:6], decision_id)
        decision_pool.append({"time": event_time, "meta": meta, "used": False})
    decision_pool.sort(key=lambda x: x["time"])

    max_gap = timedelta(hours=max(1.0, float(max_lookback_hours)))
    records: list[dict[str, Any]] = []

    for outcome in sorted(outcomes, key=lambda x: str(x.get("time", ""))):
        out = str(outcome.get("outcome", "draw")).lower()
        if out not in {"win", "loss"}:
            continue
        outcome_time = _parse_iso(outcome.get("time"))
        if outcome_time is None:
            continue
        outcome_decision_id = str(outcome.get("decision_id", "") or "").strip().lower()
        if outcome_decision_id:
            outcome_decision_id = decision_aliases.get(outcome_decision_id, outcome_decision_id)
        strategy = str(outcome.get("strategy", "")).split("(", 1)[0].strip().lower()
        side = str(outcome.get("side", "")).lower()

        best_idx = -1
        best_rank: tuple[int, int, float] | None = None
        if outcome_decision_id:
            for idx, candidate in enumerate(decision_pool):
                if candidate["used"]:
                    continue
                decision_id = str(dict(candidate.get("meta", {}) or {}).get("decision_id", "") or "").strip().lower()
                if decision_id and decision_id == outcome_decision_id:
                    best_idx = idx
                    break
        for idx, candidate in enumerate(decision_pool):
            if best_idx != -1:
                break
            if candidate["used"]:
                continue
            meta = dict(candidate.get("meta", {}) or {})
            c_strategy = str(meta.get("strategy", "")).split("(", 1)[0].strip().lower()
            c_side = str(meta.get("side", "")).lower()
            if c_side != side:
                continue
            if strategy not in {"", "unknown"} and c_strategy != strategy:
                continue
            cand_time = candidate["time"]
            if cand_time > outcome_time:
                continue
            gap = outcome_time - cand_time
            if gap > max_gap:
                continue
            same_strategy_rank = 0 if c_strategy == strategy else 1
            same_session_rank = 0 if _safe_label(meta.get("session")) == _safe_label(outcome.get("session")) else 1
            rank = (same_strategy_rank, same_session_rank, gap.total_seconds())
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_idx = idx
        if best_idx == -1:
            continue

        decision_pool[best_idx]["used"] = True
        decision = dict(decision_pool[best_idx]["meta"] or {})
        contextual = dict(decision.get("contextual_ranking", {}) or {})
        contextual_context = dict(contextual.get("context", {}) or {})
        decision_time = decision_pool[best_idx]["time"]
        confidence = float(decision.get("confidence", 0.0) or 0.0)
        trend_strength = float(decision.get("trend_strength", contextual_context.get("trend_strength", 0.0)) or 0.0)
        volatility_pct = float(decision.get("volatility_pct", contextual_context.get("volatility_pct", 0.0)) or 0.0)
        regime = _safe_label(decision.get("regime", outcome.get("regime", "unknown")))
        higher_tf_bias = _safe_label(decision.get("higher_tf_bias", "unknown"))
        side = _safe_label(decision.get("side", outcome.get("side", "flat")))

        record = {
            "ticket": str(outcome.get("ticket", "") or ""),
            "decision_id": str(decision.get("decision_id", outcome.get("decision_id", "")) or ""),
            "strategy": str(decision.get("strategy", outcome.get("strategy", "unknown")) or "unknown"),
            "side": side,
            "session": _safe_label(decision.get("session", outcome.get("session", "unknown"))),
            "hour_utc": int(decision.get("hour_utc", outcome.get("hour_utc", -1)) or -1),
            "regime": regime,
            "higher_tf_bias": higher_tf_bias,
            "htf_aligned": side in {"buy", "sell"} and higher_tf_bias == side,
            "spread_points": float(decision.get("spread_points", 0.0) or 0.0),
            "confidence": confidence,
            "same_side_support": int(decision.get("same_side_support", 0) or 0),
            "same_side_ratio": float(decision.get("same_side_ratio", 0.0) or 0.0),
            "m1_quality_factor": float(dict(decision.get("m1_quality", {}) or {}).get("factor", 1.0) or 1.0),
            "timeframe": str(decision.get("timeframe", "unknown") or "unknown"),
            "volatility_pct": volatility_pct,
            "volatility_bucket": _volatility_bucket(volatility_pct),
            "trend_strength": trend_strength,
            "trend_bucket": _trend_bucket(trend_strength),
            "confidence_bucket": _confidence_bucket(confidence),
            "setup_quality_score": float(dict(contextual.get("setup_quality", {}) or {}).get("score", 0.0) or 0.0),
            "meta_probability": float(dict(decision.get("meta_model", {}) or {}).get("probability", 0.5) or 0.5),
            "decision_time": decision_time.isoformat(),
            "time": str(outcome.get("time", "") or ""),
            "duration_minutes": round(max(0.0, (outcome_time - decision_time).total_seconds() / 60.0), 2),
            "net_pnl_zar": float(outcome.get("net_pnl_zar", 0.0) or 0.0),
            "outcome": out,
        }
        record["post_trade_lesson"] = dict(outcome.get("post_trade_lesson", {}) or build_post_trade_lesson(record, decision))
        records.append(record)
    return records


def _aggregate_by_dimension(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = _safe_label(row.get(key, "unknown"))
        current = stats.setdefault(value, {"value": value, "samples": 0, "wins": 0, "losses": 0, "pnl_sum": 0.0, "avg_confidence_sum": 0.0})
        current["samples"] += 1
        current["wins"] += 1 if str(row.get("outcome", "")).lower() == "win" else 0
        current["losses"] += 1 if str(row.get("outcome", "")).lower() == "loss" else 0
        current["pnl_sum"] += float(row.get("net_pnl_zar", 0.0) or 0.0)
        current["avg_confidence_sum"] += float(row.get("confidence", 0.0) or 0.0)
    result = []
    for item in stats.values():
        samples = max(1, int(item["samples"]))
        result.append(
            {
                "value": item["value"],
                "samples": samples,
                "wins": int(item["wins"]),
                "losses": int(item["losses"]),
                "win_rate": round(float(item["wins"]) / samples, 4),
                "avg_pnl_zar": round(float(item["pnl_sum"]) / samples, 2),
                "avg_confidence": round(float(item["avg_confidence_sum"]) / samples, 4),
            }
        )
    result.sort(key=lambda x: (float(x["win_rate"]), float(x["avg_pnl_zar"]), int(x["samples"])), reverse=True)
    return result


def _pattern_summary(records: list[dict[str, Any]], outcome: str) -> dict[str, Any]:
    filtered = [row for row in records if str(row.get("outcome", "")).lower() == outcome]
    dimensions = ["regime", "session", "strategy", "volatility_bucket", "trend_bucket", "confidence_bucket"]
    top_patterns: dict[str, list[dict[str, Any]]] = {}
    for dim in dimensions:
        ranked = [row for row in _aggregate_by_dimension(filtered, dim) if int(row["samples"]) >= 2]
        top_patterns[dim] = ranked[:3]
    return {"samples": len(filtered), "top_patterns": top_patterns}


def _confidence_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    bins = [(0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 1.01)]
    rows: list[dict[str, Any]] = []
    for low, high in bins:
        bucket = [row for row in records if low <= float(row.get("confidence", 0.0) or 0.0) < high]
        if not bucket:
            rows.append({"range": [round(low, 2), round(min(high, 1.0), 2)], "samples": 0, "win_rate": 0.0, "avg_confidence": round((low + min(high, 1.0)) / 2.0, 3), "avg_pnl_zar": 0.0})
            continue
        samples = len(bucket)
        avg_conf = sum(float(row.get("confidence", 0.0) or 0.0) for row in bucket) / samples
        wins = sum(1 for row in bucket if str(row.get("outcome", "")).lower() == "win")
        avg_pnl = sum(float(row.get("net_pnl_zar", 0.0) or 0.0) for row in bucket) / samples
        rows.append(
            {
                "range": [round(low, 2), round(min(high, 1.0), 2)],
                "samples": samples,
                "win_rate": round(wins / samples, 4),
                "avg_confidence": round(avg_conf, 4),
                "avg_pnl_zar": round(avg_pnl, 2),
            }
        )
    calibration_error = 0.0
    weighted = 0
    for row in rows:
        samples = int(row["samples"])
        if samples <= 0:
            continue
        calibration_error += abs(float(row["win_rate"]) - float(row["avg_confidence"])) * samples
        weighted += samples
    return {
        "samples": len(records),
        "bins": rows,
        "mean_abs_gap": round(calibration_error / max(1, weighted), 4),
    }


def _regime_memory(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = _aggregate_by_dimension(records, "regime")
    favorable = [row["value"] for row in ranked if int(row["samples"]) >= 3 and float(row["win_rate"]) >= 0.55 and float(row["avg_pnl_zar"]) > 0.0]
    unfavorable = [row["value"] for row in ranked if int(row["samples"]) >= 3 and (float(row["win_rate"]) <= 0.4 or float(row["avg_pnl_zar"]) < 0.0)]
    return {"by_regime": ranked, "favorable_regimes": favorable[:3], "unfavorable_regimes": unfavorable[:3]}


def _comparison_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    winners = [row for row in records if str(row.get("outcome", "")).lower() == "win"]
    losers = [row for row in records if str(row.get("outcome", "")).lower() == "loss"]
    if not winners or not losers:
        return {
            "winner_vs_loser_deltas": {},
            "main_signal_gap": "Not enough mixed outcomes yet to compare winners and losers.",
        }
    winner_conf = sum(float(row.get("confidence", 0.0) or 0.0) for row in winners) / len(winners)
    loser_conf = sum(float(row.get("confidence", 0.0) or 0.0) for row in losers) / len(losers)
    winner_trend = sum(float(row.get("trend_strength", 0.0) or 0.0) for row in winners) / len(winners)
    loser_trend = sum(float(row.get("trend_strength", 0.0) or 0.0) for row in losers) / len(losers)
    winner_duration = sum(float(row.get("duration_minutes", 0.0) or 0.0) for row in winners) / len(winners)
    loser_duration = sum(float(row.get("duration_minutes", 0.0) or 0.0) for row in losers) / len(losers)
    return {
        "winner_vs_loser_deltas": {
            "confidence": round(winner_conf - loser_conf, 4),
            "trend_strength": round(winner_trend - loser_trend, 4),
            "duration_minutes": round(winner_duration - loser_duration, 2),
        },
        "main_signal_gap": (
            "Winning trades tend to appear with stronger trend strength and better confidence than losing trades."
            if winner_trend > loser_trend or winner_conf > loser_conf
            else "Current winner and loser patterns are still too similar to claim a strong signal-quality edge."
        ),
    }


def _signal_scoring_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = outcomes_to_samples(records)
    if len(samples) < 12:
        return {
            "status": "insufficient_samples",
            "samples": len(samples),
            "score_range": [0.0, 1.0],
            "inputs": ["trend_strength", "volatility", "session", "alignment", "confidence"],
        }
    model = SignalQualityModel()
    model.fit(samples, epochs=35, lr=0.07, l2=0.0008)
    metrics = evaluate_binary_classifier(model, samples)
    return {
        "status": "trained",
        "samples": len(samples),
        "score_range": [0.0, 1.0],
        "inputs": ["trend_strength", "volatility", "session", "alignment", "confidence", "spread", "support"],
        "metrics": {
            "accuracy": round(float(metrics.get("accuracy", 0.0) or 0.0), 4),
            "brier": round(float(metrics.get("brier", 1.0) or 1.0), 4),
        },
        "model": model.to_dict(),
        "feature_example": context_to_features(records[-1]) if records else {},
    }


def build_trade_outcome_intelligence(store: KnowledgeStore, max_lookback_hours: float = 18.0) -> dict[str, Any]:
    records = build_trade_outcome_memory(store, max_lookback_hours=max_lookback_hours)
    now = datetime.now().astimezone().isoformat()
    return {
        "generated_at": now,
        "sample_count": len(records),
        "outcome_memory": {
            "trades": records[-200:],
            "fields": [
                "regime",
                "volatility_pct",
                "volatility_bucket",
                "session",
                "strategy",
                "side",
                "confidence",
                "trend_strength",
                "htf_aligned",
                "duration_minutes",
                "net_pnl_zar",
                "outcome",
                "post_trade_lesson",
            ],
        },
        "winner_patterns": _pattern_summary(records, "win"),
        "loser_patterns": _pattern_summary(records, "loss"),
        "pattern_comparison": _comparison_summary(records),
        "signal_scoring_model": _signal_scoring_model(records),
        "confidence_calibration": _confidence_calibration(records),
        "regime_memory": _regime_memory(records),
    }


def _derive_self_improvement_candidates(intelligence: dict[str, Any]) -> list[dict[str, Any]]:
    sample_count = int(intelligence.get("sample_count", 0) or 0)
    if sample_count < 12:
        return []
    loser_patterns = dict(intelligence.get("loser_patterns", {}) or {}).get("top_patterns", {}) or {}
    trend_patterns = list(dict(loser_patterns).get("trend_bucket", []) or [])
    strategy_patterns = list(dict(loser_patterns).get("strategy", []) or [])
    comparison = dict(intelligence.get("pattern_comparison", {}) or {})
    deltas = dict(comparison.get("winner_vs_loser_deltas", {}) or {})

    candidates: list[dict[str, Any]] = []
    weak_trend = next((row for row in trend_patterns if str(row.get("value", "")).lower() == "weak"), None)
    if weak_trend and int(weak_trend.get("samples", 0) or 0) >= 3:
        candidate = {
            "type": "weak_trend_filter",
            "scope": {
                "strategy": str(strategy_patterns[0].get("value", "momentum_continuation") if strategy_patterns else "momentum_continuation"),
                "regime_bucket": "weak",
            },
            "reason": "Weak-trend conditions are clustering in recent losing outcomes.",
            "evidence": {
                "sample_count": int(weak_trend.get("samples", 0) or 0),
                "win_rate": float(weak_trend.get("win_rate", 0.0) or 0.0),
                "avg_pnl_zar": float(weak_trend.get("avg_pnl_zar", 0.0) or 0.0),
                "winner_vs_loser_deltas": deltas,
            },
        }
        candidates.append(candidate)
    return candidates


def sync_self_improvement_from_outcome_intelligence(store: KnowledgeStore, intelligence: dict[str, Any]) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    existing_proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    comparison = dict(intelligence.get("pattern_comparison", {}) or {})
    deltas = dict(comparison.get("winner_vs_loser_deltas", {}) or {})
    created_insights: list[str] = []
    created_proposals: list[str] = []
    for index, candidate in enumerate(_derive_self_improvement_candidates(intelligence), start=1):
        scope = dict(candidate.get("scope", {}) or {})
        scope_key = f"{scope.get('strategy', 'unknown')}::{scope.get('regime_bucket', 'unknown')}"
        insight_id = f"insight_outcome_{scope_key.replace('::', '_')}"
        evidence = dict(candidate.get("evidence", {}) or {})
        sample_count = int(evidence.get("sample_count", intelligence.get("sample_count", 0)) or 0)
        strength = score_evidence_strength(
            sample_size_score=min(1.0, sample_count / 40.0),
            persistence_score=0.7 if sample_count >= 6 else 0.45,
            cross_window_score=0.7,
            regime_alignment_score=0.8,
            risk_impact_score=0.7,
            overreaction_risk_score=0.2,
        )
        insight = append_or_update_insight(
            store,
            {
                "insight_id": insight_id,
                "type": "performance_pattern",
                "domain": "strategy_behavior",
                "scope": scope,
                "observation": str(candidate.get("reason", "") or ""),
                "evidence": {
                    **evidence,
                    **strength,
                },
                "recommendation_type": "proposal_candidate",
                "linked_proposal_id": "",
                "active_rule_candidate": False,
            },
        )
        created_insights.append(insight_id)
        store.append_self_improvement_audit(
            {
                "event": "insight_created",
                "actor": "atlas",
                "insight_id": insight_id,
                "reason": str(candidate.get("reason", "") or ""),
                "meta": {"strength": strength["strength"], "sample_count": sample_count},
            }
        )

        proposal_id = f"proposal_outcome_{scope_key.replace('::', '_')}"
        existing = next((row for row in existing_proposals if str(row.get("proposal_id", "")) == proposal_id), None)
        proposed_change = {
            "type": "threshold_adjustment_candidate",
            "target": "signal_regime_gate_min_trend_strength_pct",
            "suggested_direction": "tighten",
            "current_reference": 0.18,
            "suggested_value": 0.22,
        }
        proposal = append_or_update_proposal(
            store,
            {
                "proposal_id": proposal_id,
                "status": str(existing.get("status", "draft") if existing else "draft"),
                "category": "strategy_logic",
                "scope": {
                    "strategies": [str(scope.get("strategy", "momentum_continuation"))],
                    "instruments": ["XAUUSD"],
                    "sessions": [],
                },
                "source": {
                    "generator": "outcome_intelligence",
                    "linked_insight_ids": [insight_id],
                },
                "current_behavior": {
                    "summary": "Continuation logic allows weaker-trend participation under current thresholds."
                },
                "proposed_change": proposed_change,
                "reason": str(candidate.get("reason", "") or ""),
                "evidence": {
                    "strength": strength["strength"],
                    "grade": strength["grade"],
                    "sample_count": sample_count,
                    "winner_vs_loser_deltas": deltas,
                },
                "validation": {
                    "required": ["backtest", "shadow_compare", "regression_tests"],
                    "completed": list(dict(existing.get("validation", {}) or {}).get("completed", []) or []) if existing else [],
                },
                "approval": {
                    "required": True,
                    "approved": bool(dict(existing.get("approval", {}) or {}).get("approved", False)) if existing else False,
                    "approved_by": str(dict(existing.get("approval", {}) or {}).get("approved_by", "")) if existing else "",
                    "approved_at": str(dict(existing.get("approval", {}) or {}).get("approved_at", "")) if existing else "",
                },
                "rollout": {
                    "required": True,
                    "rollout_id": str(dict(existing.get("rollout", {}) or {}).get("rollout_id", "")) if existing else "",
                    "promotion_ready": bool(dict(existing.get("rollout", {}) or {}).get("promotion_ready", False)) if existing else False,
                },
                "rollback_plan": {
                    "restore_snapshot_id": str(dict(existing.get("rollback_plan", {}) or {}).get("restore_snapshot_id", "")) if existing else "",
                },
            },
        )
        created_proposals.append(proposal_id)
        store.append_self_improvement_audit(
            {
                "event": "proposal_created" if existing is None else "proposal_updated",
                "actor": "atlas",
                "proposal_id": proposal_id,
                "insight_id": insight_id,
                "category": "strategy_logic",
                "status": str(proposal.get("status", "") or ""),
                "reason": str(candidate.get("reason", "") or ""),
                "meta": {"evidence_strength": strength["strength"], "sample_count": sample_count},
            }
        )
    return {
        "created_insights": created_insights,
        "created_proposals": created_proposals,
    }


def refresh_trade_outcome_intelligence(store: KnowledgeStore, max_lookback_hours: float = 18.0) -> dict[str, Any]:
    intelligence = build_trade_outcome_intelligence(store, max_lookback_hours=max_lookback_hours)
    store.update_learning_state(lambda existing: {**existing, "trade_outcome_intelligence": intelligence})
    sync_summary = sync_self_improvement_from_outcome_intelligence(store, intelligence)
    intelligence["self_improvement_sync"] = sync_summary
    return intelligence
