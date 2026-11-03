from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from atlas_ai.config.settings import Settings
from atlas_ai.learning.experiment_tracker import ExperimentTracker
from atlas_ai.learning.meta_model import LabeledSample, SignalQualityModel, context_to_features, evaluate_binary_classifier, outcomes_to_samples
from atlas_ai.learning.promotion_thresholds import meta_model_promotion_thresholds, self_improvement_rollout_thresholds
from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.self_improvement import load_self_improvement_bundle, save_validation_report


def _parse_iso(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _learning_stability_report(state: dict[str, Any], settings: Settings) -> dict[str, Any]:
    outcomes = list(dict(state.get("loss_memory", {})).get("recent_outcomes", []))
    labeled = [row for row in outcomes if str(row.get("outcome", "draw")).lower() in {"win", "loss"}]
    if not labeled:
        return {"stable": True, "recent_win_rate": 0.0, "baseline_win_rate": 0.0, "delta": 0.0, "samples": 0}
    window = max(8, int(getattr(settings, "model_recent_window", 40)))
    baseline_slice = labeled[-max(window * 3, len(labeled)) :]
    recent_slice = labeled[-window:]
    baseline = sum(1 for row in baseline_slice if str(row.get("outcome", "")).lower() == "win") / max(1, len(baseline_slice))
    recent = sum(1 for row in recent_slice if str(row.get("outcome", "")).lower() == "win") / max(1, len(recent_slice))
    delta = recent - baseline
    max_delta = float(getattr(settings, "model_max_drift_delta", 0.18))
    return {
        "stable": abs(delta) <= max_delta,
        "recent_win_rate": round(recent, 4),
        "baseline_win_rate": round(baseline, 4),
        "delta": round(delta, 4),
        "samples": len(labeled),
        "recent_samples": len(recent_slice),
        "max_allowed_delta": round(max_delta, 4),
    }


def _walk_forward_metrics(samples: list[LabeledSample], settings: Settings, promotion_thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    n = len(samples)
    thresholds = dict(promotion_thresholds or meta_model_promotion_thresholds(settings))
    min_folds = max(1, int(thresholds["min_walkforward_folds"]))
    if n < 24:
        return {"folds": [], "fold_count": 0, "avg_accuracy": 0.0, "avg_brier": 1.0, "ok": False}
    val_size = max(6, n // 6)
    min_train = max(16, n // 2)
    folds: list[dict[str, float]] = []
    train_end = min_train
    while train_end + val_size <= n and len(folds) < max(min_folds + 1, 4):
        train = samples[:train_end]
        test = samples[train_end : train_end + val_size]
        model = SignalQualityModel()
        model.fit(train, epochs=40, lr=0.07, l2=0.0008)
        metrics = evaluate_binary_classifier(model, test)
        folds.append(
            {
                "train_samples": float(len(train)),
                "test_samples": float(len(test)),
                "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
                "brier": float(metrics.get("brier", 1.0) or 1.0),
            }
        )
        train_end += val_size
    if not folds:
        return {"folds": [], "fold_count": 0, "avg_accuracy": 0.0, "avg_brier": 1.0, "ok": False}
    avg_accuracy = sum(float(f["accuracy"]) for f in folds) / len(folds)
    avg_brier = sum(float(f["brier"]) for f in folds) / len(folds)
    return {
        "folds": folds,
        "fold_count": len(folds),
        "avg_accuracy": round(avg_accuracy, 4),
        "avg_brier": round(avg_brier, 4),
        "ok": len(folds) >= min_folds,
    }


def _model_registry(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("model_registry", {}) or {})


def build_joined_decision_outcome_samples(store: KnowledgeStore, max_lookback_hours: float = 18.0) -> list[LabeledSample]:
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
            short_id = decision_id[:6]
            if short_id:
                decision_aliases.setdefault(short_id, decision_id)
        decision_pool.append({"time": event_time, "meta": meta, "used": False})
    decision_pool.sort(key=lambda x: x["time"])

    max_gap = timedelta(hours=max(1.0, float(max_lookback_hours)))
    samples: list[LabeledSample] = []
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
        unknown_strategy = strategy in {"", "unknown"}
        side = str(outcome.get("side", "")).lower()
        best_idx = -1
        best_rank: tuple[int, int, float] | None = None
        if outcome_decision_id:
            for idx, candidate in enumerate(decision_pool):
                if candidate["used"]:
                    continue
                meta = dict(candidate.get("meta", {}) or {})
                decision_id = str(meta.get("decision_id", "") or "").strip().lower()
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
            if not unknown_strategy and c_strategy != strategy:
                continue
            cand_time = candidate["time"]
            if cand_time > outcome_time:
                continue
            gap = outcome_time - cand_time
            if gap > max_gap:
                continue
            same_strategy_rank = 0 if (not unknown_strategy and c_strategy == strategy) else 1
            same_session_rank = 0 if str(meta.get("session", "")).strip().lower() == str(outcome.get("session", "")).strip().lower() else 1
            rank = (same_strategy_rank, same_session_rank, gap.total_seconds())
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_idx = idx
        if best_idx == -1:
            continue
        decision_pool[best_idx]["used"] = True
        meta = dict(decision_pool[best_idx]["meta"])
        contextual = dict(meta.get("contextual_ranking", {}) or {})
        contextual_ctx = dict(contextual.get("context", {}) or {})
        merged = {
            "decision_id": meta.get("decision_id", outcome.get("decision_id", "")),
            "strategy": meta.get("strategy", outcome.get("strategy", "")),
            "side": meta.get("side", outcome.get("side", "")),
            "session": meta.get("session", outcome.get("session", "unknown")),
            "hour_utc": meta.get("hour_utc", outcome.get("hour_utc", -1)),
            "regime": meta.get("regime", "unknown"),
            "higher_tf_bias": meta.get("higher_tf_bias", "unknown"),
            "spread_points": meta.get("spread_points", 0.0),
            "confidence": meta.get("confidence", 0.0),
            "same_side_support": meta.get("same_side_support", 0),
            "same_side_ratio": meta.get("same_side_ratio", contextual_ctx.get("same_side_ratio", 0.0)),
            "m1_quality_factor": dict(meta.get("m1_quality", {}) or {}).get("factor", 1.0),
            "timeframe": meta.get("timeframe", "unknown"),
            "volatility_pct": meta.get("volatility_pct", contextual_ctx.get("volatility_pct", 0.0)),
            "trend_strength": meta.get("trend_strength", contextual_ctx.get("trend_strength", 0.0)),
            "body_ratio": contextual_ctx.get("body_ratio", 0.0),
            "volume_ratio": contextual_ctx.get("volume_ratio", 1.0),
            "spike_ratio": contextual_ctx.get("spike_ratio", 0.0),
            "drift_delta": contextual_ctx.get("drift_delta", 0.0),
            "drift_active": contextual_ctx.get("drift_active", False),
            "htf_aligned": contextual_ctx.get("htf_aligned", False),
            "meta_probability": dict(meta.get("meta_model", {}) or {}).get("probability", contextual_ctx.get("meta_probability", 0.5)),
            "setup_quality_score": dict(contextual.get("setup_quality", {}) or {}).get("score", 0.0),
            "setup_quality_factor": dict(contextual.get("setup_quality", {}) or {}).get("factor", 1.0),
        }
        label = 1 if out == "win" else 0
        samples.append(LabeledSample(features=context_to_features(merged), label=label))
    return samples


def readiness_report(store: KnowledgeStore) -> dict[str, Any]:
    state = store.load_learning_state().get("state", {})
    loss_memory = dict(state.get("loss_memory", {}))
    outcomes = list(loss_memory.get("recent_outcomes", []))
    decision_events = store.tail_events(limit=5000, event_type="decision_sample")
    joined_samples = build_joined_decision_outcome_samples(store)
    report = {
        "decision_samples": len(decision_events),
        "closed_outcomes": len(outcomes),
        "joined_samples": len(joined_samples),
        "joined_training_ready": len(joined_samples) >= 24,
        "meta_model_dataset_source": str(dict(state.get("meta_model", {})).get("dataset_source", "untrained")),
        "meta_model_status": str(dict(state.get("meta_model", {})).get("status", "untrained")),
    }
    report["learning_stability"] = _learning_stability_report(state, Settings(
        hfm_login=0,
        hfm_password="",
        hfm_server="",
        storage_dir=store.base.as_posix(),
    ))
    remaining = max(0, 24 - int(report["joined_samples"]))
    report["samples_needed_for_joined_training"] = remaining
    report["promotion_thresholds"] = meta_model_promotion_thresholds(store=store)
    return report


def run_meta_model_experiment(settings: Settings, store: KnowledgeStore) -> dict[str, Any]:
    state = store.load_learning_state().get("state", {})
    loss_memory = dict(state.get("loss_memory", {}))
    outcomes = list(loss_memory.get("recent_outcomes", []))
    joined_samples = build_joined_decision_outcome_samples(store)
    fallback_samples = outcomes_to_samples(outcomes)
    samples = joined_samples if len(joined_samples) >= 24 else fallback_samples
    dataset_source = "decision_outcome_joined" if samples is joined_samples else "outcome_context_fallback"

    if len(samples) < 24:
        result = {
            "ok": False,
            "reason": f"Not enough labeled outcomes for training ({len(samples)} < 24).",
            "samples": len(samples),
        }
        store.log_event("training", "Meta-model experiment skipped due to low sample count.", result)
        return result

    split = max(8, int(len(samples) * 0.2))
    train = samples[:-split] if len(samples) > split else samples
    test = samples[-split:] if len(samples) > split else samples
    promotion_thresholds = meta_model_promotion_thresholds(settings, store)
    walk_forward = _walk_forward_metrics(samples, settings, promotion_thresholds)
    stability = _learning_stability_report(state, settings)

    model = SignalQualityModel()
    model.fit(train, epochs=40, lr=0.07, l2=0.0008)
    metrics = evaluate_binary_classifier(model, test)

    version = datetime.now(timezone.utc).strftime("meta_%Y%m%dT%H%M%SZ")
    meta_model_payload = {
        "version": version,
        "type": "hashed_logistic_v1",
        "dataset_source": dataset_source,
        "joined_samples": len(joined_samples),
        "fallback_samples": len(fallback_samples),
        "trained_samples": len(train),
        "eval_samples": int(metrics.get("samples", 0)),
        "metrics": metrics,
        "walk_forward": walk_forward,
        "stability": stability,
        "model": model.to_dict(),
        "status": "candidate",
    }

    tracker = ExperimentTracker(settings.storage_dir)
    previous = tracker.list_runs(name="signal_quality_meta_model", limit=1)
    baseline_metrics = dict((previous[0].get("metrics", {}) if previous else {}) or {})
    baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0) or 0.0)
    baseline_brier = float(baseline_metrics.get("brier", 1.0) or 1.0)
    accuracy = float(metrics.get("accuracy", 0.0) or 0.0)
    brier = float(metrics.get("brier", 1.0) or 1.0)
    samples_n = int(len(samples))
    min_samples = max(1, int(promotion_thresholds["min_samples"]))
    min_acc_lift = float(promotion_thresholds["min_accuracy_lift"])
    min_brier_impr = float(promotion_thresholds["min_brier_improvement"])
    registry = _model_registry(state)
    champion = dict(registry.get("champion", {}) or {})
    last_promoted_at = _parse_iso(registry.get("last_promoted_at"))
    cooldown_hours = max(0.0, float(promotion_thresholds["cooldown_hours"]))
    cooldown_active = False
    cooldown_remaining_hours = 0.0
    if last_promoted_at is not None and cooldown_hours > 0:
        elapsed = (datetime.now(timezone.utc) - last_promoted_at).total_seconds() / 3600.0
        cooldown_remaining_hours = max(0.0, cooldown_hours - elapsed)
        cooldown_active = cooldown_remaining_hours > 0
    walk_ok = bool(walk_forward.get("ok", False))
    walk_accuracy = float(walk_forward.get("avg_accuracy", 0.0) or 0.0)
    walk_brier = float(walk_forward.get("avg_brier", 1.0) or 1.0)
    promoted = bool(
        samples_n >= min_samples
        and (accuracy - baseline_accuracy) >= min_acc_lift
        and (baseline_brier - brier) >= min_brier_impr
        and walk_ok
        and walk_accuracy
        >= max(
            float(promotion_thresholds["min_walkforward_accuracy_absolute"]),
            baseline_accuracy
            + (min_acc_lift * float(promotion_thresholds["walkforward_accuracy_baseline_lift_fraction"])),
        )
        and walk_brier
        <= min(
            float(promotion_thresholds["max_walkforward_brier_absolute"]),
            baseline_brier
            - (min_brier_impr * float(promotion_thresholds["walkforward_brier_improvement_fraction"])),
        )
        and bool(stability.get("stable", True))
        and not cooldown_active
    )
    if promoted:
        meta_model_payload["status"] = "promoted"
    registry_update = {
        **registry,
        "last_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "last_evaluation": {
            "dataset_source": dataset_source,
            "samples": samples_n,
            "metrics": metrics,
            "walk_forward": walk_forward,
            "stability": stability,
        },
        "challenger": {
            "version": version,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": dataset_source,
            "samples": samples_n,
            "metrics": metrics,
            "walk_forward": walk_forward,
            "stability": stability,
            "status": meta_model_payload["status"],
        },
    }
    if promoted:
        registry_update["previous_champion"] = champion
        registry_update["champion"] = {
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "dataset_source": dataset_source,
            "samples": samples_n,
            "metrics": metrics,
            "walk_forward": walk_forward,
            "stability": stability,
        }
        registry_update["last_promoted_at"] = datetime.now(timezone.utc).isoformat()
    store.update_learning_state(lambda existing: {**existing, "meta_model": meta_model_payload, "model_registry": registry_update})
    run = tracker.log_run(
        name="signal_quality_meta_model",
        variant="hashed_logistic_v1_promoted" if promoted else "hashed_logistic_v1_candidate",
        metrics={
            **metrics,
            "baseline_accuracy": baseline_accuracy,
            "baseline_brier": baseline_brier,
            "walk_forward_accuracy": walk_accuracy,
            "walk_forward_brier": walk_brier,
            "walk_forward_ok": walk_ok,
            "stability_delta": float(stability.get("delta", 0.0) or 0.0),
            "stability_ok": bool(stability.get("stable", True)),
            "cooldown_active": cooldown_active,
            "promoted": promoted,
            "dataset_source": dataset_source,
        },
        sample_size=len(samples),
        notes="Auto-run from experiment-report mode using joined decision/outcome dataset when available.",
    )
    summary = tracker.summarize("signal_quality_meta_model")

    result = {
        "ok": True,
        "run_id": run["id"],
        "samples_total": len(samples),
        "train_samples": len(train),
        "test_samples": len(test),
        "metrics": metrics,
        "dataset_source": dataset_source,
        "joined_samples": len(joined_samples),
        "fallback_samples": len(fallback_samples),
        "walk_forward": walk_forward,
        "stability": stability,
        "promotion": {
            "promoted": promoted,
            "baseline_accuracy": baseline_accuracy,
            "baseline_brier": baseline_brier,
            "min_samples": min_samples,
            "min_accuracy_lift": min_acc_lift,
            "min_brier_improvement": min_brier_impr,
            "cooldown_active": cooldown_active,
            "cooldown_remaining_hours": round(cooldown_remaining_hours, 2),
            "walk_forward_required": int(promotion_thresholds["min_walkforward_folds"]),
            "threshold_source": "knowledge/promotion_thresholds.json",
        },
        "summary": summary,
    }
    store.log_event("training", "Meta-model experiment completed.", result)
    return result


def build_self_improvement_validation_report(
    store: KnowledgeStore,
    proposal_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    bundle = load_self_improvement_bundle(store)
    proposals = list(dict(bundle.get("proposal_queue.json", {}) or {}).get("proposals", []) or [])
    proposal = next((row for row in proposals if str(row.get("proposal_id", "")).strip() == str(proposal_id).strip()), None)
    if proposal is None:
        report = {
            "proposal_id": str(proposal_id),
            "ok": False,
            "reason": "proposal_not_found",
        }
        save_validation_report(store, str(proposal_id), report)
        return report

    if settings is None:
        settings = Settings(
            hfm_login=0,
            hfm_password="",
            hfm_server="",
            storage_dir=store.base.as_posix(),
        )

    readiness = readiness_report(store)
    evidence = dict(proposal.get("evidence", {}) or {})
    validation = dict(proposal.get("validation", {}) or {})
    completed = list(validation.get("completed", []) or [])
    required = list(validation.get("required", []) or [])
    strength = float(evidence.get("strength", 0.0) or 0.0)
    sample_count = int(evidence.get("sample_count", 0) or 0)
    promotion_thresholds = self_improvement_rollout_thresholds(store)
    min_strength = float(promotion_thresholds["min_strength"])
    min_samples = max(1, int(promotion_thresholds["min_samples"]))
    report = {
        "proposal_id": str(proposal.get("proposal_id", "")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "category": str(proposal.get("category", "") or ""),
        "status": str(proposal.get("status", "") or ""),
        "evidence": {
            "strength": strength,
            "sample_count": sample_count,
            "grade": str(evidence.get("grade", "") or ""),
        },
        "readiness": readiness,
        "validation": {
            "required": required,
            "completed": completed,
            "missing": [item for item in required if item not in completed],
        },
        "promotion_readiness": {
            "meets_strength_threshold": strength >= min_strength,
            "meets_sample_threshold": sample_count >= min_samples,
            "joined_training_ready": bool(readiness.get("joined_training_ready", False)),
            "operator_approval_required": True,
            "ready_for_rollout_review": (
                strength >= min_strength
                and sample_count >= min_samples
            ),
            "threshold_source": "knowledge/promotion_thresholds.json",
        },
    }
    save_validation_report(store, str(proposal.get("proposal_id", "")), report)
    store.append_self_improvement_audit(
        {
            "event": "validation_completed",
            "actor": "atlas",
            "proposal_id": str(proposal.get("proposal_id", "")),
            "reason": "validation_report_generated",
            "meta": {
                "strength": strength,
                "sample_count": sample_count,
                "joined_training_ready": bool(readiness.get("joined_training_ready", False)),
            },
        }
    )
    return report
