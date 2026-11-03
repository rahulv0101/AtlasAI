from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_PROMOTION_THRESHOLDS: dict[str, Any] = {
    "schema": "atlas_promotion_thresholds_v1",
    "version": 1,
    "online_learning_paths": {
        "fast_path": {
            "min_samples": 12,
            "min_evidence_strength": 0.78,
            "max_overreaction_risk": 0.35,
        },
        "medium_path": {
            "min_samples": 5,
            "min_evidence_strength": 0.5,
            "max_overreaction_risk": 0.65,
        },
        "lesson_digest": {
            "strong_min_samples": 12,
            "strong_min_evidence_strength": 0.78,
            "strong_max_overreaction_risk": 0.35,
            "moderate_min_samples": 5,
            "moderate_min_evidence_strength": 0.5,
        },
    },
    "meta_model_promotion": {
        "min_samples": 40,
        "min_accuracy_lift": 0.03,
        "min_brier_improvement": 0.01,
        "cooldown_hours": 24.0,
        "min_walkforward_folds": 3,
        "min_walkforward_accuracy_absolute": 0.5,
        "max_walkforward_brier_absolute": 1.0,
        "walkforward_accuracy_baseline_lift_fraction": 0.5,
        "walkforward_brier_improvement_fraction": 0.5,
    },
    "self_improvement_rollout": {
        "min_strength": 0.7,
        "min_samples": 40,
    },
    "evidence_strength_grades": {
        "strong_min_strength": 0.75,
        "moderate_min_strength": 0.5,
    },
}


def _project_knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "knowledge"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_promotion_thresholds(store: Any | None = None) -> dict[str, Any]:
    """Load versioned promotion thresholds from knowledge JSON.

    The JSON file is the inspectable policy source. Defaults exist only as a
    compatibility fallback for old stores or partial policy files.
    """
    knowledge_dir = Path(getattr(store, "knowledge_dir", _project_knowledge_dir()))
    path = knowledge_dir / "promotion_thresholds.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return _deep_merge(DEFAULT_PROMOTION_THRESHOLDS, dict(payload or {}))


def online_path_thresholds(store: Any | None = None) -> dict[str, Any]:
    return dict(load_promotion_thresholds(store).get("online_learning_paths", {}) or {})


def meta_model_promotion_thresholds(settings: Any | None = None, store: Any | None = None) -> dict[str, Any]:
    cfg = dict(load_promotion_thresholds(store).get("meta_model_promotion", {}) or {})
    if settings is not None:
        cfg = _deep_merge(
            {
                "min_samples": int(getattr(settings, "promotion_min_samples", cfg.get("min_samples", 40))),
                "min_accuracy_lift": float(
                    getattr(settings, "promotion_min_accuracy_lift", cfg.get("min_accuracy_lift", 0.03))
                ),
                "min_brier_improvement": float(
                    getattr(settings, "promotion_min_brier_improvement", cfg.get("min_brier_improvement", 0.01))
                ),
                "cooldown_hours": float(getattr(settings, "model_update_cooldown_hours", cfg.get("cooldown_hours", 24.0))),
                "min_walkforward_folds": int(
                    getattr(settings, "model_min_walkforward_folds", cfg.get("min_walkforward_folds", 3))
                ),
            },
            cfg,
        )
    return cfg


def self_improvement_rollout_thresholds(store: Any | None = None) -> dict[str, Any]:
    return dict(load_promotion_thresholds(store).get("self_improvement_rollout", {}) or {})


def evidence_strength_grade_thresholds(store: Any | None = None) -> dict[str, Any]:
    return dict(load_promotion_thresholds(store).get("evidence_strength_grades", {}) or {})
