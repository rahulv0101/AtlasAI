from __future__ import annotations

import json

from atlas_ai.learning.online_learning import _learning_mode
from atlas_ai.learning.promotion_thresholds import (
    load_promotion_thresholds,
    meta_model_promotion_thresholds,
    self_improvement_rollout_thresholds,
)
from atlas_ai.memory.knowledge_store import KnowledgeStore


def test_promotion_threshold_config_matches_legacy_path_decisions() -> None:
    thresholds = load_promotion_thresholds()

    assert thresholds["online_learning_paths"]["fast_path"] == {
        "min_samples": 12,
        "min_evidence_strength": 0.78,
        "max_overreaction_risk": 0.35,
    }
    assert thresholds["online_learning_paths"]["medium_path"] == {
        "min_samples": 5,
        "min_evidence_strength": 0.5,
        "max_overreaction_risk": 0.65,
    }
    assert thresholds["meta_model_promotion"] == {
        "min_samples": 40,
        "min_accuracy_lift": 0.03,
        "min_brier_improvement": 0.01,
        "cooldown_hours": 24.0,
        "min_walkforward_folds": 3,
        "min_walkforward_accuracy_absolute": 0.5,
        "max_walkforward_brier_absolute": 1.0,
        "walkforward_accuracy_baseline_lift_fraction": 0.5,
        "walkforward_brier_improvement_fraction": 0.5,
    }

    path_thresholds = thresholds["online_learning_paths"]
    assert _learning_mode(0.78, 0.35, 12, path_thresholds) == "fast_path_observe"
    assert _learning_mode(0.5, 0.65, 5, path_thresholds) == "medium_path_observe"
    assert _learning_mode(0.49, 0.65, 5, path_thresholds) == "slow_path_observe"


def test_store_promotion_threshold_file_overrides_without_code_change(tmp_path) -> None:
    store = KnowledgeStore(str(tmp_path))
    (tmp_path / "knowledge" / "promotion_thresholds.json").write_text(
        json.dumps(
            {
                "schema": "atlas_promotion_thresholds_v1",
                "meta_model_promotion": {"min_samples": 9, "cooldown_hours": 3.0},
                "self_improvement_rollout": {"min_strength": 0.8, "min_samples": 17},
            }
        ),
        encoding="utf-8",
    )

    meta_thresholds = meta_model_promotion_thresholds(store=store)
    rollout_thresholds = self_improvement_rollout_thresholds(store)

    assert meta_thresholds["min_samples"] == 9
    assert meta_thresholds["cooldown_hours"] == 3.0
    assert rollout_thresholds["min_strength"] == 0.8
    assert rollout_thresholds["min_samples"] == 17
