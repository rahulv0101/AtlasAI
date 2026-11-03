from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class LabeledSample:
    features: dict[str, float]
    label: int


class SignalQualityModel:
    """
    Lightweight hashed-feature logistic model.
    Keeps dependencies minimal while giving a probabilistic quality score.
    """

    def __init__(self, weights: dict[str, float] | None = None, bias: float = 0.0) -> None:
        self.weights = dict(weights or {})
        self.bias = float(bias)

    def _logit(self, features: dict[str, float]) -> float:
        z = self.bias
        for k, v in features.items():
            z += float(self.weights.get(k, 0.0)) * float(v)
        return z

    def predict_proba(self, features: dict[str, float]) -> float:
        z = self._logit(features)
        # Stable sigmoid.
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = math.exp(z)
        return ez / (1.0 + ez)

    def fit(self, samples: list[LabeledSample], epochs: int = 30, lr: float = 0.08, l2: float = 0.0005) -> None:
        if not samples:
            return
        positives = sum(1 for s in samples if int(s.label) > 0)
        negatives = max(0, len(samples) - positives)
        pos_weight = (len(samples) / max(1, 2 * positives)) if positives else 1.0
        neg_weight = (len(samples) / max(1, 2 * negatives)) if negatives else 1.0
        rng = random.Random(23)
        for _ in range(max(1, int(epochs))):
            order = list(samples)
            rng.shuffle(order)
            for s in order:
                y = 1.0 if int(s.label) > 0 else 0.0
                p = self.predict_proba(s.features)
                sample_weight = pos_weight if y > 0 else neg_weight
                err = (p - y) * sample_weight
                self.bias -= lr * err
                for k, v in s.features.items():
                    w = float(self.weights.get(k, 0.0))
                    grad = err * float(v) + l2 * w
                    self.weights[k] = w - lr * grad

    def to_dict(self) -> dict[str, Any]:
        return {
            "bias": float(self.bias),
            "weights": {k: float(v) for k, v in self.weights.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SignalQualityModel:
        return cls(weights=dict(payload.get("weights", {})), bias=float(payload.get("bias", 0.0)))


def context_to_features(context: dict[str, Any]) -> dict[str, float]:
    side = str(context.get("side", "flat")).lower()
    session = str(context.get("session", "unknown")).lower()
    strategy = str(context.get("strategy", "unknown")).split("(", 1)[0].strip().lower()
    hour = int(context.get("hour_utc", -1))
    regime = str(context.get("regime", "unknown")).lower()
    higher_tf_bias = str(context.get("higher_tf_bias", "unknown")).lower()
    timeframe = str(context.get("timeframe", "unknown")).upper()
    spread_points = float(context.get("spread_points", 0.0) or 0.0)
    confidence = float(context.get("confidence", 0.0) or 0.0)
    support = int(context.get("same_side_support", 0) or 0)
    m1_factor = float(context.get("m1_quality_factor", 1.0) or 1.0)
    volatility_pct = float(context.get("volatility_pct", 0.0) or 0.0)
    trend_strength = float(context.get("trend_strength", 0.0) or 0.0)
    body_ratio = float(context.get("body_ratio", 0.0) or 0.0)
    volume_ratio = float(context.get("volume_ratio", 1.0) or 1.0)
    same_side_ratio = float(context.get("same_side_ratio", 0.0) or 0.0)
    drift_delta = float(context.get("drift_delta", 0.0) or 0.0)
    meta_probability = float(context.get("meta_probability", 0.5) or 0.5)
    setup_quality_score = float(context.get("setup_quality_score", 0.0) or 0.0)
    setup_quality_factor = float(context.get("setup_quality_factor", 1.0) or 1.0)
    spike_ratio = float(context.get("spike_ratio", 0.0) or 0.0)
    htf_aligned = 1.0 if str(context.get("htf_aligned", "")).lower() in {"1", "true", "yes"} else 0.0
    drift_active = 1.0 if str(context.get("drift_active", "")).lower() in {"1", "true", "yes"} else 0.0

    if spread_points < 20:
        spread_bucket = "tight"
    elif spread_points < 45:
        spread_bucket = "normal"
    else:
        spread_bucket = "wide"

    if confidence < 0.18:
        confidence_bucket = "low"
    elif confidence < 0.35:
        confidence_bucket = "medium"
    else:
        confidence_bucket = "high"

    features: dict[str, float] = {
        "bias": 1.0,
        f"side:{side}": 1.0,
        f"session:{session}": 1.0,
        f"strategy:{strategy}": 1.0,
        f"hour_bucket:{max(0, hour // 4)}": 1.0 if hour >= 0 else 0.0,
        f"regime:{regime}": 1.0,
        f"higher_tf_bias:{higher_tf_bias}": 1.0,
        f"timeframe:{timeframe}": 1.0,
        f"spread_bucket:{spread_bucket}": 1.0,
        f"confidence_bucket:{confidence_bucket}": 1.0,
        "spread_norm": min(1.0, max(0.0, spread_points / 80.0)),
        "confidence_value": min(1.0, max(0.0, confidence)),
        "same_side_support_norm": min(1.0, max(0.0, support / 5.0)),
        "m1_quality_factor": min(1.5, max(0.5, m1_factor)),
        "volatility_pct_norm": min(1.0, max(0.0, volatility_pct / 0.12)),
        "trend_strength_norm": min(1.0, max(0.0, trend_strength / 0.45)),
        "body_ratio": min(1.0, max(0.0, body_ratio)),
        "volume_ratio_norm": min(2.0, max(0.0, volume_ratio)),
        "same_side_ratio": min(1.0, max(0.0, same_side_ratio)),
        "drift_delta_shifted": max(0.0, min(1.0, (drift_delta + 0.5) / 1.0)),
        "meta_probability": min(1.0, max(0.0, meta_probability)),
        "setup_quality_score_shifted": max(0.0, min(1.0, (setup_quality_score + 0.25) / 0.5)),
        "setup_quality_factor": min(1.5, max(0.5, setup_quality_factor)),
        "spike_ratio_norm": min(2.5, max(0.0, spike_ratio)),
        "htf_aligned": htf_aligned,
        "drift_active": drift_active,
    }
    return features


def outcomes_to_samples(outcomes: list[dict[str, Any]]) -> list[LabeledSample]:
    samples: list[LabeledSample] = []
    for row in outcomes:
        out = str(row.get("outcome", "draw")).lower()
        if out not in {"win", "loss"}:
            continue
        label = 1 if out == "win" else 0
        samples.append(LabeledSample(features=context_to_features(row), label=label))
    return samples


def evaluate_binary_classifier(model: SignalQualityModel, samples: list[LabeledSample]) -> dict[str, float]:
    if not samples:
        return {"samples": 0.0, "accuracy": 0.0, "brier": 1.0}
    correct = 0
    brier = 0.0
    for s in samples:
        p = model.predict_proba(s.features)
        y = 1.0 if s.label else 0.0
        pred = 1 if p >= 0.5 else 0
        if pred == s.label:
            correct += 1
        brier += (p - y) ** 2
    n = len(samples)
    return {
        "samples": float(n),
        "accuracy": float(correct) / n,
        "brier": float(brier) / n,
    }
