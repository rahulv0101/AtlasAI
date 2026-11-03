from atlas_ai.learning.meta_model import SignalQualityModel, evaluate_binary_classifier, outcomes_to_samples


def test_outcomes_to_samples_filters_draws():
    outcomes = [
        {"outcome": "win", "side": "buy", "session": "london", "strategy": "ema", "hour_utc": 9, "net_pnl_zar": 100},
        {"outcome": "loss", "side": "sell", "session": "new_york", "strategy": "rsi", "hour_utc": 15, "net_pnl_zar": -120},
        {"outcome": "draw", "side": "buy", "session": "asia", "strategy": "breakout", "hour_utc": 2, "net_pnl_zar": 0},
    ]
    samples = outcomes_to_samples(outcomes)
    assert len(samples) == 2


def test_model_fit_and_eval_runs():
    outcomes = []
    for i in range(30):
        outcomes.append(
            {
                "outcome": "win" if i % 2 == 0 else "loss",
                "side": "buy" if i % 3 else "sell",
                "session": "london" if i % 2 == 0 else "new_york",
                "strategy": "ema_crossover" if i % 2 == 0 else "rsi_mean_reversion",
                "hour_utc": i % 24,
                "net_pnl_zar": 150 if i % 2 == 0 else -110,
            }
        )
    samples = outcomes_to_samples(outcomes)
    model = SignalQualityModel()
    model.fit(samples, epochs=5, lr=0.05)
    metrics = evaluate_binary_classifier(model, samples)
    assert metrics["samples"] == 30.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_context_features_include_richer_trade_context():
    samples = outcomes_to_samples(
        [
            {
                "outcome": "win",
                "side": "buy",
                "session": "london",
                "strategy": "gold_micro_impulse",
                "hour_utc": 9,
                "regime": "trend",
                "higher_tf_bias": "buy",
                "spread_points": 18.0,
                "confidence": 0.42,
                "same_side_support": 3,
                "same_side_ratio": 0.4,
                "m1_quality_factor": 1.05,
                "timeframe": "M1",
                "volatility_pct": 0.041,
                "trend_strength": 0.19,
                "body_ratio": 0.61,
                "volume_ratio": 1.22,
                "drift_delta": -0.12,
                "drift_active": True,
                "htf_aligned": True,
                "meta_probability": 0.63,
            }
        ]
    )
    feats = samples[0].features
    assert feats["regime:trend"] == 1.0
    assert feats["higher_tf_bias:buy"] == 1.0
    assert feats["timeframe:M1"] == 1.0
    assert feats["spread_bucket:tight"] == 1.0
    assert feats["body_ratio"] > 0
    assert feats["meta_probability"] > 0.5
