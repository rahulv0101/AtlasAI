from types import SimpleNamespace

from atlas_ai.regime_filters import conservative_pretrade_filter_reason


def _settings(**overrides):
    base = {
        "conservative_filter_enabled": True,
        "conservative_filter_block_late_us_trend": True,
        "conservative_filter_block_post_spike": True,
        "conservative_filter_block_poor_liquidity": True,
        "conservative_filter_poor_liquidity_volume_ratio": 0.80,
        "conservative_filter_post_spike_atr_multiple": 1.6,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_conservative_filter_blocks_late_us_trend():
    reason = conservative_pretrade_filter_reason(
        _settings(),
        {"session": "late_us", "regime_name": "trend", "volume_ratio": 1.0, "spike_ratio": 1.0},
    )
    assert "late_us trend" in reason


def test_conservative_filter_blocks_post_spike():
    reason = conservative_pretrade_filter_reason(
        _settings(),
        {"session": "new_york", "regime_name": "range", "volume_ratio": 1.0, "spike_ratio": 1.7},
    )
    assert "post-spike" in reason


def test_conservative_filter_blocks_poor_liquidity():
    reason = conservative_pretrade_filter_reason(
        _settings(),
        {"session": "london", "regime_name": "range", "volume_ratio": 0.72, "spike_ratio": 1.0},
    )
    assert "poor liquidity" in reason


def test_conservative_filter_is_opt_in():
    reason = conservative_pretrade_filter_reason(
        _settings(conservative_filter_enabled=False),
        {"session": "late_us", "regime_name": "trend", "volume_ratio": 0.60, "spike_ratio": 2.0},
    )
    assert reason == ""
