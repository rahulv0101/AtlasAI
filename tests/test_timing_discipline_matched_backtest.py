from tools.timing_discipline_matched_backtest import _apply_mid_range_mixed_asia_variant, _summarize


def _timing(location="mid_range", momentum="mixed", session="asia", extra_missing=None):
    missing = ["tradable_location", "strong_or_turning_m1_momentum", *(extra_missing or [])]
    return {
        "passed": False,
        "hard_block": True,
        "side": "buy",
        "reason_code": "AWAITING_TRADABLE_LOCATION",
        "missing": missing,
        "reasons": ["entry_location_is_mid_range_or_unclear", "m1_momentum_not_confirmed"],
        "condition_values": {"observed": {"location": location, "momentum": momentum, "session": session}},
        "structured_result": {"passed": False, "reason_code": "AWAITING_TRADABLE_LOCATION"},
    }


def test_exact_mid_range_mixed_asia_combination_is_admitted():
    result = _apply_mid_range_mixed_asia_variant(_timing(), enabled=True)
    assert result["passed"] is True
    assert result["variant_match"] is True
    assert result["variant_admitted"] is True
    assert result["missing"] == []
    assert result["structured_result"]["passed"] is True


def test_variant_does_not_relax_other_combinations_or_other_checks():
    for field, value in (("location", "pullback"), ("momentum", "weak"), ("session", "london")):
        values = {"location": "mid_range", "momentum": "mixed", "session": "asia", field: value}
        original = _timing(**values)
        assert _apply_mid_range_mixed_asia_variant(original, enabled=True) == original
    result = _apply_mid_range_mixed_asia_variant(_timing(extra_missing=["structure_hold"]), enabled=True)
    assert result["passed"] is False
    assert result["missing"] == ["structure_hold"]


def test_summary_reports_path_drawdown():
    result = _summarize([{"r_multiple": 1, "strategy": "s"}, {"r_multiple": -2, "strategy": "s"},
                         {"r_multiple": .5, "strategy": "s"}])
    assert result["max_drawdown_r"] == 2
    assert result["max_drawdown_pct_at_configured_risk"] == 1
