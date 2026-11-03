from __future__ import annotations

from types import SimpleNamespace

from atlas_ai.analysis.decision_analysis import build_structured_decision_analysis
from atlas_ai.system_health import build_system_health


def _settings(**overrides):
    base = dict(
        timeframe_minutes=1,
        execute_trades=True,
        gemini_influence_enabled=False,
        meta_model_enabled=True,
        research_regime_filter_enabled=False,
        evolution_enabled=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_safety_lock_is_protective_not_fail():
    status = {
        "execute_trades": True,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "safety_layer": {"safety_lock_active": True, "safety_lock_reason": "rolling_expectancy_degradation", "recovery_ready": False, "recovery_blockers": ["safety_lock_active"]},
    }
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    assert "safety_lock_active" in out["classification"]["PROTECTIVE_LOCK"]
    assert "safety_lock_active" not in out["classification"]["FAIL"]


def test_historical_lock_metadata_present():
    status = {
        "execute_trades": False,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "safety_layer": {"safety_lock_active": True, "safety_lock_reason": "rolling_expectancy_degradation"},
    }
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    sl = out["safety_layer"]
    assert sl["safety_lock_source"] in {"historical", "expectancy"}
    assert "safety_lock_triggering_window" in sl


def test_cooldown_coherence_when_lock_active():
    status = {
        "execute_trades": False,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "safety_layer": {"safety_lock_active": True, "cooldown_active": False, "cooldown_level": "none"},
    }
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    assert out["safety_layer"]["cooldown_status"] == "not_applicable_due_to_safety_lock"
    assert out["safety_layer"]["safety_lock_supersedes_cooldown"] is True


def test_recovery_object_includes_required_actions():
    status = {
        "execute_trades": False,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "safety_layer": {"safety_lock_active": True, "recovery_ready": False, "recovery_blockers": ["safety_lock_active"]},
    }
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    rec = out["safety_layer"]["recovery"]
    assert rec["recovery_ready"] is False
    assert len(rec["required_actions"]) >= 1
    assert rec["manual_unlock_endpoint"] == "/api/safety/unlock"


def test_ai_influence_is_observer_or_proposal_only():
    status = {"execute_trades": False, "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}}}
    out = build_system_health(status, settings=_settings(), runtime_flags={"gemini_influence_enabled": True})
    assert out["ai_influence_mode"] in {"observer_only", "proposal_only"}
    assert out["ai_execution_authority"] == "execution_authority_forbidden"


def test_session_derived_from_utc_when_unknown():
    out = build_structured_decision_analysis(
        status={"action": "hold", "chosen": {"confidence": 0.4, "side": "flat"}, "context_snapshot": {"session": "unknown"}, "setup_family": "no_trade"},
        runtime={},
    )
    assert out["context_alignment"]["session"]["state"] in {"asia", "london", "new_york", "overlap", "off_hours"}


def test_macro_unknown_maps_to_unavailable():
    out = build_structured_decision_analysis(
        status={"action": "hold", "chosen": {"confidence": 0.4, "side": "flat"}, "context_snapshot": {}, "setup_family": "no_trade"},
        runtime={},
    )
    assert out["context_alignment"]["dxy"]["availability"] == "unavailable"
    assert out["context_alignment"]["us10y"]["availability"] == "unavailable"


def test_regime_price_only_fallback_in_system_health():
    status = {"execute_trades": False, "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}}, "regime": {"name": "unknown", "confidence": 0}}
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    assert out["regime_state"]["regime_source"] == "price_only_inferred"


def test_spread_fields_are_normalized_and_explained():
    out = build_structured_decision_analysis(
        status={"action": "hold", "chosen": {"confidence": 0.4, "side": "flat"}, "context_snapshot": {"spread_state": "tight", "liquidity_state": "unknown", "spread_points": 36}, "setup_family": "no_trade"},
        runtime={},
    )
    sl = out["context_alignment"]["spread_liquidity"]
    assert "spread_points" in sl and "spread_pips" in sl and "spread_explanation" in sl


def test_evolution_pool_empty_reports_fallback():
    status = {"execute_trades": False, "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}}, "learning_status_snapshot": {"evolved_pool_count": 0}}
    out = build_system_health(status, settings=_settings(evolution_enabled=True), runtime_flags={})
    assert out["evolution_state"]["evolution_status"] == "fallback_builtin"


def test_optional_filters_disabled_are_info():
    status = {"execute_trades": False, "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}}, "conservative_pretrade_filter": {"enabled": False}}
    out = build_system_health(status, settings=_settings(research_regime_filter_enabled=False), runtime_flags={})
    assert out["filter_status"]["conservative_filter"]["severity"] == "INFO"
    assert out["filter_status"]["research_filter"]["severity"] == "INFO"


def test_report_categories_consistent():
    status = {"execute_trades": True, "health": {"broker": {"connected": False, "symbol_selected": False}, "llm": {}}, "safety_layer": {"safety_lock_active": True}}
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    assert isinstance(out["classification"]["PASS"], list)
    assert isinstance(out["classification"]["PROTECTIVE_LOCK"], list)
    assert isinstance(out["classification"]["WARNING"], list)
    assert isinstance(out["classification"]["FAIL"], list)


def test_macro_unknown_is_unavailable_with_inferred_source():
    status = {
        "execute_trades": False,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "context_snapshot": {
            "dxy_alignment": "unknown",
            "us10y_alignment": "",
            "vix_regime": "unknown",
            "oil_backdrop": "unknown",
        },
    }
    out = build_system_health(status, settings=_settings(), runtime_flags={})
    assert out["macro_data"]["dxy_alignment"]["availability"] == "unavailable"
    assert out["macro_data"]["dxy_alignment"]["source"] == "price_only_inferred"
    assert out["macro_data"]["us10y_alignment"]["availability"] == "unavailable"


def test_guarded_demo_mode_label_is_exposed():
    status = {
        "execute_trades": True,
        "health": {"broker": {"connected": True, "symbol_selected": True}, "llm": {}},
        "mode_label": "LEARNING_AGGRESSIVE_DEMO",
    }
    runtime = {"startup_verification": {"profile": "guarded_demo", "execute_trades": True}}
    out = build_system_health(status, settings=_settings(), runtime_flags=runtime)
    assert out["effective_mode"] == "live"
    assert out["effective_mode_label"] in {"LEARNING_AGGRESSIVE_DEMO", "GUARDED_DEMO"}
