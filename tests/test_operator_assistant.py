from __future__ import annotations

from atlas_ai.memory.knowledge_store import KnowledgeStore
from atlas_ai.operator_assistant import AtlasAssistant
from atlas_ai.ui_server import UIDashboardSource


def _build_source(tmp_path):
    store = KnowledgeStore(str(tmp_path))
    return store, UIDashboardSource(store)


def _seed_outcome_learning(store: KnowledgeStore) -> None:
    store.save_learning_state(
        {
            "trade_outcome_intelligence": {
                "sample_count": 18,
                "winner_patterns": {
                    "samples": 9,
                    "top_patterns": {
                        "regime": [{"value": "trend", "samples": 6, "win_rate": 0.67, "avg_pnl_zar": 120.0}],
                        "session": [{"value": "london", "samples": 5, "win_rate": 0.8, "avg_pnl_zar": 140.0}],
                        "strategy": [{"value": "gold_micro_impulse", "samples": 4, "win_rate": 0.75, "avg_pnl_zar": 110.0}],
                        "volatility_bucket": [{"value": "normal", "samples": 7, "win_rate": 0.71, "avg_pnl_zar": 95.0}],
                        "confidence_bucket": [{"value": "high", "samples": 5, "win_rate": 0.8, "avg_pnl_zar": 130.0}],
                    },
                },
                "loser_patterns": {
                    "samples": 9,
                    "top_patterns": {
                        "regime": [{"value": "range", "samples": 6, "win_rate": 0.17, "avg_pnl_zar": -80.0}],
                        "session": [{"value": "late_us", "samples": 4, "win_rate": 0.25, "avg_pnl_zar": -70.0}],
                        "confidence_bucket": [{"value": "low", "samples": 5, "win_rate": 0.2, "avg_pnl_zar": -60.0}],
                    },
                },
                "pattern_comparison": {
                    "main_signal_gap": "Winning trades tend to appear with stronger trend strength and better confidence than losing trades."
                },
                "confidence_calibration": {
                    "samples": 18,
                    "mean_abs_gap": 0.08,
                    "bins": [
                        {"range": [0.0, 0.3], "samples": 8, "win_rate": 0.25, "avg_confidence": 0.22, "avg_pnl_zar": -80.0},
                        {"range": [0.3, 0.5], "samples": 3, "win_rate": 0.33, "avg_confidence": 0.38, "avg_pnl_zar": -30.0},
                        {"range": [0.5, 0.7], "samples": 2, "win_rate": 0.5, "avg_confidence": 0.58, "avg_pnl_zar": 20.0},
                        {"range": [0.7, 1.0], "samples": 5, "win_rate": 0.8, "avg_confidence": 0.76, "avg_pnl_zar": 110.0},
                    ],
                },
                "regime_memory": {
                    "by_regime": [
                        {"value": "trend", "samples": 10, "wins": 7, "losses": 3, "win_rate": 0.7, "avg_pnl_zar": 90.0},
                        {"value": "range", "samples": 8, "wins": 2, "losses": 6, "win_rate": 0.25, "avg_pnl_zar": -70.0},
                    ],
                    "favorable_regimes": ["trend"],
                    "unfavorable_regimes": ["range"],
                },
                "outcome_memory": {
                    "trades": [
                        {"strategy": "gold_micro_impulse", "session": "london", "regime": "trend", "volatility_bucket": "normal", "confidence": 0.74, "net_pnl_zar": 120.0, "outcome": "win"},
                        {"strategy": "gold_micro_impulse", "session": "new_york", "regime": "trend", "volatility_bucket": "normal", "confidence": 0.69, "net_pnl_zar": -50.0, "outcome": "loss"},
                        {"strategy": "gold_micro_impulse", "session": "london", "regime": "trend", "volatility_bucket": "normal", "confidence": 0.78, "net_pnl_zar": 140.0, "outcome": "win"},
                    ]
                },
                "signal_scoring_model": {
                    "status": "trained",
                    "samples": 18,
                    "metrics": {"accuracy": 0.72},
                },
            }
        }
    )


def test_assistant_refuses_mutating_requests(tmp_path):
    _store, source = _build_source(tmp_path)
    reply = AtlasAssistant(source).respond("enable live trading and place a trade now")

    assert reply["intent"] == "refusal_mutating_request"
    assert "will not place trades" in reply["result"]
    assert "approval-controlled" in reply["approval"]


def test_assistant_explains_block_reason_from_current_state(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "chosen": {
                "strategy": "momentum_continuation",
                "side": "buy",
                "confidence": 0.31,
            },
            "system_health": {
                "warnings": ["stale_data_detected"],
            },
        }
    )
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    reply = AtlasAssistant(source).respond("why are you not trading?")

    assert reply["intent"] == "block_reasons"
    assert "current action is `blocked`" in reply["result"] or "Atlas is not trading because the current action is `blocked`." in reply["result"]
    assert "poor liquidity" in reply["result"].lower()
    assert "system health warnings" in reply["result"].lower()


def test_assistant_summarizes_market_context_from_saved_chart_state(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "market": {"recent_close": [3000.0, 3001.0, 3003.0]},
            "regime": {"name": "trend", "trend_strength": 0.24},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "normal",
            },
        }
    )

    reply = AtlasAssistant(source).respond("study these charts")

    assert reply["intent"] == "chart_study"
    assert reply["response_kind"] == "task_result"
    assert "after studying the chart" in reply["result"].lower()
    assert "trend" in reply["result"].lower()
    assert "london" in reply["result"].lower()


def test_assistant_prepares_read_only_next_experiment_plan(tmp_path):
    store, source = _build_source(tmp_path)
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.save_status(
        {
            "comparison": {
                "run_id": "run123",
                "preset": "conservative_filter_rollout_v1",
                "decision_checklist": {"classification": "CONTINUE_TESTING"},
            }
        }
    )
    store.save_learning_state(
        {
            "evolved_strategy_pool": {
                "count": 3,
            }
        }
    )

    reply = AtlasAssistant(source).respond("prepare the next offline experiment")

    assert reply["intent"] == "planning"
    assert "offline experiment" in reply["result"].lower() or "safest next step" in reply["result"].lower()
    assert "separate explicit action" in reply["approval"].lower()


def test_chart_understanding_questions_route_to_chart_intent(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.22},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "chosen": {
                "strategy": "momentum_continuation",
                "side": "buy",
                "confidence": 0.31,
            },
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("what is your understanding of the chart right now?")

    assert reply["intent"] == "chart_understanding"
    assert "Sir," in reply["result"]
    assert "my current read on the chart" in reply["result"]
    assert "I am not taking a trade because" in reply["result"]


def test_chart_understanding_response_is_plain_english_and_respectful(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "paper",
            "message": "Trade candidate found, but EXECUTE_TRADES is disabled.",
            "regime": {"name": "range", "trend_strength": 0.08},
            "context_snapshot": {
                "session": "new_york",
                "volatility_state": "high",
                "spread_state": "normal",
                "liquidity_state": "normal",
            },
            "chosen": {
                "strategy": "gold_liquidity_sweep",
                "side": "sell",
                "confidence": 0.41,
            },
            "higher_tf_bias": "sell",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("what do you see on the chart")

    assert reply["intent"] == "chart_understanding"
    assert reply["acknowledge"].endswith("sir.")
    assert "Sir," in reply["result"]
    assert "higher-timeframe bias" in reply["result"]
    assert "to act" in reply["result"]


def test_partial_market_question_maps_to_nearest_market_intent(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
        }
    )

    reply = AtlasAssistant(source).respond("market thoughts?")

    assert reply["intent"] in {"chart_understanding", "market_context"}
    assert "Sir," in reply["result"]
    assert "hold" in reply["result"].lower() or "market" in reply["result"].lower()


def test_conversational_chart_prompt_maps_to_chart_understanding(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Spread too high (42.0 pts).",
            "regime": {"name": "range", "trend_strength": 0.07},
            "context_snapshot": {
                "session": "new_york",
                "volatility_state": "high",
                "spread_state": "wide",
                "liquidity_state": "normal",
            },
            "chosen": {"strategy": "gold_liquidity_sweep", "side": "sell", "confidence": 0.22},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("what do you think about the chart right now")

    assert reply["intent"] == "chart_understanding"
    assert "Sir," in reply["result"]
    assert "higher-timeframe bias" in reply["result"]


def test_informal_not_trading_prompt_maps_to_block_reasons(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "No high-confidence trade found."})

    reply = AtlasAssistant(source).respond("why arent you trading")

    assert reply["intent"] == "block_reasons"
    assert "not trading" in reply["result"].lower()


def test_slangy_trade_distance_prompt_maps_to_chart_understanding(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "regime": {"name": "trend", "trend_strength": 0.16},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "normal",
            },
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.29},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("are we close to a trade or nah")

    assert reply["intent"] == "chart_understanding"
    assert "waiting" in reply["result"].lower() or "need" in reply["result"].lower()


def test_simple_explanation_request_reduces_complexity(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.22},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("what do you see on the chart, explain simple")

    assert reply["intent"] == "chart_understanding"
    assert "market looks" in reply["result"].lower()
    assert "trend strength around" not in reply["result"].lower()


def test_understanding_prompt_handles_conversational_form(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "idle", "message": "Waiting for first cycle."})

    reply = AtlasAssistant(source).respond("do you understand me")

    assert reply["intent"] == "understanding"
    assert "I understand" in reply["result"] or "I understand" in reply["acknowledge"]


def test_winner_pattern_review_uses_outcome_learning_memory(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)

    reply = AtlasAssistant(source).respond("what do past winners usually look like")

    assert reply["intent"] == "winner_pattern_review"
    assert "Direct answer:" in reply["result"]
    assert "trend" in reply["result"].lower()
    assert "Supporting evidence:" in reply["result"]


def test_loser_pattern_review_uses_outcome_learning_memory(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)

    reply = AtlasAssistant(source).respond("what do past losers usually look like")

    assert reply["intent"] == "loser_pattern_review"
    assert "Direct answer:" in reply["result"]
    assert "range" in reply["result"].lower()
    assert "losing" in reply["result"].lower() or "losers" in reply["result"].lower()


def test_setup_vs_winners_compares_current_context_to_winner_profile(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)
    store.save_status(
        {
            "regime": {"name": "trend"},
            "context_snapshot": {"session": "london", "volatility_pct": 0.03},
            "chosen": {"strategy": "gold_micro_impulse", "side": "buy", "confidence": 0.71},
        }
    )

    reply = AtlasAssistant(source).respond("how does this setup compare to past winners")

    assert reply["intent"] == "setup_vs_winners"
    assert "matches past winners" in reply["result"].lower()


def test_confidence_calibration_review_reports_calibration_quality(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)

    reply = AtlasAssistant(source).respond("is your confidence calibrated well")

    assert reply["intent"] == "confidence_calibration_review"
    assert "calibrated" in reply["result"].lower()
    assert "mean absolute confidence gap" in reply["result"].lower()


def test_regime_memory_review_reports_worst_regime(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)

    reply = AtlasAssistant(source).respond("which regime has been worst for you")

    assert reply["intent"] == "regime_memory_review"
    assert "range" in reply["result"].lower()
    assert "best stored regime" in reply["result"].lower()


def test_strategy_outcome_review_reports_current_strategy_memory(tmp_path):
    store, source = _build_source(tmp_path)
    _seed_outcome_learning(store)
    store.save_status({"chosen": {"strategy": "gold_micro_impulse", "side": "buy", "confidence": 0.74}})

    reply = AtlasAssistant(source).respond("what does outcome memory say about this strategy")

    assert reply["intent"] == "strategy_outcome_review"
    assert "gold_micro_impulse" in reply["result"].lower()
    assert "labeled outcomes" in reply["result"].lower()


def test_self_assessment_routes_capability_question_away_from_understanding_intent(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "regime": {"name": "range", "trend_strength": 0.07},
            "context_snapshot": {
                "volatility_state": "low",
                "spread_state": "wide",
                "liquidity_state": "unknown",
            },
            "win_rate_focus": {"samples": 6, "recent_win_rate": 0.33},
            "drift": {"active": True},
            "system_health": {"paper_mode_confirmed": True},
            "decision_explanation": {"final_decision": "block", "reason_codes": ["no_high_confidence_trade_found"]},
        }
    )

    reply = AtlasAssistant(source).respond("how well do you understand the charts")

    assert reply["reasoning_mode"] == "SELF_ASSESSMENT"
    assert reply["intent"] == "self_assessment"
    assert "%" in reply["result"]
    assert "limitations" in reply["result"].lower()


def test_self_assessment_reports_grounded_low_confidence_when_sample_is_low(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "regime": {"name": "range", "trend_strength": 0.05},
            "context_snapshot": {
                "session": "unknown",
                "volatility_state": "low",
                "spread_state": "wide",
                "liquidity_state": "unknown",
            },
            "win_rate_focus": {"samples": 4, "recent_win_rate": 0.25},
            "drift": {"active": True},
            "system_health": {"paper_mode_confirmed": True},
            "decision_explanation": {"final_decision": "block", "reason_codes": ["no_high_confidence_trade_found"]},
        }
    )

    reply = AtlasAssistant(source).respond("between 0 to 100%, how good is your chart understanding")

    assert reply["intent"] == "self_assessment"
    assert "limited confirmed trade outcomes" in reply["result"].lower()
    assert "not fully validated" in reply["result"].lower()


def test_self_assessment_answers_weakness_question_with_grounded_limits(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "regime": {"name": "range", "trend_strength": 0.05},
            "context_snapshot": {"session": "unknown", "volatility_state": "low", "spread_state": "wide", "liquidity_state": "poor"},
            "win_rate_focus": {"samples": 8, "recent_win_rate": 0.25},
            "drift": {"active": True},
            "system_health": {"paper_mode_confirmed": True, "effective_mode": "paper"},
            "decision_explanation": {"final_decision": "block", "reason_codes": ["no_high_confidence_trade_found"]},
        }
    )

    reply = AtlasAssistant(source).respond("what are your weaknesses")

    assert reply["intent"] == "self_assessment"
    assert "weaknesses" in reply["result"].lower()
    assert "readiness" in reply["result"].lower()


def test_self_assessment_answers_strength_question_with_grounded_capabilities(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "regime": {"name": "trend", "trend_strength": 0.22},
            "context_snapshot": {"session": "london", "volatility_state": "normal", "spread_state": "tight", "liquidity_state": "normal"},
            "win_rate_focus": {"samples": 18, "recent_win_rate": 0.42},
            "drift": {"active": False},
            "system_health": {"paper_mode_confirmed": True, "effective_mode": "paper"},
            "decision_explanation": {"final_decision": "block", "reason_codes": ["weak_consensus"]},
        }
    )

    reply = AtlasAssistant(source).respond("what are you good at")

    assert reply["intent"] == "self_assessment"
    assert "good at" in reply["result"].lower()
    assert "identify trend and regime" in reply["result"].lower()


def test_capability_assessment_routes_self_reference_requirement_question(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "win_rate_focus": {"samples": 30, "recent_win_rate": 0.267, "target": 0.55},
            "drift": {"active": True, "recent_win_rate_20": 0.15, "baseline_win_rate": 0.34},
            "system_health": {"effective_mode": "paper", "paper_mode_confirmed": True},
        }
    )

    reply = AtlasAssistant(source).respond("what do you need to become a day trader")

    assert reply["reasoning_mode"] == "CAPABILITY_ASSESSMENT"
    assert reply["intent"] == "capability_assessment"
    assert "for me to operate effectively as a day trading system" in reply["result"].lower()
    assert "currently, i am limited by" in reply["result"].lower()


def test_capability_assessment_handles_live_demo_readiness_question(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "win_rate_focus": {"samples": 30, "recent_win_rate": 0.267, "target": 0.55},
            "drift": {"active": True, "recent_win_rate_20": 0.15, "baseline_win_rate": 0.34},
            "system_health": {"effective_mode": "paper", "paper_mode_confirmed": True},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")

    reply = AtlasAssistant(source).respond("are you ready for live demo")

    assert reply["intent"] == "capability_assessment"
    assert "readiness statement" in reply["result"].lower()
    assert "not ready" in reply["result"].lower()


def test_capability_assessment_understands_you_as_atlas_system(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "win_rate_focus": {"samples": 24, "recent_win_rate": 0.31, "target": 0.55},
            "drift": {"active": True, "recent_win_rate_20": 0.18, "baseline_win_rate": 0.34},
            "system_health": {"effective_mode": "paper", "paper_mode_confirmed": True},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")

    reply = AtlasAssistant(source).respond("what is missing for you")

    assert reply["intent"] == "capability_assessment"
    assert "my current state" in reply["understanding"].lower() or "i, as the atlas system" in reply["understanding"].lower()
    assert "current primary blocker" in reply["result"].lower() or "main blocker" in reply["result"].lower()


def test_capability_assessment_does_not_override_understanding_acknowledgment(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "idle", "message": "Waiting for first cycle."})

    reply = AtlasAssistant(source).respond("do you understand my task")

    assert reply["intent"] == "understanding"


def test_follow_up_explain_that_simpler_uses_previous_blocker_answer(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Spread too high (42.0 pts)."})
    store.log_event("risk", "Spread too high (42.0 pts).")
    assistant = AtlasAssistant(source)

    first = assistant.respond("what is the main blocker right now")
    follow = assistant.respond("explain that simpler")

    assert first["intent"] == "prioritization"
    assert follow["intent"] == "follow_up"
    assert "main blocker" in follow["result"].lower()
    assert "safety checks" in follow["result"].lower()


def test_follow_up_what_do_you_mean_by_that_clarifies_previous_answer(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Conservative pre-trade filter blocked trade: poor liquidity."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    assistant = AtlasAssistant(source)

    assistant.respond("what is the main blocker right now")
    follow = assistant.respond("what do you mean by that")

    assert follow["intent"] == "follow_up"
    assert "main blocker right now" in follow["result"].lower()


def test_follow_up_based_on_that_what_should_happen_next_uses_previous_subject(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Conservative pre-trade filter blocked trade: poor liquidity."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    assistant = AtlasAssistant(source)

    assistant.respond("what is the main blocker right now")
    follow = assistant.respond("based on that, what should happen next")

    assert follow["intent"] == "follow_up"
    assert "next is a safe research step" in follow["result"].lower()


def test_follow_up_how_sure_are_you_about_that_uses_previous_context(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "min_confidence_threshold": 0.50,
            "context_snapshot": {"spread_state": "tight", "liquidity_state": "poor"},
        }
    )
    assistant = AtlasAssistant(source)

    assistant.respond("what would need to change before you trade")
    follow = assistant.respond("how sure are you about that")

    assert follow["intent"] == "follow_up"
    assert "current setup confidence" in follow["result"].lower() or "i am basing that on" in follow["result"].lower()


def test_follow_up_why_does_that_matter_keeps_previous_subject(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Spread too high (42.0 pts)."})
    store.log_event("risk", "Spread too high (42.0 pts).")
    assistant = AtlasAssistant(source)

    assistant.respond("what is the main blocker right now")
    follow = assistant.respond("why does that matter")

    assert follow["intent"] == "follow_up"
    assert "main reason trades are being blocked" in follow["result"].lower()


def test_follow_up_focus_on_first_resolves_to_previous_priority_subject(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Conservative pre-trade filter blocked trade: poor liquidity."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    assistant = AtlasAssistant(source)

    assistant.respond("what is the main blocker right now")
    follow = assistant.respond("and what should I focus on first")

    assert follow["intent"] == "follow_up"
    assert "first thing to focus on" in follow["result"].lower() or "safe research step" in follow["result"].lower()


def test_follow_up_keeps_subject_across_longer_thread(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Spread too high (42.0 pts)."})
    store.log_event("risk", "Spread too high (42.0 pts).")
    assistant = AtlasAssistant(source)

    assistant.respond("what is the main blocker right now")
    assistant.respond("why does that matter")
    assistant.respond("explain that simpler")
    assistant.respond("and what should I focus on first")
    follow = assistant.respond("based on that, what should happen next")

    assert follow["intent"] == "follow_up"
    assert follow["reasoning_mode"] == "FOLLOW_UP"
    assert "safe research step" in follow["result"].lower() or "next is a safe research step" in follow["result"].lower()


def test_task_command_go_study_the_charts_routes_to_chart_study(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "regime": {"name": "trend", "trend_strength": 0.22},
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "higher_tf_bias": "buy",
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "min_confidence_threshold": 0.50,
        }
    )
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")

    reply = AtlasAssistant(source).respond("go study the charts")

    assert reply["intent"] == "chart_study"
    assert reply["response_kind"] == "task_result"
    assert "after studying the chart" in reply["result"].lower()
    assert "what looks favorable" in reply["result"].lower()
    assert "what looks unfavorable" in reply["result"].lower()


def test_block_reason_analysis_task_routes_and_reports_top_cause(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "No high-confidence trade found."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Spread too high (42.0 pts).")

    reply = AtlasAssistant(source).respond("study what is blocking trades")

    assert reply["intent"] == "block_reason_analysis"
    assert reply["response_kind"] == "task_result"
    assert "hurting trade quality most" in reply["result"].lower()
    assert "poor liquidity" in reply["result"].lower()


def test_run_review_task_routes_and_summarizes_recent_activity(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    store.log_event("system", "Execution mode reconciled at startup.")
    store.log_event("risk", "No high-confidence trade found.")
    store.log_event("paper", "Trade candidate found, but EXECUTE_TRADES is disabled.")

    reply = AtlasAssistant(source).respond("review the last session")

    assert reply["intent"] == "run_review"
    assert reply["response_kind"] == "task_result"
    assert "after reviewing the run" in reply["result"].lower()
    assert "recent activity was led by" in reply["result"].lower()


def test_change_summary_task_routes_and_reports_recent_change(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    store.log_event("risk", "Spread too high (42.0 pts).")
    store.log_event("risk", "No high-confidence trade found.")

    reply = AtlasAssistant(source).respond("summarize what changed since the last cycle")

    assert reply["intent"] == "change_summary"
    assert reply["response_kind"] == "task_result"
    assert "recent change" in reply["result"].lower() or "latest new state message" in reply["result"].lower()


def test_reasoning_mode_marks_current_state_question_as_state(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})

    reply = AtlasAssistant(source).respond("what is happening now")

    assert reply["reasoning_mode"] == "STATE"


def test_reasoning_mode_routes_how_would_you_prompt_to_planning(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "hold", "message": "No high-confidence trade found."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")

    reply = AtlasAssistant(source).respond("how would you improve this if I give you time")

    assert reply["reasoning_mode"] == "PLANNING"
    assert reply["intent"] == "planning"
    assert "if given time" in reply["result"].lower()
    assert "current status" in " ".join(reply["data_sources"]).lower() or "recent events" in " ".join(reply["data_sources"]).lower()


def test_reasoning_mode_marks_compare_query_as_comparison(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"comparison": {"run_id": "run123"}})

    reply = AtlasAssistant(source).respond("compare control vs experiment")

    assert reply["reasoning_mode"] == "COMPARISON"


def test_reasoning_mode_marks_main_blocker_question_as_prioritization(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "No high-confidence trade found."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Spread too high (42.0 pts).")

    reply = AtlasAssistant(source).respond("what matters most right now")

    assert reply["reasoning_mode"] == "PRIORITIZATION"
    assert reply["intent"] == "prioritization"
    assert "primary blocker" in reply["result"].lower()


def test_reasoning_mode_marks_setup_question_as_setup_evaluation(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "context_snapshot": {
                "session": "london",
                "volatility_state": "normal",
                "spread_state": "tight",
                "liquidity_state": "poor",
            },
            "chosen": {"strategy": "momentum_continuation", "side": "buy", "confidence": 0.31},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
        }
    )

    reply = AtlasAssistant(source).respond("explain what would need to change before you trade")

    assert reply["reasoning_mode"] == "SETUP_EVALUATION"
    assert reply["intent"] == "setup_evaluation"
    assert "current setup quality" in reply["result"].lower()
    assert "tradable" in reply["result"].lower()


def test_reasoning_mode_routes_action_question_to_action_guidance(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "context_snapshot": {
                "session": "new_york",
                "volatility_state": "normal",
                "spread_state": "normal",
                "liquidity_state": "unknown",
            },
            "regime": {"name": "range", "trend_strength": 0.08},
            "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")

    reply = AtlasAssistant(source).respond("what should I do about consensus right now")

    assert reply["reasoning_mode"] == "ACTION_GUIDANCE"
    assert reply["intent"] == "action_guidance"
    assert "my recommendation is to improve upstream" in reply["result"].lower()


def test_action_guidance_for_signals_recommends_upstream_improvement(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "hold",
            "message": "No high-confidence trade found.",
            "context_snapshot": {
                "session": "unknown",
                "volatility_state": "normal",
                "spread_state": "normal",
                "liquidity_state": "unknown",
            },
            "regime": {"name": "range", "trend_strength": 0.07},
            "chosen": {"strategy": "momentum_continuation", "side": "flat", "confidence": 0.0},
            "decision_explanation": {"reason_codes": ["no_high_confidence_trade_found"], "setup_quality_reasons": ["weak_consensus"]},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")

    reply = AtlasAssistant(source).respond("how do we improve signals")

    assert reply["intent"] == "action_guidance"
    assert "signal quality is currently" in reply["result"].lower() or "raw signal quality is currently" in reply["result"].lower()
    assert "recommendation is to improve upstream" in reply["result"].lower()


def test_action_guidance_response_has_four_operator_sections(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=buy, support=1, required=2.",
            "context_snapshot": {"session": "new_york", "volatility_state": "normal", "spread_state": "normal", "liquidity_state": "unknown"},
            "regime": {"name": "range", "trend_strength": 0.08},
            "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
        }
    )
    store.log_event("risk", "Consensus guard blocked trade: side=buy, support=1, required=2.")

    reply = AtlasAssistant(source).respond("should we change consensus or improve upstream")

    assert reply["reasoning_mode"] == "ACTION_GUIDANCE"
    assert reply["intent"] == "action_guidance"
    assert "current role:" in reply["result"].lower()
    assert "diagnosis:" in reply["result"].lower()
    assert "recommendation:" in reply["result"].lower()
    assert "reasoning:" in reply["result"].lower()


def test_action_guidance_understands_we_do_about_component_wording(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Consensus guard blocked trade: side=sell, support=1, required=2.",
            "context_snapshot": {"session": "new_york", "volatility_state": "normal", "spread_state": "normal", "liquidity_state": "unknown"},
            "regime": {"name": "range", "trend_strength": 0.08},
            "decision_explanation": {"setup_quality_reasons": ["weak_consensus"]},
            "missed_entry_learning_summary": {
                "available": True,
                "top_repair_candidate": {
                    "primary_blocker": "Consensus guard blocked trade: side=sell, support=1, required=2.",
                    "repair_priority": "high",
                    "suggested_action": "review_blocker_selectivity",
                    "samples": 150,
                    "missed_opportunity": 55,
                    "correct_hold_or_block": 25,
                },
            },
        }
    )

    reply = AtlasAssistant(source).respond("what should we do about consensus right now")

    assert reply["reasoning_mode"] == "ACTION_GUIDANCE"
    assert reply["intent"] == "action_guidance"
    assert "run experiment" in reply["result"].lower()
    assert "missed opportunities" in reply["result"].lower()


def test_action_guidance_best_approach_right_now_does_not_fall_back_to_generic_status(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.",
            "context_snapshot": {"session": "new_york", "volatility_state": "normal", "spread_state": "normal", "liquidity_state": "unknown"},
            "regime": {"name": "trend", "trend_strength": 0.19},
            "decision_explanation": {"reason_codes": ["htf_opposed"]},
        }
    )
    store.log_event("risk", "Higher-timeframe alignment required: chosen=buy, htf_bias=sell.")

    reply = AtlasAssistant(source).respond("what is the best approach right now")

    assert reply["reasoning_mode"] == "ACTION_GUIDANCE"
    assert reply["intent"] == "action_guidance"
    assert reply["intent"] != "general_operator_query"
    assert "recommendation:" in reply["result"].lower()


def test_prioritization_response_ranks_primary_secondary_and_tertiary_blockers(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status({"action": "blocked", "message": "Spread too high (42.0 pts)."})
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")
    store.log_event("risk", "Spread too high (42.0 pts).")
    store.log_event("risk", "No high-confidence trade found.")

    reply = AtlasAssistant(source).respond("rank the top 3 reasons no trades are passing")

    assert reply["intent"] == "prioritization"
    assert "Primary blocker" in reply["result"]
    assert "Secondary blocker" in reply["result"]
    assert "Tertiary blocker" in reply["result"]


def test_setup_evaluation_reports_quality_missing_conditions_and_issue_types(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "action": "blocked",
            "message": "Conservative pre-trade filter blocked trade: poor liquidity.",
            "context_snapshot": {
                "session": "new_york",
                "volatility_state": "high",
                "spread_state": "wide",
                "liquidity_state": "poor",
            },
            "regime": {"name": "range", "trend_strength": 0.08},
            "chosen": {"strategy": "gold_liquidity_sweep", "side": "sell", "confidence": 0.22},
            "higher_tf_bias": "buy",
            "min_confidence_threshold": 0.50,
            "decision_explanation": {
                "reason_codes": ["poor_liquidity"],
                "setup_quality_reasons": ["weak_consensus"],
                "penalties": ["loss_pattern_memory"],
            },
        }
    )
    store.log_event("risk", "Conservative pre-trade filter blocked trade: poor liquidity.")

    reply = AtlasAssistant(source).respond("tell me the main thing hurting trade quality")

    assert reply["intent"] in {"prioritization", "block_reason_analysis"}
    eval_reply = AtlasAssistant(source).respond("what would need to change before you trade")

    assert eval_reply["intent"] == "setup_evaluation"
    assert "current setup quality is weak" in eval_reply["result"].lower()
    assert "what is missing before it becomes tradable" in eval_reply["result"].lower()
    assert "confidence" in eval_reply["result"].lower()
    assert "higher-timeframe alignment" in eval_reply["result"].lower()
    assert "loss-pattern protection" in eval_reply["result"].lower()


def test_comparison_review_includes_lane_condition_comparison(tmp_path):
    store, source = _build_source(tmp_path)
    store.save_status(
        {
            "comparison": {
                "run_id": "run123",
                "control_status": {"message": "No high-confidence trade found."},
                "experiment_status": {"message": "Conservative pre-trade filter blocked trade: poor liquidity."},
            }
        }
    )

    reply = AtlasAssistant(source).respond("compare this condition to recent losing conditions")

    assert reply["intent"] == "comparison_review"
    assert "control lane" in reply["result"].lower()
    assert "experiment lane" in reply["result"].lower()


def test_mutation_refusal_still_includes_reasoning_metadata(tmp_path):
    _store, source = _build_source(tmp_path)

    reply = AtlasAssistant(source).respond("enable live and place a trade")

    assert reply["intent"] == "refusal_mutating_request"
    assert reply["reasoning_mode"] in {"STATE", "EXPLANATION", "PLANNING", "COMPARISON", "PROGRESSION", "ANALYSIS", "PRIORITIZATION", "SETUP_EVALUATION"}
