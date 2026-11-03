# Timing Discipline Phase 1 Notes

## Source of Operator Message

Message:

`Timing discipline blocked trade: wait for clean entry timing confirmation.`

Primary source:

- File: `atlas_ai/trader.py`
- Function/path: `AtlasTrader._run_once()` after directional candidate selection and minimum confidence threshold checks.
- Current line reference from repo search: `atlas_ai/trader.py:5201`

The message is static text assigned to `status["message"]` when:

```python
timing_discipline = dict(status.get("timing_discipline", {}) or {})
if bool(timing_discipline.get("hard_block", False)):
    status["action"] = "blocked"
    status["message"] = "Timing discipline blocked trade: wait for clean entry timing confirmation."
```

The variable explanation behind the static message is already present in `status["timing_discipline"]`.

## Where The Check Is Built

Primary structured check:

- File: `atlas_ai/trader.py`
- Function: `AtlasTrader._timing_discipline_snapshot(...)`
- Current line reference from repo search: around `atlas_ai/trader.py:3710`

The snapshot returns:

- `status`: `timing_block` or `timing_clear_for_existing_gates`
- `score`: integer 0-100
- `side`
- `missing`
- `reasons`
- `observed`
- `requirements`
- `hard_block`
- `recommendation`: `wait_for_clean_timing_confirmation` or `allow_existing_gates_to_decide`

## Exact Hard-Block Condition

Timing discipline hard-blocks only when both are true:

1. The selected signal side is directional:
   - `signal.side in {"buy", "sell"}`

2. At least one timing requirement is missing:
   - `unique_missing` is non-empty

Equivalent condition:

```python
hard_block = bool(str(getattr(signal, "side", "flat")) in {"buy", "sell"} and unique_missing)
```

This means timing discipline does not block flat/no-signal output. It blocks only after a directional candidate has already cleared earlier selection/confidence checks.

## Timing Requirements And Inputs

The check reads these inputs from `context`, `status["chart_features"]`, `status["visual_chart_read"]`, and `status["setup_quality_observation"]`.

### Location

Data sources:

- `context["entry_location_state"]`
- `context["location_state"]`
- `context["location"]`
- `visual_chart_read.visual_features.visual_location`
- `chart_features.location_state`

Pass values for fresh/tradable location:

- `pullback`
- `retest`
- `fresh_retest`
- `reaction_zone`
- `range_edge`
- `sweep_rejection`
- `structure_reclaim`

Blocking/missing outcomes:

- If location is `mid_range`, `middle`, `consolidation`, `unknown`, or empty:
  - missing: `tradable_location`
  - reason: `entry_location_is_mid_range_or_unclear`

- If location is not fresh/tradable and not mid-range:
  - missing: `fresh_entry_or_retest`
  - reason: `entry_location_is_not_fresh_retest_or_reaction_zone`

### Momentum

Data sources:

- `context["entry_momentum_state"]`
- `context["momentum_state"]`
- `context["momentum"]`
- `visual_chart_read.visual_features.visual_momentum`
- `chart_features.momentum_state`

Pass values:

- `strong`
- `turning`
- `impulsive`
- `aligned`

Blocking/missing outcome:

- missing: `strong_or_turning_m1_momentum`
- reason: `m1_momentum_not_confirmed`

### Continuation Structure Hold

Data sources:

- `context["entry_trend_state"]`
- `context["trend_state"]`
- `chart_features.trend_state`
- `context["trend_strength"]`
- `chart_features.trend_strength`
- `context["setup_type_hint"]`
- `chart_features.setup_type_hint`

Continuation-like setup hints:

- contains `continuation`
- contains `pullback`
- contains `impulse`
- contains `trend`

Pass condition:

- `trend_strength >= 0.18`, or
- `trend_state in {"trend", "uptrend", "downtrend", "bullish", "bearish"}`

Blocking/missing outcome for continuation-like setups:

- missing: `structure_hold`
- reason: `continuation_structure_not_confirmed`

### Stop-Hunt Protection

Data sources:

- `status["learning_summary"]["loss_reason_summary"]["top_reason"]`
- location fields listed above

Pass values when top loss reason is `stop_hunt`:

- `sweep_rejection`
- `structure_reclaim`
- `retest`
- `fresh_retest`

Blocking/missing outcome:

- missing: `sweep_rejection_or_structure_reclaim`
- reason: `stop_hunt_history_requires_rejection_or_reclaim`

### New York Extra Momentum Confirmation

Data source:

- `context["session"]`

Condition:

- If `session == "new_york"` and momentum is not confirmed, timing discipline adds:
  - missing: `new_york_extra_momentum_confirmation`
  - reason: `new_york_recent_performance_requires_stronger_confirmation`

This duplicates the practical momentum requirement for New York conditions but is de-duplicated in `missing` only by exact string key, so it can appear alongside `strong_or_turning_m1_momentum`.

### Continuous Visual Watch Checks

These checks apply only when:

- `status["visual_chart_read"]["visual_watch_used_for_decision"] == True`

Data sources:

- `status["visual_chart_read"]["visual_confirmation"]`
- `status["visual_chart_read"]["visual_features"]["visual_location"]`
- `status["visual_chart_read"]["visual_features"]["visual_momentum"]`
- `status["visual_chart_read"]["visual_features"]["visual_direction"]`

Blocking/missing outcomes:

- If `visual_confirmation == "does_not_confirm"`:
  - missing: `visual_confirmation`
  - reason: `continuous_visual_watch_does_not_confirm_setup`

- If `visual_location == "mid_range"`:
  - missing: `visual_tradable_location`
  - reason: `continuous_visual_watch_sees_mid_range`

- If `visual_momentum in {"weak", "unknown"}`:
  - missing: `visual_momentum_confirmation`
  - reason: `continuous_visual_watch_lacks_momentum_confirmation`

- If `visual_direction == "sideways"`:
  - missing: `visual_directional_impulse`
  - reason: `continuous_visual_watch_sees_sideways_action`

## Score Calculation

The score starts at `100` and applies penalties:

- `-35` if location is mid-range/unclear
- `-25` if momentum is not confirmed
- `-15` if location is not fresh/tradable and not mid-range
- `-15` if continuation-like setup lacks structure hold
- `-10` if top loss reason is `stop_hunt` and stop-hunt protection is missing
- `-15` if visual confirmation is `does_not_confirm`

The score is clamped to `0..100`.

Important: the current hard-block condition does not depend on a numeric score threshold. The score is explanatory. The block is determined by directional side plus non-empty `missing`.

## Pipeline Location

This check is not inside one M1 gold-specific strategy module. It is a trader-level timing gate built after signal selection.

The flow is:

1. Strategy/signal generation creates candidates.
2. A chosen signal is selected.
3. If side is `flat` or confidence is below threshold, the cycle returns `hold`.
4. If a directional candidate cleared that threshold, `status["timing_discipline"]` is checked.
5. If `timing_discipline.hard_block == True`, the cycle returns `blocked` with the static timing message.
6. Only after this point do online-learning adjustment and later risk/execution logic continue.

So the message is not the conservative filter stage itself. It is a separate trader-level timing discipline gate before later execution/risk handling.

## Related Sources

Setup playbook evidence:

- File: `atlas_ai/analysis/setup_playbook.py`
- It builds setup candidates and missing confirmation labels such as `momentum_confirmation`, `pullback_entry_location`, and `breakout_follow_through`.
- This feeds operator-visible setup/timing context but is not the direct source of the static timing discipline block message.

Decision analysis timing engine:

- File: `atlas_ai/analysis/decision_analysis.py`
- It exposes `timing_engine` fields from setup radar:
  - `state`
  - `waiting_for`
  - `trigger_state`
  - `trigger_checklist`
  - `next_confirmation`
  - `conditions_slowing_setups`
- This is observability-only and does not itself produce the static block message.

Knowledge files:

- `atlas_ai/knowledge/readiness_rules.json`
- `atlas_ai/knowledge/blockers_catalog.json`
- `atlas_ai/knowledge/lessons_learned.json`

These contain related concepts such as setup readiness, mid-range avoidance, and confirmation requirements. They are not the direct source of the exact message.

Brain evidence/synthesis:

- `atlas_ai/brain/evidence.py`
- `atlas_ai/brain/synthesis.py`
- Brain v2 can surface blocker/status evidence, including current status and why traces, but the exact timing discipline message originates from `trader.py`.

## Existing Tests

Relevant tests:

- `tests/test_trader_learning.py`
  - `test_timing_discipline_blocks_mid_range_mixed_momentum_directional_candidate`
  - `test_timing_discipline_allows_clean_retest_with_strong_momentum_to_existing_gates`
  - `test_timing_discipline_uses_continuous_visual_watch_as_execution_quality_evidence`

- `tests/test_ui_server.py`
  - `test_visual_evidence_review_exposes_proxy_only_capture_degradation`

These confirm that timing discipline blocks weak/mid-range/visually unconfirmed directional candidates and clears clean retest plus strong momentum candidates to existing gates.

