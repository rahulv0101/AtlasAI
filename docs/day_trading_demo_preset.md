# Day Trading Demo Preset

Status: built for backtesting only. Not deployed to the live `.env`.

This preset combines the operator-approved demo-learning changes into one auditable configuration. It is a deliberate risk increase for guarded demo learning, not a bug fix.

## Preset ID

`day_trading_demo_combined_v1`

Machine-readable preset:

`knowledge/day_trading_demo_preset.json`

## Changes

| Area | Setting | Old value | New preset value | Notes |
|---|---:|---:|---:|---|
| Confidence | `CONFIDENCE_THRESHOLD_DEMO` | `0.48` current guarded-demo experiment (`0.52` standard demo) | `0.44` | Reuses the existing learning-mode confidence value without turning learning mode on. |
| Structure | `STRUCTURE_SCORE_MIN_DEMO` | `0.35` | `0.24` | Reuses the existing learning-mode structure minimum. |
| Spread | `SPREAD_MAX_PIPS_DEMO` | `4.5` | `7.0` | Reuses the existing learning-mode spread ceiling. |
| Forced entries | `LEARNING_FORCE_ENTRY_PER_SESSION` | `false` | `false` | Explicitly remains disabled. This preset reuses lower thresholds only, not forced-entry logic. |
| Learning mode | `LEARNING_MODE` | `false` | `false` | Learning mode remains off. |
| Fast learning | `FAST_LEARNING_MODE` | `false` | `false` | Fast learning remains off. |
| Consensus | `MIN_SAME_SIDE_SIGNALS` | `2` | `1` | Reduces hard consensus guard support requirement. |
| Timing discipline | `TIMING_ALLOW_MID_RANGE_STRONG_MOMENTUM` | `false` | `true` | Allows `location=mid_range` only when `momentum=strong`. Momentum requirement is not weakened. |

## Timing Rule Detail

Default timing discipline still treats `mid_range` as not tradable.

The preset adds a guarded exception:

```text
if TIMING_ALLOW_MID_RANGE_STRONG_MOMENTUM=true
and location == mid_range
and momentum == strong
then the location part of timing discipline is allowed.
```

This does not loosen the momentum check. `momentum=mixed`, `weak`, or `unknown` still fails the timing momentum requirement.

Other timing blockers still apply, including visual-watch blockers if they are present.

## Safety Bounds Expected To Remain Active

| Bound | Required value |
|---|---:|
| `gate_execution_environment` | `demo` |
| `rollout_stage` | `limited_live_pilot` |
| `tiny_live_pilot.size_cap` | `0.01` |
| `tiny_live_pilot.max_trades_per_day` | `8` |
| `tiny_live_pilot.max_consecutive_losses` | `8` |
| `tiny_live_pilot.daily_loss_cap` | `0.6` |

## Non-Goals

- Does not deploy to the running backend.
- Does not restart the backend.
- Does not enable `LEARNING_FORCE_ENTRY_PER_SESSION`.
- Does not increase lot size.
- Does not remove ExecutionGate.
- Does not disable the daily loss cap.

## Backtest Gate

This preset must be backtested before live guarded-demo deployment. The next step is to run the combined preset offline and compare it against the current guarded-demo baseline.
