# Timeframe Migration Audit

This audit documents strategy-level assumptions after changing Atlas from M1 execution data to M15 execution data with H1/H4 context. It is deliberately conservative: any live/demo strategy that was not explicitly validated on M15 should not be treated as safely recalibrated just because it still produces signals.

## Current Runtime Context

- Execution timeframe: `M15`
- Higher-timeframe confirmation/context: `H1`, `H4`
- Removed pending recalibration: `gold_micro_impulse`, `gold_breakout_retest`, `gold_liquidity_sweep`, `momentum_continuation`
- Still enabled pending audit: `ema_crossover`, `pullback_trend`

## Audit Status

| Strategy | Status | Decision |
| --- | --- | --- |
| `momentum_continuation` | Audited first because it is the current live/demo chosen signal | Needs recalibration or temporary exclusion before it is allowed to create execution requests |
| `ema_crossover` | Audited after `momentum_continuation`; currently raw signal only, not active/chosen in latest status | Needs recalibration; should not be treated as M15-cleared without M15 backtest/calibration |
| `pullback_trend` | Audited after `ema_crossover`; corrected-exit M15 validation failed in-sample | Exclude from M15 execution-capable use |
| `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9)` | Audited while current paper-mode chosen signal | Exclude from execution-capable use until retrained or recalibrated on M15 |
| `evolved_mhr_hr` | Audited while active as flat paper-mode demo chromosome | Exclude from execution-capable use until retrained/revalidated on M15 |

## Consolidated Decision Table

These decisions are audit conclusions only. They do not re-enable execution, change strategy logic, or approve live/demo trading. `EXECUTE_TRADES=false` remains the required posture until fixed-window M15 backtests and paper comparisons exist.

| Strategy | Decision | Reasoning |
| --- | --- | --- |
| `momentum_continuation` | Recalibrate or exclude before execution-capable use | Lives in `minute_pack.py`; ROC/EMA/regime windows expanded by 15x on M15; `ROC(12)` moved from 12 minutes to 3 hours, EMA 9/21 moved to 2.25/5.25 hours, and no M15 calibration evidence exists. |
| `ema_crossover` | Recalibrate before treating as M15-cleared | 20/50 EMA structure can be valid on M15, but confidence scaling, fresh-cross timing, ATR sizing, and EMA guard behavior were tuned/used before the M15 migration; raw confidence reached `1.0` on M15, which requires validation. |
| `pullback_trend` | Exclude from M15 execution-capable use | Corrected-exit M15 retest fixed the one-bar exit-sizing issue but failed in-sample over 12 weeks: 42 trades, PF `0.634`, win rate `28.57%`, return `-4.22%`, SL/TP `30:12`. No directional edge was demonstrated. |
| `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9)` | Exclude until retrained/recalibrated on M15 | Current paper-mode chosen signal; parameters were selected by mutation tournament before M15 migration; one-bar-forward objective appears M1-scale; `fast=18` and `slow=87` now mean 4.5 and 21.75 hours. |
| `evolved_mhr_hr` | Exclude until retrained/revalidated on M15 | Active demo chromosome; review artifacts do not prove M15 backtest/shadow review; volume/regime windows expanded materially; some chromosome genes are not enforced by the live signal generator. |

## Proposed M15 Recalibration Starting Points

These are **proposals only** for offline tests. They must not be enabled live/demo until the baseline process produces matching fixed-window backtest/paper evidence and an explicit operator decision approves them.

### `momentum_continuation` Proposed M15 Parameters

| Parameter | Current | Proposed M15 starting point | Reasoning |
| --- | --- | --- | --- |
| ROC window | `12` bars = 3 hours on M15 | `4` bars = 1 hour | Keeps the idea as intraday continuation instead of making it a 3-hour momentum gate. One hour gives enough M15 structure without waiting half a session. |
| Fast EMA | `9` bars = 2.25 hours | `5` bars = 1.25 hours | Restores faster continuation sensitivity while still smoothing more than an M1 trigger. |
| Slow EMA | `21` bars = 5.25 hours | `13` bars = 3.25 hours | Keeps a fast/slow trend relationship but reduces the slow leg from multi-session drift toward same-session context. |
| Regime lookback | `30` bars = 7.5 hours | `12` bars = 3 hours | A 3-hour M15 regime window better matches London/New York intraday continuation. |
| Minimum trend strength | `0.18` over 7.5 hours | `0.20` over 3 hours | Slightly higher threshold because M15 bars should show cleaner movement before continuation is trusted. |
| Minimum volatility pct | `0.03` | `0.04` | M15 average movement should clear a modestly stronger activity threshold so the strategy does not fire in compressed drift. |
| ROC trigger | `0.08%` over 3 hours | `0.12%` over 1 hour | Requires clearer same-session momentum after shortening the ROC window. |
| Confidence scale | `abs(roc)/0.6` | `abs(roc)/0.9` | Slows confidence saturation on M15 so ordinary hourly movement does not overstate certainty. |
| Trend weight divisor | `0.24` | `0.35` | Reduces boost sensitivity after moving to larger M15 movement windows. |
| Stop loss | `max(ATR*1.9, 115)` | `max(ATR*1.5, 150)` | M15 ATR is larger; slightly lower ATR multiple plus higher absolute floor avoids enormous stops while respecting M15 noise. |
| Take profit | `max(ATR*2.8, 175)` | `max(ATR*2.4, 240)` | Keeps reward above stop while adapting to M15 ATR scale. |

### `ema_crossover` Proposed M15 Parameters

| Parameter | Current | Proposed M15 starting point | Reasoning |
| --- | --- | --- | --- |
| Fast EMA | `20` bars = 5 hours | `8` bars = 2 hours | Keeps a tradable intraday M15 trend measure rather than a half-session lagging line. |
| Slow EMA | `50` bars = 12.5 hours | `21` bars = 5.25 hours | Makes the slow side a same-day trend anchor instead of nearly a full trading day. |
| Minimum history | `53` bars = 13.25 hours | `24` bars = 6 hours | Enough data for EMA 21 plus buffer while reducing stale warmup. |
| Cross comparison | previous/current bar = 30 minutes | keep `2` bars | A 30-minute M15 fresh-cross check is reasonable for M15; no change proposed initially. |
| Spread confidence multiplier | `700` | `450` | M15 EMA separation is naturally larger; lower multiplier should reduce confidence saturation. |
| Slope confidence multiplier | `320` | `180` | M15 slope covers 15-minute movement; lower multiplier prevents over-crediting one candle. |
| Fresh cross bonus | `0.18` | `0.12` | M15 crosses are slower and should not receive the same bonus without confirmation. |
| Stop loss | `max(ATR*2, 100)` | `max(ATR*1.4, 150)` | M15 ATR is larger; reduce multiple and raise absolute floor. |
| Take profit | `max(ATR*3, 150)` | `max(ATR*2.2, 240)` | Maintains positive reward profile without overextending targets on M15. |

### `ema_crossover` M15 Validation History And Final Recommendation

Final recommendation: **advance `ema_crossover` to guarded paper-mode validation only** using the closed-confirm entry and mid-exit sizing. This is **not** approval for live trading and must not re-enable `EXECUTE_TRADES`.

Recommended Stage A paper-mode candidate:

- Entry: EMA `8/21`
- Confirmation: require a fully closed M15 EMA cross, wait one bar after the confirmed cross, and require current alignment to remain intact
- Stop: `max(ATR*2.0, 1100 points)`
- Target: `max(ATR*2.6, 1450 points)`
- Mode: guarded paper validation only

Evidence trail:

| Artifact | Window | Variant | Trades | Win rate | Max drawdown | Profit factor | Total return | Avg hold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_20260825T074317Z_m15_ema_pullback_proposals.json` | `2026-08-03` to `2026-08-21` | EMA `8/21`, non-wide SL/TP `max(ATR*1.4,150)` / `max(ATR*2.2,240)` | 593 | 46.54% | 11.52% | 1.211 | 41.81% | 1.01 bars |
| `baseline_20260825T082222Z_m15_ema_pullback_proposals.json` | `2026-08-03` to `2026-08-21` | EMA `8/21`, wide SL/TP `max(ATR*2.5,250)` / `max(ATR*4.0,400)` | 536 | 42.72% | 12.66% | 1.101 | 15.64% | 1.13 bars |
| `baseline_20260825T084641Z_m15_ema_pullback_proposals.json` | `2026-08-03` to `2026-08-21` | Closed-confirm entry, non-wide SL/TP | 10 | 70.00% | 0.55% | 3.232 | 3.69% | 1.00 bars |
| `baseline_20260825T084936Z_m15_ema_pullback_proposals.json` | `2026-08-03` to `2026-08-21` | Closed-confirm entry, bar-sized SL/TP `max(ATR*2.5,1470)` / `max(ATR*4.0,2350)` | 10 | 50.00% | 1.14% | 1.173 | 0.25% | 7.60 bars |
| `baseline_20260825T085225Z_m15_ema_pullback_proposals.json` | `2026-08-03` to `2026-08-21` | Closed-confirm entry, mid-exit SL/TP `max(ATR*2.0,1100)` / `max(ATR*2.6,1450)` | 10 | 50.00% | 1.09% | 1.321 | 0.69% | 5.20 bars |
| `baseline_20260825T090916Z_m15_ema_closed_confirm_mid_exit_12w.json` | `2026-06-02T12:00Z` to `2026-08-25T12:00Z` | Same closed-confirm mid-exit candidate over 12 weeks | 60 | 43.33% | 3.46% | 1.157 | 1.97% | 4.22 bars |

The 12-week candidate passed the three larger-sample checks needed for Stage A:

- Profit factor stayed meaningfully above `1.0`: `1.157`.
- Stop-loss/take-profit split stayed reasonably balanced: `30` stop-loss, `23` take-profit, `5` max-hold, `2` stale-exit. This is about `1.3x` more stop-loss exits than take-profit exits, not a `5-6x` failure skew.
- Hold times remained spread across multiple M15 bars: `1:15`, `2:13`, `3:7`, `4:4`, `5:3`, `6:5`, `7:4`, `8:1`, `10:1`, `12:7`.

Interpretation:

- The initial M15 proposal had acceptable headline metrics but was still effectively resolving on one candle; this confirmed the exit geometry was not M15-mature.
- Widening stops/targets without improving entry timing did not fix the one-bar behavior.
- Closed-candle confirmation greatly reduced trade count and improved selectivity.
- Sizing the stop around `1.5x` the median M15 bar range and target around `1.3R` produced the best balance between realistic M15 holding behavior and positive expectancy.

Decision:

`ema_crossover` should move from "needs recalibration" to **Stage A guarded paper-mode validation candidate** with the closed-confirm mid-exit configuration above. It should remain excluded from live execution until Stage A paper evidence confirms stability, drawdown, and execution quality under current broker conditions.

Implementation status:

- Added as a distinct paper-stage strategy label: `ema_crossover_closed_confirm_mid_exit`.
- Added to the M15 paper runtime strategy universe only when `execute_trades=false`.
- Added paper-compare preset: `ema_crossover_closed_confirm_mid_exit_stage_a`.
- Stage A experiment lane isolates `enabled_strategies=("ema_crossover_closed_confirm_mid_exit",)` and forces `backtest_include_evolved=false`.
- This does not change live/execution-capable default strategy configuration and does not re-enable execution.

Paper result locations:

- Per-lane state: `<comparison_root>/<run_id>/experiment/paper_compare_state.json`
- Per-lane status/events: `<comparison_root>/<run_id>/experiment/status.json` and `events.jsonl`
- Cross-lane summary: `<comparison_root>/<run_id>/comparison_summary.json`
- Tracker report: `<comparison_root>/<run_id>/comparison_report.json`

Stage A exit criteria, defined before paper results are evaluated:

`ema_crossover_closed_confirm_mid_exit` may move from Stage A guarded paper validation to Stage B guarded demo tiny-pilot review only if all criteria below are met. Meeting these criteria permits a review; it does not automatically promote the strategy or re-enable execution.

| Criterion | Required threshold | Reason |
| --- | --- | --- |
| Minimum paper trades | At least `60` closed paper trades | Matches the 12-week backtest sample size used for the recalibration decision; smaller samples are too easy to overfit. |
| Minimum elapsed paper period | At least `4` full trading weeks | Ensures the signal survives changing sessions, news weeks, and volatility patches instead of one lucky cluster. |
| Profit factor | At least `1.15`, and not below `1.10` in any two-week sub-window with at least `15` trades | The 12-week backtest profit factor was `1.157`; paper performance must hold roughly the same edge without relying on one segment. |
| Win rate | At least `42%` over the full paper sample | The backtest win rate was `43.33%`; this allows small live-paper friction while requiring the same basic hit-rate character. |
| Stop-loss vs take-profit balance | Stop-loss exits no more than `1.75x` take-profit exits over the full sample | Backtest split was `30` SL vs `23` TP, about `1.3x`; a larger skew would suggest the M15 entry edge is degrading. |
| Hold-time profile | Average hold at least `3.0` M15 bars, with no more than `40%` of trades exiting after `1` bar | Backtest average hold was `4.22` bars. This prevents regression to the original one-bar resolution problem. |
| Drawdown | Paper max drawdown no worse than `6%` | Backtest drawdown was `3.46%`; Stage A allows some live-paper variance but not unchecked risk expansion. |
| Execution quality | No repeated broker/data errors, duplicate candidate opens, or stale-heartbeat periods affecting more than `5%` of eligible cycles | Stage B should only review strategy quality, not infrastructure noise. |
| Safety posture | `execute_trades=false` throughout Stage A and no manual override/bypass of ExecutionGate | Preserves the migration safety contract. |

Stage B review packet must include the paper state file, comparison summary, exit-reason distribution, hold-time distribution, and a direct comparison against `baseline_20260825T090916Z_m15_ema_closed_confirm_mid_exit_12w.json`.

Out-of-sample validation update:

After the Stage A thresholds were defined, the exact same `ema_crossover_closed_confirm_mid_exit` configuration was tested on two older out-of-sample windows. No parameters were changed for these tests.

| Artifact | Window | Trades | Win rate | Max drawdown | Profit factor | Total return | Avg hold | Hold-time shape |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_20260825T090916Z_m15_ema_closed_confirm_mid_exit_12w.json` | tuning/validation window: `2026-06-02T12:00Z` to `2026-08-25T12:00Z` | 60 | 43.33% | 3.46% | 1.157 | 1.97% | 4.22 bars | spread across `1..12` bars |
| `baseline_20260825T110428Z_m15_ema_closed_confirm_mid_exit_oos_2025-12-01_to_2026-02-28.json` | OOS 1: `2025-12-01` to `2026-02-28` | 67 | 38.81% | 3.94% | 0.844 | -2.87% | 2.64 bars | spread across `1..12` bars, though biased toward 1-bar exits |
| `baseline_20260825T110414Z_m15_ema_closed_confirm_mid_exit_oos_2025-01-01_to_2025-03-31.json` | OOS 2: `2025-01-01` to `2025-03-31` | 8 | 25.00% | 1.60% | 0.434 | -1.60% | 6.00 bars | spread across `2..12` bars |

Exit split by window:

| Window | Stop-loss | Take-profit | Max-hold | Stale-exit |
| --- | ---: | ---: | ---: | ---: |
| `2026-06-02` to `2026-08-25` | 30 | 23 | 5 | 2 |
| `2025-12-01` to `2026-02-28` | 38 | 24 | 4 | 1 |
| `2025-01-01` to `2025-03-31` | 4 | 2 | 1 | 1 |

Conclusion after out-of-sample checks:

The edge did **not** generalize out of sample. The tuning/validation window stayed modestly positive, but both older OOS windows produced profit factor below `1.0`, with negative total return. Therefore the pre-declared Stage A -> Stage B promotion criteria should now be considered **not met** for this configuration. `ema_crossover_closed_confirm_mid_exit` should **not** advance toward guarded demo based on current evidence, regardless of how live paper trading looks over the next few weeks, unless a materially new assessment is opened and documented separately.

What did generalize:

The structural correction remains useful learning. Closed-candle confirmation plus bar-range-scaled exits prevented the strategy from collapsing back into the earlier one-bar M15 resolution problem. Hold times remained distributed across multiple bars in both OOS periods. The failure is the durability of the specific EMA `8/21` edge, not the broader lesson that M15 entries need closed-candle confirmation and M15-sized exits.

### `pullback_trend` Proposed M15 Parameters

| Parameter | Current | Proposed M15 starting point | Reasoning |
| --- | --- | --- | --- |
| Fast/trend EMA | `20` bars = 5 hours | `8` bars = 2 hours | Makes pullback reference responsive enough for M15 intraday entries. |
| Slow/trend EMA | `50` bars = 12.5 hours | `21` bars = 5.25 hours | Keeps a same-day trend filter instead of a very slow anchor. |
| Buy proximity band | `cur <= e20 * 1.0003` | `cur <= e_fast * 1.0006` | Allows a slightly wider M15 pullback zone because M15 candles have larger ranges. |
| Sell proximity band | `cur >= e20 * 0.9997` | `cur >= e_fast * 0.9994` | Symmetric wider sell pullback zone. |
| Confidence scale | `distance * 300` | `distance * 180` | M15 EMA distance is larger; lower scale reduces false confidence spikes. |
| Minimum momentum confirmation | none inside strategy | require current close to turn back with trend over latest 1-2 M15 bars | Preserves pullback entry intent by requiring evidence that pullback is ending, not just price touching EMA. |
| Stop loss | `max(ATR*1.8, 110)` | `max(ATR*2.0, 1100)` | Applies the same M15 bar-range correction learned from the EMA work. The old 110/150 point floor was far below the measured M15 median high-low range of about 735 points, so normal one-candle noise could resolve trades mechanically. |
| Take profit | `max(ATR*2.6, 170)` | `max(ATR*2.6, 1450)` | Keeps the target near the validated M15 mid-exit geometry used by the EMA candidate, about 1.3R from the 1100 point floor, while still allowing ATR expansion in genuinely volatile conditions. |

Applied pre-retest implementation:

- Added a distinct research/paper candidate named `pullback_trend_m15_mid_exit`.
- The original `pullback_trend` strategy is unchanged for M1/live compatibility.
- The candidate uses EMA `8/21`, proximity band `0.0006`, confidence multiplier `180`, stop `max(ATR*2.0, 1100)`, and target `max(ATR*2.6, 1450)`.
- The candidate is available only in the M15 paper/backtest strategy universe when `execute_trades=false`.
- Added paper-compare preset `pullback_trend_m15_mid_exit_stage_a` so retests can isolate this candidate as the experiment lane.

Weak-pullback learning guard decision:

- The existing weak-pullback guard is explicitly M1-only in `AtlasTrader._apply_entry_timing_learning()` and returns immediately when `timeframe != "M1"`.
- Therefore this guard is **not active** for the M15 pullback candidate.
- This is documented as an intentional absence for this retest, not a hidden safety filter. An M15 equivalent should be designed separately only after reviewing M15-specific pullback loss classifications; the M1 guard should not be copied blindly because its evidence names and timing assumptions refer to M1 momentum.

### `pullback_trend` M15 Validation History And Final Recommendation

Original 3-week test:

- Window: `2026-08-03` to `2026-08-21`.
- Original M15 proposal result: 6 trades, win rate `66.67%`, profit factor `2.771`, total return `+1.93%`, max drawdown `0.55%`.
- All 6 trades exited after 1 M15 bar, with exit split `4` take-profit / `2` stop-loss.
- Verdict: too sparse and too mechanically resolved to trust. The sample did not prove an M15 pullback edge.

SL/TP structural fix:

- The original proposal used small flat floors: stop `max(ATR*1.3, 150)` and target `max(ATR*2.1, 240)`.
- Those floors were far below the measured M15 XAUUSD bar range distribution used during the EMA migration work, where the median M15 high-low range was about `735` points.
- The corrected exit geometry changed the pullback retest to stop `max(ATR*2.0, 1100)` and target `max(ATR*2.6, 1450)`, matching the same bar-range-scaled correction that prevented one-bar mechanical exits in the EMA work.

Corrected-exit 3-week check:

- Window: `2026-08-03` to `2026-08-21`.
- Result: 5 closed trades, win rate `20.00%`, profit factor `0.543`, total return `-0.52%`, max drawdown `1.13%`.
- Exit split: `4` stop-loss / `1` take-profit.
- Hold distribution: 2 bars: `2`; 4 bars: `3`.
- Interpretation: the exit-sizing structure improved because trades no longer collapsed to 1-bar resolution, but the directional result was weak and the sample was still too small.

Corrected-exit 12-week check:

- Window: `2026-06-02` to `2026-08-25`.
- Artifact: `docs/baselines/baseline_20260830T095936Z_m15_pullback_corrected_exit_12w.json`.
- Result: 42 trades, win rate `28.57%`, profit factor `0.634`, total return `-4.22%`, max drawdown `4.26%`.
- Exit split: `30` stop-loss / `12` take-profit.
- Hold distribution: 1 bar: `9`; 2 bars: `13`; 3 bars: `1`; 4 bars: `10`; 5 bars: `3`; 6 bars: `3`; 8 bars: `2`; 10 bars: `1`.
- Interpretation: the exit-sizing structural issue was fixed; hold times spread across 1-10 M15 bars instead of resolving mechanically on the entry bar. However, the strategy still did not show directional edge. Stop-loss exits dominated take-profit exits by `30:12`, profit factor stayed well below `1.0`, and total return was negative across a sufficient in-sample sample.

Stricter candidate check:

- The stricter `pullback_trend_m15_mid_exit` variant, using EMA `8/21`, a `0.0006` pullback band, current-bar alignment, stop `max(ATR*2.0, 1100)`, and target `max(ATR*2.6, 1450)`, produced only 1 trade over the same 12-week window.
- That result is not usable evidence either way. It is recorded for completeness only and should not be treated as proof of edge.

Final verdict: **EXCLUDE.**

The EMA `8/21` pullback-band approach does not show a demonstrated edge on M15 gold, even after correcting the known exit-sizing bug. Unlike `ema_crossover`, this failed the larger in-sample validation window itself, so out-of-sample testing was not performed and was not needed. Running OOS validation on a strategy with profit factor `0.634`, return `-4.22%`, and SL/TP split `30:12` would not add useful promotion evidence.

Runtime policy:

- Do not enable `pullback_trend` or `pullback_trend_m15_mid_exit` for M15 execution-capable trading.
- Keep the original `pullback_trend` behavior available for M1 compatibility only.
- Any future M15 pullback work should start as a new research proposal with different entry evidence, not as a minor tweak to this excluded candidate.

### `evolved_rule(...)` Proposed M15 Approach

Preferred path: **retrain on M15 rather than hand-editing the stored champion.**

| Parameter/search space | Current champion | Proposed M15 starting point | Reasoning |
| --- | --- | --- | --- |
| Fast EMA search | champion `18` = 4.5 hours | search `5..16` M15 bars | Covers 1.25-4 hours, closer to intraday M15 signal timing. |
| Slow EMA search | champion `87` = 21.75 hours | search `18..48` M15 bars | Covers 4.5-12 hours, enough for session/day trend without inheriting M1-era 21.75-hour lag. |
| RSI period | fixed `14` = 3.5 hours | test `7`, `10`, `14` | M15 RSI(14) may be acceptable for trend context but too slow for entry; compare variants. |
| RSI buy threshold | `47.757` | search `38..48` | Keeps buy confirmation moderate but avoids near-midline triggers without evidence. |
| RSI sell threshold | `56.891` | search `52..62` | Keeps sell confirmation above midline while testing stricter M15 momentum. |
| Confidence multiplier | EMA gap `*1000` | test `*450`, `*650`, `*850` | M15 EMA gaps are larger; multiplier needs calibration to avoid inflated confidence. |
| Minimum confidence | implicit live side requires non-flat; scoring skips `<0.1` | test live gate `0.45..0.60` | M15 signals are slower; require stronger threshold candidates in paper tests. |
| Outcome horizon | next bar = previously next minute | test `1`, `2`, `4` M15 bars | M15 entries may need 15-60 minute outcome windows; next-bar-only could be too noisy. |
| Training span | stored `20000` M1 bars | at least `5000` M15 bars, preferably `10000+` if broker history allows | Gives roughly 52-104 days of M15 history, enough to include multiple regimes. |

### `evolved_mhr_hr` Proposed M15 Approach

Preferred path: **deactivate old demo chromosome and create a new M15-reviewed chromosome**, because the current one was activated from pre-M15 artifacts and some genes are not enforced by the live generator.

| Gene/parameter | Current chromosome | Proposed M15 starting point | Reasoning |
| --- | --- | --- | --- |
| `entry_ema_fast` | `17` = 4.25 hours | `6..12` search, seed `8` | Restores faster M15 entry sensitivity without going back to M1 noise. |
| `entry_ema_slow` | `55` = 13.75 hours | `18..34` search, seed `21` | Provides a same-session trend anchor. |
| `rsi_buy_threshold` | `41` | `38..45` search | Keeps buy confirmation stricter than midline. |
| `rsi_sell_threshold` | `55` | `55..62` search | Keeps sell confirmation meaningfully above midline. |
| `confirm_min_body_ratio` | `0.36` | `0.40` seed, search `0.30..0.55` | M15 entries should prefer stronger candle bodies if used as execution confirmation. |
| `confirm_volume_min_ratio` | `0.68` over 8 bars = 2 hours on M15 | `0.85` over 8 bars, or change lookback to 4 bars if code supports it | Current threshold is loose for a 2-hour volume window; stronger threshold or shorter lookback should be tested. |
| `risk_min_confidence` | `0.53` | `0.58` seed, search `0.52..0.68` | M15 demo chromosomes should clear a higher confidence bar before selection. |
| `risk_confidence_multiplier` | `1.11` | `1.00` | Avoid boosting a migrated chromosome until M15 evidence supports it. |
| `exit_sl_atr_multiple` | `0.9` | `1.2` | Current stop may be too tight on M15 ATR; start with a wider but still bounded multiple. |
| `exit_tp_atr_multiple` | `2.4` | `2.2` | Keeps reward above stop while avoiding oversized targets. |
| `exit_max_hold_bars` | `8`, but not live-enforced by generator | require implementation audit before relying on it; if enforced, test `4..8` bars | On M15 this is 1-2 hours; but current live generator does not enforce the gene. |
| `regime_excluded_hours_mask` | excludes `7,8,9,12,15,18,19` | keep initially, then test against M15 session data | It was explicitly micro-cooldown-derived; do not loosen without evidence. |
| `regime_prefer_trend` | `0` | test both `0` and `1` | M15 may benefit from trend preference, but forcing it could remove useful range behavior. |
| `regime_min_trend_strength` | `0.19`, only enforced if prefer-trend is on | test `0.20..0.35` | Reasonable M15 trend-strength range for a trend-preferred variant. |

Implementation note for `evolved_mhr_hr`: before relying on chromosome summaries, reconcile the live generator with the genes. `confirm_htf_alignment`, `exit_max_hold_bars`, `risk_max_trades_per_day`, and related genes are present in the chromosome but not enforced inside `_EvolvedGoldStrategy.generate_signal()`.

## `momentum_continuation`

Source: `atlas_ai/strategies/minute_pack.py`, `MomentumContinuationStrategy.generate_signal()`.

Related post-signal gating source: `atlas_ai/trader.py`, `AtlasTrader._apply_signal_generation_gating()`.

### Values Used By The Strategy

The strategy reads the same dataframe as the active execution timeframe. After the migration, all candle counts below are M15 candles.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| ROC window | `roc(close, 12)` | 12 minutes | 180 minutes / 3 hours |
| Fast EMA | `ema(close, 9)` | 9 minutes | 135 minutes / 2.25 hours |
| Slow EMA | `ema(close, 21)` | 21 minutes | 315 minutes / 5.25 hours |
| Regime lookback | `_recent_regime_stats(close, lookback=30)` | 30 minutes | 450 minutes / 7.5 hours |
| Minimum trend strength | `trend_strength < 0.18` blocks | 0.18% drift over about 30 minutes | 0.18% drift over about 7.5 hours |
| Minimum volatility percent | `volatility_pct < 0.03` blocks | average candle movement threshold over M1 closes | average candle movement threshold over M15 closes |
| ROC trigger | `r > 0.08` / `r < -0.08` | 0.08% 12-minute momentum | 0.08% 3-hour momentum |
| Confidence scale | `abs(r) / 0.6` | 0.6% 12-minute move saturates confidence before trend weighting | 0.6% 3-hour move saturates confidence before trend weighting |
| Trend weight divisor | `trend_strength / 0.24` | scales 30-minute trend strength | scales 7.5-hour trend strength |
| Stop loss | `max(ATR * 1.9, 115)` | ATR from M1 candles; minimum 115 points | ATR from M15 candles; minimum 115 points |
| Take profit | `max(ATR * 2.8, 175)` | ATR from M1 candles; minimum 175 points | ATR from M15 candles; minimum 175 points |

### Additional Gating Values Affecting This Strategy

Source: `AtlasTrader._apply_signal_generation_gating()`.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Low-energy trend-strength threshold | `signal_regime_gate_min_trend_strength_pct = 0.18` | short-window signal suppression threshold | same numeric threshold applied to M15-derived context |
| Momentum trend-strength threshold | `momentum_signal_min_trend_strength_pct = 0.18` | minimum short-window trend strength | same numeric threshold applied to M15-derived context |
| Momentum volatility threshold | `momentum_signal_min_volatility_pct = 0.03` | minimum short-window movement | same numeric threshold applied to M15-derived context |
| Weak trend threshold | `signal_trend_strength_weak_pct = 0.12` | confidence haircut below weak M1 trend | confidence haircut below weak M15 trend |
| Strong trend threshold | `signal_trend_strength_strong_pct = 0.28` | confidence boost above strong M1 trend | confidence boost above strong M15 trend |
| Weak trend multiplier | `signal_trend_strength_weak_confidence_multiplier = 0.55` | confidence haircut | same |
| Strong trend boost | `signal_trend_strength_strong_confidence_boost = 1.12` | confidence boost | same |

### Evidence Of M1-Scale Tuning

- The file is `atlas_ai/strategies/minute_pack.py`, which strongly suggests these strategies were originally built as minute-scale strategies.
- Project knowledge explicitly says: "M1 training should teach execution timing from recent minute structure while using M15/H1 as context only." That is not the current runtime arrangement; M15 is now the execution dataframe.
- Existing verification baseline docs refer to a fixed deterministic M1 window.
- The strategy's short windows, especially `ROC(12)`, `EMA(9)`, `EMA(21)`, and `lookback=30`, are not necessarily invalid on M15, but their real time horizons changed by 15x. That changes the strategy from short continuation timing into multi-hour trend continuation.
- The volatility and trend thresholds were not accompanied by comments or documentation proving they were calibrated for M15.
- Git history was not available in this workspace, so no commit history could be used to prove original tuning intent.

### Decision

Decision: **needs recalibration or temporary exclusion before it is allowed to create execution requests.**

Reasoning:

- It is currently the live/demo selected signal source on M15.
- Its windows now describe 2-7.5 hour behavior, not minute-level continuation.
- The confidence reached `1.0` in live/demo status after the M15 switch, which is a warning sign that the old thresholds may be too permissive or simply measuring a different phenomenon than intended.
- Risk guard blocked the current trade, but that is accidental protection from volatility, not evidence that the strategy is correctly calibrated.
- There is no documented M15 backtest, threshold recalibration, or before/after behavior comparison for this strategy.

Safe next action:

- Temporarily exclude `momentum_continuation` from execution-capable enabled strategies, or run it in observability/paper-only mode, until M15 thresholds are deliberately calibrated.
- A calibration pass should decide whether the intended M15 strategy should use the current 9/21 EMA and 12-candle ROC logic, or whether it should use separate M15-specific thresholds and confidence scaling.

## `ema_crossover`

Source: `atlas_ai/strategies/built_in.py`, `EMACrossoverStrategy.generate_signal()`.

Related post-signal behavior source: `atlas_ai/trader.py`, EMA side-switch training and EMA fresh-cross guard near the downstream execution-gate preparation path.

Latest live/demo status checked during this audit:

- `timeframe`: `M15`
- Raw `signals`: `ema_crossover:buy:1`
- `signals_active`: no `ema_crossover` entry
- `chosen`: `evolved_rule(...):flat:0`

This means `ema_crossover` is currently enabled and generating a raw directional signal, but it is not the active/chosen signal in the latest stored cycle. It can still become influential if filtering, ranking, or regime state changes.

### Values Used By The Strategy

The strategy reads the same dataframe as the active execution timeframe. After the migration, all candle counts below are M15 candles.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Minimum history | `len(df) < max(fast, slow) + 3`; defaults `fast=20`, `slow=50`, so 53 candles required | 53 minutes | 795 minutes / 13.25 hours |
| Fast EMA | `ema(close, self.fast)` with `fast=20` | 20 minutes | 300 minutes / 5 hours |
| Slow EMA | `ema(close, self.slow)` with `slow=50` | 50 minutes | 750 minutes / 12.5 hours |
| Fresh cross comparison | previous candle EMA values vs current candle EMA values | cross detected across the last 2 minutes | cross detected across the last 30 minutes |
| ATR period | `atr(df)` uses default `period=14` | 14 minutes | 210 minutes / 3.5 hours |
| Direction | `curr_fast > curr_slow` buy, `curr_fast < curr_slow` sell | fast/slow minute trend direction | fast/slow multi-hour trend direction |
| EMA spread confidence component | `(abs(curr_fast - curr_slow) / price_now) * 700.0` | confidence from minute-scale EMA separation | same multiplier applied to multi-hour EMA separation |
| Fast EMA slope confidence component | `(abs(curr_fast - prev_fast) / price_now) * 320.0` | confidence from 1-minute fast-EMA movement | confidence from 15-minute fast-EMA movement |
| Fresh cross bonus | `0.18` if previous/current EMA relation crossed | bonus for a minute-scale fresh cross | bonus for a 15-minute-bar fresh cross |
| Confidence cap | `min(1.0, spread_component + slope_component + cross_bonus)` | caps at 1.0 | caps at 1.0 |
| Stop loss | `max(ATR * 2, 100)` | ATR from M1 candles; minimum 100 points | ATR from M15 candles; minimum 100 points |
| Take profit | `max(ATR * 3, 150)` | ATR from M1 candles; minimum 150 points | ATR from M15 candles; minimum 150 points |

### Additional EMA-Specific Trader Behavior

Source: `AtlasTrader.run_cycle()` downstream of initial signal ranking.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Adaptive EMA side switch | `adaptive_ema_side_switch=True` by default | a directional EMA signal can override the chosen side to help rotate buy/sell | same override can be driven by an M15 EMA signal |
| EMA fresh-cross guard enabled | `require_ema_fresh_cross=True` by default | weak/drift regimes require EMA side agreement or confidence is penalized/blocked | same guard is driven by M15 EMA side/confidence |
| Weak regime threshold | `trend_strength < 0.18` or regime not `trend` | short-window trend regime sensitivity | same numeric threshold applied after M15 migration |
| Soft penalty | confidence multiplied by `0.92` when EMA guard disagrees | minute-scale guard disagreement haircut | same haircut after M15 migration |
| Hard-block margin | block if post-penalty confidence `< dynamic_min_conf + 0.05` | minute-scale EMA guard can block weak setups | M15 EMA guard can block weak setups |
| Severe-drift block | `severe_drift_hard_mode` plus EMA side mismatch | hard block during severe drift | same hard block driven by M15 EMA signal |

### Evidence Of M1-Scale Or Pre-Migration Tuning

- The `20/50` EMA pair is not inherently M1-only; it is a common moving-average structure that can be meaningful on several timeframes.
- The current Atlas runtime was previously using M1 execution data, and the existing deterministic baseline files identify `timeframe_minutes: 1`.
- The project knowledge states that M1 training should use M15/H1 as context, not as the execution dataframe. After the migration, M15 is now the execution dataframe.
- The confidence formula uses fixed multipliers `700.0` and `320.0`, plus a fixed `0.18` cross bonus, with no evidence in code comments or docs that these were recalibrated for M15 EMA spread/slope magnitudes.
- The raw live/demo status during this audit showed `ema_crossover` at `buy:1`, meaning the confidence cap is already saturated on M15. That may be legitimate in a strong M15 trend, but without calibration it is a warning sign rather than proof of validity.
- Git history was not available in this workspace, so original tuning intent could not be proven from commits.

### Decision

Decision: **needs recalibration; should not be treated as M15-cleared without M15 backtest/calibration.**

Reasoning:

- Unlike `momentum_continuation`, `ema_crossover` is less obviously minute-only because 20/50 EMAs can be used on M15. The issue is not that the structure is impossible on M15; it is that the confidence scaling, ATR-derived stops, fresh-cross timing, and side-switch/guard behavior were not validated after the 15x timeframe change.
- The same two-candle cross logic moved from roughly 2 minutes of confirmation to roughly 30 minutes of confirmation, changing the timing meaning.
- The fast/slow EMA pair moved from 20/50 minutes to 5/12.5 hours, changing the strategy from intraminute/short-session timing into multi-hour trend following.
- The strategy is still enabled and producing a raw directional signal, so it can become selectable if the active pool changes.
- It was not the active/chosen signal in the latest status checked during this audit, but enabled raw signals can still influence later cycles through ranking, support, or EMA side-switch/fresh-cross logic.

Safe next action:

- Keep `ema_crossover` out of execution-capable use until a deliberate M15 calibration pass confirms the `20/50` periods, confidence multipliers, `0.18` cross bonus, ATR stop/target sizing, and EMA guard behavior.
- If it remains enabled for observation, monitor it as raw/status-only and do not treat `confidence=1.0` as proof that the M15 setup is cleared.

## `pullback_trend`

Source: `atlas_ai/strategies/minute_pack.py`, `PullbackTrendStrategy.generate_signal()`.

Related post-signal behavior source: `atlas_ai/trader.py`, `AtlasTrader._apply_signal_generation_gating()` and `AtlasTrader._apply_entry_timing_learning()`.

Latest live/demo status checked during this audit:

- `timeframe`: `M15`
- Raw `signals`: `pullback_trend:flat:0`
- `signals_active`: no `pullback_trend` entry
- `chosen`: `evolved_rule(...):flat:0`

This means `pullback_trend` is currently enabled, but the latest stored cycle did not have it as an active or chosen directional signal.

### Values Used By The Strategy

The strategy reads the same dataframe as the active execution timeframe. After the migration, all candle counts below are M15 candles.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Fast/trend EMA | `ema(close, 20)` | 20 minutes | 300 minutes / 5 hours |
| Slow/trend EMA | `ema(close, 50)` | 50 minutes | 750 minutes / 12.5 hours |
| Pullback buy zone | `cur <= e20 * 1.0003` while `e20 > e50` | price at or within about 0.03% above the M1 20-EMA | price at or within about 0.03% above the M15 20-EMA |
| Pullback sell zone | `cur >= e20 * 0.9997` while `e20 < e50` | price at or within about 0.03% below the M1 20-EMA | price at or within about 0.03% below the M15 20-EMA |
| Confidence scale | `abs(e20 - cur) / cur * 300` | distance from M1 20-EMA scaled to confidence | same multiplier applied to distance from M15 20-EMA |
| Confidence cap | `min(1.0, ...)` | caps at 1.0 | caps at 1.0 |
| ATR period | `_safe_vol(df)` -> `atr(df)` default `period=14` | 14 minutes | 210 minutes / 3.5 hours |
| ATR fallback | `_safe_vol()` returns `120.0` if ATR is invalid/NaN | fallback during insufficient/noisy M1 volatility | same fallback during insufficient/noisy M15 volatility |
| Stop loss | `max(ATR * 1.8, 110)` | ATR from M1 candles; minimum 110 points | ATR from M15 candles; minimum 110 points |
| Take profit | `max(ATR * 2.6, 170)` | ATR from M1 candles; minimum 170 points | ATR from M15 candles; minimum 170 points |

### Additional Trader-Side Gating Values Affecting This Strategy

Source: `AtlasTrader._apply_signal_generation_gating()`.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Low-energy suppression threshold | `signal_regime_gate_min_trend_strength_pct = 0.18` with low volatility and range/unknown regime | suppresses weak M1 directional pullback candidates | same numeric threshold applied to M15-derived context |
| Weak trend threshold | `signal_trend_strength_weak_pct = 0.12` | confidence haircut below weak M1 trend | confidence haircut below weak M15 trend |
| Weak trend multiplier | `signal_trend_strength_weak_confidence_multiplier = 0.55` | cuts confidence in weak M1 trend | same multiplier after M15 migration |
| Strong trend threshold | `signal_trend_strength_strong_pct = 0.28` | trend boost threshold on M1 context | trend boost threshold on M15 context |
| Strong trend boost | `signal_trend_strength_strong_confidence_boost = 1.12` | boosts confidence in strong M1 trend | same boost after M15 migration |
| Regime fit boost | trend regime and strategy in `{pullback_trend, ema_crossover, ...}` gives `factor *= 1.05` | contextual ranking boost for M1 trend-family strategies | same boost after M15 migration |

Source: `AtlasTrader._apply_entry_timing_learning()`.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| M1-only entry timing learning gate | `if timeframe != "M1": return signal` | weak-pullback loss lessons can haircut `pullback_trend` | this path is skipped entirely on M15 |
| Weak pullback lesson count | `counts["weak_pullback"] >= 50` | activates pullback-specific learning haircut | not applied on M15 because of the M1-only guard |
| Pullback weak-momentum haircut | `location == "pullback" and momentum != "strong"` -> `* 0.82` | reduces weak M1 pullback confidence | not applied on M15 |
| Mid-range haircut | `location == "mid_range"` -> `* 0.90` | reduces M1 mid-range entries | not applied on M15 |

### Evidence Of M1-Scale Tuning

- The strategy lives in `atlas_ai/strategies/minute_pack.py`, which strongly suggests it was part of the minute-level strategy pack.
- The strategy does not have an explicit minimum-history guard of its own; it relies on EMA/ATR outputs. With M15 input, the EMA and ATR calculations still run, but their time meaning expands by 15x.
- Existing lessons and teaching files repeatedly refer to `M1 momentum`, `M1 backtest losses`, and pullback timing as minute-level entry behavior.
- `AtlasTrader._apply_entry_timing_learning()` explicitly exits unless `settings.timeframe` is `M1`, so the historical weak-pullback learning haircuts that mention `pullback_trend` are not protecting it after the M15 migration.
- The `0.03%` pullback proximity band around the 20-EMA and the `* 300` confidence scale have no code comments or docs proving M15 calibration.
- Git history was not available in this workspace, so original tuning intent could not be proven from commits.

### Decision

Decision: **needs recalibration or temporary exclusion before it is allowed to create execution requests.**

Reasoning:

- `pullback_trend` is a minute-pack strategy and its 20/50 EMA structure moved from 20/50 minutes to 5/12.5 hours.
- The pullback proximity rule still uses the same tight `0.03%` band, but now compares price to a much slower M15 20-EMA.
- The confidence scale and ATR stop/target sizing now operate on M15-derived EMA distance and M15 ATR, not M1 pullback structure.
- The M1-specific weak-pullback learning guard no longer applies because it is explicitly skipped outside M1.
- It is not active or chosen in the latest status, but it remains enabled and could become selectable later.

Safe next action:

- Temporarily exclude `pullback_trend` from execution-capable enabled strategies, or run it as observability-only, until an M15 calibration pass validates the 20/50 EMA periods, pullback band, confidence scaling, and ATR stop/target sizing.
- If kept visible for monitoring, treat any future directional signal from `pullback_trend` as unaudited M15 evidence, not as cleared execution evidence.

## `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9)`

Source: `atlas_ai/strategies/strategy_lab.py`, `EvolvedRuleStrategy.generate_signal()`.

Load/use path: `atlas_ai/trader.py`, `AtlasTrader._load_evolved_strategy_pool()` loads `storage/learning_state.json` and `AtlasTrader._build_signals()` appends evolved rule signals after the base strategies.

Training source: `atlas_ai/learning/historical_trainer.py`, `HistoricalTrainer.train()` calls `StrategyLab.discover_candidates()` and stores the champion in `storage/learning_state.json`.

Latest paper-mode status checked during this audit:

- `timeframe`: `M15`
- `execute_trades`: `false`
- Raw `signals`: `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9):sell:0.6754`
- `signals_active`: `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9):sell:0.6754`
- `chosen`: `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9):sell:0.6754`

This is currently the selected paper signal. Execution is paused, but this rule is still influencing observation, ranking, status, and decision traces.

### Stored Champion Metadata

Source: `storage/learning_state.json`.

| Field | Value |
| --- | --- |
| `fast` | `18` |
| `slow` | `87` |
| `rsi_buy` | `47.75710122104659` |
| `rsi_sell` | `56.89084386210996` |
| `signature` | `f18_s87_rb47.8_rs56.9` |
| `source` | `mutation_tournament` |
| `generated_at` | `2026-05-14T00:18:43.996892+02:00` |
| `mutation_train_bars` | `20000` |
| `backtest_bars` / `trained_bars` | `5000` |
| `trades` | `4919` |
| `wins` / `losses` | `2567` / `2352` |
| `win_rate` | `0.5218540353730433` |
| `profit_score` | `0.10060017669707641` |
| `confidence_floor` | `0.605929220484339` |
| `generations` / `population` | `12` / `18` |

### Values Used By The Live Rule

The live rule reads the same dataframe as the active execution timeframe. After the migration, all candle counts below are M15 candles.

| Value | Code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Fast EMA | `ema(close, fast)` with champion `fast=18` | 18 minutes | 270 minutes / 4.5 hours |
| Slow EMA | `ema(close, slow)` with champion `slow=87` | 87 minutes / 1.45 hours | 1305 minutes / 21.75 hours |
| RSI period | `rsi(close)` default `period=14` | 14 minutes | 210 minutes / 3.5 hours |
| ATR period | `atr(df)` default `period=14` | 14 minutes | 210 minutes / 3.5 hours |
| Buy condition | `e_fast[-1] > e_slow[-1] and rsi < 47.7571` | M1 fast trend up plus M1 RSI below evolved buy threshold | M15 multi-hour trend up plus M15 RSI below the same threshold |
| Sell condition | `e_fast[-1] < e_slow[-1] and rsi > 56.8908` | M1 fast trend down plus M1 RSI above evolved sell threshold | M15 multi-hour trend down plus M15 RSI above the same threshold |
| Raw confidence | `abs(e_fast[-1] - e_slow[-1]) / close[-1] * 1000`, capped at `1.0` | confidence from M1 EMA separation | same multiplier applied to M15 EMA separation |
| Flat handling | side stays `flat`; confidence forced to `0.0` | no signal if EMA/RSI combination fails | same |
| Stop loss | `max(ATR * 1.9, 100)` | ATR from M1 candles; minimum 100 points | ATR from M15 candles; minimum 100 points |
| Take profit | `max(ATR * 2.8, 160)` | ATR from M1 candles; minimum 160 points | ATR from M15 candles; minimum 160 points |

### Values Used During Evolution/Scoring

Source: `StrategyLab.evaluate_strategy()` and `StrategyLab.discover_candidates()`.

| Value | Code | Meaning on M1 training data | Meaning if rerun on M15 data |
| --- | --- | --- | --- |
| Minimum training dataframe | `len(df) < 300` rejects | requires at least 300 minutes / 5 hours | requires at least 4500 minutes / 75 hours |
| Scoring warmup | loop starts at `i=220` | first 220 minutes ignored | first 3300 minutes / 55 hours ignored |
| Outcome horizon | compares bar `i` with `i + 1` | scores next 1-minute move | would score next 15-minute move |
| Regime scoring window | `_regime_name(close[i-59:i+1])` | 60 minutes / 1 hour | 900 minutes / 15 hours |
| Confidence floor cutoff | skip if `confidence < 0.1` | M1 EMA separation below threshold ignored | same threshold would apply to M15 EMA separation |
| Pullback quality denominator | `max(bar_range, current * 0.0002)` | compares entry location to M1 bar range or 0.02% price floor | same formula would compare to M15 bar range or 0.02% floor |
| Trend maturity scale | `ema_gap * 1800.0` times candle body ratio | M1 EMA-gap maturity score | same scale would apply to M15 EMA gap |
| Mutation fast range | random `fast` from `5..35`; mutate lower bound `5` | 5-35 minutes | 75-525 minutes / 1.25-8.75 hours |
| Mutation slow range | random `slow` up to `90`; mutate must stay at least `fast+5` | up to 90 minutes / 1.5 hours | up to 1350 minutes / 22.5 hours |
| RSI buy search range | random `30..55`; mutate bounded `20..60` | thresholds evolved against M1 RSI distribution | same values would need M15 RSI distribution validation |
| RSI sell search range | random `45..70`; mutate bounded `40..80` | thresholds evolved against M1 RSI distribution | same values would need M15 RSI distribution validation |
| Evolution training span | `EVOLVED_STRATEGY_TRAIN_MAX_BARS=20000` in `.env` | 20,000 M1 bars = about 13.9 days | 20,000 M15 bars = about 208.3 days |
| Historical backtest span | `HISTORICAL_BACKTEST_MAX_BARS=5000` in `.env` | 5,000 M1 bars = about 3.5 days | 5,000 M15 bars = about 52.1 days |

### Additional Trader-Side Behavior Affecting This Rule

| Value | Code | Meaning |
| --- | --- | --- |
| Evolved confidence sanitizer | in `_sanitize_signal()`: if strategy starts with `evolved_rule(`, confidence becomes `min(0.82, conf * 0.72 + 0.08)` | prevents evolved variants bunching at `1.0`; still uses M15-derived raw confidence after the migration |
| Weak strategy hard prune | in `_filter_weak_strategies()`: evolved rules are not exempt from hard pruning | can remove poor evolved rules based on learned allocator statistics |
| Entry timing learning | `_apply_entry_timing_learning()` mentions `evolved_rule`, but returns immediately unless `timeframe == "M1"` | M1 loss-pattern timing haircuts do not apply on M15 |
| Trend-family setup classification | `_strategy_to_setup_family()` defaults evolved rules to `continuation` | status/decision analysis treats this as continuation-like unless another layer overrides |

### Evidence Of M1-Scale Training

- The champion was generated on `2026-05-14`, before this audit's M15 migration work.
- The active settings now show `TIMEFRAME=M15`, but the stored champion has not been regenerated after the M15 change; its `generated_at` remains `2026-05-14T00:18:43.996892+02:00`.
- Backup learning state around the same champion contains explicit `timeframe: M1` entries in trade/context records.
- Existing baseline files identify `timeframe_minutes: 1` and `timeframe: M1`.
- Knowledge and teaching records include M1-specific lessons for evolved-rule losses, including the `2026-05-13 Atlas-tagged evolved_rule sell loss`.
- The evolution process optimized one-bar-forward outcomes; on M1 that means the next minute, while on M15 it would mean the next 15-minute candle.
- Git history was not available in this workspace, so commit-level tuning intent could not be checked.

### Decision

Decision: **exclude from execution-capable use until retrained or recalibrated on M15.**

Reasoning:

- This is the current selected paper signal, so it is the highest-priority remaining strategy.
- The rule is not only using generic EMA/RSI values; its exact parameters were selected by a mutation tournament over historical data that appears to have been M1-scale.
- The live meaning changed substantially: `fast=18` moved from 18 minutes to 4.5 hours, `slow=87` moved from 1.45 hours to 21.75 hours, and the RSI/ATR period moved from 14 minutes to 3.5 hours.
- Its training objective scored the next bar. That is a direct timing assumption: the evolved edge was selected against next-minute movement, not next-15-minute movement.
- The M1-only entry timing learning path does not protect this rule on M15 because it exits when `timeframe != "M1"`.
- It is currently paper-only because `EXECUTE_TRADES=false`, which is the correct safety posture until the evolved rules are retrained or explicitly recalibrated.

Safe next action:

- Keep execution disabled while this rule remains active/chosen on M15.
- Either exclude `evolved_rule(...)` from execution-capable strategy selection or regenerate/revalidate the evolved pool using M15 data with documented fixed-window backtest results.
- Do not treat current paper-mode confidence as evidence of M15 safety; it is a migrated M1-evolved chromosome being evaluated on a different candle duration.

## `evolved_mhr_hr`

Source: `atlas_ai/strategies/strategy_factory.py`, `_EvolvedGoldStrategy.generate_signal()`.

Chromosome source: `storage/evolution_population.json`, active demo chromosome `mhr_hr78912151819_c604b9a2`.

Promotion/review artifacts:

- `storage/ai_demo_review_packet_mhr_hr78912151819_c604b9a2.json`
- `storage/ai_demo_activation_mhr_hr78912151819_c604b9a2.json`
- `storage/ai_demo_activations.jsonl`

Latest paper-mode status checked during this audit:

- `timeframe`: `M15`
- `execute_trades`: `false`
- Raw `signals`: `evolved_mhr_hr:flat:0`
- `signals_active`: `evolved_mhr_hr:flat:0`
- `chosen`: `evolved_rule(fast=18,slow=87,rsi_buy=47.8,rsi_sell=56.9):sell`

This means `evolved_mhr_hr` is active/selectable as a demo chromosome signal, but it was flat and not the chosen signal in the latest stored cycle.

### Stored Chromosome Metadata

Source: `storage/evolution_population.json`.

| Field | Value |
| --- | --- |
| `uid` | `mhr_hr78912151819_c604b9a2` |
| Runtime strategy name | `evolved_mhr_hr` |
| `generation` | `7` |
| `origin` | `ai_regime_repair:micro_cooldown_exclude_7_8_9_12_15_18_19` |
| `created_at` | `2026-06-01T22:55:47.718761+02:00` |
| `promotion_stage` | `demo` |
| `promotion_reason` | `demo_activated_by_explicit_operator_command` |
| `demo_active` | `true` |
| `backtest_score` | `67.74` |
| `backtest_trades` | `256` |
| `backtest_wins` / `backtest_losses` | `154` / `102` |
| `backtest_win_rate` | `0.6015625` |
| `backtest_profit_factor` | `3.552535375872932` |
| `backtest_sharpe` | `0.6192001553401771` |
| `backtest_max_drawdown` | `0.026954589380059225` |
| `shadow_passed` | `true` |
| `shadow_signals` | `263` |
| `shadow_agreement_rate` | `0.6159695817490495` |
| `shadow_degradation_pct` | `0.0` |
| `demo_trades` | `0` |

Review packet compact summary: `EMA(17/55) RSI(41/55) | SL/TP=0.9x/2.4x | stage=shadow | Backtest: score=67.7 wr=60.16% pf=3.55 sharpe=0.62 dd=2.70% trades=256`.

### Genes That Affect The Live Signal Path

The runtime strategy reads the same dataframe as the active execution timeframe. After the migration, all candle counts below are M15 candles.

| Value | Gene/code | Meaning on M1 | Meaning on M15 |
| --- | --- | --- | --- |
| Minimum history | `len(df) < max(50, entry_ema_slow + 5)`; with `entry_ema_slow=55`, requires 60 bars | 60 minutes / 1 hour | 900 minutes / 15 hours |
| Fast EMA | `entry_ema_fast=17` | 17 minutes | 255 minutes / 4.25 hours |
| Slow EMA | `entry_ema_slow=55` | 55 minutes | 825 minutes / 13.75 hours |
| EMA cross comparison | current and previous EMA relation | cross over latest 2 minutes | cross over latest 30 minutes |
| RSI period | fixed `rsi(closes, period=14)` | 14 minutes | 210 minutes / 3.5 hours |
| ATR period | fixed `atr(df, period=14)` | 14 minutes | 210 minutes / 3.5 hours |
| Regime trend-strength window | `_trend_strength(closes.tail(60))` | 60 minutes / 1 hour | 900 minutes / 15 hours |
| Regime volatility window | `closes.tail(30).diff().abs().mean()` | 30 minutes | 450 minutes / 7.5 hours |
| Allowed sessions | `regime_allowed_sessions=3` -> `("london", "new_york")` | London/New York M1 operation | same sessions on M15 |
| Excluded hours | `regime_excluded_hours_mask=824192` -> excludes hours `7,8,9,12,15,18,19` | excludes specific clock hours with M1 granularity | same clock hours; each excluded hour contains four M15 bars |
| Prefer trend filter | `regime_prefer_trend=0` | does not require trend strength | same |
| Regime minimum trend strength | `regime_min_trend_strength=0.19` | only matters if `regime_prefer_trend=1` | same, but currently not enforced because prefer-trend is off |
| Regime max volatility | `regime_max_volatility_pct=0.07` | computed but not used for blocking in current runtime path | same |
| Buy base signal | EMA fast above slow, or fresh bullish cross | M1 EMA trend/cross | M15 multi-hour EMA trend/cross |
| Sell base signal | EMA fast below slow, or fresh bearish cross | M1 EMA trend/cross | M15 multi-hour EMA trend/cross |
| Cross base confidence | `0.72` on fresh EMA cross | fresh M1 cross confidence | fresh M15 cross confidence |
| Non-cross base confidence | `0.35` when fast remains above/below slow | continuing M1 EMA direction | continuing M15 EMA direction |
| RSI buy threshold | `rsi_buy_threshold=41` | M1 RSI confirmation for buy | M15 RSI confirmation for buy |
| RSI sell threshold | `rsi_sell_threshold=55` | M1 RSI confirmation for sell | M15 RSI confirmation for sell |
| RSI confirmation boost | `+0.12` if RSI confirms side | M1 RSI boost | M15 RSI boost |
| RSI opposition penalty | `-0.22` if RSI opposes side | M1 RSI penalty | M15 RSI penalty |
| RSI divergence boost | `+0.08` using previous/current RSI and previous/current close | 1-minute divergence/momentum check | 15-minute divergence/momentum check |
| Minimum body ratio | `confirm_min_body_ratio=0.36` | latest M1 candle body/range threshold | latest M15 candle body/range threshold |
| Small body penalty | body `< min_body * 0.5`, so `< 0.18` | latest M1 candle penalty | latest M15 candle penalty |
| Body confirmation boost | body `>= 0.36` gives `+0.06` | latest M1 candle boost | latest M15 candle boost |
| Volume lookback | `df[tick_volume].tail(8).mean()` | last 8 minutes | last 120 minutes / 2 hours |
| Minimum volume ratio | `confirm_volume_min_ratio=0.68` | M1 tick-volume confirmation | M15 tick-volume confirmation |
| Low volume penalty | volume ratio `< 0.68` gives `-0.08` | M1 volume penalty | M15 volume penalty |
| Normal volume boost | volume ratio `>= 1.0` gives `+0.04` | M1 volume boost | M15 volume boost |
| Minimum confidence gate | `risk_min_confidence=0.53` | M1 generated signal must clear 0.53 | same threshold applied to M15 signal |
| Confidence multiplier | `risk_confidence_multiplier=1.11` | M1 raw confidence multiplier | same multiplier on M15 |
| Confidence compression | if confidence `> 0.82`, compress to `(confidence * 0.72) + 0.08` | anti-bunching on M1-derived signal confidence | same compression on M15-derived confidence |
| Stop loss | `exit_sl_atr_multiple=0.9`; `ATR(14) * 0.9` | M1 ATR stop | M15 ATR stop |
| Take profit | `exit_tp_atr_multiple=2.4`; `ATR(14) * 2.4` | M1 ATR target | M15 ATR target |

### Genes Present But Not Enforced By The Current Live Signal Generator

These genes exist in the chromosome, validator, explainer, or evolution/scoring path, but this audit did not find them changing the live signal emitted by `_EvolvedGoldStrategy.generate_signal()`.

| Gene | Stored value | Runtime finding |
| --- | --- | --- |
| `confirm_min_trend_strength` | `0.33` | Not referenced in `_EvolvedGoldStrategy.generate_signal()`; the runtime uses `regime_min_trend_strength` instead, and only when `regime_prefer_trend=1`. |
| `confirm_htf_alignment` | `1.0` | Not enforced inside `_EvolvedGoldStrategy.generate_signal()`; higher-timeframe alignment is handled elsewhere in trader-level gates/status. |
| `exit_trail_enable` | `1.0` | Not used by the live signal generator; it returns static SL/TP points. |
| `exit_max_hold_bars` | `8.0` | Not used by the live signal generator. ShadowTester has a hardcoded `12` bar max exit, not this gene. |
| `risk_max_risk_per_trade` | `0.005` | Not used by the signal generator; sizing/risk is handled outside the strategy. |
| `risk_max_trades_per_day` | `2.0` | Not used by the signal generator; global risk/execution policy handles trade limits. |
| `risk_loss_cooldown_bars` | `2.0` | Used in evolution-engine backtest confidence gating, not in the live signal generator. |
| `risk_loss_cooldown_confidence_lift` | `0.05` | Used in evolution-engine backtest confidence gating, not in the live signal generator. |

### M1 Dependency Assessment

- This chromosome was created on `2026-06-01`, before the current M15 migration/audit.
- Its origin is `ai_regime_repair:micro_cooldown_exclude_7_8_9_12_15_18_19`, which is explicitly micro/timing-oriented.
- The demo activation artifact says it was activated by explicit operator command, but also says `live_promoted=false` and safety fields state it does not promote live.
- The review packet does not state the training timeframe directly, but surrounding runtime/status artifacts from this period show Atlas operating on `M1`, including MT5 chart captures titled `XAUUSD,M1` and status payloads with `timeframe: M1`.
- Existing knowledge and lessons from the same period repeatedly refer to M1 execution timing and M1 momentum.
- The shadow review reports `bars=18000`. If that was M1, it covers about 12.5 days; on M15 it would cover about 187.5 days. The artifact does not document an M15 rerun.
- The strategy's live timing checks use previous/current bar comparisons, 8-bar volume windows, 30/60-bar regime windows, and 14-bar RSI/ATR. Those all change materially when moved from M1 to M15.

### Decision

Decision: **exclude from execution-capable use until retrained/revalidated on M15.**

Reasoning:

- `evolved_mhr_hr` is an active demo chromosome signal, not a static hand-audited strategy.
- Its compact summary and genes show a timing-sensitive EMA/RSI/candle-body/volume/ATR strategy.
- Several critical timing windows expanded by 15x after the M15 migration: volume confirmation moved from 8 minutes to 2 hours, regime trend from 1 hour to 15 hours, regime volatility from 30 minutes to 7.5 hours, and max-history readiness from 1 hour to 15 hours.
- The candidate was demo-activated from a pre-M15 review process and was not live-promoted. There is no artifact proving an M15-specific backtest or shadow review.
- Some risk/confirmation genes shown in the chromosome are not enforced by the live signal generator, so the compact chromosome summary overstates what the live M15 signal path actually checks.
- It is currently flat and not chosen, but it remains active/selectable in paper-mode status.

Safe next action:

- Keep `EXECUTE_TRADES=false` while this chromosome remains active on M15.
- Either deactivate this demo chromosome from the selectable strategy pool or rerun the evolution/shadow review explicitly on M15 data with documented fixed-window results.
- Do not rely on its prior shadow pass as M15 evidence; the stored artifacts do not prove that review was conducted on M15.
