# Forced Learning Probe Audit

Generated: 2026-08-31

Purpose: identify retained decision/scan records created under
`learning_force_entry_per_session` / adaptive probe logic so future strategy
performance analysis does not treat them as normal threshold-passing trades.

## Finding

`logs/scan_results.jsonl` contains 29 retained records marked with
`forced_demo_entry: true` and/or the `adaptive_probe_entry` tag.

All 29 records show:

- `confidence: 0.4`
- `can_create_execution_request: false`
- `execution_request_created: true`
- `forced_demo_entry: true`
- `exclude_from_normal_strategy_performance: true`

These records were created under a looser forced-learning/probe standard, not
the normal demo thresholds. They must be excluded from normal strategy
performance, or analyzed in a separate forced-probe bucket.

## What The Two Request Fields Mean

Source:

- `atlas_ai/decision_framework.py`, `DecisionFramework.evaluate()`
- `atlas_ai/market_scanner.py`, `MarketScanner._run_single_scan()`
- `atlas_ai/market_scanner.py`, `MarketScanner._should_apply_adaptive_probe()`

`can_create_execution_request` is the normal eligibility result from the
decision framework. In `DecisionFramework.evaluate()`, it is set only after the
buy/sell score has passed the active score threshold and the final confidence
passes the active confidence threshold:

`out.can_create_execution_request = out.confidence >= threshold and out.trade_decision != "NO_TRADE"`

`MarketScanner._run_single_scan()` copies that value into the retained
`ScanResult` before any adaptive probe override is applied. That means this
field answers: would the normal threshold path have allowed an execution
request?

`execution_request_created` is a later lifecycle flag from the scanner. It is
set to `true` only after `_create_execution_request(...)` is called.

For these 29 records, the apparently contradictory pair:

- `can_create_execution_request: false`
- `execution_request_created: true`

is not a simple logging typo. It records the forced-learning probe mechanism:
the candidate did not pass the normal request-creation threshold, then
`MarketScanner._should_apply_adaptive_probe()` allowed a one-per-session probe
because learning mode and `learning_force_entry_per_session` were enabled, no
hard blockers were active, the dynamic probe confidence floor was met, and the
buy/sell edge gap was wide enough.

When that branch fires, `_run_single_scan()` mutates the in-memory
`DecisionOutput` with `out.can_create_execution_request = True`, marks the scan
as `forced_demo_entry: true`, adds the `adaptive_probe_entry` tag, calls
`_create_execution_request(...)`, and finally records
`execution_request_created: true`.

So the retained pair means:

`normal threshold path said no; forced-learning probe override created the execution request anyway.`

This was the intended forced-entry/probe design at the time, not an accidental
contradiction in the scan writer. The mechanism is now disabled in runtime
configuration via:

- `LEARNING_MODE=false`
- `FAST_LEARNING_MODE=false`
- `LEARNING_FORCE_ENTRY_PER_SESSION=false`

Future evaluations should treat these records as forced probe observations, not
normal strategy-performance evidence.

Normal demo threshold reference after disabling learning mode:

- `confidence_threshold: 0.52`
- `structure_score_min: 0.35`
- `spread_max_pips: 4.5`

## Retained Flag Artifact

The machine-readable audit is saved at:

`storage/forced_learning_probe_audit.json`

That file lists every detected forced probe record with source file, line
number, timestamp, decision, confidence, probe markers, and
`exclude_from_normal_strategy_performance: true`.

## Broker Fill Assessment

No retained fill/order/ticket record was found that can be tied directly to
these forced probe records. The retained evidence confirms forced execution
requests/candidates, not confirmed broker fills.

Until broker statements or MT5 history prove otherwise, any matching fills from
these timestamps should be labeled:

`forced_learning_probe_trade`

and excluded from normal strategy-performance statistics.

## Detected Forced Probe Timestamps

- 2026-05-04T17:31:27.773476+00:00 BUY_CANDIDATE
- 2026-05-04T19:23:39.310853+00:00 BUY_CANDIDATE
- 2026-05-14T17:00:03.965876+00:00 BUY_CANDIDATE
- 2026-05-19T12:55:23.510607+00:00 BUY_CANDIDATE
- 2026-05-20T17:00:37.980480+00:00 SELL_CANDIDATE
- 2026-05-21T12:32:13.185571+00:00 BUY_CANDIDATE
- 2026-05-21T13:45:11.333591+00:00 BUY_CANDIDATE
- 2026-05-25T15:30:16.229067+00:00 SELL_CANDIDATE
- 2026-05-27T07:20:03.196941+00:00 BUY_CANDIDATE
- 2026-05-29T00:30:08.454018+00:00 SELL_CANDIDATE
- 2026-05-29T07:22:15.839029+00:00 SELL_CANDIDATE
- 2026-06-01T11:00:09.628506+00:00 BUY_CANDIDATE
- 2026-06-01T15:30:26.961264+00:00 BUY_CANDIDATE
- 2026-06-03T18:28:00.043891+00:00 BUY_CANDIDATE
- 2026-06-04T04:52:10.907939+00:00 SELL_CANDIDATE
- 2026-06-09T15:45:25.299219+00:00 BUY_CANDIDATE
- 2026-06-10T15:30:03.666734+00:00 BUY_CANDIDATE
- 2026-06-12T00:01:18.749437+00:00 SELL_CANDIDATE
- 2026-08-13T15:30:00.467919+00:00 BUY_CANDIDATE
- 2026-08-14T07:32:25.025256+00:00 BUY_CANDIDATE
- 2026-08-18T15:30:07.876596+00:00 BUY_CANDIDATE
- 2026-08-19T00:03:56.219302+00:00 BUY_CANDIDATE
- 2026-08-20T15:30:47.614901+00:00 SELL_CANDIDATE
- 2026-08-21T00:04:08.599053+00:00 SELL_CANDIDATE
- 2026-08-24T15:38:16.517521+00:00 SELL_CANDIDATE
- 2026-08-25T07:30:10.956913+00:00 BUY_CANDIDATE
- 2026-08-26T15:32:03.624118+00:00 BUY_CANDIDATE
- 2026-08-28T08:18:18.854924+00:00 SELL_CANDIDATE
- 2026-08-31T10:05:02.149406+00:00 BUY_CANDIDATE
