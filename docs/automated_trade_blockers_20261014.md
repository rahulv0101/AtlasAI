# Automated trading investigation: 14 October 2026

## Conclusion

In the examined post-deployment window, the trader evaluated directional candidates
but **every one stopped before execution-request creation**. Timing discipline was
the principal blocker, followed by higher-timeframe alignment. This is not evidence
that approving the control queue, restarting the UI, or lowering confidence again
would make those candidates execute.

A controlled injected signal traversed the real trader timing/risk checks, guarded
automatic approval and ExecutionGate to a simulated broker. This establishes that
the tested Python path can submit; it is not a new MT5 fill or proof that a real
candidate would pass every later check. No real order was submitted for this audit.

## Scope and retained evidence

`tools/diagnose_trade_pipeline.py` streamed 2,709 active/archive event journals,
deduplicating exact stripped JSONL records by SHA256. The all-retained event range
was 2026-03-29 through 2026-10-14T09:51:09Z. These are retained timestamps, not a
verified deployment date or continuous-uptime assertion. Scan-result logs were not
used as authoritative trader decisions.

For the current-deployment window **2026-10-13T19:24:22Z to
2026-10-14T09:51:09Z** (21:24 SAST October 13 to 11:51 SAST October 14):

| Recorded outcome | Count |
| --- | ---: |
| Final chosen-status decision records | 746 |
| Flat/non-directional holds | 661 |
| Directional candidates | 85 |
| Directional candidates stopped at timing discipline | 67 |
| Directional candidates stopped at HTF alignment | 18 |
| Final status records indicating execution or failed submission | 0 |

These are logged decision records, not 85 independent market opportunities: repeated
observations of the same setup can count multiple times. Timing accounts for 67/85
(78.8%) of directional rejections in this window. The 0.55 automatic-approval floor
was **not reached** by these records; it cannot explain their first blocking stage.

The all-retained set contains 145 statuses marked `execute`. Do not treat those as
145 independently confirmed broker fills: some report `filled` with zero order/deal
IDs, while at least one older record has a broker-order ID. Broker history/provenance
would be needed to verify them. The operator's lifetime claim of zero automated
trades cannot be conclusively proven or disproven from those status labels alone.
The stored September 3 successful gate result also must not be confused with a new
automatic trade: it corresponds to the previously documented manual plumbing test.

Artifacts:

- `docs/diagnostics/atlas_pipeline_20261014.json`
- `docs/diagnostics/timing_block_population_20261014.json`
- `docs/diagnostics/trade_investigation_runtime_20261014.json`

## What timing actually rejected

All 67 timing records were also recovered from the active journal, matching the
archive-inclusive report. Failed conditions overlap:

| Missing condition | Records |
| --- | ---: |
| Strong/turning momentum | 55 |
| Tradable location | 49 |
| Visual confirmation | 43 |
| Visual tradable location | 35 |
| Visual momentum | 9 |
| Fresh entry or retest | 8 |
| Visual directional impulse | 8 |

Four records had **only** `fresh_entry_or_retest` missing, each on
`gold_breakout_retest`, with strong momentum and location `breakout`:

| UTC timestamp | Side | Chosen conviction | Reason code |
| --- | --- | ---: | --- |
| 2026-10-14 01:16:49 | sell | 0.90 | AWAITING_RETEST_CONFIRMATION |
| 2026-10-14 03:17:12 | buy | 0.88 | AWAITING_RETEST_CONFIRMATION |
| 2026-10-14 05:46:13 | sell | 0.91 | AWAITING_RETEST_CONFIRMATION |
| 2026-10-14 05:58:46 | sell | 0.90 | AWAITING_RETEST_CONFIRMATION |

Thus confidence is plainly not the problem for these examples. In
`AtlasTrader._timing_discipline_snapshot`, `breakout` is absent from the fresh-location
allowlist. In contrast, `GoldBreakoutRetestStrategy.generate_signal` checks a
breakout and a retest within its confirmation window before returning a direction.
The timing gate consumes the separate generic entry-location classification, not
that strategy's individual breakout/retest predicates. This is a compatibility/policy
issue to review, not permission to relabel every breakout as a confirmed retest.

One other record (2026-10-14T01:37:18Z) was rejected **only** for
`visual_momentum_confirmation`: momentum_continuation sell, conviction 0.59,
location pullback and price-derived momentum strong, but visual momentum weak.
`_timing_discipline_snapshot` explicitly uses visual-watch evidence as a veto when
`visual_watch_used_for_decision` is true. Existing tests explicitly cover that veto.
Although visual-read comments describe advisory evidence, it is not merely advisory
in this path. Removing its authority would change current entry policy and was not done.

Passing timing would only advance these candidates to later checks; it would not
guarantee a broker submission or profitable outcome.

## Other restrictions, not the current first blocker

- Trader candidate consideration uses confidence 0.44; guarded automatic demo
  approval separately requires `max(min_confidence_threshold, 0.55)`.
- `_strategy_to_setup_family` maps gold_breakout_retest and gold_liquidity_sweep
  to `breakout`. The stored tiny-live pilot allows **continuation only**. Adding
  breakout to timing alone would therefore not make those strategies pilot-eligible.
- Pilot limits are 8 trades/day and 8 consecutive losses; session control is still
  6 and 2. They are inactive constraints at zero trades/losses, not the explanation
  for the first trade being absent in this window.
- Control-item approval is a separate review workflow. It is not approval of an
  ExecutionGate trade request.

## Controlled verification

Executed:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_trade_pipeline_diagnostic_report.py tests/test_current_entry_policy_diagnostic.py tests/test_pipeline_diagnostic_20260923.py tests/test_cross_process_execution_gate_sync.py
```

**18 passed.** The end-to-end diagnostic injects controlled candidate/chart evidence
and forbids real `mt5.order_send`. Observed output: momentum_continuation buy,
conviction 0.59, automatic approval approved under `guarded_demo_auto_approval_v1`,
ExecutionGate success, simulated broker acknowledgment `BRK_00000001`.

New tests preserve the observed current policy: breakout versus retest/pullback,
the independent 0.55 approval floor, archive deduplication, exclusion of
decision-sample duplicates from the decision denominator, and protection against
overwriting an existing diagnostic artifact. They do not relax a trading gate.

## Changes and next decision

Only the read-only diagnostic tool, tests and audit artifacts changed. The diagnostic
now accepts a cutoff/output filename, refuses to overwrite prior evidence, excludes
non-final event types from its decision counts, and sorts retained execution examples
by timestamp. No live restart was required.

At 2026-10-14T14:51:57Z (16:51 SAST), backend worker 21440 was still running, heartbeat
age 12.3 seconds, cycle 34208. Latest completed status was flat momentum_continuation,
conviction 0.45/raw seed 0.0, `No directional trade signal selected.` Execution
remained enabled, mode live, gate demo; zero open positions and zero trades today.
Source hashes for `.env`, settings, trader, ExecutionGate and risk manager matched
the pre-investigation baseline.

No confident claim that the entire system is fault-free follows from one controlled
test. The evidence does show that current non-execution is explained by earlier
entry-policy rejections, rather than a newly demonstrated broken order-submission path.

The next bounded step is an **offline-only** review of strategy-confirmed retest
evidence versus the generic timing location label, and compatibility with the pilot's
allowed setup family. Any resulting timing/pilot change needs explicit approval
before demo deployment. Mid-range, momentum, risk, forced-entry, drawdown and other
safety settings remain unchanged. No policy change or trade was forced to satisfy
a trade-frequency target.
