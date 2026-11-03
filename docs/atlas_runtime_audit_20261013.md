# Atlas runtime audit and repair

Updated 2026-10-13, Africa/Johannesburg. Runtime timestamps below are UTC unless
explicitly labelled SAST. This is an evidence-based snapshot, not a promise of
profitability or a certification of every historical change.

Final verification at **2026-10-13 09:08:40 SAST** is saved in
`docs/diagnostics/runtime_repair_20261013.json`: backend worker 25380 unchanged,
UI worker 9044, heartbeat fresh at cycle 32783 (158 seconds old against 450-second
threshold), HTTP health 0.188 seconds and compact status 0.328 seconds. London
session open; latest completed decision still flat at conviction 0.45, no directional
signal. Backend resident/private memory 289/1076 MiB. The complete combined targeted
test command finished with **232 passed in 31.57 seconds**. Earlier snapshots below
are retained with their original timestamps rather than presented as current.

## What Atlas is supposed to do

`docs/SYSTEM_SPEC.md`, `AGENTS.md`, `README.md`, and the running code describe a
rule-based XAUUSD trading assistant with a separately controlled execution path:

1. MT5 market/account data and higher-timeframe context feed `AtlasTrader.step()`.
2. Strategies generate directional candidates or flat signals.
3. Selection/ranking and decision analysis evaluate those candidates. Raw signal
   confidence, contextual ranking score, and final conviction are different fields.
   A positive conviction score on a flat signal is not a buy/sell opportunity.
4. Existing confidence, consensus, HTF, timing, safety and risk checks can veto a
   candidate. A veto does not itself mean a technical failure.
5. Trader-origin requests go through create, conditional guarded-demo approval,
   and `ExecutionGate.execute()`, then the MT5 adapter. Approval is not universal.
6. Journals and learning modules retain observations/evidence. Research/promotion
   requires its existing review boundaries; the current candidate meta-model is
   not trusted live execution authority.
7. Ask Atlas/Brain v2 explains evidence through intent/domain/evidence/planning/
   synthesis/relevance stages. Typed and voice input share that answer path;
   explicit commands are separate from informational questions.

The intended outcome is reliable evaluation and safe, inspectable execution when
conditions qualify, not a guaranteed number of trades every day.

## Confirmed technical faults repaired

### UI process-query overhead

Profiling `_build_compact_status_payload()` before changes measured 2.672 seconds
overall, including 2.417 seconds in two desktop reconciliation calls which launched
four subprocess queries. `/api/health` responded while `/api/status/compact` exceeded
a three-second diagnostic timeout. This proves a slow status assembly path, not the
unique cause of every historical frontend outage.

`atlas_ai/runtime_processes.py` now reuses the native `psutil` inspection pattern
already used by `tools/monitor_full_stack.py`. It reads command arguments and parent
PIDs without launching PowerShell. The UI caches only advisory desktop process
observations for five seconds; trading state and execution permissions are not cached
by this change. Bounded subprocess fallbacks remain for installations without psutil.

Final local builds measured 0.203, 0.219 and 0.235 seconds. After deployment, HTTP
compact status measured 0.281 and 0.188 seconds. Other checked endpoints (`/`,
`/api/health`, `/api/events`, `/api/runtime`) returned HTTP 200 in under 0.1 seconds
in the final endpoint check.

### Watchdog blind spots

`tools/backend_heartbeat_watchdog.py` previously checked heartbeat/process presence
only, checked backend and UI sequentially, and selected the first matching process,
which could be the virtualenv launcher rather than the worker.

It now:

- Probes both `/api/health` and `/api/status/compact`, validates their JSON shape,
  records errors and response times, and bypasses HTTP proxies for localhost.
- Alerts on the first HTTP failure; two consecutive failed samples are required
  before restarting an otherwise live UI, avoiding a restart on one slow request.
- Runs backend and UI checks on independent schedules, retaining one supervisor lock.
- Applies the same supervisor lock to restart-capable `--once` invocations.
- Identifies leaf workers in launcher/worker pairs; multiple independent workers
  cause an explicit error, not an arbitrary kill or another launch.
- Refuses a replacement while a matching process remains; UI restart additionally
  checks that the listening port has been freed.
- Requires a backend process as well as a fresh heartbeat to declare health.
- Uses native process inspection here too. Before that final improvement, overnight
  logs captured 22 backend and 23 UI inventory errors from eight-second PowerShell
  timeouts. They did not cause an automatic restart. Native checks subsequently
  measured 0.032 and 0.015 seconds.

The existing scheduled task was retained. It starts the supervisor every five
minutes; the resident supervisor checks each service roughly every 60 seconds plus
check duration. HTTP probes have five-second socket timeouts. This work verified
real healthy probes and isolated hung-response recovery tests, not a live kill drill.

## Deployment and current evidence

Only UI and watchdog workers were restarted. The M1 backend worker remained PID
25380, created 2026-10-12 11:39:49 SAST. Its heartbeat advanced from cycle 32435
at 2026-10-12T20:56:56Z to cycle 32668 at 2026-10-13T03:29:00Z. This establishes
continued evaluation across the observation interval, not measured 100% uptime.

Final native-inspection deployment: UI launcher PID 3204 at 2026-10-13 05:29:23
SAST; watchdog launcher PID 24596 at 05:29:35 SAST. Worker PIDs are recorded in
heartbeats/watchdog events. UI launch output is retained in
`logs/ui_server_20261013.stdout.log` and `.stderr.log`; watchdog stderr is retained
in `logs/watchdog_20261013.stderr.log`.

At status time 2026-10-13T03:29:43Z:

- Timeframe M1; `execute_trades=true`; `effective_mode=live`; gate environment demo.
- Thresholds: confidence 0.44, structure 0.35, spread maximum 4.5.
- Configured six strategies: ema_crossover, pullback_trend, momentum_continuation,
  gold_micro_impulse, gold_breakout_retest, gold_liquidity_sweep.
- Chosen: momentum_continuation, **flat**, conviction 0.45, raw seed 0.0.
  Message: `No directional trade signal selected.` The six raw signals were flat.
- Asia session open; operator present; kill switch/sleep/pause off; reconciliation
  clear; no current safety lock or cooldown. Risk: equity/balance 110222.43,
  daily/weekly loss fractions 0, zero open positions and zero trades today.
- Running status reports broker connected and symbol selected; macro inputs available.
  No independent competing MT5 session or broker order was opened for this audit.

The immediate reason for no trade in that cycle is the lack of a directional
candidate. A flat candidate with conviction above 0.44 is still not a valid entry.

## Constraints and unresolved items

- `AtlasTrader._guarded_demo_auto_approval_decision()` requires confidence at least
  `max(min_confidence_threshold, 0.55)`. Thus 0.44 is not the complete automatic
  execution threshold. This policy was NOT lowered.
- Auto-approval considers continuation/pullback/breakout families, but the stored
  tiny-live pilot permits **continuation only**, size <=0.01. Other families can
  still be rejected later. This policy was NOT broadened.
- Pilot limits remain 8 trades/day, 8 consecutive losses, stored daily_loss_cap 0.6.
  Session controller remains stricter: 6 trades/day, 2 consecutive losses. The user's
  earlier requested reconciliation explicitly required confirmation; none was applied.
  Do not interpret the stored loss-cap number as a currency amount or percent without
  checking its consuming calculation.
- No weekly reset was performed. Current daily and weekly drawdown were zero; there
  was no current drawdown block to remove. Rebaselining still requires confirmation.
- Optional LLM provider reports `quota_exhausted` / no remaining credits. Billing is
  external and was not changed. Rule-based decisions continue; not all AI-provider
  features can be called healthy while this remains unresolved.
- Learning status reports a stale evolved pool and a candidate meta-model. Observing
  and retaining lessons is not proof of a profitable learned edge or promotion readiness.
- Backend private memory remained about 1021 MiB (resident 365 MiB in the sample).
  This repair does not establish that all memory/performance problems are solved.
  `KnowledgeStore.maintain_events()` still keeps full normalized event strings for
  deduplication while holding its store lock; it warrants separate profiling. It was
  not asserted to be the proven cause of a particular historical stall.
- A timing snapshot on a **flat** signal may say `passed` despite missing entry
  features because it does not hard-block a non-entry. Do not read that as proof a
  hypothetical directional trade passed timing; the strategy still has to signal.

## Verification and scope

232 targeted tests passed across UI/runtime reliability, simulated pipeline,
execution safety, cross-process gate synchronization, bounded JSONL reads, Brain
observability/follow-ups, structured trade answers, risk, macro refresh, and learning
evidence provenance. `python -m compileall -q atlas_ai tools/backend_heartbeat_watchdog.py
tests/test_runtime_reliability.py` passed. This is not a claim of a full-suite run.

`test_injected_valid_candidate_reaches_real_gate_and_simulated_broker` injected a
qualifying candidate into AtlasTrader, exercised its real timing/risk/gate path and
guarded approval, and confirmed `order_submit_attempt` plus success at TestBrokerAdapter.
Actual MT5 order submission was forbidden by the test. This proves the controlled
Python execution path, not a fresh real broker fill or strategy profitability.

No trading strategies, thresholds, risk limits, approval rules, learned state or
execution policy were intentionally modified. No backtest parameter search, forced
entry or manual broker test order was performed. Read-only diagnostic requests and
normal UI startup can update their existing operational artifacts.

Git remained unavailable in the shell. Original source copies were preserved under
`docs/baselines/20261012_runtime_reliability_before/` before edits; these are a rollback
reference, not a fabricated pre-migration trading baseline. Changes are confined to
UI process inspection, watchdog supervision, shared read-only process inspection,
regression tests, and this report.

## Follow-up: backend journal memory (October 13)

The deployment and unresolved-items sections above describe the earlier UI/watchdog
repair. The following is additional work, not a claim that the earlier backend was
already running these changes.

### Confirmed allocations and fixes

1. `KnowledgeStore.maintain_events()` held full canonical event strings for
   deduplication plus a retained deque of payloads. It now spools payloads to a
   temporary file, retaining SHA256 digests and offsets. Digest matches are checked
   against the original canonical JSON so collisions cannot discard distinct events.
   Memory still scales with record-count metadata, not full history payload bytes.
   Archive order, first-occurrence deduplication, malformed-line handling and retained
   output are preserved. The existing same-store lock remains; this is not a new
   cross-process journal-rotation locking protocol.
2. Periodic housekeeping in `main.py` now reports duration and logs exceptions,
   then retries at the next scheduled interval. A failure no longer silently kills
   that housekeeping thread. Existing completion logging is retained.
3. `AtlasTrader._publish_status()` calls
   `build_blocker_selectivity_offline_collector()` on each status publication.
   Despite its name, this collector is on the live status path. It loaded the full
   missed-entry outcome-review journal (171,248,728 bytes when inspected), then
   materialized another list of reviews. It now iterates batches and accumulates
   identical all-history counters. No tail cutoff or evidence-window change was made.
   Unavailable-reference/direction exclusions and target-blocker filtering are unchanged.

### Before/after evidence

Original source copies are in `docs/baselines/20261013_maintenance_before/`.
Two repeatable synthetic benchmarks use temporary stores only:

| Operation | Input bytes | Old peak allocation | New peak allocation | Output diff |
| --- | ---: | ---: | ---: | --- |
| Event maintenance | 33,445,196 | 58,529,761 | 1,476,708 | Empty; active/archive SHA256 identical |
| Outcome collector | 34,005,040 | 102,026,245 | 63,271 | Empty; all result fields except timestamp/path identical |

Artifacts: `maintenance_20261013_before.json`, `maintenance_20261013_after.json`,
`collector_20261013_before.json`, `collector_20261013_after.json` under
`docs/baselines/`. These are memory/output equivalence checks, not backtests.
Synthetic elapsed times were **not faster**: maintenance 2.125 -> 4.818 seconds;
collector 0.333 -> 1.063 seconds. This change removes large allocations, not the
need to inspect historical rows. Timings were separate runs on a busy laptop.

88 targeted tests passed, covering the collector, missed-entry audit, housekeeping,
knowledge store, JSONL tail, controlled pipeline, runtime reliability and cross-process
ExecutionGate synchronization. Compileall passed. The collector regression explicitly
forbids full-text/list loading of its review journal, checks malformed/blank/scalar rows,
both target-filter modes, unknown outcomes, and bounded allocation with 4,000 batches.
The maintenance tests cover output equivalence, forced digest collisions, malformed
rows, bounded payload memory and failed replacement. No full test-suite or historical
trade-replay comparison was run in this follow-up.

### Deployment checkpoints and limits

- Maintenance-only worker 5800 started at 2026-10-13 14:47:31 SAST. It advanced
  heartbeat counts 32933 through 32935 initially and was still running at count 33242
  at 19:22:36Z. This was not enough to solve memory use: an early transient reached
  817.55 MiB private, and private memory was 912.67 MiB before the next restart.
- Both fixes were loaded by worker 21440 (launcher 6008), started at 21:24:22 SAST.
  HFM connection was reported at 21:24:29; heartbeat 33244 at 19:24:33Z and completed
  status at 19:24:46Z. Initial sampled private memory was 337.79 MiB.
- Watchdog supervision was restored after each controlled backend replacement.
  UI worker 9044 / launcher 3204 was not stopped or replaced.
- `.env`, settings, `trader.py`, ExecutionGate and risk-manager source hashes still
  matched the pre-deployment snapshot. No confidence, consensus, timing, approval,
  session, drawdown or pilot policy was changed. No test order was sent.
- The smaller audit-context backfill still materializes/re-writes its audit history,
  and the collector still scans the outcome journal. Neither has been claimed fully
  constant-memory/constant-time. Long-run stability needs observation beyond startup.
- Diagnostic evidence is under `docs/diagnostics/maintenance_deploy_before_20261013.json`,
  `maintenance_postrestart_20261013.json` and `collector_postrestart_20261013.json`.

### Overnight verification: October 14, 05:58 SAST

The first five completed cycles (33244-33248) stayed on worker 21440. One-second
sampling over 303.5 seconds recorded a private-memory peak of 410.46 MiB and
resident-memory peak of 191.81 MiB. These are sampled peaks, not allocator maxima.

At 2026-10-14T03:58:00Z, the **same worker** had run for about 8 hours 34 minutes:

- Private memory: **406.30 MiB**; resident: **169.45 MiB**. Compare with 912.67 MiB
  private on the maintenance-only worker before the collector fix, and 1076.32 MiB
  in the earlier pre-maintenance snapshot. This supports a substantial reduction,
  not a claim that every allocation/leak or future performance problem is solved.
- Heartbeat 33687, timestamp 03:57:29Z, age 30.38 seconds at the check.
- Retained watchdog observations span all 444 counts from 33244 through 33687,
  with no missing cycle numbers. There were 515 healthy backend checks and 510
  healthy UI checks, and **no unhealthy/restart actions** in this interval.
- 103 `duplicate_watchdog_exit` records were benign five-minute scheduled launches
  declining the already-held supervisor lock. They are not duplicate live traders.
- Cycle-start intervals across 443 consecutive pairs: median **68.247 s**,
  p90 **74.351 s**, p95 **78.437 s**, max **104.129 s**. These include the configured
  60-second polling sleep; they are not pure processing durations.
- All 33 retained maintenance runs in this interval completed; no failure events.
  Durations: median 22.177 s, p95 31.798 s, max 39.134 s. This verifies operation
  through repeated housekeeping, not just the first few cycles after restart.
- UI worker 9044 was unchanged. HTTP `/api/health` returned 200 in 0.059 s and
  `/api/status/compact` returned 200 in 0.472 s. No UI restart was needed.
- Status remained `execute_trades=true`, `effective_mode=live`, gate `demo`;
  confidence 0.44, structure 0.35, spread maximum 4.5. Policy-source hashes still
  matched. Latest completed status at the initial file read said
  `No directional trade signal selected.` Chosen momentum_continuation was flat,
  conviction 0.45, raw confidence seed 0.0; zero open positions/trades today and
  zero daily/weekly drawdown. A later API response completed during this check.

Evidence: `docs/diagnostics/memory_followup_20261014.json` and
`docs/diagnostics/memory_overnight_20261014.json`. This closes verification of these
two targeted memory fixes across the observed overnight interval. The approval-floor,
session-policy and external LLM quota issues listed above remain separate and were
not silently changed. No weekly drawdown reset was performed.
