# Atlas Verification Process

## Current Baseline Mechanism

Git is not currently usable in this workspace: `git --version` is not available on
`PATH`, and the workspace does not contain a `.git` directory. Until Git is installed
and this folder is initialized safely, behavior-sensitive changes must use the
file-based baseline mechanism below.

## When To Capture A Baseline

Before any change described as "must not change trading behavior", capture a fixed-window
baseline before editing code. After the change, run the same command again and compare
the two JSON files programmatically.

The baseline command runs Atlas's existing backtest path only. It forces
`execute_trades=False` and uses an isolated temporary storage directory, so it must not
write to live trading state or send anything through live execution.

## Fixed Window

Default window:

```powershell
2026-08-03T00:00:00+00:00 to 2026-08-07T23:59:59+00:00
```

Use the same symbol and timeframe for the before and after captures. If a task is about
a timeframe migration, pass the timeframe explicitly.

## Capture

```powershell
python scripts\capture_baseline.py --label before_<task_label> --timeframe M15
```

The output is written to:

```text
docs/baselines/baseline_<timestamp>_<label>.json
```

Each baseline includes:

- trade list
- entry and exit bars/times/prices
- trade count
- wins and losses
- win rate
- drawdown
- average hold bars
- expectancy and profit factor

## Compare

```powershell
python scripts\diff_baselines.py docs\baselines\<before>.json docs\baselines\<after>.json
```

The diff command prints a structured JSON result. It exits with code `0` when the
baseline behavior matches and `1` when metrics or the trade list changed.

For observability-only tasks, an empty diff is the expected result. If the diff is not
empty, stop and inspect the changed metrics/trades before continuing.

## Moving To Git Later

When Git becomes available, initialize the repository only after confirming it will not
discard or overwrite existing files:

```powershell
git init
git add .
git commit -m "Baseline current Atlas workspace"
```

After that, Git commits should become the primary baseline mechanism, but the JSON
baseline process should still be used for trading-behavior validation because it captures
domain-specific outputs that a source diff alone cannot prove.

## Journal Memory Changes

For journal-maintenance or reporting-reader changes, also capture allocation and
output-equivalence artifacts in isolated temporary stores before and after editing:

```powershell
.venv\Scripts\python.exe tools/benchmark_event_maintenance.py --output docs/baselines/maintenance_before.json
.venv\Scripts\python.exe tools/benchmark_missed_entry_collector.py --output docs/baselines/collector_before.json
# Apply the narrowly scoped change, then:
.venv\Scripts\python.exe tools/benchmark_event_maintenance.py --output docs/baselines/maintenance_after.json --compare docs/baselines/maintenance_before.json
.venv\Scripts\python.exe tools/benchmark_missed_entry_collector.py --output docs/baselines/collector_after.json --compare docs/baselines/collector_before.json
```

Use date/task-specific filenames in actual work; never overwrite a prior baseline.
Both tools report peak Python allocations, elapsed time and programmatic output
differences. Maintenance compares retained/archive hashes as well as counts; the
collector compares normalized all-history summary fields. An empty result diff is
required. These checks do not substitute for a trade replay when trading behavior
could be affected, or for measuring the real worker through multiple completed
cycles and a maintenance interval after deployment. Do not benchmark by rewriting
the live journals, and do not interpret one startup-memory sample as leak resolution.
