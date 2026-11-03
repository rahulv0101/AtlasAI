# Reliability Backlog

These items are intentionally deferred until after the active `setup_quality_rollout_20260404` paper experiment, unless the run becomes unstable.

## Deferred Items

1. Harden duplicate Windows child-process behavior
- Investigate why a non-venv Python child can appear under the venv `paper-compare` parent on Windows.
- Goal: ensure one authoritative comparison worker writes run artifacts.
- Constraint: do not interrupt or alter the active experiment unless duplicate behavior causes instability.

2. Tighten broker-clock normalization for freshness diagnostics
- Current diagnostics can show a negative bar age when broker/server time is ahead of local UTC reference time.
- Goal: normalize freshness calculations so diagnostics remain accurate and easy to interpret without weakening stale-data protection.
- Constraint: keep the stale-data guard strict.
