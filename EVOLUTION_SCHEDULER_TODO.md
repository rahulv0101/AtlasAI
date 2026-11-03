# Evolution Scheduler Implementation — Task Tracker

## Phase A: Create evolution_scheduler.py
- [x] Create `atlas_ai/strategies/evolution_scheduler.py`
- [x] EvolutionScheduler class with thread-safe daemon loop
- [x] evolve_generation() on interval
- [x] ShadowTester.step() for backtested candidates
- [x] PromotionGovernor rules for stage transitions
- [x] EvolutionDrawdownGuard for demo strategies
- [x] EvolutionReporter artifact writing
- [x] Scheduler state persistence
- [x] Safe failure handling (never crashes trading)
- [x] Stop signal for clean shutdown

## Phase B: Wire into main.py
- [x] Add `start_evolution_scheduler_background()`
- [x] Call in `run_live()` when evolution_enabled
- [x] Call in `run_live_with_ui()` when evolution_enabled
- [x] Add `evolution` CLI mode with --once, --generations, --dry-run, --interval-hours
- [x] Prevent duplicate scheduler threads

## Phase C: API/UI Integration
- [x] Expose scheduler state in ui_server.py /api/runtime
- [x] Fields: evolution_enabled, scheduler_status, current_generation, etc.

## Phase D: Tests
- [x] Create `tests/test_evolution_scheduler.py`
- [x] test_scheduler_runs_evolve_generation_on_interval
- [x] test_scheduler_does_not_execute_trades
- [x] test_scheduler_writes_reporter_artifacts
- [x] test_scheduler_applies_promotion_governor
- [x] test_scheduler_handles_failure_without_crashing
- [x] test_scheduler_does_not_start_duplicates
- [x] test_cli_evolution_once_runs_successfully
- [x] test_runtime_exposes_scheduler_state

## Phase E: Documentation
- [x] Update TODO_strategy_evolution.md
- [x] Update TODO.md

