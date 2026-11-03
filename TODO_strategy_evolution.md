# SAFE Autonomous Strategy Evolution Engine — Implementation Plan

## Architecture Overview
The strategy evolution engine is a **proposal-only** system. It creates, backtests, evaluates, and proposes strategies for promotion into DEMO. It NEVER executes trades. Execution remains the sole responsibility of ExecutionGate.

## Delivery Order

### Phase 1 — Core DNA and Validation ✅ COMPLETE
- [x] `StrategyGene` + `StrategyChromosome` dataclasses
- [x] `GoldValidator` with XAUUSD hard rules
- [x] `StrategyFactory` (random + guided generation)

### Phase 2 — Fitness and Evolution ✅ COMPLETE
- [x] `FitnessScorer` with multi-metric scoring
- [x] `EvolutionEngine` with population of 50, elites, crossover, mutation
- [x] Persistence to `evolution_population.json`

### Phase 3 — Shadow Testing and Promotion ✅ COMPLETE
- [x] `ShadowTester` before demo promotion
- [x] `PromotionGovernor` — demo promotion rules (max 3, 0.5% risk, 100-trade milestone)
- [x] `StrategyExplainer` plain-English output

### Phase 4 — Reporting and UI ✅ COMPLETE
- [x] `EvolutionReporter` — `leaderboard.json` + `evolution_log.json`
- [x] API endpoints: `/api/strategies/leaderboard`, `/api/strategies/evolution-log`, `/api/strategies/<id>`

### Phase 5 — Integration and Safety ✅ COMPLETE
- [x] Settings additions (EVOLUTION_ENABLED, INTERVAL, MAX_DEMO, RISK_PCT, HUMAN_APPROVAL)
- [x] Trader integration bridge (`evolution_population.json` → trader evolved pool)
- [x] Drawdown guard pausing evolved strategies
- [x] No bypass of ExecutionGate/safety guards

### Phase 6 — Tests ✅ COMPLETE
- [x] gene validity
- [x] GoldValidator rejections
- [x] fitness disqualifiers
- [x] crossover/mutation safety
- [x] shadow testing before promotion
- [x] max 3 demo strategies
- [x] no bypass of ExecutionGate
- [x] live promotion requires human approval
- [x] persistence across restart
- [x] leaderboard/explanation endpoints
- [x] trader loads demo chromosomes from population file
- [x] evolved strategies cannot execute directly (signal-only)

### Phase 7 — Evolution Scheduler ✅ COMPLETE
- [x] `EvolutionScheduler` background daemon
- [x] Wired into `main.py` for live/live-ui modes
- [x] CLI `evolution` mode with `--once`, `--generations`, `--dry-run`, `--interval-hours`
- [x] API/UI runtime exposure via `/api/runtime`
- [x] Singleton duplicate prevention
- [x] Safe failure handling without crashing trading
- [x] Scheduler state persistence
- [x] Tests: `tests/test_evolution_scheduler.py` (17 tests, all passing)

