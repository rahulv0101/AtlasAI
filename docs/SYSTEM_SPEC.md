# Atlas AI System Specification

## Purpose

Atlas AI is a modular `XAUUSD` trading system built around MetaTrader 5. This specification exists to make the system's edge, decision order, experiment rules, and operational safety contracts explicit and resistant to drift.

This document is normative for architecture, safety, and experiment discipline. It does not change trading logic on its own.

## 1. Edge Definition

### Primary Edge

The primary edge comes from rule-based signal generation on gold using a mix of:
- trend-following entries
- breakout participation
- selective mean-reversion behavior where explicitly designed

These signal families are the system's core source of expected return.

### Secondary Edge

The secondary edge comes from trade selection and operational quality improvements around the core signals:
- regime awareness
- signal ranking and allocator discipline
- conservative negative-quality blocking
- execution and spread safety
- risk management and exposure control

These layers are intended to improve trade quality and reduce bad exposures. They are not the primary source of alpha.

### What The System Is Not Relying On

The system is not currently relying on:
- the ML/meta-model for live decision authority
- post-entry outcome information in pre-trade logic
- hidden optimization or multi-factor experiment drift

The current ML/meta-model may be logged and analyzed, but it is not a trusted production edge source.

### Edge Boundaries

The following distinctions must remain clear:
- signals create the opportunity set
- filters reduce risk and remove low-quality setups
- risk management constrains loss and exposure
- execution quality preserves realized edge

Filters must never be described or implemented as the primary alpha engine.

## 2. Decision Hierarchy

The decision pipeline is ordered and must remain explicit:

1. Signal generation
   - strategies generate candidate directions and setup metadata
   - this is the core edge layer

2. Signal ranking
   - candidates are ranked and narrowed into a chosen trade idea
   - ranking may use allocator/history inputs, but must not silently replace signal logic

3. Conservative filter
   - the negative-quality filter may block the chosen setup
   - it may only veto or pass
   - it must not create trades, reverse trades, resize trades, or alter exit structure

4. Risk management
   - sizing and exposure rules are applied after the candidate survives prior checks
   - this layer may reduce risk or reject unsafe execution, but it does not create alpha

5. Execution
   - broker-safe order construction and placement
   - execution must preserve attribution and operational auditability

### Hierarchy Contract

Later layers must not silently become earlier layers:
- filters cannot become primary signal generators
- risk controls cannot become hidden signal selectors
- execution metadata must not change trade intent

## 3. Conservative Filter Contract

The conservative pre-entry filter is a negative-quality blocker and the current production candidate under paper evaluation.

### Scope

The filter:
- uses only pre-entry information
- blocks only the lowest-quality setups
- does not predict winners
- does not alter entries, exits, sizing, or strategy thresholds
- does not create new trades

### Forward-Safety Contract

Allowed inputs must be fully available at or before entry, for example:
- session
- regime classification
- spike/extension context
- liquidity/volume context
- pre-entry confidence metadata already present in the chosen setup

Forbidden inputs include:
- realized outcome
- future bars
- MFE/MAE
- exit reason
- hold duration
- post-entry labels

No post-entry or leaked outcome data may be used in live or paper-decision paths.

### Current Rule Intent

The conservative filter exists to block negative-quality conditions such as:
- weak late-session trend participation
- overstretched post-spike entries
- poor liquidity conditions

These are risk-reduction rules, not winner-prediction rules.

### Acceptance Contract

The filter may only be considered promotable if side-by-side evidence shows:
- more losers removed than winners removed
- trade count remains meaningful versus control
- expectancy does not worsen
- max drawdown does not materially worsen

### Failure Contract

The filter must be rejected or kept in testing if:
- it removes too many winners
- it collapses trade count
- expectancy worsens
- drawdown worsens materially
- any rule is found to depend on post-entry information

## 4. Experiment Isolation Rules

All experiments must obey one-variable isolation.

### Isolation Rule

For any control-vs-experiment comparison:
- only one factor may differ
- all other parameters must remain identical

### Allowed Difference

The intended experiment variable must be named explicitly. For the current paper rollout:
- control: conservative filter OFF
- experiment: conservative filter ON

All other settings must match:
- symbol
- timeframe
- strategy set
- risk settings
- overlays
- live/paper mode
- ML/meta-model state

### Invalid Experiment States

An experiment is invalid if:
- multiple parameters differ
- control or experiment unexpectedly runs live
- ML influence is enabled when the experiment is not about ML
- runtime flags drift away from the intended preset

## 5. Paper Comparison Metrics

Paper comparison exists to evaluate trade-quality changes, not just raw win rate.

### Required Lane Metrics

Each lane must report:
- trades taken
- filtered/skipped trades
- trades closed
- PnL per closed trade
- win rate
- expectancy
- profit factor
- drawdown
- average win
- average loss

### Required Cross-Lane Metrics

The comparison report must also include:
- trades taken by both lanes
- trades taken only by control
- trades taken only by experiment
- trades filtered by experiment
- losers avoided
- winners removed
- net trade-quality impact
- net expectancy of removed trades

Win-rate improvement alone is not sufficient evidence. Removed-trade expectancy is required so a filter cannot look better merely by shrinking the sample.

## 6. ML / Meta-Model Role

The ML/meta-model is currently a research component, not a production decision authority.

### Allowed Role

The ML layer may:
- log scores
- support research analysis
- be evaluated offline
- be tested in controlled paper comparisons

### Forbidden Role

The ML layer must not:
- influence production or production-style comparisons by default
- silently boost or veto trades in the current conservative-filter rollout
- be treated as trusted edge without controlled promotion evidence

### Promotion Preconditions

The ML layer may only be considered for promotion after passing all of:
- out-of-sample validation
- walk-forward validation
- controlled paper comparison against a locked baseline
- decision checklist review

Until then, ML outputs are observational.

## 7. Promotion Pipeline

All nontrivial changes should move through a staged promotion path:

1. Research
   - idea generation and exploratory analysis
   - failure condition: relies on leaked or unavailable features

2. Offline validation
   - backtest or labeled-trade study against a locked baseline
   - failure condition: expectancy worsens, sample collapses, or results are not explainable

3. Paper comparison
   - side-by-side control vs experiment
   - failure condition: live/paper safety breach, experiment drift, or harmful trade-quality outcomes

4. Decision checklist
   - machine-readable acceptance logic over paper results
   - failure condition: insufficient sample, mixed evidence, or failed hard checks

5. Promotion
   - limited rollout only after prior evidence is complete
   - failure condition: any unresolved safety, leakage, or drift concern

No stage may be skipped for changes that influence trading decisions.

## 8. System Design Principles

- edge comes from signals first, then disciplined filtering and execution quality
- filters block bad trades; they do not predict good trades
- the system must remain explainable
- all live logic must be forward-safe
- outcome leakage is forbidden
- simpler rules are preferred over complex opaque models
- experiments must be reproducible and isolated
- operational safety outranks speculative improvement
- paper evidence is required before promotion

## 9. Operational Contracts

### Live / Paper Routing

- effective mode must be explicit
- mismatches between configured mode and runtime mode must fail loudly unless deliberately overridden
- paper experiments must force `execute_trades=false`

### Attribution

- comments, logs, and run manifests must preserve enough metadata to reconstruct what happened
- experiment lanes must remain separately attributable

### Drift Resistance

- specs, presets, and reports must agree
- experiment comparisons should be reviewable after the fact without relying on memory or operator intuition

