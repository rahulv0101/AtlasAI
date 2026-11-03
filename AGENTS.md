# AGENTS.md

## Project
Atlas is a trading assistant and operator interface. The goal is to improve clarity, reasoning, observability, and interaction quality without causing drift in core trading behavior.

## Non-negotiable rules
- Do not silently change core trading purpose.
- Do not silently change strategy logic.
- Do not silently change risk policy.
- Do not silently change execution policy.
- Do not introduce uncontrolled self-modification.
- Do not introduce hidden parameter drift.
- Do not let conversational memory alter trading decisions unless explicitly requested and clearly bounded.
- Do not let voice features broaden system actions beyond explicit command routing.

## Architecture priorities
Prefer small, clean, incremental changes over large rewrites.

Keep these responsibilities separated:
- intent detection
- domain detection
- evidence collection
- answer planning
- synthesis
- relevance validation
- memory
- knowledge access
- voice input/output
- command routing

Preserve Brain v2 modularity.

## Answering rules
Atlas must answer the user’s actual question, not dump nearby state.

Required answer flow:
1. normalize question
2. detect likely intent candidates
3. detect likely answer domain
4. collect relevant evidence
5. select answer plan
6. synthesize response
7. run relevance check before returning

If relevance is weak:
- reject the draft answer
- narrow evidence
- regenerate
- prefer the closest useful grounded answer

Current system state must not be the universal fallback for every unclear question.

## Relevance guard
Protect against irrelevant fallback leakage.

Examples of bad answers that must be blocked:
- self-knowledge question answered with HOLD / PAPER / RUN
- capability question answered with blocked trade state
- limitation question answered with generic runtime status

Questions about:
- capabilities
- limitations
- self-knowledge
- intelligence
must prefer self-model and structured knowledge over trading-state dumps.

## Voice and typed input
Voice and typed input must share the same answer path.

Required flow:
speech -> transcription -> Ask Atlas service -> Brain v2 answer engine -> response -> optional TTS

Do not build a separate voice brain.

Voice mode should:
- be shorter than standard UI mode
- answer directly first
- include minimal supporting detail unless asked

## Memory rules
Use memory for continuity, not hidden control changes.

Memory may:
- resolve follow-up questions
- preserve prior answer context
- improve continuity

Memory must not:
- override current state
- invent facts
- silently change trading logic
- broaden command execution

Store compact structured context such as:
- prior question
- selected intent
- selected domain
- selected plan
- main conclusion
- supporting reasons
- confidence
- referenced entities

## Knowledge rules
Prefer structured knowledge over prompt blobs.

Keep knowledge in repo-visible structured files where practical, such as:
- knowledge/system_overview.json
- knowledge/market_knowledge.json
- knowledge/strategy_knowledge.json
- knowledge/risk_policy.json
- knowledge/readiness_rules.json
- knowledge/blockers_catalog.json
- knowledge/experiment_history.json
- knowledge/lessons_learned.json
- knowledge/glossary.json

Do not hallucinate knowledge that is not present in state, rules, memory, or structured knowledge.

## Adaptation and learning rules
Atlas may observe and explain learning/adaptation speed, but must not become reckless.

Allowed:
- observe chart conditions
- score evidence strength
- classify learning mode
- report adaptation speed
- explain regime shifts
- recommend safe adjustments
- make bounded confidence adjustments if already part of design

Not allowed:
- uncontrolled retraining
- silent strategy rewrites
- hidden policy changes
- overreactive adaptation from weak evidence

Preferred learning policy:
- slow_path for weak, mixed, noisy, or short-lived evidence
- medium_path for moderate evidence
- fast_path for strong, persistent, aligned evidence

Fast-path must still use guardrails:
- max confidence delta
- minimum confirmation threshold
- persistence requirement
- hysteresis or cooldown
- overreaction scoring

## Command safety
System commands must be explicit.

Examples:
- "Atlas wake"
- "Atlas sleep"
- "start system"
- "close system"

Informational questions must not trigger actions.

Command execution should remain isolated from informational answering.

## Style rules
Atlas should sound:
- calm
- concise
- operator-focused
- grounded
- professional

Do not:
- repeat honorifics in every sentence
- use robotic filler
- over-explain simple answers
- use status dumps as generic fallback

Do:
- answer directly first
- support with reasons when useful
- separate facts from inference
- state uncertainty clearly when needed

## Testing expectations
When changing Atlas, prefer adding or updating tests for:
- self-knowledge queries
- capabilities queries
- limitations queries
- status queries
- blocker queries
- readiness queries
- comparison queries
- follow-up continuity
- voice mode brevity
- relevance guard failures
- adaptation and learning-speed reporting

Permanent failure cases to prevent:
- "How much do you know about the markets?" answered with trade-block state
- "What can you do?" answered with HOLD or PAPER status
- "What are your limitations?" answered with run metadata
- follow-up "Why?" losing prior context
- voice path returning different reasoning than typed path

## Implementation guidance
Before making major changes:
- inspect existing Brain v2 files
- preserve compatibility where possible
- refactor incrementally
- keep debug output inspectable
- avoid duplicate logic when a shared module can be reused

Prefer adding clean adapters rather than breaking working paths.

## Debugging and observability
Changes should keep or improve inspectability.

Where useful, expose:
- intent candidates
- selected domain
- evidence used
- evidence rejected
- answer plan
- relevance result
- learning mode
- evidence strength
- overreaction/stability metrics

## Success standard
A good change makes Atlas:
- more relevant
- more explainable
- more stable
- easier to inspect
- safer to extend

A bad change makes Atlas:
- more verbose but not smarter
- faster but less stable
- more adaptive but less trustworthy
- more feature-rich but harder to reason about
