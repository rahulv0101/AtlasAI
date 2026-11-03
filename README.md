# Atlas AI

Atlas AI is a self-improving gold (XAUUSD) trading system designed for HFM via MetaTrader 5.

## Architecture And Operating Contracts
- System specification and experiment rules: [`docs/SYSTEM_SPEC.md`](docs/SYSTEM_SPEC.md)
- The specification makes the system edge, decision hierarchy, conservative-filter contract, ML constraints, and promotion pipeline explicit.
- Treat that document as the source of truth for architecture, experiment isolation, and paper-comparison interpretation.

## Features
- HFM integration through MT5 login and order execution.
- Auto-opens HFM terminal when the system starts.
- Gold-focused strategy set: EMA trend, RSI mean reversion, breakout.
- StrategyLab that evolves new rule-based strategies from market data.
- Persistent learning memory saved to disk (`storage/learning_state.json`).
- Event and status telemetry saved for observability (`storage/events.jsonl`, `storage/status.json`).
- Gemini support for tactical trade notes (optional with `GEMINI_API_KEY`).
- Optional OpenAI fallback (`OPENAI_API_KEY`) when Gemini is unavailable.
- Optional Gemini decision influence with capped confidence adjustments and veto mode.
- Futuristic desktop UI with startup animation and live feedback.
- Risk engine: per-trade risk sizing, max daily drawdown guard, and max open trades.
- Execution preflight checks for spread, volume bounds/step, stops distance, and free margin.
- Additional safety guards: max trades/day, consecutive-loss cooldown, spread/slippage limits.
- Backtest mode, live mode, UI mode, and live-with-UI mode.

## Setup
1. Install Python 3.10+.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in HFM credentials.
4. If Atlas cannot find your terminal automatically, set `HFM_TERMINAL_PATH` to your `terminal64.exe` location.
5. Ensure MetaTrader 5 terminal (x64) is installed and logged in to the same HFM account.
6. Set `EXECUTE_TRADES=true` in `.env` when you are ready for real order execution (keep `false` for paper mode).
7. Optional LLM controls in `.env`:
   `OPENAI_API_KEY`, `OPENAI_MODEL` (fallback provider)
8. Optional Gemini influence controls in `.env`:
   `GEMINI_INFLUENCE_ENABLED`, `GEMINI_MAX_CONFIDENCE_BOOST`, `GEMINI_MAX_CONFIDENCE_CUT`.
9. Optional execution safety controls in `.env`:
   `MAX_TRADES_PER_DAY`, `MAX_CONSECUTIVE_LOSSES`, `COOLDOWN_MINUTES_AFTER_LOSS_STREAK`, `MAX_SPREAD_POINTS`, `MAX_SLIPPAGE_POINTS`, `DAILY_PROFIT_TARGET_ZAR`.
10. Optional learning/quality controls in `.env`:
   `EVENT_MAX_LINES`, `HIGHER_TIMEFRAME_CONFIRM`, `MIN_CONFIDENCE_TREND`, `MIN_CONFIDENCE_RANGE`, `MIN_CONFIDENCE_HIGH_VOLATILITY`.
11. Optional intelligence controls in `.env`:
   `EXPLORATION_EPSILON`, `META_MODEL_ENABLED`, `META_MODEL_CONFIDENCE_CAP`,
   `PROMOTION_MIN_SAMPLES`, `PROMOTION_MIN_ACCURACY_LIFT`, `PROMOTION_MIN_BRIER_IMPROVEMENT`,
   `BACKTEST_SPREAD_COST_BPS`, `BACKTEST_SLIPPAGE_COST_BPS`.

## Run
- Backtest:
  `python -m atlas_ai.main backtest`
- Live:
  `python -m atlas_ai.main live --poll-seconds 60`
- UI only (reads saved telemetry):
  `python -m atlas_ai.main ui`
- Live + UI desktop:
  `python -m atlas_ai.main live-ui --poll-seconds 60`
- Atlas wake launcher:
  `python -m atlas_ai.atlas_full_launcher --json`
- Atlas no-console full stack launcher:
  `atlas_fullstack_startup.pyw`
- Atlas sleep shutdown:
  `python -m atlas_ai.atlas_full_shutdown --json`
- Atlas wake phrase trigger:
  `python -m atlas_ai.atlas_wake_listener --trigger "Atlas wake" --json`
- Atlas sleep phrase trigger:
  `python -m atlas_ai.atlas_wake_listener --trigger "Atlas sleep" --json`
- Atlas audio listener:
  `python -m atlas_ai.atlas_audio_listener`
- Atlas audio watchdog no-console entrypoint:
  `atlas_audio_watchdog.py`
- Experiment report (meta-model training + experiment log):
  `python -m atlas_ai.main experiment-report`

## Test
- Preferred:
  `python -m pytest -q`
- Bare `pytest -q` is also supported from the repo root; pytest is configured to add `.` to `PYTHONPATH` so `atlas_ai` imports resolve even if a global `pytest.exe` is used.

## Notes
- This system is research-grade and should be paper-traded first.
- No strategy guarantees profit; market conditions can change quickly.
- Keep strict risk controls when moving to real capital.
- Windows startup should use the no-console entrypoints only.
- Legacy `.bat` and `.cmd` wrappers are compatibility shims and should not be used for background autostart.

## Historical learning
- Atlas AI can train from previous months/years before live operation.
- Configure in .env:
  - ENABLE_HISTORICAL_TRAINING=true|false
  - HISTORICAL_TRAIN_MONTHS=24
  - HISTORICAL_TRAIN_MAX_BARS=30000
  - HISTORICAL_RETRAIN_HOURS=6

