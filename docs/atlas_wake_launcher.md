# Atlas Wake Launcher

This launcher starts the existing Atlas paper stack without VS Code.

## Audio Setup

Install the local offline speech model:

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_vosk_setup --json
```

Then start the always-on microphone listener:

```powershell
.venv\Scripts\pythonw.exe atlas_audio_watchdog.py
```

Legacy compatibility wrappers:

- `atlas_audio_setup.bat`
- `atlas_audio_listen.bat`

Preferred background entrypoint:

- `atlas_audio_watchdog.py`

By default the listener works in voice-only mode for:

- `Atlas wake`
- `Atlas sleep`

Optional clap gate:

- Set `ATLAS_AUDIO_CLAP_ENABLED=true` or put `"audio_clap_enabled": true` in `atlas_wake_config.json`
- Then use double clap, followed by `Atlas wake`

## Direct Python Run

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_full_launcher --json
```

No-console full stack startup:

```text
C:\Users\User\Desktop\Atlas AI\atlas_fullstack_startup.pyw
```

Direct shutdown:

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_full_shutdown --json
```

## Wake Trigger

One-shot trigger:

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_wake_listener --trigger "Atlas wake" --json
```

One-shot sleep trigger:

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_wake_listener --trigger "Atlas sleep" --json
```

Interactive stdin listener:

```powershell
.venv\Scripts\python.exe -m atlas_ai.atlas_wake_listener --stdin --json
```

## Batch Wrappers

- `atlas_wake.bat`
- `atlas_sleep.bat`
- `atlas_wake_listen.bat`
- `atlas_audio_listen.bat`
- `atlas_audio_setup.bat`

These wrappers are kept for compatibility. For no-console startup, prefer the `.pyw` and `pythonw.exe` paths.

## Optional Config

Copy `atlas_wake_config.json.example` to `atlas_wake_config.json` if you want to override launcher paths or commands in one place.

## Task Scheduler

For no-console full stack startup, point Windows Task Scheduler or shortcuts at:

```text
C:\Users\User\Desktop\Atlas AI\atlas_fullstack_startup.pyw
```

For audio-listener-only startup, point Windows Task Scheduler at:

```text
C:\Users\User\Desktop\Atlas AI\atlas_audio_watchdog.py
```

For backend and UI-server heartbeat supervision, point Windows Task Scheduler at:

```text
C:\Users\User\Desktop\Atlas AI\atlas_backend_watchdog.bat
```

The watchdog checks two independent heartbeat files:

- Backend: `storage/heartbeat.json`
- UI server: `storage/ui_heartbeat.json`

If the backend heartbeat becomes `stale` or `unknown`, it writes
`storage/backend_staleness_alert.json`, appends `logs/backend_heartbeat_watchdog.jsonl`,
and restarts the M1 live backend through `python -m atlas_ai.main live --poll-seconds 60
--mode-override live`.

If the UI-server heartbeat becomes `stale` or `unknown`, or if the ui-server process is
missing, it writes `storage/ui_staleness_alert.json`, appends the same watchdog log with
schema `atlas_ui_heartbeat_watchdog_v1`, and restarts the UI server through
`python -m atlas_ai.main ui-server`.

The backend and UI checks run independently inside the watchdog. A failure in one check is
logged under that component and does not prevent the other check from running. This is
operational supervision only; it does not change trading, risk, strategy, or ExecutionGate
policy.

Interim backend staleness threshold:

- Effective date: 2026-09-22
- Setting: `BACKEND_HEARTBEAT_STALE_AFTER_SECONDS=450`
- Basis: recent cycle-start intervals had a 252.9-second P95, while retained outage
  gaps were measured in hours. The previous 180-second threshold could restart a
  backend that was still completing a slow cycle.
- Scope: operational watchdog timing only. This is an interim mitigation while cycle
  latency is investigated; it does not change trading, risk, strategy, or execution
  policy.
- Verification: after restarting the resident watchdog so it loaded the new setting,
  one backend worker completed five consecutive cycle intervals (`25084` through
  `25089`) without a watchdog restart. Watchdog records reported
  `staleness_threshold_seconds=450` and `action=healthy` throughout the run.

Verified backend recovery test:

- Date: 2026-08-31
- Test: deliberately terminated the M1 live backend process while leaving the watchdog and
  scheduled task in place.
- Result: the watchdog detected the stale heartbeat and relaunched the backend without a
  manual restart.
- Downtime: approximately 2 minutes 33 seconds from backend termination to fresh heartbeat.
- Scheduler evidence: the repeating scheduled task fired successfully with `LastTaskResult`
  `0` after the task action was fixed to run through `cmd.exe /c` with the quoted batch
  path.
- Lock evidence: duplicate scheduled launches were recorded as `duplicate_watchdog_exit`,
  confirming overlapping watchdog instances were prevented.
- Coverage note: the deliberate kill created a brief monitoring/trading gap during the
  test window; no unintended trade was observed in live status during the drill.

Verified UI-server recovery test:

- Date: 2026-09-01
- Test: deliberately terminated only the ui-server parent/worker processes while leaving
  the M1 live backend and watchdog running.
- Killed UI processes: parent PID `17840`, worker PID `4496`.
- Result: the watchdog detected `process_missing` and relaunched the UI server without a
  manual restart.
- Recovered UI processes: parent PID `6240`, worker PID `7160`.
- Downtime: approximately 46 seconds from ui-server termination to `/api/status`
  responding with HTTP `200`.
- Watchdog evidence: `logs/backend_heartbeat_watchdog.jsonl` recorded
  `schema=atlas_ui_heartbeat_watchdog_v1`, `action=restart_ui_server`, and
  `restart_result.started_ui_server.pid=6240`.
- UI heartbeat evidence: `storage/ui_heartbeat.json` advanced to worker PID `7160` with
  `cycle_status=running`.
- Backend isolation evidence: backend parent/worker PIDs remained `16576` / `8188`, and
  `storage/heartbeat.json` remained fresh during the UI kill-and-recover drill.

### HTTP responsiveness and worker selection (2026-10-13)

The supervisor now checks both `/api/health` and `/api/status/compact`, including
response shape and elapsed time. A fresh `ui_heartbeat.json` is no longer sufficient
to classify a hung HTTP/status path as healthy. One failed HTTP sample raises an
alert; two consecutive failures trigger recovery of an otherwise live UI. Existing
missing-process/stale-heartbeat recovery remains in place.

Backend and UI supervision now run on separate resident loops, so a delayed backend
probe cannot hold up UI probing. Restart-capable `--once` also obeys the watchdog lock.
The repeating five-minute scheduled task still starts the resident supervisor; each
resident check runs roughly every 60 seconds plus its own check duration.

Process inspection uses the already-installed psutil library without spawning
PowerShell, with bounded Windows subprocess fallbacks if the library is unavailable.
Workers are distinguished from virtualenv launchers using parent-child relationships.
Multiple independent workers are reported for operator review. Restart is withheld
if a matching process remains; a UI replacement also requires the port to be free.

Verification: isolated tests cover hung HTTP with fresh heartbeat, recovery debounce,
independent checks, failed-stop duplicate protection, and worker selection. Healthy
HTTP probes were verified against the deployed UI. This update did **not** repeat a
live kill-and-recover drill; the older drills above are historical evidence only.
See `docs/atlas_runtime_audit_20261013.md` for measured timings and remaining issues.
