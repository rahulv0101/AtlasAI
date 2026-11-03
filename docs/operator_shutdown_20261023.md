# Operator-Requested Shutdown

On 2026-10-23 (operator/client date), the operator requested all Atlas processes
and automatic restart launchers remain stopped until an explicit restart request.

Stopped: M1 backend, UI server, backend/UI heartbeat watchdog, audio watchdog.
Disabled scheduled tasks (both previously enabled):
- Atlas Backend Heartbeat Watchdog
- Atlas Audio Listener

Verification: neither task enabled; no matching Atlas runtime or launcher process
remaining; TCP port 8765 not listening; no Atlas Windows startup command found.
Unrelated applications and MT5 were left alone. Trading parameters were not changed.

Do not restart Atlas or re-enable these tasks until the operator requests it.
AI Core repair work was interrupted by this shutdown request; do not treat that
work as fully tested or deployed merely because files were edited.
