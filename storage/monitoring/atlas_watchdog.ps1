param(
    [int]$IntervalSeconds = 60,
    [int]$StaleMinutes = 5
)

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Project = Split-Path -Parent $Root
$Log = Join-Path $Root 'monitoring\atlas_watchdog.log'
$Insights = Join-Path $Root 'monitoring\atlas_monitor_insights.jsonl'

function Write-WatchLog($Message) {
    $ts = (Get-Date).ToString('o')
    Add-Content -Path $Log -Value "$ts $Message" -Encoding UTF8
}

function Write-Insight($Object) {
    $Object | Add-Member -NotePropertyName ts -NotePropertyValue (Get-Date).ToString('o') -Force
    Add-Content -Path $Insights -Value ($Object | ConvertTo-Json -Depth 8 -Compress) -Encoding UTF8
}

function Get-LiveProcesses {
    Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'atlas_ai\.main live' }
}

function Start-Backend {
    $py = Join-Path $Project '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { $py = 'python.exe' }
    Start-Process -FilePath $py -ArgumentList @('-m','atlas_ai.main','live','--poll-seconds','60','--mode-override','live') -WorkingDirectory $Project -WindowStyle Hidden
    Write-WatchLog 'backend_restart_requested reason=not_running_or_stale'
}

function Get-Mt5Snapshot {
    $py = Join-Path $Project '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) { return @{ error = 'venv_python_missing' } }
    $code = @'
import json
import MetaTrader5 as mt5
from atlas_ai.config.settings import load_settings
s = load_settings()
ok = mt5.initialize()
acc = mt5.account_info() if ok else None
pos = mt5.positions_get(symbol=s.symbol) if ok else None
pos = pos or []
print(json.dumps({
    "connected": bool(ok and acc),
    "balance": float(getattr(acc, "balance", 0.0)) if acc else None,
    "equity": float(getattr(acc, "equity", 0.0)) if acc else None,
    "profit": float(getattr(acc, "profit", 0.0)) if acc else None,
    "positions": len(pos),
    "floating_profit": round(sum(float(getattr(p, "profit", 0.0)) for p in pos), 2),
}))
'@
    try {
        $raw = $code | & $py -
        return $raw | ConvertFrom-Json
    } catch {
        return @{ error = $_.Exception.Message }
    }
}

function Get-ConfusionSignals($Status) {
    $signals = New-Object System.Collections.Generic.List[string]
    if ($Status.market_regime.regime -eq 'unknown') { $signals.Add('market_regime_unknown') }
    if ($Status.context_snapshot.event_risk_state -eq 'unknown') { $signals.Add('event_risk_unknown') }
    if ($Status.context_snapshot.news_risk_classification -eq 'unknown') { $signals.Add('news_risk_unknown') }
    if ($Status.context_snapshot.dxy_alignment -eq 'unknown') { $signals.Add('dxy_alignment_unknown') }
    if ($Status.context_snapshot.us10y_alignment -eq 'unknown') { $signals.Add('us10y_alignment_unknown') }
    if ($Status.context_snapshot.vix_alignment -eq 'unknown') { $signals.Add('vix_alignment_unknown') }
    if ($Status.context_snapshot.oil_alignment -eq 'unknown') { $signals.Add('oil_alignment_unknown') }
    if ($Status.decision_analysis.macro_context.dxy.availability -eq 'unavailable') { $signals.Add('dxy_data_unavailable') }
    if ($Status.decision_analysis.macro_context.us10y.availability -eq 'unavailable') { $signals.Add('us10y_data_unavailable') }
    if ($Status.decision_analysis.macro_context.vix.availability -eq 'unavailable') { $signals.Add('vix_data_unavailable') }
    if ($Status.decision_analysis.macro_context.oil.availability -eq 'unavailable') { $signals.Add('oil_data_unavailable') }
    if ($Status.setup_family -eq 'no_trade') { $signals.Add('setup_family_no_trade') }
    if ($Status.chart_features.momentum_state -in @('mixed', 'weak')) { $signals.Add("momentum_$($Status.chart_features.momentum_state)") }
    if ($Status.chart_features.location_state -eq 'mid_range') { $signals.Add('entry_location_mid_range') }
    return @($signals)
}

Write-WatchLog "monitor_started interval=$IntervalSeconds stale_minutes=$StaleMinutes"
Write-Insight @{ event = 'monitor_started'; interval_seconds = $IntervalSeconds; stale_minutes = $StaleMinutes }

while ($true) {
    try {
        $live = @(Get-LiveProcesses)
        if ($live.Count -eq 0) {
            Write-Insight @{ event = 'backend_down'; action_taken = 'restart_backend' }
            Start-Backend
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $status = $null
        try { $status = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/status' -TimeoutSec 8 } catch {}
        if ($null -eq $status) {
            Write-WatchLog 'status_api_unavailable backend_process_present=true'
            Write-Insight @{ event = 'status_api_unavailable'; backend_process_count = $live.Count }
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $stale = $false
        $ageMin = $null
        try {
            $updated = [datetime]$status.updated_at
            $ageMin = ((Get-Date) - $updated).TotalMinutes
            $stale = $ageMin -gt $StaleMinutes
        } catch {}

        if ($stale) {
            Write-WatchLog "status_stale age_minutes=$([Math]::Round($ageMin,2)) action=restart_backend"
            Write-Insight @{ event = 'status_stale'; age_minutes = [Math]::Round($ageMin, 2); action_taken = 'restart_backend' }
            foreach ($p in $live) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
            Start-Sleep -Seconds 2
            Start-Backend
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $mt5 = Get-Mt5Snapshot
        $confusion = Get-ConfusionSignals $status
        $blockers = @($status.decision_analysis.hard_blockers)
        $topBlockers = @($status.pre_trade_memo.top_blockers | ForEach-Object { "$($_.name):$($_.actual)/$($_.target)" })

        Write-WatchLog "ok action=$($status.action) blocker_class=$($status.blocker_class) primary='$($status.primary_blocker)' weekly=$($status.risk.weekly_loss_pct) daily=$($status.risk.daily_loss_pct) positions=$($status.risk.open_positions) confusion=$($confusion -join ',')"
        Write-Insight @{
            event = 'monitor_tick'
            action = $status.action
            message = $status.message
            blocker_class = $status.blocker_class
            primary_blocker = $status.primary_blocker
            hard_blockers = $blockers
            top_blockers = $topBlockers
            confusion_signals = $confusion
            risk = @{
                balance = $status.risk.balance
                equity = $status.risk.equity
                weekly_loss_pct = $status.risk.weekly_loss_pct
                daily_loss_pct = $status.risk.daily_loss_pct
                trades_today = $status.risk.trades_today
                open_positions = $status.risk.open_positions
            }
            market = @{
                session = $status.session
                regime = $status.market_regime.regime
                trend_structure = $status.chart_features.trend_structure
                momentum = $status.chart_features.momentum_state
                location = $status.chart_features.location_state
                spread_points = $status.risk.spread_points
            }
            mt5 = $mt5
        }
    } catch {
        Write-WatchLog "monitor_error $($_.Exception.Message)"
        Write-Insight @{ event = 'monitor_error'; error = $_.Exception.Message }
    }
    Start-Sleep -Seconds $IntervalSeconds
}
