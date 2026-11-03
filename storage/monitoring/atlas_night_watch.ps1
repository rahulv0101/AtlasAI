$ErrorActionPreference = "SilentlyContinue"

$Project = "C:\Users\User\Desktop\Atlas AI"
$Python = Join-Path $Project ".venv\Scripts\python.exe"
$Log = Join-Path $Project "storage\monitoring\atlas_night_watch.log"
$Summary = Join-Path $Project "storage\monitoring\atlas_night_watch_summary.jsonl"
$EndAt = (Get-Date).AddHours(10)

function Write-NightLog {
    param([string]$Message)
    $ts = (Get-Date).ToString("o")
    Add-Content -Path $Log -Value "$ts $Message"
}

function Write-Summary {
    param([hashtable]$Payload)
    $Payload["ts"] = (Get-Date).ToString("o")
    Add-Content -Path $Summary -Value ($Payload | ConvertTo-Json -Compress -Depth 8)
}

Write-NightLog "night_watch_started end_at=$($EndAt.ToString("o"))"

while ((Get-Date) -lt $EndAt) {
    try {
        $monitor = Get-CimInstance Win32_Process | Where-Object {
            ($_.Name -like "python*") -and ($_.CommandLine -like "*atlas_monitor.py*")
        }
        if (-not $monitor) {
            Start-Process -FilePath $Python -ArgumentList @("storage\monitoring\atlas_monitor.py", "60", "5") -WorkingDirectory $Project -WindowStyle Hidden
            Write-NightLog "monitor_restart_requested reason=missing"
            Start-Sleep -Seconds 5
        }

        $statusPath = Join-Path $Project "storage\status.json"
        if (Test-Path $statusPath) {
            $statusFile = Get-Item $statusPath
            $ageSeconds = [math]::Round(((Get-Date) - $statusFile.LastWriteTime).TotalSeconds, 1)
            $status = Get-Content -Raw $statusPath | ConvertFrom-Json
            Write-Summary @{
                event = "night_watch_snapshot"
                status_age_seconds = $ageSeconds
                action = [string]$status.action
                blocker_class = [string]$status.blocker_class
                message = [string]$status.message
                mt5_connected = [bool]$status.system_health.broker_connectivity_status.connected
                live_execute = [bool]$status.system_health.mode_truth.runtime_execute_trades
                safety_lock = [bool]$status.system_health.safety_layer.safety_lock_active
                online_learning_applied = [bool]$status.online_learning_influence.applied
            }
            if ($ageSeconds -gt 300) {
                Write-NightLog "status_stale age_seconds=$ageSeconds restart_requested"
                & $Python -m atlas_ai.atlas_full_launcher --json *> $null
            }
        } else {
            Write-NightLog "status_missing restart_requested"
            & $Python -m atlas_ai.atlas_full_launcher --json *> $null
        }
    } catch {
        Write-NightLog "night_watch_error $($_.Exception.Message)"
    }

    Start-Sleep -Seconds 300
}

Write-NightLog "night_watch_finished"
