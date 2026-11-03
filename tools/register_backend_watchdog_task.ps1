param(
    [string]$TaskName = "Atlas Backend Heartbeat Watchdog",
    [string]$RepoRoot = "C:\Users\User\Desktop\Atlas AI"
)

$ErrorActionPreference = "Continue"

$reportPath = Join-Path $RepoRoot "storage\scheduler_admin_probe.json"
$watchdogBat = Join-Path $RepoRoot "atlas_backend_watchdog.bat"

function Capture-Text {
    param([scriptblock]$Block)
    try {
        (& $Block 2>&1 | Out-String).Trim()
    } catch {
        ($_ | Out-String).Trim()
    }
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$isAdminRole = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$whoamiGroups = Capture-Text { whoami /groups }
$netSession = Capture-Text { net session }

$trivialTaskName = "Atlas Permission Probe Elevated"
$trivialCreate = Capture-Text {
    $action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c exit 0"
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    Register-ScheduledTask -TaskName $trivialTaskName -Action $action -Trigger $trigger -Force
}

$trivialQuery = Capture-Text {
    Get-ScheduledTask -TaskName $trivialTaskName -ErrorAction Stop |
        Select-Object TaskName, TaskPath, State |
        ConvertTo-Json -Compress
}

$actualCreate = Capture-Text {
    $startBoundary = (Get-Date).AddMinutes(1).ToString("yyyy-MM-ddTHH:mm:ss")
    $userSid = $identity.User.Value
    $author = [Security.SecurityElement]::Escape($identity.Name)
    $repo = [Security.SecurityElement]::Escape($RepoRoot)
    $bat = [Security.SecurityElement]::Escape("/c `"$watchdogBat`"")
    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>$author</Author>
    <Description>Starts the Atlas backend heartbeat watchdog every 5 minutes. The watchdog lock prevents duplicate supervisors.</Description>
  </RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <StartBoundary>$startBoundary</StartBoundary>
      <Enabled>true</Enabled>
      <Repetition>
        <Interval>PT5M</Interval>
        <StopAtDurationEnd>false</StopAtDurationEnd>
      </Repetition>
    </TimeTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$userSid</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>cmd.exe</Command>
      <Arguments>$bat</Arguments>
      <WorkingDirectory>$repo</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@
    Register-ScheduledTask -TaskName $TaskName -Xml $xml -Force
}

$actualQuery = Capture-Text {
    Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop |
        Select-Object TaskName, TaskPath, State |
        ConvertTo-Json -Compress
}

$schtasksQuery = Capture-Text {
    schtasks /Query /TN $TaskName /V /FO LIST
}

$report = [ordered]@{
    checked_at = (Get-Date).ToUniversalTime().ToString("o")
    user = $identity.Name
    is_in_administrator_role = $isAdminRole
    net_session_output = $netSession
    whoami_groups = $whoamiGroups
    trivial_task_create_output = $trivialCreate
    trivial_task_query_output = $trivialQuery
    actual_task_create_output = $actualCreate
    actual_task_query_output = $actualQuery
    schtasks_query_output = $schtasksQuery
    watchdog_bat = $watchdogBat
}

$report | ConvertTo-Json -Depth 5 | Set-Content -Path $reportPath -Encoding UTF8
Write-Host "Wrote scheduler report to $reportPath"
