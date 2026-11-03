param(
    [int]$PollSeconds = 60,
    [string]$ComparisonRoot = ".\\storage_compare_runs",
    [string]$RunId = "",
    [string]$PaperPreset = "conservative_filter_rollout_v1",
    [switch]$Resume,
    [switch]$AllowMarketClosed
)

$ErrorActionPreference = "Stop"

$argsList = @("-m", "atlas_ai.main", "paper-compare", "--poll-seconds", "$PollSeconds", "--comparison-root", $ComparisonRoot, "--paper-preset", $PaperPreset)
if ($RunId) {
    $argsList += @("--comparison-run-id", $RunId)
}
if ($Resume) {
    $argsList += "--resume"
}
if ($AllowMarketClosed) {
    $argsList += "--allow-market-closed"
}

& ".\\.venv\\Scripts\\python.exe" @argsList
