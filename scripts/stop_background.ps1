$root = Split-Path -Parent $PSScriptRoot
$pidFiles = @(
    Join-Path $root "runs\realistic\train.pid",
    Join-Path $root "runs\realistic\web.pid",
    Join-Path $root "runs\current\train.pid",
    Join-Path $root "runs\current\web.pid"
)

foreach ($pidFile in $pidFiles) {
    if (-not (Test-Path -LiteralPath $pidFile)) {
        continue
    }
    $processId = Get-Content -LiteralPath $pidFile | Select-Object -First 1
    if (-not $processId) {
        continue
    }
    $process = Get-Process -Id ([int]$processId) -ErrorAction SilentlyContinue
    if ($process) {
        Stop-Process -Id $process.Id -Force
        Write-Host "Stopped process $($process.Id) from $pidFile"
    }
}
