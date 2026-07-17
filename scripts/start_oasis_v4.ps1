$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $Root "runs\aaai_v4"
$PidPath = Join-Path $RunRoot "pipeline.pid"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Existing = $null
if (Test-Path -LiteralPath $PidPath) {
    $PidText = Get-Content -LiteralPath $PidPath | Select-Object -First 1
    if ($PidText) { $Existing = Get-Process -Id ([int]$PidText) -ErrorAction SilentlyContinue }
}

if ($Existing) {
    Write-Host "OASIS v4 pipeline already running, PID $($Existing.Id)"
    exit 0
}

$Process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $Root "scripts\run_oasis_v4.ps1")) `
    -WorkingDirectory $Root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $RunRoot "pipeline.stdout.log") `
    -RedirectStandardError (Join-Path $RunRoot "pipeline.stderr.log") `
    -PassThru
Set-Content -LiteralPath $PidPath -Value $Process.Id
Write-Host "Started OASIS v4 train-and-test pipeline, PID $($Process.Id)"
