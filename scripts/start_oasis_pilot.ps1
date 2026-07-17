$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root "runs\oasis_pilot"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
$Process = Start-Process powershell.exe `
  -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "$PSScriptRoot\run_pilot_and_baselines.ps1"
  ) `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput "$RunDir\pilot.stdout.log" `
  -RedirectStandardError "$RunDir\pilot.stderr.log" `
  -PassThru
$Process.PriorityClass = "BelowNormal"
$Process.Id | Set-Content -Encoding ascii "$RunDir\pilot.pid"
Write-Output $Process.Id
