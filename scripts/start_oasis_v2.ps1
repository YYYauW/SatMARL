$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root "runs\oasis_v2"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$stdout = Join-Path $RunDir "train.stdout.log"
$stderr = Join-Path $RunDir "train.stderr.log"
$process = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "run_oasis_v2.ps1")
  ) `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Set-Content -LiteralPath (Join-Path $RunDir "train.pid") -Value $process.Id
Write-Output "Started OASIS v2 with PID $($process.Id)"
