$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$QueueDir = Join-Path $Root "runs\aaai_multiseed_queue"
New-Item -ItemType Directory -Force -Path $QueueDir | Out-Null

$process = Start-Process `
  -FilePath "powershell.exe" `
  -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $PSScriptRoot "queue_aaai_multiseed.ps1")
  ) `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $QueueDir "queue.stdout.log") `
  -RedirectStandardError (Join-Path $QueueDir "queue.stderr.log") `
  -PassThru

Set-Content -LiteralPath (Join-Path $QueueDir "queue.pid") -Value $process.Id
Write-Output "Started AAAI multi-seed queue with PID $($process.Id)"
