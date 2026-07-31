$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "runs\dashboard"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Process = Start-Process "D:\anaconda\envs\satenv\python.exe" `
  -ArgumentList @("$PSScriptRoot\serve_dashboard.py", "--port", "8766") `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput "$LogDir\server.stdout.log" `
  -RedirectStandardError "$LogDir\server.stderr.log" `
  -PassThru
$Process.Id | Set-Content -Encoding ascii "$LogDir\server.pid"
Write-Output $Process.Id
