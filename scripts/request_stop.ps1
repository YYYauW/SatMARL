$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "runs\realistic"
$stopFile = Join-Path $runDir "STOP"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
Set-Content -LiteralPath $stopFile -Value "stop requested $(Get-Date -Format o)"
Write-Host "Stop requested. Training will save a checkpoint and exit after the current episode."
Write-Host "STOP file: $stopFile"
