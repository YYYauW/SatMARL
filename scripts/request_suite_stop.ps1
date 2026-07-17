$root = Split-Path -Parent $PSScriptRoot
$baseDir = Join-Path $root "runs\comparison"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null
Set-Content -LiteralPath (Join-Path $baseDir "STOP") -Value "stop requested $(Get-Date -Format o)"
foreach ($algorithm in @("ps_dqn", "ippo", "mappo", "qmix")) {
    $runDir = Join-Path $baseDir $algorithm
    if (Test-Path -LiteralPath $runDir) {
        Set-Content -LiteralPath (Join-Path $runDir "STOP") -Value "stop requested $(Get-Date -Format o)"
    }
}
Write-Host "Algorithm suite stop requested. Active trainer will checkpoint after its current episode."
