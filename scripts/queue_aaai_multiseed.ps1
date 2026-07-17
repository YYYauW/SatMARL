$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\satenv\python.exe"
$PrimarySuite = Join-Path $Root "runs\aaai_v3\suite.json"

while ($true) {
  if (Test-Path -LiteralPath $PrimarySuite) {
    $state = Get-Content -LiteralPath $PrimarySuite -Raw | ConvertFrom-Json
    if ($state.status -eq "complete") { break }
    if ($state.status -in @("failed", "stopped")) {
      throw "Primary AAAI suite ended with status '$($state.status)'"
    }
  }
  Start-Sleep -Seconds 30
}

foreach ($Seed in @(501, 601, 701, 801)) {
  $RunDir = Join-Path $Root "runs\aaai_v3_seed_$Seed"
  & $Python (Join-Path $PSScriptRoot "run_algorithm_suite.py") `
    --base-dir $RunDir `
    --satellites 16 --tasks 768 --planes 4 --max-steps 240 `
    --candidate-k 24 --neighbor-k 6 `
    --step-duration-seconds 30 --point-observation-seconds 5 `
    --fov-deg 45 --max-off-nadir-deg 45 `
    --min-task-window 40 --max-task-window 160 `
    --planning-lookahead-steps 30 `
    --task-layout global_random --curriculum-visible-fraction 0 `
    --scenario-seed-cycle 0 --episodes 500 --seed $Seed `
    --eval-seed 9001 --eval-episodes 50 `
    --eval-task-layout global_random `
    --hidden-dim 128 --checkpoint-every 10 --device cuda
  if ($LASTEXITCODE -ne 0) {
    throw "AAAI suite for seed $Seed failed with exit code $LASTEXITCODE"
  }
}
