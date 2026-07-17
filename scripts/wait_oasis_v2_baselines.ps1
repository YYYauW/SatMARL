$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\satenv\python.exe"
$RunDir = Join-Path $Root "runs\oasis_v2"
$EvalSummary = Join-Path $RunDir "eval\summary.json"

while (-not (Test-Path -LiteralPath $EvalSummary)) {
  Start-Sleep -Seconds 30
}

$BaselineDir = Join-Path $RunDir "baselines"
New-Item -ItemType Directory -Force -Path $BaselineDir | Out-Null
& $Python (Join-Path $PSScriptRoot "evaluate_heuristics.py") `
  --satellites 16 --tasks 768 --planes 4 --max-steps 240 `
  --candidate-k 24 --neighbor-k 6 `
  --step-duration-seconds 30 --point-observation-seconds 5 `
  --fov-deg 45 --max-off-nadir-deg 45 `
  --min-task-window 40 --max-task-window 160 `
  --planning-lookahead-steps 30 --task-layout global_random `
  --episodes 50 --seed 9001 `
  --output (Join-Path $BaselineDir "summary.json")
