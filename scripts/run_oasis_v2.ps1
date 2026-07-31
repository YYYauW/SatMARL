param(
  [int]$Episodes = 300,
  [string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\satenv\python.exe"
$RunDir = Join-Path $Root "runs\oasis_v2"

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

& $Python (Join-Path $PSScriptRoot "train_oasis.py") `
  --satellites 16 `
  --tasks 768 `
  --planes 4 `
  --max-steps 240 `
  --candidate-k 24 `
  --neighbor-k 6 `
  --episodes $Episodes `
  --seed 131 `
  --scenario-seed-cycle 0 `
  --task-layout mixed `
  --curriculum-visible-fraction 0.65 `
  --curriculum-end-fraction 0.15 `
  --target-opportunity-rate 0.12 `
  --step-duration-seconds 30 `
  --point-observation-seconds 5 `
  --fov-deg 45 `
  --max-off-nadir-deg 45 `
  --min-task-window 40 `
  --max-task-window 160 `
  --planning-lookahead-steps 30 `
  --min-decision-samples 512 `
  --max-buffered-episodes 4 `
  --hidden-dim 128 `
  --hidden-layers 2 `
  --minibatch-size 256 `
  --update-epochs 4 `
  --checkpoint-every 10 `
  --metrics-every 1 `
  --device $Device `
  --run-dir $RunDir

if ($LASTEXITCODE -ne 0) {
  throw "OASIS v2 training failed with exit code $LASTEXITCODE"
}

$EvalDir = Join-Path $RunDir "eval"
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
& $Python (Join-Path $PSScriptRoot "evaluate_marl.py") `
  --checkpoint (Join-Path $RunDir "checkpoints\latest.pt") `
  --output (Join-Path $EvalDir "rollout.json") `
  --seed 9001 `
  --eval-episodes 50 `
  --eval-task-layout global_random `
  --device $Device

if ($LASTEXITCODE -ne 0) {
  throw "OASIS v2 held-out evaluation failed with exit code $LASTEXITCODE"
}

$BaselineDir = Join-Path $RunDir "baselines"
New-Item -ItemType Directory -Force -Path $BaselineDir | Out-Null
& $Python (Join-Path $PSScriptRoot "evaluate_heuristics.py") `
  --satellites 16 `
  --tasks 768 `
  --planes 4 `
  --max-steps 240 `
  --candidate-k 24 `
  --neighbor-k 6 `
  --step-duration-seconds 30 `
  --point-observation-seconds 5 `
  --fov-deg 45 `
  --max-off-nadir-deg 45 `
  --min-task-window 40 `
  --max-task-window 160 `
  --planning-lookahead-steps 30 `
  --task-layout global_random `
  --episodes 50 `
  --seed 9001 `
  --output (Join-Path $BaselineDir "summary.json")

if ($LASTEXITCODE -ne 0) {
  throw "OASIS v2 heuristic evaluation failed with exit code $LASTEXITCODE"
}
