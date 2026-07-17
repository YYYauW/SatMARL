param(
  [int]$Episodes = 100,
  [ValidateSet("cpu", "cuda")][string]$Device = "cpu"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\satenv\python.exe"

& $Python "$PSScriptRoot\train_oasis.py" `
  --satellites 16 `
  --tasks 192 `
  --planes 4 `
  --max-steps 48 `
  --candidate-k 16 `
  --neighbor-k 6 `
  --episodes $Episodes `
  --hidden-dim 128 `
  --hidden-layers 2 `
  --minibatch-size 256 `
  --update-epochs 2 `
  --scenario-seed-cycle 32 `
  --target-opportunity-rate 0.06 `
  --device $Device `
  --run-dir "$Root\runs\oasis_pilot"

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python "$PSScriptRoot\evaluate_heuristics.py" `
  --satellites 16 `
  --tasks 192 `
  --planes 4 `
  --max-steps 48 `
  --candidate-k 16 `
  --neighbor-k 6 `
  --episodes 10 `
  --seed 24001 `
  --output "$Root\runs\baselines\summary.json"

exit $LASTEXITCODE
