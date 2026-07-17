$root = Split-Path -Parent $PSScriptRoot
$python = "D:\anaconda\envs\satenv\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "satenv python not found: $python"
}

$runDir = Join-Path $root "runs\realistic"
New-Item -ItemType Directory -Force -Path (Join-Path $runDir "eval") | Out-Null

$stopFile = Join-Path $runDir "STOP"
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}

& $python (Join-Path $root "scripts\train_dqn.py") `
    --satellites 256 `
    --tasks 1024 `
    --planes 8 `
    --max-steps 128 `
    --candidate-k 16 `
    --neighbor-k 6 `
    --episodes 20000 `
    --seed 23 `
    --learning-starts 2048 `
    --batch-size 256 `
    --buffer-size 120000 `
    --store-agents 128 `
    --hidden-dim 256 `
    --hidden-layers 2 `
    --activation relu `
    --layer-norm `
    --optimizer adamw `
    --weight-decay 0.01 `
    --lr 0.0003 `
    --gamma 0.97 `
    --loss-type huber `
    --huber-delta 1.0 `
    --double-dqn `
    --reward-scale 1.0 `
    --reward-clip 10.0 `
    --epsilon-start 0.8 `
    --epsilon-end 0.05 `
    --epsilon-decay-steps 100000 `
    --target-update-steps 800 `
    --target-tau 1.0 `
    --train-frequency 1 `
    --gradient-steps 1 `
    --grad-clip 10.0 `
    --checkpoint-every 10 `
    --metrics-every 1 `
    --run-dir (Join-Path $root "runs\realistic") `
    --resume

$trainExit = $LASTEXITCODE
if ($trainExit -ne 0) {
    Write-Error "Training exited with code $trainExit; evaluation skipped."
    exit $trainExit
}

$metricsPath = Join-Path $runDir "metrics.json"
$metrics = Get-Content -Raw -LiteralPath $metricsPath | ConvertFrom-Json
if ($metrics.status -ne "complete") {
    Write-Host "Training status is '$($metrics.status)'; evaluation skipped."
    exit 0
}

& $python (Join-Path $root "scripts\evaluate_policy.py") `
    --checkpoint (Join-Path $runDir "checkpoints\latest.pt") `
    --output (Join-Path $runDir "eval\rollout.json") `
    --seed 9001

exit $LASTEXITCODE
