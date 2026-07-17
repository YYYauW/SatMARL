$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = "D:\anaconda\envs\satenv\python.exe"
$RunDir = Join-Path $Root "runs\aaai_v4\oasis"
$PipelinePath = Join-Path $Root "runs\aaai_v4\pipeline.json"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "satenv python not found: $Python"
}

New-Item -ItemType Directory -Force -Path (Join-Path $RunDir "eval") | Out-Null

function Write-PipelineState([string]$Status, [string]$Stage, [string]$Message) {
    $payload = [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        updated_at = [DateTimeOffset]::Now.ToUnixTimeSeconds()
        run_dir = $RunDir
        training_seed = 4401
        evaluation_seed_start = 9501
        evaluation_episodes = 50
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $PipelinePath -Encoding UTF8
}

try {
    Write-PipelineState "running" "training" "Semantic opportunity balancing with an annealed realistic curriculum"
    & $Python (Join-Path $Root "scripts\train_oasis.py") `
        --satellites 16 --tasks 768 --planes 4 --max-steps 240 `
        --candidate-k 24 --neighbor-k 6 `
        --step-duration-seconds 30 --point-observation-seconds 5 `
        --fov-deg 45 --max-off-nadir-deg 45 `
        --min-observation-elevation-deg 3 --optical-min-sun-elevation-deg 8 `
        --min-task-window 40 --max-task-window 160 --planning-lookahead-steps 30 `
        --task-layout mixed --curriculum-visible-fraction 0.65 `
        --curriculum-end-fraction 0 --target-opportunity-rate 0.06 `
        --curriculum-feedback-gain 1.0 --adaptive-curriculum `
        --scenario-seed-cycle 0 --episodes 500 --seed 4401 `
        --hidden-dim 128 --hidden-layers 2 --activation relu --layer-norm `
        --lr 0.0003 --weight-decay 0.01 --gamma 0.97 --gae-lambda 0.95 `
        --opportunity-gae --opportunity-balancing --semantic-opportunity-balancing `
        --min-decision-samples 512 --max-buffered-episodes 4 `
        --clip-coef 0.2 --value-clip-coef 0.2 --entropy-coef 0.02 `
        --value-coef 0.5 --update-epochs 4 --minibatch-size 256 `
        --normalize-advantages --reward-scale 1 --reward-clip 10 --grad-clip 10 `
        --checkpoint-every 10 --metrics-every 1 --device cuda `
        --run-dir $RunDir --resume
    if ($LASTEXITCODE -ne 0) { throw "OASIS v4 training failed with exit code $LASTEXITCODE" }

    $Metrics = Get-Content -LiteralPath (Join-Path $RunDir "metrics.json") -Raw | ConvertFrom-Json
    if ($Metrics.status -ne "complete") {
        Write-PipelineState $Metrics.status "training" "Training did not reach complete status; evaluation skipped"
        exit 0
    }

    Write-PipelineState "running" "evaluation" "Evaluating v4 on unseen global-random seeds 9501-9550"
    & $Python (Join-Path $Root "scripts\evaluate_marl.py") `
        --checkpoint (Join-Path $RunDir "checkpoints\latest.pt") `
        --output (Join-Path $RunDir "eval\rollout.json") `
        --seed 9501 --eval-episodes 50 --eval-task-layout global_random --device cuda
    if ($LASTEXITCODE -ne 0) { throw "OASIS v4 evaluation failed with exit code $LASTEXITCODE" }

    foreach ($Reference in @("oasis", "ps_dqn")) {
        Write-PipelineState "running" "reference_evaluation" "Evaluating AAAI v3 $Reference on the same unseen seeds"
        $ReferenceDir = Join-Path $Root "runs\aaai_v4\reference_$Reference"
        New-Item -ItemType Directory -Force -Path (Join-Path $ReferenceDir "eval") | Out-Null
        & $Python (Join-Path $Root "scripts\evaluate_marl.py") `
            --checkpoint (Join-Path $Root "runs\aaai_v3\$Reference\checkpoints\latest.pt") `
            --output (Join-Path $ReferenceDir "eval\rollout.json") `
            --seed 9501 --eval-episodes 50 --eval-task-layout global_random --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Reference evaluation for $Reference failed with exit code $LASTEXITCODE" }
    }

    Write-PipelineState "complete" "complete" "Training and paired independent evaluation complete"
}
catch {
    Write-PipelineState "failed" "failed" $_.Exception.Message
    throw
}
