$root = Split-Path -Parent $PSScriptRoot
$python = "D:\anaconda\envs\satenv\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "satenv python not found: $python"
}

$baseDir = Join-Path $root "runs\comparison_dense"
New-Item -ItemType Directory -Force -Path $baseDir | Out-Null
$pidFile = Join-Path $baseDir "suite.pid"
$existing = $null
if (Test-Path -LiteralPath $pidFile) {
    $existing = Get-Process -Id ([int](Get-Content -LiteralPath $pidFile | Select-Object -First 1)) -ErrorAction SilentlyContinue
}

if (-not $existing) {
    $process = Start-Process -FilePath $python `
        -ArgumentList @(
            (Join-Path $root "scripts\run_algorithm_suite.py"),
            "--base-dir", $baseDir,
            "--satellites", "64",
            "--tasks", "1024",
            "--planes", "8",
            "--max-steps", "96",
            "--candidate-k", "24",
            "--neighbor-k", "8",
            "--task-layout", "mixed",
            "--curriculum-visible-fraction", "0.75",
            "--curriculum-ground-track-jitter-deg", "4.0",
            "--curriculum-time-jitter-steps", "4",
            "--curriculum-payload-match-probability", "0.85",
            "--min-observation-elevation-deg", "2.0",
            "--max-off-nadir-deg", "45.0",
            "--optical-min-sun-elevation-deg", "4.0",
            "--min-task-window", "28",
            "--max-task-window", "72",
            "--planning-lookahead-steps", "24",
            "--scenario-seed-cycle", "64",
            "--episodes", "2000",
            "--seed", "131",
            "--eval-seed", "19001",
            "--eval-episodes", "30",
            "--eval-task-layout", "global_random",
            "--hidden-dim", "384",
            "--hidden-layers", "2",
            "--device", "cuda",
            "--resume"
        ) `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $baseDir "suite_stdout.log") `
        -RedirectStandardError (Join-Path $baseDir "suite_stderr.log") `
        -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id
    Write-Host "Started dense algorithm suite PID $($process.Id)"
} else {
    Write-Host "Dense algorithm suite already running PID $($existing.Id)"
}

$keepAwakePidFile = Join-Path $baseDir "keep_awake.pid"
$keepAwake = $null
if (Test-Path -LiteralPath $keepAwakePidFile) {
    $keepAwake = Get-Process -Id ([int](Get-Content -LiteralPath $keepAwakePidFile | Select-Object -First 1)) -ErrorAction SilentlyContinue
}
if (-not $keepAwake) {
    $keepAwake = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $PSScriptRoot "keep_awake_while_training.ps1"),
            "-PidFile", $pidFile
        ) `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath $keepAwakePidFile -Value $keepAwake.Id
}

Write-Host "Dense comparison board: http://localhost:8765/web/compare.html?run=comparison_dense"
