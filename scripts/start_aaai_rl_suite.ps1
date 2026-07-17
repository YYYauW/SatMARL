$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root "runs\aaai_v3"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$stdout = Join-Path $RunDir "suite.stdout.log"
$stderr = Join-Path $RunDir "suite.stderr.log"
$process = Start-Process `
  -FilePath "D:\anaconda\envs\satenv\python.exe" `
  -ArgumentList @(
    (Join-Path $PSScriptRoot "run_algorithm_suite.py"),
    "--base-dir", $RunDir,
    "--satellites", "16",
    "--tasks", "768",
    "--planes", "4",
    "--max-steps", "240",
    "--candidate-k", "24",
    "--neighbor-k", "6",
    "--step-duration-seconds", "30",
    "--point-observation-seconds", "5",
    "--fov-deg", "45",
    "--max-off-nadir-deg", "45",
    "--min-task-window", "40",
    "--max-task-window", "160",
    "--planning-lookahead-steps", "30",
    "--task-layout", "global_random",
    "--curriculum-visible-fraction", "0",
    "--scenario-seed-cycle", "0",
    "--episodes", "500",
    "--seed", "401",
    "--eval-seed", "9001",
    "--eval-episodes", "50",
    "--eval-task-layout", "global_random",
    "--hidden-dim", "128",
    "--checkpoint-every", "10",
    "--device", "cuda"
  ) `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -PassThru

Set-Content -LiteralPath (Join-Path $RunDir "suite.pid") -Value $process.Id
Write-Output "Started AAAI RL suite with PID $($process.Id)"
