$root = Split-Path -Parent $PSScriptRoot
$python = "D:\anaconda\envs\satenv\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "satenv python not found: $python"
}

$baseDir = Join-Path $root "runs\comparison"
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
            "--tasks", "256",
            "--planes", "8",
            "--max-steps", "96",
            "--episodes", "2000",
            "--seed", "31",
            "--eval-seed", "9001",
            "--eval-episodes", "20",
            "--device", "cuda",
            "--resume"
        ) `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $baseDir "suite_stdout.log") `
        -RedirectStandardError (Join-Path $baseDir "suite_stderr.log") `
        -PassThru
    Set-Content -LiteralPath $pidFile -Value $process.Id
    Write-Host "Started algorithm suite PID $($process.Id)"
} else {
    Write-Host "Algorithm suite already running PID $($existing.Id)"
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

Write-Host "Comparison board: http://localhost:8765/web/compare.html"
