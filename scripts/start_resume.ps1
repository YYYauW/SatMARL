$root = Split-Path -Parent $PSScriptRoot
$python = "D:\anaconda\envs\satenv\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "satenv python not found: $python"
}

$runDir = Join-Path $root "runs\realistic"
New-Item -ItemType Directory -Force -Path (Join-Path $runDir "checkpoints") | Out-Null

$stopFile = Join-Path $runDir "STOP"
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}

$existingTrain = $null
$trainPidFile = Join-Path $runDir "train.pid"
if (Test-Path -LiteralPath $trainPidFile) {
    $existingTrain = Get-Process -Id ([int](Get-Content -LiteralPath $trainPidFile | Select-Object -First 1)) -ErrorAction SilentlyContinue
}

if (-not $existingTrain) {
    $train = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $root "scripts\run_train_then_eval.ps1")
        ) `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runDir "train_stdout.log") `
        -RedirectStandardError (Join-Path $runDir "train_stderr.log") `
        -PassThru
    Set-Content -LiteralPath $trainPidFile -Value $train.Id
    Write-Host "Started train-then-eval pipeline PID $($train.Id)"
} else {
    Write-Host "Training already running PID $($existingTrain.Id)"
}

$existingWeb = $null
$webPidFile = Join-Path $runDir "web.pid"
if (Test-Path -LiteralPath $webPidFile) {
    $existingWeb = Get-Process -Id ([int](Get-Content -LiteralPath $webPidFile | Select-Object -First 1)) -ErrorAction SilentlyContinue
}

$urlFile = Join-Path $runDir "web_url.txt"
if (-not $existingWeb) {
    $port = 8765
    $used = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($used) {
        $port = 8766
    }
    $serverArgs = @("-m", "http.server", [string]$port)
    $server = Start-Process -FilePath $python `
        -ArgumentList $serverArgs `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runDir "web_stdout.log") `
        -RedirectStandardError (Join-Path $runDir "web_stderr.log") `
        -PassThru
    Set-Content -LiteralPath $webPidFile -Value $server.Id
    Set-Content -LiteralPath $urlFile -Value "http://localhost:$port/web/index.html"
    Write-Host "Started web board PID $($server.Id)"
} else {
    Write-Host "Web board already running PID $($existingWeb.Id)"
}

if (Test-Path -LiteralPath $urlFile) {
    Write-Host (Get-Content -LiteralPath $urlFile | Select-Object -First 1)
}
