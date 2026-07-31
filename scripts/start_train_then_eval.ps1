$root = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $root "runs\realistic"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$stopFile = Join-Path $runDir "STOP"
if (Test-Path -LiteralPath $stopFile) {
    Remove-Item -LiteralPath $stopFile -Force
}

$trainPidFile = Join-Path $runDir "train.pid"
$existingTrain = $null
if (Test-Path -LiteralPath $trainPidFile) {
    $pidText = Get-Content -LiteralPath $trainPidFile | Select-Object -First 1
    if ($pidText) {
        $existingTrain = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    }
}

if (-not $existingTrain) {
    $pipeline = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @(
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $root "scripts\run_train_then_eval.ps1")
        ) `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $runDir "train_stdout.log") `
        -RedirectStandardError (Join-Path $runDir "train_stderr.log") `
        -PassThru
    Set-Content -LiteralPath $trainPidFile -Value $pipeline.Id
    Write-Host "Started train-then-eval pipeline PID $($pipeline.Id)"
} else {
    Write-Host "Training pipeline already running PID $($existingTrain.Id)"
}

$python = "D:\anaconda\envs\satenv\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "satenv python not found: $python"
}

$webPidFile = Join-Path $runDir "web.pid"
$existingWeb = $null
if (Test-Path -LiteralPath $webPidFile) {
    $pidText = Get-Content -LiteralPath $webPidFile | Select-Object -First 1
    if ($pidText) {
        $existingWeb = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
    }
}

$urlFile = Join-Path $runDir "web_url.txt"
if (-not $existingWeb) {
    $port = 8765
    $used = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($used) {
        $port = 8766
    }
    $server = Start-Process -FilePath $python `
        -ArgumentList @("-m", "http.server", [string]$port) `
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

Write-Host "Training board: $(Get-Content -LiteralPath $urlFile | Select-Object -First 1)"
$evalUrl = (Get-Content -LiteralPath $urlFile | Select-Object -First 1) -replace "/web/index.html", "/web/eval.html"
Write-Host "Evaluation replay: $evalUrl"
