$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\start_resume.ps1"
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew
Register-ScheduledTask `
    -TaskName "SatMarlResumeTraining" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Resume satellite MARL training board from D:\Condadata\sat_marl_env" `
    -Force | Out-Null
Write-Host "Installed scheduled task: SatMarlResumeTraining"

