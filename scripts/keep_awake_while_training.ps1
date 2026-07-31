param(
    [Parameter(Mandatory = $true)]
    [string]$PidFile
)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class TrainingPowerState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
"@

$continuous = [Convert]::ToUInt32("80000000", 16)
$systemRequired = [uint32]0x00000001
try {
    while (Test-Path -LiteralPath $PidFile) {
        $suitePid = [int](Get-Content -LiteralPath $PidFile | Select-Object -First 1)
        if (-not (Get-Process -Id $suitePid -ErrorAction SilentlyContinue)) {
            break
        }
        [void][TrainingPowerState]::SetThreadExecutionState(
            $continuous -bor $systemRequired
        )
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][TrainingPowerState]::SetThreadExecutionState($continuous)
}
