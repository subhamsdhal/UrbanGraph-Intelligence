# ==============================================================================
# UrbanGraph Intelligence -- Localhost Stop Script
# Platform: Windows 11 / PowerShell 5.1+ / Python 3.11+
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$Force
)

$ErrorActionPreference = "SilentlyContinue"

if (Test-Path (Join-Path $PSScriptRoot "urbangraph.pid")) {
    $WorkspaceRoot = $PSScriptRoot
} elseif (Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) "urbangraph.pid")) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
} else {
    $WorkspaceRoot = $PSScriptRoot
}

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Stopping Localhost Production Server" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$PidFile = Join-Path $WorkspaceRoot "urbangraph.pid"
if (Test-Path $PidFile) {
    $TargetPid = (Get-Content $PidFile -ErrorAction SilentlyContinue).Trim()
    if ($TargetPid) {
        Write-Host "[*] Read PID $TargetPid from $PidFile..." -ForegroundColor Yellow
        $TargetProcess = Get-Process -Id $TargetPid -ErrorAction SilentlyContinue
        if ($TargetProcess) {
            Write-Host "[*] Terminating process $TargetPid ($($TargetProcess.ProcessName))..." -ForegroundColor Yellow
            Stop-Process -Id $TargetPid -Force
            Start-Sleep -Milliseconds 800
            if (Get-Process -Id $TargetPid -ErrorAction SilentlyContinue) {
                Write-Warning "Process $TargetPid did not terminate immediately. Retrying..."
                Stop-Process -Id $TargetPid -Force
            }
            Write-Host "[+] Successfully terminated process PID $TargetPid." -ForegroundColor Green
        }
    }
    Remove-Item -Path $PidFile -Force -ErrorAction SilentlyContinue
}

# Ensure port 8000 listeners are also freed (workers, detached processes)
$ActiveConns = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
if ($ActiveConns) {
    foreach ($conn in $ActiveConns) {
        $ProcId = $conn.OwningProcess
        if ($ProcId -and $ProcId -ne 0) {
            $Proc = Get-Process -Id $ProcId -ErrorAction SilentlyContinue
            if ($Proc) {
                Write-Host "[*] Terminating listener process on port 8000 (PID: $ProcId - $($Proc.ProcessName))..." -ForegroundColor Yellow
                Stop-Process -Id $ProcId -Force
            }
        }
    }
    Start-Sleep -Milliseconds 800
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  UrbanGraph server stopped cleanly." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
exit 0
