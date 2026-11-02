# ==============================================================================
# UrbanGraph Intelligence -- Localhost Test Runner Script
# Platform: Windows 11 / PowerShell 5.1+ / Python 3.11+
# ==============================================================================
[CmdletBinding()]
param (
    [string]$Partition = "smoke",
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Localhost Test Suite Execution" -ForegroundColor Cyan
Write-Host "  Partition: $Partition" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check Python Venv
if (-not $VenvPath) {
    if (Test-Path "C:\urbangraph-cleanroom-final\Scripts\python.exe") {
        $VenvPath = "C:\urbangraph-cleanroom-final"
    } elseif (Test-Path (Join-Path $WorkspaceRoot ".venv-local\Scripts\python.exe")) {
        $VenvPath = Join-Path $WorkspaceRoot ".venv-local"
    } else {
        Write-Error "Virtual environment not found. Please run install_local.ps1 first."
        exit 1
    }
}
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

$env:PYTHONPATH = (Join-Path $WorkspaceRoot "src")

switch ($Partition.ToLower()) {
    "smoke" {
        Write-Host "[*] Executing fast smoke tests (distributed auth & simulation contracts)..." -ForegroundColor Yellow
        & $VenvPython -m pytest tests/security/test_distributed_token_store.py -v
    }
    "auth" {
        Write-Host "[*] Executing real multi-process distributed auth test..." -ForegroundColor Yellow
        & $VenvPython scripts/test_real_distributed_auth.py
    }
    "packaged" {
        Write-Host "[*] Executing full clean-room packaged acceptance test..." -ForegroundColor Yellow
        & $VenvPython scripts/verify_e2e_packaged_acceptance.py
    }
    "all" {
        Write-Host "[*] Executing all local security, contract, and regression suites..." -ForegroundColor Yellow
        & $VenvPython -m pytest tests/security/test_distributed_token_store.py -v
        & $VenvPython scripts/test_real_distributed_auth.py
        & $VenvPython scripts/verify_e2e_packaged_acceptance.py
    }
    default {
        Write-Host "[*] Executing custom test target: $Partition..." -ForegroundColor Yellow
        & $VenvPython -m pytest $Partition -v
    }
}

if ($LASTEXITCODE -eq 0) {
    Write-Host "======================================================================" -ForegroundColor Green
    Write-Host "  TEST SUITE COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host "======================================================================" -ForegroundColor Red
    Write-Host "  TEST SUITE FAILED WITH EXIT CODE $LASTEXITCODE" -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    exit $LASTEXITCODE
}
