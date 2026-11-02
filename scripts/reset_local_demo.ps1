# ==============================================================================
# UrbanGraph Intelligence -- Localhost Demo Reset Script
# Platform: Windows 11 / PowerShell 5.1+ / Python 3.11+
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$Force
)

$ErrorActionPreference = "SilentlyContinue"
$WorkspaceRoot = $PSScriptRoot

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Reset Localhost Demo State" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Stop running instance if any
$StopScript = Join-Path $WorkspaceRoot "stop_local.ps1"
if (Test-Path $StopScript) {
    & $StopScript
}

# 2. Reset token database
$TokenDb = Join-Path $WorkspaceRoot "logs\urbangraph_tokens.db"
$TokenDbWal = Join-Path $WorkspaceRoot "logs\urbangraph_tokens.db-wal"
$TokenDbShm = Join-Path $WorkspaceRoot "logs\urbangraph_tokens.db-shm"

if (Test-Path $TokenDb) {
    Remove-Item -Path $TokenDb, $TokenDbWal, $TokenDbShm -Force -ErrorAction SilentlyContinue
    Write-Host "[+] Reset centralized token repository: $TokenDb" -ForegroundColor Green
}

# 3. Clean server logs
$ServerLog = Join-Path $WorkspaceRoot "logs\urbangraph_server.log"
if (Test-Path $ServerLog) {
    Remove-Item -Path $ServerLog -Force -ErrorAction SilentlyContinue
    Write-Host "[+] Cleared server log: $ServerLog" -ForegroundColor Green
}

# 4. Clean ephemeral pytest and pycache
Get-ChildItem -Path $WorkspaceRoot -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $WorkspaceRoot -Filter ".pytest_cache" -Recurse -Directory -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[+] Cleaned ephemeral caches (__pycache__, .pytest_cache)." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  Demo state has been safely reset. Ready for clean start." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
exit 0
