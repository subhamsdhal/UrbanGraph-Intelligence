[CmdletBinding()]
param (
    [int]$Port = 8000,
    [string]$BindHost = "127.0.0.1",
    [string]$VenvPath = "",
    [switch]$NoBrowser,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

if (Test-Path (Join-Path $PSScriptRoot "release\urbangraph-0.2.0rc3-py3-none-any.whl")) {
    $WorkspaceRoot = $PSScriptRoot
} elseif (Test-Path (Join-Path $PSScriptRoot "dist\urbangraph-0.2.0rc3-py3-none-any.whl")) {
    $WorkspaceRoot = $PSScriptRoot
} elseif (Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) "release\urbangraph-0.2.0rc3-py3-none-any.whl")) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
} elseif (Test-Path (Join-Path (Split-Path -Parent $PSScriptRoot) "dist\urbangraph-0.2.0rc3-py3-none-any.whl")) {
    $WorkspaceRoot = Split-Path -Parent $PSScriptRoot
} else {
    $WorkspaceRoot = $PSScriptRoot
}

$TargetUrl = "http://${BindHost}:${Port}"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Starting Localhost Production Server" -ForegroundColor Cyan
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
Write-Host "--> Using Python runtime: $VenvPython" -ForegroundColor Green

# 2. Check Port Availability
$PortActive = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
if ($PortActive) {
    try {
        $Health = Invoke-RestMethod -Uri "$TargetUrl/healthz" -TimeoutSec 2 -ErrorAction Stop
        if ($Health.status -eq "ok") {
            Write-Host "--> UrbanGraph Intelligence is already running on $TargetUrl" -ForegroundColor Yellow
            if (-not $NoBrowser) {
                Start-Process $TargetUrl
            }
            exit 0
        }
    } catch {
        Write-Error "Port $Port is already in use by another process. Please free the port or specify another port."
        exit 1
    }
}

# 3. Setup Directories and Environment
$LogsDir = Join-Path $WorkspaceRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}
$PidFile = Join-Path $WorkspaceRoot "urbangraph.pid"
$LogFileOut = Join-Path $LogsDir "urbangraph_server.stdout.log"
$LogFileErr = Join-Path $LogsDir "urbangraph_server.stderr.log"
$TokenDb = Join-Path $LogsDir "urbangraph_tokens.db"

$env:URBANGRAPH_HOST = $BindHost
$env:URBANGRAPH_PORT = "$Port"
$env:URBANGRAPH_WORKERS = "1"
$env:URBANGRAPH_TOKEN_STORE_PATH = $TokenDb

if (-not $env:URBANGRAPH_ADMIN_TOKEN) {
    $env:URBANGRAPH_ADMIN_TOKEN = "ug_admin_staging_sec_7f9a2b8e4c1d6f30a9e8b7c6d5e4f3a2"
}
if (-not $env:URBANGRAPH_WORKER_TOKEN) {
    $env:URBANGRAPH_WORKER_TOKEN = "ug_worker_staging_sec_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
}
if (-not $env:URBANGRAPH_READER_TOKEN) {
    $env:URBANGRAPH_READER_TOKEN = "ug_reader_staging_sec_9f8e7d6c5b4a3f2e1d0c9b8a7f6e5d4c"
}

$FixturesCandidates = @(
    (Join-Path $WorkspaceRoot "src\urbangraph\fixtures\eg006"),
    (Join-Path $WorkspaceRoot "fixtures\eg006"),
    (Join-Path $WorkspaceRoot "URBANGRAPH_LOCALHOST_FINAL\fixtures\eg006"),
    (Join-Path (Split-Path -Parent $WorkspaceRoot) "fixtures\eg006")
)
foreach ($fc in $FixturesCandidates) {
    if (Test-Path (Join-Path $fc "manifest.json")) {
        $env:URBANGRAPH_DEMO_FIXTURE_ROOT = $fc
        Write-Host "--> Configured Fixture Root: $fc" -ForegroundColor Green
        break
    }
}

# 4. Launch Server
Write-Host "--> Launching server on $TargetUrl..." -ForegroundColor Yellow
if ($Foreground) {
    & $VenvPython -m urbangraph.production_server
    exit $LASTEXITCODE
} else {
    $EnvCmd = "cmd.exe /c `"set URBANGRAPH_HOST=$BindHost&& set URBANGRAPH_PORT=$Port&& set URBANGRAPH_WORKERS=1&& set URBANGRAPH_ADMIN_TOKEN=$($env:URBANGRAPH_ADMIN_TOKEN)&& set URBANGRAPH_WORKER_TOKEN=$($env:URBANGRAPH_WORKER_TOKEN)&& set URBANGRAPH_READER_TOKEN=$($env:URBANGRAPH_READER_TOKEN)&& set URBANGRAPH_TOKEN_STORE_PATH=$TokenDb&& `"$VenvPython`" -m urbangraph.production_server`""
    $CimProc = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
        CommandLine = $EnvCmd
        CurrentDirectory = $WorkspaceRoot
    } -ErrorAction SilentlyContinue

    if ($CimProc -and $CimProc.ReturnValue -eq 0 -and $CimProc.ProcessId) {
        $ProcId = $CimProc.ProcessId
    } else {
        $proc = Start-Process -FilePath $VenvPython -ArgumentList @("-m", "urbangraph.production_server") -WorkingDirectory $WorkspaceRoot -RedirectStandardOutput $LogFileOut -RedirectStandardError $LogFileErr -WindowStyle Hidden -PassThru
        $ProcId = $proc.Id
    }
    $ProcId | Out-File -FilePath $PidFile -Encoding ascii -Force
    Write-Host "--> Server process launched with PID $ProcId" -ForegroundColor Green
}

# 5. Wait for Readiness Probe
Write-Host "--> Polling /readyz probe for readiness (timeout 25s)..." -ForegroundColor Yellow
$ReadyUrl = "$TargetUrl/readyz"
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$IsReady = $false

while ($Stopwatch.Elapsed.TotalSeconds -lt 25) {
    Start-Sleep -Milliseconds 800
    try {
        $Response = Invoke-RestMethod -Uri $ReadyUrl -TimeoutSec 2 -ErrorAction Stop
        if ($Response.ready -eq $true -or $Response.status -eq "ready" -or $Response.status -eq "ok") {
            $IsReady = $true
            break
        }
    } catch {
        # Waiting for server to become ready
    }
}

if (-not $IsReady) {
    Write-Error "Server failed to report ready status within 25 seconds. Check logs at: $LogFileErr"
    exit 1
}

$Elapsed = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 2)
Write-Host "--> Server reported READY in ${Elapsed}s" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  UrbanGraph Intelligence is ONLINE on Windows Localhost" -ForegroundColor Green
Write-Host "  URL: $TargetUrl" -ForegroundColor Cyan
Write-Host "  API Docs: $TargetUrl/docs" -ForegroundColor Cyan
Write-Host "  Prometheus Metrics: $TargetUrl/metrics" -ForegroundColor Cyan
Write-Host "  Server Logs: $LogFileOut" -ForegroundColor DarkGray
Write-Host "  PID file: $PidFile" -ForegroundColor DarkGray
Write-Host "======================================================================" -ForegroundColor Green

# 6. Open Browser
if (-not $NoBrowser) {
    Write-Host "--> Opening default web browser..." -ForegroundColor Yellow
    Start-Process $TargetUrl
}

exit 0
