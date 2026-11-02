# ==============================================================================
# UrbanGraph Intelligence -- Localhost Installation Script
# Platform: Windows 11 / PowerShell 5.1+ / Python 3.11+
# ==============================================================================
[CmdletBinding()]
param (
    [string]$VenvPath = "",
    [string]$WheelPath = ""
)

$ErrorActionPreference = "Stop"
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Localhost Cleanroom Installer" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Locate Python
$PythonCmd = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $PythonCmd) {
    Write-Error "Python executable not found on PATH. Please install Python 3.11+."
    exit 1
}

$PyVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "[+] Detected System Python: $PyVersion ($($PythonCmd.Source))" -ForegroundColor Green

$PyMajor = [int](& python -c "import sys; print(sys.version_info.major)")
$PyMinor = [int](& python -c "import sys; print(sys.version_info.minor)")
if ($PyMajor -lt 3 -or ($PyMajor -eq 3 -and $PyMinor -lt 11)) {
    Write-Error "Python 3.11+ is required. Found $PyVersion."
    exit 1
}

# 2. Determine Workspace and Virtual Environment Path
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

if (-not $VenvPath) {
    if (Test-Path "C:\urbangraph-cleanroom-final\Scripts\python.exe") {
        $VenvPath = "C:\urbangraph-cleanroom-final"
        Write-Host "[+] Using verified cleanroom environment: $VenvPath" -ForegroundColor Cyan
    } else {
        $VenvPath = Join-Path $WorkspaceRoot ".venv-local"
    }
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[*] Creating fresh virtual environment at: $VenvPath" -ForegroundColor Yellow
    & python -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment at $VenvPath"
        exit 1
    }
}
Write-Host "[+] Virtual environment ready: $VenvPath" -ForegroundColor Green

# 3. Locate Wheel
if (-not $WheelPath) {
    $CandidateWheels = @(
        (Join-Path $WorkspaceRoot "dist\urbangraph-0.2.0rc3-py3-none-any.whl"),
        (Join-Path $WorkspaceRoot "release\urbangraph-0.2.0rc3-py3-none-any.whl"),
        (Join-Path $WorkspaceRoot "URBANGRAPH_LOCALHOST_FINAL\release\urbangraph-0.2.0rc3-py3-none-any.whl"),
        (Join-Path (Split-Path -Parent $WorkspaceRoot) "dist\urbangraph-0.2.0rc3-py3-none-any.whl"),
        (Join-Path (Split-Path -Parent $WorkspaceRoot) "URBANGRAPH_LOCALHOST_FINAL\release\urbangraph-0.2.0rc3-py3-none-any.whl")
    )
    foreach ($cand in $CandidateWheels) {
        if (Test-Path $cand) {
            $WheelPath = $cand
            break
        }
    }
}

if (-not $WheelPath -or -not (Test-Path $WheelPath)) {
    Write-Error "Could not locate urbangraph-0.2.0rc3 wheel artifact. Please build wheel first."
    exit 1
}

$WheelItem = Get-Item $WheelPath
$WheelHash = (Get-FileHash -Path $WheelPath -Algorithm SHA256).Hash
Write-Host "[+] Installing Wheel: $($WheelItem.Name) ($($WheelItem.Length) bytes)" -ForegroundColor Green
Write-Host "    SHA-256: $WheelHash" -ForegroundColor DarkGray

# Authoritative hash validation
$ExpectedWheelHash = "B0CFC7297BBC6C5F76B790F15BD5CCA40873E48C7D34089CC32460FB360B3C97"
if ($WheelHash -ne $ExpectedWheelHash) {
    Write-Error "Wheel SHA-256 mismatch! Expected $ExpectedWheelHash, got $WheelHash"
    exit 1
}
Write-Host "[+] Wheel integrity verified (matches official release SHA-256)" -ForegroundColor Green

# 4. Install dependencies and wheel into venv
Write-Host "[*] Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet

$HasDeps = $false
try {
    & $VenvPython -c "import fastapi, torch, scipy" 2>$null
    if ($LASTEXITCODE -eq 0) { $HasDeps = $true }
} catch {
    $HasDeps = $false
}

if (-not $HasDeps) {
    Write-Host "[*] Dependencies missing in target environment. Installing runtime dependencies..." -ForegroundColor Yellow
    & $VenvPython -m pip install "$WheelPath[demo,neural,optimization,data]" --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[*] Extras installation encountered error; installing pinned core dependencies..." -ForegroundColor Yellow
        & $VenvPython -m pip install fastapi uvicorn pydantic scipy "torch>=2.0" "pyarrow<18" slowapi prometheus_client networkx pandas shapely python-multipart --quiet
    }
}

Write-Host "[*] Installing wheel into virtual environment..." -ForegroundColor Yellow
& $VenvPython -m pip install --force-reinstall --no-deps $WheelPath --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install wheel $WheelPath into virtual environment."
    exit 1
}

Write-Host "[*] Verifying pip package consistency..." -ForegroundColor Yellow
& $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip check reported broken requirements."
    exit 1
}

# 5. Verify Installation
Write-Host "[*] Verifying module imports and CLI entrypoints..." -ForegroundColor Yellow
$VerifyCode = @"
import urbangraph
import energraph.demo.api
import urbangraph.production_server
print(f'SUCCESS: urbangraph imported from {urbangraph.__file__}')
"@
& $VenvPython -c $VerifyCode
if ($LASTEXITCODE -ne 0) {
    Write-Error "Verification failed: unable to import installed modules."
    exit 1
}

$ServerExe = Join-Path $VenvPath "Scripts\urbangraph-server.exe"
if (Test-Path $ServerExe) {
    Write-Host "[*] Verifying CLI binary: $ServerExe --check-only" -ForegroundColor Yellow
    & $ServerExe --check-only
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CLI check-only preflight verification failed."
        exit 1
    }
}

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  INSTALLATION COMPLETE: UrbanGraph Intelligence is ready on Localhost" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
exit 0
