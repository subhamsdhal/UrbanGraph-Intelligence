param(
    [string]$PythonPath = '',
    [string]$PackagePath = '',
    [ValidateRange(1, 65535)][int]$ApiPort = 8000,
    [ValidateRange(1, 65535)][int]$FrontendPort = 5173
)
$ErrorActionPreference = 'Stop'
if ($ApiPort -eq $FrontendPort) { throw 'API and frontend ports must differ.' }

function Assert-LoopbackPortFree([int]$Port) {
    $listener = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, $Port)
    try {
        $listener.Start()
    }
    catch {
        throw "Loopback port $Port is unavailable; no existing process was stopped."
    }
    finally {
        $listener.Stop()
    }
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe' }
if (-not $PackagePath) { $PackagePath = Join-Path $projectRoot 'tests\fixtures\eg006' }
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) { throw 'Python environment is missing; see EG-006 setup instructions.' }
if (-not (Test-Path -LiteralPath $PackagePath -PathType Container)) { throw 'Approved synthetic package is missing.' }
$env:ENERGRAPH_DEMO_PACKAGE = (Resolve-Path -LiteralPath $PackagePath).Path
$env:ENERGRAPH_DEMO_FRONTEND_ORIGIN = "http://127.0.0.1:$FrontendPort"
$env:ENERGRAPH_DEMO_API_PORT = [string]$ApiPort
$env:PYTHONDONTWRITEBYTECODE = '1'
$frontRoot = Join-Path $projectRoot 'frontend'
if (-not (Test-Path -LiteralPath (Join-Path $frontRoot 'node_modules') -PathType Container)) { throw 'Frontend dependencies are missing; run npm ci in frontend.' }
$nodeCommand = Get-Command node -ErrorAction Stop
Assert-LoopbackPortFree $ApiPort
Assert-LoopbackPortFree $FrontendPort
$pythonArgs = @('-B', '-m', 'uvicorn', 'energraph.demo.api:create_app', '--factory', '--host', '127.0.0.1', '--port', [string]$ApiPort)
$api = $null
$ui = $null
try {
    $api = Start-Process -FilePath $PythonPath -ArgumentList $pythonArgs -WorkingDirectory $projectRoot -PassThru -WindowStyle Hidden
    $ui = Start-Process -FilePath $nodeCommand.Source -ArgumentList @('node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', [string]$FrontendPort, '--strictPort') -WorkingDirectory $frontRoot -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 2
    if ($api.HasExited -or $ui.HasExited) { throw 'A demo process exited during startup; check package validation, dependencies, and ports.' }
    Write-Host "EnerGraph synthetic demo: http://127.0.0.1:$FrontendPort"
    Write-Host "Local API documentation: http://127.0.0.1:$ApiPort/docs"
    Write-Host 'Press Enter to stop the two processes started by this script.'
    [void](Read-Host)
}
finally {
    foreach ($process in @($ui, $api)) {
        if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
    }
}
