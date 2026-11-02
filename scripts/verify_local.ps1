# ==============================================================================
# UrbanGraph Intelligence -- Localhost Verification & Health Audit
# Platform: Windows 11 / PowerShell 5.1+ / Python 3.11+
# ==============================================================================
[CmdletBinding()]
param (
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$Token = "ug_worker_staging_sec_1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d"
)

$ErrorActionPreference = "Continue"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  UrbanGraph Intelligence -- Localhost Verification Suite" -ForegroundColor Cyan
Write-Host "  Target Endpoint: $BaseUrl" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$Results = @()
$AllPassed = $true
$AuthHeaders = @{
    "Authorization" = "Bearer $Token"
    "Accept" = "application/json"
}

function Run-Step {
    param (
        [string]$StepId,
        [string]$Name,
        [scriptblock]$Action
    )
    $Sw = [System.Diagnostics.Stopwatch]::StartNew()
    $Status = "PASS"
    $Detail = ""
    try {
        $Detail = & $Action
        if (-not $Detail) { $Detail = "Assertion satisfied" }
    } catch {
        $Status = "FAIL"
        $Detail = $_.Exception.Message
        $script:AllPassed = $false
    }
    $Sw.Stop()
    $DurationMs = [math]::Round($Sw.Elapsed.TotalMilliseconds, 1)

    $Color = if ($Status -eq "PASS") { "Green" } else { "Red" }
    Write-Host ("[{0}] {1,-38} | {2,6} | {3,7} ms | {4}" -f $StepId, $Name, $Status, $DurationMs, $Detail) -ForegroundColor $Color

    $script:Results += [PSCustomObject]@{
        Step = $StepId
        Name = $Name
        Status = $Status
        DurationMs = $DurationMs
        Detail = $Detail
    }
}

# 1. Healthz
Run-Step "V-01" "Liveness Probe (/healthz)" {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/healthz" -Method Get -TimeoutSec 5
    if ($resp.status -ne "ok") { throw "Status expected 'ok', got '$($resp.status)'" }
    return "Liveness confirmed (status: ok)"
}

# 2. Readyz
Run-Step "V-02" "Readiness Probe (/readyz)" {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/readyz" -Method Get -TimeoutSec 5
    if ($resp.ready -ne $true -and $resp.status -ne "ready") { throw "Ready expected true, got '$($resp.ready)'" }
    return "Engine and fixtures ready (ready: true)"
}

# 3. Metrics
Run-Step "V-03" "Prometheus Telemetry (/metrics)" {
    $raw = Invoke-WebRequest -Uri "$BaseUrl/metrics" -Method Get -TimeoutSec 5 -UseBasicParsing
    if ($raw.StatusCode -ne 200 -or -not ($raw.Content -like "*urbangraph_requests_total*" -or $raw.Content -like "*urbangraph_uptime_seconds*")) {
        throw "Prometheus metrics exposition format not found"
    }
    return "Prometheus scrape endpoint operational"
}

# 4. Meta
Run-Step "V-04" "Metadata API (/api/v1/meta)" {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/meta" -Method Get -Headers $AuthHeaders -TimeoutSec 5
    if (-not $resp.manifest_sha256 -or $resp.dataset_kind -ne "synthetic") {
        throw "Invalid metadata payload"
    }
    return "Synthetic metadata verified (SHA: $($resp.manifest_sha256.Substring(0,10))...)"
}

# 5. Regions
Run-Step "V-05" "Regional Topologies (/api/v1/regions)" {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/regions" -Method Get -Headers $AuthHeaders -TimeoutSec 5
    if ($resp.regions.Count -lt 3) { throw "Expected >= 3 regions, got $($resp.regions.Count)" }
    $ids = ($resp.regions | ForEach-Object { $_.id }) -join ", "
    return "$($resp.regions.Count) regions verified ($ids)"
}

# 6. Graph
Run-Step "V-06" "Schematic Graph (/api/v1/graph)" {
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/graph" -Method Get -Headers $AuthHeaders -TimeoutSec 5
    if ($resp.regions.Count -lt 3 -or $resp.edges.Count -lt 3) { throw "Invalid graph structure" }
    return "Graph topology verified ($($resp.regions.Count) nodes, $($resp.edges.Count) edges)"
}

# 7. Simulate - DC Power Flow
Run-Step "V-07" "DC Power Flow Contingency (/api/v1/simulate)" {
    $body = @{
        scenario_type = "dc_power_flow"
        region_id = "SYN_A"
        load_scale = 1.0
        line_outages = @()
    } | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/simulate" -Method Post -Headers $AuthHeaders -Body $body -ContentType "application/json" -TimeoutSec 8
    if (-not $resp.feasible -or $resp.branch_flows.Count -eq 0) { throw "Power flow simulation did not converge" }
    return "DC Power flow converged ($($resp.branch_flows.Count) branches, feasible: $($resp.feasible))"
}

# 8. Simulate - HiGHS MILP Storage
Run-Step "V-08" "HiGHS Storage Arbitrage (/api/v1/simulate)" {
    $body = @{
        scenario_type = "battery_optimization"
        region_id = "SYN_A"
        load_scale = 1.0
        battery_capacity_ratio = 0.15
        battery_efficiency = 0.90
    } | ConvertTo-Json
    $resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/simulate" -Method Post -Headers $AuthHeaders -Body $body -ContentType "application/json" -TimeoutSec 8
    if (-not $resp.realized_cost_CU) { throw "HiGHS optimizer did not produce realized financial metrics" }
    return "HiGHS MILP solved (Net benefit: $($resp.realized_cost_CU.net_cost_CU) CU)"
}

# 9. Frontend Root & Security Headers
Run-Step "V-09" "SPA Bundle & CSP Headers (/)" {
    $resp = Invoke-WebRequest -Uri "$BaseUrl/" -Method Get -TimeoutSec 5 -UseBasicParsing
    if ($resp.StatusCode -ne 200) { throw "HTTP $($resp.StatusCode)" }
    $csp = $resp.Headers["Content-Security-Policy"]
    if (-not $csp) { throw "Content-Security-Policy header missing" }
    if (-not ($resp.Content -like "*EnerGraph*" -or $resp.Content -like "*UrbanGraph*")) { throw "HTML missing title/branding" }
    return "React SPA bundle served with hardened CSP headers"
}

# 10. Security RBAC Enforcement
Run-Step "V-10" "Security RBAC 401 Enforcement (/api/v1/predict)" {
    $body = @{
        region_id = "SYN_A"
        origin_utc = "2026-03-01T00:00:00Z"
    } | ConvertTo-Json
    try {
        $resp = Invoke-WebRequest -Uri "$BaseUrl/api/v1/predict" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 5 -ErrorAction Stop -UseBasicParsing
        throw "Expected HTTP 401, got $($resp.StatusCode)"
    } catch {
        if ($_.Exception.Response.StatusCode -eq 401) {
            return "Enforced 401 Unauthorized for uncredentialed request"
        } else {
            throw "Expected HTTP 401, got: $($_.Exception.Message)"
        }
    }
}

Write-Host "======================================================================" -ForegroundColor Cyan
if ($AllPassed) {
    Write-Host "  ALL 10 VERIFICATION STEPS PASSED (100% HEALTH AUDIT SUCCESS)" -ForegroundColor Green
    Write-Host "======================================================================" -ForegroundColor Green
    exit 0
} else {
    Write-Host "  VERIFICATION FAILED: One or more audit checks failed." -ForegroundColor Red
    Write-Host "======================================================================" -ForegroundColor Red
    exit 1
}
