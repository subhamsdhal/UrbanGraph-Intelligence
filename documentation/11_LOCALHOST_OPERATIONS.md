# 11 - Localhost Operations Runbook

## 1. Launching Local Services
```powershell
.\scripts\start_local.ps1
```
The script:
1. Validates Python environment and package installation.
2. Checks port 8000 availability.
3. Launches background server process and records PID in `urbangraph.pid`.
4. Polls healthcheck endpoint until HTTP 200 is confirmed.

## 2. Stopping Local Services
```powershell
.\scripts\stop_local.ps1
```
Cleanly terminates the background process and removes the PID file.

## 3. Running Verification Suite
```powershell
.\scripts\verify_local.ps1
```
Executes a 10-step comprehensive diagnostic validating API endpoints, database health, model inference, and frontend bundle integrity.
