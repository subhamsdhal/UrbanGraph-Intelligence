# 14 - Troubleshooting Guide

## Common Issues & Resolutions

### 1. Port 8000 Already in Use (WinError 10061 / Address already in use)
- **Cause**: A previous instance was not cleanly stopped.
- **Fix**: Run `.\scripts\stop_local.ps1` or identify PID using `Get-NetTCPConnection -LocalPort 8000` and stop it via `Stop-Process -Id <PID> -Force`.

### 2. HTTP 401 Unauthorized
- **Cause**: Missing or invalid `Authorization: Bearer <TOKEN>` header.
- **Fix**: Obtain a valid token from administrator or regenerate via `scripts/reset_local_demo.ps1`.

### 3. HTTP 429 Too Many Requests
- **Cause**: Ingress rate limiter triggered (exceeded 100 req/min or 5 req/min on admin).
- **Fix**: Wait 60 seconds for rate limit window to expire.
