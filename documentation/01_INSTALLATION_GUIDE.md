# 01 - Installation Guide: UrbanGraph Intelligence

## Overview
This document provides instructions for installing and running UrbanGraph Intelligence across supported environments:
1. Production Docker Container (Recommended)
2. Local Windows PowerShell Virtual Environment
3. Kubernetes / Helm Deployment

---

## 1. Prerequisites

### Hardware Requirements
- **CPU**: 4 cores minimum (8 cores recommended for concurrent MILP solving and neural inference).
- **RAM**: 8 GB minimum (16 GB recommended).
- **Disk**: 5 GB free disk space.

### Software Dependencies
- **Docker**: Docker Engine 24+ and Docker Compose v2 (for container deployment).
- **Python**: Python 3.10, 3.11, or 3.12 (for local development).
- **PowerShell**: Windows PowerShell 5.1 or PowerShell Core 7+ (for Windows local scripts).

---

## 2. Docker Production Deployment (Recommended)

The production deployment runs a two-tier architecture: Caddy reverse proxy on port 8000 and the hardened FastAPI backend on internal port 8080.

```bash
# 1. Navigate to docker directory
cd docker

# 2. Build and launch services
docker compose -f docker-compose.production.yml up -d --build

# 3. Verify container health
docker compose -f docker-compose.production.yml ps
```

Healthcheck endpoints:
- Ingress: `http://localhost:8000/api/v1/health`
- Proxy Status: `http://localhost:8000/` (serves the React SPA)

---

## 3. Local Windows Development Installation

### Step 1: Create Python Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Pre-built Release Wheel
```powershell
pip install --upgrade pip
pip install release/urbangraph-0.2.0rc3-py3-none-any.whl
```

### Step 3: Install Dev Dependencies (Optional)
```powershell
pip install -e ".[dev]"
```

### Step 4: Launch Localhost Server
```powershell
.\scripts\start_local.ps1
```

### Step 5: Verification
Run the 10-point local verification suite:
```powershell
.\scripts\verify_local.ps1
```

---

## 4. Kubernetes / Helm Staging Deployment

```bash
# Lint the Helm chart
helm lint infrastructure/helm/energraph-ai

# Dry-run template render
helm template urbangraph infrastructure/helm/energraph-ai -f infrastructure/helm/energraph-ai/values-production.yaml

# Deploy to cluster
helm upgrade --install urbangraph infrastructure/helm/energraph-ai \
  --namespace staging --create-namespace \
  -f infrastructure/helm/energraph-ai/values-production.yaml
```

---

## 5. Post-Installation Verification Checklist

- [ ] HTTP GET `/api/v1/health` returns `{"status": "ok", "version": "0.2.0rc3"}`.
- [ ] Root page `http://localhost:8000` loads the React frontend without JavaScript console errors.
- [ ] Centralized token store initializes `/app/security/tokens.db` (or `./security/tokens.db`).
- [ ] Pre-trained neural weights load from `models/urbangraphformer_synth.pt`.
