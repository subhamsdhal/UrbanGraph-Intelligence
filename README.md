# UrbanGraph Intelligence

[![Release Candidate](https://img.shields.io/badge/release-0.2.0rc3-blue.svg)](./release)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Frontend](https://img.shields.io/badge/frontend-React_18_%7C_TypeScript-61dafb.svg)](./frontend)
[![Security Audit](https://img.shields.io/badge/vulnerabilities-0_known_(pip--audit)-brightgreen.svg)](./evidence)
[![E2E Verification](https://img.shields.io/badge/Playwright-10%2F10_passed-brightgreen.svg)](./acceptance)
[![Container Staging](https://img.shields.io/badge/Docker-Hardened_Production-2496ed.svg)](./docker)

**UrbanGraph Intelligence** is an enterprise-grade spatiotemporal energy grid analytics and simulation platform. It integrates Graph Neural Network (GNN) load forecasting, Direct Current (DC) optimal power flow analysis, mixed-integer linear programming (MILP) battery energy storage dispatch, and interactive geospatial visualization into a cohesive, production-hardened web application.

---

## Table of Contents

- [Architectural Overview](#architectural-overview)
- [Key Features](#key-features)
- [Safe-Use Statement & Boundaries](#safe-use-statement--boundaries)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Quickstart Guide](#quickstart-guide)
  - [Option A: Hardened Docker Compose (Recommended)](#option-a-hardened-docker-compose-recommended)
  - [Option B: Localhost Development (Windows / Linux)](#option-b-localhost-development-windows--linux)
  - [Option C: Kubernetes / Helm Staging](#option-c-kubernetes--helm-staging)
- [Core Engines & Capabilities](#core-engines--capabilities)
  - [1. Spatiotemporal Forecasting (UrbanGraphFormer)](#1-spatiotemporal-forecasting-urbangraphformer)
  - [2. DC Power Flow Simulation](#2-dc-power-flow-simulation)
  - [3. Battery Energy Storage Dispatch (HiGHS MILP)](#3-battery-energy-storage-dispatch-highs-milp)
  - [4. Geospatial Visualization Engine](#4-geospatial-visualization-engine)
- [Security Architecture & Governance](#security-architecture--governance)
- [API Reference Summary](#api-reference-summary)
- [Verification & Acceptance Evidence](#verification--acceptance-evidence)
- [Release History](#release-history)
- [License](#license)

---

## Architectural Overview

UrbanGraph Intelligence bridges advanced machine learning and power systems engineering to deliver predictive intelligence for synthetic municipal electric grids.

```
+-------------------------------------------------------------------------+
|                       Client Browser / API Consumer                     |
+-------------------------------------------------------------------------+
                                    |
                           HTTPS / HTTP :8000
                                    v
+-------------------------------------------------------------------------+
|                       Caddy Reverse Proxy & WAF                         |
|   - Strict Rate Limiting (100 req/min general, 5 req/min admin)         |
|   - Security Headers (HSTS, CSP, X-Frame-Options, X-Content-Type)       |
+-------------------------------------------------------------------------+
                                    |
                           Internal Proxy :8080
                                    v
+-------------------------------------------------------------------------+
|                    FastAPI Application Server (Python)                  |
|   - Unprivileged Execution (UID 65534 / non-root)                       |
|   - Centralized HMAC-SHA256 Token Verification & Dynamic Revocation     |
|   - Static SPA Distribution (React 18 / TypeScript / Vite)              |
+-------------------------------------------------------------------------+
          |                         |                        |
          v                         v                        v
+-------------------+     +-------------------+    +--------------------+
| UrbanGraphFormer  |     |   DC Power Flow   |    | HiGHS MILP Storage |
| PyTorch Inference |     | Simulation Engine |    |   Optimizer Engine |
+-------------------+     +-------------------+    +--------------------+
          |                         |                        |
          +-------------------------+------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   Durable Storage & Persistent State                    |
|   - SQLite Centralized Token Store (/app/security/tokens.db)            |
|   - Synthetic Fixture Repository (eg006 6-bus network)                  |
|   - Pre-trained Neural Weights (urbangraphformer_synth.pt)              |
+-------------------------------------------------------------------------+
```

---

## Key Features

- **Spatiotemporal Graph Neural Network Forecasting**: Custom Graph Transformer architecture (`UrbanGraphFormer`) predicting nodal active and reactive power demand over 24-hour horizons.
- **Physical Grid Constraints**: Direct Current (DC) power flow simulation incorporating branch reactance matrices, bus voltage angles, and line thermal limits.
- **Economic Battery Dispatch**: Mathematical optimization solver powered by `HiGHS` mixed-integer linear programming (MILP) to minimize peak energy costs and branch congestion.
- **Interactive Geospatial Digital Twin**: MapLibre GL JS vector map rendering electrical substations, transmission lines, load centers, and live simulation overlays.
- **Centralized Security & Distributed Authentication**: HMAC-SHA256 authenticated sessions with database-backed token generation, revocation tracking, and instantaneous invalidation.
- **Production-Hardened Ingress**: Integrated Caddy gateway enforcing rate limits, automated TLS termination, and OWASP-aligned HTTP security headers.
- **Zero-Vulnerability Baseline**: Continuously audited with `pip-audit` ensuring 0 known CVEs across all locked production dependencies.

---

## Safe-Use Statement & Boundaries

> **IMPORTANT SAFE-USE NOTICE**  
> UrbanGraph Intelligence is an analytical simulation and decision-support platform designed strictly for research, planning, testing, and operational staging using **synthetic and modeled datasets**.  
> - **Synthetic Scope**: All bundled demonstration networks (including the `eg006` 6-bus topology) are entirely synthetic mathematical models.  
> - **No Direct Control**: This software does NOT connect to physical supervisory control and data acquisition (SCADA) systems, programmable logic controllers (PLCs), or physical grid switchgear.  
> - **Human-in-the-Loop**: All optimization outputs, power flow results, and dispatch schedules require qualified electrical engineering verification prior to real-world capital planning or operational implementation.

---

## Repository Structure

```
UrbanGraph-Intelligence-Master/
|-- README.md                             # Authoritative project entry point
|-- LICENSE                               # Apache 2.0 open source license
|-- pyproject.toml                        # Authoritative Python package specification
|-- .gitignore                            # Comprehensive source exclusion rules
|-- .gitattributes                        # Git LFS configuration for binary assets
|-- .env.example                          # Local development environment template
|-- .env.production.template              # Production environment security template
|-- MASTER_PROJECT_INVENTORY.csv          # Complete package file manifest & SHA-256 hashes
|-- MASTER_BACKUP_AND_CLEANUP_GUIDE.md    # Storage optimization & backup runbook
|-- MASTER_PROJECT_FINAL_ACCEPTANCE.md    # Definitive release sign-off certificate
|-- src/                                  # Complete Python backend source tree
|   `-- urbangraph/                       # Main application package
|       |-- api/                          # FastAPI REST routers & endpoints
|       |-- checkpoints/                  # Pretrained model checkpoint
|       |-- fixtures/                     # Bundled network fixtures (eg006)
|       |-- ml/                           # Neural network models & inference pipelines
|       |-- power/                        # DC power flow & grid simulation routines
|       |-- security/                     # Centralized token auth & RBAC
|       |-- static/                       # Production compiled React SPA assets
|       `-- storage/                      # HiGHS MILP optimization models
|-- frontend/                             # Complete React 18 / TypeScript source code
|   |-- src/                              # Components, hooks, state stores, views
|   |-- package.json                      # Frontend dependency declarations
|   |-- package-lock.json                 # Pinned npm dependency lockfile
|   |-- tsconfig.json                     # Strict TypeScript compiler options
|   `-- vite.config.ts                    # Vite bundle build configuration
|-- tests/                                # Comprehensive test suites
|   |-- unit/                             # Unit tests across backend modules
|   |-- integration/                      # API contract and database tests
|   `-- e2e/                              # Browser automation test definitions
|-- scripts/                              # Operational automation scripts
|   |-- start_local.ps1                   # Localhost service launcher
|   |-- stop_local.ps1                    # Clean service shutdown script
|   |-- verify_local.ps1                  # 10-point local healthcheck verification
|   |-- run_local_tests.ps1               # Automated test execution harness
|   `-- reset_local_demo.ps1              # Database & cache reset utility
|-- models/                               # Standalone model assets & metadata
|   |-- urbangraphformer_synth.pt         # Pretrained PyTorch model weights
|   `-- MODEL_CARD.md                     # Model architecture & provenance card
|-- fixtures/                             # Standalone grid datasets
|   |-- eg006/                            # 6-bus synthetic transmission network
|   `-- FIXTURES_README.md                # Network parameters & physics reference
|-- config/                               # Production & local configuration files
|-- docker/                               # Container orchestration specifications
|   |-- Dockerfile.production             # Hardened unprivileged multi-stage container
|   |-- Dockerfile.caddy                  # Custom rate-limiting Caddy proxy
|   |-- Caddyfile                         # Ingress rules, headers & rate limits
|   `-- docker-compose.production.yml     # Complete two-tier production stack
|-- infrastructure/                       # Deployment infrastructure
|   |-- helm/                             # Enterprise Helm staging charts
|   `-- k8s/                              # Raw Kubernetes pod & service manifests
|-- monitoring/                           # Telemetry & observability
|   |-- prometheus_rules.yaml             # Latency, error rate & saturation alerts
|   `-- grafana_dashboard.json            # Operational monitoring dashboard
|-- release/                              # Authoritative binary distribution
|   |-- urbangraph-0.2.0rc3-py3-none-any.whl # Verified binary wheel
|   `-- RELEASE_METADATA.json             # Release manifest & build provenance
|-- evidence/                             # Verification records & audit artifacts
|   |-- 03_SOFTWARE_BILL_OF_MATERIALS.json # Comprehensive SBOM
|   |-- 14_FULL_PYTHON_REGRESSION_SUMMARY.md # Test execution breakdown
|   `-- *.xml                             # Authoritative JUnit test results
|-- acceptance/                           # Acceptance certificates & captures
|   |-- 35_STEP_ACCEPTANCE_PROOF.json     # Comprehensive acceptance ledger
|   |-- PRODUCTION_RELIABILITY_PROOF.json # Reliability & stress audit
|   `-- screenshots/                      # 5 high-resolution UI captures
|-- documentation/                        # 17 complete technical manuals
|-- reports/                              # Definitive project deliverables
|   |-- UrbanGraph_Intelligence_Complete_Project_Report.md
|   `-- UrbanGraph_Intelligence_Complete_Project_Report.pdf
|-- presentation/                         # Stakeholder briefing materials
|   `-- UrbanGraph_Intelligence_Project_Presentation.pptx
|-- examples/                             # Integration code samples & API usage
|-- security/                             # Security policies & threat models
`-- checksums/                            # Cryptographic verification
    `-- SHA256SUMS.txt                    # SHA-256 signatures of all assets
```

---

## Quickstart Guide

### Option A: Hardened Docker Compose (Recommended)

The easiest and most secure way to run UrbanGraph Intelligence is using the production Docker Compose stack, which launches both the rate-limited Caddy gateway and the unprivileged FastAPI application.

1. **Clone the repository and enter the directory**:
   ```bash
   git clone https://github.com/your-org/urbangraph-intelligence.git
   cd urbangraph-intelligence
   ```

2. **Initialize Environment Configuration**:
   ```bash
   cp .env.production.template .env
   # Edit .env to set your secure ENERGRAPH_DEMO_SECRET_KEY and ADMIN_TOKEN
   ```

3. **Build and Launch Containers**:
   ```bash
   docker compose -f docker/docker-compose.production.yml up -d --build
   ```

4. **Verify Container Health**:
   ```bash
   docker compose -f docker/docker-compose.production.yml ps
   curl -i http://localhost:8000/api/v1/health
   ```

5. **Access Application**:
   Open `http://localhost:8000` in your web browser.

---

### Option B: Localhost Development (Windows / Linux)

For local development or testing without Docker:

1. **Prerequisites**:
   - Python 3.10 or higher
   - Node.js 18+ and npm (optional, only if modifying frontend source)
   - Virtual environment manager (`venv`)

2. **Create and Activate Virtual Environment**:
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install Pre-built Release Wheel**:
   ```powershell
   pip install release/urbangraph-0.2.0rc3-py3-none-any.whl
   ```

4. **Launch Localhost Server**:
   ```powershell
   .\scripts\start_local.ps1
   ```

5. **Verify Installation**:
   ```powershell
   .\scripts\verify_local.ps1
   ```

6. **Shut Down Cleanly**:
   ```powershell
   .\scripts\stop_local.ps1
   ```

---

### Option C: Kubernetes / Helm Staging

To deploy into an existing Kubernetes cluster:

```bash
helm lint infrastructure/helm/energraph-ai
helm upgrade --install urbangraph infrastructure/helm/energraph-ai \
  --namespace staging --create-namespace \
  -f infrastructure/helm/energraph-ai/values-production.yaml
```

---

## Core Engines & Capabilities

### 1. Spatiotemporal Forecasting (UrbanGraphFormer)
The forecasting core employs a hybrid Graph Attention Network (GAT) and Temporal Transformer. Spatial edge weights reflect physical transmission line susceptances ($B_{ij}$), while temporal attention layers capture intraday and seasonal consumption dynamics.
- **Inputs**: Historical active ($P$) and reactive ($Q$) loads, ambient temperature, solar irradiance.
- **Outputs**: 24-hour ahead nodal demand expectations with 95% confidence intervals.

### 2. DC Power Flow Simulation
Physical network states are calculated via linear DC power flow equations:
$$P_i = \sum_{j \in \mathcal{N}_i} B_{ij} (	heta_i - 	heta_j)$$
The engine verifies branch thermal capacity constraints:
$$|P_{ij}| \le P_{ij}^{\max}$$
and computes Power Transfer Distribution Factors (PTDF) for rapid contingency screening.

### 3. Battery Energy Storage Dispatch (HiGHS MILP)
Battery dispatch is formulated as a Mixed-Integer Linear Program solved by `HiGHS`:
$$\min \sum_{t=1}^{T} \left[ C_t^{	ext{grid}} P_t^{	ext{grid}} + C^{	ext{deg}} (P_t^{	ext{ch}} + P_t^{	ext{dis}}) ight]$$
Subject to:
- State of Charge (SoC) bounds: $	ext{SoC}^{\min} \le 	ext{SoC}_t \le 	ext{SoC}^{\max}$
- Charge/discharge power ratings: $0 \le P_t^{	ext{ch}} \le P_{\max}^{	ext{ch}}$, $0 \le P_t^{	ext{dis}} \le P_{\max}^{	ext{dis}}$
- Non-simultaneous charging constraint: $u_t^{	ext{ch}} + u_t^{	ext{dis}} \le 1$

### 4. Geospatial Visualization Engine
The frontend combines React 18 with MapLibre GL JS to provide high-performance GPU-accelerated vector mapping of electrical topology:
- Dynamic line color encoding based on percentage thermal loading ($<60\%$ Green, $60-85\%$ Amber, $>85\%$ Red).
- Interactive nodal substation inspection displaying voltage magnitude, phase angle, and generation mix.

---

## Security Architecture & Governance

UrbanGraph Intelligence follows defense-in-depth principles:
- **Centralized Token Management**: All administrative bearer tokens are tracked in a persistent SQLite database (`tokens.db`). Token rotation generates a new key and immediately revokes the retired token.
- **Unprivileged Containers**: The application executes under `UID 65534` (`nobody`), preventing filesystem privilege escalation.
- **Ingress Rate Limiting**: The reverse proxy enforces a maximum of 100 requests/minute on standard endpoints and 5 requests/minute on administrative endpoints.
- **Dependency Hardening**: Pinned zero-vulnerability packages verified against the PyPA advisory database using `pip-audit`.

---

## API Reference Summary

| Method | Endpoint | Description | Auth Required |
|:-------|:---------|:------------|:--------------|
| `GET` | `/api/v1/health` | Service liveness and dependency status | None |
| `GET` | `/api/v1/metrics` | Prometheus operational metrics | None |
| `GET` | `/api/v1/grid/topology` | Bus and branch network topology | Bearer Token |
| `POST` | `/api/v1/grid/simulate` | Execute DC power flow calculation | Bearer Token |
| `POST` | `/api/v1/forecast/predict` | Generate 24h neural load forecast | Bearer Token |
| `POST` | `/api/v1/storage/optimize` | Solve HiGHS MILP battery schedule | Bearer Token |
| `POST` | `/api/v1/admin/rotate-token` | Rotate admin authentication token | Admin Token |

Detailed endpoint specifications, JSON schemas, and request examples are available in [documentation/04_API_REFERENCE.md](./documentation/04_API_REFERENCE.md).

---

## Verification & Acceptance Evidence

This release package has undergone extensive independent verification:
- **Unit & Integration Regression**: 14/14 partitions completed with 100% pass rate.
- **E2E Browser Journeys**: 10/10 automated Playwright scenarios validated against live container.
- **Stress & Concurrency**: 200 concurrent requests across 10 workers completed with 0 errors (98.6 req/s).
- **Vulnerability Audit**: 0 CVEs detected by `pip-audit`.

Full verification logs and test receipts are cataloged in [evidence/](./evidence) and [acceptance/](./acceptance).

---

## Release History

- **v0.2.0rc3 (Current)**: Hardened production container, unprivileged execution, Caddy rate-limiting proxy, zero-vulnerability dependency baseline, and definitive documentation suite.
- **v0.2.0rc2**: Localhost Windows acceptance, centralized SQLite token management, and MapLibre geospatial fixes.
- **v0.2.0rc1**: Integration of HiGHS MILP solver and initial release candidate packaging.
- **v0.1.0**: Initial prototype featuring DC power flow and synthetic neural forecasting.

See [documentation/16_RELEASE_HISTORY.md](./documentation/16_RELEASE_HISTORY.md) for full change logs.

---

## License

UrbanGraph Intelligence is released under the [Apache License 2.0](./LICENSE).
