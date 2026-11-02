# UrbanGraph Intelligence — Telemetry Architecture Decision & Worker Topology

**Decision Date:** 2026-11-01  
**Status:** APPROVED (Option A — Preferred Simple Production Deployment)  

---

## 1. Problem Statement & Root Cause

In multi-worker Uvicorn (`--workers > 1`), independent operating system processes are spawned to handle concurrent HTTP connections.
`UrbanGraphMetrics` operates as an in-memory Python singleton. Each worker increments its own private memory counter.
When an external Prometheus scraper requests `/metrics`, Uvicorn routes the request to one worker, which exports only its local in-memory counters.
Consequently, multi-worker telemetry results in counter dispersion across worker PIDs.

---

## 2. Architecture Decision: Option A (Recommended Production Standard)

**Topology:** Run **1 Uvicorn Worker per Container Replica (`--workers 1`)**.
**Horizontal Scaling:** Scale container Pods horizontally using Kubernetes Replicas and Horizontal Pod Autoscaling (HPA).
**Telemetry Scraping:**
- Expose each individual Pod's `/metrics` endpoint to Prometheus via a standard Kubernetes `PodMonitor` or `ServiceMonitor`.
- Prometheus scrapes each Pod directly, aggregating cluster-wide requests, latencies, and error rates in Prometheus PromQL:
  `sum(rate(urbangraph_http_requests_total[5m])) by (endpoint)`
- This guarantees 100% telemetry accuracy without complex multi-process file locks or inter-process communication overhead.

---

## 3. Alternative Option B (Native Multi-Process) Classification

- Implementing `prometheus_client` with `PROMETHEUS_MULTIPROC_DIR` is available as a future enhancement for monolithic multi-worker bare-metal hosts.
- For containerized and cloud-native Kubernetes deployments, Option A is the industry-standard cloud-native architecture.
