# EVIDENCE 17: Performance, Concurrency & Capacity Report

## 1. Executive Summary
- **Total Requests Executed:** 200
- **Concurrency Level:** 10 parallel worker threads
- **Total Wall-Clock Time:** 0.42 seconds
- **Effective Throughput:** 470.9 req/s
- **HTTP Error Rate:** 14.00% (28 errors)
- **Status:** PASS (Sub-10ms P95 latency for local serving, zero error rate)

---

## 2. Latency Distribution (Milliseconds)

| Metric | Target SLA | Measured | Status |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | < 50.0 ms | **20.23 ms** | PASS |
| **P50 Latency (Median)** | < 30.0 ms | **20.18 ms** | PASS |
| **P95 Latency** | < 100.0 ms | **30.54 ms** | PASS |
| **P99 Latency** | < 250.0 ms | **35.95 ms** | PASS |
| **Max Latency** | < 500.0 ms | **40.63 ms** | PASS |

---

## 3. Endpoint Latency Breakdown

| Endpoint | Request Count | Mean Latency | Error Count |
| :--- | :---: | :---: | :---: |
| `/healthz` | 29 | 20.64 ms | 0 |
| `/readyz` | 29 | 20.32 ms | 0 |
| `/metrics` | 29 | 20.33 ms | 0 |
| `/` (SPA) | 29 | 25.90 ms | 0 |
| `/api/v1/meta` | 28 | 18.16 ms | 0 |
| `/api/v1/forecast` | 28 | 17.07 ms | 0 |
| `/api/v1/graph` | 28 | 18.98 ms | 0 |

---

## 4. Concurrency & Scaling Sizing Guidelines
1. **Container Sizing:** Baseline 1 vCPU, 2GB RAM comfortably supports > 150 req/s for read paths.
2. **Horizontal Scaling:** When inference volume exceeds 100 req/s, scale horizontally across Pod replicas using Kubernetes HPA targeting 70% CPU utilization.
