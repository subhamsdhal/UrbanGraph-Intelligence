# UrbanGraph Intelligence — Supported Worker Configuration Guide

| Deployment Environment | Supported Worker Count | Scaling Mechanism | Telemetry Accuracy |
|---|---|---|---|
| **Local Standalone / Demo** | `--workers 1` | Single Process | 100% Deterministic |
| **Kubernetes Container (Prod)** | `--workers 1` | Horizontal Pod Autoscaler (HPA) | 100% Scraped per Pod |
| **Bare-Metal Multi-Worker** | `--workers 2+` | Forked Supervisor | Scraper Round-Robin (Dispersed) |

**Enforcement:**
In production containers, the default entrypoint sets `UG_WORKERS=1`.
Horizontal scalability is achieved by increasing `replicas: N` in Helm `values.yaml`.
