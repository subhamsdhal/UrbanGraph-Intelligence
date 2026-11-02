# EVIDENCE 12: Deployment Configuration Audit

## 1. Resource Allocations
- Container CPU Request: `500m`
- Container CPU Limit: `2000m`
- Container Memory Request: `1Gi`
- Container Memory Limit: `4Gi`

## 2. Health & Liveness Probes
- Liveness Probe: `GET /healthz` on port 8000, initialDelay: 15s, period: 10s.
- Readiness Probe: `GET /readyz` on port 8000, initialDelay: 10s, period: 5s.

## 3. Storage & Persistence
- Root filesystem is mounted read-only (`readOnlyRootFilesystem: true`).
- Ephemeral writable scratch space is provided via dedicated `emptyDir` mounted at `/tmp` with memory backing.
- Model checkpoints are loaded from pre-baked image layers or mounted read-only volume (`/opt/urbangraph/checkpoints`).

## 4. Network Security & Ingress
- ClusterIP default exposure; Ingress TLS terminated via cert-manager.
- NetworkPolicy isolates internal traffic, restricting ingress to authorized API gateways.
