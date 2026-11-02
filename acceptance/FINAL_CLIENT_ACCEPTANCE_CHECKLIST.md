# UrbanGraph Intelligence — Client Acceptance Checklist
**Candidate:** `0.2.0rc3`  
**Client Target:** Enterprise Staging Cluster  

---

## Verification Criteria
- [x] Wheel package installs cleanly in isolated environment (R01)
- [x] Query parameter tokens eliminated; Bearer header enforced (R02)
- [x] Timing-safe token comparison verified via `hmac.compare_digest` (R03)
- [x] RBAC enforcement verified across admin, worker, reader (R04)
- [x] Standard HTTP security headers active (R05)
- [x] Single-port unified hosting active (R06)
- [x] MapLibre GL JS v6.16.0 & Deck.gl rendering verified (R07)
- [x] WCAG 2.1 AA accessibility compliant (0 Axe-core violations) (R08)
- [x] Responsive layout verified (0 horizontal overflow) (R09)
- [x] UrbanGraphFormer checkpoint integrity verified (R10)
- [x] 24-step forecasting horizon validated (R11)
- [x] Conformal prediction uncertainty intervals verified (R12)
- [x] Real-time sensor noise resilience verified (R13)
- [x] DC Kirchhoff power flow constraints enforced (R14)
- [x] BESS state-of-charge limits enforced (R15)
- [x] Docker image `urbangraph:0.2.0rc3` verified non-root (UID 65534) (R16)
- [x] Helm configuration enforces single-worker per pod (R17)
- [x] Prometheus `/metrics` exposition verified (R18)
- [x] Liveness and readiness probes verified (R19)
- [x] Kafka streaming adapter verified (R20)
- [x] Fast socket rebind (< 3s) verified (R21)
- [x] Disaster recovery and state snapshot verified (R22)
- [x] Clean-room rollback verified (R23)
- [x] Full Python regression passes (924 passed, 10 skipped, 0 failed) (R24)
- [x] Advisory boundary enforced on power grid actuation (R25)
- [x] External production gates clearly registered (G-01 to G-09) (R26)
