# 10 - Security Architecture: UrbanGraph Intelligence

## 1. Security Principles
- **Defense in Depth**: Multi-layer security across reverse proxy, application framework, and database.
- **Least Privilege**: Application runs under unprivileged user `UID 65534` (`nobody`).
- **Zero-Vulnerability Baseline**: Pinned dependencies verified with `pip-audit`.
- **Zero In-Memory Drift**: Centralized SQLite database handles token validity and instant revocation.

---

## 2. Role-Based Access Control (RBAC) Matrix

| Endpoint | Method | Viewer | Operator | Admin |
|:---------|:-------|:-------|:---------|:------|
| `/api/v1/health` | GET | Allowed | Allowed | Allowed |
| `/api/v1/grid/topology` | GET | Allowed | Allowed | Allowed |
| `/api/v1/grid/simulate` | POST | Denied | Allowed | Allowed |
| `/api/v1/forecast/predict` | POST | Denied | Allowed | Allowed |
| `/api/v1/storage/optimize` | POST | Denied | Allowed | Allowed |
| `/api/v1/admin/rotate-token` | POST | Denied | Denied | Allowed |
| `/api/v1/admin/audit-logs` | GET | Denied | Denied | Allowed |
