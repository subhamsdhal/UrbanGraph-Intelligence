# 03 - Administrator Manual: UrbanGraph Intelligence

## 1. Operational Overview
The system administrator is responsible for managing authentication tokens, security policies, rate limits, database persistence, and system observability.

---

## 2. Centralized Token Management

### 2.1 Token Store Architecture
Authentication tokens are stored centrally in a SQLite database located at `ENERGRAPH_SECURITY_DIR/tokens.db`. This eliminates multi-replica inconsistency and enables instantaneous revocation.

Schema:
```sql
CREATE TABLE tokens (
    token_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_revoked BOOLEAN DEFAULT FALSE
);
```

### 2.2 Token Rotation Procedure
Tokens should be rotated every 30 days or immediately upon suspected compromise.

**Via Admin REST API**:
```bash
curl -X POST http://localhost:8000/api/v1/admin/rotate-token \
  -H "Authorization: Bearer <CURRENT_ADMIN_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin", "validity_hours": 24}'
```
Response:
```json
{
  "status": "success",
  "new_token": "ug_adm_9f8e7d6c5b4a3...",
  "expires_at": "2026-11-03T16:00:00Z",
  "revoked_token_id": "ug_adm_1a2b3c4d5e..."
}
```

### 2.3 Manual Revocation via CLI
```bash
python -m urbangraph.security.revoke --token-id "ug_adm_1a2b3c4d5e..."
```

---

## 3. Rate Limiting Configuration
Caddy enforces ingress rate limits:
- Standard endpoints (`/api/v1/*`): 100 requests per minute per client IP.
- Sensitive endpoints (`/api/v1/admin/*`): 5 requests per minute per client IP.

To modify rate limits, edit `docker/Caddyfile`:
```caddy
rate_limit {
    zone dynamic_zone {
        key {remote_host}
        events 100
        window 1m
    }
}
```

---

## 4. Prometheus Metrics & Alerting
Metrics are exposed at `http://localhost:8000/api/v1/metrics`:
- `urbangraph_http_requests_total`: Total API requests partitioned by route, method, and status.
- `urbangraph_simulation_seconds`: Histogram of DC power flow and HiGHS solve latencies.
- `urbangraph_active_tokens`: Gauge of non-expired, non-revoked bearer tokens.
