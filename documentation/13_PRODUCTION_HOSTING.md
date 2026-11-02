# 13 - Production Hosting Guide

## 1. Hosting Architecture Requirements
- **Stateful Persistence**: The token database requires an ACID-compliant POSIX filesystem mount. Ephemeral instance storage (e.g., standard Cloud Run without volume mounts) or object storage FUSE layers (e.g., GCS FUSE) are prohibited for SQLite write locks.
- **Single Replica / Distributed Token Store**: For multi-replica deployments, tokens must be backed by a shared network filesystem with file-locking support (NFS/EFS) or migrated to PostgreSQL.
- **SSL / TLS Termination**: Terminate HTTPS at the ingress load balancer or Caddy gateway.
