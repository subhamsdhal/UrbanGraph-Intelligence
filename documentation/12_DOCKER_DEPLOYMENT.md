# 12 - Docker Deployment Guide

## 1. Multi-Stage Hardened Container
The production image `Dockerfile.production` uses a two-stage build:
1. **Builder Stage**: Installs development tools, builds wheels, compiles frontend SPA.
2. **Runtime Stage**: Clean `python:3.12-slim` image, non-root user (`UID 65534`), pinned zero-CVE dependencies.

## 2. Running with Docker Compose
```bash
docker compose -f docker/docker-compose.production.yml up -d
```
Services:
- `caddy`: Custom Caddy gateway handling TLS termination and rate limits.
- `urbangraph-backend`: Hardened FastAPI application server.
- `urbangraph-security-data`: Named persistent volume for `tokens.db`.
