# 16 - Release History & Version Log

## [0.2.0rc3] - 2026-11-02
- **Production Hardening**: Created `Dockerfile.production` running unprivileged non-root user (`UID 65534`).
- **Dependency Audit**: Pinned all dependencies; remediated 120 vulnerabilities down to 0 CVEs via `pip-audit`.
- **Ingress Proxy**: Integrated Caddy gateway with automated rate limiting (100 req/min general, 5 req/min admin).
- **E2E Playwright Suite**: 10/10 automated browser journey tests validated against live container.
- **Master Packaging**: Consolidated all assets into authoritative GitHub-ready repository.

## [0.2.0rc2] - 2026-11-01
- **Localhost Acceptance**: 10/10 Windows localhost verification checks passed.
- **Centralized Tokens**: Implemented SQLite centralized token management resolving distributed authentication drift.
- **Geospatial Fixes**: Resolved MapLibre GL runtime container loading issues.

## [0.2.0rc1] - 2026-10-31
- **HiGHS MILP Solver**: Integrated mixed-integer linear programming battery energy storage optimization.
- **Release Candidate Wheel**: Packaged `urbangraph-0.2.0rc3-py3-none-any.whl`.

## [0.1.0] - 2026-10-18
- Initial functional prototype with DC power flow and synthetic neural load forecasting.
