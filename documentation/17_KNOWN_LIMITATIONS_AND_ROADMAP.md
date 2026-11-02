# 17 - Known Limitations and Roadmap

## 1. Known Limitations & Safe Boundaries
- **Synthetic Scope**: Demonstration networks (including `eg006`) are mathematical models.
- **No SCADA Interface**: Does not connect to live electrical utility telemetry.
- **Linear DC Approximation**: Reactive power and voltage magnitude variations are modeled under linear assumptions; full non-linear AC optimal power flow is scheduled for v0.3.0.
- **SQLite Concurrency**: Token store uses SQLite; suitable for single-node deployments. Multi-region deployments should migrate to PostgreSQL.

---

## 2. Development Roadmap (v0.3.0 & Beyond)

### Phase 1: Advanced Power Systems (v0.3.0)
- AC Optimal Power Flow (AC-OPF) solver integration via Interior Point Methods (Ipopt).
- Dynamic contingency analysis (N-1 line outage criteria).

### Phase 2: Enterprise Cloud Architecture (v0.4.0)
- Native PostgreSQL token and session backend.
- OpenID Connect (OIDC) / OAuth2 single sign-on integration.
- Distributed Celery / Redis task queue for massive Monte Carlo grid simulations.
