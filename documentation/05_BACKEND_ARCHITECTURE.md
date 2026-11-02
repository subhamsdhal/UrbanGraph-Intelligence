# 05 - Backend Architecture: UrbanGraph Intelligence

## 1. Architectural Philosophy
The backend is engineered for high throughput, sub-10ms simulation times, asynchronous concurrency, and defense-in-depth security. Built on **FastAPI** and **Pydantic v2**, it strictly decouples routing, validation, domain physics, optimization modeling, and persistence.

---

## 2. Layered Architecture

```
+-----------------------------------------------------------+
|                      FastAPI Routers                      |
|         (api/routers/{health, grid, forecast, admin})     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                   Pydantic v2 Contracts                   |
|         (schemas/{grid_schema, forecast_schema, ...})     |
+-----------------------------------------------------------+
                             |
                             v
+-----------------------------------------------------------+
|                      Service Layer                        |
|       - PowerFlowService (DC power flow matrix math)      |
|       - InferenceService (PyTorch GNN inference)          |
|       - OptimizationService (HiGHS MILP formulation)      |
|       - TokenService (SQLite HMAC centralized auth)       |
+-----------------------------------------------------------+
```

---

## 3. Core Modules

- `urbangraph.power`: Implements nodal admittance matrix formulation, susceptance matrix inversion ($B^{-1}$), and PTDF calculation using `scipy.sparse` and `numpy`.
- `urbangraph.ml`: Contains the `UrbanGraphFormer` architecture, spatial graph attention convolution, temporal self-attention, and PyTorch CPU inference pipelines.
- `urbangraph.storage`: Implements battery dispatch mixed-integer linear programming models using `highspy`.
- `urbangraph.security`: Provides centralized SQLite token verification, HMAC-SHA256 signature checking, and constant-time string comparisons.
