# 04 - API Reference: UrbanGraph Intelligence

## Base URL
- Production / Docker Ingress: `http://localhost:8000/api/v1`
- Direct Backend: `http://localhost:8080/api/v1`

---

## 1. System Endpoints

### GET `/health`
Returns system health, database connectivity, and version metadata.
- **Auth**: None
- **Response**:
  ```json
  {
    "status": "healthy",
    "version": "0.2.0rc3",
    "timestamp": "2026-11-02T16:00:00Z",
    "components": {
      "database": "connected",
      "model_loaded": true,
      "solver_available": true
    }
  }
  ```

### GET `/metrics`
Prometheus exposition format operational metrics.
- **Auth**: None

---

## 2. Grid Simulation Endpoints

### GET `/grid/topology`
Retrieves buses, branches, generators, and loads for a designated fixture.
- **Auth**: Bearer Token (`viewer`, `operator`, `admin`)
- **Query Params**:
  - `network_id` (string, optional, default: `eg006`)
- **Response**:
  ```json
  {
    "network_id": "eg006",
    "base_mva": 100.0,
    "buses": [
      {"id": 1, "type": "slack", "base_kv": 230.0, "p_mw": 0.0, "q_mvar": 0.0},
      {"id": 2, "type": "pv", "base_kv": 230.0, "p_mw": 50.0, "q_mvar": 0.0}
    ],
    "branches": [
      {"from_bus": 1, "to_bus": 2, "r_pu": 0.01, "x_pu": 0.05, "rating_mva": 100.0}
    ]
  }
  ```

### POST `/grid/simulate`
Executes linear DC power flow simulation.
- **Auth**: Bearer Token (`operator`, `admin`)
- **Request Body**:
  ```json
  {
    "network_id": "eg006",
    "load_scaling": 1.10,
    "slack_bus": 1
  }
  ```
- **Response**:
  ```json
  {
    "converged": true,
    "execution_time_ms": 4.2,
    "bus_angles_deg": {"1": 0.0, "2": -1.45, "3": -2.80, "4": -3.12, "5": -3.95, "6": -4.21},
    "branch_flows_mw": {
      "branch_1_2": 42.1,
      "branch_1_4": 38.5,
      "branch_2_3": 15.2
    },
    "max_loading_percent": 76.4,
    "congested_branches": []
  }
  ```

---

## 3. Machine Learning & Optimization Endpoints

### POST `/forecast/predict`
Generates 24-hour nodal demand forecasts using `UrbanGraphFormer`.
- **Auth**: Bearer Token
- **Request Body**:
  ```json
  {
    "network_id": "eg006",
    "horizon_hours": 24,
    "temperature_c": 22.5
  }
  ```

### POST `/storage/optimize`
Computes optimal battery charging/discharging schedule via `HiGHS` MILP.
- **Auth**: Bearer Token
- **Request Body**:
  ```json
  {
    "battery_capacity_mwh": 20.0,
    "power_rating_mw": 5.0,
    "initial_soc": 0.50,
    "objective": "peak_shaving"
  }
  ```

---

## 4. Administrative Endpoints

### POST `/admin/rotate-token`
Rotates administrative credentials and revokes previous tokens.
- **Auth**: Bearer Token (`admin` role required)
- **Rate Limit**: 5 requests / minute
