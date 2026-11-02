"""Local-only read-only FastAPI application for approved synthetic fixtures."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import hmac
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import PlainTextResponse

from urbangraph.monitoring.observability import PrometheusExporter, get_global_metrics
from urbangraph.monitoring.drift import FeatureDriftDetector
from urbangraph.serving.resilience import RuntimeResilienceController
from urbangraph.data import DataError
from urbangraph.security.token_rotation import TokenRotationEngine, RoleType

from energraph.demo.contracts import (
    Battery,
    Dispatch,
    DispatchPoint,
    ForecastPoint,
    Forecasts,
    Graph,
    GraphEdge,
    GraphNode,
    Meta,
    Region,
    Regions,
    Series,
    SeriesPoint,
    ResilientPredictRequest,
    ResilientForecastPoint,
    ResilientPredictResponse,
    DriftCheckRequest,
    DriftMonitoringStatus,
    SimulationRequest,
    SimulationResponse,
)
from energraph.demo.fixtures import DemoData, load_demo
from energraph.optimization.physical import (
    BatterySpecification,
    SyntheticTariff,
    check_schedule,
    realized_cost,
)
from energraph.optimization.solver import optimize


def _time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(422, "Expected ISO-8601 UTC timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != UTC.utcoffset(result):
        raise HTTPException(422, "Expected ISO-8601 UTC timestamp")
    return result


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"schema_version": "eg006-demo-v1", "error": {"code": code, "message": message}},
    )


def create_app(package: Path | None = None, model_checkpoint: Path | None = None) -> FastAPI:
    """Validate the complete package before exposing any route."""
    if package is None:
        configured = os.environ.get("ENERGRAPH_DEMO_PACKAGE")
        if not configured:
            raise ValueError("ENERGRAPH_DEMO_PACKAGE must name the approved synthetic package")
        package = Path(configured)
    data: DemoData = load_demo(package)
    app = FastAPI(title="EnerGraph Synthetic Demo API", version="eg006-demo-v1")

    # Load trained model checkpoint if provided or available
    trained_model: Any = None
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    if model_checkpoint is None:
        configured_cp = os.environ.get("URBANGRAPH_MODEL_CHECKPOINT")
        if configured_cp:
            model_checkpoint = Path(configured_cp)
        else:
            default_cp = (
                Path(__file__).resolve().parents[3]
                / "tests"
                / "fixtures"
                / "checkpoints"
                / "urbangraphformer_synth.pt"
            )
            if default_cp.exists():
                model_checkpoint = default_cp

    if model_checkpoint is not None and Path(model_checkpoint).exists():
        try:
            from urbangraph.neural.urbangraphformer import UrbanGraphFormer
            trained_model = UrbanGraphFormer.load_checkpoint(str(model_checkpoint))
            trained_model.eval()
            saved_meta = torch.load(str(model_checkpoint), map_location="cpu", weights_only=False).get("metadata", {})
            if "mean" in saved_meta and "std" in saved_meta:
                scaler_mean = np.array(saved_meta["mean"], dtype=np.float32)
                scaler_std = np.array(saved_meta["std"], dtype=np.float32)
        except Exception:
            trained_model = None

    # Build topological graph adjacency and initialize RuntimeResilienceController
    sensor_ids = [str(node["id"]) for node in data.graph["regions"]]
    sensor_index = {s: idx for idx, s in enumerate(sensor_ids)}
    n_nodes = len(sensor_ids)
    adj_tensor = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for edge in data.graph["edges"]:
        u = sensor_index.get(str(edge["from"]))
        v = sensor_index.get(str(edge["to"]))
        if u is not None and v is not None:
            w = float(edge["weight"])
            adj_tensor[u, v] = w
            adj_tensor[v, u] = w
    resilience_controller = RuntimeResilienceController(sensor_ids, adj_tensor, inflation_multiplier=1.5)

    # Initialize FeatureDriftDetector with baseline reference observations
    ref_data: dict[str, np.ndarray] = {}
    for s in sensor_ids:
        if s in data.series and len(data.series[s]) >= 20:
            ref_data[s] = np.array([float(pt["load_MW"]) for pt in data.series[s][:200]], dtype=np.float64)
    drift_detector = FeatureDriftDetector(ref_data, min_batch_size=10)
    latest_drift_report: dict[str, Any] = {
        "overall_status": "STABLE",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "sample_count": len(next(iter(ref_data.values()))) if ref_data else 0,
        "is_drifted": False,
        "metrics_by_feature": {},
        "alerts": [],
    }

    origin = os.environ.get("ENERGRAPH_DEMO_FRONTEND_ORIGIN", "http://127.0.0.1:5173")
    try:
        parsed_origin = urlsplit(origin)
        valid_origin = (
            parsed_origin.scheme == "http"
            and parsed_origin.hostname in {"127.0.0.1", "localhost"}
            and parsed_origin.port is not None
            and 1 <= parsed_origin.port <= 65535
            and origin == f"http://{parsed_origin.hostname}:{parsed_origin.port}"
        )
    except ValueError:
        valid_origin = False
    if not valid_origin:
        raise ValueError("Only an explicit localhost frontend origin is allowed")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
        label = {
            400: "bad_request",
            404: "not_found",
            405: "method_not_allowed",
            422: "invalid_input",
        }.get(error.status_code, "request_error")
        return _error(error.status_code, label, str(error.detail))

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        _request: Request, _error_info: RequestValidationError
    ) -> JSONResponse:
        return _error(422, "invalid_input", "Invalid request parameters")

    token_store_path = os.environ.get("URBANGRAPH_TOKEN_STORE_PATH")
    token_engine = TokenRotationEngine(
        rotation_ttl_seconds=3600.0,
        grace_period_seconds=900.0,
        store_path=token_store_path,
    )
    app.state.token_engine = token_engine

    def resolve_token_role(candidate: str) -> str | None:
        static_admin = (
            os.environ.get("URBANGRAPH_ADMIN_TOKEN")
            or os.environ.get("ENERGRAPH_API_KEY")
            or os.environ.get("URBANGRAPH_API_TOKEN")
        )
        if static_admin and hmac.compare_digest(candidate, static_admin):
            return "admin"
        static_worker = os.environ.get("URBANGRAPH_WORKER_TOKEN")
        if static_worker and hmac.compare_digest(candidate, static_worker):
            return "worker"
        static_reader = os.environ.get("URBANGRAPH_READER_TOKEN")
        if static_reader and hmac.compare_digest(candidate, static_reader):
            return "reader"
        # Check dynamic rotating credentials by role
        for role in ("admin", "worker", "reader"):
            if token_engine.validate_token(role, candidate):
                return role
        return None

    @app.middleware("http")
    async def no_cache_and_telemetry(request: Request, call_next: Any) -> Any:
        corr_id = request.headers.get("X-Correlation-ID") or os.urandom(8).hex()
        static_token = (
            os.environ.get("URBANGRAPH_ADMIN_TOKEN")
            or os.environ.get("ENERGRAPH_API_KEY")
            or os.environ.get("URBANGRAPH_API_TOKEN")
        )
        
        # Authentication and RBAC enforcement on /api/v1/ routes
        if request.url.path.startswith("/api/v1/"):
            auth_header = request.headers.get("Authorization", "")
            token_str = auth_header[7:].strip() if auth_header.startswith("Bearer ") else ""
            role = resolve_token_role(token_str) if token_str else None

            # If authentication is required or if a token was supplied:
            if static_token or token_str:
                if not role:
                    err_resp = _error(401, "unauthorized", "Invalid or missing Bearer token authorization")
                    err_resp.headers["X-Correlation-ID"] = corr_id
                    return err_resp

                # Store validated role on request state
                request.state.role = role

                # RBAC Rule 1: Administrative endpoints require 'admin' role
                if request.url.path.startswith("/api/v1/admin/"):
                    if role != "admin":
                        err_resp = _error(403, "forbidden", f"Role '{role}' is not authorized for administrative operations; requires 'admin'")
                        err_resp.headers["X-Correlation-ID"] = corr_id
                        return err_resp

                # RBAC Rule 2: Neural predictive inference requires 'worker' or 'admin' role
                if request.url.path == "/api/v1/predict":
                    if role not in ("admin", "worker"):
                        err_resp = _error(403, "forbidden", f"Role '{role}' is not authorized for neural predictive inference; requires 'worker' or 'admin'")
                        err_resp.headers["X-Correlation-ID"] = corr_id
                        return err_resp

        start_t = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Correlation-ID"] = corr_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'"
        get_global_metrics().record_request(request.url.path, response.status_code, latency_ms)
        return response

    def region_or_404(region_id: str) -> None:
        if region_id not in data.series:
            raise HTTPException(404, "Unknown synthetic region")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "uptime_seconds": str(get_global_metrics().get_summary()["uptime_seconds"])}

    @app.get("/readyz")
    def readyz() -> dict[str, bool]:
        report = get_global_metrics().check_health()
        return {"ready": report.ready}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return PrometheusExporter.to_prometheus_text(get_global_metrics())

    @app.get("/api/v1/monitoring/health")
    def monitoring_health() -> dict[str, Any]:
        report = get_global_metrics().check_health()
        return report.to_dict()

    @app.get("/api/v1/meta", response_model=Meta)
    def meta() -> Meta:
        rows = data.series["SYN_A"]
        return Meta(
            manifest_sha256=data.manifest_sha256,
            regions=3,
            origins_per_region=7,
            first_utc=str(rows[0]["timestamp_utc"]),
            last_utc=str(rows[-1]["timestamp_utc"]),
            disclaimer="Invented data; no real-grid, calibrated-risk or economic claim.",
        )

    @app.get("/api/v1/regions", response_model=Regions)
    def regions() -> Regions:
        return Regions(
            manifest_sha256=data.manifest_sha256,
            regions=[
                Region(id=str(node["id"]), label=str(node["label"]), origins_utc=list(data.origins))
                for node in data.graph["regions"]
            ],
        )

    @app.get("/api/v1/series", response_model=Series)
    def series(
        region_id: str = Query(min_length=1, max_length=32),
        start_utc: str = Query(),
        end_utc: str = Query(),
    ) -> Series:
        region_or_404(region_id)
        start, end = _time(start_utc), _time(end_utc)
        rows = data.series[region_id]
        if (
            start >= end
            or start < _time(str(rows[0]["timestamp_utc"]))
            or end > _time(str(rows[-1]["timestamp_utc"]))
        ):
            raise HTTPException(400, "Time range must be ordered and inside the synthetic fixture")
        selected = [
            SeriesPoint.model_validate(row)
            for row in rows
            if start <= _time(str(row["timestamp_utc"])) <= end
        ]
        if len(selected) > 2400:
            raise HTTPException(400, "Series range too large")
        return Series(
            manifest_sha256=data.manifest_sha256,
            region_id=region_id,
            points=selected,
            count=len(selected),
        )

    @app.get("/api/v1/forecasts", response_model=Forecasts)
    def forecasts(
        region_id: str = Query(min_length=1, max_length=32), origin_utc: str = Query()
    ) -> Forecasts:
        region_or_404(region_id)
        origin = _time(origin_utc).isoformat().replace("+00:00", "Z")
        rows = data.forecasts.get((region_id, origin))
        if rows is None:
            raise HTTPException(404, "Unknown synthetic forecast origin")
        return Forecasts(
            manifest_sha256=data.manifest_sha256,
            region_id=region_id,
            origin_utc=origin,
            points=[ForecastPoint.model_validate(row) for row in rows],
        )

    @app.get("/api/v1/graph", response_model=Graph)
    def graph() -> Graph:
        nodes = data.graph["regions"]
        edges = data.graph["edges"]
        return Graph(
            manifest_sha256=data.manifest_sha256,
            regions=[GraphNode.model_validate(node) for node in nodes],
            edges=[
                GraphEdge(
                    source=str(edge["from"]), target=str(edge["to"]), weight=float(edge["weight"])
                )
                for edge in edges
            ],
        )

    @app.get("/api/v1/dispatch", response_model=Dispatch)
    def dispatch(
        region_id: str = Query(min_length=1, max_length=32), origin_utc: str = Query()
    ) -> Dispatch:
        region_or_404(region_id)
        origin = _time(origin_utc).isoformat().replace("+00:00", "Z")
        rows = data.dispatch.get((region_id, origin))
        if rows is None:
            raise HTTPException(404, "No illustrative dispatch for this selection")
        return Dispatch(
            manifest_sha256=data.manifest_sha256,
            region_id=region_id,
            origin_utc=origin,
            battery=Battery.model_validate(data.battery),
            points=[DispatchPoint.model_validate(row) for row in rows],
        )

    @app.post("/api/v1/predict", response_model=ResilientPredictResponse)
    @app.post("/api/v1/forecast/resilient", response_model=ResilientPredictResponse)
    def predict_resilient(request: ResilientPredictRequest) -> ResilientPredictResponse:
        region_or_404(request.region_id)
        origin = _time(request.origin_utc).isoformat().replace("+00:00", "Z")
        base_rows = data.forecasts.get((request.region_id, origin))
        if base_rows is None:
            raise HTTPException(404, "Unknown synthetic forecast origin")

        if not (0.50 <= request.confidence_level <= 0.99):
            raise HTTPException(422, "Confidence level must be between 0.50 and 0.99")

        for dropped in request.injected_outages:
            if dropped not in sensor_ids:
                raise HTTPException(422, f"Unknown sensor ID in injected_outages: {dropped}")

        # Construct time-series sensor input tensor: shape (1, T=12, N, 1)
        history_len = 12
        x_np = np.zeros((1, history_len, n_nodes, 1), dtype=np.float32)
        for idx, s in enumerate(sensor_ids):
            rows = data.series.get(s, [])
            preceding = [r for r in rows if _time(str(r["timestamp_utc"])) <= _time(origin)]
            if len(preceding) >= history_len:
                selected_hist = preceding[-history_len:]
            else:
                selected_hist = rows[:history_len]
            vals = [float(pt["load_MW"]) for pt in selected_hist]
            if len(vals) < history_len:
                vals = vals + [vals[-1] if vals else 100.0] * (history_len - len(vals))
            x_np[0, :, idx, 0] = vals

        # If custom_series is provided, overwrite values
        if request.custom_series:
            for item in request.custom_series:
                s_id = str(item.get("sensor_id", request.region_id))
                if s_id in sensor_index and "values" in item:
                    custom_vals = [float(v) for v in item["values"][:history_len]]
                    if len(custom_vals) == history_len:
                        x_np[0, :, sensor_index[s_id], 0] = custom_vals

        # Inject sensor outages (NaN) for dropped sensors
        for dropped in request.injected_outages:
            d_idx = sensor_index[dropped]
            x_np[0, :, d_idx, 0] = np.nan

        x_tensor = torch.from_numpy(x_np)

        def base_forecaster_fn(imputed_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if trained_model is not None:
                with torch.no_grad():
                    m_inp = imputed_tensor.clone()
                    if scaler_mean is not None and scaler_std is not None:
                        mean_t = torch.tensor(scaler_mean, dtype=torch.float32).view(1, 1, n_nodes, 1)
                        std_t = torch.tensor(scaler_std, dtype=torch.float32).view(1, 1, n_nodes, 1)
                        m_inp = (m_inp - mean_t) / std_t

                    num_graphs = getattr(trained_model.config, "num_graphs", 3)
                    identity_adj = torch.eye(n_nodes, dtype=torch.float32)
                    uniform_adj = torch.ones((n_nodes, n_nodes), dtype=torch.float32) / max(1, n_nodes)
                    g_list = [adj_tensor, identity_adj, uniform_adj][:num_graphs]
                    while len(g_list) < num_graphs:
                        g_list.append(identity_adj)
                    g_adjs = torch.stack(g_list, dim=0)
                    node_m = mask.unsqueeze(0)
                    preds = trained_model(m_inp, g_adjs, node_mask=node_m)  # (1, H, N)
                    if preds.shape[1] != 24:
                        preds = torch.nn.functional.interpolate(
                            preds.permute(0, 2, 1), size=24, mode="linear", align_corners=True
                        ).permute(0, 2, 1)
                    return preds

            preds = torch.zeros((1, 24, n_nodes), dtype=torch.float32)
            for idx, s in enumerate(sensor_ids):
                s_rows = data.forecasts.get((s, origin))
                if s_rows:
                    base_pts = [float(pt["point_MW"]) for pt in s_rows[:24]]
                    preds[0, :, idx] = torch.tensor(base_pts, dtype=torch.float32)
                else:
                    mean_val = float(imputed_tensor[0, :, idx, 0].mean().item())
                    preds[0, :, idx] = torch.full((24,), max(10.0, mean_val), dtype=torch.float32)

                if mask[idx].item() == 0.0:
                    orig_mean = max(1.0, float(data.series[s][0]["load_MW"]))
                    imputed_mean = max(1.0, float(imputed_tensor[0, :, idx, 0].mean().item()))
                    scale_ratio = min(2.0, max(0.5, imputed_mean / orig_mean))
                    preds[0, :, idx] *= scale_ratio
            return preds

        predictions, uncertainty_scales, outage_rep = resilience_controller.process(x_tensor, base_forecaster_fn)

        curr_batch: dict[str, np.ndarray] = {
            s: x_tensor[0, :, idx, 0].numpy().astype(np.float64) for idx, s in enumerate(sensor_ids)
        }
        drift_rep = drift_detector.check_drift(curr_batch)
        if drift_rep.is_drifted:
            get_global_metrics().record_drift_alert()
        nonlocal latest_drift_report
        latest_drift_report = drift_rep.to_dict()

        get_global_metrics().record_outage(len(outage_rep.dropped_sensors))

        target_idx = sensor_index[request.region_id]
        inflation = float(uncertainty_scales[0, 0, target_idx].item())
        is_imputed = request.region_id in outage_rep.dropped_sensors

        response_points: list[ResilientForecastPoint] = []
        for h_idx, base_pt in enumerate(base_rows):
            h_num = h_idx + 1
            raw_pt = float(base_pt["point_MW"])
            adjusted_pt = float(predictions[0, h_idx, target_idx].item())
            nom_lower = float(base_pt["lower_MW"])
            nom_upper = float(base_pt["upper_MW"])
            half_width = max(1.0, (nom_upper - nom_lower) / 2.0)
            inflated_half = half_width * inflation
            adj_lower = max(0.0, adjusted_pt - inflated_half)
            adj_upper = adjusted_pt + inflated_half

            response_points.append(
                ResilientForecastPoint(
                    horizon_hours=h_num,
                    target_utc=str(base_pt["target_utc"]),
                    point_MW=round(adjusted_pt, 2),
                    truth_MW=float(base_pt["truth_MW"]),
                    lower_MW=round(adj_lower, 2),
                    upper_MW=round(adj_upper, 2),
                    raw_point_MW=round(raw_pt, 2),
                    uncertainty_inflation_factor=round(inflation, 4),
                    is_imputed=is_imputed,
                )
            )

        return ResilientPredictResponse(
            schema_version="eg008-resilient-inference-v1",
            region_id=request.region_id,
            origin_utc=origin,
            outage_report=outage_rep.to_dict(),
            points=response_points,
            drift_detected=drift_rep.is_drifted,
            drift_status=drift_rep.overall_status,
            model_id="urbangraphformer-trained-v1" if trained_model is not None else "urbangraph-resilience-v1",
        )

    @app.post("/api/v1/simulate", response_model=SimulationResponse)
    def simulate_grid(req: SimulationRequest) -> SimulationResponse:
        nodes = [str(r["id"]) for r in data.graph["regions"]]
        n = len(nodes)
        node_idx = {name: i for i, name in enumerate(nodes)}

        # Base net power injection per bus (MW): Slack bus balances, load buses consume
        P_inj = np.zeros(n, dtype=np.float64)
        for i, name in enumerate(nodes):
            scale = req.load_scale if name == req.region_id else 1.0
            P_inj[i] = -100.0 * scale
        P_inj[0] = -np.sum(P_inj[1:])  # Slack generation balances total demand

        # Susceptance matrix B
        B = np.zeros((n, n), dtype=np.float64)
        branches = []
        for edge in data.graph["edges"]:
            u_name = str(edge["from"])
            v_name = str(edge["to"])
            u = node_idx.get(u_name)
            v = node_idx.get(v_name)
            if u is None or v is None:
                continue
            corridor_id = f"{u_name}-{v_name}"
            corridor_rev = f"{v_name}-{u_name}"
            is_out = (corridor_id in req.line_outages) or (corridor_rev in req.line_outages)
            weight = float(edge["weight"])
            susceptance = 0.0 if is_out else max(weight * 10.0, 1.0)
            branches.append({
                "source": u_name,
                "target": v_name,
                "susceptance": susceptance,
                "is_outage": is_out,
                "rating_MW": 120.0,
            })
            if not is_out:
                B[u, v] -= susceptance
                B[v, u] -= susceptance
                B[u, u] += susceptance
                B[v, v] += susceptance

        # Check connectivity of non-slack buses
        theta = np.zeros(n, dtype=np.float64)
        violations: list[str] = []
        feasible = True
        status = "CONVERGED_FEASIBLE"

        # Solve DC power flow: theta_reduced = inv(B_reduced) * P_reduced
        B_red = B[1:, 1:]
        P_red = P_inj[1:]
        try:
            cond = np.linalg.cond(B_red)
            if cond > 1e12 or np.isnan(cond):
                feasible = False
                status = "INFEASIBLE_ISLANDING"
                violations.append("Grid topological islanding detected: line outages disconnected one or more regional buses.")
            else:
                theta_red = np.linalg.solve(B_red, P_red)
                theta[1:] = theta_red
        except Exception:
            feasible = False
            status = "INFEASIBLE_SINGULAR"
            violations.append("Susceptance matrix singular: grid topology lacks reference path to slack generator.")

        # Compute branch flows P_ij = b_ij * (theta_i - theta_j)
        branch_flows = []
        for br in branches:
            u = node_idx[br["source"]]
            v = node_idx[br["target"]]
            if br["is_outage"] or not feasible:
                flow = 0.0
            else:
                flow = float(br["susceptance"] * (theta[u] - theta[v]))
            rating = br["rating_MW"]
            loading_pct = float(abs(flow) / rating * 100.0) if rating > 0 else 0.0
            overload = abs(flow) > rating
            if overload:
                violations.append(f"Thermal rating violation on {br['source']}-{br['target']}: {abs(flow):.1f} MW exceeds {rating:.1f} MW limit ({loading_pct:.1f}%).")
                feasible = False
                status = "INFEASIBLE_OVERLOAD"
            branch_flows.append({
                "corridor": f"{br['source']}-{br['target']}",
                "flow_MW": round(flow, 2),
                "rating_MW": rating,
                "loading_pct": round(loading_pct, 1),
                "is_outage": br["is_outage"],
                "is_overloaded": overload,
            })

        bus_angles = {name: round(float(theta[i]), 4) for i, name in enumerate(nodes)}

        # Battery Optimization scenario if requested
        batt_schedule = None
        cost_cu = None
        if req.scenario_type == "battery_optimization":
            try:
                load_series = [float(pt["load_MW"]) for pt in data.series.get(req.region_id, [])[:24]]
                if len(load_series) < 24:
                    load_series = [100.0 * req.load_scale] * 24
                loads_arr = np.array(load_series, dtype=np.float64) * req.load_scale
                b_spec = BatterySpecification(
                    training_median_MW=float(np.median(loads_arr)),
                    capacity_ratio=req.battery_capacity_ratio,
                    roundtrip_efficiency=req.battery_efficiency,
                )
                tariff = SyntheticTariff(peak=req.tariff_peak, degradation=req.tariff_degradation)
                sched, _ = optimize(loads_arr[None, :], b_spec, tariff)
                chk = check_schedule(sched, b_spec, loads_arr)
                rcost = realized_cost(sched, b_spec, tariff, loads_arr)
                soc_list = chk.get("soc_MWh", [0.0] * 25)
                batt_schedule = [
                    {
                        "hour": h,
                        "charge_MW": round(float(sched.charge[h]), 2),
                        "discharge_MW": round(float(sched.discharge[h]), 2),
                        "soc_MWh": round(float(soc_list[h + 1]), 2) if h + 1 < len(soc_list) else 0.0,
                    }
                    for h in range(24)
                ]
                cost_cu = {
                    "energy_cost_CU": round(float(rcost["energy_CU"]), 2),
                    "peak_demand_charge_CU": round(float(rcost["peak_CU"]), 2),
                    "battery_degradation_CU": round(float(rcost["degradation_CU"]), 2),
                    "net_cost_CU": round(float(rcost["total_CU"]), 2),
                    "peak_import_MW": round(float(chk["peak_import_MW"]), 2),
                }
            except Exception as e:
                violations.append(f"Battery optimization failed: {str(e)}")

        return SimulationResponse(
            schema_version="eg014-grid-simulation-v1",
            scenario_type=req.scenario_type,
            region_id=req.region_id,
            feasible=feasible,
            status=status,
            branch_flows=branch_flows,
            bus_angles=bus_angles,
            constraint_violations=violations,
            battery_schedule=batt_schedule,
            realized_cost_CU=cost_cu,
            advisory_notice="Local simulation only. No physical SCADA actuation.",
        )

    @app.get("/api/v1/monitoring/drift", response_model=DriftMonitoringStatus)
    def monitoring_drift_status() -> DriftMonitoringStatus:
        summary = get_global_metrics().get_summary()
        return DriftMonitoringStatus(
            schema_version="eg021-observability-v1",
            overall_status=latest_drift_report.get("overall_status", "STABLE"),
            drift_alerts_total=summary["drift_alerts_total"],
            sensor_outages_total=summary["sensor_outages_total"],
            reference_features=list(ref_data.keys()),
            sample_count=len(next(iter(ref_data.values()))) if ref_data else 0,
            latest_report=latest_drift_report,
        )

    @app.post("/api/v1/monitoring/drift/check")
    def monitoring_drift_check(req: DriftCheckRequest) -> dict[str, Any]:
        curr = {k: np.array(v, dtype=np.float64) for k, v in req.features.items()}
        rep = drift_detector.check_drift(curr)
        if rep.is_drifted:
            get_global_metrics().record_drift_alert()
        nonlocal latest_drift_report
        latest_drift_report = rep.to_dict()
        return rep.to_dict()

    @app.post("/api/v1/admin/rotate-token")
    def admin_rotate_token(req: dict[str, Any]) -> dict[str, Any]:
        role = req.get("role")
        if role not in ("admin", "worker", "reader"):
            raise HTTPException(422, "Invalid role; must be 'admin', 'worker', or 'reader'")
        try:
            new_tok = token_engine.rotate_token(role)
        except DataError as e:
            raise HTTPException(400, str(e))
        latest_audit = token_engine.get_audit_log()[-1]
        return {
            "status": "rotated",
            "role": role,
            "new_token": new_tok,
            "audit_entry": latest_audit,
        }

    @app.post("/api/v1/admin/revoke-token")
    def admin_revoke_token(req: dict[str, Any]) -> dict[str, Any]:
        role = req.get("role")
        candidate_token = req.get("token")
        if role not in ("admin", "worker", "reader") or not candidate_token:
            raise HTTPException(422, "Missing or invalid role or token")
        success = token_engine.revoke_token(role, candidate_token)
        if not success:
            raise HTTPException(404, "Token not found for revocation")
        latest_audit = token_engine.get_audit_log()[-1]
        return {
            "status": "revoked",
            "role": role,
            "audit_entry": latest_audit,
        }

    @app.get("/api/v1/admin/tokens")
    def admin_list_tokens() -> dict[str, Any]:
        return {
            "status": "ok",
            "inventory": token_engine.get_active_inventory(),
            "audit_log": token_engine.get_audit_log(),
        }

    return app
