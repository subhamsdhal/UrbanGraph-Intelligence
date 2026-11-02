"""Read-only API contract checks against approved synthetic fixtures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from energraph.demo.api import create_app
from energraph.demo.fixtures import FixtureError


@pytest.fixture(scope="module")
def client() -> TestClient:
    root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("ENERGRAPH_DEMO_PACKAGE")
    app = create_app(Path(configured) if configured else root / "tests" / "fixtures" / "eg006")
    return TestClient(app)


def test_meta_regions_graph_and_openapi(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    meta = client.get("/api/v1/meta")
    assert meta.status_code == 200
    assert meta.json()["dataset_kind"] == "synthetic"
    regions = client.get("/api/v1/regions").json()["regions"]
    assert [region["id"] for region in regions] == ["SYN_A", "SYN_B", "SYN_C"]
    assert all(len(region["origins_utc"]) == 7 for region in regions)
    graph = client.get("/api/v1/graph").json()
    assert graph["coordinate_system"] == "synthetic_unit_square"
    assert len(graph["regions"]) == 3
    assert "/api/v1/forecasts" in client.get("/openapi.json").json()["paths"]


def test_all_forecast_populations(client: TestClient) -> None:
    for region in client.get("/api/v1/regions").json()["regions"]:
        for origin in region["origins_utc"]:
            response = client.get(
                "/api/v1/forecasts", params={"region_id": region["id"], "origin_utc": origin}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["interval_kind"] == "illustrative_uncalibrated"
            assert body["unit"] == "MW"
            assert [point["horizon_hours"] for point in body["points"]] == list(range(1, 25))


def test_series_dispatch_and_negative_contracts(client: TestClient) -> None:
    meta = client.get("/api/v1/meta").json()
    series = client.get(
        "/api/v1/series",
        params={"region_id": "SYN_A", "start_utc": meta["first_utc"], "end_utc": meta["last_utc"]},
    )
    assert series.status_code == 200
    assert series.json()["count"] == 2400
    origins = client.get("/api/v1/regions").json()["regions"][0]["origins_utc"]
    dispatch = client.get(
        "/api/v1/dispatch", params={"region_id": "SYN_A", "origin_utc": origins[0]}
    )
    assert dispatch.status_code == 200
    assert len(dispatch.json()["points"]) == 24
    for endpoint in ("/api/v1/series", "/api/v1/forecasts", "/api/v1/dispatch"):
        bad = client.get(
            endpoint,
            params={
                "region_id": "REAL_DE",
                "origin_utc": origins[0],
                "start_utc": meta["first_utc"],
                "end_utc": meta["last_utc"],
            },
        )
        assert bad.status_code == 404
        assert bad.json()["error"]["code"] == "not_found"
    invalid = client.get(
        "/api/v1/forecasts", params={"region_id": "SYN_A", "origin_utc": "not-a-time"}
    )
    assert invalid.status_code == 422
    mutation = client.post("/api/v1/regions")
    assert mutation.status_code == 405
    assert mutation.json()["error"]["code"] == "method_not_allowed"


def test_corrupt_package_prevents_app_creation(tmp_path: Path) -> None:
    approved = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eg006"
    package = tmp_path / "package"
    shutil.copytree(approved, package)
    (package / "fixtures" / "graph.json").write_bytes(b"corrupt")
    with pytest.raises(FixtureError):
        create_app(package)


def test_alternative_frontend_origin_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENERGRAPH_DEMO_FRONTEND_ORIGIN", "http://127.0.0.1:15173")
    package = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eg006"
    with TestClient(create_app(package)) as local_client:
        response = local_client.options(
            "/api/v1/meta",
            headers={"Origin": "http://127.0.0.1:15173", "Access-Control-Request-Method": "GET"},
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:15173"


def test_external_frontend_origin_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENERGRAPH_DEMO_FRONTEND_ORIGIN", "http://example.com:15173")
    package = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eg006"
    with pytest.raises(ValueError, match="localhost"):
        create_app(package)


def test_monitoring_and_healthz_routes(client: TestClient) -> None:
    res_healthz = client.get("/healthz")
    assert res_healthz.status_code == 200
    assert res_healthz.json()["status"] == "ok"

    res_readyz = client.get("/readyz")
    assert res_readyz.status_code == 200
    assert res_readyz.json()["ready"] is True

    res_metrics = client.get("/metrics")
    assert res_metrics.status_code == 200
    assert "urbangraph_requests_total" in res_metrics.text

    res_mon = client.get("/api/v1/monitoring/health")
    assert res_mon.status_code == 200
    assert res_mon.json()["status"] in {"HEALTHY", "DEGRADED"}


def test_resilient_predict_normal_and_outages(client: TestClient) -> None:
    origins = client.get("/api/v1/regions").json()["regions"][0]["origins_utc"]
    valid_origin = origins[0]

    # 1. Normal prediction without outages
    req_normal = {
        "region_id": "SYN_A",
        "origin_utc": valid_origin,
        "injected_outages": [],
        "confidence_level": 0.90,
    }
    res_normal = client.post("/api/v1/predict", json=req_normal)
    assert res_normal.status_code == 200
    data_normal = res_normal.json()
    assert data_normal["schema_version"] == "eg008-resilient-inference-v1"
    assert data_normal["region_id"] == "SYN_A"
    assert data_normal["outage_report"]["degradation_level"] == "NORMAL"
    assert data_normal["outage_report"]["uncertainty_inflation_factor"] == 1.0
    assert len(data_normal["points"]) == 24
    assert not any(p["is_imputed"] for p in data_normal["points"])

    # 2. Moderate outage (1 sensor down: SYN_B)
    req_mod = {
        "region_id": "SYN_A",
        "origin_utc": valid_origin,
        "injected_outages": ["SYN_B"],
        "confidence_level": 0.90,
    }
    res_mod = client.post("/api/v1/predict", json=req_mod)
    assert res_mod.status_code == 200
    data_mod = res_mod.json()
    assert data_mod["outage_report"]["degradation_level"] in {"MODERATE_OUTAGE", "SEVERE_OUTAGE"}
    assert data_mod["outage_report"]["uncertainty_inflation_factor"] > 1.0
    assert data_mod["outage_report"]["dropped_sensors"] == ["SYN_B"]

    # 3. Severe outage on target sensor itself (SYN_A down -> requires harmonic spatial imputation)
    req_target_down = {
        "region_id": "SYN_A",
        "origin_utc": valid_origin,
        "injected_outages": ["SYN_A"],
        "confidence_level": 0.90,
    }
    res_td = client.post("/api/v1/predict", json=req_target_down)
    assert res_td.status_code == 200
    data_td = res_td.json()
    assert data_td["outage_report"]["imputation_applied"] in {"HARMONIC_SPATIAL", "HYBRID"}
    assert all(p["is_imputed"] for p in data_td["points"])
    # Target dropped node receives higher uncertainty inflation
    assert data_td["points"][0]["uncertainty_inflation_factor"] > data_mod["points"][0]["uncertainty_inflation_factor"]
    # Verify bounds are valid and positive
    assert all(p["upper_MW"] > p["lower_MW"] >= 0.0 for p in data_td["points"])

    # 4. Alias route /api/v1/forecast/resilient
    res_alias = client.post("/api/v1/forecast/resilient", json=req_normal)
    assert res_alias.status_code == 200
    assert res_alias.json()["region_id"] == "SYN_A"


def test_resilient_predict_negative_inputs(client: TestClient) -> None:
    origins = client.get("/api/v1/regions").json()["regions"][0]["origins_utc"]
    valid_origin = origins[0]

    # Unknown region -> 404
    bad_region = client.post("/api/v1/predict", json={"region_id": "NON_EXISTENT", "origin_utc": valid_origin})
    assert bad_region.status_code == 404
    assert bad_region.json()["error"]["code"] == "not_found"

    # Unknown origin -> 404
    bad_origin = client.post("/api/v1/predict", json={"region_id": "SYN_A", "origin_utc": "2099-01-01T00:00:00Z"})
    assert bad_origin.status_code == 404

    # Invalid timestamp format -> 422
    bad_time = client.post("/api/v1/predict", json={"region_id": "SYN_A", "origin_utc": "invalid-time-format"})
    assert bad_time.status_code == 422

    # Invalid confidence level -> 422
    bad_conf = client.post(
        "/api/v1/predict",
        json={"region_id": "SYN_A", "origin_utc": valid_origin, "confidence_level": 0.10},
    )
    assert bad_conf.status_code == 422

    # Unknown sensor in injected_outages -> 422
    bad_sensor = client.post(
        "/api/v1/predict",
        json={"region_id": "SYN_A", "origin_utc": valid_origin, "injected_outages": ["GHOST_SENSOR"]},
    )
    assert bad_sensor.status_code == 422


def test_runtime_feature_drift_monitoring(client: TestClient) -> None:
    # 1. Check baseline status
    status_res = client.get("/api/v1/monitoring/drift")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["schema_version"] == "eg021-observability-v1"
    assert "SYN_A" in status_data["reference_features"]

    # 2. Check small sample suppression (below min_batch_size=10)
    small_check = client.post(
        "/api/v1/monitoring/drift/check",
        json={"features": {"SYN_A": [100.0, 102.0, 101.0]}},
    )
    assert small_check.status_code == 200
    assert small_check.json()["is_drifted"] is False
    assert "below threshold" in small_check.json()["alerts"][0]["message"]

    # 3. Simulate severe distribution shift (e.g. massive 10x values across 25 samples)
    drifted_vals = [float(v) for v in np.random.normal(1500.0, 20.0, size=25)]
    drift_check = client.post(
        "/api/v1/monitoring/drift/check",
        json={"features": {"SYN_A": drifted_vals}},
    )
    assert drift_check.status_code == 200
    res_data = drift_check.json()
    assert res_data["is_drifted"] is True
    assert res_data["overall_status"] in {"MODERATE_DRIFT", "SEVERE_DRIFT"}
    assert res_data["metrics_by_feature"]["SYN_A"]["psi"] > 0.25

    # 4. Verify Prometheus metrics reflects the drift alert counter
    metrics_res = client.get("/metrics")
    assert "urbangraph_drift_alerts_total" in metrics_res.text


def test_trained_urbangraphformer_model_inference() -> None:
    root = Path(__file__).resolve().parents[2]
    checkpoint_path = root / "tests" / "fixtures" / "checkpoints" / "urbangraphformer_synth.pt"
    package = root / "tests" / "fixtures" / "eg006"
    assert checkpoint_path.exists(), "Trained UrbanGraphFormer checkpoint must exist"

    app = create_app(package, model_checkpoint=checkpoint_path)
    with TestClient(app) as local_client:
        origins = local_client.get("/api/v1/regions").json()["regions"][0]["origins_utc"]
        res = local_client.post(
            "/api/v1/predict",
            json={
                "region_id": "SYN_A",
                "origin_utc": origins[0],
                "injected_outages": [],
                "confidence_level": 0.90,
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["model_id"] == "urbangraphformer-trained-v1"
        assert len(body["points"]) == 24
        # Verify predictions are non-trivial positive floats from the neural model
        assert all(p["point_MW"] > 0 for p in body["points"])
        # Compare against dropped sensor imputation
        res_drop = local_client.post(
            "/api/v1/predict",
            json={
                "region_id": "SYN_A",
                "origin_utc": origins[0],
                "injected_outages": ["SYN_A"],
                "confidence_level": 0.90,
            },
        )
        assert res_drop.status_code == 200
        body_drop = res_drop.json()
        assert body_drop["model_id"] == "urbangraphformer-trained-v1"
        assert all(p["is_imputed"] for p in body_drop["points"])



