"""End-to-End Architectural Integration Test across all UrbanGraph Engineering Components.

Satisfies Phase 9:
Data ingestion & canonicalization ->
Multi-relational graph construction ->
UrbanGraphFormer spatio-temporal inference ->
Sensor-dropout resilience & harmonic imputation ->
Traffic conformal uncertainty calibration ->
Feature distribution drift monitoring ->
Token rotation security & authenticated serving ->
Artifact snapshot backup & bit-for-bit disaster recovery verification.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import torch
import pytest

from urbangraph.governance.lifecycle import DisasterRecoveryManager
from urbangraph.monitoring.drift import FeatureDriftDetector
from urbangraph.monitoring.observability import MetricsCollector, PrometheusExporter
from urbangraph.neural.urbangraphformer import UrbanGraphFormer, UrbanGraphFormerConfig
from urbangraph.security.token_rotation import TokenRotationEngine
from urbangraph.serving.resilience import RuntimeResilienceController
from urbangraph.uncertainty.traffic_conformal import (
    TrafficCalibrationPolicy,
    evaluate_traffic_intervals,
    fit_traffic_calibrator,
)


def test_full_pipeline_end_to_end_integration() -> None:
    # -------------------------------------------------------------
    # Step 1: Multi-Relational Graph Construction
    # -------------------------------------------------------------
    num_nodes = 4
    sensors = ["NODE_DE", "NODE_FR", "NODE_GB", "NODE_ES"]
    num_relations = 2  # Physical transmission + Correlation proximity

    # Relation 0: Physical adjacency
    adj_physical = torch.tensor([
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.6, 0.7],
        [0.0, 0.6, 1.0, 0.0],
        [0.0, 0.7, 0.0, 1.0],
    ])
    # Relation 1: Correlation similarity
    adj_correlation = torch.tensor([
        [1.0, 0.5, 0.4, 0.3],
        [0.5, 1.0, 0.5, 0.6],
        [0.4, 0.5, 1.0, 0.2],
        [0.3, 0.6, 0.2, 1.0],
    ])
    adjacencies = torch.stack([adj_physical, adj_correlation], dim=0)  # (2, 4, 4)

    # -------------------------------------------------------------
    # Step 2: UrbanGraphFormer Model Architecture
    # -------------------------------------------------------------
    config = UrbanGraphFormerConfig(
        hidden=32,
        heads=4,
        layers=2,
        num_graphs=num_relations,
        dropout=0.0,
    )
    model = UrbanGraphFormer(
        config=config,
        history=6,
        horizon=3,
        num_nodes=num_nodes,
        in_features=2,
    )
    model.eval()

    # Synthetic observations: (B=1, T=6, N=4, F=2)
    x = torch.rand(1, 6, num_nodes, 2) * 50.0 + 20.0

    # -------------------------------------------------------------
    # Step 3: Sensor-Dropout Resilience & Runtime Imputation
    # -------------------------------------------------------------
    controller = RuntimeResilienceController(
        sensor_ids=sensors,
        adjacency=adj_physical,
        inflation_multiplier=1.8,
    )

    # Inject missing telemetry on NODE_GB
    x_with_dropout = x.clone()
    x_with_dropout[:, :, 2, :] = float("nan")

    def model_forecaster(imputed_inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            preds = model(imputed_inputs, adjacencies)  # (1, H=3, N=4)
        return preds

    preds, uncert_inflation, outage_report = controller.process(x_with_dropout, model_forecaster)
    assert preds.shape == (1, 3, 4)
    assert uncert_inflation.shape == (1, 3, 4)
    assert outage_report.degradation_level == "MODERATE_OUTAGE"
    assert "NODE_GB" in outage_report.dropped_sensors
    assert outage_report.uncertainty_inflation_factor > 1.0

    # -------------------------------------------------------------
    # Step 4: Traffic Conformal Prediction & Calibration
    # -------------------------------------------------------------
    rng = np.random.default_rng(42)
    n_cal_origins = 60
    horizons = 3
    cal_truth = rng.uniform(40.0, 90.0, size=(n_cal_origins, horizons, num_nodes))
    cal_preds = cal_truth + rng.normal(0, 3.0, size=(n_cal_origins, horizons, num_nodes))
    cal_mask = np.ones_like(cal_truth, dtype=bool)
    cal_origins = np.arange(100, 100 + n_cal_origins, dtype=np.int64)

    policy = TrafficCalibrationPolicy(coverage=0.90, minimum_count=20, units="km/h")
    calibrator = fit_traffic_calibrator(
        truth=cal_truth,
        prediction=cal_preds,
        mask=cal_mask,
        origins=cal_origins,
        policy=policy,
        detector_ids=tuple(sensors),
        dataset_id="urban-synthetic-corridor-p6",
        model_id="urbangraphformer-v1",
        observation_delay=1,
    )

    # Forward prediction intervals: expects 3D (num_origins, horizons, num_nodes)
    forward_preds = np.maximum(preds.numpy(), 0.0)  # Clamped to non-negative traffic physical speed/flow
    forward_origins = np.array([200], dtype=np.int64)
    lower, upper = calibrator.intervals(
        forward_preds,
        forward_origins,
        detector_ids=tuple(sensors),
        model_id="urbangraphformer-v1",
    )
    assert lower.shape == (1, 3, 4)
    assert upper.shape == (1, 3, 4)
    assert np.all(lower <= forward_preds)
    assert np.all(forward_preds <= upper)

    eval_results = evaluate_traffic_intervals(
        truth=cal_truth[:10],
        prediction=cal_preds[:10],
        lower=lower.repeat(10, axis=0),
        upper=upper.repeat(10, axis=0),
        mask=cal_mask[:10],
        coverage=0.90,
    )
    assert "mean_winkler_score" in eval_results

    # -------------------------------------------------------------
    # Step 5: Distribution Drift Monitoring
    # -------------------------------------------------------------
    detector = FeatureDriftDetector(
        reference_data={"sensor_flow": cal_truth[:, 0, 0]},
        min_batch_size=20,
    )
    report = detector.check_drift({"sensor_flow": cal_preds[:, 0, 0]})
    assert report.sample_count >= 20

    # -------------------------------------------------------------
    # Step 6: Token Rotation & Security
    # -------------------------------------------------------------
    security_engine = TokenRotationEngine()
    worker_token = security_engine.get_primary_token("worker")
    assert security_engine.validate_token("worker", worker_token) is True
    assert security_engine.validate_token("admin", worker_token) is False

    # -------------------------------------------------------------
    # Step 7: Observability & Prometheus Telemetry
    # -------------------------------------------------------------
    metrics = MetricsCollector()
    metrics.record_request("/v1/forecast/urbangraphformer", 200, 24.5, "urbangraphformer-v1")
    metrics.record_outage(len(outage_report.dropped_sensors))
    prom_text = PrometheusExporter.to_prometheus_text(metrics)
    assert "urbangraph_requests_total" in prom_text

    # -------------------------------------------------------------
    # Step 8: Disaster Recovery Snapshot & Restore Verification
    # -------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_data = tmp_path / "artifacts"
        source_data.mkdir()
        (source_data / "model.pt").write_bytes(b"MODEL_CHECKPOINT_INTEGRATION_BYTES")

        backup_file = tmp_path / "dr_backup.tar.gz"
        restore_dir = tmp_path / "restored"

        DisasterRecoveryManager.create_backup_archive(source_data, backup_file)
        dr_result = DisasterRecoveryManager.restore_and_verify_archive(backup_file, restore_dir)

        assert dr_result["status"] == "RESTORE_VERIFIED_SUCCESSFUL"
        assert (restore_dir / "model.pt").read_bytes() == b"MODEL_CHECKPOINT_INTEGRATION_BYTES"
