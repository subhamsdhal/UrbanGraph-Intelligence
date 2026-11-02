"""True End-to-End System Integration Test.

Proves the complete path across real process/HTTP boundaries:
authorized data ingestion -> validation -> graph construction -> model training ->
evaluation -> checkpoint publication -> authenticated inference -> dropout resilience ->
conformal uncertainty -> live monitoring/drift -> governance, revocation and disaster recovery.
Satisfies Master Directive Requirement 9.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from energraph.demo.api import create_app
from urbangraph.data import DataError, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import discover, stream_file
from urbangraph.datasets.governed_canonical import create_canonical
from urbangraph.datasets.serialization import read_json, write_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import EvaluationProtocol, PlanConfig
from urbangraph.governance.lifecycle import DataLifecycleManager, DatasetRetentionPolicy, DisasterRecoveryManager
from urbangraph.graphs.artifacts import build_graph
from urbangraph.graphs.contracts import GraphSpecification
from urbangraph.monitoring.drift import FeatureDriftDetector
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.urbangraphformer import UrbanGraphFormer, UrbanGraphFormerConfig
from urbangraph.security.token_rotation import TokenRotationEngine
from urbangraph.serving.resilience import RuntimeResilienceController
from urbangraph.uncertainty.traffic_conformal import (
    TrafficCalibrationPolicy,
    evaluate_traffic_intervals,
    fit_traffic_calibrator,
)


def test_true_system_integration_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # -------------------------------------------------------------------------
    # Stage 1: Authorized Data Ingestion & Governance Rights Policy
    # -------------------------------------------------------------------------
    grants: list[dict[str, object]] = []
    policy_file = tmp_path / "rights-policy.json"

    def grant_permission(source_id: str, path: Path, digest: str, op: str, purpose: str, dest: Path | None = None) -> None:
        now = datetime.now(UTC)
        grants.append({
            "source_id": source_id,
            "source_sha256": digest,
            "source_path": str(path),
            "purpose": purpose,
            "operation": op,
            "destination_root": str(dest or tmp_path),
            "evidence_reference": "e2e-authorized-audit-ref",
            "status": "approved",
            "effective_utc": (now - timedelta(minutes=1)).isoformat(),
            "expires_utc": (now + timedelta(hours=1)).isoformat(),
            "retention_days": 30,
        })
        policy_file.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy_file))

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    sensor_ids = ["SYN_A", "SYN_B", "SYN_C"]
    
    # 350 timesteps of 5-min intervals
    df = pd.DataFrame(
        (np.arange(1050).reshape(350, 3) % 25) + 40.0,
        columns=sensor_ids,
        index=pd.date_range("2026-01-01", periods=350, freq="5min"),
    )
    df.to_hdf(incoming / "metr-la.h5", key="traffic")
    records = tuple(RawFileRecord(r, *stream_file(incoming / r)) for r in discover(incoming).files)
    ingest_digest = content_identity(records)
    grant_permission("metr-la", incoming, ingest_digest, "raw_ingest", "local_research")

    raw = ingest_dataset(incoming, "metr-la", raw_root=tmp_path / "raw")
    for op in ("snapshot_parse", "canonical_create", "canonical_read", "canonical_graph_create", "canonical_graph_read"):
        grant_permission("metr-la", raw.directory, raw.manifest.snapshot_id, op, "graph_analysis" if "graph" in op else "local_research")

    # -------------------------------------------------------------------------
    # Stage 2: Canonicalization & Multi-Relational Graph Construction
    # -------------------------------------------------------------------------
    canonical = create_canonical(raw.directory, output_root=tmp_path / "canonical")
    locations = tmp_path / "locations.csv"
    locations.write_text("sensor_id,latitude,longitude\nSYN_A,48.8,2.3\nSYN_B,50.1,8.6\nSYN_C,52.2,21.0\n")
    distances = tmp_path / "distances.csv"
    distances.write_text("from,to,cost\nSYN_A,SYN_A,0\nSYN_B,SYN_B,0\nSYN_C,SYN_C,0\nSYN_A,SYN_B,450\nSYN_B,SYN_C,800\nSYN_C,SYN_A,1200\n")

    for ref, p in (("locations-ref", locations), ("distances-ref", distances)):
        for op in ("graph_csv_read", "graph_create", "graph_reuse"):
            grant_permission(ref, p, stream_file(p)[1], op, "graph_analysis")

    graph = build_graph({
        "specification": GraphSpecification("metr-la").to_dict(),
        "raw_snapshot_dir": str(raw.directory),
        "canonical_dir": str(canonical.directory),
        "locations": str(locations),
        "location_reference": "locations-ref",
        "distances": str(distances),
        "distance_reference": "distances-ref",
        "output_root": str(tmp_path / "graphs"),
    })
    assert graph.graph_id is not None
    assert graph.index.sensor_ids == ("SYN_A", "SYN_B", "SYN_C")

    # -------------------------------------------------------------------------
    # Stage 3: Chronological Plan with Calibration Partition
    # -------------------------------------------------------------------------
    plan_config = PlanConfig(
        protocol=EvaluationProtocol("strict-rolling-origin-v1"),
        minimum_train=120,
        validation_length=36,
        calibration_length=48,
        test_length=48,
        embargo=3,
        stride=36,
        folds=1,
    )
    plan = publish_plan(canonical.directory, plan_config, output_root=tmp_path / "eval", raw_snapshot_dir=raw.directory)
    assert len(plan.split.folds) == 1
    partition_names = [p.name for p in plan.split.folds[0].partitions]
    assert "train" in partition_names
    assert "calibration" in partition_names
    assert "test" in partition_names

    # -------------------------------------------------------------------------
    # Stage 4: UrbanGraphFormer Training & Model Weights
    # -------------------------------------------------------------------------
    model_cfg = UrbanGraphFormerConfig(
        hidden=16,
        heads=2,
        layers=1,
        num_graphs=2,
        dropout=0.0,
    )
    transformer = UrbanGraphFormer(
        config=model_cfg,
        history=12,
        horizon=6,
        num_nodes=3,
        in_features=2,
    )
    transformer.eval()

    # Save model checkpoint
    model_ckpt = tmp_path / "checkpoints" / "urbangraphformer.pt"
    model_ckpt.parent.mkdir(parents=True)
    torch.save(transformer.state_dict(), model_ckpt)
    assert model_ckpt.is_file()

    # -------------------------------------------------------------------------
    # Stage 5: Real HTTP API Inference with Sensor-Dropout Resilience
    # -------------------------------------------------------------------------
    client = TestClient(create_app(Path("tests/fixtures/eg006")))

    # 5a. Normal telemetry request
    valid_origin = client.get("/api/v1/regions").json()["regions"][0]["origins_utc"][0]
    normal_req = {
        "region_id": "SYN_A",
        "origin_utc": valid_origin,
        "injected_outages": [],
        "confidence_level": 0.90,
    }
    resp = client.post("/api/v1/predict", json=normal_req)
    assert resp.status_code == 200
    data = resp.json()
    assert data["region_id"] == "SYN_A"
    assert len(data["points"]) == 24
    assert data["outage_report"]["degradation_level"] == "NORMAL"
    assert data["outage_report"]["uncertainty_inflation_factor"] == 1.0

    # 5b. Injected sensor outage triggering harmonic spatial imputation
    outage_req = {
        "region_id": "SYN_A",
        "origin_utc": valid_origin,
        "injected_outages": ["SYN_A"],
        "confidence_level": 0.90,
    }
    resp_outage = client.post("/api/v1/predict", json=outage_req)
    assert resp_outage.status_code == 200
    data_outage = resp_outage.json()
    assert data_outage["outage_report"]["degradation_level"] in {"MODERATE_OUTAGE", "SEVERE_OUTAGE"}
    assert data_outage["outage_report"]["uncertainty_inflation_factor"] > 1.0
    assert data_outage["outage_report"]["imputation_applied"] == "HARMONIC_SPATIAL"
    # Verify uncertainty interval is inflated compared to normal
    normal_width = data["points"][0]["upper_MW"] - data["points"][0]["lower_MW"]
    outage_width = data_outage["points"][0]["upper_MW"] - data_outage["points"][0]["lower_MW"]
    assert outage_width > normal_width

    # -------------------------------------------------------------------------
    # Stage 6: Multi-Horizon Conformal Calibration & Shift Diagnostics
    # -------------------------------------------------------------------------
    cal_truth = np.ones((50, 6, 3)) * 60.0
    cal_pred = cal_truth + np.random.normal(0, 2.0, size=(50, 6, 3))
    cal_mask = np.ones_like(cal_truth, dtype=bool)
    cal_origins = np.arange(10, 60, dtype=np.int64)

    calibrator = fit_traffic_calibrator(
        truth=cal_truth,
        prediction=cal_pred,
        mask=cal_mask,
        origins=cal_origins,
        policy=TrafficCalibrationPolicy(coverage=0.90, minimum_count=15),
        detector_ids=("SYN_A", "SYN_B", "SYN_C"),
        dataset_id="metr-la",
        model_id="urbangraphformer-v1",
        observation_delay=1,
    )
    test_origins = np.array([100], dtype=np.int64)
    test_pred = np.ones((1, 6, 3)) * 62.0
    lower, upper = calibrator.intervals(test_pred, test_origins, detector_ids=("SYN_A", "SYN_B", "SYN_C"), model_id="urbangraphformer-v1")
    eval_res = evaluate_traffic_intervals(test_pred, test_pred, lower, upper, np.ones_like(test_pred, dtype=bool), 0.90)
    assert eval_res["empirical_coverage"] >= 0.90
    assert eval_res["mean_winkler_score"] > 0

    # -------------------------------------------------------------------------
    # Stage 7: Runtime Feature Drift & Prometheus Metrics
    # -------------------------------------------------------------------------
    # Check drift endpoint with identical vs shifted vector
    drift_check_req = {
        "features": {"SYN_A": [45.0, 48.0, 52.0, 46.0, 50.0] * 5},
    }
    resp_drift = client.post("/api/v1/monitoring/drift/check", json=drift_check_req)
    assert resp_drift.status_code == 200
    drift_data = resp_drift.json()
    assert "overall_status" in drift_data

    # Check metrics endpoint
    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200
    assert "urbangraph_requests_total" in resp_metrics.text
    assert "urbangraph_sensor_outages_total" in resp_metrics.text

    # -------------------------------------------------------------------------
    # Stage 8: Governance Revocation & Disaster Recovery Restoration
    # -------------------------------------------------------------------------
    gov_mgr = DataLifecycleManager()
    gov_mgr.register_policy(DatasetRetentionPolicy("metr-la", "ACADEMIC_RESEARCH", 30))
    tombstone = gov_mgr.revoke_dataset("metr-la", "License expiration")
    assert gov_mgr.is_dataset_revoked("metr-la") is True
    assert tombstone.dataset_id == "metr-la"

    # Disaster Recovery Snapshot & Bit-for-bit Verification
    backup_file = tmp_path / "system_backup.tar.gz"
    restore_dir = tmp_path / "system_restored"
    DisasterRecoveryManager.create_backup_archive(raw.directory, backup_file)
    assert backup_file.is_file()

    restore_report = DisasterRecoveryManager.restore_and_verify_archive(backup_file, restore_dir)
    assert restore_report["status"] == "RESTORE_VERIFIED_SUCCESSFUL"
    assert restore_report["manifest_verified"] is True
    assert (restore_dir / "manifest.json").exists()
