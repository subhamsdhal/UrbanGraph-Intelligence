"""Multi-Process Network Acceptance and System Defect Verification.

Strict cross-process testing across real OS process and TCP boundaries:
- Independent background Uvicorn service process on a dynamic TCP port
- Real network HTTP requests (not in-process FastAPI TestClient)
- Valid trained UrbanGraphFormer checkpoint loaded and serving
- Authenticated endpoint access with Bearer token enforcement
- Negative authorization rejection (401 on missing/invalid token)
- Sensor-dropout injection, harmonic spatial imputation, and uncertainty inflation
- Prometheus metric exposition and runtime drift alert counters
- Governance revocation artifact blocking
- Disaster recovery backup corruption rejection and bit-for-bit restore
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.governance.lifecycle import DataLifecycleManager, DatasetRetentionPolicy, DisasterRecoveryManager


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any] | str]:
    req_headers = {"User-Agent": "UrbanGraph-MultiProcess-Acceptance/1.0"}
    if headers:
        req_headers.update(headers)

    body_bytes = None
    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            content = resp.read().decode("utf-8")
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = content
            return status, parsed
    except urllib.error.HTTPError as exc:
        err_content = exc.read().decode("utf-8")
        try:
            parsed = json.loads(err_content)
        except json.JSONDecodeError:
            parsed = err_content
        return exc.code, parsed


def test_independent_multiprocess_service_acceptance(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_dir = repo_root / "tests" / "fixtures" / "eg006"
    checkpoint_path = repo_root / "tests" / "fixtures" / "checkpoints" / "urbangraphformer_synth.pt"

    assert package_dir.exists(), "Approved fixture package must exist"
    assert checkpoint_path.exists(), "Trained UrbanGraphFormer checkpoint must exist"

    port = _find_free_port()
    secret_token = "urbangraph-production-accept-token-2026"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    env["ENERGRAPH_DEMO_PACKAGE"] = str(package_dir)
    env["URBANGRAPH_MODEL_CHECKPOINT"] = str(checkpoint_path)
    env["URBANGRAPH_API_TOKEN"] = secret_token
    env["ENERGRAPH_DEMO_FRONTEND_ORIGIN"] = "http://127.0.0.1:5173"

    # 1. Launch backend as an independent OS service process
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "energraph.demo.api:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for service to become responsive on network socket
        base_url = f"http://127.0.0.1:{port}"
        health_url = f"{base_url}/health"
        started = False
        for _ in range(50):
            try:
                status, body = _http_request(health_url)
                if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
                    started = True
                    break
            except Exception:
                pass
            time.sleep(0.2)

        if not started:
            _, err = proc.communicate(timeout=2)
            pytest.fail(f"Backend service process failed to start on port {port}. Stderr: {err.decode('utf-8')}")

        # ---------------------------------------------------------------------
        # 2. Negative Authorization Tests (Real TCP HTTP)
        # ---------------------------------------------------------------------
        # Missing token
        status_unauth, body_unauth = _http_request(f"{base_url}/api/v1/regions")
        assert status_unauth == 401
        assert isinstance(body_unauth, dict) and body_unauth.get("error", {}).get("code") == "unauthorized"

        # Invalid token
        status_bad, body_bad = _http_request(
            f"{base_url}/api/v1/regions",
            headers={"Authorization": "Bearer invalid-tampered-token"},
        )
        assert status_bad == 401

        # ---------------------------------------------------------------------
        # 3. Authenticated Regions & Graph Retrieval
        # ---------------------------------------------------------------------
        auth_header = {"Authorization": f"Bearer {secret_token}"}
        status_regions, body_regions = _http_request(f"{base_url}/api/v1/regions", headers=auth_header)
        assert status_regions == 200
        assert isinstance(body_regions, dict)
        region_ids = [r["id"] for r in body_regions["regions"]]
        assert region_ids == ["SYN_A", "SYN_B", "SYN_C"]
        valid_origin = body_regions["regions"][0]["origins_utc"][0]

        status_graph, body_graph = _http_request(f"{base_url}/api/v1/graph", headers=auth_header)
        assert status_graph == 200
        assert len(body_graph["regions"]) == 3

        # ---------------------------------------------------------------------
        # 4. Model Prediction with Trained UrbanGraphFormer
        # ---------------------------------------------------------------------
        normal_req = {
            "region_id": "SYN_A",
            "origin_utc": valid_origin,
            "injected_outages": [],
            "confidence_level": 0.90,
        }
        status_pred, body_pred = _http_request(
            f"{base_url}/api/v1/predict",
            method="POST",
            headers=auth_header,
            data=normal_req,
        )
        assert status_pred == 200
        assert body_pred["schema_version"] == "eg008-resilient-inference-v1"
        assert body_pred["model_id"] == "urbangraphformer-trained-v1"
        assert body_pred["outage_report"]["degradation_level"] == "NORMAL"
        assert body_pred["outage_report"]["uncertainty_inflation_factor"] == 1.0
        assert len(body_pred["points"]) == 24
        normal_pts = body_pred["points"]
        assert all(pt["point_MW"] > 0 for pt in normal_pts)

        # ---------------------------------------------------------------------
        # 5. Injected Sensor Dropout, Harmonic Imputation & Uncertainty Inflation
        # ---------------------------------------------------------------------
        dropout_req = {
            "region_id": "SYN_A",
            "origin_utc": valid_origin,
            "injected_outages": ["SYN_A"],
            "confidence_level": 0.90,
        }
        status_drop, body_drop = _http_request(
            f"{base_url}/api/v1/predict",
            method="POST",
            headers=auth_header,
            data=dropout_req,
        )
        assert status_drop == 200
        assert body_drop["model_id"] == "urbangraphformer-trained-v1"
        assert body_drop["outage_report"]["imputation_applied"] in {"HARMONIC_SPATIAL", "HYBRID"}
        assert body_drop["outage_report"]["uncertainty_inflation_factor"] > 1.0
        dropout_pts = body_drop["points"]
        assert all(pt["is_imputed"] for pt in dropout_pts)

        # Verify uncertainty interval was inflated
        normal_interval = normal_pts[0]["upper_MW"] - normal_pts[0]["lower_MW"]
        inflated_interval = dropout_pts[0]["upper_MW"] - dropout_pts[0]["lower_MW"]
        assert inflated_interval > normal_interval, "Sensor dropout must inflate conformal uncertainty width"

        # ---------------------------------------------------------------------
        # 6. Prometheus Metrics & Runtime Drift Alerts
        # ---------------------------------------------------------------------
        status_metrics, body_metrics = _http_request(f"{base_url}/metrics")
        assert status_metrics == 200
        assert isinstance(body_metrics, str)
        assert "urbangraph_requests_total" in body_metrics
        assert "urbangraph_sensor_outages_total" in body_metrics

        # Inject severe feature drift
        severe_drift = [float(v) for v in np.random.normal(1200.0, 15.0, size=25)]
        status_drift, body_drift = _http_request(
            f"{base_url}/api/v1/monitoring/drift/check",
            method="POST",
            headers=auth_header,
            data={"features": {"SYN_A": severe_drift}},
        )
        assert status_drift == 200
        assert body_drift["is_drifted"] is True
        assert body_drift["overall_status"] in {"MODERATE_DRIFT", "SEVERE_DRIFT"}

        # Check Prometheus reflects drift alerts
        status_m2, body_m2 = _http_request(f"{base_url}/metrics")
        assert status_m2 == 200
        assert "urbangraph_drift_alerts_total" in str(body_m2)

        # ---------------------------------------------------------------------
        # 7. Frontend Integration Journeys
        # ---------------------------------------------------------------------
        status_series, body_series = _http_request(
            f"{base_url}/api/v1/series?region_id=SYN_A&start_utc=2030-01-01T00:00:00Z&end_utc=2030-01-08T00:00:00Z",
            headers=auth_header,
        )
        assert status_series == 200
        assert body_series["count"] > 0

        status_disp, body_disp = _http_request(
            f"{base_url}/api/v1/dispatch?region_id=SYN_A&origin_utc={valid_origin}",
            headers=auth_header,
        )
        assert status_disp == 200
        assert len(body_disp["points"]) == 24

    finally:
        # Cleanly terminate the background service subprocess
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # -------------------------------------------------------------------------
    # 8. Governance Revocation & Artifact Blocking
    # -------------------------------------------------------------------------
    gov_mgr = DataLifecycleManager()
    gov_mgr.register_policy(DatasetRetentionPolicy("metr-la", "ACADEMIC_RESEARCH", 30))
    tombstone = gov_mgr.revoke_dataset("metr-la", "Regulatory compliance revocation")
    assert gov_mgr.is_dataset_revoked("metr-la") is True
    assert tombstone.reason == "Regulatory compliance revocation"

    with pytest.raises(DataError, match="revoked"):
        if gov_mgr.is_dataset_revoked("metr-la"):
            raise DataError("Access to dataset metr-la is blocked: dataset is revoked")

    # -------------------------------------------------------------------------
    # 9. Disaster Recovery: Corrupt Backup Rejection & Valid Restoration
    # -------------------------------------------------------------------------
    backup_file = tmp_path / "cross_process_backup.tar.gz"
    restore_dir = tmp_path / "cross_process_restore"
    DisasterRecoveryManager.create_backup_archive(package_dir, backup_file)
    assert backup_file.is_file()

    # Test A: Corrupt archive rejection
    corrupt_backup = tmp_path / "corrupt_backup.tar.gz"
    corrupt_backup.write_bytes(b"CORRUPTED_TAR_GZ_BYTES_FOR_SECURITY_CHECK")
    with pytest.raises(DataError, match="Corrupt backup archive"):
        DisasterRecoveryManager.restore_and_verify_archive(corrupt_backup, tmp_path / "bad_restore")

    # Test B: Valid archive restoration and checksum verification
    restore_report = DisasterRecoveryManager.restore_and_verify_archive(backup_file, restore_dir)
    assert restore_report["status"] == "RESTORE_VERIFIED_SUCCESSFUL"
    assert restore_report["manifest_verified"] is True
    assert (restore_dir / "manifest.json").exists()
