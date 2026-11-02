"""Unit tests for R21 Observability, Telemetry, Prometheus Exporter, and Drift Detection."""

from __future__ import annotations

import concurrent.futures
import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.monitoring.drift import FeatureDriftDetector, compute_ks, compute_psi
from urbangraph.monitoring.observability import (
    MetricsCollector,
    PrometheusExporter,
    redact_secrets,
)


def test_metrics_collection_and_percentiles() -> None:
    collector = MetricsCollector()
    collector.register_model_artifact("urbangraphformer-v1", "1.0", "abcdef123456")

    # Record 100 requests with varying latencies
    for i in range(1, 101):
        status = 200 if i <= 95 else 500
        collector.record_request("/v1/forecast", status, latency_ms=float(i), model_id="urbangraphformer-v1")

    collector.record_outage(2)
    collector.record_drift_alert()

    summary = collector.get_summary()
    assert summary["total_requests"] == 100
    assert summary["total_errors"] == 5
    assert summary["error_rate"] == 0.05
    assert summary["sensor_outages_total"] == 2
    assert summary["drift_alerts_total"] == 1
    assert summary["predictions_by_model"]["urbangraphformer-v1"] == 100

    percentiles = summary["latency"]
    assert 48.0 <= percentiles["p50_ms"] <= 52.0
    assert 88.0 <= percentiles["p90_ms"] <= 92.0
    assert 97.0 <= percentiles["p99_ms"] <= 100.0


def test_prometheus_exposition_format() -> None:
    collector = MetricsCollector()
    collector.record_request("/v1/forecast", 200, 45.0)
    collector.record_request("/v1/forecast", 422, 12.0)
    collector.record_outage(1)

    prom_text = PrometheusExporter.to_prometheus_text(collector)
    assert "# HELP urbangraph_requests_total" in prom_text
    assert "# TYPE urbangraph_requests_total counter" in prom_text
    assert 'urbangraph_requests_total{endpoint="/v1/forecast",status="200"} 1' in prom_text
    assert 'urbangraph_requests_total{endpoint="/v1/forecast",status="422"} 1' in prom_text
    assert "urbangraph_sensor_outages_total 1" in prom_text
    assert "urbangraph_latency_p50_milliseconds" in prom_text


def test_secret_redaction() -> None:
    sensitive_dict = {
        "user": "operator",
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload",
        "nested": {
            "api_key": "ug-secret-key-9999",
            "password": "supersecretpassword",
            "log_line": "Received token=abc12345xyz from client",
        },
        "safe_field": 42,
    }

    cleaned = redact_secrets(sensitive_dict)
    assert cleaned["authorization"] == "[REDACTED]"
    assert cleaned["nested"]["api_key"] == "[REDACTED]"
    assert cleaned["nested"]["password"] == "[REDACTED]"
    assert "[REDACTED]" in cleaned["nested"]["log_line"]
    assert cleaned["safe_field"] == 42


def test_health_check_reporting() -> None:
    collector = MetricsCollector()
    # Initial healthy state
    health = collector.check_health()
    assert health.status == "HEALTHY"
    assert health.ready is True

    # Inject high error rate -> triggers DEGRADED / UNHEALTHY
    for _ in range(50):
        collector.record_request("/v1/forecast", 500, 100.0)

    degraded_health = collector.check_health()
    assert degraded_health.status in {"DEGRADED", "UNHEALTHY"}


def test_psi_and_ks_drift_detection() -> None:
    rng = np.random.default_rng(42)
    ref_data = rng.normal(50.0, 10.0, size=200)

    # 1. No drift (drawn from same distribution)
    curr_stable = rng.normal(50.0, 10.0, size=200)
    psi_stable = compute_psi(ref_data, curr_stable)
    ks_stable = compute_ks(ref_data, curr_stable)
    assert psi_stable < 0.10
    assert ks_stable < 0.15

    # 2. Significant drift (mean shifted to 75, std expanded to 25)
    curr_drifted = rng.normal(75.0, 25.0, size=200)
    psi_drifted = compute_psi(ref_data, curr_drifted)
    ks_drifted = compute_ks(ref_data, curr_drifted)
    assert psi_drifted > 0.25
    assert ks_drifted > 0.20

    # Test FeatureDriftDetector workflow
    detector = FeatureDriftDetector(reference_data={"sensor_flow": ref_data})

    report_stable = detector.check_drift({"sensor_flow": curr_stable})
    assert report_stable.is_drifted is False
    assert report_stable.overall_status == "STABLE"

    report_drifted = detector.check_drift({"sensor_flow": curr_drifted})
    assert report_drifted.is_drifted is True
    assert report_drifted.overall_status == "SEVERE_DRIFT"
    assert len(report_drifted.alerts) > 0
    assert report_drifted.alerts[0].severity == "CRITICAL"


def test_false_alert_suppression_on_small_samples() -> None:
    rng = np.random.default_rng(77)
    ref = rng.normal(0, 1, size=100)
    detector = FeatureDriftDetector(reference_data={"feature": ref}, min_batch_size=20)

    # Only 5 samples (even if skewed, should suppress alert)
    small_batch = np.array([10.0, 12.0, 11.0, 15.0, 14.0])
    report = detector.check_drift({"feature": small_batch})
    assert report.is_drifted is False
    assert "suppressed" in report.alerts[0].message


def test_concurrent_metric_recording() -> None:
    collector = MetricsCollector()

    def record_worker(worker_id: int) -> None:
        for _ in range(50):
            collector.record_request("/v1/forecast", 200, latency_ms=10.0 + worker_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(record_worker, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    summary = collector.get_summary()
    assert summary["total_requests"] == 400
