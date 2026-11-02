"""Tests for runtime sensor-dropout resilience and fault tolerance."""

import pytest
import torch

from urbangraph.data import DataError
from urbangraph.serving.resilience import (
    HarmonicSpatialImputer,
    RuntimeResilienceController,
    SensorOutageDetector,
)


def test_sensor_outage_detection() -> None:
    sensors = ["s1", "s2", "s3", "s4"]
    detector = SensorOutageDetector(sensors)

    # 4 time steps, 4 sensors, 1 feature
    x = torch.ones(4, 4, 1)

    # All sensors healthy
    mask, dropped = detector.detect(x)
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert dropped == []

    # Sensor s2 has NaNs
    x[:, 1, :] = float("nan")
    mask, dropped = detector.detect(x)
    assert mask.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert dropped == ["s2"]

    # Sensor s4 is dead/zero line
    x[:, 3, :] = 0.0
    # Make time longer than 4 steps to trigger dead line
    x_long = torch.ones(8, 4, 1)
    x_long[:, 1, :] = float("nan")
    x_long[:, 3, :] = 0.0
    mask_long, dropped_long = detector.detect(x_long)
    assert mask_long.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert set(dropped_long) == {"s2", "s4"}


def test_harmonic_spatial_imputation() -> None:
    torch.manual_seed(42)
    b, t, n, f = 1, 5, 4, 1
    imputer = HarmonicSpatialImputer()

    # Adjacency: line graph 0 - 1 - 2 - 3
    adj = torch.zeros(n, n)
    adj[0, 1] = adj[1, 0] = 1.0
    adj[1, 2] = adj[2, 1] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0

    # Node 0 has value 10.0, Node 2 has value 20.0
    x = torch.zeros(b, t, n, f)
    x[:, :, 0, :] = 10.0
    x[:, :, 2, :] = 20.0

    # Node 1 is dropped (NaN)
    x[:, :, 1, :] = float("nan")
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])

    imputed, method = imputer.impute(x, adj, mask)
    assert method == "HARMONIC_SPATIAL"
    assert not torch.isnan(imputed).any()

    # Node 1 is neighbor to Node 0 (weight 1.0) and Node 2 (weight 1.0)
    # Harmonic mean should be (10.0 * 1.0 + 20.0 * 1.0) / 2.0 = 15.0
    expected_val = 15.0
    torch.testing.assert_close(
        imputed[0, :, 1, 0],
        torch.full((t,), expected_val),
        rtol=1e-4,
        atol=1e-4,
    )


def test_causal_isolation_no_future_leakage() -> None:
    """Verifies that imputation never borrows from future timestamps."""
    b, t, n, f = 1, 8, 3, 1
    imputer = HarmonicSpatialImputer()

    # Complete topological isolation (no edges)
    adj = torch.zeros(n, n)

    # Node 0 has readings until t=3, then drops out at t=4..7
    x = torch.zeros(b, t, n, f)
    x[:, 0:4, 0, :] = 5.0
    x[:, 4:8, 0, :] = float("nan")

    mask = torch.tensor([0.0, 1.0, 1.0])  # node 0 marked as dropped

    imputed, method = imputer.impute(x, adj, mask)
    assert method == "TEMPORAL_FALLBACK"

    # Future timestamps t=4..7 must not affect past timestamps t=0..3
    x_perturbed_future = x.clone()
    # Modify hypothetical future unobserved data
    x_perturbed_future[:, 7, 0, :] = 999.0

    imputed_perturbed, _ = imputer.impute(x_perturbed_future, adj, mask)

    torch.testing.assert_close(
        imputed[:, 0:4, :, :],
        imputed_perturbed[:, 0:4, :, :],
        atol=1e-6,
        rtol=1e-6,
    )


def test_resilience_controller_end_to_end() -> None:
    sensors = ["DE", "FR", "GB", "ES"]
    n = len(sensors)
    adj = torch.eye(n) + 0.5

    controller = RuntimeResilienceController(
        sensor_ids=sensors,
        adjacency=adj,
        inflation_multiplier=2.0,
        consecutive_recovery_threshold=2,
    )

    # Define mock forecaster: doubles input
    def mock_forecaster(imputed_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, t, n_nodes, f = imputed_x.shape
        # Return (B, H=2, N)
        return imputed_x[:, -1, :, 0].unsqueeze(1).expand(-1, 2, -1) * 2.0

    b, t, f = 1, 6, 1
    x = torch.ones(b, t, n, f) * 100.0

    # 1. Normal run (no dropout)
    preds, uncert, report = controller.process(x, mock_forecaster)
    assert report.degradation_level == "NORMAL"
    assert report.dropout_ratio == 0.0
    assert report.uncertainty_inflation_factor == 1.0
    assert preds.shape == (1, 2, 4)

    # 2. Outage run: FR and ES drop out (NaN)
    x_outage = x.clone()
    x_outage[:, :, 1, :] = float("nan")
    x_outage[:, :, 3, :] = float("nan")

    preds_outage, uncert_outage, report_outage = controller.process(x_outage, mock_forecaster)
    assert report_outage.degradation_level == "SEVERE_OUTAGE"
    assert report_outage.dropout_ratio == 0.50
    assert set(report_outage.dropped_sensors) == {"FR", "ES"}
    # Inflation factor: 1.0 + 2.0 * 0.50 = 2.0
    assert report_outage.uncertainty_inflation_factor == 2.0
    # Dropped nodes receive extra 1.2x penalty -> 2.4
    assert uncert_outage[0, 0, 1].item() == pytest.approx(2.4, rel=1e-3)
    assert not torch.isnan(preds_outage).any()

    # 3. Recovery run: sensors come back online
    preds_recovered, _, report_recovered = controller.process(x, mock_forecaster)
    assert report_recovered.degradation_level == "NORMAL"
    assert report_recovered.dropout_ratio == 0.0
    # Counters increment towards healthy threshold
    assert report_recovered.metadata["recovery_counters"]["FR"] >= 1
