"""Unit tests for R09 Traffic Conformal Uncertainty and Prediction Intervals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.uncertainty.traffic_conformal import (
    TrafficCalibrationPolicy,
    TrafficConformalCalibrator,
    TrafficDistributionShiftDiagnostics,
    evaluate_traffic_intervals,
    finite_quantile,
    fit_traffic_calibrator,
)


def test_traffic_policy_validation() -> None:
    # Valid
    p = TrafficCalibrationPolicy(coverage=0.90, grouping="detector-horizon")
    assert p.coverage == 0.90
    assert p.units == "km/h"

    # Invalid coverage
    with pytest.raises(DataError, match="coverage must be strictly between 0 and 1"):
        TrafficCalibrationPolicy(coverage=1.2)

    # Invalid grouping
    with pytest.raises(DataError, match="Unsupported grouping"):
        TrafficCalibrationPolicy(grouping="invalid_grouping")  # type: ignore


def test_finite_quantile_calculation() -> None:
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    # n=10, coverage=0.90 -> rank = ceil(11 * 0.90) = ceil(9.9) = 10 -> score 10.0
    val, rank = finite_quantile(scores, 0.90)
    assert rank == 10
    assert val == 10.0

    # Negative scores forbidden
    with pytest.raises(DataError, match="cannot be negative"):
        finite_quantile(np.array([-1.0, 2.0]), 0.9)


def test_traffic_calibration_fitting_and_intervals() -> None:
    rng = np.random.default_rng(42)
    n_origins = 100
    horizons = 4   # 15, 30, 45, 60 min
    n_detectors = 3
    detector_ids = ("DET_01", "DET_02", "DET_03")

    truth = rng.uniform(40.0, 100.0, size=(n_origins, horizons, n_detectors))
    # Prediction has noise ~ Normal(0, 5)
    noise = rng.normal(0.0, 5.0, size=(n_origins, horizons, n_detectors))
    prediction = truth + noise
    mask = np.ones_like(truth, dtype=bool)
    origins = np.arange(100, 100 + n_origins, dtype=np.int64)

    policy = TrafficCalibrationPolicy(coverage=0.90, grouping="detector-horizon", minimum_count=20)
    calibrator = fit_traffic_calibrator(
        truth=truth,
        prediction=prediction,
        mask=mask,
        origins=origins,
        policy=policy,
        detector_ids=detector_ids,
        dataset_id="traffic-synthetic-corridor-v1",
        model_id="urbangraphformer-v1",
        observation_delay=1,
    )

    assert calibrator.horizons == 4
    assert calibrator.thresholds.shape == (4, 3)
    assert np.all(calibrator.thresholds > 0)

    # Inference on forward evaluation origins
    test_origins = np.array([300, 301, 302], dtype=np.int64)
    test_pred = rng.uniform(50.0, 80.0, size=(3, 4, 3))
    lower, upper = calibrator.intervals(
        test_pred,
        test_origins,
        detector_ids=detector_ids,
        model_id="urbangraphformer-v1",
    )

    assert lower.shape == (3, 4, 3)
    assert upper.shape == (3, 4, 3)
    assert np.all(lower <= test_pred)
    assert np.all(test_pred <= upper)
    assert np.all(lower >= 0.0)  # Non-negative speed guaranteed


def test_temporal_isolation_leakage_prevention() -> None:
    n_origins = 50
    horizons = 2
    n_detectors = 2
    truth = np.ones((n_origins, horizons, n_detectors)) * 50.0
    prediction = np.ones((n_origins, horizons, n_detectors)) * 52.0
    mask = np.ones_like(truth, dtype=bool)
    origins = np.arange(10, 10 + n_origins, dtype=np.int64)

    calibrator = fit_traffic_calibrator(
        truth=truth,
        prediction=prediction,
        mask=mask,
        origins=origins,
        policy=TrafficCalibrationPolicy(minimum_count=10),
        detector_ids=("DET_A", "DET_B"),
        dataset_id="synthetic-traffic",
        model_id="test-model",
        observation_delay=2,
    )

    # Origin before calibration window ends (last origin is 59 + 2 + 2 = 63)
    # Attempting to use origin 60 should raise DataError
    invalid_origins = np.array([50, 60], dtype=np.int64)
    pred = np.ones((2, 2, 2)) * 50.0

    with pytest.raises(DataError, match="chronological availability violation"):
        calibrator.intervals(
            pred,
            invalid_origins,
            detector_ids=("DET_A", "DET_B"),
            model_id="test-model",
        )


def test_evaluation_metrics_and_winkler() -> None:
    rng = np.random.default_rng(123)
    truth = rng.uniform(30.0, 90.0, size=(50, 3, 2))
    pred = truth + rng.normal(0, 3.0, size=(50, 3, 2))
    mask = np.ones_like(truth, dtype=bool)

    # Synthetic intervals: +/- 7 km/h
    lower = np.maximum(pred - 7.0, 0.0)
    upper = pred + 7.0

    eval_results = evaluate_traffic_intervals(
        truth=truth,
        prediction=pred,
        lower=lower,
        upper=upper,
        mask=mask,
        coverage=0.90,
        units="km/h",
    )

    assert "empirical_coverage" in eval_results
    assert 0.80 <= eval_results["empirical_coverage"] <= 1.0
    assert eval_results["mean_winkler_score"] > 0
    assert eval_results["units"] == "km/h"
    assert eval_results["simultaneous_trajectory_coverage"] is not None


def test_distribution_shift_diagnostics() -> None:
    rng = np.random.default_rng(999)
    baseline_residuals = np.abs(rng.normal(0, 4.0, size=100))
    diagnostics = TrafficDistributionShiftDiagnostics(baseline_residuals)

    # 1. Normal traffic (same distribution)
    current_residuals = np.abs(rng.normal(0, 4.0, size=100))
    report_normal = diagnostics.diagnose(current_residuals, empirical_coverage=0.89, nominal_coverage=0.90)
    assert report_normal["severity"] == "NORMAL"
    assert report_normal["recommended_action"] == "NONE"

    # 2. Severe traffic shockwave / rain storm (drifted variance)
    drifted_residuals = np.abs(rng.normal(0, 15.0, size=100))
    report_drift = diagnostics.diagnose(drifted_residuals, empirical_coverage=0.72, nominal_coverage=0.90)
    assert report_drift["severity"] in {"MILD_DRIFT", "SEVERE_DRIFT"}
    assert report_drift["recommended_action"] in {"MONITOR_AND_QUEUE_RECALIBRATION", "IMMEDIATE_RECALIBRATION_REQUIRED"}


def test_serialization_roundtrip(tmp_path: Path) -> None:
    n_origins = 30
    horizons = 2
    n_detectors = 2
    truth = np.ones((n_origins, horizons, n_detectors)) * 60.0
    pred = truth + 2.0
    mask = np.ones_like(truth, dtype=bool)
    origins = np.arange(10, 10 + n_origins, dtype=np.int64)

    calibrator = fit_traffic_calibrator(
        truth=truth,
        prediction=pred,
        mask=mask,
        origins=origins,
        policy=TrafficCalibrationPolicy(minimum_count=10),
        detector_ids=("D1", "D2"),
        dataset_id="traffic-synth",
        model_id="m1",
    )

    json_file = tmp_path / "traffic_calibrator.json"
    calibrator.save_json(json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    restored = TrafficConformalCalibrator.from_dict(data)
    assert restored.detector_ids == calibrator.detector_ids
    assert restored.model_id == calibrator.model_id
    assert np.allclose(restored.thresholds, calibrator.thresholds)
