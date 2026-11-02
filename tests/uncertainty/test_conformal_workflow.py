"""Integration tests for R09 Traffic Conformal Uncertainty Workflow."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import EvaluationProtocol, PlanConfig, SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.uncertainty.traffic_conformal import TrafficCalibrationPolicy
from urbangraph.uncertainty.workflow import (
    calibrate_and_evaluate_workflow,
    generate_window_predictions,
)


class MockTrafficModel(nn.Module):
    """Deterministic mock neural model predicting smoothed moving average."""

    def __init__(self, history: int = 12, horizon: int = 12) -> None:
        super().__init__()
        self.history = history
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, T, N, F]
        # Return [B, H, N, 1] with slight offset
        val = x[:, -1:, :, 0:1].expand(-1, self.horizon, -1, -1)
        return val + 0.05


@pytest.fixture
def traffic_fixture(tmp_path: Path) -> dict[str, object]:
    # 500 timesteps at 5-minute intervals = ~41 hours
    n_rows = 500
    dates = pd.date_range("2026-01-01 00:00:00", periods=n_rows, freq="5min")
    sensor_ids = ("SENSOR_EAST", "SENSOR_WEST", "SENSOR_CENTRAL")
    
    # Generate realistic traffic speed series (40-100 km/h)
    rng = np.random.default_rng(2026)
    base_speed = 70.0 + 15.0 * np.sin(np.linspace(0, 4 * np.pi, n_rows))[:, None]
    noise = rng.normal(0, 3.0, size=(n_rows, len(sensor_ids)))
    values = np.clip(base_speed + noise, 20.0, 110.0)
    
    df = pd.DataFrame(values, index=dates, columns=list(sensor_ids))
    dataset = EvaluationDataset("metr-synth", df, "synthetic-lineage-123")
    
    # Plan split with dedicated train, validation, calibration, test
    # e.g., rolling with minimum_train=200, val=50, cal=80, test=100
    config = PlanConfig(
        protocol=EvaluationProtocol("strict-rolling-origin-v1"),
        minimum_train=180,
        validation_length=40,
        calibration_length=80,
        test_length=100,
        embargo=5,
        stride=50,
        folds=1,
    )
    split = plan_splits(dates, config)
    
    from urbangraph.neural.preprocessing import fit_preprocessing

    scaler = fit_preprocessing(dataset, split, fold=0, mode="per_sensor")
    
    cal_windows = NeuralWindows(dataset, split, fold=0, partition="calibration", scaler=scaler)
    test_windows = NeuralWindows(dataset, split, fold=0, partition="test", scaler=scaler)
    
    return {
        "dataset": dataset,
        "split": split,
        "cal_windows": cal_windows,
        "test_windows": test_windows,
        "sensor_ids": sensor_ids,
        "model": MockTrafficModel(history=12, horizon=12),
    }


def test_generate_window_predictions(traffic_fixture: dict[str, object]) -> None:
    model: nn.Module = traffic_fixture["model"]  # type: ignore
    cal_windows: NeuralWindows = traffic_fixture["cal_windows"]  # type: ignore

    truth, pred, mask, origins = generate_window_predictions(model, cal_windows, batch_size=32)

    assert len(truth) == len(cal_windows)
    assert truth.shape == (len(cal_windows), 12, 3)
    assert pred.shape == (len(cal_windows), 12, 3)
    assert mask.shape == (len(cal_windows), 12, 3)
    assert origins.shape == (len(cal_windows),)
    assert np.all(np.isfinite(pred))
    assert np.all(np.isfinite(truth))


def test_calibrate_and_evaluate_workflow_end_to_end(
    traffic_fixture: dict[str, object], tmp_path: Path
) -> None:
    model: nn.Module = traffic_fixture["model"]  # type: ignore
    cal_windows: NeuralWindows = traffic_fixture["cal_windows"]  # type: ignore
    test_windows: NeuralWindows = traffic_fixture["test_windows"]  # type: ignore

    policy = TrafficCalibrationPolicy(coverage=0.90, grouping="detector-horizon", minimum_count=20)
    out_dir = tmp_path / "conformal_output"

    report = calibrate_and_evaluate_workflow(
        model=model,
        calibration_windows=cal_windows,
        test_windows=test_windows,
        policy=policy,
        dataset_id="traffic-synth-corridor",
        model_id="mock-urbangraphformer-v1",
        batch_size=32,
        output_dir=out_dir,
    )

    assert report["dataset_id"] == "traffic-synth-corridor"
    assert report["model_id"] == "mock-urbangraphformer-v1"
    assert report["policy"]["coverage"] == 0.90
    assert report["evaluation"]["nominal_coverage"] == 0.90
    assert 0.70 <= report["evaluation"]["empirical_coverage"] <= 1.0
    assert report["evaluation"]["mean_winkler_score"] > 0
    assert report["evaluation"]["mean_width"] > 0
    assert report["shift_diagnostics"]["ks_statistic"] >= 0.0

    # Verify persisted artifacts
    assert (out_dir / "traffic_calibrator.json").exists()
    assert (out_dir / "conformal_evaluation.json").exists()

    with open(out_dir / "traffic_calibrator.json", "r", encoding="utf-8") as f:
        calibrator_data = json.load(f)
        assert calibrator_data["horizons"] == 12
        assert len(calibrator_data["detector_ids"]) == 3

    with open(out_dir / "conformal_evaluation.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        assert "evaluation" in eval_data
        assert "shift_diagnostics" in eval_data


def test_workflow_detects_distribution_drift(traffic_fixture: dict[str, object]) -> None:
    cal_windows: NeuralWindows = traffic_fixture["cal_windows"]  # type: ignore
    test_windows: NeuralWindows = traffic_fixture["test_windows"]  # type: ignore

    # Model for calibration
    cal_model = MockTrafficModel(history=12, horizon=12)

    # Corrupted / drifted model on test causing large residual shift
    class DriftedTrafficModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            val = x[:, -1:, :, 0:1].expand(-1, 12, -1, -1)
            # Severe prediction bias simulating sensor degradation or shock
            return val + 35.0

    drifted_model = DriftedTrafficModel()

    # Step 1: Calibrate with normal model
    cal_truth, cal_pred, cal_mask, cal_origins = generate_window_predictions(cal_model, cal_windows)
    from urbangraph.uncertainty.traffic_conformal import fit_traffic_calibrator, evaluate_traffic_intervals, TrafficDistributionShiftDiagnostics

    calibrator = fit_traffic_calibrator(
        truth=cal_truth,
        prediction=cal_pred,
        mask=cal_mask,
        origins=cal_origins,
        policy=TrafficCalibrationPolicy(coverage=0.90, minimum_count=20),
        detector_ids=cal_windows.sensor_ids,
        dataset_id="test",
        model_id="test",
    )

    # Step 2: Test on drifted model predictions
    test_truth, test_pred, test_mask, test_origins = generate_window_predictions(drifted_model, test_windows)
    lower, upper = calibrator.intervals(test_pred, test_origins, detector_ids=test_windows.sensor_ids, model_id="test")

    eval_results = evaluate_traffic_intervals(test_truth, test_pred, lower, upper, test_mask, 0.90)
    
    cal_residuals = np.abs(cal_truth[cal_mask] - cal_pred[cal_mask])
    test_residuals = np.abs(test_truth[test_mask] - test_pred[test_mask])

    diagnostics = TrafficDistributionShiftDiagnostics(cal_residuals)
    report = diagnostics.diagnose(test_residuals, float(eval_results["empirical_coverage"]), 0.90)

    assert report["severity"] in {"MILD_DRIFT", "SEVERE_DRIFT"}
    assert report["recommended_action"] != "NONE"
