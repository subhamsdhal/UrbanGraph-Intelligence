"""Traffic Conformal Uncertainty Workflow: End-to-end multi-horizon calibration and held-out evaluation.

Integrates strictly separated chronological calibration and held-out test splits,
reporting empirical coverage, interval widths, Winkler scores, and shift diagnostics.
Satisfies Requirement R09.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from urbangraph.data import DataError
from urbangraph.evaluation.windows import validity_mask
from urbangraph.neural.datasets import NeuralWindows, collate
from urbangraph.uncertainty.traffic_conformal import (
    TrafficCalibrationPolicy,
    TrafficConformalCalibrator,
    TrafficDistributionShiftDiagnostics,
    evaluate_traffic_intervals,
    fit_traffic_calibrator,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def generate_window_predictions(
    model: nn.Module,
    windows: NeuralWindows,
    *,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[FloatArray, FloatArray, BoolArray, IntArray]:
    """Execute model inference across chronological windows returning aligned arrays.
    
    Returns:
        truth: (n_origins, horizon, n_detectors) float64
        prediction: (n_origins, horizon, n_detectors) float64
        mask: (n_origins, horizon, n_detectors) bool
        origins: (n_origins,) int64
    """
    model.eval()
    n_windows = len(windows)
    if n_windows == 0:
        raise DataError("Cannot generate predictions on empty window set")

    all_origins = windows.origins
    all_truth = windows.raw_targets(all_origins)  # (n_windows, horizon, n_detectors)
    all_mask = validity_mask(all_truth, windows.validity)
    all_pred = np.empty_like(all_truth, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            stop = min(start + batch_size, n_windows)
            batch_samples = [windows[i] for i in range(start, stop)]
            x, _, _, _ = collate(batch_samples)
            x_dev = x.to(device)
            # Model outputs normalized forecast [B, H, N, 1] or [B, H, N]
            out = model(x_dev).detach().cpu().numpy().astype(np.float64)
            if out.ndim == 4 and out.shape[-1] == 1:
                out = out[..., 0]
            # Inverse transform from normalized space to original engineering units
            pred_unscaled = windows.scaler.inverse_transform(out, sensor_ids=windows.sensor_ids)
            all_pred[start:stop] = pred_unscaled

    if not np.isfinite(all_pred).all():
        raise DataError("Non-finite predictions encountered during window evaluation")

    return (
        np.asarray(all_truth, dtype=np.float64),
        np.asarray(all_pred, dtype=np.float64),
        np.asarray(all_mask, dtype=np.bool_),
        np.asarray(all_origins, dtype=np.int64),
    )


def calibrate_and_evaluate_workflow(
    model: nn.Module,
    calibration_windows: NeuralWindows,
    test_windows: NeuralWindows,
    *,
    policy: TrafficCalibrationPolicy | None = None,
    dataset_id: str = "traffic-network-v1",
    model_id: str = "urbangraphformer-v1",
    batch_size: int = 64,
    device: str = "cpu",
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete traffic conformal calibration and held-out evaluation workflow.
    
    1. Chronological calibration predictions generated.
    2. TrafficConformalCalibrator fitted with finite-sample guarantees.
    3. Forward held-out test predictions evaluated with prediction intervals.
    4. Empirical coverage, interval width, Winkler score, and trajectory coverage computed.
    5. Two-sample Kolmogorov-Smirnov shift diagnostics evaluated between calibration and test.
    6. Artifacts saved if output_dir specified.
    """
    if policy is None:
        policy = TrafficCalibrationPolicy(coverage=0.90, grouping="detector-horizon")

    if calibration_windows.sensor_ids != test_windows.sensor_ids:
        raise DataError("Sensor ID mismatch between calibration and test windows")

    detector_ids = calibration_windows.sensor_ids

    # 1. Calibration Phase
    cal_truth, cal_pred, cal_mask, cal_origins = generate_window_predictions(
        model, calibration_windows, batch_size=batch_size, device=device
    )

    calibrator = fit_traffic_calibrator(
        truth=cal_truth,
        prediction=cal_pred,
        mask=cal_mask,
        origins=cal_origins,
        policy=policy,
        detector_ids=detector_ids,
        dataset_id=dataset_id,
        model_id=model_id,
    )

    # 2. Evaluation Phase on Held-Out Test Set
    test_truth, test_pred, test_mask, test_origins = generate_window_predictions(
        model, test_windows, batch_size=batch_size, device=device
    )

    lower, upper = calibrator.intervals(
        test_pred,
        test_origins,
        detector_ids=detector_ids,
        model_id=model_id,
    )

    eval_results = evaluate_traffic_intervals(
        truth=test_truth,
        prediction=test_pred,
        lower=lower,
        upper=upper,
        mask=test_mask,
        coverage=policy.coverage,
        units=policy.units,
    )

    # 3. Distribution Shift Diagnostics
    cal_residuals = np.abs(cal_truth[cal_mask] - cal_pred[cal_mask])
    test_residuals = np.abs(test_truth[test_mask] - test_pred[test_mask])

    diagnostics = TrafficDistributionShiftDiagnostics(cal_residuals)
    shift_report = diagnostics.diagnose(
        test_residuals,
        empirical_coverage=float(eval_results["empirical_coverage"]),
        nominal_coverage=float(policy.coverage),
    )

    report = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "policy": policy.to_dict(),
        "detector_ids": list(detector_ids),
        "calibration": {
            "origin_count": len(cal_origins),
            "first_origin": int(cal_origins[0]),
            "last_origin": int(cal_origins[-1]),
            "last_available_origin": int(calibrator.last_available_origin),
        },
        "evaluation": eval_results,
        "shift_diagnostics": shift_report,
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        calibrator.save_json(out_p / "traffic_calibrator.json")
        with open(out_p / "conformal_evaluation.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
