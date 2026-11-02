"""Original-unit interval metrics; trajectories are distinct from marginal events."""

from __future__ import annotations

import numpy as np

from energraph.uncertainty.conformal import FloatArray, validate_arrays
from urbangraph.data import DataError


def interval_metrics(
    truth: FloatArray,
    prediction: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    mask: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    coverage: float,
) -> dict[str, object]:
    validate_arrays(truth, prediction, mask)
    if (
        lower.shape != truth.shape
        or upper.shape != truth.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or np.any(lower > prediction)
        or np.any(prediction > upper)
        or not 0 < coverage < 1
        or not mask.any()
    ):
        raise DataError("Invalid symmetric interval or empty evaluation population")
    hit = (truth >= lower) & (truth <= upper)
    width = upper - lower
    score = width + 2 / (1 - coverage) * (
        np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0)
    )
    error = truth[mask] - prediction[mask]
    actual = float(hit[mask].mean())
    complete = np.asarray(mask.all(axis=1), dtype=np.bool_)
    simultaneous = np.asarray(hit.all(axis=1), dtype=np.bool_)[complete]
    return {
        "valid_test_count": int(mask.sum()),
        "prediction_availability": 1.0,
        "coverage": actual,
        "nominal_coverage": coverage,
        "absolute_coverage_error": abs(actual - coverage),
        "miss_frequency": 1 - actual,
        "undercoverage_gap": max(coverage - actual, 0),
        "mean_width_MW": float(width[mask].mean()),
        "median_width_MW": float(np.median(width[mask])),
        "interval_score_MW": float(score[mask].mean()),
        "mae_MW": float(np.abs(error).mean()),
        "rmse_MW": float(np.sqrt((error**2).mean())),
        "complete_trajectories": int(complete.sum()),
        "simultaneous_coverage": float(simultaneous.mean()) if len(simultaneous) else None,
    }


def dependence(
    scores: FloatArray, origins: np.ndarray[tuple[int, ...], np.dtype[np.int64]]
) -> dict[str, object]:
    if scores.ndim != 3 or scores.shape[0] != len(origins) or not np.isfinite(scores).all():
        raise DataError("Invalid dependence population")
    series = scores.mean(axis=(1, 2))
    lag_one = (
        float(np.corrcoef(series[:-1], series[1:])[0, 1])
        if len(series) > 2 and np.std(series[:-1]) > 0 and np.std(series[1:]) > 0
        else None
    )
    targets = origins[:, None] + np.arange(1, scores.shape[1] + 1)
    return {
        "origins": len(origins),
        "unique_target_rows": len(np.unique(targets)),
        "target_row_occurrences": targets.size,
        "overlapping_target_occurrences": targets.size - len(np.unique(targets)),
        "origin_mean_absolute_error_lag1_correlation": lag_one,
        "independence": "not established; separated origins are not necessarily independent",
    }
