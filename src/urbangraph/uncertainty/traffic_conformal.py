"""Traffic Conformal Uncertainty: finite-sample guaranteed prediction intervals for urban traffic sensors.

Satisfies Requirement R09:
- Multi-horizon traffic contracts (speed km/h, flow veh/h, occupancy %).
- Strict chronological calibration splits preventing temporal lookahead.
- Non-conformity residual thresholds with finite-sample coverage guarantees.
- Comprehensive evaluation: empirical coverage, interval width, Winkler score, simultaneous trajectory coverage.
- Distribution-shift diagnostics: residual distribution testing and non-stationarity alerts.
- Artifact serialization for reproducible pipeline deployment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TrafficCalibrationPolicy:
    """Configuration contract for traffic conformal calibration."""

    coverage: float = 0.90
    grouping: Literal["global", "detector", "horizon", "detector-horizon"] = "detector-horizon"
    minimum_count: int = 30
    origin_spacing: int = 1
    clip_non_negative: bool = True
    units: str = "km/h"
    version: str = "traffic-conformal-1.0"

    def __post_init__(self) -> None:
        if not np.isfinite(self.coverage) or not 0.0 < self.coverage < 1.0:
            raise DataError("Traffic calibration coverage must be strictly between 0 and 1")
        if self.grouping not in {"global", "detector", "horizon", "detector-horizon"}:
            raise DataError(f"Unsupported grouping strategy: {self.grouping}")
        if not isinstance(self.minimum_count, int) or self.minimum_count < 5:
            raise DataError("Minimum sample count must be an integer >= 5")
        if not isinstance(self.origin_spacing, int) or self.origin_spacing < 1:
            raise DataError("Origin spacing must be an integer >= 1")
        if not isinstance(self.clip_non_negative, bool):
            raise DataError("clip_non_negative must be a boolean")
        if not self.units or not isinstance(self.units, str):
            raise DataError("Traffic units must be a non-empty string (e.g. 'km/h')")
        if self.version != "traffic-conformal-1.0":
            raise DataError("Unsupported policy version")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def finite_quantile(scores: FloatArray, coverage: float) -> tuple[float, int]:
    """Compute exact empirical quantile satisfying P(score <= q) >= coverage."""
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise DataError("Finite quantile requires non-empty array of finite positive scores")
    if np.any(scores < 0.0):
        raise DataError("Non-conformity scores cannot be negative")
    n = len(scores)
    rank = ceil((n + 1) * coverage)
    if rank > n:
        raise DataError(f"Insufficient samples (n={n}) to guarantee coverage {coverage} without extrapolation")
    val = float(np.partition(scores, rank - 1)[rank - 1])
    return val, rank


def validate_traffic_arrays(truth: FloatArray, prediction: FloatArray, mask: BoolArray) -> None:
    """Validate aligned [origins, horizons, detectors] arrays."""
    if (
        truth.ndim != 3
        or prediction.shape != truth.shape
        or mask.shape != truth.shape
        or mask.dtype != np.bool_
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth[mask]).all()
    ):
        raise DataError("Expected aligned finite [origins, horizons, detectors] traffic arrays")


@dataclass(frozen=True)
class TrafficConformalCalibrator:
    """Calibrated prediction interval generator for multi-horizon traffic networks."""

    policy: TrafficCalibrationPolicy
    detector_ids: tuple[str, ...]
    dataset_id: str
    model_id: str
    last_available_origin: int
    horizons: int
    thresholds: FloatArray  # (horizons, num_detectors)
    scales: FloatArray      # (horizons, num_detectors)
    counts: IntArray        # (horizons, num_detectors)
    ranks: IntArray         # (horizons, num_detectors)
    fallback: BoolArray     # (horizons, num_detectors)

    def intervals(
        self,
        prediction: FloatArray,
        origins: IntArray,
        *,
        detector_ids: tuple[str, ...],
        model_id: str,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute [lower, upper] prediction intervals for given model forecast."""
        if (
            prediction.ndim != 3
            or prediction.shape[1:] != self.thresholds.shape
            or prediction.shape[0] != len(origins)
            or origins.ndim != 1
            or len(origins) == 0
            or not np.isfinite(prediction).all()
            or detector_ids != self.detector_ids
            or model_id != self.model_id
            or np.any(origins < self.last_available_origin)
            or np.any(np.diff(origins) <= 0)
        ):
            raise DataError("Model, detector_ids, shape, or chronological availability violation")

        half_width = self.thresholds * self.scales  # (horizons, detectors)
        lower = prediction - half_width
        upper = prediction + half_width

        if self.policy.clip_non_negative:
            lower = np.maximum(lower, 0.0)

        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise DataError("Non-finite interval endpoints produced")

        return lower, upper

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "detector_ids": list(self.detector_ids),
            "dataset_id": self.dataset_id,
            "model_id": self.model_id,
            "last_available_origin": int(self.last_available_origin),
            "horizons": int(self.horizons),
            "thresholds": self.thresholds.tolist(),
            "scales": self.scales.tolist(),
            "counts": self.counts.tolist(),
            "ranks": self.ranks.tolist(),
            "fallback": self.fallback.tolist(),
        }

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrafficConformalCalibrator:
        policy_dict = data["policy"]
        policy = TrafficCalibrationPolicy(
            coverage=float(policy_dict["coverage"]),
            grouping=policy_dict["grouping"],
            minimum_count=int(policy_dict["minimum_count"]),
            origin_spacing=int(policy_dict["origin_spacing"]),
            clip_non_negative=bool(policy_dict.get("clip_non_negative", True)),
            units=str(policy_dict.get("units", "km/h")),
            version=str(policy_dict.get("version", "traffic-conformal-1.0")),
        )
        return cls(
            policy=policy,
            detector_ids=tuple(data["detector_ids"]),
            dataset_id=str(data["dataset_id"]),
            model_id=str(data["model_id"]),
            last_available_origin=int(data["last_available_origin"]),
            horizons=int(data["horizons"]),
            thresholds=np.array(data["thresholds"], dtype=np.float64),
            scales=np.array(data["scales"], dtype=np.float64),
            counts=np.array(data["counts"], dtype=np.int64),
            ranks=np.array(data["ranks"], dtype=np.int64),
            fallback=np.array(data["fallback"], dtype=np.bool_),
        )


def fit_traffic_calibrator(
    truth: FloatArray,
    prediction: FloatArray,
    mask: BoolArray,
    origins: IntArray,
    *,
    policy: TrafficCalibrationPolicy,
    detector_ids: tuple[str, ...],
    dataset_id: str,
    model_id: str,
    observation_delay: int = 1,
    scales: FloatArray | None = None,
) -> TrafficConformalCalibrator:
    """Fit a conformal calibrator on a dedicated chronological calibration split."""
    validate_traffic_arrays(truth, prediction, mask)
    n_origins, horizons, n_detectors = truth.shape

    if (
        origins.shape != (n_origins,)
        or origins.dtype != np.int64
        or np.any(origins < 0)
        or np.any(np.diff(origins) <= 0)
        or len(detector_ids) != n_detectors
        or len(set(detector_ids)) != n_detectors
        or not dataset_id
        or not model_id
        or not isinstance(observation_delay, int)
        or observation_delay < 0
    ):
        raise DataError("Invalid calibration split metadata, chronology, or detector IDs")

    scale_matrix = np.ones((horizons, n_detectors), dtype=np.float64) if scales is None else scales.copy()
    if scale_matrix.shape != (horizons, n_detectors) or not np.isfinite(scale_matrix).all() or np.any(scale_matrix <= 0):
        raise DataError("Scales must be positive finite matrix matching [horizons, detectors]")

    # Subsample origins according to spacing
    selected_indices: list[int] = []
    for i, origin in enumerate(origins):
        if not selected_indices or origin - origins[selected_indices[-1]] >= policy.origin_spacing:
            selected_indices.append(i)

    residuals = np.abs(truth[selected_indices] - prediction[selected_indices]) / scale_matrix
    valid_mask = mask[selected_indices]

    pooled_residuals = residuals[valid_mask]
    if len(pooled_residuals) < policy.minimum_count:
        raise DataError(f"Insufficient valid calibration samples ({len(pooled_residuals)} < {policy.minimum_count})")

    global_thresh, global_rank = finite_quantile(pooled_residuals, policy.coverage)

    thresholds = np.zeros((horizons, n_detectors), dtype=np.float64)
    counts = np.zeros((horizons, n_detectors), dtype=np.int64)
    ranks = np.zeros((horizons, n_detectors), dtype=np.int64)
    fallback = np.zeros((horizons, n_detectors), dtype=np.bool_)

    for h in range(horizons):
        for d in range(n_detectors):
            h_slice = slice(h, h + 1) if "horizon" in policy.grouping else slice(None)
            d_slice = slice(d, d + 1) if "detector" in policy.grouping else slice(None)

            sub_scores = residuals[:, h_slice, d_slice][valid_mask[:, h_slice, d_slice]]
            cnt = len(sub_scores)
            counts[h, d] = cnt

            if cnt < policy.minimum_count or ceil((cnt + 1) * policy.coverage) > cnt:
                thresholds[h, d] = global_thresh
                ranks[h, d] = global_rank
                fallback[h, d] = True
            else:
                q_val, r_val = finite_quantile(sub_scores, policy.coverage)
                thresholds[h, d] = q_val
                ranks[h, d] = r_val

    last_usable_row = int(origins[-1]) + horizons + observation_delay

    return TrafficConformalCalibrator(
        policy=policy,
        detector_ids=detector_ids,
        dataset_id=dataset_id,
        model_id=model_id,
        last_available_origin=last_usable_row,
        horizons=horizons,
        thresholds=thresholds,
        scales=scale_matrix,
        counts=counts,
        ranks=ranks,
        fallback=fallback,
    )


def evaluate_traffic_intervals(
    truth: FloatArray,
    prediction: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    mask: BoolArray,
    coverage: float,
    units: str = "km/h",
) -> dict[str, object]:
    """Evaluate empirical coverage, interval width, Winkler score, and trajectory coverage."""
    validate_traffic_arrays(truth, prediction, mask)
    if (
        lower.shape != truth.shape
        or upper.shape != truth.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or not 0.0 < coverage < 1.0
        or not mask.any()
    ):
        raise DataError("Invalid interval dimensions or evaluation array")

    hit = (truth >= lower) & (truth <= upper)
    width = upper - lower

    # Winkler score: width + (2 / (1 - alpha)) * penalty
    alpha = 1.0 - coverage
    penalty_lower = np.maximum(lower - truth, 0.0)
    penalty_upper = np.maximum(truth - upper, 0.0)
    winkler = width + (2.0 / alpha) * (penalty_lower + penalty_upper)

    abs_error = np.abs(truth[mask] - prediction[mask])
    empirical_coverage = float(hit[mask].mean())

    # Trajectory simultaneous coverage (all horizons for a given origin & detector hit)
    complete_trajectories = np.asarray(mask.all(axis=1), dtype=np.bool_)
    simultaneous_hit = np.asarray(hit.all(axis=1), dtype=np.bool_)[complete_trajectories]

    return {
        "valid_eval_count": int(mask.sum()),
        "nominal_coverage": float(coverage),
        "empirical_coverage": empirical_coverage,
        "coverage_gap": float(max(coverage - empirical_coverage, 0.0)),
        "mean_width": float(width[mask].mean()),
        "median_width": float(np.median(width[mask])),
        "mean_winkler_score": float(winkler[mask].mean()),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt((abs_error**2).mean())),
        "units": units,
        "complete_trajectories": int(complete_trajectories.sum()),
        "simultaneous_trajectory_coverage": float(simultaneous_hit.mean()) if len(simultaneous_hit) > 0 else None,
    }


class TrafficDistributionShiftDiagnostics:
    """Diagnoses temporal non-stationarity, peak congestion shifts, and distribution drift."""

    def __init__(self, baseline_residuals: FloatArray, alert_threshold: float = 0.05) -> None:
        if baseline_residuals.ndim != 1 or len(baseline_residuals) < 20:
            raise DataError("Baseline requires at least 20 1D residual observations")
        self.baseline = np.sort(baseline_residuals)
        self.alert_threshold = alert_threshold

    def kolmogorov_smirnov_statistic(self, current_residuals: FloatArray) -> float:
        """Compute 2-sample Kolmogorov-Smirnov statistic D between baseline and current."""
        if current_residuals.ndim != 1 or len(current_residuals) == 0:
            raise DataError("Current residuals must be non-empty 1D array")
        curr = np.sort(current_residuals)
        all_vals = np.concatenate([self.baseline, curr])
        cdf1 = np.searchsorted(self.baseline, all_vals, side="right") / len(self.baseline)
        cdf2 = np.searchsorted(curr, all_vals, side="right") / len(curr)
        return float(np.max(np.abs(cdf1 - cdf2)))

    def diagnose(self, current_residuals: FloatArray, empirical_coverage: float, nominal_coverage: float) -> dict[str, object]:
        """Produce structured drift report."""
        ks_stat = self.kolmogorov_smirnov_statistic(current_residuals)
        coverage_drop = max(nominal_coverage - empirical_coverage, 0.0)

        if ks_stat > 0.20 or coverage_drop > 0.10:
            severity = "SEVERE_DRIFT"
            recommended_action = "IMMEDIATE_RECALIBRATION_REQUIRED"
        elif ks_stat > 0.10 or coverage_drop > 0.05:
            severity = "MILD_DRIFT"
            recommended_action = "MONITOR_AND_QUEUE_RECALIBRATION"
        else:
            severity = "NORMAL"
            recommended_action = "NONE"

        return {
            "ks_statistic": ks_stat,
            "nominal_coverage": nominal_coverage,
            "empirical_coverage": empirical_coverage,
            "coverage_drop": coverage_drop,
            "severity": severity,
            "recommended_action": recommended_action,
        }
