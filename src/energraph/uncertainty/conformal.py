"""Finite-sample residual calibration without interpolated quantiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CalibrationPolicy:
    coverage: float = 0.9
    grouping: str = "region-horizon"
    minimum_count: int = 30
    origin_spacing: int = 1
    normalized: bool = False
    version: str = "energy-conformal-1.0"

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.coverage)
            or not 0 < self.coverage < 1
            or self.grouping not in {"global", "region", "horizon", "region-horizon"}
            or type(self.minimum_count) is not int
            or self.minimum_count < 1
            or type(self.origin_spacing) is not int
            or self.origin_spacing < 1
            or type(self.normalized) is not bool
            or self.version != "energy-conformal-1.0"
        ):
            raise DataError("Invalid calibration policy")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def finite_quantile(scores: FloatArray, coverage: float) -> tuple[float, int]:
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise DataError("Calibration requires nonempty finite scores")
    if (scores < 0).any() or not np.isfinite(coverage) or not 0 < coverage < 1:
        raise DataError("Invalid score or coverage")
    rank = ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise DataError("Insufficient samples for finite conformal threshold")
    return float(np.partition(scores, rank - 1)[rank - 1]), rank


def validate_arrays(truth: FloatArray, prediction: FloatArray, mask: NDArray[np.bool_]) -> None:
    if (
        truth.ndim != 3
        or not all(truth.shape)
        or prediction.shape != truth.shape
        or mask.shape != truth.shape
        or mask.dtype != np.bool_
        or not np.isfinite(prediction).all()
        or not np.isfinite(truth[mask]).all()
    ):
        raise DataError("Expected aligned finite [origins,horizons,regions] predictions and mask")


@dataclass(frozen=True)
class ConformalCalibrator:
    policy: CalibrationPolicy
    region_ids: tuple[str, ...]
    population_id: str
    model_id: str
    last_available_row: int
    thresholds: FloatArray
    scales: FloatArray
    counts: NDArray[np.int64]
    ranks: NDArray[np.int64]
    fallback: NDArray[np.bool_]

    def intervals(
        self,
        prediction: FloatArray,
        origins: NDArray[np.int64],
        *,
        region_ids: tuple[str, ...],
        model_id: str,
    ) -> tuple[FloatArray, FloatArray]:
        if (
            prediction.ndim != 3
            or prediction.shape[1:] != self.thresholds.shape
            or prediction.shape[0] != len(origins)
            or origins.ndim != 1
            or not len(origins)
            or not np.isfinite(prediction).all()
            or region_ids != self.region_ids
            or model_id != self.model_id
            or np.any(origins < self.last_available_row)
            or np.any(np.diff(origins) <= 0)
        ):
            raise DataError("Interval model, region, shape or temporal availability mismatch")
        width = self.thresholds * self.scales
        lower, upper = prediction - width, prediction + width
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise DataError("Nonfinite interval endpoints")
        return lower, upper

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.to_dict(),
            "region_ids": list(self.region_ids),
            "population_id": self.population_id,
            "model_id": self.model_id,
            "last_available_row": self.last_available_row,
            "thresholds": self.thresholds.tolist(),
            "scales": self.scales.tolist(),
            "counts": self.counts.tolist(),
            "ranks": self.ranks.tolist(),
            "fallback": self.fallback.tolist(),
        }


def fit_calibrator(
    truth: FloatArray,
    prediction: FloatArray,
    mask: NDArray[np.bool_],
    origins: NDArray[np.int64],
    *,
    policy: CalibrationPolicy,
    region_ids: tuple[str, ...],
    population_id: str,
    model_id: str,
    observation_delay: int,
    scales: FloatArray | None = None,
) -> ConformalCalibrator:
    validate_arrays(truth, prediction, mask)
    n, horizons, regions = truth.shape
    if (
        origins.shape != (n,)
        or origins.dtype != np.int64
        or np.any(origins < 0)
        or np.any(np.diff(origins) <= 0)
        or len(region_ids) != regions
        or len(set(region_ids)) != regions
        or any(not r for r in region_ids)
        or not population_id
        or not model_id
        or type(observation_delay) is not int
        or observation_delay < 0
    ):
        raise DataError("Invalid calibration population or identity")
    scale = np.ones((horizons, regions), dtype=np.float64) if scales is None else scales.copy()
    if (
        scale.shape != (horizons, regions)
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
        or (not policy.normalized and not np.all(scale == 1))
        or (policy.normalized and scales is None)
    ):
        raise DataError("Explicit positive permitted-data scales required")
    selected: list[int] = []
    for i, origin in enumerate(origins):
        if not selected or origin - origins[selected[-1]] >= policy.origin_spacing:
            selected.append(i)
    residual = np.abs(truth[selected] - prediction[selected]) / scale
    valid = mask[selected]
    pooled = residual[valid]
    if len(pooled) < policy.minimum_count:
        raise DataError("Insufficient global calibration support")
    global_q, global_rank = finite_quantile(pooled, policy.coverage)
    thresholds = np.empty((horizons, regions), dtype=np.float64)
    counts = np.empty((horizons, regions), dtype=np.int64)
    ranks = counts.copy()
    fallback = np.zeros((horizons, regions), dtype=np.bool_)
    for h in range(horizons):
        for r in range(regions):
            hs = slice(h, h + 1) if "horizon" in policy.grouping else slice(None)
            rs = slice(r, r + 1) if "region" in policy.grouping else slice(None)
            scores = residual[:, hs, rs][valid[:, hs, rs]]
            count = len(scores)
            if count < policy.minimum_count or ceil((count + 1) * policy.coverage) > count:
                thresholds[h, r], ranks[h, r] = global_q, global_rank
                fallback[h, r] = True
            else:
                thresholds[h, r], ranks[h, r] = finite_quantile(scores, policy.coverage)
            counts[h, r] = count
    return ConformalCalibrator(
        policy,
        region_ids,
        population_id,
        model_id,
        int(origins[-1]) + horizons + observation_delay,
        thresholds,
        scale,
        counts,
        ranks,
        fallback,
    )
