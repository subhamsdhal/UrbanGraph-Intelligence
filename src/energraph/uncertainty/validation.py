"""Structural and temporal validation of numeric forecast populations."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from energraph.uncertainty.conformal import validate_arrays
from urbangraph.data import DataError


def validate_partitions(
    arrays: dict[str, NDArray[Any]],
    region_ids: tuple[str, ...],
    delay: int,
) -> None:
    """Reject malformed axes and any unavailable earlier-partition labels.

    Fitted-model training lineage is additionally checked by reconstruction against
    the immutable source model/plan. This validator never fits a transform.
    """
    if not region_ids or len(set(region_ids)) != len(region_ids) or delay < 0:
        raise DataError("Invalid population axis or delay")
    previous_end: int | None = None
    epoch_offsets: list[int] = []
    for partition in ("validation", "calibration", "test"):
        required = [
            partition + suffix
            for suffix in ("_truth", "_origins", "_targets_ns", "_lightgbm", "_persistence")
        ]
        if any(key not in arrays for key in required):
            raise DataError("Missing population arrays")
        truth = arrays[partition + "_truth"]
        origins = arrays[partition + "_origins"]
        targets = arrays[partition + "_targets_ns"]
        if (
            truth.ndim != 3
            or truth.shape[1:] != (24, len(region_ids))
            or origins.dtype != np.int64
            or origins.shape != (len(truth),)
            or not len(origins)
            or np.any(origins < 0)
            or np.any(np.diff(origins) <= 0)
            or targets.dtype != np.int64
            or targets.shape != (len(origins), 24)
        ):
            raise DataError("Invalid origin, target or region/horizon alignment")
        if previous_end is not None and int(origins[0]) < previous_end:
            raise DataError("Partition label availability overlap")
        previous_end = int(origins[-1]) + 24 + delay
        offsets = targets - (origins[:, None] + np.arange(1, 25)) * 3600 * 10**9
        if not np.all(offsets == offsets[0, 0]):
            raise DataError("Target timestamp/horizon mismatch")
        epoch_offsets.append(int(offsets[0, 0]))
        for model in ("lightgbm", "persistence"):
            validate_arrays(truth, arrays[partition + "_" + model], np.isfinite(truth))
    if len(set(epoch_offsets)) != 1:
        raise DataError("Partition timestamp origins differ")
