"""NumPy metrics in original target units, with explicit masks and populations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import PROTOCOLS, MetricResult, TargetValidityPolicy
from urbangraph.evaluation.windows import validity_mask

MetricName = Literal["mae", "rmse", "mape"]
DEFAULT_VALIDITY = TargetValidityPolicy()


def metric(
    name: MetricName,
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    validity: TargetValidityPolicy | None = None,
    mask: NDArray[np.bool_] | None = None,
    mape_zero: str = "exclude",
    value_space: str = "original",
) -> MetricResult:
    """Without a validity policy all targets must be finite. Masks are never broadcast."""
    if name not in {"mae", "rmse", "mape"} or mape_zero not in {"exclude", "error"}:
        raise DataError("Unknown metric or MAPE zero policy")
    if value_space != "original":
        raise DataError("Inverse-transform predictions and targets before physical-unit metrics")
    a, p = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if a.shape != p.shape or a.size == 0:
        raise DataError("Metric shapes must match and be nonempty")
    selected = np.ones(a.shape, dtype=bool) if validity is None else validity_mask(a, validity)
    if mask is not None:
        if mask.dtype != np.bool_ or mask.shape != a.shape:
            raise DataError("Explicit mask must be boolean and match target shape exactly")
        selected &= mask
    if np.any(selected & ~np.isfinite(a)):
        raise DataError("Nonfinite targets require an explicit masking policy")
    zero_count = int(np.count_nonzero(selected & (a == 0))) if name == "mape" else 0
    if name == "mape":
        if mape_zero == "error" and zero_count:
            raise DataError("MAPE has zero denominators")
        selected &= a != 0
    count = int(selected.sum())
    if not count:
        raise DataError("Empty valid metric population")
    if np.any(~np.isfinite(p[selected])):
        raise DataError("Nonfinite prediction at a scored target")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        errors = np.abs(p[selected] - a[selected])
        if name == "mape":
            errors = errors / np.abs(a[selected])
        if not np.all(np.isfinite(errors)):
            raise DataError("Metric arithmetic overflow")
        largest = float(errors.max())
        # Rescale first so neither summation nor squaring unnecessarily overflows.
        normalized = errors / largest if largest else errors
        value = (
            float(np.sqrt(np.mean(normalized**2)) * largest)
            if name == "rmse"
            else float(np.mean(normalized) * largest)
        )
        if name == "mape":
            value *= 100.0
    if not np.isfinite(value):
        raise DataError("Metric result exceeds float64 range")
    return MetricResult(name, value, count, int(a.size) - count, zero_count)


def mae(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> MetricResult:
    return metric("mae", actual, predicted)


def rmse(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> MetricResult:
    return metric("rmse", actual, predicted)


def mape(
    actual: NDArray[np.float64], predicted: NDArray[np.float64], *, zero_policy: str = "exclude"
) -> MetricResult:
    return metric("mape", actual, predicted, mape_zero=zero_policy)


def masked_mae(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    validity: TargetValidityPolicy = DEFAULT_VALIDITY,
    mask: NDArray[np.bool_] | None = None,
) -> MetricResult:
    return metric("mae", actual, predicted, validity=validity, mask=mask)


def masked_rmse(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    validity: TargetValidityPolicy = DEFAULT_VALIDITY,
    mask: NDArray[np.bool_] | None = None,
) -> MetricResult:
    return metric("rmse", actual, predicted, validity=validity, mask=mask)


def masked_mape(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    validity: TargetValidityPolicy = DEFAULT_VALIDITY,
    mask: NDArray[np.bool_] | None = None,
    zero_policy: str = "exclude",
) -> MetricResult:
    return metric("mape", actual, predicted, validity=validity, mask=mask, mape_zero=zero_policy)


def evaluate_predictions(
    actual: NDArray[np.float64],
    predicted: NDArray[np.float64],
    *,
    sensor_ids: tuple[str, ...],
    protocol: str,
    validity: TargetValidityPolicy = DEFAULT_VALIDITY,
    mask: NDArray[np.bool_] | None = None,
    fold_ids: tuple[str, ...] | None = None,
    partitions: tuple[str, ...] | None = None,
    value_space: str = "original",
    units: str | None = None,
) -> dict[str, object]:
    """[samples,horizon,sensors,1], micro means; overlapping forecast cells count separately."""
    a, p = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if (
        a.ndim != 4
        or a.shape != p.shape
        or a.shape[-1] != 1
        or not a.size
        or a.shape[2] != len(sensor_ids)
        or len(set(sensor_ids)) != len(sensor_ids)
        or any(not isinstance(s, str) or not s for s in sensor_ids)
        or protocol not in PROTOCOLS
        or value_space != "original"
    ):
        raise DataError("Invalid evaluation dimensions, labels, protocol or value space")
    selected = validity_mask(a, validity)
    if mask is not None:
        if mask.dtype != np.bool_ or mask.shape != a.shape:
            raise DataError("Evaluation mask must match shape and be boolean")
        selected &= mask
    if not selected.any():
        raise DataError("Empty valid metric population")

    def summary(
        truth: NDArray[np.float64], prediction: NDArray[np.float64], valid: NDArray[np.bool_]
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in ("mae", "rmse", "mape"):
            scored = valid & (truth != 0) if name == "mape" else valid
            if not scored.any():
                result[name] = {
                    "value": None,
                    "valid_count": 0,
                    "masked_count": int(truth.size),
                    "status": "empty_population",
                    "zero_denominator_count": int(np.count_nonzero(valid & (truth == 0)))
                    if name == "mape"
                    else 0,
                }
            else:
                result[name] = metric(
                    name, truth, prediction, validity=validity, mask=valid
                ).to_dict()
        return result

    result: dict[str, object] = {
        "protocol": protocol,
        "units": units,
        "value_space": value_space,
        "mape_units": "percent",
        "mape_zero_policy": "exclude",
        "mask_zero": validity.mask_zero,
        "aggregation": "micro over forecast cells; repeated targets counted per forecast",
        "overall": summary(a, p, selected),
        "by_horizon": [
            {"horizon_step": h + 1, "metrics": summary(a[:, h], p[:, h], selected[:, h])}
            for h in range(a.shape[1])
        ],
        "by_sensor": [
            {"sensor_id": sensor, "metrics": summary(a[:, :, i], p[:, :, i], selected[:, :, i])}
            for i, sensor in enumerate(sensor_ids)
        ],
    }
    for name, labels in (("by_fold", fold_ids), ("by_partition", partitions)):
        if labels is not None:
            if len(labels) != a.shape[0] or any(not isinstance(s, str) or not s for s in labels):
                raise DataError("One nonempty grouping label is required per forecast sample")
            groups = []
            label_array = np.asarray(labels)
            for label in sorted(set(labels)):
                rows = label_array == label
                groups.append(
                    {"label": label, "metrics": summary(a[rows], p[rows], selected[rows])}
                )
            result[name] = groups
    return result
