"""Vectorized descriptive audits. These statistics are never fitted transforms."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from urbangraph.datasets.contracts import CanonicalDataset, Finding, QualityReport


def _statistics(values: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
    # All-missing slices are expected and represented by JSON null, not warnings
    # emitted by NumPy. Arithmetic overflows become explicit audit findings below.
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return {
            "min": np.nanmin(values, axis=0),
            "max": np.nanmax(values, axis=0),
            "mean": np.nanmean(values, axis=0),
            "median": np.nanmedian(values, axis=0),
            "std": np.nanstd(values, axis=0, ddof=0),
            "q25": np.nanpercentile(values, 25, axis=0),
            "q75": np.nanpercentile(values, 75, axis=0),
        }


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def profile_dataset(dataset: CanonicalDataset) -> QualityReport:
    """Audit a canonical [T,N] matrix without mutating values or fitting anything."""
    frame = dataset.observations
    # Fix reduction traversal order across HDF and Arrow memory layouts. This may
    # allocate a float64 audit buffer; the source columns keep their native dtype.
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64, copy=False))
    rows, columns = values.shape
    finite_mask = np.isfinite(values)
    nan_counts = np.isnan(values).sum(axis=0)
    inf_counts = np.isinf(values).sum(axis=0)
    zero_counts = (values == 0).sum(axis=0)
    negative_counts = ((values < 0) & finite_mask).sum(axis=0)
    finite_counts = finite_mask.sum(axis=0)
    first_rows = finite_mask.argmax(axis=0)
    last_rows = rows - 1 - finite_mask[::-1].argmax(axis=0)
    finite = np.where(finite_mask, values, np.nan)
    stats = _statistics(finite)
    overall = _statistics(finite.reshape(-1, 1))
    # Use native column values here so distinct large integers cannot collapse
    # into a false constant flag when audit arithmetic converts to float64.
    constant = (finite_counts >= 2) & np.array(
        [frame.iloc[:, i][finite_mask[:, i]].nunique() == 1 for i in range(columns)], dtype=bool
    )
    low_variance = (
        (finite_counts >= 2) & (stats["std"] > 0) & (stats["std"] < dataset.config.low_variance_std)
    )
    findings = list(dataset.findings)
    sensors: list[dict[str, object]] = []
    index = pd.DatetimeIndex(frame.index)
    for i, sensor in enumerate(dataset.sensors):
        if np.dtype(sensor.dtype).kind in "iu" and (
            (frame.iloc[:, i] > 2**53).any() or (frame.iloc[:, i] < -(2**53)).any()
        ):
            findings.append(
                Finding(
                    "audit_precision",
                    "WARNING",
                    "Large integers preserved exactly; float64 descriptive statistics may round",
                    {"sensor_id": sensor.identifier, "position": i},
                )
            )
        sensor_stats = {name: _json_number(float(array[i])) for name, array in stats.items()}
        sensor_profile: dict[str, object] = {
            "sensor_id": sensor.identifier,
            "position": i,
            "observation_count": rows,
            "missing_count": int(nan_counts[i]),
            "nan_count": int(nan_counts[i]),
            "missing_percentage": float(nan_counts[i] * 100 / rows),
            "infinity_count": int(inf_counts[i]),
            "zero_count": int(zero_counts[i]),
            "negative_count": int(negative_counts[i]),
            "finite_count": int(finite_counts[i]),
            "constant": bool(constant[i]),
            "statistics": sensor_stats,
            "first_finite_timestamp": index[int(first_rows[i])].isoformat()
            if finite_counts[i]
            else None,
            "last_finite_timestamp": index[int(last_rows[i])].isoformat()
            if finite_counts[i]
            else None,
        }
        sensors.append(sensor_profile)
        for check, condition, description in (
            ("all_missing_sensor", nan_counts[i] == rows, "Entire sensor history is NaN"),
            (
                "constant_sensor",
                constant[i],
                "At least two finite observations with identical values",
            ),
            (
                "low_variance_sensor",
                low_variance[i],
                "Finite population standard deviation is below configured threshold",
            ),
            (
                "extreme_missingness",
                nan_counts[i] / rows >= dataset.config.extreme_missing_fraction,
                "Missing fraction meets configured audit threshold",
            ),
            (
                "statistics_overflow",
                finite_counts[i] > 0 and any(v is None for v in sensor_stats.values()),
                "Some descriptive statistics are not representable as finite float64",
            ),
        ):
            if condition:
                findings.append(
                    Finding(
                        check,
                        "WARNING",
                        description,
                        {
                            "sensor_id": sensor.identifier,
                            "position": i,
                            "finite_count": int(finite_counts[i]),
                            "missing_count": int(nan_counts[i]),
                        },
                    )
                )
    for check, count, severity, description in (
        (
            "infinite_values",
            int(inf_counts.sum()),
            "WARNING",
            "Infinite observations preserved and excluded from finite statistics",
        ),
        (
            "negative_values",
            int(negative_counts.sum()),
            "WARNING",
            "Negative finite observations preserved",
        ),
        (
            "zero_values",
            int(zero_counts.sum()),
            "INFO",
            "Zero observations retained as valid numerical values",
        ),
        (
            "uneven_coverage",
            int(finite_counts.max() - finite_counts.min()),
            "WARNING",
            "Sensor finite-observation counts differ",
        ),
    ):
        if count:
            findings.append(
                Finding(
                    check,
                    "INFO" if severity == "INFO" else "WARNING",
                    description,
                    {"count": count},
                )
            )
    if np.isfinite(finite).any() and any(not np.isfinite(v[0]) for v in overall.values()):
        findings.append(
            Finding(
                "statistics_overflow",
                "WARNING",
                "Some overall descriptive statistics overflow float64",
                {},
            )
        )
    findings = [
        replace(f, severity="ERROR") if f.check_id in dataset.config.reject_checks else f
        for f in findings
    ]
    metrics: dict[str, object] = {
        "timestamp_count": rows,
        "sensor_count": columns,
        "observation_count": rows * columns,
        "observed_count": rows * columns - int(nan_counts.sum()),
        "finite_count": int(finite_counts.sum()),
        "missing_count": int(nan_counts.sum()),
        "missing_percentage": float(nan_counts.sum() * 100 / values.size),
        "nan_count": int(nan_counts.sum()),
        "infinity_count": int(inf_counts.sum()),
        "zero_count": int(zero_counts.sum()),
        "negative_count": int(negative_counts.sum()),
        "constant_sensor_count": int(constant.sum()),
        "all_missing_sensor_count": int((nan_counts == rows).sum()),
        "statistics": {name: _json_number(float(array[0])) for name, array in overall.items()},
        "statistics_policy": (
            "finite observations only; population std ddof=0; float64 audit arithmetic"
        ),
        "temporal": dataset.temporal,
        "configuration": dataset.config.to_dict(),
        "source_numeric_bytes": int(frame.memory_usage(index=False, deep=True).sum()),
        "audit_float64_matrix_bytes": int(values.nbytes),
    }
    return QualityReport(
        dataset.dataset_id, dataset.raw_snapshot_id, metrics, tuple(sensors), tuple(findings)
    )
