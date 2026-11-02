"""Causal selected-cell features; labels never enter this API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from urbangraph.baselines.contracts import IntArray
from urbangraph.baselines.history import HistoricalAccess
from urbangraph.boosting.contracts import FeatureSpec
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import FeatureAvailabilityPolicy


@dataclass(frozen=True)
class FeatureRows:
    values: NDArray[np.float32]
    names: tuple[str, ...]
    groups: tuple[str, ...]
    unavailable_seasonal: NDArray[np.bool_]


def build_features(
    spec: FeatureSpec,
    history: HistoricalAccess,
    sensors: IntArray,
    horizons: IntArray,
    target_labels: pd.DatetimeIndex,
    sensor_ids: tuple[str, ...],
    delay: int,
) -> FeatureRows:
    rows = len(sensors)
    if (
        sensors.shape != horizons.shape
        or rows != len(target_labels)
        or np.any(horizons < 1)
        or len(set(sensor_ids)) != len(sensor_ids)
        or np.any(sensors >= len(sensor_ids))
    ):
        raise DataError("Misaligned feature row identities")
    availability = FeatureAvailabilityPolicy(True, True, delay)
    availability.validate_request("speed", "historical_observed", future=False)
    columns: list[NDArray[np.float64]] = []
    names: list[str] = []
    groups: list[str] = []

    def add(name: str, group: str, values: NDArray[np.float64], missing: bool = False) -> None:
        clean = np.asarray(values, dtype=np.float64).copy()
        clean[~np.isfinite(clean)] = np.nan
        columns.append(clean)
        names.append(name)
        groups.append(group)
        if missing and spec.missing_indicators:
            columns.append((~np.isfinite(clean)).astype(np.float64))
            names.append(name + "_missing")
            groups.append(group)

    lookback = max((2, *spec.lags, *spec.rolling))
    past = history.retrieve_sensors(history.latest[:, None] + np.arange(1 - lookback, 1), sensors)
    for lag in spec.lags:
        add(f"lag_{lag}", "lags", past[:, -lag], True)
    add("horizon", "horizon", horizons.astype(np.float64))
    level = int(spec.variant[1])
    if level >= 1:
        availability.validate_request("time_of_day", "known_future", future=True)
        availability.validate_request("day_of_week", "known_future", future=True)
        tod = (
            np.asarray(
                target_labels.hour * 3600 + target_labels.minute * 60 + target_labels.second,
                dtype=np.float64,
            )
            / 86400
        )
        dow = np.asarray(target_labels.dayofweek, dtype=np.float64)
        for name, values in (
            ("tod_sin", np.sin(2 * np.pi * tod)),
            ("tod_cos", np.cos(2 * np.pi * tod)),
            ("dow_sin", np.sin(2 * np.pi * dow / 7)),
            ("dow_cos", np.cos(2 * np.pi * dow / 7)),
            ("weekend", (dow >= 5).astype(np.float64)),
        ):
            add(name, "calendar", values)
    if level >= 2:
        for window in spec.rolling:
            values = past[:, -window:]
            finite = np.isfinite(values)
            count = finite.sum(axis=1)
            total = np.where(finite, values, 0).sum(axis=1)
            mean = np.divide(total, count, out=np.full(rows, np.nan), where=count > 0)
            squared = np.where(finite, (values - mean[:, None]) ** 2, 0).sum(axis=1)
            std = np.sqrt(np.divide(squared, count, out=np.full(rows, np.nan), where=count > 0))
            low = np.where(finite, values, np.inf).min(axis=1)
            high = np.where(finite, values, -np.inf).max(axis=1)
            for suffix, val in (("mean", mean), ("std", std), ("min", low), ("max", high)):
                add(f"rolling_{window}_{suffix}", "rolling", val, True)
        add("difference_1", "rolling", past[:, -1] - past[:, -2], True)
    unavailable = np.zeros(rows, dtype=bool)
    if level >= 3:
        for days in (1, 7):
            if spec.seasonal_policy == "positional":
                positions = history.latest + delay + horizons - 288 * days
                seasonal = history.retrieve_sensors(positions[:, None], sensors)[:, 0]
            else:
                if target_labels.tz is not None:
                    raise DataError("Timestamp-label seasonal features require naive labels")
                seasonal = history.matching_sensors(
                    target_labels - pd.Timedelta(days=days), sensors
                )[:, 0]
            unavailable |= ~np.isfinite(seasonal)
            add(f"seasonal_{days}d", "seasonal", seasonal, True)
    if level >= 4:
        add("sensor_id", "sensor", sensors.astype(np.float64))
    matrix = np.column_stack(columns)
    with np.errstate(over="ignore"):
        result = matrix.astype(np.float32)
    if np.any(np.isfinite(matrix) & ~np.isfinite(result)):
        raise DataError("Feature exceeds float32 range")
    return FeatureRows(result, tuple(names), tuple(groups), unavailable)
