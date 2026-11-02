"""Lazy forecast samples: allocate one window, never all overlapping tensors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import ForecastOrigin, SplitManifest, TargetValidityPolicy
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits


def validity_mask(values: NDArray[np.float64], policy: TargetValidityPolicy) -> NDArray[np.bool_]:
    mask = np.isfinite(values)
    if policy.mask_zero:
        mask &= values != 0
    return mask


def numerical_values(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Explicit float64 model view. Reject integer precision loss, preserve NaN/inf."""
    for i, dtype in enumerate(frame.dtypes):
        if dtype.kind in "iu" and (
            (frame.iloc[:, i] > 2**53).any() or (frame.iloc[:, i] < -(2**53)).any()
        ):
            raise DataError("Model float64 conversion would lose large-integer precision")
    return frame.to_numpy(dtype=np.float64, copy=True)


def calendar_features(
    index: pd.DatetimeIndex, sensors: int, tod: bool, dow: bool
) -> NDArray[np.float64]:
    features: list[NDArray[np.float64]] = []
    if tod:
        # Source wall-clock fields preserve timezone, including repeated DST wall times.
        seconds = np.asarray(index.hour * 3600 + index.minute * 60 + index.second, dtype=np.float64)
        seconds += np.asarray(index.microsecond, dtype=np.float64) / 1e6
        seconds += np.asarray(index.nanosecond, dtype=np.float64) / 1e9
        features.append((seconds / 86400)[:, None])
    if dow:
        features.append(np.eye(7, dtype=np.float64)[np.asarray(index.dayofweek)])
    flat = (
        np.concatenate(features, axis=1)
        if features
        else np.empty((len(index), 0), dtype=np.float64)
    )
    return np.broadcast_to(flat[:, None, :], (len(index), sensors, flat.shape[1]))


@dataclass(frozen=True)
class ForecastSample:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    input_valid: NDArray[np.bool_]
    target_valid: NDArray[np.bool_]
    future_known: NDArray[np.float64]
    input_positions: tuple[int, ...]
    target_positions: tuple[int, ...]
    input_timestamps: tuple[str, ...]
    target_timestamps: tuple[str, ...]
    origin: ForecastOrigin
    sensor_ids: tuple[str, ...]
    fold_id: str
    partition: str
    feature_metadata: dict[str, object]

    @property
    def reference_y(self) -> NDArray[np.float64]:
        """Upstream generator also appends calendar channels to y; not future inputs."""
        return np.concatenate((self.y, self.future_known), axis=-1)

    def preview(self) -> dict[str, object]:
        return {
            "fold": self.fold_id,
            "partition": self.partition,
            "origin": self.origin.timestamp,
            "origin_position": self.origin.position,
            "input_positions": list(self.input_positions),
            "target_positions": list(self.target_positions),
            "input_timestamps": list(self.input_timestamps),
            "target_timestamps": list(self.target_timestamps),
            "sensor_ids": list(self.sensor_ids),
            "x_shape": list(self.x.shape),
            "y_shape": list(self.y.shape),
            "valid_targets": int(self.target_valid.sum()),
            "feature_metadata": self.feature_metadata,
        }


class WindowCollection:
    def __init__(
        self, dataset: EvaluationDataset, split: SplitManifest, *, fold: int, partition: str
    ) -> None:
        # Recompute boundaries from timestamps to reject fabricated or mismatched partitions.
        if plan_splits(dataset.timestamps, split.config) != split:
            raise DataError("Split does not match dataset timestamps/configuration")
        if type(fold) is not int or not 0 <= fold < len(split.folds):
            raise DataError("Invalid fold index")
        self.dataset = dataset
        self.split = split
        self.fold = split.folds[fold]
        self.partition = self.fold.partition(partition)

    def __len__(self) -> int:
        return self.partition.sample_count

    def __getitem__(self, item: int) -> ForecastSample:
        if type(item) is not int or not 0 <= item < len(self):
            raise IndexError("Window index out of bounds")
        remaining = item
        origin = -1
        for run in self.partition.samples:
            size = run.stop - run.start
            if remaining < size:
                origin = run.start + remaining
                break
            remaining -= size
        config = self.split.config
        history, horizon = config.window.history, config.window.horizon
        latest = origin - config.features.observed_delay_steps
        inputs = tuple(range(latest - history + 1, latest + 1))
        targets = tuple(range(origin + 1, origin + horizon + 1))
        frame = self.dataset.frame
        input_values = numerical_values(frame.iloc[inputs[0] : inputs[-1] + 1])[:, :, None]
        target_values = numerical_values(frame.iloc[targets[0] : targets[-1] + 1])[:, :, None]
        index = self.dataset.timestamps
        f = config.features
        historical_calendar = calendar_features(
            index[list(inputs)], len(frame.columns), f.time_of_day, f.day_of_week
        )
        future_calendar = calendar_features(
            index[list(targets)], len(frame.columns), f.time_of_day, f.day_of_week
        )
        x = np.concatenate((input_values, historical_calendar), axis=-1)
        input_mask = np.concatenate(
            (
                validity_mask(input_values, config.validity),
                np.ones(historical_calendar.shape, dtype=bool),
            ),
            axis=-1,
        )
        return ForecastSample(
            x,
            target_values,
            input_mask,
            validity_mask(target_values, config.validity),
            future_calendar,
            inputs,
            targets,
            tuple(index[i].isoformat() for i in inputs),
            tuple(index[i].isoformat() for i in targets),
            ForecastOrigin(origin, index[origin].isoformat(), latest),
            self.dataset.sensor_ids,
            self.fold.fold_id,
            self.partition.name,
            {
                "speed": "historical_observed",
                "calendar": "known_future",
                "y": "target_only",
                "unknown_future_measurements": "forbidden",
                "static_metadata": "not implemented",
                "timezone": str(index.tz) if index.tz else None,
                "observed_delay_steps": f.observed_delay_steps,
            },
        )
