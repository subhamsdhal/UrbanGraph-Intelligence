"""Bounded observation access; prediction code receives no targets or dataset object."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from urbangraph.baselines.contracts import FloatArray, IntArray
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import BENCHMARK_PROTOCOLS, SplitManifest, TargetValidityPolicy
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import calendar_features, numerical_values, validity_mask


class HistoricalAccess:
    """Policy guard for trusted models, not a sandbox for arbitrary Python plugins."""

    def __init__(
        self,
        values: FloatArray,
        timestamps: pd.DatetimeIndex,
        origins: IntArray,
        *,
        delay: int,
        lower: int,
        validity: TargetValidityPolicy,
    ) -> None:
        if (
            values.ndim != 2
            or len(values) != len(timestamps)
            or origins.ndim != 1
            or not origins.size
            or delay < 0
            or lower < 0
            or np.any(origins - delay < lower)
            or np.any(origins >= len(values))
        ):
            raise DataError("Invalid historical access boundaries")
        self.__values = values
        self.__timestamps = timestamps
        self.latest = origins.copy() - delay
        self.lower = lower
        self.validity = validity

    def retrieve(self, positions: IntArray) -> FloatArray:
        if positions.ndim != 2 or len(positions) != len(self.latest):
            raise DataError("One position row per forecast origin required")
        if np.any(positions > self.latest[:, None]):
            raise DataError("Future or delayed traffic observations are unavailable")
        allowed = positions >= self.lower
        result = self.__values[np.maximum(positions, self.lower)].copy()
        result[~allowed] = np.nan
        result[~validity_mask(result, self.validity)] = np.nan
        return np.asarray(result, dtype=np.float64)

    def preceding(self, steps: int) -> FloatArray:
        if type(steps) is not int or steps < 1:
            raise DataError("Positive history length required")
        return self.retrieve(self.latest[:, None] + np.arange(1 - steps, 1, dtype=np.int64))

    def retrieve_sensors(self, positions: IntArray, sensors: IntArray) -> FloatArray:
        """Retrieve one declared sensor per origin without copying the whole sensor axis."""
        if (
            positions.ndim != 2
            or len(positions) != len(self.latest)
            or sensors.shape != self.latest.shape
            or sensors.dtype != np.int64
            or np.any(sensors < 0)
            or np.any(sensors >= self.__values.shape[1])
        ):
            raise DataError("Invalid selected-sensor historical query")
        if np.any(positions > self.latest[:, None]):
            raise DataError("Future or delayed traffic observations are unavailable")
        result = np.asarray(
            self.__values[np.maximum(positions, self.lower), sensors[:, None]], dtype=np.float64
        ).copy()
        result[positions < self.lower] = np.nan
        result[~validity_mask(result, self.validity)] = np.nan
        return result

    def matching_sensors(self, labels: pd.DatetimeIndex, sensors: IntArray) -> FloatArray:
        """Naive timestamp-label matching under the same origin and context guards."""
        source = np.asarray(self.__timestamps.as_unit("ns").astype("int64"), dtype=np.int64)
        requested = np.asarray(labels.as_unit("ns").astype("int64"), dtype=np.int64)
        positions = np.asarray(np.searchsorted(source, requested), dtype=np.int64)
        clipped = np.minimum(positions, len(source) - 1)
        positions = np.where(source[clipped] == requested, positions, -1)
        return self.retrieve_sensors(positions[:, None], sensors)

    def matching(self, labels: pd.DatetimeIndex, shape: tuple[int, int]) -> FloatArray:
        source = np.asarray(self.__timestamps.as_unit("ns").astype("int64"), dtype=np.int64)
        requested = np.asarray(labels.as_unit("ns").astype("int64"), dtype=np.int64)
        positions = np.asarray(np.searchsorted(source, requested), dtype=np.int64)
        clipped = np.minimum(positions, len(source) - 1)
        positions = np.where(source[clipped] == requested, positions, -1).reshape(shape)
        return self.retrieve(positions)


@dataclass(frozen=True)
class ForecastBatch:
    x: FloatArray
    origins: IntArray
    target_positions: IntArray
    target_labels: pd.DatetimeIndex
    sensor_ids: tuple[str, ...]
    history: HistoricalAccess
    protocol: str
    delay: int


@dataclass(frozen=True)
class TrainingData:
    values: FloatArray
    timestamps: pd.DatetimeIndex
    sensor_ids: tuple[str, ...]
    validity: TargetValidityPolicy
    interval_seconds: int
    positional: bool
    delay: int
    lineage: dict[str, object]


def training_data(dataset: EvaluationDataset, split: SplitManifest, fold: int) -> TrainingData:
    if plan_splits(dataset.timestamps, split.config) != split:
        raise DataError("Training split does not match dataset")
    train = split.folds[fold].partition("train")
    start, stop = train.start, train.stop
    benchmark = split.config.protocol.name in BENCHMARK_PROTOCOLS
    if benchmark:
        start = train.samples[0].start - split.config.window.history + 1
        stop = train.samples[-1].stop
    return TrainingData(
        numerical_values(dataset.frame.iloc[start:stop]),
        dataset.timestamps[start:stop],
        dataset.sensor_ids,
        split.config.validity,
        split.config.window.interval_seconds,
        split.config.protocol.name == "dcrnn-positional-v1",
        split.config.features.observed_delay_steps,
        {
            "canonical": dataset.lineage,
            "split_id": identity(split.to_dict()),
            "fold": split.folds[fold].fold_id,
            "fit_rows": [start, stop],
            "basis": "unique training input rows"
            if benchmark
            else "unique training partition rows",
        },
    )


def make_batch(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    partition: str,
    origins: IntArray,
    values: FloatArray,
) -> ForecastBatch:
    part = split.folds[fold].partition(partition)
    permitted = np.zeros(origins.shape, dtype=bool)
    for run in part.samples:
        permitted |= (origins >= run.start) & (origins < run.stop)
    if not permitted.all():
        raise DataError("Origins outside verified evaluation population")
    c = split.config
    lower = part.start if c.protocol.context == "isolated" else 0
    access = HistoricalAccess(
        values,
        dataset.timestamps,
        origins,
        delay=c.features.observed_delay_steps,
        lower=lower,
        validity=c.validity,
    )
    positions = origins[:, None] + np.arange(1, c.window.horizon + 1, dtype=np.int64)
    inputs = access.preceding(c.window.history)
    input_positions = access.latest[:, None] + np.arange(1 - c.window.history, 1)
    calendar = calendar_features(
        dataset.timestamps[input_positions.ravel()],
        len(dataset.sensor_ids),
        c.features.time_of_day,
        c.features.day_of_week,
    )
    x = np.concatenate(
        (
            inputs[..., None],
            calendar.reshape(len(origins), c.window.history, len(dataset.sensor_ids), -1),
        ),
        axis=-1,
    )
    return ForecastBatch(
        x,
        origins.copy(),
        positions,
        dataset.timestamps[positions.ravel()],
        dataset.sensor_ids,
        access,
        c.protocol.name,
        c.features.observed_delay_steps,
    )
