"""No-fit models, train-only seasonal averages and regularized direct autoregression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from urbangraph.baselines.contracts import (
    MODEL_VERSION,
    FloatArray,
    IntArray,
    ModelConfig,
    Prediction,
)
from urbangraph.baselines.history import ForecastBatch, TrainingData
from urbangraph.baselines.state import decode_array, encode_array
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.windows import validity_mask


def _last(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    valid = np.isfinite(values)
    indices = np.where(valid, np.arange(values.shape[1])[None, :, None], -1).max(axis=1)
    selected = np.take_along_axis(values, np.maximum(indices, 0)[:, None, :], axis=1)[:, 0]
    selected[indices < 0] = np.nan
    return selected, indices.astype(np.float64)


def slots(index: pd.DatetimeIndex, mode: str) -> IntArray:
    tod = np.asarray(index.hour * 12 + index.minute // 5, dtype=np.int64)
    if mode == "global":
        return np.zeros(len(index), dtype=np.int64)
    if mode == "time-of-week":
        tod += np.asarray(index.dayofweek, dtype=np.int64) * 288
    return tod


@dataclass(frozen=True)
class FittedState:
    sensor_ids: tuple[str, ...]
    parameters: FloatArray
    support: FloatArray
    lineage: dict[str, object]

    def __post_init__(self) -> None:
        if (
            not self.sensor_ids
            or len(set(self.sensor_ids)) != len(self.sensor_ids)
            or not np.isfinite(self.parameters).all()
            or not np.isfinite(self.support).all()
            or np.any(self.support < 0)
            or (
                self.parameters.ndim == 2
                and (
                    self.parameters.shape != self.support.shape
                    or self.parameters.shape[1] != len(self.sensor_ids)
                )
            )
            or (
                self.parameters.ndim == 3
                and (
                    self.parameters.shape[0] != len(self.sensor_ids)
                    or self.support.shape != (len(self.sensor_ids),)
                )
            )
            or self.parameters.ndim not in {2, 3}
        ):
            raise DataError("Invalid fitted state dimensions or values")

    @classmethod
    def from_dict(cls, value: object) -> FittedState:
        obj = object_dict(value)
        sensors = obj.get("sensor_ids")
        if (
            set(obj) != {"sensor_ids", "parameters", "support", "lineage"}
            or not isinstance(sensors, list | tuple)
            or any(not isinstance(s, str) or not s for s in sensors)
        ):
            raise DataError("Invalid fitted state schema")
        return cls(
            tuple(str(s) for s in sensors),
            decode_array(obj["parameters"]),
            decode_array(obj["support"]),
            object_dict(obj["lineage"]),
        )

    def to_dict(self) -> dict[str, object]:
        # Unsupported parameters use finite placeholders plus explicit support, never forecasts.
        return {
            "sensor_ids": self.sensor_ids,
            "parameters": encode_array(self.parameters),
            "support": encode_array(self.support),
            "lineage": self.lineage,
        }

    @property
    def state_id(self) -> str:
        return identity(self.to_dict())


@dataclass(frozen=True)
class Baseline:
    config: ModelConfig
    state: FittedState | None = None
    interval_seconds: int = 300

    def __post_init__(self) -> None:
        if type(self.interval_seconds) is not int or self.interval_seconds not in {300, 3600}:
            raise DataError("Supported baseline cadence is 300 or 3600 seconds")

    def predict(self, batch: ForecastBatch) -> Prediction:
        c = self.config
        if self.state is not None and batch.sensor_ids != self.state.sensor_ids:
            raise DataError("Fitted sensor ordering mismatch")
        b, horizon = batch.target_positions.shape
        history = batch.history.preceding(c.history_fallback)
        last, last_index = _last(history)
        persistence = np.broadcast_to(last[:, None, :], (b, horizon, len(batch.sensor_ids))).copy()
        fallback = np.zeros(persistence.shape, dtype=bool)
        if c.model == "persistence":
            result = persistence
            fallback = np.broadcast_to(
                (last_index < history.shape[1] - 1)[:, None, :], result.shape
            )
        elif c.model == "drift":
            first_indices = np.argmax(np.isfinite(history), axis=1)
            first = np.take_along_axis(history, first_indices[:, None, :], axis=1)[:, 0]
            span = last_index - first_indices
            slope = np.divide(last - first, span, out=np.full(last.shape, np.nan), where=span > 0)
            distance = (
                np.arange(1, horizon + 1)[None, :, None]
                + batch.delay
                + history.shape[1]
                - 1
                - last_index[:, None, :]
            )
            result = last[:, None, :] + slope[:, None, :] * distance
            if c.bounds is not None:
                result = np.clip(result, *c.bounds)
        elif c.model in {"daily-seasonal", "weekly-seasonal"}:
            days = 1 if c.model == "daily-seasonal" else 7
            if c.seasonal_policy == "positional":
                period = (86400 // self.interval_seconds) * days
                reference = batch.target_positions - period
                if self.interval_seconds == 3600:
                    # Latest same-season observation that satisfies declared publication latency.
                    unavailable = np.maximum(reference - batch.history.latest[:, None], 0)
                    reference -= ((unavailable + period - 1) // period) * period
                result = batch.history.retrieve(reference)
            else:
                # Naive source labels: calendar-label matching, no elapsed-time claim.
                if batch.target_labels.tz is not None:
                    raise DataError("Timestamp-label seasonal mode requires naive timestamps")
                result = batch.history.matching(
                    batch.target_labels - pd.Timedelta(days=days), (b, horizon)
                )
        elif c.model == "historical-average":
            state = self._state()
            indices = slots(batch.target_labels, c.average)
            result = state.parameters[indices].reshape(persistence.shape).copy()
            supported = state.support[indices].reshape(persistence.shape) > 0
            result[~supported] = np.nan
        else:
            state = self._state()
            lagged = batch.history.preceding(c.lags)
            result = np.einsum("bln,nhl->bhn", lagged, state.parameters[:, :, 1:])
            result += state.parameters[:, :, 0].T[None]
            result[:, :, state.support == 0] = np.nan
            if result.shape != persistence.shape:
                raise DataError("Fitted horizon mismatch")
        missing = ~np.isfinite(result)
        if c.model != "persistence" and c.fallback == "persistence":
            fallback = missing & np.isfinite(persistence)
            result = np.where(missing, persistence, result)
        available = np.isfinite(result)
        result[~available] = np.nan
        return Prediction(
            result[..., None], available[..., None], (fallback & available)[..., None]
        )

    def _state(self) -> FittedState:
        if self.state is None:
            raise DataError("Model requires train-only fitted state")
        return self.state


def fit_model(
    config: ModelConfig, training: TrainingData, horizon: int, *, interval_seconds: int = 300
) -> Baseline:
    if config.model not in {"historical-average", "autoregressive"}:
        raise DataError("No-fit models must be constructed without training")
    if interval_seconds not in {300, 3600} or training.interval_seconds != interval_seconds:
        raise DataError("Statistical baseline cadence must match the explicit sampling contract")
    if interval_seconds != 300 and config.model != "autoregressive":
        raise DataError("Hourly fitting currently supports autoregression only")
    values = training.values
    valid = validity_mask(values, training.validity)
    lineage = training.lineage | {
        "model": config.to_dict(),
        "version": MODEL_VERSION,
        "mask_zero": training.validity.mask_zero,
        "horizon": horizon,
    }
    if interval_seconds != 300:
        lineage["interval_seconds"] = interval_seconds
    support: FloatArray
    parameters: FloatArray
    if config.model == "historical-average":
        groups = slots(training.timestamps, config.average)
        count = {"global": 1, "time-of-day": 288, "time-of-week": 2016}[config.average]
        sums = np.zeros((count, values.shape[1]), dtype=np.float64)
        support = np.zeros_like(sums)
        np.add.at(sums, groups, np.where(valid, values, 0))
        np.add.at(support, groups, valid)
        parameters = np.divide(sums, support, out=np.zeros_like(sums), where=support > 0)
    else:
        lags, delay = config.lags, training.delay
        origins = np.arange(lags - 1 + delay, len(values) - horizon, dtype=np.int64)
        if not origins.size:
            raise DataError("Insufficient training rows for autoregression")
        if not training.positional:
            ns = np.asarray(training.timestamps.as_unit("ns").astype("int64"))
            bad = np.r_[0, np.cumsum(np.diff(ns) != training.interval_seconds * 10**9)]
            origins = origins[bad[origins + horizon] == bad[origins - delay - lags + 1]]
        inputs = origins[:, None] - delay + np.arange(1 - lags, 1)
        targets = origins[:, None] + np.arange(1, horizon + 1)
        parameters = np.zeros((values.shape[1], horizon, lags + 1), dtype=np.float64)
        support = np.zeros(values.shape[1], dtype=np.float64)
        # One sensor at a time bounds memory; multi-output ridge solves all horizons together.
        for sensor in range(values.shape[1]):
            keep = valid[inputs, sensor].all(axis=1) & valid[targets, sensor].all(axis=1)
            x, y = values[inputs[keep], sensor], values[targets[keep], sensor]
            support[sensor] = len(x)
            if len(x) < lags + 1:
                support[sensor] = 0
                continue
            xm, ym = x.mean(axis=0), y.mean(axis=0)
            centered = x - xm
            coef = np.linalg.solve(
                centered.T @ centered + config.ridge * np.eye(lags), centered.T @ (y - ym)
            )
            parameters[sensor, :, 1:] = coef.T
            parameters[sensor, :, 0] = ym - xm @ coef
    if not np.isfinite(parameters).all():
        raise DataError("Nonfinite statistical fitted parameters")
    return Baseline(
        config, FittedState(training.sensor_ids, parameters, support, lineage), interval_seconds
    )
