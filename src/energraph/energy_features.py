"""Causal horizon-conditioned energy features with explicit hindsight separation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energraph.variables import variables
from urbangraph.data import DataError


@dataclass(frozen=True)
class EnergyFeatures:
    inputs: tuple[str, ...] = ("load",)
    latencies: tuple[int, ...] = (24,)
    lags: tuple[int, ...] = (0, 1, 2, 6, 12, 23, 47, 143)
    rolling: tuple[int, ...] = (24, 168)
    calendar: bool = True
    weather_mode: str = "W0"
    version: str = "energy-features-1.0"

    def __post_init__(self) -> None:
        selected = variables(self.inputs)
        if (
            self.version != "energy-features-1.0"
            or self.weather_mode not in {"W0", "W1"}
            or len(self.latencies) != len(selected)
            or any(type(n) is not int or not 0 <= n <= 168 for n in self.latencies)
            or type(self.calendar) is not bool
            or any(type(n) is not int or not 0 <= n < 168 for n in self.lags)
            or any(type(n) is not int or not 1 <= n <= 168 for n in self.rolling)
            or not self.lags
            or len(set(self.lags)) != len(self.lags)
            or len(set(self.rolling)) != len(self.rolling)
        ):
            raise DataError("Invalid energy feature/availability policy")
        if any(v.observation_type == "known-calendar" for v in selected):
            raise DataError("Calendar must be generated from forecast timestamps")
        if self.weather_mode == "W0" and any(v.kind == "weather" for v in selected):
            raise DataError("Reanalysis requires explicitly labeled W1 hindsight mode")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def require_issued_weather(
    valid_time: datetime, origin: datetime, issued_at: datetime | None
) -> None:
    if any(t.tzinfo is None for t in (valid_time, origin)) or issued_at is None:
        raise DataError("W3 requires actual timezone-aware forecast issue and valid times")
    if issued_at.tzinfo is None or issued_at > origin or valid_time < issued_at:
        raise DataError("Weather forecast was not issued at origin or has invalid lead time")


def build_features(
    frame: pd.DataFrame,
    origins: NDArray[np.int64],
    horizons: NDArray[np.int64],
    spec: EnergyFeatures,
    *,
    lower: int = 0,
) -> tuple[NDArray[np.float32], tuple[str, ...], tuple[str, ...]]:
    if (
        origins.ndim != 1
        or origins.shape != horizons.shape
        or not len(origins)
        or np.any(horizons < 1)
        or np.any(horizons > 24)
        or np.any(origins < lower)
        or np.any(origins + horizons >= len(frame))
        or lower < 0
        or str(pd.DatetimeIndex(frame.index).tz) != "UTC"
    ):
        raise DataError("Invalid energy feature population or UTC axis")
    if np.any(
        np.diff(np.asarray(pd.DatetimeIndex(frame.index).as_unit("ns").astype("int64")))
        != 3600 * 10**9
    ):
        raise DataError("Causal positional features require a continuous hourly axis")
    cols: list[NDArray[np.float64]] = []
    names: list[str] = []
    groups: list[str] = []

    def append(value: NDArray[np.float64], name: str, group: str) -> None:
        cols.append(value)
        names.append(name)
        groups.append(group)

    append(horizons.astype(np.float64), "horizon", "horizon")
    for name, latency in zip(spec.inputs, spec.latencies, strict=True):
        if name not in frame:
            raise DataError("Missing requested variable")
        values = frame[name].to_numpy(dtype=np.float64)
        available = origins - latency

        def retrieve(
            positions: NDArray[np.int64],
            values: NDArray[np.float64] = values,
            available: NDArray[np.int64] = available,
        ) -> NDArray[np.float64]:
            if np.any(positions > available):
                raise DataError("Future observation feature rejected")
            result = values[np.clip(positions, 0, len(values) - 1)].copy()
            result[positions < lower] = np.nan
            return result

        for lag in spec.lags:
            value = retrieve(available - lag)
            append(value, f"{name}_lag_{lag}", name)
            append(np.isnan(value).astype(np.float64), f"{name}_lag_{lag}_missing", name)
        for window in spec.rolling:
            history = np.stack([retrieve(available - k) for k in range(window)], axis=1)
            finite = np.isfinite(history)
            count = finite.sum(axis=1)
            mean = np.divide(
                np.nansum(history, axis=1),
                count,
                out=np.full(len(origins), np.nan),
                where=count > 0,
            )
            append(mean, f"{name}_mean_{window}", name)
            append(1 - count / window, f"{name}_missing_fraction_{window}", name)
        # Seasonal target reference rolled back until publication latency is met.
        for period in (24, 168):
            steps = ((horizons + latency + period - 1) // period) * period
            append(retrieve(origins + horizons - steps), f"{name}_seasonal_{period}", name)
    if spec.calendar:
        target = pd.DatetimeIndex(frame.index[origins + horizons]).tz_convert("Europe/Berlin")
        for label, calendar_values, period in (
            ("hour", target.hour, 24),
            ("weekday", target.dayofweek, 7),
            ("dayofyear", target.dayofyear - 1, 366),
        ):
            angle = np.asarray(calendar_values, dtype=np.float64) * (2 * np.pi / period)
            append(np.sin(angle), f"calendar_{label}_sin", "calendar")
            append(np.cos(angle), f"calendar_{label}_cos", "calendar")
    output = np.asarray(np.column_stack(cols), dtype=np.float32)
    if np.isinf(output).any():
        raise DataError("Nonfinite feature overflow")
    return output, tuple(names), tuple(groups)
