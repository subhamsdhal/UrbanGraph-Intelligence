"""Strict model configuration and prediction-only input contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.contracts import integer

MODEL_IDS = (
    "persistence",
    "drift",
    "daily-seasonal",
    "weekly-seasonal",
    "historical-average",
    "autoregressive",
)
MODEL_VERSION = "1.0"
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ModelConfig:
    model: str = "persistence"
    schema_version: str = "1.0"
    history_fallback: int = 12
    fallback: str = "persistence"
    seasonal_policy: str = "positional"
    average: str = "time-of-week"
    lags: int = 12
    ridge: float = 1.0
    bounds: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.model not in MODEL_IDS or self.schema_version != "1.0":
            raise DataError("Unknown model or model configuration schema")
        integer(self.history_fallback, "history_fallback", 1)
        integer(self.lags, "lags", 1)
        if self.lags > 288 or self.history_fallback > 2016:
            raise DataError("History limits exceeded")
        if self.fallback not in {"persistence", "unavailable"}:
            raise DataError("Unknown fallback policy")
        if self.seasonal_policy not in {"positional", "timestamp-label"}:
            raise DataError("Unknown seasonal temporal policy")
        if self.average not in {"global", "time-of-day", "time-of-week"}:
            raise DataError("Unknown historical average grouping")
        if type(self.ridge) not in (int, float) or not np.isfinite(self.ridge) or self.ridge <= 0:
            raise DataError("Ridge regularization must be finite and positive")
        if self.bounds is not None and (
            len(self.bounds) != 2
            or any(type(x) not in (int, float) for x in self.bounds)
            or not np.all(np.isfinite(self.bounds))
            or self.bounds[0] >= self.bounds[1]
        ):
            raise DataError("Bounds require two increasing finite numbers")
        if self.bounds is not None and self.model != "drift":
            raise DataError("Output bounds apply only to drift")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def model_config(value: object) -> ModelConfig:
    obj = object_dict(value)
    defaults = ModelConfig().to_dict()
    if set(obj) - set(defaults):
        raise DataError("Unknown model configuration fields")
    d = defaults | obj
    for key in ("model", "schema_version", "fallback", "seasonal_policy", "average"):
        if not isinstance(d[key], str):
            raise DataError(f"{key} must be text")
    ridge = d["ridge"]
    if isinstance(ridge, bool) or not isinstance(ridge, int | float):
        raise DataError("ridge must be numeric")
    bounds = d["bounds"]
    parsed: tuple[float, float] | None = None
    if bounds is not None:
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise DataError("Invalid bounds")
        if any(type(x) not in (int, float) for x in bounds):
            raise DataError("Bounds must be numeric")
        parsed = (float(bounds[0]), float(bounds[1]))
    return ModelConfig(
        str(d["model"]),
        str(d["schema_version"]),
        integer(d["history_fallback"], "history_fallback", 1),
        str(d["fallback"]),
        str(d["seasonal_policy"]),
        str(d["average"]),
        integer(d["lags"], "lags", 1),
        float(ridge),
        parsed,
    )


@dataclass(frozen=True)
class Prediction:
    values: FloatArray
    available: BoolArray
    fallback: BoolArray

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 4
            or self.values.shape[-1] != 1
            or self.available.shape != self.values.shape
            or self.fallback.shape != self.values.shape
            or self.available.dtype != np.bool_
            or self.fallback.dtype != np.bool_
            or np.any(self.available != np.isfinite(self.values))
            or np.any(self.fallback & ~self.available)
        ):
            raise DataError("Invalid prediction shape, availability or fallback mask")
