"""Validated, serializable evaluation configuration and compact index contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

from urbangraph.data import DataError
from urbangraph.datasets.serialization import canonical_json, object_dict

EVALUATION_VERSION = "1.0"
REFERENCE_COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
REFERENCE_URL = (
    f"https://github.com/liyaguang/DCRNN/blob/{REFERENCE_COMMIT}/scripts/generate_training_data.py"
)
POSITIONAL_PROTOCOL = "dcrnn-positional-v1"
BENCHMARK_PROTOCOLS = ("dcrnn-sample-first-v1", POSITIONAL_PROTOCOL)
PROTOCOLS = (
    "dcrnn-sample-first-v1",
    "strict-chronological-v1",
    "strict-rolling-origin-v1",
    POSITIONAL_PROTOCOL,
)
PARTITIONS = ("train", "validation", "calibration", "test")


def integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DataError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class ForecastWindowSpecification:
    history: int = 12
    horizon: int = 12
    interval_seconds: int = 300

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            integer(value, name, 1)
        if self.interval_seconds > 86400:
            raise DataError("Cadence must not exceed one day")

    @property
    def input_offsets(self) -> tuple[int, ...]:
        return tuple(range(1 - self.history, 1))

    @property
    def target_offsets(self) -> tuple[int, ...]:
        return tuple(range(1, self.horizon + 1))


@dataclass(frozen=True)
class FeatureAvailabilityPolicy:
    time_of_day: bool = True
    day_of_week: bool = False
    observed_delay_steps: int = 0

    def __post_init__(self) -> None:
        if type(self.time_of_day) is not bool or type(self.day_of_week) is not bool:
            raise DataError("Calendar flags must be boolean")
        integer(self.observed_delay_steps, "observed_delay_steps")

    def validate_request(self, feature: str, category: str, *, future: bool) -> None:
        known = {"time_of_day": self.time_of_day, "day_of_week": self.day_of_week}
        if feature == "speed" and category == "historical_observed" and not future:
            return
        if category == "known_future" and known.get(feature, False):
            return
        raise DataError("Unavailable or unsupported feature request; future measurements forbidden")


@dataclass(frozen=True)
class TargetValidityPolicy:
    mask_zero: bool = False

    def __post_init__(self) -> None:
        if type(self.mask_zero) is not bool:
            raise DataError("mask_zero must be boolean")


@dataclass(frozen=True)
class PartitionBoundary:
    name: str
    start: int | str
    stop: int | str

    def __post_init__(self) -> None:
        if self.name not in PARTITIONS:
            raise DataError("Unknown partition name")
        for value in (self.start, self.stop):
            if type(value) is int:
                integer(value, "boundary")
            elif not isinstance(value, str) or not value.strip():
                raise DataError("Boundary must be a row index or ISO timestamp")


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str = "strict-rolling-origin-v1"
    context: str = "isolated"

    def __post_init__(self) -> None:
        if self.name not in PROTOCOLS or self.context not in {
            "isolated",
            "operational",
            "benchmark",
        }:
            raise DataError("Unknown evaluation protocol/context")
        if (self.name in BENCHMARK_PROTOCOLS) != (self.context == "benchmark"):
            raise DataError("Benchmark and strict context policies cannot be mixed")


@dataclass(frozen=True)
class PlanConfig:
    protocol: EvaluationProtocol = field(default_factory=EvaluationProtocol)
    window: ForecastWindowSpecification = field(default_factory=ForecastWindowSpecification)
    features: FeatureAvailabilityPolicy = field(default_factory=FeatureAvailabilityPolicy)
    validity: TargetValidityPolicy = field(default_factory=TargetValidityPolicy)
    scaling: str = "none"
    boundaries: tuple[PartitionBoundary, ...] = ()
    minimum_train: int = 72
    validation_length: int = 36
    calibration_length: int = 36
    test_length: int = 36
    stride: int = 36
    folds: int = 2
    embargo: int = 0
    seed: int | None = None

    def __post_init__(self) -> None:
        for name, cls in (
            ("protocol", EvaluationProtocol),
            ("window", ForecastWindowSpecification),
            ("features", FeatureAvailabilityPolicy),
            ("validity", TargetValidityPolicy),
        ):
            if not isinstance(getattr(self, name), cls):
                raise DataError(f"Invalid {name} contract")
        for name in ("minimum_train", "validation_length", "test_length", "stride", "folds"):
            integer(getattr(self, name), name, 1)
        for name in ("calibration_length", "embargo"):
            integer(getattr(self, name), name)
        if self.seed is not None:
            integer(self.seed, "seed")
        if self.scaling not in {"none", "global", "per_sensor"}:
            raise DataError("Scaling must be none, global or per_sensor")
        if not isinstance(self.boundaries, tuple) or any(
            not isinstance(b, PartitionBoundary) for b in self.boundaries
        ):
            raise DataError("Invalid boundary contracts")
        if self.protocol.name == PROTOCOLS[1]:
            if not self.boundaries:
                raise DataError("Chronological protocol requires explicit boundaries")
        elif self.boundaries:
            raise DataError("Boundaries apply only to chronological protocol")
        if self.protocol.name in BENCHMARK_PROTOCOLS and (
            self.window != ForecastWindowSpecification()
            or self.features.observed_delay_steps != 0
            or not self.features.time_of_day
            or self.embargo != 0
        ):
            raise DataError(
                "Benchmark requires reference offsets, cadence, time-of-day and no embargo"
            )

    def to_dict(self) -> dict[str, object]:
        # JSON normalize tuples, without permissive custom encoders.
        import json

        return object_dict(json.loads(canonical_json(asdict(self))))


def config_from_dict(value: object) -> PlanConfig:
    obj = object_dict(value)
    allowed = set(PlanConfig().to_dict())
    if set(obj) - allowed:
        raise DataError("Unknown evaluation configuration fields")
    base = PlanConfig().to_dict()
    base.update(obj)

    def nested(key: str, defaults: dict[str, object]) -> dict[str, object]:
        supplied = object_dict(base[key])
        if set(supplied) - set(defaults):
            raise DataError(f"Unknown {key} fields")
        return defaults | supplied

    proto = nested("protocol", asdict(EvaluationProtocol()))
    window = nested("window", asdict(ForecastWindowSpecification()))
    feature = nested("features", asdict(FeatureAvailabilityPolicy()))
    validity = nested("validity", asdict(TargetValidityPolicy()))
    if not isinstance(proto["name"], str) or not isinstance(proto["context"], str):
        raise DataError("Protocol fields must be strings")
    if type(feature["time_of_day"]) is not bool or type(feature["day_of_week"]) is not bool:
        raise DataError("Calendar flags must be boolean")
    if type(validity["mask_zero"]) is not bool or not isinstance(base["scaling"], str):
        raise DataError("Invalid scaling or validity policy")
    boundaries = base["boundaries"]
    if not isinstance(boundaries, list):
        raise DataError("Boundaries must be a list")
    parsed: list[PartitionBoundary] = []
    for item in boundaries:
        b = object_dict(item)
        if set(b) != {"name", "start", "stop"} or not isinstance(b["name"], str):
            raise DataError("Invalid boundary fields")
        start, stop = b["start"], b["stop"]
        if not isinstance(start, int | str) or not isinstance(stop, int | str):
            raise DataError("Invalid boundary types")
        parsed.append(PartitionBoundary(b["name"], start, stop))
    return PlanConfig(
        protocol=EvaluationProtocol(proto["name"], proto["context"]),
        window=ForecastWindowSpecification(
            *(integer(window[k], k, 1) for k in ("history", "horizon", "interval_seconds"))
        ),
        features=FeatureAvailabilityPolicy(
            feature["time_of_day"],
            feature["day_of_week"],
            integer(feature["observed_delay_steps"], "delay"),
        ),
        validity=TargetValidityPolicy(validity["mask_zero"]),
        scaling=base["scaling"],
        boundaries=tuple(parsed),
        minimum_train=integer(base["minimum_train"], "minimum_train", 1),
        validation_length=integer(base["validation_length"], "validation_length", 1),
        calibration_length=integer(base["calibration_length"], "calibration_length"),
        test_length=integer(base["test_length"], "test_length", 1),
        stride=integer(base["stride"], "stride", 1),
        folds=integer(base["folds"], "folds", 1),
        embargo=integer(base["embargo"], "embargo"),
        seed=None if base["seed"] is None else integer(base["seed"], "seed"),
    )


@dataclass(frozen=True)
class SampleIndex:
    """Compact contiguous range of forecast origins, [start, stop)."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        integer(self.start, "sample start")
        if integer(self.stop, "sample stop", 1) <= self.start:
            raise DataError("Empty sample range")


@dataclass(frozen=True)
class TemporalPartition:
    name: str
    start: int
    stop: int
    first_timestamp: str
    last_timestamp: str
    stop_timestamp_exclusive: str | None
    samples: tuple[SampleIndex, ...]
    rejected_gap_samples: int

    @property
    def sample_count(self) -> int:
        return sum(s.stop - s.start for s in self.samples)


@dataclass(frozen=True)
class RollingOriginFold:
    fold_id: str
    partitions: tuple[TemporalPartition, ...]

    def partition(self, name: str) -> TemporalPartition:
        for partition in self.partitions:
            if partition.name == name:
                return partition
        raise DataError("Unknown partition for fold")


@dataclass(frozen=True)
class SplitManifest:
    config: PlanConfig
    folds: tuple[RollingOriginFold, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "folds": [asdict(f) for f in self.folds],
            "boundary_semantics": "row indices [start, stop); last timestamp inclusive",
            "protocol_version": self.config.protocol.name,
        }


@dataclass(frozen=True)
class ForecastOrigin:
    position: int
    timestamp: str
    latest_available_position: int


@dataclass(frozen=True)
class MetricResult:
    metric: Literal["mae", "rmse", "mape"]
    value: float
    valid_count: int
    masked_count: int
    zero_denominator_count: int
    aggregation: str = "micro over valid forecast cells"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
