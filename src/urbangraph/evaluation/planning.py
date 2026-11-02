"""Pure chronological planning. Targets define strict partition ownership."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import (
    BENCHMARK_PROTOCOLS,
    POSITIONAL_PROTOCOL,
    PROTOCOLS,
    PlanConfig,
    RollingOriginFold,
    SampleIndex,
    SplitManifest,
    TemporalPartition,
)


def validate_timestamps(index: pd.DatetimeIndex) -> None:
    if (
        not isinstance(index, pd.DatetimeIndex)
        or index.empty
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
    ):
        raise DataError("Expected nonempty ordered unique timestamps without NaT")
    try:
        ns = np.asarray(index.as_unit("ns").astype("int64"), dtype=np.int64)
    except (ValueError, OverflowError) as exc:
        raise DataError("Timestamps must fit nanoseconds") from exc
    if int(ns[-1]) - int(ns[0]) > np.iinfo(np.int64).max:
        raise DataError("Timestamp span exceeds safe cadence arithmetic")


def resolve_boundary(index: pd.DatetimeIndex, boundary: int | str, interval: int) -> int:
    if type(boundary) is int:
        if 0 <= boundary <= len(index):
            return boundary
        raise DataError("Boundary index outside dataset")
    try:
        timestamp = pd.Timestamp(boundary)
        if pd.isna(timestamp) or (timestamp.tzinfo is None) != (index.tz is None):
            raise DataError("Boundary timezone awareness must match timestamps")
        if timestamp == index[-1] + pd.Timedelta(seconds=interval):
            return len(index)
        matches = np.flatnonzero(index == timestamp)
        found = int(matches[0]) if len(matches) else -1
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataError("Invalid timestamp boundary") from exc
    if found < 0:
        raise DataError("Timestamp boundary must exactly match a row or nominal end sentinel")
    return int(found)


def _ranges(origins: NDArray[np.int64]) -> tuple[SampleIndex, ...]:
    if not len(origins):
        raise DataError("Partition has no usable forecast windows")
    cuts = np.flatnonzero(np.diff(origins) != 1) + 1
    return tuple(
        SampleIndex(int(group[0]), int(group[-1]) + 1) for group in np.split(origins, cuts)
    )


def _partition(
    index: pd.DatetimeIndex,
    config: PlanConfig,
    name: str,
    start: int,
    stop: int,
    origin_limits: tuple[int, int] | None = None,
) -> TemporalPartition:
    if not 0 <= start < stop <= len(index):
        raise DataError("Invalid or empty chronological partition")
    history, horizon = config.window.history, config.window.horizon
    delay = config.features.observed_delay_steps
    if origin_limits is None:
        first = start + history - 1 + delay
        if config.protocol.context == "operational" and name != "train":
            # Delay the first prediction so even the latest pre-partition fitted
            # training observation is available. Targets skipped by latency are explicit.
            first = max(start - 1 + delay, history - 1 + delay)
        last = stop - horizon
    else:
        first, last = origin_limits
    origins = np.arange(first, max(first, last), dtype=np.int64)
    ns = np.asarray(index.as_unit("ns").astype("int64"), dtype=np.int64)
    # Prefix sums let every candidate inspect its entire input-to-target span in O(1).
    bad = np.concatenate(([0], np.cumsum(np.diff(ns) != config.window.interval_seconds * 10**9)))
    starts = origins - delay - history + 1
    ends = origins + horizon
    keep = bad[ends] == bad[starts]
    rejected = int(np.count_nonzero(~keep))
    if config.protocol.name == POSITIONAL_PROTOCOL:
        # Explicit positional compatibility preserves every reference origin.
        # Discontinuities are reported in the plan; zero samples are rejected here.
        keep = np.ones(len(origins), dtype=bool)
        rejected = 0
    elif origin_limits is not None and rejected:
        raise DataError("Benchmark cadence mismatch; refusing to change reference sample indexing")
    runs = _ranges(origins[keep])
    return TemporalPartition(
        name,
        start,
        stop,
        index[start].isoformat(),
        index[stop - 1].isoformat(),
        index[stop].isoformat() if stop < len(index) else None,
        runs,
        rejected,
    )


def _strict_fold(
    index: pd.DatetimeIndex,
    config: PlanConfig,
    boundaries: list[tuple[str, int, int]],
    number: int,
) -> RollingOriginFold:
    names = [b[0] for b in boundaries]
    if names not in (
        ["train", "validation", "test"],
        ["train", "validation", "calibration", "test"],
    ):
        raise DataError("Partitions must be ordered train, validation, optional calibration, test")
    for previous, current in zip(boundaries, boundaries[1:], strict=False):
        if current[1] < previous[2] + config.embargo:
            raise DataError("Partitions overlap or violate the row embargo")
    return RollingOriginFold(
        f"fold-{number:04d}", tuple(_partition(index, config, *b) for b in boundaries)
    )


def plan_splits(index: pd.DatetimeIndex, config: PlanConfig) -> SplitManifest:
    validate_timestamps(index)
    if config.protocol.name not in BENCHMARK_PROTOCOLS and (
        config.window.history + config.window.horizon + config.features.observed_delay_steps
        > len(index)
    ):
        raise DataError("Insufficient rows for history, observation latency and forecast horizon")
    if config.protocol.name in BENCHMARK_PROTOCOLS:
        if index.tz is not None:
            raise DataError(
                "Benchmark calendar parity is supported only for naive source timestamps"
            )
        count = len(index) - 23
        # Deliberately use Python float multiplication + ties-to-even round, as upstream.
        train, test = round(count * 0.7), round(count * 0.2)
        validation = count - train - test
        if min(train, validation, test) <= 0:
            raise DataError(
                "Benchmark split has an empty partition; tiny upstream slicing is unsupported"
            )
        cuts = (0, train, train + validation, count)
        partitions = []
        for name, a, b in zip(("train", "validation", "test"), cuts, cuts[1:], strict=False):
            first, stop = 11 + a, 11 + b
            # Benchmark partition boundaries describe the target union, which may overlap.
            partitions.append(_partition(index, config, name, first + 1, stop + 12, (first, stop)))
        return SplitManifest(config, (RollingOriginFold("fold-0000", tuple(partitions)),))
    if config.protocol.name == PROTOCOLS[1]:
        boundaries = [
            (
                b.name,
                resolve_boundary(index, b.start, config.window.interval_seconds),
                resolve_boundary(index, b.stop, config.window.interval_seconds),
            )
            for b in config.boundaries
        ]
        return SplitManifest(config, (_strict_fold(index, config, boundaries, 0),))
    folds = []
    for number in range(config.folds):
        train_end = config.minimum_train + number * config.stride
        rolling = [("train", 0, train_end)]
        cursor = train_end
        for name, length in (
            ("validation", config.validation_length),
            ("calibration", config.calibration_length),
            ("test", config.test_length),
        ):
            if not length:
                continue
            start = cursor + config.embargo
            cursor = start + length
            rolling.append((name, start, cursor))
        folds.append(_strict_fold(index, config, rolling, number))
    return SplitManifest(config, tuple(folds))
