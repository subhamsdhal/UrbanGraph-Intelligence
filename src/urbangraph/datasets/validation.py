"""Structural validation and deterministic temporal diagnostics; never sort or fill."""

from __future__ import annotations

import math
import unicodedata
from typing import NoReturn

import numpy as np
import pandas as pd

from urbangraph.datasets.contracts import DatasetConfig, Finding, SensorIdentity, StructuralError


def _fail(check: str, description: str, **context: object) -> NoReturn:
    raise StructuralError((Finding(check, "ERROR", description, context),))


def validate_frame(frame: pd.DataFrame) -> tuple[SensorIdentity, ...]:
    if frame.empty:
        _fail("empty_matrix", "At least one timestamp and one sensor are required")
    if not isinstance(frame.index, pd.DatetimeIndex):
        _fail("timestamp_type", "A DatetimeIndex is required; strings are not inferred")
    if frame.index.hasnans:
        _fail("missing_timestamps", "Timestamp axis contains NaT")
    if frame.index.has_duplicates:
        _fail("duplicate_timestamps", "Timestamp instants must be unique")
    if not frame.index.is_monotonic_increasing:
        _fail("timestamp_order", "Timestamp order must be increasing; no implicit sorting")
    if isinstance(frame.columns, pd.MultiIndex):
        _fail("sensor_axis", "MultiIndex sensor identifiers are unsupported")
    if frame.columns.has_duplicates:
        _fail("duplicate_sensors", "Original sensor identifiers must be unique")
    sensors: list[SensorIdentity] = []
    identifiers: set[str] = set()
    for position, (label, dtype) in enumerate(zip(frame.columns, frame.dtypes, strict=True)):
        if label is None or label is pd.NA or label is pd.NaT:
            _fail("missing_sensor_id", "Missing sensor identifier", position=position)
        if not isinstance(label, str | int | float | np.integer | np.floating) or isinstance(
            label, bool
        ):
            _fail(
                "sensor_type",
                "Identifiers must be strings or finite real scalars",
                position=position,
            )
        if isinstance(label, float | np.floating) and not math.isfinite(float(label)):
            _fail(
                "missing_sensor_id", "Sensor identifier is missing or nonfinite", position=position
            )
        original = str(label)
        if not original.strip():
            _fail("missing_sensor_id", "Blank sensor identifier", position=position)
        identifier = unicodedata.normalize("NFC", original)
        if identifier in identifiers:
            _fail(
                "sensor_normalization_collision",
                "Normalized identifiers collide",
                position=position,
            )
        identifiers.add(identifier)
        if (
            not isinstance(dtype, np.dtype)
            or dtype.kind not in "iuf"
            or dtype.itemsize > 8
            or not dtype.isnative
            or (dtype.kind == "f" and dtype.itemsize < 4)
        ):
            _fail(
                "observation_dtype",
                "Only native-endian NumPy real integer and float32/64 columns are supported",
                position=position,
                dtype=str(dtype),
            )
        sensors.append(SensorIdentity(identifier, type(label).__name__, original, str(dtype)))
    return tuple(sensors)


def temporal_profile(
    index: pd.DatetimeIndex, config: DatasetConfig
) -> tuple[dict[str, object], tuple[Finding, ...]]:
    # ns ensures a single exact timestamp representation in canonical Parquet.
    try:
        index = index.as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise StructuralError(
            (Finding("timestamp_range", "ERROR", "Timestamp outside nanosecond range", {}),)
        ) from exc
    ns = np.asarray(index.astype("int64"), dtype=np.int64)
    span = int(ns[-1]) - int(ns[0])
    if span > np.iinfo(np.int64).max:
        _fail("timestamp_span", "Time span exceeds safe int64 difference range")
    step = config.nominal_interval_seconds * 1_000_000_000
    differences = np.diff(ns)
    offsets = ns - ns[0]
    on_grid = int(np.count_nonzero(offsets % step == 0))
    expected = span // step + 1
    gaps = np.flatnonzero(differences > step)
    intervals, counts = np.unique(differences, return_counts=True)
    examples = [
        {
            "after_row": int(i),
            "start": index[int(i)].isoformat(),
            "end": index[int(i) + 1].isoformat(),
            "interval_seconds": float(differences[i] / 1e9),
            "missing_aligned_steps": int(differences[i] // step - 1)
            if differences[i] % step == 0
            else None,
        }
        for i in gaps[: config.max_gap_examples]
    ]
    aware = index.tz is not None
    local_repeats = int(index.tz_localize(None).duplicated().sum()) if aware else 0
    irregular = int(np.count_nonzero(differences != step))
    temporal: dict[str, object] = {
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "timezone": str(index.tz) if aware else None,
        "timezone_aware": aware,
        "timezone_policy": "preserve-source",
        "timestamp_unit": "ns",
        "source_order_preserved": True,
        "nominal_interval_seconds": config.nominal_interval_seconds,
        "observed_slots": len(index),
        "expected_grid_slots": expected,
        "observed_on_grid_slots": on_grid,
        "missing_grid_slots": expected - on_grid,
        "off_grid_timestamps": len(index) - on_grid,
        "gap_count": len(gaps),
        "irregular_interval_count": irregular,
        "interval_histogram": [
            {"seconds": float(v / 1e9), "count": int(c)}
            for v, c in zip(intervals, counts, strict=True)
        ],
        "gap_examples": examples,
        "gap_examples_omitted": max(0, len(gaps) - len(examples)),
        "repeated_local_wall_timestamps": local_repeats,
    }
    findings: list[Finding] = []
    if not aware:
        findings.append(
            Finding(
                "timezone_naive",
                "INFO",
                "Timezone unknown; DST ambiguity cannot be resolved from naive timestamps",
                {},
            )
        )
    for check, count, description in (
        (
            "wall_time_repeats",
            local_repeats,
            "Repeated local wall times; distinct aware instants preserved",
        ),
        ("irregular_intervals", irregular, "Observed intervals differ from nominal cadence"),
        ("temporal_gaps", len(gaps), "Intervals larger than nominal cadence"),
        (
            "off_grid_timestamps",
            len(index) - on_grid,
            "Timestamps off the grid anchored at first observation",
        ),
    ):
        if count:
            findings.append(Finding(check, "WARNING", description, {"count": count}))
    return temporal, tuple(findings)
