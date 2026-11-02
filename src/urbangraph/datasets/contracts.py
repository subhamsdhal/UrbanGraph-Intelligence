"""Typed canonical contracts and explicit audit-only transformation policy."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from urbangraph.data import DataError, DatasetManifest, RawFileRecord

if TYPE_CHECKING:
    import pandas as pd

CANONICAL_VERSION = "1.0"
PARSER_VERSION = "pandas-hdf-v1"
PROFILE_VERSION = "traffic-audit-v1"
QUALITY_CHECKS = frozenset(
    {
        "timezone_naive",
        "wall_time_repeats",
        "irregular_intervals",
        "temporal_gaps",
        "off_grid_timestamps",
        "all_missing_sensor",
        "constant_sensor",
        "low_variance_sensor",
        "extreme_missingness",
        "uneven_coverage",
        "infinite_values",
        "negative_values",
        "zero_values",
        "statistics_overflow",
        "audit_precision",
    }
)


@dataclass(frozen=True)
class Finding:
    check_id: str
    severity: Literal["INFO", "WARNING", "ERROR"]
    description: str
    context: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class StructuralError(DataError):
    """A source cannot be represented without concealing a structural defect."""

    def __init__(self, findings: tuple[Finding, ...]) -> None:
        self.findings = findings
        super().__init__("Structural validation failed: " + ", ".join(f.check_id for f in findings))


@dataclass(frozen=True)
class DatasetConfig:
    hdf_key: str | None = None
    payload_file: str | None = None
    nominal_interval_seconds: int = 300
    low_variance_std: float = 0.01
    extreme_missing_fraction: float = 0.5
    max_gap_examples: int = 20
    reject_checks: tuple[str, ...] = ()
    units: str | None = None
    units_reference: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.nominal_interval_seconds) is not int
            or not 1 <= self.nominal_interval_seconds <= 86400
        ):
            raise DataError("Nominal interval must be an integer from 1 to 86400 seconds")
        if type(self.max_gap_examples) is not int or not 0 <= self.max_gap_examples <= 10000:
            raise DataError("Gap example limit must be an integer from 0 to 10000")
        for value in (self.low_variance_std, self.extreme_missing_fraction):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise DataError("Quality thresholds must be finite numbers")
        if self.low_variance_std < 0 or not 0 <= self.extreme_missing_fraction <= 1:
            raise DataError("Invalid quality threshold range")
        if not isinstance(self.reject_checks, tuple) or any(
            c not in QUALITY_CHECKS for c in self.reject_checks
        ):
            raise DataError("Unknown quality check selected for rejection")
        if len(set(self.reject_checks)) != len(self.reject_checks):
            raise DataError("Duplicate rejection check")
        for text_value in (self.hdf_key, self.payload_file, self.units, self.units_reference):
            if text_value is not None and (
                not isinstance(text_value, str) or not text_value.strip()
            ):
                raise DataError("Optional text must be nonblank or null")
        if (self.units is None) != (self.units_reference is None):
            raise DataError("Units require an explicit source reference, and vice versa")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["reject_checks"] = sorted(self.reject_checks)
        result["low_variance_std"] = float(self.low_variance_std)
        result["extreme_missing_fraction"] = float(self.extreme_missing_fraction)
        return result


@dataclass(frozen=True)
class SensorIdentity:
    identifier: str
    original_type: str
    original_value: str
    dtype: str


@dataclass(frozen=True)
class RawLineage:
    snapshot_dir: Path
    manifest: DatasetManifest
    manifest_sha256: str
    input_file: RawFileRecord


@dataclass(frozen=True)
class CanonicalDataset:
    """Logical [T,N] matrix backed by dtype-preserving numeric DataFrame columns.

    Frozen metadata does not make the contained DataFrame immutable. Treat it as
    read-only; materialization re-parses its verified raw source rather than trusting
    a caller-mutated matrix.
    """

    observations: pd.DataFrame
    sensors: tuple[SensorIdentity, ...]
    lineage: RawLineage
    config: DatasetConfig
    temporal: dict[str, object]
    source_axes: dict[str, object]
    findings: tuple[Finding, ...]

    @property
    def dataset_id(self) -> str:
        return self.lineage.manifest.dataset_id

    @property
    def raw_snapshot_id(self) -> str:
        return self.lineage.manifest.snapshot_id

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(sensor.identifier for sensor in self.sensors)

    def schema_dict(self) -> dict[str, object]:
        return {
            "schema_version": CANONICAL_VERSION,
            "parser_version": PARSER_VERSION,
            "dataset_id": self.dataset_id,
            "raw_snapshot_id": self.raw_snapshot_id,
            "timestamp_column": "timestamp",
            "sensor_columns": [f"sensor_{i:06d}" for i in range(len(self.sensors))],
            "sensors": [asdict(sensor) for sensor in self.sensors],
            "timestamp_count": len(self.observations),
            "sensor_count": len(self.sensors),
            "temporal": self.temporal,
            "source_axes": self.source_axes,
            "units": self.config.units,
            "units_reference": self.config.units_reference,
            "missingness": "source NaN preserved; Parquet nulls round-trip to NaN; zeros valid",
        }


@dataclass(frozen=True)
class QualityReport:
    dataset_id: str
    raw_snapshot_id: str
    metrics: dict[str, object]
    sensors: tuple[dict[str, object], ...]
    findings: tuple[Finding, ...]

    @property
    def accepted(self) -> bool:
        return not any(f.severity == "ERROR" for f in self.findings)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_version": PROFILE_VERSION,
            "purpose": "descriptive audit only; never training preprocessing parameters",
            "dataset_id": self.dataset_id,
            "raw_snapshot_id": self.raw_snapshot_id,
            "accepted": self.accepted,
            "metrics": self.metrics,
            "sensors": list(self.sensors),
            "findings": [finding.to_dict() for finding in self.findings],
        }
