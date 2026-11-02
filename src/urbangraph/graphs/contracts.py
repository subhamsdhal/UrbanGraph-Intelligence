"""Validated graph specifications and exact canonical sensor indexing."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError, DatasetIdentity
from urbangraph.evaluation.dataset import identity

FloatMatrix = NDArray[np.float64]


def sensor_axis(values: tuple[str, ...]) -> tuple[str, ...]:
    if not values or any(not isinstance(v, str) or not v or v != v.strip() for v in values):
        raise DataError("Nonempty exact string sensor IDs required")
    normalized = tuple(unicodedata.normalize("NFC", v) for v in values)
    if normalized != values or len(set(values)) != len(values):
        raise DataError("Duplicate or noncanonical sensor identifiers")
    return values


@dataclass(frozen=True)
class SensorNodeIndex:
    sensor_ids: tuple[str, ...]
    source_order: tuple[str, ...]

    def __post_init__(self) -> None:
        sensor_axis(self.sensor_ids)
        sensor_axis(self.source_order)
        if set(self.sensor_ids) != set(self.source_order):
            raise DataError("Source and canonical sensor sets must match exactly")

    @property
    def permutation(self) -> tuple[int, ...]:
        lookup = {v: i for i, v in enumerate(self.source_order)}
        return tuple(lookup[v] for v in self.sensor_ids)

    @property
    def fingerprint(self) -> str:
        return identity(list(self.sensor_ids))

    def to_dict(self) -> dict[str, object]:
        return {
            "sensor_ids": list(self.sensor_ids),
            "source_order": list(self.source_order),
            "canonical_to_source": list(self.permutation),
            "axis_sha256": self.fingerprint,
        }


@dataclass(frozen=True)
class GraphSource:
    name: str
    sha256: str
    size_bytes: int
    reference: str
    layout: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GraphSpecification:
    dataset_id: str
    method: str = "road-gaussian"
    version: str = "graph-1.0"
    direction: str = "directed"
    self_loops: str = "preserve"
    normalization: str = "none"
    threshold: float = 0.1
    duplicate_edges: str = "reject"
    outside_nodes: str = "filter"
    k: int = 8
    radius_km: float = 25.0
    geographic_weight: str = "gaussian"
    geographic_scale_km: float = 5.0

    def __post_init__(self) -> None:
        DatasetIdentity(self.dataset_id)
        if self.version != "graph-1.0" or self.method not in {"road-gaussian", "geographic-knn"}:
            raise DataError("Unsupported graph schema/method")
        choices = {
            "direction": {"directed", "symmetric-max"},
            "self_loops": {"preserve", "remove", "unit"},
            "normalization": {"none", "random-walk", "reverse-walk", "symmetric"},
            "duplicate_edges": {"reject", "last"},
            "outside_nodes": {"reject", "filter"},
            "geographic_weight": {"gaussian", "inverse-distance"},
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise DataError(f"Invalid {name} policy")
        for name in ("threshold", "radius_km", "geographic_scale_km"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise DataError(f"Invalid {name}")
        if not 0 <= self.threshold <= 1 or self.radius_km <= 0 or self.geographic_scale_km <= 0:
            raise DataError("Graph parameter out of range")
        if type(self.k) is not int or self.k < 1:
            raise DataError("k must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GraphArtifact:
    directory: Path
    graph_id: str
    index: SensorNodeIndex
    adjacency: FloatMatrix
    descriptor: dict[str, object]


@dataclass(frozen=True)
class GraphVerificationResult:
    valid: bool
    graph_id: str | None
    issues: tuple[str, ...] = ()
