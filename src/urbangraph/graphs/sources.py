"""Bounded CSV adapters; graph pickles are never opened."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.data.rights import require_raw_authorization
from urbangraph.graphs.contracts import FloatMatrix, GraphSource, SensorNodeIndex


def read_csv(
    path: str | Path, *, reference: str, layout: str
) -> tuple[list[list[str]], GraphSource]:
    expected_digest = require_raw_authorization(
        reference, path, "graph_analysis", Path(path).parent, operation="graph_csv_read"
    )
    source = safe_absolute(path)
    if source.suffix.lower() != ".csv" or not reference.strip():
        raise DataError("CSV and explicit provenance reference required; pickle is forbidden")
    size, digest = stream_file(source)
    if digest != expected_digest:
        raise DataError("Graph source content does not match rights grant")
    if size > 32 * 1024**2:
        raise DataError("Graph CSV exceeds 32 MiB limit")
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or stream_file(source) != (size, digest):
        raise DataError("Empty or changing graph source")
    return rows, GraphSource(source.name, digest, size, reference, layout)


def read_locations(
    path: str | Path,
    sensors: tuple[str, ...],
    *,
    reference: str,
    layout: str = "locations-header",
    coordinate_reference: str | None = None,
) -> tuple[SensorNodeIndex, FloatMatrix | None, GraphSource]:
    rows, source = read_csv(path, reference=reference, layout=layout)
    if layout == "locations-header":
        header = rows.pop(0)
        if len(set(header)) != len(header) or not {"sensor_id", "latitude", "longitude"} <= set(
            header
        ):
            raise DataError("Explicit coordinate headers required")
        if any(len(row) != len(header) for row in rows):
            raise DataError("Malformed location row")
        columns = [header.index(k) for k in ("sensor_id", "latitude", "longitude")]
        rows = [[row[i] for i in columns] for row in rows]
    elif layout != "bay-headerless":
        raise DataError("Unsupported location layout")
    if any(len(row) != 3 for row in rows):
        raise DataError("Expected three location fields")
    index = SensorNodeIndex(sensors, tuple(row[0] for row in rows))
    if layout == "bay-headerless" and not coordinate_reference:
        return index, None, source
    try:
        coordinates = np.asarray([[float(row[1]), float(row[2])] for row in rows])
    except ValueError as exc:
        raise DataError("Invalid coordinate encoding") from exc
    if (
        not np.isfinite(coordinates).all()
        or (np.abs(coordinates[:, 0]) > 90).any()
        or (np.abs(coordinates[:, 1]) > 180).any()
    ):
        raise DataError("Coordinates out of latitude/longitude range")
    return index, coordinates[list(index.permutation)].copy(), source


def read_distances(
    path: str | Path,
    index: SensorNodeIndex,
    *,
    reference: str,
    layout: str,
    duplicate_policy: str,
    outside_policy: str,
) -> tuple[FloatMatrix, GraphSource, dict[str, object]]:
    rows, source = read_csv(path, reference=reference, layout=layout)
    if layout == "distance-header":
        if rows.pop(0) != ["from", "to", "cost"]:
            raise DataError("Expected from,to,cost header")
    elif layout != "distance-headerless":
        raise DataError("Unsupported distance layout")
    if duplicate_policy not in {"reject", "last"} or outside_policy not in {"reject", "filter"}:
        raise DataError("Invalid distance policy")
    lookup = {v: i for i, v in enumerate(index.sensor_ids)}
    distances = np.full((len(lookup), len(lookup)), np.inf)
    outside, duplicate = 0, 0
    for row in rows:
        if len(row) != 3 or row[0] != row[0].strip() or row[1] != row[1].strip():
            raise DataError("Malformed distance record")
        try:
            distance = float(row[2])
        except ValueError as exc:
            raise DataError("Invalid distance") from exc
        if not np.isfinite(distance) or distance < 0:
            raise DataError("Distance must be finite and nonnegative")
        if row[0] not in lookup or row[1] not in lookup:
            if outside_policy == "reject":
                raise DataError("Unknown distance sensor")
            outside += 1
            continue
        i, j = lookup[row[0]], lookup[row[1]]
        if np.isfinite(distances[i, j]):
            duplicate += 1
            if duplicate_policy == "reject":
                raise DataError("Duplicate distance edge")
        distances[i, j] = distance
    return (
        distances,
        source,
        {
            "source_rows": len(rows),
            "filtered_outside_rows": outside,
            "duplicate_rows": duplicate,
            "distance_units": None,
        },
    )
