"""Read explicit text/CSV sensor axes. Pickles are never opened."""

import csv
import math
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute, stream_file


def inspect_sensor_metadata(
    path: str | Path | None, sensor_ids: tuple[str, ...], *, layout: str = "locations-header"
) -> dict[str, object]:
    if path is None:
        return {"status": "UNKNOWN", "reason": "No spatial metadata supplied"}
    source = safe_absolute(path)
    if source.suffix.lower() not in {".txt", ".csv"}:
        raise DataError("Only text/CSV metadata is supported; pickle deserialization is forbidden")
    if layout not in {"locations-header", "ids", "bay-headerless"}:
        raise DataError("Unknown metadata layout")
    if not source.is_file():
        return {"status": "UNKNOWN", "reason": "Metadata unavailable"}
    if source.stat().st_size > 16 * 1024 * 1024:
        raise DataError("Sensor metadata exceeds 16 MiB")
    before = stream_file(source)
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    identifiers: list[str] = []
    invalid_coordinates = 0
    if layout == "ids":
        identifiers = [value.strip() for row in rows for value in row]
    elif layout == "bay-headerless":
        # Upstream file is headerless; column meanings are not guessed here.
        if any(len(row) != 3 for row in rows):
            raise DataError("Expected three-column headerless metadata")
        identifiers = [row[0].strip() for row in rows]
    else:
        if not rows or not {"sensor_id", "latitude", "longitude"} <= set(rows[0]):
            raise DataError("Explicit sensor_id, latitude, longitude headers are required")
        header = rows.pop(0)
        for row in rows:
            if len(row) != len(header):
                raise DataError("Malformed metadata row")
            identifiers.append(row[header.index("sensor_id")].strip())
            try:
                lat, lon = (float(row[header.index(k)]) for k in ("latitude", "longitude"))
            except ValueError as exc:
                raise DataError("Invalid coordinate encoding") from exc
            invalid_coordinates += int(
                not (
                    math.isfinite(lat)
                    and math.isfinite(lon)
                    and -90 <= lat <= 90
                    and -180 <= lon <= 180
                )
            )
    if (
        not identifiers
        or any(not s for s in identifiers)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise DataError("Empty or duplicate metadata sensor IDs")
    if stream_file(source) != before:
        raise DataError("Metadata changed during inspection")
    missing = sorted(set(sensor_ids) - set(identifiers))
    extra = sorted(set(identifiers) - set(sensor_ids))
    return {
        "status": "WARNING" if missing or extra or invalid_coordinates else "PASS",
        "sha256": before[1],
        "layout": layout,
        "matched_count": len(set(sensor_ids) & set(identifiers)),
        "missing_ids": missing,
        "extra_ids": extra,
        "ordering_matches": tuple(identifiers) == sensor_ids,
        "invalid_coordinates": invalid_coordinates if layout == "locations-header" else None,
        "coordinate_policy": "Named columns only; headerless coordinate meanings unverified",
    }
