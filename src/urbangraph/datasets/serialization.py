"""Strict JSON helpers and validated transformation configuration loading."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.verification import MAX_MANIFEST_BYTES, _unique_keys
from urbangraph.datasets.contracts import DatasetConfig


def object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DataError("Expected a JSON object")
    return {str(key): item for key, item in value.items()}


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise DataError("JSON artifact exceeds 16 MiB limit")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> dict[str, object]:
    safe_absolute(path)
    with path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise DataError("JSON artifact exceeds 16 MiB limit")
    try:
        value: object = json.loads(raw, object_pairs_hook=_unique_keys)
        canonical_json(value)  # Reject nonfinite numbers accepted by Python's JSON reader.
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataError("Invalid JSON artifact") from exc
    return object_dict(value)


def config_from_dict(value: object) -> DatasetConfig:
    obj = object_dict(value)
    if set(obj) != {field.name for field in fields(DatasetConfig)}:
        raise DataError("Invalid transformation configuration fields")
    text: dict[str, str | None] = {}
    for key in ("hdf_key", "payload_file", "units", "units_reference"):
        item = obj[key]
        if item is not None and not isinstance(item, str):
            raise DataError("Invalid configuration text")
        text[key] = item
    interval, limit = obj["nominal_interval_seconds"], obj["max_gap_examples"]
    low, missing = obj["low_variance_std"], obj["extreme_missing_fraction"]
    rejected = obj["reject_checks"]
    if type(interval) is not int or type(limit) is not int:
        raise DataError("Invalid configuration integers")
    if (
        isinstance(low, bool)
        or isinstance(missing, bool)
        or not isinstance(low, int | float)
        or not isinstance(missing, int | float)
    ):
        raise DataError("Invalid configuration thresholds")
    if not isinstance(rejected, list) or not all(isinstance(c, str) for c in rejected):
        raise DataError("Invalid configuration rejection policy")
    return DatasetConfig(
        hdf_key=text["hdf_key"],
        payload_file=text["payload_file"],
        units=text["units"],
        units_reference=text["units_reference"],
        nominal_interval_seconds=interval,
        max_gap_examples=limit,
        low_variance_std=float(low),
        extreme_missing_fraction=float(missing),
        reject_checks=tuple(str(c) for c in rejected),
    )
