"""Read-only evaluation input from a fully verified UG-002 artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.datasets.governed_canonical import read_canonical
from urbangraph.datasets.materialization import load_materialization
from urbangraph.datasets.serialization import canonical_json, object_dict, read_json


@dataclass(frozen=True)
class EvaluationDataset:
    directory: Path
    frame: pd.DataFrame
    lineage: dict[str, object]

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame.index)

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(str(s) for s in self.frame.columns)


def open_dataset(
    directory: str | Path, *, raw_snapshot_dir: str | Path | None = None
) -> EvaluationDataset:
    """Legacy name retained; protected observations require upstream identity."""
    if raw_snapshot_dir is None:
        raise DataError("Raw snapshot identity required for evaluation read")
    path = safe_absolute(directory)
    frame = load_materialization(path, raw_snapshot_dir=raw_snapshot_dir)
    before = stream_file(path / "manifest.json")[1]
    return _evaluation_dataset(path, frame, before)


def open_governed_dataset(
    directory: str | Path, *, raw_snapshot_dir: str | Path
) -> EvaluationDataset:
    """Open actual evaluation input only after current upstream rights recheck."""
    path = safe_absolute(directory)
    frame = read_canonical(path, raw_snapshot_dir=raw_snapshot_dir)
    before = stream_file(path / "manifest.json")[1]
    return _evaluation_dataset(path, frame, before)


def _evaluation_dataset(path: Path, frame: pd.DataFrame, before: str) -> EvaluationDataset:
    manifest = read_json(path / "manifest.json")
    if stream_file(path / "manifest.json")[1] != before:
        raise DataError("Canonical manifest changed during evaluation loading")
    descriptor = object_dict(manifest["descriptor"])
    schema = read_json(path / "schema.json")
    lineage: dict[str, object] = {
        "dataset_id": descriptor["dataset_id"],
        "raw_snapshot_id": descriptor["raw_snapshot_id"],
        "canonical_id": manifest["transformation_id"],
        "canonical_manifest_sha256": before,
        "canonical_schema_version": descriptor["canonical_schema_version"],
        "timestamp_count": len(frame),
        "sensor_ids": list(frame.columns),
        "timestamp_sha256": hashlib.sha256(
            np.asarray(frame.index.astype("int64"), dtype="<i8").tobytes()
        ).hexdigest(),
        "temporal": schema["temporal"],
        "units": schema["units"],
    }
    return EvaluationDataset(path, frame, lineage)


def identity(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
