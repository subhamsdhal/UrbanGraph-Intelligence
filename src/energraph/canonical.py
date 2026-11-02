"""OPSD selected-column streaming parser and immutable energy Parquet materialization."""

from __future__ import annotations

import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    governed_payload,
)
from energraph.broker_tree import inspect_tree
from energraph.contracts import EnergyVariable
from urbangraph.data import DataError, DatasetSnapshot, verify_snapshot
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.rights import (
    require_snapshot_authorization,
    require_verified_snapshot_authorization,
)
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import validate_timestamps


def parse_csv(
    path: Path,
    variable: EnergyVariable,
    *,
    raw_snapshot_dir: Path | None = None,
    destination: Path | None = None,
    chunk_rows: int = 8192,
) -> pd.DataFrame:
    """Protected direct read; internal verified creators use the private parser."""
    if raw_snapshot_dir is None or destination is None:
        raise DataError("Current source authorization required for energy CSV read")
    snapshot = safe_absolute(raw_snapshot_dir)
    approved = require_verified_snapshot_authorization(
        snapshot, "local_research", destination, operation="energy_source_read"
    )
    checked = verify_snapshot(snapshot)
    if not checked.valid or checked.manifest is None or checked.manifest.snapshot_id != approved:
        raise DataError("Energy source differs from current read authority")
    path = safe_absolute(path)
    if path.parent != snapshot / "payload" or path.name not in {
        item.relative_path for item in checked.manifest.files
    }:
        raise DataError("Energy CSV is not in the authorized source inventory")
    return _parse_csv_verified(path, variable, chunk_rows=chunk_rows)


def _parse_csv_verified(
    path: Path, variable: EnergyVariable, *, chunk_rows: int = 8192
) -> pd.DataFrame:
    path = safe_absolute(path)
    before = stream_file(path)
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise DataError("Positive chunk size required")
    with path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    if len(header) != len(set(header)) or not {"utc_timestamp", variable.column} <= set(header):
        raise DataError("Missing or duplicate CSV columns")
    parts = []
    try:
        for chunk in pd.read_csv(
            path,
            usecols=["utc_timestamp", variable.column],
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_rows,
        ):
            labels = chunk["utc_timestamp"]
            if not labels.str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:00:00Z").all():
                raise DataError("OPSD requires explicit UTC Z timestamps on exact hours")
            index = pd.DatetimeIndex(pd.to_datetime(labels, utc=True, errors="raise"))
            values = np.array(
                [float(v) if v else np.nan for v in chunk[variable.column]], dtype=np.float64
            )
            if np.isinf(values).any() or any(
                v and not np.isfinite(float(v)) for v in chunk[variable.column]
            ):
                raise DataError(
                    "Only empty fields represent missingness; nonfinite literals rejected"
                )
            parts.append(pd.DataFrame({variable.column: values}, index=index))
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataError("Invalid OPSD CSV values") from exc
    if not parts:
        raise DataError("Empty energy input")
    frame = pd.concat(parts)
    frame.index.name = "utc_timestamp"
    validate_timestamps(pd.DatetimeIndex(frame.index))
    if stream_file(path) != before:
        raise DataError("Energy CSV changed during parsing")
    return frame


def quality(frame: pd.DataFrame) -> dict[str, object]:
    index = pd.DatetimeIndex(frame.index)
    validate_timestamps(index)
    if str(index.tz) != "UTC" or np.any(
        np.asarray(index.as_unit("ns").astype("int64"), dtype=np.int64) % (3600 * 10**9)
    ):
        raise DataError("UTC exact-hour canonical axis required")
    delta = np.diff(np.asarray(index.as_unit("ns").astype("int64"), dtype=np.int64)) // (
        3600 * 10**9
    )
    if np.any(delta < 1):
        raise DataError("Invalid hourly cadence")
    values = frame.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise DataError("Infinite canonical values forbidden")
    return {
        "rows": len(frame),
        "first_timestamp": index[0].isoformat(),
        "last_timestamp": index[-1].isoformat(),
        "missing_values": int(np.isnan(values).sum()),
        "zero_values": int((values == 0).sum()),
        "negative_values": int((values < 0).sum()),
        "missing_hours": int(np.maximum(delta - 1, 0).sum()),
        "gap_after_rows": np.flatnonzero(delta != 1).tolist(),
        "source_order_preserved": True,
        "interpolation": "none",
    }


def logical_hash(frame: pd.DataFrame) -> str:
    h = hashlib.sha256(np.asarray(frame.index.astype("int64"), dtype="<i8").tobytes())
    values = frame.to_numpy(dtype="<f8", copy=True)
    values[np.isnan(values)] = np.nan
    h.update(values.tobytes())
    return h.hexdigest()


def materialize(
    snapshot: Path,
    root: Path,
    variable: EnergyVariable | None = None,
    *,
    filename: str = "time_series_60min_singleindex.csv",
    broker_context: BrokerPublicationContext | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> DatasetSnapshot | PendingPublication:
    approved_identity = require_snapshot_authorization(
        snapshot, "canonicalize", root, operation="energy_canonicalize"
    )
    variable = variable or EnergyVariable()
    raw = verify_snapshot(snapshot)
    if not raw.valid or raw.manifest is None:
        raise DataError("Raw snapshot must verify before parsing")
    if raw.manifest.snapshot_id != approved_identity:
        raise DataError("Energy raw snapshot differs from rights grant")
    if (broker_context is None) != (raw_broker_reference is None):
        raise DataError("Broker publication requires both broker context and raw identity")
    if broker_context is not None and raw_broker_reference is not None:
        record = broker_context.broker.authorize_directory(
            raw_broker_reference.artifact_id,
            purpose="canonicalize",
            operation="energy_canonicalize",
        )
        if (
            raw_broker_reference.artifact_type != "raw"
            or record.kind != "raw"
            or record.content_sha256 != raw_broker_reference.content_sha256
            or record.revision != raw_broker_reference.revision
            or inspect_tree(snapshot).digest != record.content_sha256
        ):
            raise DataError("Raw snapshot differs from the currently authorized broker source")
    if filename not in {f.relative_path for f in raw.manifest.files}:
        raise DataError("Requested CSV not in verified source inventory")
    frame = _parse_csv_verified(snapshot / "payload" / filename, variable)
    descriptor = {
        "schema_version": "energy-canonical-1.0",
        "variable": variable.to_dict(),
        "raw_snapshot_id": raw.manifest.snapshot_id,
        "raw_manifest_sha256": stream_file(snapshot / "manifest.json")[1],
        "file": filename,
        "logical_sha256": logical_hash(frame),
        "timestamp_policy": "UTC interval start",
        "interval_seconds": 3600,
        "value_dtype": "float64",
        "missing_value": "NaN from empty CSV only",
        "quality": quality(frame),
    }
    with tempfile.TemporaryDirectory(prefix="energraph-canonical-") as tmp:
        stage = Path(tmp)
        frame.to_parquet(stage / "observations.parquet", engine="pyarrow", index=True)
        roundtrip = pd.read_parquet(stage / "observations.parquet")
        pd.testing.assert_frame_equal(frame, roundtrip, check_exact=True, check_freq=False)
        write_json(stage / "descriptor.json", descriptor)
        write_json(stage / "raw-manifest.json", raw.manifest.to_dict())
        result = _publish_governed(
            stage,
            root,
            "canonical",
            raw_snapshot_dir=snapshot,
            broker_context=broker_context,
            broker_parents=(raw_broker_reference,) if raw_broker_reference is not None else None,
        )
    if isinstance(result, PendingPublication):
        return result
    open_energy(
        result.directory,
        raw_snapshot_dir=snapshot,
        broker_context=broker_context,
        broker_reference=result.broker_reference,
        raw_broker_reference=raw_broker_reference,
    )
    return result


def open_energy(
    directory: Path,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    broker_reference: BrokerArtifactRef | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> EvaluationDataset:
    if raw_snapshot_dir is None:
        raise DataError("Current upstream source context required for energy read")
    raw_snapshot_dir = safe_absolute(raw_snapshot_dir)
    brokered = broker_reference is not None or broker_context is not None
    if brokered:
        if broker_reference is None or raw_broker_reference is None or broker_context is None:
            raise DataError("Complete broker canonical identity is required for protected read")
        if (
            broker_reference.artifact_type != "canonical"
            or raw_broker_reference.artifact_type != "raw"
        ):
            raise DataError("Broker canonical or raw identity has the wrong type")
        base = governed_payload(
            broker_reference,
            {"observations.parquet", "descriptor.json", "raw-manifest.json"},
            kind="canonical",
            broker=broker_context.broker,
            broker_parents=(raw_broker_reference,),
            materialization_root=broker_context.materialization_root,
        )
        cached = safe_absolute(directory) / "payload"
        cached_files = discover(cached).files
        expected_files = (
            "descriptor.json",
            "governance.json",
            "observations.parquet",
            "raw-manifest.json",
        )
        if tuple(sorted(cached_files)) != expected_files:
            raise DataError("Local canonical cache inventory differs from broker content")
        for name in expected_files:
            if stream_file(cached / name) != stream_file(base / name):
                raise DataError("Local canonical cache differs from broker-authorized content")
    else:
        base = governed_payload(
            directory,
            {"observations.parquet", "descriptor.json", "raw-manifest.json"},
            raw_snapshot_dir=raw_snapshot_dir,
            kind="canonical",
        )
    descriptor = read_json(base / "descriptor.json")
    if descriptor.get("schema_version") != "energy-canonical-1.0":
        raise DataError("Unsupported energy canonical schema")
    variable = EnergyVariable.from_dict(object_dict(descriptor["variable"]))
    frame = pd.read_parquet(base / "observations.parquet")
    if list(frame.columns) != [variable.column] or any(
        d != np.dtype("float64") for d in frame.dtypes
    ):
        raise DataError("Canonical energy columns/dtype mismatch")
    if (
        quality(frame) != descriptor["quality"]
        or logical_hash(frame) != descriptor["logical_sha256"]
    ):
        raise DataError("Canonical logical content mismatch")
    raw = read_json(base / "raw-manifest.json")
    if (
        raw.get("snapshot_id") != descriptor["raw_snapshot_id"]
        or raw.get("snapshot_id") != raw_snapshot_dir.name
        or stream_file(raw_snapshot_dir / "manifest.json")[1] != descriptor["raw_manifest_sha256"]
    ):
        raise DataError("Canonical raw lineage mismatch")
    lineage: dict[str, object] = {
        "dataset_id": "opsd-de-hourly-load",
        "canonical_id": directory.name,
        "canonical_descriptor_id": identity(descriptor),
        "variable": variable.to_dict(),
        "units": variable.unit,
        "temporal": {"timezone": "UTC", "interval_seconds": 3600},
    }
    return EvaluationDataset(directory, frame, lineage)
