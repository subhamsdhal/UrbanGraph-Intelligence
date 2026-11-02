"""Pandas/PyTables adapter for trusted research inputs, gated by raw verification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from urbangraph.data import DataError, verify_snapshot
from urbangraph.data.contracts import validate_relative_path
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.data.rights import require_snapshot_authorization
from urbangraph.datasets.contracts import CanonicalDataset, DatasetConfig, RawLineage
from urbangraph.datasets.registry import get_dataset_spec
from urbangraph.datasets.validation import temporal_profile, validate_frame


def parse_snapshot(
    snapshot_dir: str | Path,
    *,
    dataset_id: str | None = None,
    config: DatasetConfig | None = None,
) -> CanonicalDataset:
    """Read an entire pandas DataFrame; no streaming or preprocessing is implied.

    HDF/PyTables can deserialize pickled object metadata. Verification establishes
    byte integrity, not trust. Open only trusted research files in this process.
    """
    approved_identity = require_snapshot_authorization(
        snapshot_dir, "local_research", Path(snapshot_dir).parent, operation="snapshot_parse"
    )
    config = config or DatasetConfig()
    directory = safe_absolute(snapshot_dir)
    verified = verify_snapshot(directory)
    if not verified.valid or verified.manifest is None:
        raise DataError("Raw snapshot failed integrity verification")
    manifest = verified.manifest
    if manifest.snapshot_id != approved_identity:
        raise DataError("Raw snapshot identity differs from rights grant")
    if dataset_id is not None and dataset_id != manifest.dataset_id:
        raise DataError("Requested dataset identity differs from raw manifest")
    spec = get_dataset_spec(manifest.dataset_id)
    relative = config.payload_file or spec.filename
    validate_relative_path(relative)
    matches = [record for record in manifest.files if record.relative_path == relative]
    if len(matches) != 1:
        raise DataError("Expected HDF payload is absent from verified inventory")
    manifest_digest = stream_file(directory / "manifest.json")[1]
    try:
        with pd.HDFStore(str(directory / "payload" / relative), mode="r") as store:
            keys = store.keys()
            if not keys:
                raise DataError("HDF store contains no pandas objects")
            if config.hdf_key is None:
                if len(keys) != 1:
                    raise DataError("Ambiguous multi-key HDF store; supply an explicit HDF key")
                selected = keys[0]
            else:
                selected = "/" + config.hdf_key.lstrip("/")
                if selected not in keys:
                    raise DataError("Requested HDF key does not exist")
            loaded = store.get(selected)
    except DataError:
        raise
    except Exception as exc:
        # Bound the external deserialization boundary; do not print raw exception
        # strings which may include payloads or sensitive source paths.
        raise DataError(f"Cannot read pandas-compatible HDF store ({type(exc).__name__})") from exc
    if not isinstance(loaded, pd.DataFrame):
        raise DataError("Selected HDF object must be a pandas DataFrame")
    sensors = validate_frame(loaded)
    resolved = replace(config, hdf_key=selected, payload_file=relative)
    index = pd.DatetimeIndex(loaded.index)
    temporal, findings = temporal_profile(index, resolved)
    axes: dict[str, object] = {
        "timestamp_dtype": str(index.dtype),
        "sensor_axis_dtype": str(loaded.columns.dtype),
        "timestamp_axis_name": repr(index.name),
        "sensor_axis_name": repr(loaded.columns.name),
    }
    # Copy the DataFrame container/axes only, leaving numeric buffers unchanged.
    frame = loaded.copy(deep=False)
    frame.index = index.as_unit("ns").rename("timestamp")
    frame.columns = pd.Index([sensor.identifier for sensor in sensors], name="sensor_id")
    again = verify_snapshot(directory)
    if (
        not again.valid
        or again.manifest != manifest
        or stream_file(directory / "manifest.json")[1] != manifest_digest
    ):
        raise DataError("Raw snapshot changed during parsing")
    return CanonicalDataset(
        frame,
        sensors,
        RawLineage(directory, manifest, manifest_digest, matches[0]),
        resolved,
        temporal,
        axes,
        findings,
    )
