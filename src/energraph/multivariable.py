"""Exact multivariable canonical tables and descriptive renewable quality audits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    broker_parent_references,
    governed_payload,
    require_committed_snapshot,
    resolve_broker_reference,
)
from energraph.broker_tree import inspect_tree
from energraph.canonical import _parse_csv_verified, logical_hash, quality
from energraph.contracts import EnergyVariable
from energraph.lineage import authorize_sources, resolve_parents
from energraph.variables import variables
from urbangraph.data import DataError, DatasetSnapshot, verify_snapshot
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.data.rights import (
    require_snapshot_authorization,
    require_verified_snapshot_authorization,
)
from urbangraph.datasets.serialization import read_json, write_json


def audit(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"axis": quality(frame), "variables": {}}
    columns: dict[str, object] = {}
    for name in frame:
        values = frame[name]
        valid = values.dropna()
        columns[str(name)] = {
            "finite": len(valid),
            "missing": int(values.isna().sum()),
            "zero": int((values == 0).sum()),
            "negative": int((values < 0).sum()),
            "first_valid": str(valid.index[0]) if len(valid) else None,
            "last_valid": str(valid.index[-1]) if len(valid) else None,
            "maximum": float(valid.max()) if len(valid) else None,
            "max_hourly_change": float(
                np.nanmax(np.abs(np.diff(values.to_numpy(dtype=np.float64))))
            )
            if len(valid) > 1
            else None,
        }
        capacity = str(name) + "_capacity"
        if capacity in frame:
            matched = values.notna() & frame[capacity].notna()
            result[str(name) + "_capacity_check"] = {
                "matched": int(matched.sum()),
                "above_capacity": int((matched & (values > frame[capacity])).sum()),
                "policy": "flag only; no clipping or capacity extrapolation",
            }
    result["variables"] = columns
    if {"wind", "wind_onshore", "wind_offshore"} <= set(frame.columns):
        difference = (frame.wind - frame.wind_onshore - frame.wind_offshore).dropna()
        result["wind_components"] = {
            "matched": len(difference),
            "max_absolute_difference_MW": float(difference.abs().max())
            if len(difference)
            else None,
            "nonzero_difference": int((difference.abs() > 1e-6).sum()),
            "policy": "use upstream total; never add total and components",
        }
    return result


def parse_variables(
    path: Path,
    names: tuple[str, ...],
    *,
    raw_snapshot_dir: Path | None = None,
    destination: Path | None = None,
) -> pd.DataFrame:
    if raw_snapshot_dir is None or destination is None:
        raise DataError("Current source authorization required for multivariable CSV read")
    raw_snapshot_dir, path = safe_absolute(raw_snapshot_dir), safe_absolute(path)
    approved = require_verified_snapshot_authorization(
        raw_snapshot_dir, "local_research", destination, operation="energy_source_read"
    )
    raw = verify_snapshot(raw_snapshot_dir)
    if not raw.valid or raw.manifest is None or raw.manifest.snapshot_id != approved:
        raise DataError("Multivariable source differs from current read authority")
    if path.parent != raw_snapshot_dir / "payload" or path.name not in {
        item.relative_path for item in raw.manifest.files
    }:
        raise DataError("Multivariable CSV is not in the authorized source inventory")
    return _parse_variables_verified(path, names)


def _parse_variables_verified(path: Path, names: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for var in variables(names):
        if var.observation_type == "known-calendar":
            raise DataError("Calendar is derived, not a CSV observation")
        # The existing strict parser operates on opaque numeric units; this adapter
        # records actual units in the versioned variable registry, not its legacy label.
        legacy = EnergyVariable(column=var.upstream)
        parts.append(
            _parse_csv_verified(path, legacy).rename(columns={var.upstream: var.identifier})
        )
    frame = pd.concat(parts, axis=1)
    quality(frame)
    return frame


def materialize_multi(
    snapshot: Path,
    root: Path,
    filename: str,
    names: tuple[str, ...],
    *,
    broker_context: BrokerPublicationContext | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> DatasetSnapshot | PendingPublication:
    approved_identity = require_snapshot_authorization(
        snapshot, "canonicalize", root, operation="energy_multivariable"
    )
    raw = verify_snapshot(snapshot)
    if raw.valid and raw.manifest is not None and raw.manifest.snapshot_id != approved_identity:
        raise DataError("Energy raw snapshot differs from rights grant")
    if not raw.valid or raw.manifest is None:
        raise DataError("Source snapshot failed integrity verification")
    if filename not in {f.relative_path for f in raw.manifest.files}:
        raise DataError("Source filename absent from verified inventory")
    if (broker_context is None) != (raw_broker_reference is None):
        raise DataError("Broker publication requires both broker context and raw identity")
    if broker_context is not None and raw_broker_reference is not None:
        record = broker_context.broker.authorize_directory(
            raw_broker_reference.artifact_id,
            purpose="canonicalize",
            operation="energy_multivariable",
        )
        if (
            raw_broker_reference.artifact_type != "raw"
            or record.kind != "raw"
            or record.content_sha256 != raw_broker_reference.content_sha256
            or record.revision != raw_broker_reference.revision
            or inspect_tree(snapshot).digest != record.content_sha256
        ):
            raise DataError("Multivariable raw snapshot differs from broker authority")
    source = snapshot / "payload" / filename
    before = stream_file(source)
    frame = _parse_variables_verified(source, names)
    if stream_file(source) != before:
        raise DataError("Source changed during multivariable parsing")
    descriptor = {
        "schema_version": "energy-multivariable-1.0",
        "variables": [v.to_dict() for v in variables(names)],
        "layout": "UTC rows; (DE country, variable) columns; variables are attributes",
        "source_manifest": raw.manifest.to_dict(),
        "source_file": filename,
        "source_sha256": before[1],
        "logical_sha256": logical_hash(frame),
        "quality": audit(frame),
        "mask": "per-variable finite; zero valid",
    }
    with tempfile.TemporaryDirectory(prefix="energy-multi-") as tmp:
        stage = Path(tmp)
        frame.to_parquet(stage / "observations.parquet")
        pd.testing.assert_frame_equal(
            frame,
            pd.read_parquet(stage / "observations.parquet"),
            check_exact=True,
            check_freq=False,
        )
        write_json(stage / "descriptor.json", descriptor)
        result = _publish_governed(
            stage,
            root,
            "multivariable",
            raw_snapshot_dir=snapshot,
            broker_context=broker_context,
            broker_parents=(raw_broker_reference,) if raw_broker_reference is not None else None,
        )
    if isinstance(result, PendingPublication):
        return result
    open_multi(
        result.directory,
        raw_snapshot_dir=snapshot,
        broker_context=broker_context,
        broker_reference=result.broker_reference,
        raw_broker_reference=raw_broker_reference,
    )
    return result


def materialize_joined_multi(
    inputs: tuple[tuple[Path, str, tuple[str, ...]], ...], root: Path
) -> DatasetSnapshot:
    """Join two independently granted invented/raw variable sources on one UTC axis.

    Both sources are verified and authorized before either protected CSV is read.
    This is an exact same-axis join; no interpolation or implicit weather alignment.
    """
    if len(inputs) != 2:
        raise DataError("Exactly two independently identified sources required")
    ordered = tuple(sorted(inputs, key=lambda item: str(safe_absolute(item[0]))))
    snapshots = tuple(safe_absolute(item[0]) for item in ordered)
    if len(set(snapshots)) != 2:
        raise DataError("Duplicate joined source")
    names = tuple(name for _, _, group in ordered for name in group)
    variables(names)
    graph = resolve_parents(snapshots)
    authorize_sources(
        graph.sources,
        purpose="canonicalize",
        operation="energy_multivariable",
        destination=safe_absolute(root),
    )
    frames: list[pd.DataFrame] = []
    files: list[dict[str, object]] = []
    for snapshot, filename, group in ordered:
        checked = verify_snapshot(snapshot)
        if (
            not checked.valid
            or checked.manifest is None
            or filename not in {item.relative_path for item in checked.manifest.files}
        ):
            raise DataError("Joined source file absent from verified raw inventory")
        source = snapshot / "payload" / filename
        before = stream_file(source)
        frame = _parse_variables_verified(source, group)
        if stream_file(source) != before:
            raise DataError("Joined source changed during parsing")
        frames.append(frame)
        files.append(
            {
                "snapshot_id": checked.manifest.snapshot_id,
                "file": filename,
                "source_sha256": before[1],
                "variables": list(group),
            }
        )
    if not frames[0].index.equals(frames[1].index):
        raise DataError("Joined sources need an identical exact UTC axis")
    frame = pd.concat(frames, axis=1)
    descriptor = {
        "schema_version": "energy-multivariable-joined-1.0",
        "variables": [value.to_dict() for value in variables(names)],
        "source_snapshot_ids": [snapshot.name for snapshot in snapshots],
        "source_files": files,
        "layout": "exact same UTC rows; distinct DE variable columns",
        "logical_sha256": logical_hash(frame),
        "quality": audit(frame),
        "mask": "per-variable finite; zero valid",
    }
    with tempfile.TemporaryDirectory(prefix="energy-joined-multi-") as tmp:
        stage = Path(tmp)
        frame.to_parquet(stage / "observations.parquet")
        pd.testing.assert_frame_equal(
            frame,
            pd.read_parquet(stage / "observations.parquet"),
            check_exact=True,
            check_freq=False,
        )
        write_json(stage / "descriptor.json", descriptor)
        result = require_committed_snapshot(
            _publish_governed(stage, root, "multivariable", parents=snapshots)
        )
    open_multi(result.directory, raw_snapshot_dirs=snapshots)
    return result


def project_multi(
    parent: Path,
    raw_snapshot_dirs: tuple[Path, ...],
    root: Path,
    names: tuple[str, ...],
) -> DatasetSnapshot:
    """Select an exact variable subset from an authorized governed table."""
    variables(names)
    parent = safe_absolute(parent)
    sources = resolve_parents((parent,)).sources
    if sources != resolve_parents(raw_snapshot_dirs).sources:
        raise DataError("Projected table raw source set does not match parent closure")
    authorize_sources(
        sources,
        purpose="canonicalize",
        operation="energy_multivariable_project",
        destination=safe_absolute(root),
    )
    frame, _ = open_multi(parent, raw_snapshot_dirs=raw_snapshot_dirs)
    if any(name not in frame for name in names):
        raise DataError("Projected variable absent from verified parent")
    projected = frame.loc[:, list(names)].copy()
    descriptor = {
        "schema_version": "energy-multivariable-projection-1.0",
        "variables": [value.to_dict() for value in variables(names)],
        "source_snapshot_ids": [Path(str(item["path"])).name for item in sources],
        "parent_id": parent.name,
        "layout": "exact same UTC rows; selected variable columns",
        "logical_sha256": logical_hash(projected),
        "quality": audit(projected),
        "mask": "per-variable finite; zero valid",
    }
    with tempfile.TemporaryDirectory(prefix="energy-project-multi-") as tmp:
        stage = Path(tmp)
        projected.to_parquet(stage / "observations.parquet")
        pd.testing.assert_frame_equal(
            projected,
            pd.read_parquet(stage / "observations.parquet"),
            check_exact=True,
            check_freq=False,
        )
        write_json(stage / "descriptor.json", descriptor)
        result = require_committed_snapshot(
            _publish_governed(stage, root, "multivariable-projection", parents=(parent,))
        )
    open_multi(result.directory, raw_snapshot_dirs=raw_snapshot_dirs, parent_artifact=parent)
    return result


def open_multi(
    directory: Path,
    *,
    raw_snapshot_dir: Path | None = None,
    raw_snapshot_dirs: tuple[Path, ...] | None = None,
    parent_artifact: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    broker_reference: BrokerArtifactRef | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if (raw_snapshot_dir is None) == (raw_snapshot_dirs is None):
        raise DataError("Current upstream source context required for multivariable read")
    snapshots = raw_snapshot_dirs if raw_snapshot_dirs is not None else (raw_snapshot_dir,)
    if snapshots is None or any(snapshot is None for snapshot in snapshots):
        raise DataError("Incomplete multivariable source context")
    snapshots = tuple(safe_absolute(snapshot) for snapshot in snapshots if snapshot is not None)
    parent_set = (safe_absolute(parent_artifact),) if parent_artifact is not None else snapshots
    brokered = broker_context is not None or broker_reference is not None
    if brokered:
        if (
            broker_context is None
            or broker_reference is None
            or raw_broker_reference is None
            or parent_artifact is not None
            or len(snapshots) != 1
            or broker_reference.artifact_type != "multivariable"
            or raw_broker_reference.artifact_type != "raw"
        ):
            raise DataError("Complete broker multivariable identity required")
        resolved = resolve_broker_reference(
            broker_context.broker,
            broker_reference.artifact_id,
            purpose="local_research",
            operation="energy_multivariable_read",
        )
        direct = broker_parent_references(
            broker_context.broker,
            resolved,
            purpose="local_research",
            operation="energy_multivariable_read",
        )
        raw_record = broker_context.broker.authorize_directory(
            raw_broker_reference.artifact_id,
            purpose="local_research",
            operation="energy_multivariable_read",
        )
        if (
            resolved != broker_reference
            or direct != (raw_broker_reference,)
            or raw_record.content_sha256 != raw_broker_reference.content_sha256
            or inspect_tree(snapshots[0]).digest != raw_record.content_sha256
        ):
            raise DataError("Multivariable broker source identity changed")
    elif (
        broker_context is not None
        or broker_reference is not None
        or raw_broker_reference is not None
    ):
        raise DataError("Incomplete broker multivariable context")
    if not brokered and resolve_parents(parent_set).sources != resolve_parents(snapshots).sources:
        raise DataError("Multivariable parent context differs from raw source closure")
    if brokered:
        assert broker_context is not None and broker_reference is not None
        assert raw_broker_reference is not None
        base = governed_payload(
            broker_reference,
            {"descriptor.json", "observations.parquet"},
            kind="multivariable",
            broker=broker_context.broker,
            broker_parents=(raw_broker_reference,),
            materialization_root=broker_context.materialization_root,
        )
    else:
        base = governed_payload(
            directory,
            {"descriptor.json", "observations.parquet"},
            parents=parent_set,
            kind="multivariable-projection" if parent_artifact is not None else "multivariable",
        )
    desc = read_json(base / "descriptor.json")
    schema = desc.get("schema_version")
    if schema not in {
        "energy-multivariable-1.0",
        "energy-multivariable-joined-1.0",
        "energy-multivariable-projection-1.0",
    }:
        raise DataError("Unsupported multivariable schema")
    if schema == "energy-multivariable-1.0":
        if len(snapshots) != 1:
            raise DataError("Single-source multivariable context required")
        verified = verify_snapshot(snapshots[0])
        if (
            not verified.valid
            or verified.manifest is None
            or desc.get("source_manifest") != verified.manifest.to_dict()
        ):
            raise DataError("Multivariable raw lineage mismatch")
    elif desc.get("source_snapshot_ids") != [snapshot.name for snapshot in snapshots]:
        raise DataError("Joined multivariable raw lineage mismatch")
    if schema == "energy-multivariable-joined-1.0":
        if len(snapshots) != 2 or parent_artifact is not None:
            raise DataError("Joined multivariable requires two direct raw parents")
        recorded = desc.get("source_files")
        if not isinstance(recorded, list) or len(recorded) != 2:
            raise DataError("Incomplete joined source inventory")
        for snapshot, item in zip(snapshots, recorded, strict=True):
            if not isinstance(item, dict) or set(item) != {
                "snapshot_id",
                "file",
                "source_sha256",
                "variables",
            }:
                raise DataError("Invalid joined source record")
            checked = verify_snapshot(snapshot)
            if not checked.valid or checked.manifest is None:
                raise DataError("Joined raw source failed integrity")
            matches = [
                file for file in checked.manifest.files if file.relative_path == item["file"]
            ]
            if (
                len(matches) != 1
                or item["snapshot_id"] != checked.manifest.snapshot_id
                or item["source_sha256"] != matches[0].sha256
            ):
                raise DataError("Joined source file identity mismatch")
    if schema == "energy-multivariable-projection-1.0" and (
        parent_artifact is None or desc.get("parent_id") != parent_artifact.name
    ):
        raise DataError("Projected multivariable parent mismatch")
    frame = pd.read_parquet(base / "observations.parquet")
    expected = [v.to_dict() for v in variables(tuple(str(c) for c in frame.columns))]
    if (
        desc.get("variables") != expected
        or any(d != np.dtype("float64") for d in frame.dtypes)
        or desc.get("logical_sha256") != logical_hash(frame)
        or desc.get("quality") != audit(frame)
    ):
        raise DataError("Multivariable logical content mismatch")
    return frame, desc
