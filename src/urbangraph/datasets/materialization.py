"""Versioned, integrity-checked canonical artifacts with verified raw lineage."""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from urbangraph.data import DataError, RawFileRecord, verify_snapshot
from urbangraph.data.contracts import manifest_from_dict
from urbangraph.data.filesystem import copy_file, discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.datasets.canonical import (
    WRITER_POLICY,
    _decode_observations,
    assert_equivalent,
    write_observations,
)
from urbangraph.datasets.contracts import (
    CANONICAL_VERSION,
    PARSER_VERSION,
    PROFILE_VERSION,
    CanonicalDataset,
    DatasetConfig,
    RawLineage,
    SensorIdentity,
)
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.profiling import profile_dataset
from urbangraph.datasets.serialization import (
    canonical_json,
    config_from_dict,
    object_dict,
    read_json,
    write_json,
)
from urbangraph.datasets.validation import temporal_profile, validate_frame
from urbangraph.reproducibility import collect_environment_info

_ARTIFACTS = ("observations.parquet", "quality.json", "raw-manifest.json", "schema.json")
_DEPENDENCIES = ("numpy", "pandas", "tables", "pyarrow", "pytz", "tzdata", "blosc2", "numexpr")


def dependency_versions() -> dict[str, str]:
    result = {name: version(name) for name in _DEPENDENCIES}
    result.update(
        python=platform.python_version(),
        hdf5=str(importlib.import_module("tables").hdf5_version),
        arrow_cpp=str(importlib.import_module("pyarrow").cpp_version),
    )
    return result


def transformation_descriptor(
    dataset: CanonicalDataset, dependencies: dict[str, str] | None = None
) -> dict[str, object]:
    return {
        "identity_version": "canonical-transformation-v1",
        "dataset_id": dataset.dataset_id,
        "raw_snapshot_id": dataset.raw_snapshot_id,
        "raw_manifest_sha256": dataset.lineage.manifest_sha256,
        "input_file": asdict(dataset.lineage.input_file),
        "canonical_schema_version": CANONICAL_VERSION,
        "parser_version": PARSER_VERSION,
        "profile_version": PROFILE_VERSION,
        "configuration": dataset.config.to_dict(),
        "dependencies": dependency_versions() if dependencies is None else dependencies,
        "writer": WRITER_POLICY,
    }


def transformation_identity(descriptor: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(descriptor)).hexdigest()


@dataclass(frozen=True)
class MaterializationResult:
    directory: Path
    transformation_id: str
    reused: bool


@dataclass(frozen=True)
class CanonicalVerification:
    valid: bool
    issues: tuple[str, ...]
    transformation_id: str | None = None


def _load_verified(
    directory: Path, *, enforce_location: bool
) -> tuple[pd.DataFrame, dict[str, object]]:
    directory = safe_absolute(directory)
    tree = discover(directory)
    if tree.directories or set(tree.files) != {"manifest.json", *_ARTIFACTS}:
        raise DataError("Canonical artifact inventory is incomplete or contains unexpected entries")
    manifest = read_json(directory / "manifest.json")
    if (
        set(manifest)
        != {
            "schema_version",
            "transformation_id",
            "descriptor",
            "created_at",
            "environment",
            "artifacts",
        }
        or manifest["schema_version"] != "1.0"
    ):
        raise DataError("Unsupported canonical manifest schema")
    created = manifest["created_at"]
    if not isinstance(created, str):
        raise DataError("Invalid canonical creation time")
    try:
        offset = datetime.fromisoformat(created).utcoffset()
    except ValueError as exc:
        raise DataError("Invalid canonical creation time") from exc
    if offset is None or offset.total_seconds() != 0:
        raise DataError("Canonical creation time must be UTC")
    descriptor = object_dict(manifest["descriptor"])
    transform_id = transformation_identity(descriptor)
    if manifest["transformation_id"] != transform_id:
        raise DataError("Canonical transformation identity mismatch")
    if enforce_location and (
        directory.name != transform_id
        or directory.parent.name != descriptor.get("raw_snapshot_id")
        or directory.parent.parent.name != descriptor.get("dataset_id")
    ):
        raise DataError("Canonical directory identity mismatch")
    inventory = manifest["artifacts"]
    if not isinstance(inventory, list) or len(inventory) != len(_ARTIFACTS):
        raise DataError("Invalid artifact checksum inventory")
    records: list[RawFileRecord] = []
    for item in inventory:
        obj = object_dict(item)
        if set(obj) != {"relative_path", "size_bytes", "sha256"}:
            raise DataError("Invalid artifact checksum fields")
        relative, size, digest = obj["relative_path"], obj["size_bytes"], obj["sha256"]
        if not isinstance(relative, str) or type(size) is not int or not isinstance(digest, str):
            raise DataError("Invalid artifact checksum types")
        records.append(RawFileRecord(relative, size, digest))
    if tuple(record.relative_path for record in records) != _ARTIFACTS:
        raise DataError("Invalid artifact ordering or names")
    for record in records:
        if stream_file(directory / record.relative_path) != (record.size_bytes, record.sha256):
            raise DataError(f"Artifact checksum mismatch: {record.relative_path}")
    raw = manifest_from_dict(read_json(directory / "raw-manifest.json"))
    # Reuse the raw environment validator without changing its public contract.
    environment_check = raw.to_dict()
    environment_check["creator_environment"] = manifest["environment"]
    manifest_from_dict(environment_check)
    raw_digest = stream_file(directory / "raw-manifest.json")[1]
    config = config_from_dict(descriptor.get("configuration"))
    matches = [record for record in raw.files if record.relative_path == config.payload_file]
    if len(matches) != 1 or not config.hdf_key or not config.hdf_key.startswith("/"):
        raise DataError("Canonical input reference is absent or unresolved")
    dependencies_obj = object_dict(descriptor.get("dependencies"))
    if set(dependencies_obj) != {*_DEPENDENCIES, "python", "hdf5", "arrow_cpp"} or any(
        not isinstance(v, str) or not v for v in dependencies_obj.values()
    ):
        raise DataError("Invalid dependency provenance")
    dependencies = {key: str(value) for key, value in dependencies_obj.items()}
    schema = read_json(directory / "schema.json")
    frame = _decode_observations(directory / "observations.parquet", schema)
    validate_frame(frame)
    sensor_objects = schema.get("sensors")
    if not isinstance(sensor_objects, list):
        raise DataError("Missing sensor metadata")
    sensors: list[SensorIdentity] = []
    for value in sensor_objects:
        sensor = object_dict(value)
        if set(sensor) != {"identifier", "original_type", "original_value", "dtype"} or not all(
            isinstance(v, str) for v in sensor.values()
        ):
            raise DataError("Invalid original sensor identity metadata")
        sensors.append(SensorIdentity(**{key: str(item) for key, item in sensor.items()}))
    temporal, findings = temporal_profile(pd.DatetimeIndex(frame.index), config)
    source_axes = object_dict(schema.get("source_axes"))
    if set(source_axes) != {
        "timestamp_dtype",
        "sensor_axis_dtype",
        "timestamp_axis_name",
        "sensor_axis_name",
    } or not all(isinstance(v, str) for v in source_axes.values()):
        raise DataError("Invalid source axis metadata")
    restored = CanonicalDataset(
        frame,
        tuple(sensors),
        RawLineage(directory, raw, raw_digest, matches[0]),
        config,
        temporal,
        source_axes,
        findings,
    )
    if schema != restored.schema_dict() or descriptor != transformation_descriptor(
        restored, dependencies
    ):
        raise DataError("Canonical schema or lineage descriptor is inconsistent")
    report = profile_dataset(restored)
    if not report.accepted or read_json(directory / "quality.json") != report.to_dict():
        raise DataError("Quality report differs from canonical data or policy")
    if discover(directory) != tree or read_json(directory / "manifest.json") != manifest:
        raise DataError("Canonical output changed during verification")
    return frame, manifest


def verify_materialization(
    directory: str | Path, *, raw_snapshot_dir: str | Path | None = None
) -> CanonicalVerification:
    try:
        if raw_snapshot_dir is None:
            raise DataError("Raw snapshot identity required for canonical verification")
        # A supplied path selects a grant; it is not authority. Establish current
        # source identity and authorization before touching canonical artifacts.
        expected_snapshot = require_verified_snapshot_authorization(
            raw_snapshot_dir,
            "local_research",
            Path(directory),
            operation="canonical_read",
        )
        return _verify_materialization_after_authorization(
            directory, raw_snapshot_dir=raw_snapshot_dir, expected_snapshot=expected_snapshot
        )
    except (DataError, OSError, ValueError, TypeError, KeyError) as exc:
        message = (
            str(exc)
            if isinstance(exc, DataError)
            else f"Cannot verify canonical artifacts ({type(exc).__name__})"
        )
        return CanonicalVerification(False, (message,))


def _verify_materialization_after_authorization(
    directory: str | Path,
    *,
    raw_snapshot_dir: str | Path,
    expected_snapshot: str,
) -> CanonicalVerification:
    """Internal verification for a caller that already established current authority."""
    try:
        artifact = Path(directory)
        raw = verify_snapshot(raw_snapshot_dir)
        if not raw.valid or raw.manifest is None or raw.manifest.snapshot_id != expected_snapshot:
            raise DataError("External raw snapshot differs from current canonical rights")
        _, manifest = _load_verified(artifact, enforce_location=True)
        descriptor = object_dict(manifest["descriptor"])
        if (
            descriptor["raw_snapshot_id"] != expected_snapshot
            or raw.manifest.dataset_id != descriptor["dataset_id"]
            or stream_file(Path(raw_snapshot_dir) / "manifest.json")[1]
            != descriptor["raw_manifest_sha256"]
        ):
            raise DataError("External raw snapshot does not match canonical lineage")
        return CanonicalVerification(True, (), str(manifest["transformation_id"]))
    except (DataError, OSError, ValueError, TypeError, KeyError) as exc:
        message = (
            str(exc)
            if isinstance(exc, DataError)
            else f"Cannot verify canonical artifacts ({type(exc).__name__})"
        )
        return CanonicalVerification(False, (message,))


def load_materialization(
    directory: str | Path, *, raw_snapshot_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load observations only through a current upstream-rights decision."""
    if raw_snapshot_dir is None:
        raise DataError("Raw snapshot identity required for canonical read")
    from urbangraph.datasets.governed_canonical import read_canonical

    return read_canonical(directory, raw_snapshot_dir=raw_snapshot_dir)


def materialize_snapshot(
    snapshot_dir: str | Path,
    *,
    output_root: str | Path = "data/interim",
    dataset_id: str | None = None,
    config: DatasetConfig | None = None,
) -> MaterializationResult:
    directory, root = safe_absolute(snapshot_dir), safe_absolute(output_root)
    from urbangraph.data.rights import require_snapshot_authorization

    approved = require_snapshot_authorization(
        directory, "local_research", root, operation="canonical_create"
    )
    verified = verify_snapshot(directory)
    if not verified.valid or verified.manifest is None or verified.manifest.snapshot_id != approved:
        raise DataError("Canonical source identity differs from current creation grant")
    # A UG-001 snapshot's grandparent is its managed raw root. Protect that whole tree.
    raw_root = directory.parent.parent
    if root.is_relative_to(raw_root) or raw_root.is_relative_to(root):
        raise DataError("Canonical output and managed raw root must be disjoint")
    dataset = parse_snapshot(directory, dataset_id=dataset_id, config=config)
    report = profile_dataset(dataset)
    if not report.accepted:
        raise DataError("Configured quality rejection policy failed")
    descriptor = transformation_descriptor(dataset)
    transform_id = transformation_identity(descriptor)
    namespace = safe_absolute(root / dataset.dataset_id / dataset.raw_snapshot_id)
    namespace.mkdir(parents=True, exist_ok=True)
    destination = namespace / transform_id
    with _writer_lock(namespace):
        if os.path.lexists(destination):
            frame, existing = _load_verified(destination, enforce_location=True)
            if existing["descriptor"] != descriptor:
                raise DataError("Existing canonical transformation conflicts")
            assert_equivalent(dataset.observations, frame)
            return MaterializationResult(destination, transform_id, True)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=namespace))
        try:
            write_observations(dataset, stage / "observations.parquet")
            write_json(stage / "schema.json", dataset.schema_dict())
            write_json(stage / "quality.json", report.to_dict())
            copy_file(directory / "manifest.json", stage / "raw-manifest.json")
            records = []
            for name in _ARTIFACTS:
                size, digest = stream_file(stage / name)
                records.append(RawFileRecord(name, size, digest))
            write_json(
                stage / "manifest.json",
                {
                    "schema_version": "1.0",
                    "transformation_id": transform_id,
                    "descriptor": descriptor,
                    "created_at": datetime.now(UTC).isoformat(),
                    "environment": collect_environment_info().to_dict(),
                    "artifacts": [asdict(record) for record in records],
                },
            )
            restored, _ = _load_verified(stage, enforce_location=False)
            assert_equivalent(dataset.observations, restored)
            raw = verify_snapshot(directory)
            if (
                not raw.valid
                or raw.manifest != dataset.lineage.manifest
                or stream_file(directory / "manifest.json")[1] != dataset.lineage.manifest_sha256
            ):
                raise DataError("Raw snapshot changed during materialization")
            _publish(stage, destination)
            return MaterializationResult(destination, transform_id, False)
        finally:
            try:
                _cleanup_stage(stage, namespace)
            except OSError as exc:
                raise DataError(
                    "Canonical staging cleanup failed; manual recovery required"
                ) from exc
