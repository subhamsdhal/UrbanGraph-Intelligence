"""Append-only local raw-data ingestion with private staging and atomic publication."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from urbangraph.data.contracts import (
    DataError,
    DatasetIdentity,
    DatasetManifest,
    DatasetSnapshot,
    DatasetSource,
    RawFileRecord,
    content_identity,
)
from urbangraph.data.filesystem import copy_file, discover, reject_link, safe_absolute, stream_file
from urbangraph.data.rights import require_raw_authorization
from urbangraph.data.verification import MAX_MANIFEST_BYTES, _verify_snapshot, verify_snapshot
from urbangraph.reproducibility import collect_environment_info

_LOGGER = logging.getLogger("urbangraph.data")


@contextmanager
def _writer_lock(dataset_dir: Path) -> Iterator[None]:
    """Serialize cooperating writers; never automatically break a stale lock."""
    lock = dataset_dir / ".ingest.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise DataError(
            "Dataset writer lock exists; another writer or manual recovery is required"
        ) from exc
    owned = os.fstat(descriptor)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        current = reject_link(lock)
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise DataError("Writer lock was replaced; refusing to remove it")
        lock.unlink()


def _cleanup_stage(stage: Path, dataset_dir: Path) -> None:
    # Check the exact absolute target before recursive deletion, on Windows too.
    if stage.parent != dataset_dir or not stage.name.startswith(".staging-"):
        raise DataError("Refusing cleanup outside the owned staging directory")
    safe_absolute(stage)
    if stage.exists():
        reject_link(stage)
        shutil.rmtree(stage)


def _publish(stage: Path, destination: Path) -> None:
    # Never use replace(). The writer lock excludes other cooperating ingestors.
    # Trusted local control is required to rule out hostile check/rename races.
    if os.path.lexists(destination):
        raise DataError("Snapshot destination appeared before publication; refusing overwrite")
    stage.rename(destination)


def ingest_dataset(
    source_dir: str | Path,
    dataset_id: str,
    *,
    raw_root: str | Path = "data/raw",
    source: DatasetSource | None = None,
) -> DatasetSnapshot:
    digest = require_raw_authorization(dataset_id, source_dir, "local_research", raw_root)
    return _ingest_verified_source(
        source_dir, dataset_id, raw_root=raw_root, source=source, expected_digest=digest
    )


def _ingest_verified_source(
    source_dir: str | Path,
    dataset_id: str,
    *,
    raw_root: str | Path,
    source: DatasetSource | None,
    expected_digest: str,
    before_publish: Callable[[], None] | None = None,
) -> DatasetSnapshot:
    """Copy opaque regular files without changing source bytes or directory entries.

    Equal inventories reuse the original verified manifest, including its original
    provenance. Caller-provided metadata is not silently merged on subsequent calls.
    Trees must be stable and under trusted local control during the operation.
    """
    identity = DatasetIdentity(dataset_id)
    if source is not None and not isinstance(source, DatasetSource):
        raise DataError("Source provenance must be a DatasetSource")
    origin = safe_absolute(source_dir)
    root = safe_absolute(raw_root)
    if origin.is_relative_to(root) or root.is_relative_to(origin):
        raise DataError("Source and managed raw root must be disjoint, non-nested directories")
    tree = discover(origin)
    if not tree.files:
        raise DataError("Source directory contains no regular files")
    approved_inventory = tuple(
        RawFileRecord(relative, *stream_file(origin / relative)) for relative in tree.files
    )
    if content_identity(approved_inventory) != expected_digest:
        raise DataError("Source content does not match rights grant")
    dataset_dir = safe_absolute(root / identity.dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    safe_absolute(dataset_dir)
    with _writer_lock(dataset_dir):
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=dataset_dir))
        try:
            records = []
            for relative in tree.files:
                size, digest = copy_file(origin / relative, stage / "payload" / relative)
                records.append(RawFileRecord(relative, size, digest))
            inventory = tuple(records)
            # A second pass detects ordinary changes to already-copied source files.
            for record in inventory:
                if stream_file(origin / record.relative_path) != (record.size_bytes, record.sha256):
                    raise DataError("Source content changed during ingestion")
            if discover(origin) != tree:
                raise DataError("Source inventory changed during ingestion")
            manifest = DatasetManifest(
                dataset_id=identity.dataset_id,
                snapshot_id=content_identity(inventory),
                created_at=datetime.now(UTC).isoformat(),
                source=source if source is not None else DatasetSource(),
                files=inventory,
                creator_environment=collect_environment_info(),
            )
            encoded = (
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
            ).encode("utf-8")
            if len(encoded) > MAX_MANIFEST_BYTES:
                raise DataError("Generated manifest exceeds 16 MiB limit")
            with (stage / "manifest.json").open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            validation = _verify_snapshot(stage, enforce_location=False)
            if not validation.valid:
                reason = "; ".join(issue.message for issue in validation.issues)
                raise DataError(f"Staged snapshot failed integrity verification: {reason}")
            destination = dataset_dir / manifest.snapshot_id
            if before_publish is not None:
                before_publish()
            if os.path.lexists(destination):
                existing = verify_snapshot(destination)
                if not existing.valid or existing.manifest is None:
                    raise DataError(
                        "Existing snapshot is corrupted or conflicting; no files were replaced"
                    )
                if (
                    existing.manifest.files != inventory
                    or existing.manifest.dataset_id != dataset_id
                ):
                    raise DataError("Existing snapshot inventory conflicts with requested content")
                _LOGGER.info(
                    "Verified and reused dataset %s snapshot %s", dataset_id, manifest.snapshot_id
                )
                return DatasetSnapshot(destination, existing.manifest, reused=True)
            _publish(stage, destination)
            _LOGGER.info("Created dataset %s snapshot %s", dataset_id, manifest.snapshot_id)
            return DatasetSnapshot(destination, manifest)
        finally:
            try:
                _cleanup_stage(stage, dataset_dir)
            except OSError as exc:
                raise DataError("Staging cleanup failed; manual recovery is required") from exc
