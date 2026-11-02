import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from urbangraph.data import (
    DataError,
    DatasetSource,
    filesystem,
    ingest_dataset,
    ingestion,
    verify_snapshot,
)
from urbangraph.data.contracts import manifest_from_dict
from urbangraph.data.rights import require_raw_authorization


def _source_state(source: Path) -> dict[str, tuple[bytes | None, int]]:
    return {
        path.relative_to(source).as_posix(): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in [source, *source.rglob("*")]
    }


def test_invented_grant_is_exact_and_expires(
    source_dir: Path,
    tmp_path: Path,
    invented_source_grant: Callable[[Path, str, Path], None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "approved-raw"
    invented_source_grant(source_dir, "metr-la", destination)
    digest = require_raw_authorization("metr-la", source_dir, "local_research", destination)
    assert len(digest) == 64
    for source, identity, purpose, root in (
        (tmp_path / "other-source", "metr-la", "local_research", destination),
        (source_dir, "pems-bay", "local_research", destination),
        (source_dir, "metr-la", "public_redistribution", destination),
        (source_dir, "metr-la", "local_research", tmp_path / "other-raw"),
    ):
        with pytest.raises(DataError):
            require_raw_authorization(identity, source, purpose, root)
    (source_dir / "metr-la.h5").write_bytes(b"changed")
    with pytest.raises(DataError, match="content does not match rights grant"):
        ingest_dataset(source_dir, "metr-la", raw_root=destination)
    policy = tmp_path / "invented-rights-policy.json"
    document = json.loads(policy.read_text(encoding="utf-8"))
    document["grants"][0]["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    policy.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    with pytest.raises(DataError, match="outside validity window"):
        require_raw_authorization("metr-la", source_dir, "local_research", destination)


def test_success_and_source_unchanged(
    source_dir: Path,
    tmp_path: Path,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    (source_dir / "empty-dir").mkdir()
    # A source manifest name is harmless inside the payload namespace.
    (source_dir / "manifest.json").write_bytes(b"opaque source manifest")
    before = _source_state(source_dir)
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    provenance = DatasetSource("local-fixture", "Synthetic bytes", "Test author", "Not a benchmark")
    snapshot = ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw", source=provenance)
    assert _source_state(source_dir) == before
    assert snapshot.directory.parent.name == "metr-la"
    assert snapshot.directory.name == snapshot.manifest.snapshot_id
    assert snapshot.manifest.source == provenance
    assert not snapshot.reused
    manifest = json.loads(snapshot.manifest_path.read_text("utf-8"))
    assert manifest_from_dict(manifest) == snapshot.manifest
    assert manifest["schema_version"] == "1.0"
    assert manifest["manifest_version"] == 1
    assert manifest["created_at"].endswith("+00:00")
    assert manifest["files"] == sorted(manifest["files"], key=lambda row: row["relative_path"])
    for record in snapshot.manifest.files:
        assert (snapshot.directory / "payload" / record.relative_path).read_bytes() == (
            source_dir / record.relative_path
        ).read_bytes()
    assert verify_snapshot(snapshot.directory).valid
    assert not (snapshot.directory / "payload" / "empty-dir").exists()
    assert not list(snapshot.directory.parent.glob(".staging-*"))
    assert not (snapshot.directory.parent / ".ingest.lock").exists()


def test_idempotent_preserves_first_manifest(
    source_dir: Path,
    tmp_path: Path,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    root = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", root)
    first = ingest_dataset(source_dir, "metr-la", raw_root=root, source=DatasetSource("first"))
    before = _source_state(first.directory)
    again = ingest_dataset(source_dir, "metr-la", raw_root=root, source=DatasetSource("second"))
    assert again.reused
    assert again.directory == first.directory
    assert again.manifest == first.manifest
    assert again.manifest.source.reference == "first"
    assert _source_state(first.directory) == before


def test_identity_independent_of_metadata_root_and_dataset(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw-a")
    first = ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw-a")
    invented_source_grant(source_dir, "pems-bay", tmp_path / "raw-b")
    second = ingest_dataset(
        source_dir, "pems-bay", raw_root=tmp_path / "raw-b", source=DatasetSource("different")
    )
    assert first.manifest.snapshot_id == second.manifest.snapshot_id
    assert first.manifest.source != second.manifest.source


def test_new_content_is_additive(
    source_dir: Path,
    tmp_path: Path,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    root = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", root)
    first = ingest_dataset(source_dir, "metr-la", raw_root=root)
    (source_dir / "metr-la.h5").write_bytes(b"different")
    with pytest.raises(DataError, match="content does not match rights grant"):
        ingest_dataset(source_dir, "metr-la", raw_root=root)
    invented_source_grant(source_dir, "metr-la", root)
    second = ingest_dataset(source_dir, "metr-la", raw_root=root)
    assert first.directory != second.directory
    assert verify_snapshot(first.directory).valid
    assert verify_snapshot(second.directory).valid


def test_invented_source_grant_rejects_outside_tmp_path(
    source_dir: Path, tmp_path: Path, invented_source_grant
) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-outside-source")
    outside.mkdir()
    (outside / "fixture.bin").write_bytes(b"not the scoped fixture")
    with pytest.raises(ValueError, match="under this test's tmp_path"):
        invented_source_grant(outside, "metr-la", tmp_path / "raw")
    with pytest.raises(ValueError, match="does not support"):
        invented_source_grant(source_dir, "unrelated-external-dataset", tmp_path / "raw")


def test_invented_source_grant_rejects_parent_traversal(
    source_dir: Path, tmp_path: Path, invented_source_grant
) -> None:
    escaped = source_dir / ".." / "source"
    with pytest.raises(DataError, match="Parent traversal"):
        invented_source_grant(escaped, "metr-la", tmp_path / "raw")


def test_invented_source_grant_rejects_link_escape(tmp_path: Path, invented_source_grant) -> None:
    outside = tmp_path.parent / (tmp_path.name + "-link-target")
    outside.mkdir()
    (outside / "source.csv").write_text("synthetic only\n", encoding="utf-8")
    link = tmp_path / "linked-source"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host cannot create directory symlinks: {type(exc).__name__}")
    with pytest.raises(DataError, match="Symlinks and Windows reparse points"):
        invented_source_grant(link, "metr-la", tmp_path / "raw")


def test_zero_byte_resource(
    tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty.bin").touch()
    invented_source_grant(source, "future-city", tmp_path / "raw")
    snapshot = ingest_dataset(source, "future-city", raw_root=tmp_path / "raw")
    assert snapshot.manifest.total_bytes == 0
    assert verify_snapshot(snapshot.directory).valid


@pytest.mark.parametrize("kind", ["missing", "empty", "file"])
def test_invalid_source(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source"
    if kind == "empty":
        source.mkdir()
    elif kind == "file":
        source.write_bytes(b"not a directory")
    with pytest.raises((DataError, OSError)):
        ingest_dataset(source, "metr-la", raw_root=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("layout", ["inside-source", "source-inside-raw", "equal"])
def test_recursive_ingestion_prevented(
    source_dir: Path,
    tmp_path: Path,
    layout: str,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    root = source_dir / "raw" if layout == "inside-source" else tmp_path
    if layout == "equal":
        root = source_dir
    before = _source_state(source_dir)
    invented_source_grant(source_dir, "metr-la", root)
    with pytest.raises(DataError, match="disjoint"):
        ingest_dataset(source_dir, "metr-la", raw_root=root)
    assert _source_state(source_dir) == before


def test_invalid_dataset_before_writes(source_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(DataError):
        ingest_dataset(source_dir, "../outside", raw_root=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("failure_point", ["copy", "publish", "interrupt"])
def test_failure_cleanup(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    root = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", root)
    existing = ingest_dataset(source_dir, "metr-la", raw_root=root)
    original = _source_state(existing.directory)
    (source_dir / "new.bin").write_bytes(b"new")
    invented_source_grant(source_dir, "metr-la", root)
    source_before = _source_state(source_dir)

    def fail_copy(source: Path, destination: Path) -> tuple[int, str]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"partial")
        raise OSError("simulated full disk")

    def fail_publish(stage: Path, destination: Path) -> None:
        assert (stage / "manifest.json").is_file()
        assert not destination.exists()
        raise OSError("simulated rename failure")

    def interrupted(source: Path, destination: Path) -> tuple[int, str]:
        raise KeyboardInterrupt

    if failure_point == "publish":
        monkeypatch.setattr(ingestion, "_publish", fail_publish)
    else:
        monkeypatch.setattr(
            ingestion, "copy_file", interrupted if failure_point == "interrupt" else fail_copy
        )
    expected = KeyboardInterrupt if failure_point == "interrupt" else OSError
    with pytest.raises(expected):
        ingest_dataset(source_dir, "metr-la", raw_root=root)
    assert list(existing.directory.parent.iterdir()) == [existing.directory]
    assert _source_state(existing.directory) == original
    assert _source_state(source_dir) == source_before


def test_source_changes_abort(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    original_copy = filesystem.copy_file

    def changing(source: Path, destination: Path) -> tuple[int, str]:
        result = original_copy(source, destination)
        source.write_bytes(b"changed during ingestion")
        return result

    monkeypatch.setattr(ingestion, "copy_file", changing)
    with pytest.raises(DataError, match="changed"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert not list((tmp_path / "raw" / "metr-la").iterdir())


def test_source_addition_aborts(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    original_copy = filesystem.copy_file

    def adding(source: Path, destination: Path) -> tuple[int, str]:
        result = original_copy(source, destination)
        (source_dir / "new.bin").write_bytes(b"new")
        return result

    monkeypatch.setattr(ingestion, "copy_file", adding)
    with pytest.raises(DataError, match="inventory changed"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert not list((tmp_path / "raw" / "metr-la").iterdir())


def test_stale_writer_lock_is_preserved(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    directory = tmp_path / "raw" / "metr-la"
    directory.mkdir(parents=True)
    lock = directory / ".ingest.lock"
    lock.write_text("unknown writer", encoding="utf-8")
    with pytest.raises(DataError, match="writer lock"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert lock.read_text("utf-8") == "unknown writer"
    assert list(directory.iterdir()) == [lock]


def test_corruption_not_repaired(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    root = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", root)
    snapshot = ingest_dataset(source_dir, "metr-la", raw_root=root)
    payload = snapshot.directory / "payload" / "metr-la.h5"
    payload.write_bytes(b"corrupted")
    before = _source_state(snapshot.directory)
    with pytest.raises(DataError, match="corrupted or conflicting"):
        ingest_dataset(source_dir, "metr-la", raw_root=root)
    assert _source_state(snapshot.directory) == before


def test_existing_empty_destination_is_not_replaced(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "first")
    first = ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "first")
    root = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", root)
    conflict = root / "metr-la" / first.manifest.snapshot_id
    conflict.mkdir(parents=True)
    with pytest.raises(DataError, match="conflicting"):
        ingest_dataset(source_dir, "metr-la", raw_root=root)
    assert conflict.is_dir()
    assert not list(conflict.iterdir())


def test_publish_refuses_destination_that_appears(tmp_path: Path) -> None:
    stage, target = tmp_path / "stage", tmp_path / "target"
    stage.mkdir()
    (stage / "payload").write_bytes(b"new")
    target.mkdir()
    with pytest.raises(DataError, match="refusing overwrite"):
        ingestion._publish(stage, target)
    assert not list(target.iterdir())
    assert (stage / "payload").read_bytes() == b"new"


def test_cleanup_rejects_unowned_path(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(DataError, match="Refusing cleanup"):
        ingestion._cleanup_stage(unrelated, tmp_path)
    assert unrelated.exists()


def test_streaming_hash_and_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    monkeypatch.setattr(filesystem, "CHUNK_SIZE", 7)
    path, copied = tmp_path / "source", tmp_path / "copy"
    content = bytes(range(256)) * 5
    path.write_bytes(content)
    assert filesystem.copy_file(path, copied) == (len(content), hashlib.sha256(content).hexdigest())
    assert copied.read_bytes() == content
    with pytest.raises(FileExistsError):
        filesystem.copy_file(path, copied)


def test_hard_links_rejected(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    # Pin the valid invented source first; a hard-linked source cannot itself
    # be inventoried for a new grant and must fail during ingestion.
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    try:
        os.link(source_dir / "metr-la.h5", source_dir / "alias.h5")
    except OSError as exc:
        pytest.skip(f"Host cannot create hard links: {type(exc).__name__}")
    with pytest.raises(DataError, match="single hard link"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")


def test_staged_corruption_never_published(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    original_copy = filesystem.copy_file

    def corrupt_copy(source: Path, destination: Path) -> tuple[int, str]:
        result = original_copy(source, destination)
        destination.write_bytes(b"broken copy")
        return result

    monkeypatch.setattr(ingestion, "copy_file", corrupt_copy)
    with pytest.raises(DataError, match="Staged snapshot failed integrity"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert not list((tmp_path / "raw" / "metr-la").iterdir())


def test_cleanup_failure_reported(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")

    def fail_copy(source: Path, destination: Path) -> tuple[int, str]:
        raise OSError("simulated copy error")

    def fail_cleanup(stage: Path, dataset_dir: Path) -> None:
        raise PermissionError("simulated permission error")

    monkeypatch.setattr(ingestion, "copy_file", fail_copy)
    monkeypatch.setattr(ingestion, "_cleanup_stage", fail_cleanup)
    with pytest.raises(DataError, match="cleanup failed"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    leftovers = list((tmp_path / "raw" / "metr-la").iterdir())
    assert len(leftovers) == 1
    assert leftovers[0].name.startswith(".staging-")
    assert not verify_snapshot(leftovers[0]).valid


def test_active_writer_excludes_second_writer(
    source_dir: Path, tmp_path: Path, invented_source_grant: Callable[[Path, str, Path], None]
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    namespace = tmp_path / "raw" / "metr-la"
    namespace.mkdir(parents=True)
    with ingestion._writer_lock(namespace):
        lock_bytes = (namespace / ".ingest.lock").read_bytes()
        with pytest.raises(DataError, match="writer lock"):
            ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
        assert (namespace / ".ingest.lock").read_bytes() == lock_bytes
    assert not list(namespace.iterdir())


def test_manifest_write_bound_cleans_staging(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant: Callable[[Path, str, Path], None],
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    monkeypatch.setattr(ingestion, "MAX_MANIFEST_BYTES", 1)
    with pytest.raises(DataError, match="manifest exceeds"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert not list((tmp_path / "raw" / "metr-la").iterdir())
