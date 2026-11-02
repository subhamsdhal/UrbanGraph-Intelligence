import json
from pathlib import Path

import pytest

from urbangraph.data import DatasetSnapshot, ingest_dataset, verify_snapshot


@pytest.fixture
def snapshot(source_dir: Path, tmp_path: Path, invented_source_grant) -> DatasetSnapshot:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    return ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")


def test_successful_verification(snapshot: DatasetSnapshot) -> None:
    result = verify_snapshot(snapshot.directory)
    assert result.valid
    assert result.manifest == snapshot.manifest
    assert result.checked_files == 2
    assert result.issues == ()


@pytest.mark.parametrize("mutation", ["corrupt", "truncate", "missing", "extra", "extra-dir"])
def test_integrity_failures(snapshot: DatasetSnapshot, mutation: str) -> None:
    payload = snapshot.directory / "payload" / "metr-la.h5"
    if mutation == "corrupt":
        payload.write_bytes(b"X" * payload.stat().st_size)
    elif mutation == "truncate":
        payload.write_bytes(b"X")
    elif mutation == "missing":
        payload.unlink()
    elif mutation == "extra":
        (snapshot.directory / "payload" / "surprise.txt").write_bytes(b"unknown")
    else:
        (snapshot.directory / "empty-extra").mkdir()
    result = verify_snapshot(snapshot.directory)
    assert not result.valid
    codes = {issue.code for issue in result.issues}
    assert {
        "corrupt": "digest",
        "truncate": "size",
        "missing": "missing",
        "extra": "unexpected",
        "extra-dir": "unexpected_directory",
    }[mutation] in codes
    if mutation == "truncate":
        assert "digest" in codes


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "2.0"),
        ("manifest_version", 2),
        ("manifest_version", True),
        ("identity_version", "unknown"),
        ("snapshot_id", "0" * 64),
        ("created_at", "not-a-date"),
        ("created_at", "2026-10-09T12:00:00"),
        ("created_at", "2026-10-09T12:00:00+05:30"),
        ("dataset_id", "../outside"),
        ("source", {"reference": "missing fields"}),
        ("files", []),
        ("files", {}),
        ("creator_environment", {}),
    ],
)
def test_invalid_manifest_fields(snapshot: DatasetSnapshot, field: str, value: object) -> None:
    manifest = json.loads(snapshot.manifest_path.read_text("utf-8"))
    manifest[field] = value
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_snapshot(snapshot.directory)
    assert not result.valid
    assert result.issues[0].code == "invalid"


@pytest.mark.parametrize("mutation", ["missing-field", "unknown-field", "reverse", "duplicate"])
def test_manifest_structure(snapshot: DatasetSnapshot, mutation: str) -> None:
    manifest = json.loads(snapshot.manifest_path.read_text("utf-8"))
    if mutation == "missing-field":
        del manifest["snapshot_id"]
    elif mutation == "unknown-field":
        manifest["extra"] = True
    elif mutation == "reverse":
        manifest["files"].reverse()
    else:
        manifest["files"].insert(0, manifest["files"][0])
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not verify_snapshot(snapshot.directory).valid


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relative_path", "../../outside"),
        ("relative_path", "C:/outside"),
        ("relative_path", "a\\b"),
        ("size_bytes", -1),
        ("size_bytes", True),
        ("sha256", "invalid"),
    ],
)
def test_invalid_file_record(snapshot: DatasetSnapshot, field: str, value: object) -> None:
    manifest = json.loads(snapshot.manifest_path.read_text("utf-8"))
    manifest["files"][0][field] = value
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not verify_snapshot(snapshot.directory).valid


@pytest.mark.parametrize("raw", [b"{", b"[]", b"null", b"\xff", b'{"a":1,"a":2}'])
def test_invalid_json(snapshot: DatasetSnapshot, raw: bytes) -> None:
    snapshot.manifest_path.write_bytes(raw)
    assert not verify_snapshot(snapshot.directory).valid


def test_missing_manifest(snapshot: DatasetSnapshot) -> None:
    snapshot.manifest_path.unlink()
    assert not verify_snapshot(snapshot.directory).valid


def test_missing_snapshot(tmp_path: Path) -> None:
    result = verify_snapshot(tmp_path / "absent")
    assert not result.valid
    assert result.manifest is None
    assert result.checked_files == 0


def test_moved_snapshot_name_rejected(snapshot: DatasetSnapshot) -> None:
    moved = snapshot.directory.with_name("0" * 64)
    snapshot.directory.rename(moved)
    result = verify_snapshot(moved)
    assert not result.valid
    assert any(issue.code == "location" for issue in result.issues)


def test_manifest_size_bound(snapshot: DatasetSnapshot, monkeypatch: pytest.MonkeyPatch) -> None:
    from urbangraph.data import verification

    monkeypatch.setattr(verification, "MAX_MANIFEST_BYTES", 10)
    result = verify_snapshot(snapshot.directory)
    assert not result.valid
    assert any("limit" in issue.message for issue in result.issues)


def test_staging_never_verifies_as_published(snapshot: DatasetSnapshot) -> None:
    stage = snapshot.directory.with_name(".staging-interrupted")
    snapshot.directory.rename(stage)
    assert not verify_snapshot(stage).valid
