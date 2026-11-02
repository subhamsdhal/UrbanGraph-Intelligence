import hashlib
import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from urbangraph.data import DataError, filesystem, ingest_dataset, verify_snapshot
from urbangraph.data.contracts import IDENTITY_VERSION


@pytest.mark.parametrize("link_kind", ["symlink", "reparse"])
def test_link_detection_without_os_privileges(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
    invented_source_grant,
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    target = source_dir / "metr-la.h5"
    original = Path.lstat

    def lstat(path: Path) -> os.stat_result:
        result = original(path)
        if path == target:
            return cast(
                os.stat_result,
                SimpleNamespace(
                    st_mode=stat.S_IFLNK if link_kind == "symlink" else result.st_mode,
                    st_file_attributes=0x400 if link_kind == "reparse" else 0,
                ),
            )
        return result

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(DataError, match="Symlinks"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "location", ["source-file", "source-parent", "destination-parent", "snapshot"]
)
def test_real_symlinks(
    source_dir: Path, tmp_path: Path, location: str, invented_source_grant
) -> None:
    raw = tmp_path / "raw"
    invented_source_grant(source_dir, "metr-la", raw)
    target = tmp_path / "external"
    target.mkdir()
    (target / "a.bin").write_bytes(b"external")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Host cannot create symlinks: {type(exc).__name__}")
    if location == "source-file":
        (source_dir / "alias").symlink_to(target / "a.bin")
    elif location == "source-parent":
        source_dir = link
    elif location == "destination-parent":
        raw = link / "raw"
    else:
        snapshot = ingest_dataset(source_dir, "metr-la", raw_root=raw)
        (snapshot.directory / "payload" / "alias").symlink_to(target / "a.bin")
        assert not verify_snapshot(snapshot.directory).valid
        return
    with pytest.raises(DataError, match="Symlinks"):
        ingest_dataset(source_dir, "metr-la", raw_root=raw)
    assert (target / "a.bin").read_bytes() == b"external"
    assert not (target / "raw").exists()


def test_parent_traversal_arguments_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="Parent traversal"):
        filesystem.safe_absolute(tmp_path / "folder" / ".." / "outside")


def test_special_file_rejected(
    source_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invented_source_grant,
) -> None:
    invented_source_grant(source_dir, "metr-la", tmp_path / "raw")
    original = filesystem.reject_link

    def special(path: Path) -> os.stat_result:
        info = original(path)
        if path.name == "metr-la.h5":
            return os.stat_result((stat.S_IFIFO, *tuple(info)[1:]))
        return info

    monkeypatch.setattr(filesystem, "reject_link", special)
    with pytest.raises(DataError, match="regular files"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")


def test_unsafe_actual_name_rejected(source_dir: Path, tmp_path: Path, monkeypatch) -> None:
    # '?' is uncreatable on Windows, so use a decomposed Unicode path on all hosts.
    (source_dir / "e\u0301.bin").write_bytes(b"noncanonical")
    # Bind this negative test's authority to the exact invented tree, including
    # the deliberately invalid path spelling. No production parser is bypassed.
    files = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            content = path.read_bytes()
            files.append(
                {
                    "relative_path": path.relative_to(source_dir).as_posix(),
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
    identity = hashlib.sha256(
        json.dumps(
            {"identity_version": IDENTITY_VERSION, "files": files},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    now = datetime.now(UTC)
    policy = {
        "version": 1,
        "grants": [
            {
                "source_id": "metr-la",
                "source_sha256": identity,
                "source_path": str(source_dir),
                "purpose": "local_research",
                "operation": "raw_ingest",
                "destination_root": str(tmp_path / "raw"),
                "evidence_reference": "invented-test-fixture-only-invalid-path",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        ],
    }
    policy_path = tmp_path / "invented-rights-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy_path))
    with pytest.raises(DataError, match="NFC"):
        ingest_dataset(source_dir, "metr-la", raw_root=tmp_path / "raw")


def test_lstat_fstat_ctime_semantics_differ(
    source_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.fstat

    def fstat(descriptor: int) -> os.stat_result:
        info = original(descriptor)
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns,
                st_ctime_ns=1,
            ),
        )

    monkeypatch.setattr(os, "fstat", fstat)
    assert filesystem.stream_file(source_dir / "metr-la.h5")[0] > 0


def test_handle_change_detected(source_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.fstat
    calls = 0

    def fstat(descriptor: int) -> os.stat_result:
        nonlocal calls
        calls += 1
        info = original(descriptor)
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_dev=info.st_dev,
                st_ino=info.st_ino,
                st_size=info.st_size,
                st_mtime_ns=info.st_mtime_ns,
                st_ctime_ns=calls,
            ),
        )

    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(DataError, match="changed while reading"):
        filesystem.stream_file(source_dir / "metr-la.h5")
