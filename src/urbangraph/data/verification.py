"""Read-only manifest and payload verification; never repairs snapshots."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

from urbangraph.data.contracts import DataError, DatasetManifest, manifest_from_dict
from urbangraph.data.filesystem import discover, reject_link, safe_absolute, stream_file

MAX_MANIFEST_BYTES = 16 * 1024 * 1024


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DataError("Duplicate JSON object key")
        result[key] = value
    return result


def read_manifest(path: Path) -> DatasetManifest:
    """Read bounded JSON, rejecting duplicate keys and unsupported contracts."""
    safe_absolute(path)
    # Validate type/link safety before opening, including the manifest itself.
    info = reject_link(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise DataError("Manifest must be a regular file with a single hard link")
    with path.open("rb") as handle:
        raw = handle.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise DataError("Manifest exceeds 16 MiB limit")
    try:
        value: object = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataError("Invalid manifest JSON") from exc
    return manifest_from_dict(value)


@dataclass(frozen=True)
class VerificationIssue:
    code: str
    message: str
    relative_path: str | None = None


@dataclass(frozen=True)
class VerificationResult:
    snapshot_dir: Path
    manifest: DatasetManifest | None
    issues: tuple[VerificationIssue, ...]
    checked_files: int

    @property
    def valid(self) -> bool:
        return self.manifest is not None and not self.issues


def _verify_snapshot(directory: Path, *, enforce_location: bool) -> VerificationResult:
    manifest: DatasetManifest | None = None
    issues: list[VerificationIssue] = []
    checked = 0
    try:
        directory = safe_absolute(directory)
        tree = discover(directory)
        manifest = read_manifest(directory / "manifest.json")
        if enforce_location and (
            directory.name != manifest.snapshot_id or directory.parent.name != manifest.dataset_id
        ):
            issues.append(
                VerificationIssue("location", "Snapshot directory does not match identity")
            )
        expected = {"manifest.json"} | {
            f"payload/{record.relative_path}" for record in manifest.files
        }
        expected_dirs = {"payload"}
        for path in expected:
            parts = path.split("/")
            expected_dirs.update("/".join(parts[:i]) for i in range(1, len(parts)))
        for missing in sorted(expected - set(tree.files)):
            issues.append(VerificationIssue("missing", "Expected file is missing", missing))
        for extra in sorted(set(tree.files) - expected):
            issues.append(VerificationIssue("unexpected", "Unexpected file", extra))
        for extra in sorted(set(tree.directories) - expected_dirs):
            issues.append(VerificationIssue("unexpected_directory", "Unexpected directory", extra))
        for record in manifest.files:
            relative = f"payload/{record.relative_path}"
            if relative not in tree.files:
                continue
            size, digest = stream_file(directory / relative)
            checked += 1
            if size != record.size_bytes:
                issues.append(VerificationIssue("size", "File size mismatch", record.relative_path))
            if digest != record.sha256:
                issues.append(VerificationIssue("digest", "SHA-256 mismatch", record.relative_path))
        if discover(directory) != tree:
            issues.append(VerificationIssue("changed", "Snapshot tree changed during verification"))
        if read_manifest(directory / "manifest.json") != manifest:
            issues.append(VerificationIssue("changed", "Manifest changed during verification"))
    except DataError as exc:
        issues.append(VerificationIssue("invalid", str(exc)))
    except OSError as exc:
        issues.append(
            VerificationIssue("filesystem", f"Filesystem operation failed: {type(exc).__name__}")
        )
    return VerificationResult(directory, manifest, tuple(issues), checked)


def verify_snapshot(snapshot_dir: str | Path) -> VerificationResult:
    """Verify a <dataset>/<content-id> directory, including unexpected entries."""
    return _verify_snapshot(Path(snapshot_dir), enforce_location=True)
