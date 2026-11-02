"""Validated, versioned raw-data contracts. No dataset format assumptions."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from urbangraph.reproducibility import EnvironmentInfo, GitStatus

SCHEMA_VERSION = "1.0"
MANIFEST_VERSION = 1
IDENTITY_VERSION = "sha256-path-size-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_RESERVED = {"con", "prn", "aux", "nul", "conin$", "conout$"} | {
    f"{prefix}{suffix}" for prefix in ("com", "lpt") for suffix in "123456789¹²³"
}


class DataError(ValueError):
    """Invalid input, unsafe structure, invalid manifest or snapshot conflict."""


def validate_relative_path(value: str) -> None:
    """Require an unambiguous NFC POSIX path portable to Windows."""
    if not isinstance(value, str) or not value or unicodedata.normalize("NFC", value) != value:
        raise DataError("Relative paths must be nonempty NFC strings")
    for component in value.split("/"):
        if (
            not component
            or component in {".", ".."}
            or component.endswith((".", " "))
            or component.split(".")[0].rstrip(" ").casefold() in _RESERVED
            or any(ord(c) < 32 or ord(c) == 127 or c in '<>:"\\|?*' for c in component)
        ):
            raise DataError("Unsafe or nonportable relative path")
        try:
            if len(component.encode("utf-8")) > 255:
                raise DataError("Path component exceeds 255 UTF-8 bytes")
        except UnicodeError as exc:
            raise DataError("Invalid Unicode path") from exc


def validate_path_inventory(paths: tuple[str, ...]) -> None:
    """Reject duplicate paths, case aliases and file/directory collisions."""
    seen: set[str] = set()
    components: dict[str, str] = {}
    for path in paths:
        validate_relative_path(path)
        if path in seen:
            raise DataError("Duplicate relative path")
        seen.add(path)
        parts = path.split("/")
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            key = prefix.casefold()
            if key in components and components[key] != prefix:
                raise DataError("Case-normalized path collision")
            components[key] = prefix
    for path in seen:
        parts = path.split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            raise DataError("File/directory path collision")


@dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dataset_id, str)
            or len(self.dataset_id) > 64
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", self.dataset_id) is None
            or self.dataset_id in _RESERVED
        ):
            raise DataError(
                "Dataset identifier must be a portable lowercase slug (1-64 characters)"
            )


METR_LA = DatasetIdentity("metr-la")
PEMS_BAY = DatasetIdentity("pems-bay")


@dataclass(frozen=True)
class DatasetSource:
    reference: str | None = None
    description: str | None = None
    attribution: str | None = None
    licensing_notes: str | None = None

    def __post_init__(self) -> None:
        for value in (self.reference, self.description, self.attribution, self.licensing_notes):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataError("Provided source metadata must be nonblank text or null")


@dataclass(frozen=True)
class RawFileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_relative_path(self.relative_path)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise DataError("File size must be a nonnegative integer")
        if not isinstance(self.sha256, str) or _DIGEST.fullmatch(self.sha256) is None:
            raise DataError("Expected a lowercase SHA-256 digest")


def content_identity(files: tuple[RawFileRecord, ...]) -> str:
    """Hash a domain-separated canonical inventory; exclude descriptive metadata."""
    if not files:
        raise DataError("An inventory must contain at least one file")
    validate_path_inventory(tuple(record.relative_path for record in files))
    canonical = {
        "identity_version": IDENTITY_VERSION,
        "files": [asdict(record) for record in sorted(files, key=lambda item: item.relative_path)],
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    snapshot_id: str
    created_at: str
    source: DatasetSource
    files: tuple[RawFileRecord, ...]
    creator_environment: EnvironmentInfo
    schema_version: str = SCHEMA_VERSION
    manifest_version: int = MANIFEST_VERSION
    identity_version: str = IDENTITY_VERSION

    def __post_init__(self) -> None:
        DatasetIdentity(self.dataset_id)
        if self.schema_version != SCHEMA_VERSION:
            raise DataError("Unsupported manifest schema version")
        if type(self.manifest_version) is not int or self.manifest_version != MANIFEST_VERSION:
            raise DataError("Unsupported manifest version")
        if self.identity_version != IDENTITY_VERSION:
            raise DataError("Unsupported content identity version")
        if not isinstance(self.created_at, str):
            raise DataError("Creation timestamp must be a UTC ISO-8601 string")
        try:
            timestamp = datetime.fromisoformat(self.created_at)
        except ValueError as exc:
            raise DataError("Invalid creation timestamp") from exc
        offset = timestamp.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise DataError("Creation timestamp must include UTC timezone")
        if not isinstance(self.source, DatasetSource) or not isinstance(
            self.creator_environment, EnvironmentInfo
        ):
            raise DataError("Invalid provenance object")
        if not isinstance(self.files, tuple) or not all(
            isinstance(record, RawFileRecord) for record in self.files
        ):
            raise DataError("Inventory must be a tuple of file records")
        if self.files != tuple(sorted(self.files, key=lambda item: item.relative_path)):
            raise DataError("Manifest inventory must be sorted by relative path")
        if self.snapshot_id != content_identity(self.files):
            raise DataError("Snapshot identity does not match manifest inventory")

    @property
    def total_bytes(self) -> int:
        return sum(record.size_bytes for record in self.files)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = asdict(self)
        result["files"] = [asdict(record) for record in self.files]
        return result


@dataclass(frozen=True)
class BrokerArtifactReference:
    """Untrusted lookup handle resolved against the broker before protected use."""

    artifact_id: str
    content_sha256: str
    revision: int
    artifact_type: str


@dataclass(frozen=True)
class DatasetSnapshot:
    directory: Path
    manifest: DatasetManifest
    reused: bool = False
    broker_reference: BrokerArtifactReference | None = None

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"


def _object(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DataError("Manifest object has missing or unknown fields")
    return {str(key): item for key, item in value.items()}


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise DataError("Manifest field must be text")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _integer(value: object) -> int:
    if type(value) is not int:
        raise DataError("Manifest field must be an integer")
    return value


def _environment(value: object) -> EnvironmentInfo:
    obj = _object(
        value,
        {
            "python_version",
            "platform",
            "project_version",
            "git_available",
            "git_status",
            "git_commit",
        },
    )
    status = _text(obj["git_status"])
    git_status: GitStatus
    if status == "clean":
        git_status = "clean"
    elif status == "dirty":
        git_status = "dirty"
    elif status == "unavailable":
        git_status = "unavailable"
    elif status == "not-a-repository":
        git_status = "not-a-repository"
    elif status == "error":
        git_status = "error"
    else:
        raise DataError("Invalid Git status")
    available = obj["git_available"]
    if not isinstance(available, bool):
        raise DataError("Git availability must be boolean")
    commit = _optional_text(obj["git_commit"])
    if commit is not None and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None:
        raise DataError("Invalid Git commit")
    return EnvironmentInfo(
        python_version=_text(obj["python_version"]),
        platform=_text(obj["platform"]),
        project_version=_text(obj["project_version"]),
        git_available=available,
        git_status=git_status,
        git_commit=commit,
    )


def manifest_from_dict(value: object) -> DatasetManifest:
    """Validate all fields before accepting a deserialized manifest."""
    obj = _object(
        value,
        {
            "schema_version",
            "manifest_version",
            "identity_version",
            "dataset_id",
            "snapshot_id",
            "created_at",
            "source",
            "files",
            "creator_environment",
        },
    )
    source = _object(obj["source"], {"reference", "description", "attribution", "licensing_notes"})
    inventory = obj["files"]
    if not isinstance(inventory, list):
        raise DataError("Manifest inventory must be a JSON array")
    records: list[RawFileRecord] = []
    for value in inventory:
        record = _object(value, {"relative_path", "size_bytes", "sha256"})
        records.append(
            RawFileRecord(
                relative_path=_text(record["relative_path"]),
                size_bytes=_integer(record["size_bytes"]),
                sha256=_text(record["sha256"]),
            )
        )
    return DatasetManifest(
        dataset_id=_text(obj["dataset_id"]),
        snapshot_id=_text(obj["snapshot_id"]),
        created_at=_text(obj["created_at"]),
        source=DatasetSource(**{key: _optional_text(item) for key, item in source.items()}),
        files=tuple(records),
        creator_environment=_environment(obj["creator_environment"]),
        schema_version=_text(obj["schema_version"]),
        manifest_version=_integer(obj["manifest_version"]),
        identity_version=_text(obj["identity_version"]),
    )
