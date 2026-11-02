"""Versioned wire contract for an independently controlled closure broker.

This parser validates shape only. Authority requires authenticated broker IPC,
current rights, and protected registry custody before any artifact metadata read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from urbangraph.data import DataError
from urbangraph.data.contracts import validate_path_inventory, validate_relative_path


@dataclass(frozen=True)
class BrokerRecord:
    artifact_id: str
    content_sha256: str
    kind: str
    parents: tuple[tuple[str, str], ...]
    sources: tuple[tuple[str, str], ...]
    transformation: str
    revision: int
    issuer: str
    audit_id: str
    snapshot: dict[str, object] | None = None


def _digest(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise DataError("Invalid broker digest")
    return value


def _identifier(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 128
        or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_." for ch in value)
    ):
        raise DataError("Invalid broker identity")
    return value


def parse_committed_record(value: object) -> BrokerRecord:
    """Fail closed on malformed, ambiguous, or uncommitted broker records."""
    common = {
        "version",
        "state",
        "artifact_id",
        "content_sha256",
        "kind",
        "parents",
        "sources",
        "transformation",
        "revision",
        "issuer",
        "audit_id",
    }
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or (
            (value.get("version") == 1 and set(value) != common)
            or (value.get("version") == 2 and set(value) != common | {"snapshot"})
            or value.get("version") not in {1, 2}
        )
    ):
        raise DataError("Unsupported broker record schema")
    if type(value["version"]) is not int or value["version"] not in {1, 2}:
        raise DataError("Unsupported broker contract version")
    if value["state"] != "committed":
        raise DataError("Broker artifact not committed")
    if type(value["revision"]) is not int or value["revision"] < 1:
        raise DataError("Invalid broker revision")
    if not isinstance(value["parents"], list) or not isinstance(value["sources"], list):
        raise DataError("Invalid broker lineage")
    if not value["sources"] or len(value["parents"]) > 64 or len(value["sources"]) > 64:
        raise DataError("Incomplete or excessive broker lineage")

    def pairs(items: list[object], key: str) -> tuple[tuple[str, str], ...]:
        result: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict) or set(item) != {key, "content_sha256"}:
                raise DataError("Invalid broker lineage entry")
            result.append((_identifier(item[key]), _digest(item["content_sha256"])))
        if [name for name, _ in result] != sorted({name for name, _ in result}):
            raise DataError("Duplicate or unordered broker lineage")
        return tuple(result)

    snapshot: dict[str, object] | None = None
    if value["version"] == 2:
        claimed = value["snapshot"]
        if (
            not isinstance(claimed, dict)
            or set(claimed) != {"schema", "directories", "files"}
            or claimed["schema"] != "broker-tree-v1"
            or not isinstance(claimed["directories"], list)
            or not isinstance(claimed["files"], list)
            or len(claimed["directories"]) > 4096
            or not 1 <= len(claimed["files"]) <= 4096
        ):
            raise DataError("Invalid broker directory inventory")
        directories = claimed["directories"]
        files = claimed["files"]
        if any(not isinstance(path, str) for path in directories):
            raise DataError("Invalid broker directory path")
        for path in directories:
            validate_relative_path(path)
        if directories != sorted(set(directories)):
            raise DataError("Unordered broker directory inventory")
        paths: list[str] = []
        total = 0
        for item in files:
            if (
                not isinstance(item, list)
                or len(item) != 3
                or not isinstance(item[0], str)
                or type(item[1]) is not int
                or not 0 <= item[1] <= 64 * 1024 * 1024
            ):
                raise DataError("Invalid broker directory file entry")
            validate_relative_path(item[0])
            _digest(item[2])
            total += item[1]
            paths.append(item[0])
        if total > 256 * 1024 * 1024 or paths != sorted(paths):
            raise DataError("Invalid broker directory size or ordering")
        validate_path_inventory(tuple(paths))
        if set(directories) & set(paths):
            raise DataError("Broker file/directory collision")
        encoded = json.dumps(
            claimed, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        actual_digest = hashlib.sha256(b"urbangraph-broker-tree-v1\x00" + encoded).hexdigest()
        if actual_digest != value["content_sha256"]:
            raise DataError("Broker directory inventory digest mismatch")
        snapshot = claimed
    artifact_id = _identifier(value["artifact_id"])
    return BrokerRecord(
        artifact_id=artifact_id,
        content_sha256=_digest(value["content_sha256"]),
        kind=_identifier(value["kind"]),
        parents=pairs(value["parents"], "artifact_id"),
        sources=pairs(value["sources"], "source_id"),
        transformation=_identifier(value["transformation"]),
        revision=value["revision"],
        issuer=_identifier(value["issuer"]),
        audit_id=_identifier(value["audit_id"]),
        snapshot=snapshot,
    )
