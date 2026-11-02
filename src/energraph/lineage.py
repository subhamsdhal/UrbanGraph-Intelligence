"""Versioned, default-deny lineage checks for local derived artifacts.

Saved paths and metadata are selectors, never permission decisions. A deployment
must protect its policy and artifact ancestors against same-user modification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from urbangraph.data import DataError, RawFileRecord, content_identity, verify_snapshot
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.data.rights import (
    AuthorizationDecision,
    AuthorizationRequest,
    require_authorization_decisions,
)
from urbangraph.datasets.serialization import canonical_json, read_json


@dataclass(frozen=True)
class FileParent:
    path: Path
    source_id: str


@dataclass(frozen=True)
class ParentGraph:
    direct: tuple[dict[str, object], ...]
    sources: tuple[dict[str, object], ...]


def _key(item: dict[str, object]) -> tuple[str, str]:
    return str(item["type"]), str(item["path"])


def _source_groups(groups: list[tuple[dict[str, object], ...]]) -> tuple[dict[str, object], ...]:
    found: dict[tuple[str, str], dict[str, object]] = {}
    for group in groups:
        for source in group:
            key = _key(source)
            if key in found and found[key] != source:
                raise DataError("Conflicting upstream source identities")
            found[key] = source
    return tuple(found[key] for key in sorted(found))


def _raw(path: Path) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    checked = verify_snapshot(path)
    if not checked.valid or checked.manifest is None:
        raise DataError("Raw parent failed integrity verification")
    if checked.manifest.dataset_id.startswith("energraph-"):
        raise DataError("Derived or legacy artifact cannot impersonate a raw source")
    source: dict[str, object] = {
        "type": "raw",
        "source_id": checked.manifest.dataset_id,
        "path": str(path),
        "content_sha256": checked.manifest.snapshot_id,
        "manifest_sha256": stream_file(path / "manifest.json")[1],
    }
    return source, (source,)


def _file(path: Path, source_id: str) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if not source_id.strip():
        raise DataError("File parent requires a source identity")
    size, digest = stream_file(path)
    source: dict[str, object] = {
        "type": "file",
        "source_id": source_id,
        "path": str(path),
        "content_sha256": digest,
        "size_bytes": size,
    }
    return source, (source,)


def _derived(
    path: Path, ancestors: frozenset[Path]
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    from energraph.artifacts import _GOVERNED

    if path in ancestors or len(ancestors) >= 16:
        raise DataError("Cyclic or excessively deep derived lineage")
    checked = verify_snapshot(path)
    if not checked.valid or checked.manifest is None:
        raise DataError("Derived parent failed integrity verification")
    record = read_json(path / "payload" / "governance.json")
    required = {
        "schema",
        "version",
        "artifact_type",
        "direct_parents",
        "source_closure",
        "transformation",
        "purpose",
        "commit_operation",
        "destination_root",
        "output_payload_sha256",
        "publication_state",
        "policy_evidence",
        "rights_expires_utc",
        "retention_days",
    }
    if (
        set(record) != required
        or record["schema"] != "energraph-governed-derived"
        or type(record["version"]) is not int
        or record["version"] != 2
    ):
        raise DataError("Unsupported or incomplete derived lineage schema")
    if (
        record["destination_root"] != str(path.parent.parent)
        or checked.manifest.dataset_id != "energraph-" + str(record["artifact_type"])
        or record["publication_state"] != "prepared-for-atomic-rename"
        or record["commit_operation"] != "derived_commit"
    ):
        raise DataError("Derived identity or publication state mismatch")
    direct = record["direct_parents"]
    if not isinstance(direct, list) or not direct:
        raise DataError("Missing derived parent set")
    graph = _resolve_claims(direct, ancestors | {path})
    if list(graph.direct) != direct or list(graph.sources) != record["source_closure"]:
        raise DataError("Forged, missing or inconsistent derived lineage")
    transform = record["transformation"]
    if not isinstance(transform, dict) or set(transform) != {
        "name",
        "version",
        "source_path",
        "source_sha256",
        "configuration_sha256",
    }:
        raise DataError("Invalid transformation identity")
    module = safe_absolute(str(transform["source_path"]))
    kind = record["artifact_type"]
    if not isinstance(kind, str) or kind not in _GOVERNED:
        raise DataError("Unknown derived transformation kind")
    operation, _, source_module, purpose, _ = _GOVERNED[kind]
    if (
        module != safe_absolute(Path(__file__).parent / source_module)
        or transform["name"] != operation
        or record["purpose"] != purpose
        or type(transform["version"]) is not int
        or transform["version"] != 1
        or stream_file(module)[1] != transform["source_sha256"]
    ):
        raise DataError("Unsupported or changed transformation implementation")
    descriptor = read_json(path / "payload" / "descriptor.json")
    if hashlib.sha256(canonical_json(descriptor)).hexdigest() != transform["configuration_sha256"]:
        raise DataError("Derived transformation configuration changed")
    files = tuple(
        RawFileRecord(item.relative_path, item.size_bytes, item.sha256)
        for item in checked.manifest.files
        if item.relative_path != "governance.json"
    )
    if not files or record["output_payload_sha256"] != content_identity(files):
        raise DataError("Derived output content identity mismatch")
    if type(record["retention_days"]) is not int or record["retention_days"] < 1:
        raise DataError("Invalid derived retention")
    try:
        expires = datetime.fromisoformat(str(record["rights_expires_utc"]))
        created = datetime.fromisoformat(checked.manifest.created_at)
    except ValueError as exc:
        raise DataError("Invalid derived validity timestamps") from exc
    if (
        expires.tzinfo is None
        or created.tzinfo is None
        or datetime.now(UTC)
        >= min(
            expires.astimezone(UTC),
            created.astimezone(UTC) + timedelta(days=record["retention_days"]),
        )
    ):
        raise DataError("Derived artifact outside validity or retention")
    evidence = record["policy_evidence"]
    if (
        not isinstance(evidence, list)
        or len(evidence) != 2 * len(graph.sources)
        or any(not isinstance(item, str) or not item for item in evidence)
    ):
        raise DataError("Incomplete historical policy references")
    parent: dict[str, object] = {
        "type": "derived",
        "path": str(path),
        "content_sha256": checked.manifest.snapshot_id,
        "manifest_sha256": stream_file(path / "manifest.json")[1],
        "governance_sha256": stream_file(path / "payload" / "governance.json")[1],
    }
    return parent, graph.sources


def _resolve_claims(claims: list[object], ancestors: frozenset[Path]) -> ParentGraph:
    direct: list[dict[str, object]] = []
    groups: list[tuple[dict[str, object], ...]] = []
    for claimed in claims:
        if not isinstance(claimed, dict) or "type" not in claimed or "path" not in claimed:
            raise DataError("Invalid derived parent reference")
        path = safe_absolute(str(claimed["path"]))
        if claimed["type"] == "raw":
            if set(claimed) != {"type", "source_id", "path", "content_sha256", "manifest_sha256"}:
                raise DataError("Invalid raw parent schema")
            actual, closure = _raw(path)
        elif claimed["type"] == "file":
            if set(claimed) != {"type", "source_id", "path", "content_sha256", "size_bytes"}:
                raise DataError("Invalid file parent schema")
            actual, closure = _file(path, str(claimed["source_id"]))
        elif claimed["type"] == "derived":
            # This recursively verifies descriptive lineage only. Protected
            # reads and publication require current rights or broker
            # authorization before invoking this parser.
            actual, closure = _derived(path, ancestors)
        else:
            raise DataError("Unsupported derived parent type")
        if actual != claimed:
            raise DataError("Derived parent identity changed")
        direct.append(actual)
        groups.append(closure)
    keys = [_key(item) for item in direct]
    if keys != sorted(set(keys)):
        raise DataError("Duplicate or unordered derived parents")
    return ParentGraph(tuple(direct), _source_groups(groups))


def resolve_parents(parents: tuple[Path | FileParent, ...]) -> ParentGraph:
    """Recompute direct identities and transitive sources from verified objects."""
    claims: list[dict[str, object]] = []
    for parent in parents:
        if isinstance(parent, FileParent):
            path = safe_absolute(parent.path)
            if any(component.startswith("energraph-") for component in path.parts):
                raise DataError("Derived parent closure authority unavailable")
            claim, _ = _file(path, parent.source_id)
        else:
            path = safe_absolute(parent)
            if any(component.startswith("energraph-") for component in path.parts):
                raise DataError("Derived parent closure authority unavailable")
            claim, _ = _raw(path)
        claims.append(claim)
    claims.sort(key=_key)
    return _resolve_claims(list(claims), frozenset())


def resolve_derived(directory: Path) -> ParentGraph:
    """Validate the complete saved v2 graph and return its source closure."""
    _, sources = _derived(safe_absolute(directory), frozenset())
    return ParentGraph((), sources)


def authorize_sources(
    sources: tuple[dict[str, object], ...], *, purpose: str, operation: str, destination: Path
) -> tuple[AuthorizationDecision, ...]:
    """Authorize all independently verified upstream sources in one policy read."""
    decisions = require_authorization_decisions(
        tuple(
            AuthorizationRequest(
                str(item["source_id"]), str(item["path"]), purpose, destination, operation
            )
            for item in sources
        )
    )
    for source, decision in zip(sources, decisions, strict=True):
        if source["content_sha256"] != decision.source_sha256:
            raise DataError("Upstream content differs from current rights decision")
    return decisions
