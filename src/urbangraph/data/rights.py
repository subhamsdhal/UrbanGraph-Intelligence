"""Default-deny local rights decisions for external raw-source reads.

Deployment must provide an administrator-owned configuration outside the caller's
write authority. Application code cannot establish that OS boundary by itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urbangraph.data.contracts import DataError
from urbangraph.data.filesystem import safe_absolute

_CONFIG_VARIABLE = "URBANGRAPH_RIGHTS_POLICY"
_REQUIRED = {
    "source_id",
    "source_sha256",
    "source_path",
    "purpose",
    "operation",
    "destination_root",
    "evidence_reference",
    "status",
    "effective_utc",
    "expires_utc",
    "retention_days",
}


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataError("Conflicting rights policy")
        result[key] = value
    return result


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise DataError("Invalid rights policy time")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataError("Invalid rights policy time") from exc
    if parsed.tzinfo is None:
        raise DataError("Rights policy time lacks timezone")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class AuthorizationRequest:
    source_id: str
    source_path: str | Path
    purpose: str
    destination: str | Path
    operation: str


@dataclass(frozen=True)
class AuthorizationDecision:
    """Policy-authored source decision, distinct from verified source bytes."""

    source_id: str
    source_sha256: str
    source_path: Path
    purpose: str
    operation: str
    destination_root: Path
    evidence_reference: str
    effective_utc: datetime
    expires_utc: datetime
    retention_days: int


def require_authorization_decisions(
    requests: tuple[AuthorizationRequest, ...],
) -> tuple[AuthorizationDecision, ...]:
    """Decide a complete upstream set from one bounded policy-file read.

    OS control of the configured policy file/directory is still required. This
    prevents mixing grants from separate observed policy revisions, not a
    hostile same-user path-replacement race after the read.
    """
    if not requests:
        raise DataError("Empty upstream authorization set")
    if len({(r.source_id, r.operation, str(r.source_path)) for r in requests}) != len(requests):
        raise DataError("Duplicate upstream authorization request")
    policy = _load_policy()
    grants = policy["grants"]
    if not isinstance(grants, list):
        raise DataError("Trusted rights policy grants invalid")
    indexed: dict[tuple[str, str, str, Path, Path], dict[str, Any]] = {}
    for item in grants:
        if not isinstance(item, dict) or set(item) != _REQUIRED:
            raise DataError("Trusted rights grant schema invalid")
        if any(not isinstance(item[k], str) or not item[k] for k in _REQUIRED - {"retention_days"}):
            raise DataError("Trusted rights grant field invalid")
        if type(item["retention_days"]) is not int or item["retention_days"] < 1:
            raise DataError("Trusted rights retention invalid")
        if not Path(item["source_path"]).is_absolute():
            raise DataError("Rights grant paths must be absolute")
        identity = (
            item["source_id"],
            item["purpose"],
            item["operation"],
            safe_absolute(item["source_path"]),
            safe_absolute(item["destination_root"]),
        )
        if identity in indexed:
            raise DataError("Conflicting rights grants")
        indexed[identity] = item
    now = datetime.now(UTC)
    results: list[AuthorizationDecision] = []
    for request in requests:
        source_key = (
            request.source_id,
            request.purpose,
            request.operation,
            safe_absolute(request.source_path),
        )
        candidates = [
            item
            for key, item in indexed.items()
            if key[:4] == source_key and safe_absolute(request.destination).is_relative_to(key[4])
        ]
        if len(candidates) != 1:
            if candidates:
                raise DataError("Ambiguous destination rights grants")
            raise DataError("No matching rights grant")
        grant = candidates[0]
        if grant["status"] != "approved" or not grant["evidence_reference"].strip():
            raise DataError("Rights grant not approved")
        if not (_time(grant["effective_utc"]) <= now < _time(grant["expires_utc"])):
            raise DataError("Rights grant outside validity window")
        if (
            not Path(grant["source_path"]).is_absolute()
            or not Path(grant["destination_root"]).is_absolute()
        ):
            raise DataError("Rights grant paths must be absolute")
        expected_source = safe_absolute(grant["source_path"])
        expected_root = safe_absolute(grant["destination_root"])
        if safe_absolute(request.source_path) != expected_source or not safe_absolute(
            request.destination
        ).is_relative_to(expected_root):
            raise DataError("Rights grant source or destination mismatch")
        digest = grant["source_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise DataError("Rights grant content identity invalid")
        results.append(
            AuthorizationDecision(
                source_id=str(grant["source_id"]),
                source_sha256=digest,
                source_path=expected_source,
                purpose=str(grant["purpose"]),
                operation=str(grant["operation"]),
                destination_root=expected_root,
                evidence_reference=str(grant["evidence_reference"]),
                effective_utc=_time(grant["effective_utc"]),
                expires_utc=_time(grant["expires_utc"]),
                retention_days=int(grant["retention_days"]),
            )
        )
    return tuple(results)


def require_authorizations(
    requests: tuple[AuthorizationRequest, ...],
) -> tuple[str, ...]:
    """Compatibility digest view of the same strict, single-policy decision set."""
    return tuple(decision.source_sha256 for decision in require_authorization_decisions(requests))


def _load_policy() -> dict[str, Any]:
    location = os.environ.get(_CONFIG_VARIABLE)
    if not location:
        raise DataError("Trusted rights policy unavailable")
    if not Path(location).is_absolute():
        raise DataError("Trusted rights policy path must be absolute")
    try:
        policy_path = safe_absolute(location)
        if not policy_path.is_absolute() or not policy_path.is_file():
            raise DataError("Trusted rights policy unavailable")
        checkout = Path(__file__).resolve().parents[3]
        if policy_path.is_relative_to(checkout):
            raise DataError("Trusted rights policy must be outside checkout")
        if policy_path.stat().st_size > 65536:
            raise DataError("Trusted rights policy exceeds size limit")
        raw = policy_path.read_bytes()
        if len(raw) > 65536:
            raise DataError("Trusted rights policy exceeds size limit")
        policy = json.loads(raw, object_pairs_hook=_unique_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise DataError("Trusted rights policy invalid or unavailable") from exc
    if (
        not isinstance(policy, dict)
        or set(policy) != {"version", "grants"}
        or type(policy["version"]) is not int
        or policy["version"] != 1
    ):
        raise DataError("Trusted rights policy schema invalid")
    return policy


def require_raw_authorization(
    source_id: str,
    source_path: str | Path,
    purpose: str,
    destination: str | Path,
    *,
    operation: str = "raw_ingest",
) -> str:
    """Authorize source/path/purpose/destination before inspecting source bytes.

    The process environment is deployment configuration, never a CLI/API argument.
    Secure deployment must protect that environment and the configured file from
    ingestion callers. In the current same-user Windows setup this is unverified.
    """
    return require_authorizations(
        (AuthorizationRequest(source_id, source_path, purpose, destination, operation),)
    )[0]


def require_snapshot_authorization(
    snapshot: str | Path,
    purpose: str,
    destination: str | Path,
    *,
    operation: str,
) -> str:
    """Pre-authorize an exact immutable raw snapshot path before verification/parse.

    The path's parent/name are selectors, never proof. The returned pinned content
    identity must be compared with the independently verified manifest afterward.
    """
    path = Path(snapshot)
    return require_raw_authorization(
        path.parent.name, path, purpose, destination, operation=operation
    )


def require_verified_snapshot_authorization(
    snapshot: str | Path,
    purpose: str,
    destination: str | Path,
    *,
    operation: str,
) -> str:
    """Authorize first, then compare the grant to independently verified raw bytes."""
    from urbangraph.data import verify_snapshot

    expected = require_snapshot_authorization(snapshot, purpose, destination, operation=operation)
    verified = verify_snapshot(snapshot)
    if not verified.valid or verified.manifest is None or verified.manifest.snapshot_id != expected:
        raise DataError("Raw snapshot differs from current purpose-specific rights grant")
    return expected
