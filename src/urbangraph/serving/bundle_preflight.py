"""Strict in-memory preflight for a future, synthetic-only serving bundle.

This validates bytes supplied by a trusted caller. It does not open paths, load a
model, grant data rights, approve promotion, or authorize inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SCHEMA = "ug-bundle-preflight-1"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
ROLE_FILES = {
    "model": "model.bin",
    "transform": "transform.json",
    "graph": "graph.json",
    "calibration": "calibration.json",
}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class BundlePreflightError(ValueError):
    """A proposed bundle does not meet the non-executing contract."""


@dataclass(frozen=True)
class ExpectedBundle:
    """Independent identity supplied by a trusted future promotion controller."""

    source_manifest_sha256: str
    code_commit: str
    model_id: str
    graph_id: str
    transform_id: str
    calibration_id: str
    sensor_ids: tuple[str, ...]
    horizon_hours: int
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_manifest_sha256, str) or not HEX_SHA256.fullmatch(
            self.source_manifest_sha256
        ):
            raise BundlePreflightError("Trusted source digest is invalid")
        if not isinstance(self.code_commit, str) or not HEX_COMMIT.fullmatch(self.code_commit):
            raise BundlePreflightError("Trusted code commit is invalid")
        for value in (self.model_id, self.graph_id, self.transform_id, self.calibration_id):
            if not isinstance(value, str) or not IDENTITY.fullmatch(value):
                raise BundlePreflightError("Trusted artifact identity is invalid")
        if (
            not isinstance(self.sensor_ids, tuple)
            or not self.sensor_ids
            or any(not isinstance(sensor, str) for sensor in self.sensor_ids)
            or len(set(self.sensor_ids)) != len(self.sensor_ids)
            or any(not IDENTITY.fullmatch(sensor) for sensor in self.sensor_ids)
            or type(self.horizon_hours) is not int
            or self.horizon_hours < 1
            or not isinstance(self.unit, str)
            or not self.unit
            or len(self.unit) > 32
        ):
            raise BundlePreflightError("Trusted serving contract is invalid")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundlePreflightError("Duplicate manifest field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BundlePreflightError(f"Nonfinite manifest value: {value}")


def _object(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BundlePreflightError("Manifest shape differs from contract")
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise BundlePreflightError("Invalid SHA-256 value")
    return value


def validate_bundle(
    manifest_bytes: bytes, payloads: Mapping[str, bytes], expected: ExpectedBundle
) -> str:
    """Return a content ID after structural and byte-integrity validation only.

    The source classification is pinned to synthetic. The caller must independently
    authenticate `expected` and keep payloads under trusted control.
    """
    if not isinstance(manifest_bytes, bytes) or len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise BundlePreflightError("Manifest missing or oversized")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_constant,
        )
        canonical = (
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, OverflowError) as exc:
        raise BundlePreflightError("Manifest is not valid finite UTF-8 JSON") from exc
    if manifest_bytes != canonical:
        raise BundlePreflightError("Manifest is not canonical")
    top = _object(
        manifest,
        {
            "schema_version",
            "source_class",
            "source_manifest_sha256",
            "code_commit",
            "contract",
            "files",
        },
    )
    if top["schema_version"] != SCHEMA or top["source_class"] != "synthetic":
        raise BundlePreflightError("Unsupported schema or unapproved source class")
    if (
        _sha(top["source_manifest_sha256"]) != expected.source_manifest_sha256
        or top["code_commit"] != expected.code_commit
    ):
        raise BundlePreflightError("Source or code identity differs from trusted expectation")
    contract = _object(
        top["contract"],
        {
            "model_id",
            "graph_id",
            "transform_id",
            "calibration_id",
            "sensor_ids",
            "horizon_hours",
            "unit",
        },
    )
    intended = {
        "model_id": expected.model_id,
        "graph_id": expected.graph_id,
        "transform_id": expected.transform_id,
        "calibration_id": expected.calibration_id,
        "sensor_ids": list(expected.sensor_ids),
        "horizon_hours": expected.horizon_hours,
        "unit": expected.unit,
    }
    if contract != intended or type(contract["horizon_hours"]) is not int:
        raise BundlePreflightError("Serving identities, sensor order or units are incompatible")
    files = _object(top["files"], set(ROLE_FILES))
    if set(payloads) != set(ROLE_FILES.values()):
        raise BundlePreflightError("Payload inventory differs from contract")
    for role, name in ROLE_FILES.items():
        record = _object(files[role], {"name", "size_bytes", "sha256"})
        data = payloads[name]
        if not isinstance(data, bytes) or len(data) > MAX_PAYLOAD_BYTES:
            raise BundlePreflightError("Payload missing or oversized")
        if (
            record["name"] != name
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] != len(data)
            or _sha(record["sha256"]) != hashlib.sha256(data).hexdigest()
        ):
            raise BundlePreflightError("Payload identity or digest mismatch")
    return hashlib.sha256(manifest_bytes).hexdigest()
