"""Synthetic-only, non-executing bundle contract checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from energraph.demo.fixtures import BUILTIN_MANIFEST_SHA256
from urbangraph.serving.bundle_preflight import (
    ROLE_FILES,
    SCHEMA,
    BundlePreflightError,
    ExpectedBundle,
    validate_bundle,
)

CODE_COMMIT = "f7a6119115e0afc82efc35c42f66533871d30ced"


def _case() -> tuple[dict[str, object], dict[str, bytes], ExpectedBundle]:
    expected = ExpectedBundle(
        source_manifest_sha256=BUILTIN_MANIFEST_SHA256,
        code_commit=CODE_COMMIT,
        model_id="synthetic-contract-model",
        graph_id="schematic-graph",
        transform_id="identity-transform",
        calibration_id="uncalibrated-declaration",
        sensor_ids=("SYN_A", "SYN_B", "SYN_C"),
        horizon_hours=24,
        unit="MW",
    )
    payloads = {
        name: f"invented test bytes for {role}".encode() for role, name in ROLE_FILES.items()
    }
    manifest: dict[str, object] = {
        "schema_version": SCHEMA,
        "source_class": "synthetic",
        "source_manifest_sha256": expected.source_manifest_sha256,
        "code_commit": expected.code_commit,
        "contract": {
            "model_id": expected.model_id,
            "graph_id": expected.graph_id,
            "transform_id": expected.transform_id,
            "calibration_id": expected.calibration_id,
            "sensor_ids": list(expected.sensor_ids),
            "horizon_hours": expected.horizon_hours,
            "unit": expected.unit,
        },
        "files": {
            role: {
                "name": name,
                "size_bytes": len(payloads[name]),
                "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            }
            for role, name in ROLE_FILES.items()
        },
    }
    return manifest, payloads, expected


def _encode(manifest: dict[str, object]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_valid_synthetic_contract_is_deterministic() -> None:
    manifest, payloads, expected = _case()
    identity = validate_bundle(_encode(manifest), payloads, expected)
    assert identity == hashlib.sha256(_encode(manifest)).hexdigest()
    assert validate_bundle(_encode(manifest), payloads, expected) == identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "future"),
        ("source_class", "rights_cleared"),
        ("source_class", "restricted_or_unknown"),
        ("source_manifest_sha256", "0" * 64),
        ("code_commit", "0" * 40),
    ],
)
def test_reject_wrong_source_or_schema(field: str, value: str) -> None:
    manifest, payloads, expected = _case()
    manifest[field] = value
    with pytest.raises(BundlePreflightError):
        validate_bundle(_encode(manifest), payloads, expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("sensor_ids", ["SYN_B", "SYN_A", "SYN_C"]),
        ("graph_id", "other"),
        ("transform_id", "other"),
        ("calibration_id", "other"),
        ("horizon_hours", 25),
        ("horizon_hours", True),
        ("unit", "kW"),
    ],
)
def test_reject_incompatible_serving_contract(field: str, value: object) -> None:
    manifest, payloads, expected = _case()
    contract = manifest["contract"]
    assert isinstance(contract, dict)
    contract[field] = value
    with pytest.raises(BundlePreflightError):
        validate_bundle(_encode(manifest), payloads, expected)


def test_reject_forged_trusted_expectation() -> None:
    manifest, payloads, expected = _case()
    with pytest.raises(BundlePreflightError):
        validate_bundle(_encode(manifest), payloads, replace(expected, graph_id="other"))


@pytest.mark.parametrize("defect", ["tamper", "missing", "extra", "size", "digest", "path"])
def test_reject_payload_inventory_or_integrity(defect: str) -> None:
    manifest, payloads, expected = _case()
    files = manifest["files"]
    assert isinstance(files, dict)
    model = files["model"]
    assert isinstance(model, dict)
    if defect == "tamper":
        payloads["model.bin"] += b"x"
    elif defect == "missing":
        del payloads["model.bin"]
    elif defect == "extra":
        payloads["secret.txt"] = b"x"
    elif defect == "size":
        model["size_bytes"] = True
    elif defect == "digest":
        model["sha256"] = "0" * 64
    else:
        model["name"] = "../model.bin"
    with pytest.raises(BundlePreflightError):
        validate_bundle(_encode(manifest), payloads, expected)


@pytest.mark.parametrize("defect", ["extra", "duplicate", "nonfinite", "noncanonical", "utf8"])
def test_reject_manifest_shape_or_encoding(defect: str) -> None:
    manifest, payloads, expected = _case()
    data = _encode(manifest)
    if defect == "extra":
        manifest["unexpected"] = "x"
        data = _encode(manifest)
    elif defect == "duplicate":
        data = b'{"schema_version":"a","schema_version":"b"}\n'
    elif defect == "nonfinite":
        data = b'{"x":NaN}\n'
    elif defect == "noncanonical":
        data = json.dumps(manifest, indent=2).encode()
    else:
        data = b"\xff"
    with pytest.raises(BundlePreflightError):
        validate_bundle(data, payloads, expected)


def test_reject_invalid_independent_identity() -> None:
    _, _, expected = _case()
    with pytest.raises(BundlePreflightError):
        replace(expected, source_manifest_sha256="unknown")
    with pytest.raises(BundlePreflightError):
        replace(expected, sensor_ids=("SYN_A", "SYN_A"))
