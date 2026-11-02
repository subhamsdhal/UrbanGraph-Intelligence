"""Request/response compatibility without loading a model or external data."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from urbangraph.inference_contract import (
    SCHEMA,
    InferenceContractError,
    TrustedContract,
    parse_request,
    validate_response,
)


def _case() -> tuple[TrustedContract, dict[str, object], dict[str, object]]:
    trusted = TrustedContract("a" * 64, "invented-model", ("SYN_A", "SYN_B"), 2, 2, "MW")
    request: dict[str, object] = {
        "schema_version": SCHEMA,
        "bundle_sha256": trusted.bundle_sha256,
        "request_id": "example-1",
        "origin_utc": "2026-01-01T01:00:00Z",
        "sensor_ids": ["SYN_A", "SYN_B"],
        "unit": "MW",
        "history": [
            {"timestamp_utc": "2026-01-01T00:00:00Z", "values": [10.0, 20.0]},
            {"timestamp_utc": "2026-01-01T01:00:00Z", "values": [11.0, 21.0]},
        ],
    }
    response: dict[str, object] = {
        "schema_version": SCHEMA,
        "bundle_sha256": trusted.bundle_sha256,
        "model_id": trusted.model_id,
        "request_id": "example-1",
        "origin_utc": "2026-01-01T01:00:00Z",
        "sensor_ids": ["SYN_A", "SYN_B"],
        "unit": "MW",
        "predictions": [
            {"target_utc": "2026-01-01T02:00:00Z", "values": [12.0, 22.0]},
            {"target_utc": "2026-01-01T03:00:00Z", "values": [13.0, 23.0]},
        ],
    }
    return trusted, request, response


def _json(value: dict[str, object]) -> bytes:
    return json.dumps(value, allow_nan=True).encode()


def test_valid_point_contract_round_trip() -> None:
    trusted, request, response = _case()
    parsed = parse_request(_json(request), trusted)
    assert parsed.request_id == "example-1"
    assert len(parsed.history) == 2
    validate_response(_json(response), trusted, parsed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "future"),
        ("bundle_sha256", "b" * 64),
        ("sensor_ids", ["SYN_B", "SYN_A"]),
        ("unit", "kW"),
        ("origin_utc", "2026-01-01T01:00:00+00:00"),
        ("request_id", "../bad"),
    ],
)
def test_request_identity_and_schema_rejections(field: str, value: object) -> None:
    trusted, request, _ = _case()
    request[field] = value
    with pytest.raises(InferenceContractError):
        parse_request(_json(request), trusted)


@pytest.mark.parametrize(
    "defect", ["gap", "duplicate", "future", "missing", "nan", "bool", "huge", "extra"]
)
def test_reject_invalid_history(defect: str) -> None:
    trusted, request, _ = _case()
    history = request["history"]
    assert isinstance(history, list)
    if defect == "gap":
        history[0]["timestamp_utc"] = "2025-12-31T23:00:00Z"
    elif defect == "duplicate":
        history[0]["timestamp_utc"] = history[1]["timestamp_utc"]
    elif defect == "future":
        history[1]["timestamp_utc"] = "2026-01-01T02:00:00Z"
    elif defect == "missing":
        history[1]["values"] = [11.0]
    elif defect == "nan":
        history[1]["values"] = [float("nan"), 21.0]
    elif defect == "bool":
        history[1]["values"] = [True, 21.0]
    elif defect == "huge":
        history[1]["values"] = [10**1000, 21.0]
    else:
        history[1]["extra"] = "forbidden"
    with pytest.raises(InferenceContractError):
        parse_request(_json(request), trusted)


@pytest.mark.parametrize(
    "defect", ["wrong-model", "wrong-request", "gap", "interval", "nan", "wrong-vector"]
)
def test_reject_response_mismatch_or_unapproved_interval(defect: str) -> None:
    trusted, request, response = _case()
    parsed = parse_request(_json(request), trusted)
    if defect == "wrong-model":
        response["model_id"] = "other"
    elif defect == "wrong-request":
        response["request_id"] = "other"
    else:
        rows = response["predictions"]
        assert isinstance(rows, list)
        if defect == "gap":
            rows[1]["target_utc"] = "2026-01-01T04:00:00Z"
        elif defect == "interval":
            rows[0]["lower"] = [0.0, 0.0]
        elif defect == "nan":
            rows[0]["values"] = [float("nan"), 22.0]
        else:
            rows[0]["values"] = [12.0]
    with pytest.raises(InferenceContractError):
        validate_response(_json(response), trusted, parsed)


def test_reject_duplicate_fields_and_invalid_json() -> None:
    trusted, _, _ = _case()
    for payload in (b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff", b"{}", b"x" * 256_001):
        with pytest.raises(InferenceContractError):
            parse_request(payload, trusted)


def test_trusted_contract_rejects_duplicates_and_unsupported_units() -> None:
    trusted, _, _ = _case()
    with pytest.raises(InferenceContractError):
        replace(trusted, sensor_ids=("SYN_A", "SYN_A"))
    with pytest.raises(InferenceContractError):
        replace(trusted, unit="kW")
    with pytest.raises(InferenceContractError):
        replace(trusted, horizon_hours=True)


def test_messages_are_not_mutated() -> None:
    trusted, request, response = _case()
    original_request, original_response = deepcopy(request), deepcopy(response)
    parsed = parse_request(_json(request), trusted)
    validate_response(_json(response), trusted, parsed)
    assert request == original_request and response == original_response


def test_missing_trusted_context_is_rejected() -> None:
    trusted, request, response = _case()
    with pytest.raises(InferenceContractError, match="Trusted contract is required"):
        parse_request(_json(request), None)  # type: ignore[arg-type]
    parse_request(_json(request), trusted)
    with pytest.raises(InferenceContractError, match="parsed request are required"):
        validate_response(_json(response), trusted, None)  # type: ignore[arg-type]
