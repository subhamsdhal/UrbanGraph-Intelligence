"""Non-executing, synthetic-only request/response boundary for future inference.

No model is loaded. A trusted caller must supply an independently approved bundle
identity and authenticate the caller before any future service uses these contracts.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

SCHEMA = "ug-inference-contract-1"
MAX_JSON_BYTES = 256_000
MAX_SENSORS = 512
MAX_HISTORY = 720
MAX_HORIZON = 168
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_UTC_HOUR = re.compile(r"\d{4}-\d\d-\d\dT\d\d:00:00Z\Z")


class InferenceContractError(ValueError):
    """An inference message is malformed or incompatible with a trusted contract."""


@dataclass(frozen=True)
class TrustedContract:
    bundle_sha256: str
    model_id: str
    sensor_ids: tuple[str, ...]
    history_hours: int
    horizon_hours: int
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.bundle_sha256, str) or not _DIGEST.fullmatch(self.bundle_sha256):
            raise InferenceContractError("Trusted bundle digest is invalid")
        if not isinstance(self.model_id, str) or not _LABEL.fullmatch(self.model_id):
            raise InferenceContractError("Trusted model ID is invalid")
        if (
            not isinstance(self.sensor_ids, tuple)
            or not 1 <= len(self.sensor_ids) <= MAX_SENSORS
            or any(not isinstance(s, str) or not _LABEL.fullmatch(s) for s in self.sensor_ids)
            or len(set(self.sensor_ids)) != len(self.sensor_ids)
        ):
            raise InferenceContractError("Trusted sensor axis is invalid")
        if (
            type(self.history_hours) is not int
            or not 1 <= self.history_hours <= MAX_HISTORY
            or type(self.horizon_hours) is not int
            or not 1 <= self.horizon_hours <= MAX_HORIZON
            or self.unit != "MW"
        ):
            raise InferenceContractError("Trusted temporal or unit contract is invalid")


@dataclass(frozen=True)
class Request:
    request_id: str
    origin_utc: datetime
    history: tuple[tuple[datetime, tuple[float, ...]], ...]


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InferenceContractError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise InferenceContractError("Nonfinite JSON constant")


def _decode(payload: bytes) -> object:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_JSON_BYTES:
        raise InferenceContractError("JSON message missing or oversized")
    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject_constant
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InferenceContractError("Invalid JSON message") from exc


def _shape(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise InferenceContractError("JSON fields differ from contract")
    return value


def _hour(value: object) -> datetime:
    if not isinstance(value, str) or not _UTC_HOUR.fullmatch(value):
        raise InferenceContractError("Timestamp must be an exact UTC hour")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError as exc:
        raise InferenceContractError("Invalid UTC timestamp") from exc


def _values(value: object, count: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise InferenceContractError("Value count differs from sensor axis")
    try:
        numbers = tuple(float(item) for item in value if type(item) in (int, float))
    except (OverflowError, ValueError) as exc:
        raise InferenceContractError("Sensor values must be finite numbers") from exc
    if len(numbers) != count or any(not math.isfinite(item) for item in numbers):
        raise InferenceContractError("Sensor values must be finite numbers")
    return numbers


def parse_request(payload: bytes, trusted: TrustedContract) -> Request:
    """Validate complete, aligned hourly history; no model execution occurs."""
    if not isinstance(trusted, TrustedContract):
        raise InferenceContractError("Trusted contract is required")
    obj = _shape(
        _decode(payload),
        {
            "schema_version",
            "bundle_sha256",
            "request_id",
            "origin_utc",
            "sensor_ids",
            "unit",
            "history",
        },
    )
    if obj["schema_version"] != SCHEMA or obj["bundle_sha256"] != trusted.bundle_sha256:
        raise InferenceContractError("Unsupported schema or bundle identity")
    if obj["sensor_ids"] != list(trusted.sensor_ids) or obj["unit"] != trusted.unit:
        raise InferenceContractError("Sensor order or unit differs from trusted bundle")
    request_id = obj["request_id"]
    if not isinstance(request_id, str) or not _LABEL.fullmatch(request_id):
        raise InferenceContractError("Invalid request ID")
    origin = _hour(obj["origin_utc"])
    history = obj["history"]
    if not isinstance(history, list) or len(history) != trusted.history_hours:
        raise InferenceContractError("History length differs from trusted bundle")
    parsed: list[tuple[datetime, tuple[float, ...]]] = []
    start = origin - timedelta(hours=trusted.history_hours - 1)
    for index, row in enumerate(history):
        values = _shape(row, {"timestamp_utc", "values"})
        timestamp = _hour(values["timestamp_utc"])
        if timestamp != start + timedelta(hours=index):
            raise InferenceContractError("History has a gap, duplicate or future timestamp")
        parsed.append((timestamp, _values(values["values"], len(trusted.sensor_ids))))
    return Request(request_id, origin, tuple(parsed))


def validate_response(payload: bytes, trusted: TrustedContract, request: Request) -> None:
    """Reject mismatched or invented interval claims in a point-forecast response."""
    if not isinstance(trusted, TrustedContract) or not isinstance(request, Request):
        raise InferenceContractError("Trusted contract and parsed request are required")
    obj = _shape(
        _decode(payload),
        {
            "schema_version",
            "bundle_sha256",
            "model_id",
            "request_id",
            "origin_utc",
            "sensor_ids",
            "unit",
            "predictions",
        },
    )
    if (
        obj["schema_version"] != SCHEMA
        or obj["bundle_sha256"] != trusted.bundle_sha256
        or obj["model_id"] != trusted.model_id
        or obj["request_id"] != request.request_id
        or _hour(obj["origin_utc"]) != request.origin_utc
        or obj["sensor_ids"] != list(trusted.sensor_ids)
        or obj["unit"] != trusted.unit
    ):
        raise InferenceContractError("Response does not match the trusted request and bundle")
    predictions = obj["predictions"]
    if not isinstance(predictions, list) or len(predictions) != trusted.horizon_hours:
        raise InferenceContractError("Prediction horizon differs from trusted bundle")
    for index, row in enumerate(predictions, start=1):
        prediction = _shape(row, {"target_utc", "values"})
        if _hour(prediction["target_utc"]) != request.origin_utc + timedelta(hours=index):
            raise InferenceContractError("Prediction timestamp or horizon is misaligned")
        _values(prediction["values"], len(trusted.sensor_ids))
