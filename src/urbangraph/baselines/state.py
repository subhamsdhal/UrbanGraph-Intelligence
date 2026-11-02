"""Lossless bounded numerical state encoding; never pickle or executable payloads."""

import base64
import hashlib
import math
import zlib

import numpy as np

from urbangraph.baselines.contracts import FloatArray
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict

MAX_STATE_BYTES = 64 * 1024 * 1024


def encode_array(values: FloatArray) -> dict[str, object]:
    raw = np.asarray(values, dtype="<f8").tobytes(order="C")
    if len(raw) > MAX_STATE_BYTES or not np.isfinite(values).all():
        raise DataError("Invalid or oversized fitted numerical state")
    return {
        "encoding": "zlib-base64",
        "dtype": "<f8",
        "shape": list(values.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
    }


def decode_array(value: object) -> FloatArray:
    obj = object_dict(value)
    shape, data = obj.get("shape"), obj.get("data")
    if (
        set(obj) != {"encoding", "dtype", "shape", "sha256", "data"}
        or obj["encoding"] != "zlib-base64"
        or obj["dtype"] != "<f8"
        or not isinstance(shape, list)
        or not 1 <= len(shape) <= 3
        or any(type(n) is not int or n < 1 for n in shape)
        or not isinstance(data, str)
    ):
        raise DataError("Invalid fitted array encoding")
    dimensions = tuple(int(n) for n in shape)
    size = math.prod(dimensions) * 8
    if size > MAX_STATE_BYTES:
        raise DataError("Oversized fitted array")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(data, validate=True), size + 1)
        if len(raw) != size or not decoder.eof or decoder.unused_data:
            raise DataError("Fitted array size/stream mismatch")
    except (ValueError, zlib.error) as exc:
        raise DataError("Invalid fitted array compression") from exc
    if hashlib.sha256(raw).hexdigest() != obj["sha256"]:
        raise DataError("Fitted array checksum mismatch")
    result = np.frombuffer(raw, dtype="<f8").reshape(dimensions).copy()
    if not np.isfinite(result).all():
        raise DataError("Nonfinite fitted array")
    return result
