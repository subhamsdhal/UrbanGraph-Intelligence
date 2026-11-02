"""Wide Parquet encoding with explicit physical-column/sensor identity mapping."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from urbangraph.data import DataError
from urbangraph.datasets.contracts import CanonicalDataset
from urbangraph.datasets.serialization import canonical_json

WRITER_POLICY = {
    "layout": "wide-v1",
    "compression": "zstd",
    "row_group_size": 65536,
    "parquet_version": "2.6",
}


def write_observations(dataset: CanonicalDataset, path: Path) -> None:
    """Preserve each numeric dtype; map floating NaN to Arrow null explicitly."""
    if path.exists():
        raise DataError("Refusing to overwrite observations")
    frame = dataset.observations
    arrays = [pa.array(frame.index)]
    arrays.extend(
        pa.array(frame.iloc[:, i].to_numpy(copy=False), from_pandas=True)
        for i in range(frame.shape[1])
    )
    names = ["timestamp", *(f"sensor_{i:06d}" for i in range(frame.shape[1]))]
    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata(
        {b"urbangraph.schema": canonical_json(dataset.schema_dict())}
    )
    # Exclusive creation is enforced even on Windows; Parquet owns no leaked handles.
    with path.open("xb") as handle:
        pq.write_table(table, handle, compression="zstd", row_group_size=65536, version="2.6")
        handle.flush()
        os.fsync(handle.fileno())


def read_observations(
    path: Path, schema: dict[str, object], *, raw_snapshot_dir: Path | None = None
) -> pd.DataFrame:
    """Read observations only through the current governed canonical source context."""
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for canonical observation read")
    path = Path(path)
    if path.name != "observations.parquet":
        raise DataError("Canonical observation path required")
    from urbangraph.datasets.governed_canonical import read_canonical
    from urbangraph.datasets.serialization import read_json

    frame = read_canonical(path.parent, raw_snapshot_dir=raw_snapshot_dir)
    if read_json(path.parent / "schema.json") != schema:
        raise DataError("Canonical observation schema differs from requested schema")
    return frame


def _decode_observations(path: Path, schema: dict[str, object]) -> pd.DataFrame:
    """Decode bytes only after a caller has established its authorized boundary."""
    try:
        with path.open("rb") as handle:
            table = pq.read_table(handle)
        metadata = table.schema.metadata or {}
        embedded = json.loads(metadata[b"urbangraph.schema"])
        if embedded != schema:
            raise DataError("Parquet embedded schema differs from schema artifact")
        sensor_columns = schema.get("sensor_columns")
        sensors = schema.get("sensors")
        if not isinstance(sensor_columns, list) or not isinstance(sensors, list):
            raise DataError("Invalid sensor schema")
        if table.column_names != ["timestamp", *sensor_columns]:
            raise DataError("Parquet physical column ordering mismatch")
        if table.num_rows != schema.get("timestamp_count") or len(sensors) != schema.get(
            "sensor_count"
        ):
            raise DataError("Parquet dimensions differ from schema")
        if (
            not pa.types.is_timestamp(table.schema.field(0).type)
            or table.schema.field(0).type.unit != "ns"
        ):
            raise DataError("Parquet timestamp column must have nanosecond type")
        if table.column(0).null_count:
            raise DataError("Parquet timestamps contain nulls")
        identifiers: list[str] = []
        dtypes: list[str] = []
        for i, item in enumerate(sensors):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("identifier"), str)
                or not isinstance(item.get("dtype"), str)
            ):
                raise DataError("Invalid sensor metadata")
            identifiers.append(item["identifier"])
            dtypes.append(item["dtype"])
            dtype = np.dtype(item["dtype"])
            if table.schema.field(i + 1).type != pa.from_numpy_dtype(dtype):
                raise DataError("Parquet numeric dtype mismatch")
            if dtype.kind in "iu" and table.column(i + 1).null_count:
                raise DataError("Unexpected null in nonnullable source integer column")
        frame = cast(pd.DataFrame, table.to_pandas(ignore_metadata=True))
        index = pd.DatetimeIndex(frame.pop("timestamp"), name="timestamp")
        frame.index = index
        frame.columns = pd.Index(identifiers, name="sensor_id")
        if [str(dtype) for dtype in frame.dtypes] != dtypes:
            raise DataError("Restored observation dtypes differ from source")
        return frame
    except DataError:
        raise
    except Exception as exc:
        raise DataError(f"Cannot decode canonical Parquet ({type(exc).__name__})") from exc


def assert_equivalent(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    """Exact dtype/order/value comparison; NaN locations must match."""
    try:
        pd.testing.assert_frame_equal(expected, actual, check_exact=True, check_freq=False)
        # Equality treats +0/-0 alike; preserve sign bits as part of numerical fidelity.
        for i, dtype in enumerate(expected.dtypes):
            if isinstance(dtype, np.dtype) and dtype.kind == "f":
                left, right = expected.iloc[:, i].to_numpy(), actual.iloc[:, i].to_numpy()
                if not np.array_equal(np.signbit(left[left == 0]), np.signbit(right[right == 0])):
                    raise DataError("Signed zero changed in Parquet round trip")
    except AssertionError as exc:
        raise DataError(
            "Parquet round trip changed timestamps, sensors, dtypes or observations"
        ) from exc
