import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import discover, stream_file
from urbangraph.datasets.contracts import DatasetConfig, StructuralError
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.validation import validate_frame

from . import SnapshotFactory


def test_non_native_numeric_byte_order() -> None:
    dtype = np.dtype("float64").newbyteorder("S")
    frame = pd.DataFrame(
        {"sensor": np.array([1, 2], dtype=dtype)},
        index=pd.date_range("2024-01-01", periods=2, freq="5min"),
    )
    with pytest.raises(StructuralError, match="observation_dtype") as error:
        validate_frame(frame)
    assert "native-endian" in error.value.findings[0].description
    assert error.value.findings[0].context["dtype"] == str(dtype)


@pytest.mark.parametrize("dataset_id", ["metr-la", "pems-bay"])
@pytest.mark.parametrize("storage", ["fixed", "table"])
def test_hdf_adapters(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, dataset_id: str, storage: str
) -> None:
    raw = snapshot_factory(
        traffic_frame, dataset_id, storage="fixed" if storage == "fixed" else "table"
    )
    before = (raw.directory / "payload" / f"{dataset_id}.h5").read_bytes()
    parsed = parse_snapshot(raw.directory, dataset_id=dataset_id)
    assert parsed.dataset_id == dataset_id
    assert parsed.raw_snapshot_id == raw.manifest.snapshot_id
    assert parsed.sensor_ids == ("sensor-Z", "001", "A")
    pd.testing.assert_frame_equal(
        parsed.observations, traffic_frame, check_names=False, check_freq=False
    )
    assert parsed.config.hdf_key == "/traffic"
    assert parsed.lineage.input_file == raw.manifest.files[0]
    assert (raw.directory / "payload" / f"{dataset_id}.h5").read_bytes() == before


def test_multiple_keys(snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame) -> None:
    raw = snapshot_factory(traffic_frame, other=traffic_frame * 2)
    with pytest.raises(DataError, match="Ambiguous"):
        parse_snapshot(raw.directory)
    selected = parse_snapshot(raw.directory, config=DatasetConfig(hdf_key="other"))
    assert selected.observations.iloc[0, 0] == 20
    with pytest.raises(DataError, match="does not exist"):
        parse_snapshot(raw.directory, config=DatasetConfig(hdf_key="absent"))


def test_identity_mismatch(snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame) -> None:
    raw = snapshot_factory(traffic_frame)
    with pytest.raises(DataError, match="differs"):
        parse_snapshot(raw.directory, dataset_id="pems-bay")


@pytest.mark.parametrize("kind", ["missing-file", "invalid-hdf", "series", "arbitrary-hdf"])
def test_unsupported_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "metr-la.h5"
    if kind == "series":
        pd.Series([1.0, 2.0]).to_hdf(payload, key="traffic", mode="w")
    elif kind == "arbitrary-hdf":
        import tables  # type: ignore[import-untyped]

        with tables.open_file(payload, mode="w") as handle:
            handle.create_array("/", "matrix", np.zeros((2, 2)))
    else:
        (source / ("other.bin" if kind == "missing-file" else "metr-la.h5")).write_bytes(b"not HDF")
    # This invented source is authorized only for this test's exact bytes and
    # destination. A dataset label alone grants no rights to real METR-LA.
    records = tuple(
        RawFileRecord(relative, *stream_file(source / relative))
        for relative in discover(source).files
    )
    now = datetime.now(UTC)
    policy = tmp_path / "invented-unsupported-payload-rights.json"
    policy.write_text(
        json.dumps(
            {
                "version": 1,
                "grants": [
                    {
                        "source_id": "metr-la",
                        "source_sha256": content_identity(records),
                        "source_path": str(source),
                        "purpose": "local_research",
                        "operation": "raw_ingest",
                        "destination_root": str(tmp_path / "raw"),
                        "evidence_reference": "invented-unsupported-payload-test-only",
                        "status": "approved",
                        "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                        "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                        "retention_days": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    raw = ingest_dataset(source, "metr-la", raw_root=tmp_path / "raw")
    with pytest.raises(DataError):
        parse_snapshot(raw.directory)


def test_corrupted_raw_rejected(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = snapshot_factory(traffic_frame)
    (raw.directory / "payload" / "metr-la.h5").write_bytes(b"corrupt")

    def must_not_open(*args: object, **kwargs: object) -> None:
        pytest.fail("Parser opened HDF before verifying raw integrity")

    monkeypatch.setattr(pd, "HDFStore", must_not_open)
    with pytest.raises(DataError, match="integrity"):
        parse_snapshot(raw.directory)


@pytest.mark.parametrize("kind", ["duplicate", "unsorted", "nat", "string-index"])
def test_bad_timestamps(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, kind: str
) -> None:
    frame = traffic_frame.copy()
    expected = "timestamp"
    if kind == "duplicate":
        frame.index = pd.DatetimeIndex(
            [frame.index[0], frame.index[0], frame.index[2], frame.index[3]]
        )
        expected = "duplicate_timestamps"
    elif kind == "unsorted":
        frame = frame.iloc[::-1]
    elif kind == "nat":
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    else:
        frame.index = pd.Index(["a", "b", "c", "d"])
    raw = snapshot_factory(frame)
    with pytest.raises(StructuralError, match=expected):
        parse_snapshot(raw.directory)


@pytest.mark.parametrize(
    "labels,check",
    [
        (["a", "a", "b"], "duplicate_sensors"),
        ([1, "1", "b"], "sensor_normalization_collision"),
        (["é", "e\u0301", "b"], "sensor_normalization_collision"),
        (["", "a", "b"], "missing_sensor_id"),
        ([np.nan, "a", "b"], "missing_sensor_id"),
        ([True, "a", "b"], "sensor_type"),
    ],
)
def test_sensor_structural_checks(
    traffic_frame: pd.DataFrame, labels: list[object], check: str
) -> None:
    traffic_frame.columns = pd.Index(labels, dtype=object)
    with pytest.raises(StructuralError, match=check):
        validate_frame(traffic_frame)


def test_nonnumeric_rejected(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame
) -> None:
    traffic_frame["A"] = ["1", "2", "3", "4"]
    raw = snapshot_factory(traffic_frame)
    with pytest.raises(StructuralError, match="observation_dtype"):
        parse_snapshot(raw.directory)


def test_original_sensor_identity(snapshot_factory: SnapshotFactory) -> None:
    frame = pd.DataFrame([[1.0, 2.0]], index=pd.date_range("2024", periods=1), columns=[715, 209])
    parsed = parse_snapshot(snapshot_factory(frame).directory)
    assert parsed.sensor_ids == ("715", "209")
    assert parsed.sensors[0].original_value == "715"
    assert parsed.sensors[0].original_type == "int"


def test_upstream_nan_label_conversion_is_rejected(snapshot_factory: SnapshotFactory) -> None:
    # pandas fixed HDF decodes its reserved nan_rep spelling as a missing axis
    # label. Do not silently invent a sensor identity after that information loss.
    frame = pd.DataFrame({"nan": [1.0]}, index=pd.date_range("2024", periods=1))
    raw = snapshot_factory(frame)
    with pytest.raises(StructuralError, match="missing_sensor_id"):
        parse_snapshot(raw.directory)


@pytest.mark.parametrize(
    "offsets,expected_gaps,missing,off_grid",
    [
        ([0, 5, 10], 0, 0, 0),
        ([0, 5, 15], 1, 1, 0),
        ([0, 7, 12], 1, 2, 2),
        ([0], 0, 0, 0),
    ],
)
def test_temporal_grid(
    snapshot_factory: SnapshotFactory,
    offsets: list[int],
    expected_gaps: int,
    missing: int,
    off_grid: int,
) -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=n) for n in offsets]
    )
    frame = pd.DataFrame({"a": np.ones(len(index))}, index=index)
    parsed = parse_snapshot(snapshot_factory(frame).directory)
    assert parsed.temporal["gap_count"] == expected_gaps
    assert parsed.temporal["missing_grid_slots"] == missing
    assert parsed.temporal["off_grid_timestamps"] == off_grid
    assert parsed.temporal["timezone"] is None
    assert parsed.temporal["timezone_aware"] is False
    assert len(parsed.observations) == len(offsets)


def test_timezone_dst_preservation(snapshot_factory: SnapshotFactory) -> None:
    index = pd.date_range("2024-11-03 00:55", periods=30, freq="5min", tz="America/Los_Angeles")
    frame = pd.DataFrame({"a": np.arange(30, dtype=float)}, index=index)
    parsed = parse_snapshot(snapshot_factory(frame).directory)
    pd.testing.assert_index_equal(parsed.observations.index, index, check_names=False)
    assert parsed.temporal["timezone"] == "America/Los_Angeles"
    assert parsed.temporal["gap_count"] == 0
    assert parsed.temporal["repeated_local_wall_timestamps"] == 12


def test_explicit_units_require_evidence() -> None:
    with pytest.raises(DataError, match="Units require"):
        DatasetConfig(units="mph")
    assert DatasetConfig().units is None
    assert (
        replace(DatasetConfig(), units="mph", units_reference="user supplied document").units
        == "mph"
    )
