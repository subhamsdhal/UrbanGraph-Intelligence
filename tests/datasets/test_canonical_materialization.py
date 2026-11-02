import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError, verify_snapshot
from urbangraph.datasets import materialization
from urbangraph.datasets.canonical import assert_equivalent
from urbangraph.datasets.contracts import CanonicalDataset, DatasetConfig
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.materialization import (
    load_materialization,
    materialize_snapshot,
    transformation_descriptor,
    transformation_identity,
    verify_materialization,
)

from . import SnapshotFactory


def test_invalid_creation_metadata(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    path = result.directory / "manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    manifest["created_at"] = "not-a-date"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert not verify_materialization(result.directory, raw_snapshot_dir=raw.directory).valid
    with pytest.raises(DataError, match="creation time"):
        materialize_snapshot(raw.directory, output_root=tmp_path / "interim")


def test_roundtrip_lineage_and_reuse(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    raw_before = {
        p.relative_to(raw.directory).as_posix(): p.read_bytes()
        for p in raw.directory.rglob("*")
        if p.is_file()
    }
    parsed = parse_snapshot(raw.directory)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    assert not result.reused
    assert result.directory.parent.name == raw.manifest.snapshot_id
    assert_equivalent(
        parsed.observations,
        load_materialization(result.directory, raw_snapshot_dir=raw.directory),
    )
    assert verify_materialization(result.directory, raw_snapshot_dir=raw.directory).valid
    manifest = json.loads((result.directory / "manifest.json").read_text("utf-8"))
    assert manifest["descriptor"]["input_file"] == {
        "relative_path": "metr-la.h5",
        "size_bytes": raw.manifest.files[0].size_bytes,
        "sha256": raw.manifest.files[0].sha256,
    }
    assert set(manifest["descriptor"]["dependencies"]) >= {
        "python",
        "pandas",
        "numpy",
        "tables",
        "pyarrow",
        "hdf5",
    }
    assert (result.directory / "raw-manifest.json").read_bytes() == raw.manifest_path.read_bytes()
    output_before = {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in result.directory.iterdir()
    }
    repeated = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    assert repeated.reused and repeated.transformation_id == result.transformation_id
    assert output_before == {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in result.directory.iterdir()
    }
    assert raw_before == {
        p.relative_to(raw.directory).as_posix(): p.read_bytes()
        for p in raw.directory.rglob("*")
        if p.is_file()
    }
    assert verify_snapshot(raw.directory).valid


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64"])
def test_dtype_and_value_preservation(
    snapshot_factory: SnapshotFactory, tmp_path: Path, dtype: str
) -> None:
    values: list[int | float] = (
        [0, 2**53 + 1, 2**53 + 3] if "int" in dtype else [-0.0, np.nan, np.inf]
    )
    frame = pd.DataFrame(
        {"timestamp": np.array(values, dtype=dtype), "other": np.array([1, 2, 3], dtype="int32")},
        index=pd.date_range("2024", periods=3, freq="5min"),
    )
    raw = snapshot_factory(frame)
    parsed = parse_snapshot(raw.directory)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    actual = load_materialization(result.directory, raw_snapshot_dir=raw.directory)
    assert_equivalent(parsed.observations, actual)
    assert actual.dtypes.iloc[0] == np.dtype(dtype)


def test_timezone_roundtrip(snapshot_factory: SnapshotFactory, tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"a": [1.0] * 30},
        index=pd.date_range("2024-11-03 00:55", periods=30, freq="5min", tz="America/Los_Angeles"),
    )
    raw = snapshot_factory(frame, "pems-bay")
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    pd.testing.assert_index_equal(
        load_materialization(result.directory, raw_snapshot_dir=raw.directory).index,
        frame.index,
        check_names=False,
    )


def test_transformation_identity(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    dataset = parse_snapshot(raw.directory)
    descriptor = transformation_descriptor(dataset)
    original = transformation_identity(descriptor)
    assert original == transformation_identity(dict(reversed(list(descriptor.items()))))
    for field in (
        "parser_version",
        "canonical_schema_version",
        "raw_snapshot_id",
        "raw_manifest_sha256",
        "profile_version",
    ):
        changed = {**descriptor, field: "changed"}
        assert transformation_identity(changed) != original
    changed_versions = json.loads(json.dumps(descriptor))
    changed_versions["dependencies"]["pandas"] = "future-version"
    assert transformation_identity(changed_versions) != original
    with_key = parse_snapshot(raw.directory, config=DatasetConfig(hdf_key="/traffic"))
    assert transformation_identity(transformation_descriptor(with_key)) == original
    default = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    changed_output = materialize_snapshot(
        raw.directory, output_root=tmp_path / "interim", config=DatasetConfig(low_variance_std=2.0)
    )
    assert default.directory != changed_output.directory
    assert verify_materialization(default.directory, raw_snapshot_dir=raw.directory).valid


@pytest.mark.parametrize(
    "mutation", ["parquet", "missing", "unexpected", "manifest", "quality", "schema"]
)
def test_corrupt_outputs_not_reused(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path, mutation: str
) -> None:
    raw = snapshot_factory(traffic_frame)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    target = (
        result.directory
        / {
            "parquet": "observations.parquet",
            "missing": "schema.json",
            "unexpected": "extra.bin",
            "manifest": "manifest.json",
            "quality": "quality.json",
            "schema": "schema.json",
        }[mutation]
    )
    if mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(b"invalid artifact")
    before = {p.name: p.read_bytes() for p in result.directory.iterdir()}
    assert not verify_materialization(result.directory, raw_snapshot_dir=raw.directory).valid
    with pytest.raises(DataError):
        materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    assert before == {p.name: p.read_bytes() for p in result.directory.iterdir()}
    assert verify_snapshot(raw.directory).valid


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_partial_failure_cleanup(
    snapshot_factory: SnapshotFactory,
    traffic_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    raw = snapshot_factory(traffic_frame)
    existing = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    before = (existing.directory / "manifest.json").read_bytes()

    def broken_write(dataset: CanonicalDataset, path: Path) -> None:
        path.write_bytes(b"partial")
        raise OSError("simulated failed write")

    def broken_publish(stage: Path, destination: Path) -> None:
        assert (stage / "manifest.json").exists()
        raise OSError("simulated failed rename")

    if failure == "write":
        monkeypatch.setattr(materialization, "write_observations", broken_write)
    else:
        monkeypatch.setattr(materialization, "_publish", broken_publish)
    with pytest.raises(OSError):
        materialize_snapshot(
            raw.directory,
            output_root=tmp_path / "interim",
            config=DatasetConfig(low_variance_std=3.0),
        )
    assert list(existing.directory.parent.iterdir()) == [existing.directory]
    assert (existing.directory / "manifest.json").read_bytes() == before
    assert verify_snapshot(raw.directory).valid


def test_raw_root_overlap_rejected(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame
) -> None:
    raw = snapshot_factory(traffic_frame)
    with pytest.raises(DataError, match="disjoint"):
        materialize_snapshot(raw.directory, output_root=raw.directory / "canonical")
    assert verify_snapshot(raw.directory).valid


def test_quality_gate_prevents_publication(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    with pytest.raises(DataError, match="quality rejection"):
        materialize_snapshot(
            raw.directory,
            output_root=tmp_path / "interim",
            config=DatasetConfig(reject_checks=("negative_values",)),
        )
    assert not (tmp_path / "interim").exists()


def test_external_raw_lineage_rejection(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    other = snapshot_factory(traffic_frame + 1)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    assert not verify_materialization(result.directory, raw_snapshot_dir=other.directory).valid


def test_staging_not_valid_public_output(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame, tmp_path: Path
) -> None:
    raw = snapshot_factory(traffic_frame)
    result = materialize_snapshot(raw.directory, output_root=tmp_path / "interim")
    stage = result.directory.with_name(".staging-interrupted")
    result.directory.rename(stage)
    assert not verify_materialization(stage, raw_snapshot_dir=raw.directory).valid
