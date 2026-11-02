import json

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError
from urbangraph.datasets.contracts import DatasetConfig
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.profiling import profile_dataset

from . import SnapshotFactory


def test_quality_counts_and_preservation(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame
) -> None:
    parsed = parse_snapshot(snapshot_factory(traffic_frame).directory)
    before = parsed.observations.copy(deep=True)
    report = profile_dataset(parsed)
    assert report.accepted
    for key, expected in {
        "timestamp_count": 4,
        "sensor_count": 3,
        "observation_count": 12,
        "missing_count": 1,
        "nan_count": 1,
        "infinity_count": 1,
        "zero_count": 1,
        "negative_count": 1,
        "constant_sensor_count": 1,
        "finite_count": 10,
    }.items():
        assert report.metrics[key] == expected
    assert report.metrics["missing_percentage"] == pytest.approx(100 / 12)
    assert report.sensors[0]["statistics"] == {
        "min": 10.0,
        "max": 30.0,
        "mean": 20.0,
        "median": 20.0,
        "std": 10.0,
        "q25": 15.0,
        "q75": 25.0,
    }
    assert report.sensors[0]["first_finite_timestamp"] == "2024-01-01T00:00:00"
    assert report.sensors[0]["last_finite_timestamp"] == "2024-01-01T00:15:00"
    assert {
        "zero_values",
        "negative_values",
        "infinite_values",
        "constant_sensor",
        "temporal_gaps",
        "uneven_coverage",
    } <= {f.check_id for f in report.findings}
    pd.testing.assert_frame_equal(parsed.observations, before)
    assert report.to_dict() == profile_dataset(parsed).to_dict()
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))["accepted"] is True


def test_missing_constant_low_variance(snapshot_factory: SnapshotFactory) -> None:
    frame = pd.DataFrame(
        {"missing": [np.nan] * 3, "zero": [0.0] * 3, "low": [1.0, 1.001, 1.002]},
        index=pd.date_range("2024", periods=3, freq="5min"),
    )
    report = profile_dataset(parse_snapshot(snapshot_factory(frame).directory))
    assert report.metrics["missing_count"] == 3
    assert report.metrics["zero_count"] == 3
    assert report.metrics["constant_sensor_count"] == 1
    statistics = report.sensors[0]["statistics"]
    assert isinstance(statistics, dict)
    assert all(value is None for value in statistics.values())
    assert report.sensors[0]["first_finite_timestamp"] is None
    assert {"all_missing_sensor", "extreme_missingness", "low_variance_sensor"} <= {
        f.check_id for f in report.findings
    }


def test_selected_rejection_policy(
    snapshot_factory: SnapshotFactory, traffic_frame: pd.DataFrame
) -> None:
    raw = snapshot_factory(traffic_frame)
    strict = parse_snapshot(raw.directory, config=DatasetConfig(reject_checks=("negative_values",)))
    report = profile_dataset(strict)
    assert not report.accepted
    assert any(f.check_id == "negative_values" and f.severity == "ERROR" for f in report.findings)
    assert profile_dataset(parse_snapshot(raw.directory)).accepted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_variance_std": -1.0},
        {"low_variance_std": float("nan")},
        {"extreme_missing_fraction": 2.0},
        {"nominal_interval_seconds": 0},
        {"max_gap_examples": -1},
        {"reject_checks": ("unknown",)},
    ],
)
def test_invalid_policy(kwargs: dict[str, object]) -> None:
    from urbangraph.datasets.serialization import config_from_dict

    obj = DatasetConfig().to_dict()
    obj.update(kwargs)
    with pytest.raises(DataError):
        config_from_dict(obj)


def test_all_nonfinite_report_is_json(snapshot_factory: SnapshotFactory) -> None:
    frame = pd.DataFrame(
        {"missing-sensor": [np.nan, np.nan], "infinite-sensor": [np.inf, -np.inf]},
        index=pd.date_range("2024", periods=2, freq="5min"),
    )
    report = profile_dataset(parse_snapshot(snapshot_factory(frame).directory))
    assert report.metrics["finite_count"] == 0
    json.dumps(report.to_dict(), allow_nan=False)


def test_gap_examples_bounded(snapshot_factory: SnapshotFactory) -> None:
    frame = pd.DataFrame({"a": [1.0] * 10}, index=pd.date_range("2024", periods=10, freq="10min"))
    dataset = parse_snapshot(
        snapshot_factory(frame).directory, config=DatasetConfig(max_gap_examples=2)
    )
    assert dataset.temporal["gap_count"] == 9
    examples = dataset.temporal["gap_examples"]
    assert isinstance(examples, list)
    assert len(examples) == 2
    assert dataset.temporal["gap_examples_omitted"] == 7


def test_large_integers_not_falsely_constant(snapshot_factory: SnapshotFactory) -> None:
    frame = pd.DataFrame(
        {"a": np.array([2**53, 2**53 + 1], dtype="int64")},
        index=pd.date_range("2024", periods=2, freq="5min"),
    )
    report = profile_dataset(parse_snapshot(snapshot_factory(frame).directory))
    assert report.metrics["constant_sensor_count"] == 0
    assert any(f.check_id == "audit_precision" for f in report.findings)
