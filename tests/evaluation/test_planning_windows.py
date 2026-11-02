from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import (
    EvaluationProtocol,
    FeatureAvailabilityPolicy,
    ForecastWindowSpecification,
    PartitionBoundary,
    PlanConfig,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import WindowCollection


@pytest.mark.parametrize(
    ("rows", "expected"), [(38, (10, 2, 3)), (48, (18, 2, 5)), (58, (24, 4, 7)), (100, (54, 8, 15))]
)
def test_reference_rounding(rows: int, expected: tuple[int, ...]) -> None:
    config = PlanConfig(protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark"))
    split = plan_splits(pd.date_range("2024", periods=rows, freq="5min"), config)
    assert tuple(p.sample_count for p in split.folds[0].partitions) == expected
    assert config.window.input_offsets == (-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0)
    assert config.window.target_offsets == (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def test_reference_toy_indices_and_features(dataset: EvaluationDataset) -> None:
    data = replace(dataset, frame=dataset.frame.iloc[:38])
    config = PlanConfig(protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark"))
    split = plan_splits(data.timestamps, config)
    train = WindowCollection(data, split, fold=0, partition="train")
    validation = WindowCollection(data, split, fold=0, partition="validation")
    test = WindowCollection(data, split, fold=0, partition="test")
    assert [train[i].origin.position for i in range(len(train))] == list(range(11, 21))
    assert [validation[i].origin.position for i in range(len(validation))] == [21, 22]
    assert [test[i].origin.position for i in range(len(test))] == [23, 24, 25]
    np.testing.assert_array_equal(train[0].x[:, 0, 0], np.arange(12))
    np.testing.assert_allclose(train[0].x[:, 0, 1], np.arange(12) / 288)
    np.testing.assert_array_equal(train[0].reference_y[:, 0, 0], np.arange(12, 24))
    np.testing.assert_allclose(train[0].reference_y[:, 0, 1], np.arange(12, 24) / 288)
    assert set(train[9].target_positions) & set(validation[0].target_positions) == set(
        range(22, 33)
    )
    assert train[0].x.shape == (12, 2, 2)
    assert train[0].y.shape == (12, 2, 1)


def test_expanding_folds(config: PlanConfig, dataset: EvaluationDataset) -> None:
    split = plan_splits(dataset.timestamps, config)
    assert [(p.name, p.start, p.stop) for p in split.folds[0].partitions] == [
        ("train", 0, 20),
        ("validation", 20, 30),
        ("calibration", 30, 40),
        ("test", 40, 50),
    ]
    assert [(p.start, p.stop) for p in split.folds[1].partitions] == [
        (0, 30),
        (30, 40),
        (40, 50),
        (50, 60),
    ]
    assert split.folds[0].fold_id == "fold-0000"
    assert split == plan_splits(dataset.timestamps.copy(), config)


@pytest.mark.parametrize("context", ["isolated", "operational"])
def test_boundary_context(config: PlanConfig, dataset: EvaluationDataset, context: str) -> None:
    config = replace(config, protocol=EvaluationProtocol(context=context))
    split = plan_splits(dataset.timestamps, config)
    windows = WindowCollection(dataset, split, fold=0, partition="validation")
    sample = windows[0]
    assert sample.input_positions == ((20, 21, 22) if context == "isolated" else (17, 18, 19))
    assert sample.target_positions == ((23, 24) if context == "isolated" else (20, 21))
    assert sample.sensor_ids == ("sensor-Z", "001")
    assert sample.x.shape == (3, 2, 2) and sample.y.shape == (2, 2, 1)
    np.testing.assert_array_equal(sample.x[:, 1, 0], 2 * np.array(sample.input_positions) + 1)
    before = dataset.frame.copy(deep=True)
    sample.x[:] = -100
    pd.testing.assert_frame_equal(dataset.frame, before)
    with pytest.raises(IndexError):
        windows[-1]


def test_explicit_timestamp_and_index_boundaries(
    config: PlanConfig, dataset: EvaluationDataset
) -> None:
    boundaries = (
        PartitionBoundary("train", 0, 20),
        PartitionBoundary("validation", 22, 35),
        PartitionBoundary("test", 37, 100),
    )
    indexed = replace(
        config,
        protocol=EvaluationProtocol("strict-chronological-v1", "operational"),
        boundaries=boundaries,
        embargo=2,
    )
    timed = replace(
        indexed,
        boundaries=tuple(
            PartitionBoundary(
                b.name,
                dataset.timestamps[int(b.start)].isoformat(),
                (
                    dataset.timestamps[int(b.stop)]
                    if b.stop != 100
                    else dataset.timestamps[-1] + pd.Timedelta(minutes=5)
                ).isoformat(),
            )
            for b in boundaries
        ),
    )
    assert (
        plan_splits(dataset.timestamps, indexed).folds
        == plan_splits(dataset.timestamps, timed).folds
    )
    assert (
        plan_splits(dataset.timestamps, indexed).folds[0].partition("test").stop_timestamp_exclusive
        is None
    )


@pytest.mark.parametrize("bad", [(-1, 20, 30), (0, 20, 101), (0, 25, 24)])
def test_invalid_boundaries(
    config: PlanConfig, dataset: EvaluationDataset, bad: tuple[int, int, int]
) -> None:
    with pytest.raises(DataError):
        boundaries = (
            PartitionBoundary("train", bad[0], bad[1]),
            PartitionBoundary("validation", bad[1], bad[2]),
            PartitionBoundary("test", bad[2], 100),
        )
        plan_splits(
            dataset.timestamps,
            replace(
                config,
                protocol=EvaluationProtocol("strict-chronological-v1"),
                boundaries=boundaries,
            ),
        )


@pytest.mark.parametrize("rows", [1, 23, 24, 28])
def test_insufficient_benchmark(rows: int) -> None:
    with pytest.raises(DataError, match="empty"):
        plan_splits(
            pd.date_range("2024", periods=rows, freq="5min"),
            PlanConfig(protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark")),
        )


def test_gap_rejection_is_not_silent_parity(config: PlanConfig, dataset: EvaluationDataset) -> None:
    changed = dataset.frame.copy()
    times = changed.index.to_list()
    times[10:] = [t + pd.Timedelta(minutes=5) for t in times[10:]]
    changed.index = pd.DatetimeIndex(times)
    data = replace(dataset, frame=changed)
    strict = plan_splits(data.timestamps, config)
    assert strict.folds[0].partition("train").rejected_gap_samples == 4
    windows = WindowCollection(data, strict, fold=0, partition="train")
    assert {windows[i].origin.position for i in range(len(windows))}.isdisjoint({8, 9, 10, 11})
    with pytest.raises(DataError, match="cadence"):
        plan_splits(
            data.timestamps,
            PlanConfig(protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark")),
        )


def test_config_roundtrip_and_validation(config: PlanConfig) -> None:
    assert config_from_dict(config.to_dict()) == config
    for value in (
        {"typo": 1},
        {"folds": True},
        {"validity": {"mask_zero": 1}},
        {"features": {"future_speed": True}},
        {"window": {"history": 0}},
    ):
        with pytest.raises(DataError):
            config_from_dict(value)
    with pytest.raises(DataError):
        EvaluationProtocol("dcrnn-sample-first-v1", "isolated")
    with pytest.raises(DataError):
        replace(config, minimum_train=4, window=ForecastWindowSpecification())
        # Length is validated against the actual index by the planner.
        plan_splits(
            pd.date_range("2024", periods=100, freq="5min"), replace(config, minimum_train=4)
        )


def test_calendar_awareness_and_latency(config: PlanConfig, dataset: EvaluationDataset) -> None:
    frame = dataset.frame.copy()
    frame.index = pd.date_range("2024-11-03 00:00", periods=100, freq="5min", tz="America/New_York")
    data = replace(dataset, frame=frame)
    cfg = replace(
        config,
        protocol=EvaluationProtocol(context="operational"),
        features=FeatureAvailabilityPolicy(day_of_week=True, observed_delay_steps=2),
    )
    split = plan_splits(data.timestamps, cfg)
    sample = WindowCollection(data, split, fold=0, partition="validation")[0]
    assert sample.origin.position == 21
    assert sample.input_positions[-1] == 19
    assert sample.x.shape == (3, 2, 9)
    assert sample.feature_metadata["timezone"] == "America/New_York"
    assert sample.future_known[0, 0, -1] == 1  # Sunday one-hot.
