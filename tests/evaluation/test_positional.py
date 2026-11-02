"""Independent positional reference equations and strict cadence regression checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.benchmarks.pipeline import evaluation_configs
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.contracts import (
    POSITIONAL_PROTOCOL,
    EvaluationProtocol,
    FeatureAvailabilityPolicy,
    PlanConfig,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.preprocessing import fit_standardizer
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.evaluation.windows import WindowCollection


def positional() -> PlanConfig:
    return PlanConfig(protocol=EvaluationProtocol(POSITIONAL_PROTOCOL, "benchmark"))


def jump_dataset() -> EvaluationDataset:
    index = pd.date_range("2017-03-11 17:40", periods=300, freq="5min")
    index = pd.DatetimeIndex([t + pd.Timedelta(hours=int(i >= 100)) for i, t in enumerate(index)])
    frame = pd.DataFrame(
        {"sensor-Z": np.arange(300, dtype=float), "001": np.arange(300, dtype=float) * 2},
        index=index,
    )
    return EvaluationDataset(Path("unused"), frame, {"fixture": "synthetic"})


@pytest.mark.parametrize(
    "rows,counts",
    [
        (38, (10, 2, 3)),
        (48, (18, 2, 5)),
        (58, (24, 4, 7)),
        (34272, (23974, 3425, 6850)),
        (52116, (36465, 5209, 10419)),
    ],
)
def test_positional_counts(rows: int, counts: tuple[int, ...]) -> None:
    index = pd.date_range("2017", periods=rows, freq="5min")
    split = plan_splits(index, positional())
    assert tuple(p.sample_count for p in split.folds[0].partitions) == counts
    legacy = plan_splits(
        index, PlanConfig(protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark"))
    )
    assert split.folds == legacy.folds


def test_reference_windows_and_calendar() -> None:
    dataset = jump_dataset()
    before = dataset.frame.copy(deep=True)
    split = plan_splits(dataset.timestamps, positional())
    # Independently express the pinned reference's timestamp day fraction and offsets.
    index = dataset.timestamps.values
    fraction = (index - index.astype("datetime64[D]")) / np.timedelta64(1, "D")
    speed = dataset.frame.to_numpy()[:, :, None]
    reference = np.concatenate(
        (speed, np.broadcast_to(fraction[:, None, None], speed.shape)), axis=2
    )
    reference_origins = list(range(11, len(index) - 12))
    sample_position = 0
    for part in split.folds[0].partitions:
        windows = WindowCollection(dataset, split, fold=0, partition=part.name)
        for i in range(len(windows)):
            origin = reference_origins[sample_position]
            sample = windows[i]
            np.testing.assert_array_equal(sample.x, reference[origin + np.arange(-11, 1)])
            np.testing.assert_array_equal(sample.reference_y, reference[origin + np.arange(1, 13)])
            assert sample.sensor_ids == ("sensor-Z", "001")
            assert sample.input_timestamps == tuple(
                t.isoformat() for t in dataset.timestamps[origin - 11 : origin + 1]
            )
            assert sample.origin.position == origin
            sample_position += 1
    assert sample_position == len(index) - 23
    sample = WindowCollection(dataset, split, fold=0, partition="train")[99 - 11]
    assert sample.input_timestamps[-1] == "2017-03-12T01:55:00"
    assert sample.target_timestamps[0] == "2017-03-12T03:00:00"
    assert pd.Timestamp(sample.target_timestamps[-1]) - pd.Timestamp(sample.origin.timestamp) == (
        pd.Timedelta(minutes=120)
    )
    assert sample.feature_metadata["timezone"] is None and dataset.timestamps.tz is None
    pd.testing.assert_frame_equal(dataset.frame, before)
    with pytest.raises(DataError, match="cadence"):
        plan_splits(
            dataset.timestamps,
            replace(
                positional(), protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark")
            ),
        )
    diagnostics = temporal_diagnostics(dataset.timestamps, split)
    assert diagnostics["discontinuity_count"] == 1
    examples = diagnostics["examples"]
    assert isinstance(examples, list)
    edge = object_dict(examples[0])
    assert edge["after_row"] == 99 and edge["skipped_naive_grid_labels"] == 12
    assert edge["previous_interval_seconds"] == edge["following_interval_seconds"] == 300
    assert edge["affected_origin_ranges"] == [
        {
            "fold_id": "fold-0000",
            "partition": "train",
            "origin_start": 88,
            "origin_stop_exclusive": 111,
            "count": 23,
        }
    ]


@pytest.mark.parametrize("context", ["isolated", "operational"])
def test_strict_unchanged_and_availability(context: str) -> None:
    dataset = jump_dataset()
    configs = evaluation_configs(300)
    for name in ("chronological", "rolling"):
        cfg = configs[name]
        cfg = replace(cfg, protocol=replace(cfg.protocol, context=context))
        split = plan_splits(dataset.timestamps, cfg)
        for number, fold in enumerate(split.folds):
            assert fold.partition("train").rejected_gap_samples == 23
            assert fold.partition("calibration").stop <= fold.partition("test").start
            for part in fold.partitions:
                samples = WindowCollection(dataset, split, fold=number, partition=part.name)
                for i in range(len(samples)):
                    s = samples[i]
                    assert not (s.input_positions[0] <= 99 < s.target_positions[-1])
                    assert s.input_positions[-1] <= s.origin.position < s.target_positions[0]
            fitted = fit_standardizer(dataset, split, fold=number)
            assert fitted.lineage["fit_rows"] == [0, fold.partition("train").stop]
        with pytest.raises(DataError):
            cfg.features.validate_request("speed", "historical_observed", future=True)


def test_explicit_policy_and_bad_configuration() -> None:
    assert PlanConfig().protocol.name == "strict-rolling-origin-v1"
    assert evaluation_configs(300)["benchmark"].protocol.name == "dcrnn-sample-first-v1"
    assert config_from_dict(positional().to_dict()) == positional()
    for context in ("isolated", "operational"):
        with pytest.raises(DataError):
            EvaluationProtocol(POSITIONAL_PROTOCOL, context)
    with pytest.raises(DataError):
        evaluation_configs(300, benchmark_protocol="unknown")
    with pytest.raises(DataError):
        replace(positional(), features=FeatureAvailabilityPolicy(observed_delay_steps=1))
    with pytest.raises(DataError):
        plan_splits(pd.date_range("2017", periods=100, freq="5min", tz="UTC"), positional())
