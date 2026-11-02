import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError
from urbangraph.evaluation.artifacts import publish_plan, verify_plan
from urbangraph.evaluation.contracts import EvaluationProtocol, PartitionBoundary, PlanConfig
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits, resolve_boundary
from urbangraph.evaluation.preprocessing import FittedStandardizer, fit_standardizer
from urbangraph.evaluation.windows import WindowCollection


def test_no_calibration_and_overlapping_folds(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    split = plan_splits(dataset.timestamps, replace(config, calibration_length=0, stride=1))
    assert [p.name for p in split.folds[0].partitions] == ["train", "validation", "test"]
    left, right = (f.partition("test") for f in split.folds)
    assert left.start < right.start < left.stop  # legal repetition across distinct folds.
    for fold in split.folds:
        assert fold.partition("train").stop <= fold.partition("test").start


def test_embargo_cannot_be_ignored(dataset: EvaluationDataset, config: PlanConfig) -> None:
    config = replace(
        config,
        protocol=EvaluationProtocol("strict-chronological-v1"),
        embargo=2,
        boundaries=(
            PartitionBoundary("train", 0, 20),
            PartitionBoundary("validation", 21, 40),
            PartitionBoundary("test", 42, 60),
        ),
    )
    with pytest.raises(DataError, match="embargo"):
        plan_splits(dataset.timestamps, config)


@pytest.mark.parametrize("boundary", ["2024-01-01T00:00:01", "2024-01-01T00:00:00Z", "not-a-date"])
def test_boundary_requires_exact_time_and_awareness(
    dataset: EvaluationDataset, boundary: str
) -> None:
    with pytest.raises(DataError):
        resolve_boundary(dataset.timestamps, boundary, 300)


@pytest.mark.parametrize("defect", ["duplicate", "reverse", "nat"])
def test_structural_temporal_errors(
    dataset: EvaluationDataset, config: PlanConfig, defect: str
) -> None:
    times: list[pd.Timestamp | None] = list(dataset.timestamps)
    if defect == "duplicate":
        times[3] = times[2]
    elif defect == "reverse":
        times.reverse()
    else:
        times[3] = None
    with pytest.raises(DataError, match="timestamps"):
        plan_splits(pd.DatetimeIndex(times), config)


def test_all_cross_gap_candidates_are_rejected(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    index = pd.date_range("2024", periods=100, freq="10min")
    with pytest.raises(DataError, match="no usable"):
        plan_splits(index, config)


def test_window_masks_preserve_original_semantics(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    frame = dataset.frame.copy()
    frame.iloc[3, 0] = np.nan
    frame.iloc[3, 1] = np.inf
    frame.iloc[4, 0] = -np.inf
    frame.iloc[4, 1] = 0
    data = replace(dataset, frame=frame)
    split = plan_splits(data.timestamps, config)
    sample = WindowCollection(data, split, fold=0, partition="train")[0]
    np.testing.assert_array_equal(sample.target_valid[..., 0], [[False, False], [False, True]])
    assert np.isnan(sample.y[0, 0, 0]) and np.isposinf(sample.y[0, 1, 0])
    assert np.isneginf(sample.y[1, 0, 0]) and sample.y[1, 1, 0] == 0


def test_fitted_state_roundtrip_and_invalid_parameters(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    state = fit_standardizer(dataset, plan_splits(dataset.timestamps, config), fold=0)
    payload = json.loads(json.dumps(state.to_dict()))
    assert FittedStandardizer.from_dict(payload) == state
    for value in (0, -1, float("nan"), float("inf")):
        changed = json.loads(json.dumps(payload))
        changed["scales"][0] = value
        with pytest.raises(DataError):
            FittedStandardizer.from_dict(changed)


def test_plan_identity_changes_with_policy(
    canonical_dir: Path, raw_snapshot_dir: Path, config: PlanConfig, tmp_path: Path
) -> None:
    original = publish_plan(
        canonical_dir, config, output_root=tmp_path / "plans", raw_snapshot_dir=raw_snapshot_dir
    )
    changed = publish_plan(
        canonical_dir,
        replace(config, scaling="global"),
        output_root=tmp_path / "plans",
        raw_snapshot_dir=raw_snapshot_dir,
    )
    assert original.plan_id != changed.plan_id
    assert verify_plan(
        original.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    ).valid
    assert verify_plan(
        changed.directory, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    ).valid
