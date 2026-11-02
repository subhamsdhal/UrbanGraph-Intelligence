import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import (
    EvaluationProtocol,
    FeatureAvailabilityPolicy,
    PlanConfig,
    TargetValidityPolicy,
)
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.preprocessing import fit_standardizer
from urbangraph.evaluation.windows import WindowCollection, numerical_values, validity_mask


@pytest.mark.parametrize("context", ["isolated", "operational"])
def test_no_future_inputs_and_disjoint_labels(
    dataset: EvaluationDataset, config: PlanConfig, context: str
) -> None:
    cfg = replace(config, protocol=EvaluationProtocol(context=context), embargo=2)
    split = plan_splits(dataset.timestamps, cfg)
    for fold_number, fold in enumerate(split.folds):
        target_sets: list[set[int]] = []
        for partition in fold.partitions:
            targets: set[int] = set()
            windows = WindowCollection(dataset, split, fold=fold_number, partition=partition.name)
            for i in range(len(windows)):
                sample = windows[i]
                assert max(sample.input_positions) <= sample.origin.latest_available_position
                assert min(sample.target_positions) > sample.origin.position
                assert min(sample.target_positions) >= partition.start
                assert max(sample.target_positions) < partition.stop
                if context == "isolated":
                    assert min(sample.input_positions) >= partition.start
                targets.update(sample.target_positions)
            for previous in target_sets:
                assert previous.isdisjoint(targets)
            target_sets.append(targets)


@pytest.mark.parametrize("partition", ["validation", "calibration", "test"])
def test_future_distribution_cannot_affect_fit(
    dataset: EvaluationDataset, config: PlanConfig, partition: str
) -> None:
    split = plan_splits(dataset.timestamps, config)
    fitted = fit_standardizer(dataset, split, fold=0)
    boundary = split.folds[0].partition(partition)
    changed = dataset.frame.copy(deep=True)
    changed.iloc[boundary.start : boundary.stop] = 1e12
    modified = replace(dataset, frame=changed)
    again = fit_standardizer(modified, split, fold=0)
    assert again.to_dict() == fitted.to_dict()
    assert fitted.means == pytest.approx((9.5, 20.0))
    assert fitted.valid_counts == (20, 20)


def test_each_fold_fits_its_own_train_rows(dataset: EvaluationDataset, config: PlanConfig) -> None:
    split = plan_splits(dataset.timestamps, config)
    first = fit_standardizer(dataset, split, fold=0)
    second = fit_standardizer(dataset, split, fold=1)
    assert first.means == pytest.approx((9.5, 20))
    assert second.means == pytest.approx((14.5, 30))
    assert first.lineage["fit_rows"] == [0, 20]
    assert second.lineage["fit_rows"] == [0, 30]
    frame = dataset.frame.copy()
    frame.iloc[20:30] += 1000  # prior-fold validation, now legitimate training history.
    changed = replace(dataset, frame=frame)
    assert fit_standardizer(changed, split, fold=0).means == first.means
    assert fit_standardizer(changed, split, fold=1).means != second.means


@pytest.mark.parametrize("mode", ["global", "per_sensor"])
def test_scaler_roundtrip_and_no_refit(
    dataset: EvaluationDataset, config: PlanConfig, mode: str
) -> None:
    split = plan_splits(dataset.timestamps, replace(config, scaling=mode))
    scaler = fit_standardizer(dataset, split, fold=0)
    before = scaler.to_dict()
    original = dataset.frame.copy(deep=True)
    values = numerical_values(dataset.frame.iloc[40:50])
    transformed = scaler.transform(values, sensor_ids=dataset.sensor_ids)
    restored = scaler.inverse_transform(transformed, sensor_ids=dataset.sensor_ids)
    np.testing.assert_allclose(restored, values, rtol=1e-14)
    assert scaler.to_dict() == before
    pd.testing.assert_frame_equal(dataset.frame, original)
    with pytest.raises(DataError, match="aligned"):
        scaler.transform(values, sensor_ids=tuple(reversed(dataset.sensor_ids)))


def test_fit_has_no_implicit_full_dataset_default(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    signature = inspect.signature(fit_standardizer)
    assert signature.parameters["split"].default is inspect.Parameter.empty
    assert signature.parameters["fold"].default is inspect.Parameter.empty
    with pytest.raises(DataError, match="configured"):
        fit_standardizer(
            dataset, plan_splits(dataset.timestamps, replace(config, scaling="none")), fold=0
        )
    split = plan_splits(dataset.timestamps, config)
    train = split.folds[0].partition("train")
    fabricated = replace(
        split,
        folds=(
            replace(
                split.folds[0],
                partitions=(replace(train, stop=100), *split.folds[0].partitions[1:]),
            ),
            *split.folds[1:],
        ),
    )
    with pytest.raises(DataError, match="match"):
        fit_standardizer(dataset, fabricated, fold=0)


@pytest.mark.parametrize(
    ("feature", "category", "future"),
    [
        ("speed", "historical_observed", True),
        ("weather", "unknown_future", True),
        ("speed", "target", True),
        ("weather", "known_future", True),
        ("latitude", "static", False),
    ],
)
def test_unavailable_feature_rejected(feature: str, category: str, future: bool) -> None:
    with pytest.raises(DataError, match="Unavailable"):
        FeatureAvailabilityPolicy().validate_request(feature, category, future=future)


def test_valid_feature_requests() -> None:
    policy = FeatureAvailabilityPolicy()
    policy.validate_request("speed", "historical_observed", future=False)
    policy.validate_request("time_of_day", "known_future", future=True)


def test_all_invalid_constant_and_zero_policy(
    dataset: EvaluationDataset, config: PlanConfig
) -> None:
    frame = dataset.frame.copy()
    frame.iloc[:20, 0] = 0
    frame.iloc[:20, 1] = 4
    frame.iloc[0, 1] = np.inf
    data = replace(dataset, frame=frame)
    split = plan_splits(data.timestamps, config)
    fitted = fit_standardizer(data, split, fold=0)
    assert fitted.means == (0.0, 4.0) and fitted.scales == (1.0, 1.0)
    assert fitted.valid_counts == (20, 19)
    zero_masked = plan_splits(
        data.timestamps, replace(config, validity=TargetValidityPolicy(mask_zero=True))
    )
    with pytest.raises(DataError, match="All-invalid"):
        fit_standardizer(data, zero_masked, fold=0)
    values = np.array([np.nan, np.inf, -np.inf, 0, -1, 2], dtype=np.float64)
    np.testing.assert_array_equal(
        validity_mask(values, TargetValidityPolicy()), [False, False, False, True, True, True]
    )
    np.testing.assert_array_equal(
        validity_mask(values, TargetValidityPolicy(True)), [False, False, False, False, True, True]
    )


def test_benchmark_fit_scope_is_input_union(dataset: EvaluationDataset) -> None:
    config = PlanConfig(
        protocol=EvaluationProtocol("dcrnn-sample-first-v1", "benchmark"), scaling="per_sensor"
    )
    split = plan_splits(dataset.timestamps, config)
    fitted = fit_standardizer(dataset, split, fold=0)
    assert fitted.lineage["fit_rows"] == [0, 65]
    assert fitted.means == pytest.approx((32, 65))
    frame = dataset.frame.copy()
    frame.iloc[65:] = 1e8
    assert fit_standardizer(replace(dataset, frame=frame), split, fold=0).means == fitted.means


def test_large_integer_precision_is_explicit() -> None:
    with pytest.raises(DataError, match="precision"):
        numerical_values(pd.DataFrame({"sensor": [2**60, 2**60 + 1]}))
