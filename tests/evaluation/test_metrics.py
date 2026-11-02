import math

import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import TargetValidityPolicy
from urbangraph.evaluation.metrics import (
    evaluate_predictions,
    mae,
    mape,
    masked_mae,
    masked_mape,
    masked_rmse,
    metric,
    rmse,
)


def test_independent_metric_values() -> None:
    actual = np.array([0, 2, -4, 10], dtype=np.float64)
    predicted = np.array([1, 4, -2, 14], dtype=np.float64)
    assert mae(actual, predicted).value == pytest.approx(2.25)
    assert rmse(actual, predicted).value == pytest.approx(math.sqrt(25 / 4))
    result = mape(actual, predicted)
    assert result.value == pytest.approx(100 * 1.9 / 3)
    assert (result.valid_count, result.masked_count, result.zero_denominator_count) == (3, 1, 1)
    with pytest.raises(DataError, match="zero denominators"):
        mape(actual, predicted, zero_policy="error")


def test_masks_never_score_excluded_targets() -> None:
    actual = np.array([np.nan, np.inf, -np.inf, 0, -2, 4], dtype=np.float64)
    predictions = np.array([np.nan, np.nan, np.nan, 3, -1, 2], dtype=np.float64)
    result = masked_mae(actual, predictions)
    assert result.value == 2 and result.valid_count == 3 and result.masked_count == 3
    assert masked_mae(actual, predictions, validity=TargetValidityPolicy(True)).value == 1.5
    assert masked_rmse(actual, predictions).value == pytest.approx(math.sqrt(14 / 3))
    assert masked_mape(actual, predictions).value == 50
    explicit = np.array([False, False, False, False, True, False])
    assert masked_mae(actual, predictions, mask=explicit).value == 1
    with pytest.raises(DataError, match="Nonfinite targets"):
        mae(actual, predictions)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_at_valid_target_errors(bad: float) -> None:
    with pytest.raises(DataError, match="prediction"):
        masked_mae(np.array([1.0]), np.array([bad]))


def test_empty_shape_mask_and_units_errors() -> None:
    one = np.array([1.0])
    with pytest.raises(DataError, match="Empty"):
        masked_mae(np.array([np.nan]), one)
    with pytest.raises(DataError, match="Empty"):
        mape(np.array([0.0]), one)
    with pytest.raises(DataError, match="shape"):
        masked_mae(one, one, mask=np.array([[True]]))
    with pytest.raises(DataError, match="Inverse"):
        metric("mae", one, one, value_space="standardized")
    with pytest.raises(DataError, match="shapes"):
        mae(one, np.ones(2))


def test_stable_accumulation_and_overflow() -> None:
    assert rmse(np.zeros(3), np.full(3, 1e200)).value == pytest.approx(1e200)
    assert mae(np.zeros(3), np.full(3, 1e308)).value == pytest.approx(1e308)
    with pytest.raises(DataError, match="overflow"):
        mae(np.array([-1e308]), np.array([1e308]))


def test_horizon_sensor_fold_partition_aggregation() -> None:
    truth = np.full((2, 2, 2, 1), 10.0)
    errors = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)[..., None]
    report = evaluate_predictions(
        truth,
        truth + errors,
        sensor_ids=("Z", "A"),
        protocol="strict-rolling-origin-v1",
        fold_ids=("f0", "f1"),
        partitions=("validation", "test"),
    )
    # Check JSON-compatible structure independently of dataclass representation.
    import json

    value = json.loads(json.dumps(report))
    assert value["overall"]["mae"]["value"] == 4.5
    assert [r["metrics"]["mae"]["value"] for r in value["by_horizon"]] == [3.5, 5.5]
    assert [r["metrics"]["mae"]["value"] for r in value["by_sensor"]] == [4, 5]
    assert [r["metrics"]["mae"]["value"] for r in value["by_fold"]] == [2.5, 6.5]
    assert {r["label"]: r["metrics"]["mae"]["value"] for r in value["by_partition"]} == {
        "validation": 2.5,
        "test": 6.5,
    }
    truth[:, :, 0] = np.nan
    masked = json.loads(
        json.dumps(
            evaluate_predictions(
                truth, truth + errors, sensor_ids=("Z", "A"), protocol="dcrnn-sample-first-v1"
            )
        )
    )
    assert masked["by_sensor"][0]["metrics"]["mae"]["status"] == "empty_population"
    assert masked["overall"]["mae"]["valid_count"] == 4
