from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.baselines.contracts import ModelConfig, model_config
from urbangraph.baselines.history import HistoricalAccess, make_batch, training_data
from urbangraph.baselines.models import Baseline, fit_model
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import (
    EvaluationProtocol,
    FeatureAvailabilityPolicy,
    PlanConfig,
    TargetValidityPolicy,
)
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import WindowCollection, numerical_values


def dataset(rows: int = 4000) -> EvaluationDataset:
    values = np.arange(rows, dtype=np.float64)
    return EvaluationDataset(
        Path("."),
        pd.DataFrame(
            {"Z": values, "001": 2 * values},
            index=pd.date_range("2024-01-01", periods=rows, freq="5min"),
        ),
        {"test": True},
    )


def config(context: str = "operational", delay: int = 0) -> PlanConfig:
    return PlanConfig(
        protocol=EvaluationProtocol(context=context),
        minimum_train=2500,
        validation_length=300,
        calibration_length=300,
        test_length=500,
        folds=1,
        features=FeatureAvailabilityPolicy(observed_delay_steps=delay),
    )


@pytest.mark.parametrize("delay", [0, 2])
@pytest.mark.parametrize("context", ["isolated", "operational"])
def test_persistence_drift_and_existing_window_parity(delay: int, context: str) -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config(context, delay))
    origin = split.folds[0].partition("test").samples[0].start
    batch = make_batch(
        data, split, 0, "test", np.array([origin, origin + 1]), numerical_values(data.frame)
    )
    original = WindowCollection(data, split, fold=0, partition="test")[0]
    np.testing.assert_array_equal(batch.x[0], original.x)
    assert batch.sensor_ids == ("Z", "001")
    persistence = Baseline(ModelConfig()).predict(batch)
    assert persistence.values.shape == (2, 12, 2, 1)
    np.testing.assert_array_equal(
        persistence.values[:, :, 0, 0],
        np.broadcast_to(np.array([origin - delay, origin + 1 - delay])[:, None], (2, 12)),
    )
    drift = Baseline(ModelConfig(model="drift")).predict(batch)
    np.testing.assert_allclose(
        drift.values[..., 0], numerical_values(data.frame)[batch.target_positions]
    )


@pytest.mark.parametrize("model,lag", [("daily-seasonal", 288), ("weekly-seasonal", 2016)])
@pytest.mark.parametrize("policy", ["positional", "timestamp-label"])
def test_seasonal(model: str, lag: int, policy: str) -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    batch = make_batch(data, split, 0, "test", np.array([3200, 3201]), numerical_values(data.frame))
    pred = Baseline(ModelConfig(model=model, seasonal_policy=policy)).predict(batch)
    np.testing.assert_array_equal(
        pred.values[..., 0], numerical_values(data.frame)[batch.target_positions - lag]
    )
    assert not pred.fallback.any()


def test_dst_label_matching_differs_and_never_repairs() -> None:
    data = dataset()
    index = data.timestamps.to_numpy().copy()
    index[3000:] += np.timedelta64(60, "m")
    data.frame.index = pd.DatetimeIndex(index)
    split = plan_splits(data.timestamps, config())
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    positional = Baseline(ModelConfig(model="daily-seasonal")).predict(batch)
    labels = Baseline(ModelConfig(model="daily-seasonal", seasonal_policy="timestamp-label"))
    matched = labels.predict(batch)
    np.testing.assert_array_equal(
        matched.values - positional.values,
        np.broadcast_to(np.array([12, 24])[None, None, :, None], matched.values.shape),
    )
    assert data.timestamps.tz is None
    np.testing.assert_array_equal(data.timestamps.to_numpy(), index)


def test_unavailable_history_context_and_fallback() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config("isolated"))
    batch = make_batch(data, split, 0, "test", np.array([3120]), numerical_values(data.frame))
    pred = Baseline(ModelConfig(model="weekly-seasonal")).predict(batch)
    assert pred.fallback.all()
    missing = Baseline(ModelConfig(model="weekly-seasonal", fallback="unavailable")).predict(batch)
    assert not missing.available.any()
    assert np.isnan(missing.values).all()


def test_future_access_delay_lower_bound_and_copies() -> None:
    data = dataset(30)
    values = numerical_values(data.frame)
    access = HistoricalAccess(
        values, data.timestamps, np.array([20]), delay=2, lower=10, validity=TargetValidityPolicy()
    )
    with pytest.raises(DataError, match="Future"):
        access.retrieve(np.array([[19]]))
    assert np.isnan(access.retrieve(np.array([[9]]))).all()
    past = access.retrieve(np.array([[18]]))
    past[:] = -999
    assert values[18, 0] == 18


@pytest.mark.parametrize("mask_zero", [False, True])
def test_zero_missing_and_last_valid(mask_zero: bool) -> None:
    data = dataset()
    data.frame.iloc[3199, 0] = 0
    data.frame.iloc[3200, 0] = np.nan
    split = plan_splits(
        data.timestamps, replace(config(), validity=TargetValidityPolicy(mask_zero))
    )
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    result = Baseline(ModelConfig()).predict(batch)
    assert result.values[0, 0, 0, 0] == (3198 if mask_zero else 0)
    assert result.fallback[0, :, 0].all()
    assert not result.fallback[0, :, 1].any()
    unavailable = Baseline(ModelConfig(history_fallback=1)).predict(batch)
    assert not unavailable.available[0, :, 0].any()


@pytest.mark.parametrize("model", ["historical-average", "autoregressive"])
def test_train_only_fitted_parameters_and_sensor_order(model: str) -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    cfg = ModelConfig(model=model, average="global", lags=3)
    first = fit_model(cfg, training_data(data, split, 0), 12)
    assert first.state is not None
    data.frame.iloc[2500:] = 9e6
    second = fit_model(cfg, training_data(data, split, 0), 12)
    assert second.state is not None
    assert first.state.state_id == second.state.state_id
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    with pytest.raises(DataError, match="sensor"):
        first.predict(replace(batch, sensor_ids=("001", "Z")))


@pytest.mark.parametrize("mode", ["global", "time-of-day", "time-of-week"])
def test_historical_average_known_and_unseen(mode: str) -> None:
    data = dataset()
    data.frame[:] = 7.0
    split = plan_splits(data.timestamps, config())
    model = fit_model(
        ModelConfig(model="historical-average", average=mode), training_data(data, split, 0), 12
    )
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    assert np.all(model.predict(batch).values == 7)


def test_direct_ar_linear_signal_and_missing_fit() -> None:
    data = dataset()
    data.frame.iloc[100:110, 0] = np.nan
    split = plan_splits(data.timestamps, config())
    model = fit_model(
        ModelConfig(model="autoregressive", lags=3), training_data(data, split, 0), 12
    )
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    np.testing.assert_allclose(
        model.predict(batch).values[..., 0],
        numerical_values(data.frame)[batch.target_positions],
        atol=1e-5,
    )
    assert model.state is not None
    assert model.state.support[0] < model.state.support[1]


@pytest.mark.parametrize(
    "bad",
    [
        {"model": "neural"},
        {"unknown": 1},
        {"ridge": 0},
        {"ridge": True},
        {"lags": 0},
        {"lags": True},
        {"fallback": "groundtruth"},
        {"seasonal_policy": "UTC"},
        {"bounds": [1, 0]},
        {"schema_version": "2"},
    ],
)
def test_configuration_rejection(bad: dict[str, object]) -> None:
    with pytest.raises(DataError):
        model_config(bad)


def test_no_fit_and_required_fit() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    with pytest.raises(DataError, match="No-fit"):
        fit_model(ModelConfig(), training_data(data, split, 0), 12)
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    with pytest.raises(DataError, match="requires"):
        Baseline(ModelConfig(model="historical-average")).predict(batch)


def test_benchmark_fit_scope_and_future_perturbation() -> None:
    data = dataset()
    split = plan_splits(
        data.timestamps, PlanConfig(protocol=EvaluationProtocol("dcrnn-positional-v1", "benchmark"))
    )
    training = training_data(data, split, 0)
    assert len(training.values) == split.folds[0].partition("train").samples[-1].stop
    batch = make_batch(data, split, 0, "test", np.array([3500]), numerical_values(data.frame))
    first = Baseline(ModelConfig(model="weekly-seasonal")).predict(batch)
    data.frame.iloc[3501:] = -9999
    later = make_batch(data, split, 0, "test", np.array([3500]), numerical_values(data.frame))
    np.testing.assert_array_equal(
        first.values, Baseline(ModelConfig(model="weekly-seasonal")).predict(later).values
    )
