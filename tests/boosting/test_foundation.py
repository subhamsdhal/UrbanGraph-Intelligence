import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.baselines.history import HistoricalAccess
from urbangraph.boosting.contracts import BoostConfig, FeatureSpec, config_from_dict
from urbangraph.boosting.features import build_features
from urbangraph.boosting.models import BudgetExceeded, load_model, train_model
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import TargetValidityPolicy


def test_configuration_roundtrip() -> None:
    config = BoostConfig()
    assert config_from_dict(config.to_dict()) == config
    with pytest.raises(DataError):
        config_from_dict({"threads": True})
    with pytest.raises(DataError):
        config_from_dict({"unknown": 1})


@pytest.mark.parametrize("variant", ["F0", "F1", "F2", "F3", "F4"])
def test_features_causal_zero_preserving(variant: str) -> None:
    values = np.arange(8000, dtype=np.float64).reshape(4000, 2)
    values[3000, 0] = 0
    labels = pd.date_range("2024-01-01", periods=4000, freq="5min")
    origins = np.array([3000, 3000], dtype=np.int64)
    sensors = np.array([0, 1], dtype=np.int64)
    horizons = np.array([1, 12], dtype=np.int64)

    def features() -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
        history = HistoricalAccess(
            values, labels, origins, delay=0, lower=0, validity=TargetValidityPolicy()
        )
        return build_features(
            FeatureSpec(variant=variant),
            history,
            sensors,
            horizons,
            labels[origins + horizons],
            ("A", "B"),
            0,
        ).values

    before = features()
    assert before[0, 0] == 0
    values[3001:] = -100000
    np.testing.assert_array_equal(features(), before)


@pytest.mark.parametrize("family", ["lightgbm", "xgboost"])
def test_native_train_reload(tmp_path: Path, family: str) -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 3)).astype(np.float32)
    x[:, 2] = np.arange(200) % 2
    x[::13, 0] = np.nan
    y = np.nan_to_num(x[:, 0]).astype(np.float64) * 2 + x[:, 2]
    config = BoostConfig(
        family=family,
        trees=5,
        threads=1,
        min_child_samples=2,
        min_child_weight=1,
        variants=("F0", "F4"),
    )
    names, groups = ("lag_1", "horizon", "sensor_id"), ("lags", "horizon", "sensor")
    model, info = train_model(
        config,
        FeatureSpec(),
        names,
        groups,
        ("A", "B"),
        12,
        x[:150],
        y[:150],
        x[150:],
        y[150:],
        time.perf_counter() + 60,
    )
    predicted = model.predict(x, names, ("A", "B"))
    path = tmp_path / ("model.txt" if family == "lightgbm" else "model.json")
    model.save(path)
    restored = load_model(
        path, config, FeatureSpec(), names, groups, ("A", "B"), 12, model.best_iteration
    )
    np.testing.assert_allclose(restored.predict(x, names, ("A", "B")), predicted, atol=1e-6)
    assert info["best_iteration"] == model.best_iteration
    with pytest.raises(DataError):
        model.save(path)
    with pytest.raises(DataError):
        model.predict(x, names, ("B", "A"))
    with pytest.raises(BudgetExceeded):
        train_model(
            replace(config, trees=1),
            FeatureSpec(),
            names,
            groups,
            ("A", "B"),
            12,
            x[:150],
            y[:150],
            x[150:],
            y[150:],
            time.perf_counter() - 1,
        )


@pytest.mark.parametrize("delay", [0, 2])
def test_selected_history_guards(delay: int) -> None:
    values = np.arange(40, dtype=np.float64).reshape(20, 2)
    labels = pd.date_range("2024-01-01", periods=20, freq="5min")
    origins = np.array([10], dtype=np.int64)
    sensors = np.array([0], dtype=np.int64)
    access = HistoricalAccess(
        values, labels, origins, delay=delay, lower=4, validity=TargetValidityPolicy()
    )
    with pytest.raises(DataError, match="unavailable"):
        access.retrieve_sensors(np.array([[11 - delay]], dtype=np.int64), sensors)
    assert np.isnan(access.retrieve_sensors(np.array([[3]], dtype=np.int64), sensors)).all()
    with pytest.raises(DataError):
        access.retrieve_sensors(np.array([[4]], dtype=np.int64), np.array([-1], dtype=np.int64))
    features = build_features(
        FeatureSpec(lags=(1,), rolling=(1,)),
        access,
        sensors,
        np.array([1], dtype=np.int64),
        labels[[11]],
        ("A", "B"),
        delay,
    )
    assert features.values[0, 0] == values[10 - delay, 0]


@pytest.mark.parametrize("policy", ["positional", "timestamp-label"])
def test_seasonal_missing_and_context(policy: str) -> None:
    values = np.ones((100, 1), dtype=np.float64)
    labels = pd.date_range("2024-01-01", periods=100, freq="5min")
    origins = np.array([50], dtype=np.int64)
    access = HistoricalAccess(
        values, labels, origins, delay=0, lower=40, validity=TargetValidityPolicy()
    )
    features = build_features(
        FeatureSpec(seasonal_policy=policy),
        access,
        np.array([0], dtype=np.int64),
        np.array([12], dtype=np.int64),
        labels[[62]],
        ("A",),
        0,
    )
    assert features.unavailable_seasonal.all()
    assert np.isnan(features.values[0, features.names.index("seasonal_1d")])
    assert features.values[0, features.names.index("seasonal_1d_missing")] == 1


def test_dst_policy_and_rolling_values() -> None:
    values = np.arange(4000, dtype=np.float64)[:, None]
    labels = pd.date_range("2024-01-01", periods=4000, freq="5min")
    shifted = labels.to_numpy().copy()
    shifted[3000:] += np.timedelta64(60, "m")
    labels = pd.DatetimeIndex(shifted)
    origin = np.array([3050], dtype=np.int64)
    access = HistoricalAccess(
        values, labels, origin, delay=0, lower=0, validity=TargetValidityPolicy()
    )

    def features(
        policy: str,
    ) -> tuple[tuple[str, ...], np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
        row = build_features(
            FeatureSpec(seasonal_policy=policy),
            access,
            np.array([0], dtype=np.int64),
            np.array([12], dtype=np.int64),
            labels[[3062]],
            ("S",),
            0,
        )
        return row.names, row.values

    names, positional = features("positional")
    _, timestamp = features("timestamp-label")
    assert positional[0, names.index("seasonal_1d")] == 3062 - 288
    assert timestamp[0, names.index("seasonal_1d")] == 3062 - 288 + 12
    assert positional[0, names.index("rolling_3_mean")] == 3049
    assert positional[0, names.index("rolling_3_std")] == pytest.approx(np.std([3048, 3049, 3050]))
    assert positional[0, names.index("difference_1")] == 1
    assert positional[0, names.index("horizon")] == 12
