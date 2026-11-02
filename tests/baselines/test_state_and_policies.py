from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from urbangraph.baselines.contracts import ModelConfig
from urbangraph.baselines.history import make_batch, training_data
from urbangraph.baselines.models import Baseline, FittedState, fit_model
from urbangraph.baselines.runner import select_ar
from urbangraph.baselines.state import decode_array, encode_array
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import code_provenance
from urbangraph.evaluation.compatibility import compatible_provenance
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import numerical_values

from .test_models import config, dataset


def test_fitted_state_lossless_roundtrip_and_tamper() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    fitted = fit_model(ModelConfig(model="historical-average"), training_data(data, split, 0), 12)
    assert fitted.state is not None
    restored = FittedState.from_dict(fitted.state.to_dict())
    assert restored.state_id == fitted.state.state_id
    np.testing.assert_array_equal(restored.parameters, fitted.state.parameters)
    encoded = encode_array(np.array([1.0, 2.0]))
    encoded["sha256"] = "0" * 64
    with pytest.raises(DataError, match="checksum"):
        decode_array(encoded)
    encoded["shape"] = [1000000000]
    with pytest.raises(DataError, match="Oversized"):
        decode_array(encoded)


def test_validation_selection_independent_of_calibration_and_test() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    first, selection = select_ar(data, split, 0, ModelConfig(model="autoregressive", lags=3), 128)
    data.frame.iloc[2800:] = -123456.0
    second, again = select_ar(data, split, 0, ModelConfig(model="autoregressive", lags=3), 128)
    assert first.state is not None and second.state is not None
    assert first.state.state_id == second.state.state_id
    assert selection == again


def test_drift_bounds_and_insufficient_trend_fallback() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    bounded = Baseline(ModelConfig(model="drift", bounds=(0, 100))).predict(batch)
    assert np.all(bounded.values == 100)
    no_trend = Baseline(ModelConfig(model="drift", history_fallback=1)).predict(batch)
    assert no_trend.fallback.all()


def test_unseen_average_slots_explicit_fallback() -> None:
    data = dataset()
    split = plan_splits(data.timestamps, config())
    training = training_data(data, split, 0)
    shortened = replace(training, values=training.values[:20], timestamps=training.timestamps[:20])
    fitted = fit_model(ModelConfig(model="historical-average"), shortened, 12)
    batch = make_batch(data, split, 0, "test", np.array([3200]), numerical_values(data.frame))
    assert fitted.predict(batch).fallback.all()


def test_exact_ug003s_compatibility_only() -> None:
    old = read_json(Path(__file__).parents[1] / "evaluation/fixtures/ug003s-code.json")
    recorded_dependencies = object_dict(old["dependencies"])
    candidate = {**code_provenance(), "dependencies": recorded_dependencies}
    assert compatible_provenance(old, candidate, "dcrnn-positional-v1")
    changed = dict(old)
    changed["source_sha256"] = {"unknown.py": "0" * 64}
    assert not compatible_provenance(changed, candidate, "dcrnn-positional-v1")
    changed_dependencies = {**recorded_dependencies, "hdf5": "unknown"}
    assert not compatible_provenance(
        old, {**candidate, "dependencies": changed_dependencies}, "dcrnn-positional-v1"
    )
