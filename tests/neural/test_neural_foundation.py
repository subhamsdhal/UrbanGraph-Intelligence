from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import (
    FeatureAvailabilityPolicy,
    ForecastWindowSpecification,
    PlanConfig,
)
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import WindowCollection
from urbangraph.neural.checkpointing import load_checkpoint
from urbangraph.neural.contracts import NeuralConfig, config_from_dict
from urbangraph.neural.dataloaders import initialize
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.losses import masked_loss
from urbangraph.neural.models import TemporalForecaster
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.trainer import fit


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
def test_forward_gradient_and_sensor_independence(family: str) -> None:
    config = NeuralConfig(family=family, hidden=8, feedforward=16)
    initialize(config)
    model = TemporalForecaster(config, 3, 2, ("z", "a"), 2).eval()
    x = torch.randn(2, 3, 2, 2)
    x[..., 1] = 1
    original = model(x)
    assert original.shape == (2, 2, 2, 1)
    changed = x.clone()
    changed[:, :, 1, 0] += 100
    torch.testing.assert_close(model(changed)[:, :, 0], original[:, :, 0])
    x[..., 1] = 0
    missing = model(x)
    assert torch.isfinite(missing).all()
    torch.autograd.backward(missing.square().mean())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(DataError, match="aligned"):
        model(x, sensor_ids=("a", "z"))
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(DataError, match="sanitized"):
        model(x)


@pytest.mark.parametrize("kind", ["mae", "mse", "huber"])
def test_invalid_labels_do_not_poison_gradients(kind: str) -> None:
    prediction = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    target = torch.tensor([0.0, float("nan"), float("inf")])
    mask = torch.tensor([True, True, False])
    loss = masked_loss(prediction, target, mask, kind)
    torch.autograd.backward(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0] != 0
    assert prediction.grad[1:].tolist() == [0.0, 0.0]
    with pytest.raises(DataError, match="All-invalid"):
        masked_loss(prediction, target, torch.zeros(3, dtype=torch.bool), kind)


@pytest.mark.parametrize(
    "value",
    [
        {"hidden": 0},
        {"seed": True},
        {"workers": 1},
        {"device": "auto"},
        {"unknown": 1},
        {"precision": "float16"},
        {"dropout": float("nan")},
    ],
)
def test_invalid_configuration(value: dict[str, object]) -> None:
    with pytest.raises(DataError):
        config_from_dict(value)


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
def test_training_checkpoint_and_leakage(tmp_path: Path, family: str) -> None:
    values = np.arange(100, dtype=np.float64)
    frame = pd.DataFrame(
        {"z": values, "a": 2 * values}, index=pd.date_range("2024-01-01", periods=100, freq="5min")
    )
    frame.iloc[4, 0] = np.nan
    data = EvaluationDataset(tmp_path, frame, {"fixture": "neural-test"})
    plan = PlanConfig(
        window=ForecastWindowSpecification(3, 2),
        minimum_train=20,
        validation_length=10,
        calibration_length=10,
        test_length=10,
        stride=10,
        folds=2,
    )
    split = plan_splits(data.timestamps, plan)
    scaler = fit_preprocessing(data, split, 0, "global")
    modified = frame.copy()
    modified.iloc[20:] = 1e9
    assert (
        fit_preprocessing(replace(data, frame=modified), split, 0, "global").to_dict()
        == scaler.to_dict()
    )
    train = NeuralWindows(data, split, 0, "train", scaler)
    validation = NeuralWindows(data, split, 0, "validation", scaler)
    assert torch.isfinite(train[0].x).all()
    assert train.sensor_ids == ("z", "a")
    config = NeuralConfig(
        family=family,
        hidden=8,
        feedforward=16,
        epochs=2,
        batch_size=4,
        threads=1,
        trial_seconds=120,
    )
    result = fit(train, validation, config, tmp_path / "checkpoints", {"fixture": True})
    restored = TemporalForecaster(config, 3, 2, train.sensor_ids, 2)
    state, _ = load_checkpoint(result.checkpoint)
    restored.load_state_dict(state["best_model"])  # type: ignore[arg-type]
    restored.eval()
    sample = train[0].x.unsqueeze(0)
    torch.testing.assert_close(restored(sample), result.model(sample), rtol=0, atol=0)
    resumed = fit(
        train,
        validation,
        config,
        tmp_path / "checkpoints",
        {"fixture": True},
        resume=result.checkpoint,
    )
    torch.testing.assert_close(resumed.model(sample), result.model(sample), rtol=0, atol=0)
    continued = fit(
        train,
        validation,
        config,
        tmp_path / "continued",
        {"fixture": True},
        resume=tmp_path / "checkpoints" / "epoch-0001",
    )
    torch.testing.assert_close(continued.model(sample), result.model(sample), rtol=0, atol=0)
    with (result.checkpoint / "state.pt").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(DataError, match="checksum"):
        load_checkpoint(result.checkpoint)


@pytest.mark.parametrize("delay", [0, 2])
@pytest.mark.parametrize("calendar", [False, True])
def test_window_parity_and_causal_fill(tmp_path: Path, delay: int, calendar: bool) -> None:
    values = np.arange(100, dtype=np.float64)
    frame = pd.DataFrame(
        {"z": values, "a": values * 2}, index=pd.date_range("2024-01-01", periods=100, freq="5min")
    )
    frame.iloc[1, 0] = np.nan
    data = EvaluationDataset(tmp_path, frame, {"fixture": "parity"})
    config = PlanConfig(
        window=ForecastWindowSpecification(3, 2),
        minimum_train=20,
        validation_length=12,
        calibration_length=12,
        test_length=12,
        stride=12,
        folds=2,
        features=FeatureAvailabilityPolicy(observed_delay_steps=delay),
    )
    split = plan_splits(data.timestamps, config)
    scaler = fit_preprocessing(data, split, 0, "per_sensor")
    adapter = NeuralWindows(data, split, 0, "train", scaler, calendar=calendar)
    windows = WindowCollection(data, split, fold=0, partition="train")
    assert len(adapter) == len(windows)
    for i in range(len(adapter)):
        sample, reference = adapter[i], windows[i]
        assert sample.origin == reference.origin.position
        np.testing.assert_array_equal(sample.mask.numpy(), reference.target_valid)
        np.testing.assert_array_equal(
            adapter.raw_targets(adapter.origins[i : i + 1])[0, ..., None], reference.y
        )
        assert sample.x.shape[-1] == (4 if calendar else 2)
    first = adapter[0]
    assert first.x[0, 0, 1] == 1  # observed zero is valid
    assert first.x[1, 0, 1] == 0  # missing indicator
    assert first.x[1, 0, 0] == first.x[0, 0, 0]  # causal carry, no next-row interpolation
    with pytest.raises(DataError, match="Only train/validation"):
        NeuralWindows(data, split, 0, "test", scaler, maximum=2)
