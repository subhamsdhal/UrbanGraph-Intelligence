from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import ForecastWindowSpecification, PlanConfig
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.neural import checkpointing
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.dataloaders import initialize
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.losses import masked_loss
from urbangraph.neural.models import TemporalForecaster
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.trainer import BudgetExceeded, fit


def inputs(tmp_path: Path) -> tuple[NeuralWindows, NeuralWindows]:
    values = np.arange(100, dtype=np.float64)
    frame = pd.DataFrame({"a": values}, index=pd.date_range("2024-01-01", periods=100, freq="5min"))
    dataset = EvaluationDataset(tmp_path, frame, {"fixture": "safety"})
    split = plan_splits(
        dataset.timestamps,
        PlanConfig(
            window=ForecastWindowSpecification(3, 2),
            minimum_train=20,
            validation_length=10,
            calibration_length=10,
            test_length=10,
            stride=10,
            folds=2,
        ),
    )
    scaler = fit_preprocessing(dataset, split, 0, "global")
    return (
        NeuralWindows(dataset, split, 0, "train", scaler),
        NeuralWindows(dataset, split, 0, "validation", scaler),
    )


def test_expired_budget_never_publishes_checkpoint(tmp_path: Path) -> None:
    train, validation = inputs(tmp_path)
    with pytest.raises(BudgetExceeded):
        fit(
            train,
            validation,
            NeuralConfig(epochs=1, threads=1),
            tmp_path / "epochs",
            {},
            deadline=0,
        )
    assert not list(tmp_path.rglob("checkpoint.json"))


def test_failed_publication_cleans_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, validation = inputs(tmp_path)

    def fail(stage: Path, destination: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(checkpointing, "_publish", fail)
    root = tmp_path / "epochs"
    with pytest.raises(OSError, match="simulated"):
        fit(train, validation, NeuralConfig(epochs=1, threads=1), root, {})
    assert not list(root.glob(".staging-*"))
    assert not list(root.glob("epoch-*"))


def test_resume_rejects_changed_config_and_lineage(tmp_path: Path) -> None:
    train, validation = inputs(tmp_path)
    config = NeuralConfig(epochs=1, threads=1, hidden=8)
    result = fit(train, validation, config, tmp_path / "epochs", {"source": "a"})
    cases: tuple[tuple[NeuralConfig, dict[str, object]], ...] = (
        (replace(config, seed=1), {"source": "a"}),
        (config, {"source": "b"}),
    )
    for altered, lineage in cases:
        with pytest.raises(DataError, match="mismatch"):
            fit(train, validation, altered, tmp_path / "epochs", lineage, resume=result.checkpoint)
    with pytest.raises(DataError, match="overwrite"):
        fit(train, validation, config, tmp_path / "epochs", {"source": "a"})


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
def test_gradient_accumulation_matches_large_batch(tmp_path: Path, family: str) -> None:
    train, validation = inputs(tmp_path)
    base = NeuralConfig(
        family=family,
        hidden=8,
        feedforward=16,
        epochs=1,
        threads=1,
        batch_size=4,
        accumulation=1,
        dropout=0,
        residual=False,
    )
    full = fit(train, validation, base, tmp_path / "full", {})
    accumulated = fit(
        train, validation, replace(base, batch_size=2, accumulation=2), tmp_path / "accumulated", {}
    )
    for key, value in full.model.state_dict().items():
        actual = accumulated.model.state_dict()[key]
        if key.endswith("self_attn.in_proj_bias"):
            # Key-bias is a softmax-invariant offset. Its exact gradient is zero;
            # float32 roundoff is amplified by Adam near its epsilon. Check the
            # identifiable query/value parameters at the original tolerance.
            indices = torch.cat((torch.arange(8), torch.arange(16, 24)))
            torch.testing.assert_close(value[indices], actual[indices], rtol=1e-4, atol=1e-5)
        else:
            torch.testing.assert_close(value, actual, rtol=1e-4, atol=1e-5)
    samples = torch.stack([train[i].x for i in range(len(train))])
    torch.testing.assert_close(
        full.model(samples), accumulated.model(samples), rtol=1e-4, atol=1e-5
    )


def test_transformer_accumulated_gradients_including_null_key_bias() -> None:
    config = NeuralConfig(family="transformer", hidden=8, feedforward=16, residual=False, threads=1)
    initialize(config)
    full = TemporalForecaster(config, 3, 2, ("a",), 2).double()
    accumulated = TemporalForecaster(config, 3, 2, ("a",), 2).double()
    accumulated.load_state_dict(full.state_dict())
    x = torch.randn(4, 3, 1, 2, dtype=torch.float64)
    x[..., 1] = 1
    y = torch.randn(4, 2, 1, 1, dtype=torch.float64)
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[0, 0] = False  # unequal valid counts exercise weighted accumulation
    count = int(valid.sum())
    torch.autograd.backward(masked_loss(full(x), y, valid))
    for start in (0, 2):
        mask = valid[start : start + 2]
        loss = masked_loss(accumulated(x[start : start + 2]), y[start : start + 2], mask)
        torch.autograd.backward(loss * int(mask.sum()) / count)
    for (name, parameter), (_, other) in zip(
        full.named_parameters(), accumulated.named_parameters(), strict=True
    ):
        assert parameter.grad is not None and other.grad is not None
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-12, atol=1e-14)
        if name.endswith("self_attn.in_proj_bias"):
            torch.testing.assert_close(
                parameter.grad[8:16], torch.zeros(8, dtype=torch.float64), rtol=0, atol=1e-14
            )


def test_early_stopping_and_scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from urbangraph.neural import trainer

    train, validation = inputs(tmp_path)

    def constant_validation(
        model: TemporalForecaster, dataset: NeuralWindows, config: NeuralConfig
    ) -> dict[str, object]:
        return {"mae": {"value": 1.0}}

    monkeypatch.setattr(trainer, "validate", constant_validation)
    config = NeuralConfig(epochs=10, patience=2, threads=1, hidden=8)
    result = fit(train, validation, config, tmp_path / "epochs", {})
    assert result.report["best_epoch"] == 1
    history = result.report["history"]
    assert isinstance(history, list)
    assert len(history) == 3
    assert history[-1]["learning_rate"] == config.learning_rate / 2
    assert len(list((tmp_path / "epochs").glob("epoch-*"))) == 3


def test_seed_reproducibility_and_variation(tmp_path: Path) -> None:
    train, validation = inputs(tmp_path)
    config = NeuralConfig(epochs=1, threads=1, hidden=8)
    first = fit(train, validation, config, tmp_path / "first", {})
    second = fit(train, validation, config, tmp_path / "second", {})
    changed = fit(train, validation, replace(config, seed=2027), tmp_path / "changed", {})
    x = train[0].x.unsqueeze(0)
    torch.testing.assert_close(first.model(x), second.model(x), rtol=0, atol=0)
    assert not torch.equal(first.model(x), changed.model(x))
