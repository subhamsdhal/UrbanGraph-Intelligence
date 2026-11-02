from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import EvaluationProtocol, PlanConfig
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.dataloaders import initialize
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.models import TemporalForecaster
from urbangraph.neural.preprocessing import fit_preprocessing


def test_positional_discontinuity_is_not_filled_or_retimed(tmp_path: Path) -> None:
    labels = pd.date_range("2017-03-11 17:40", periods=300, freq="5min")
    index = pd.DatetimeIndex([t + pd.Timedelta(hours=int(i >= 100)) for i, t in enumerate(labels)])
    frame = pd.DataFrame({"z": np.arange(300, dtype=np.float64)}, index=index)
    dataset = EvaluationDataset(tmp_path, frame, {"fixture": "DST-like"})
    split = plan_splits(
        dataset.timestamps,
        PlanConfig(protocol=EvaluationProtocol("dcrnn-positional-v1", "benchmark")),
    )
    scaler = fit_preprocessing(dataset, split, 0, "global")
    adapter = NeuralWindows(dataset, split, 0, "train", scaler, calendar=True)
    position = int(np.flatnonzero(adapter.origins == 99)[0])
    sample = adapter[position]
    assert sample.origin == 99
    np.testing.assert_array_equal(
        adapter.raw_targets(adapter.origins[position : position + 1])[0, :, 0], np.arange(100, 112)
    )
    assert frame.index[100] - frame.index[99] == pd.Timedelta(minutes=65)
    expected_phase = (1 * 3600 + 55 * 60) / 86400 * 2 * np.pi
    assert sample.x[-1, 0, 2].item() == pytest.approx(np.sin(expected_phase))
    assert frame.shape == (300, 1)


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
def test_cpu_cuda_shape_and_finite_outputs(family: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable; CPU path covered separately")
    config = NeuralConfig(family=family, hidden=8, feedforward=16, device="cuda")
    initialize(config)
    cpu = TemporalForecaster(config, 3, 2, ("a", "b"), 2).eval()
    gpu = TemporalForecaster(config, 3, 2, ("a", "b"), 2).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.zeros(2, 3, 2, 2)
    x[..., 1] = 1
    x[:, :, 1, 1] = 0
    with torch.no_grad():
        expected, actual = cpu(x), gpu(x.cuda()).cpu()
    assert expected.shape == actual.shape == (2, 2, 2, 1)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("family", ["gru", "lstm", "transformer"])
def test_short_history_rejected(family: str) -> None:
    model = TemporalForecaster(NeuralConfig(family=family), 12, 12, ("a",), 2)
    with pytest.raises(DataError, match="aligned"):
        model(torch.zeros(1, 11, 1, 2))
