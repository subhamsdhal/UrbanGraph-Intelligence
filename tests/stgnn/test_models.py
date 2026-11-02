from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import PlanConfig
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.graphs.contracts import GraphArtifact, SensorNodeIndex
from urbangraph.graphs.normalization import sparse_edges
from urbangraph.neural.checkpointing import load_checkpoint
from urbangraph.neural.dataloaders import initialize
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.trainer import fit_model
from urbangraph.stgnn.contracts import GraphModelConfig, TrainingConfig
from urbangraph.stgnn.graph_batching import WeightedMessages, pyg_adjacency
from urbangraph.stgnn.models import CausalGate, GraphForecaster, diffuse


def graph() -> GraphArtifact:
    return GraphArtifact(
        Path("synthetic"),
        "synthetic",
        SensorNodeIndex(("a", "b", "c"), ("a", "b", "c")),
        np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]),
        {},
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_pyg_orientation_gradients_batch_isolation(device: str) -> None:
    initialize(TrainingConfig(threads=1))
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a = np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(pyg_adjacency(a).numpy(), a)
    index, weights = sparse_edges(a)
    x = torch.tensor(
        [[[2.0], [5.0], [7.0]], [[20.0], [50.0], [70.0]]], device=device, requires_grad=True
    )
    result = WeightedMessages().to(device)(
        x, torch.from_numpy(index).to(device), torch.from_numpy(weights).to(device).float()
    )
    expected = torch.tensor([[[0.0], [1.0], [10.0]], [[0.0], [10.0], [100.0]]], device=device)
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(
        x.grad, torch.tensor([[[0.5], [2.0], [0.0]], [[0.5], [2.0], [0.0]]], device=device)
    )
    # PyG gathers source->destination; row-operator diffusion gathers the reverse orientation.
    torch.testing.assert_close(
        diffuse(x, pyg_adjacency(a).to(device)),
        WeightedMessages().to(device)(
            x,
            torch.from_numpy(index[::-1].copy()).to(device),
            torch.from_numpy(weights).to(device).float(),
        ),
    )


@pytest.mark.parametrize("family", ["stgcn", "dcrnn", "graph-wavenet", "urbangraphformer"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_models_forward_backward_schema_and_batch(family: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    initialize(TrainingConfig(device=device, threads=1))
    config = GraphModelConfig(
        family=family,
        hidden=4,
        blocks=1,
        connectivity="static-adaptive" if family == "graph-wavenet" else "static",
    )
    model = GraphForecaster(config, graph(), 4, 3, graph().index.sensor_ids).to(device).eval()
    x = torch.randn(2, 4, 3, 2, device=device)
    x[..., 1] = 1
    x[:, :, 0] = 0  # all-missing normalized train-mean sensor input, explicit indicator zero
    result = model(x)
    assert result.shape == (2, 3, 3, 1) and torch.isfinite(result).all()
    changed = x.clone()
    changed[1, :, :, 0] += 10
    torch.testing.assert_close(result[0], model(changed)[0])
    result.square().mean().backward()
    if family == "urbangraphformer":
        # Strict check: all forecasting-active parameters (head, attention, encoder, graph fusion)
        # MUST receive valid, finite gradients.
        forecasting_params = [
            (name, p)
            for name, p in model.named_parameters()
            if not (name.startswith("transformer.mask_token") or name.startswith("transformer.reconstruction_head"))
        ]
        assert len(forecasting_params) > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in forecasting_params)
        # Pretraining-only parameters must remain completely inactive during pure forecasting
        pretrain_params = [
            (name, p)
            for name, p in model.named_parameters()
            if name.startswith("transformer.mask_token") or name.startswith("transformer.reconstruction_head")
        ]
        assert len(pretrain_params) == 5
        assert all(p.grad is None for _, p in pretrain_params)
    else:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(DataError, match="aligned"):
        model(x, sensor_ids=("b", "a", "c"))
    with pytest.raises(TypeError):
        model(x, future_targets=torch.zeros_like(result))
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(DataError, match="sanitized"):
        model(x)


def test_urbangraphformer_masked_pretraining_backward() -> None:
    """Verifies that during self-supervised masked pretraining:
    mask_token, reconstruction_head, and shared representation parameters
    all receive finite, non-zero gradients.
    """
    initialize(TrainingConfig(device="cpu", threads=1))
    config = GraphModelConfig(
        family="urbangraphformer",
        hidden=4,
        blocks=1,
        connectivity="static",
    )
    model = GraphForecaster(config, graph(), 4, 3, graph().index.sensor_ids).to("cpu")
    x = torch.randn(2, 4, 3, 2)
    adj = torch.stack(model.supports(), dim=0)

    # 25% masking with guaranteed masked entry
    mask = (torch.rand(2, 4, 3) > 0.75).float()
    mask[0, 1, 0] = 1.0

    recon = model.transformer.reconstruct(x, adj, mask=mask)
    assert recon.shape == x.shape and torch.isfinite(recon).all()

    # Pretraining reconstruction loss computed over masked positions
    loss = (recon - x).pow(2).mul(mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1.0)
    loss.backward()

    # 1. mask_token must receive finite, non-zero gradient
    assert model.transformer.mask_token.grad is not None
    assert torch.isfinite(model.transformer.mask_token.grad).all()
    assert float(model.transformer.mask_token.grad.abs().sum()) > 0.0

    # 2. reconstruction_head parameters must all receive finite, non-zero gradients
    recon_params = list(model.transformer.reconstruction_head.parameters())
    assert len(recon_params) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in recon_params)
    assert all(float(p.grad.abs().sum()) > 0.0 for p in recon_params)

    # 3. shared representation parameters must all receive finite, non-zero gradients
    shared_params = [
        p for name, p in model.transformer.named_parameters()
        if "reconstruction_head" not in name and "mask_token" not in name and "head" not in name
    ]
    assert len(shared_params) > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in shared_params)
    assert all(float(p.grad.abs().sum()) > 0.0 for p in shared_params)


def test_causal_filter_and_adaptive_adjacency() -> None:
    initialize(TrainingConfig(threads=1))
    gate = CausalGate(4, 2)
    x = torch.randn(2, 12, 3, 4)
    modified = x.clone()
    modified[:, 7:] += 100
    torch.testing.assert_close(gate(x)[:, :7], gate(modified)[:, :7])
    model = GraphForecaster(
        GraphModelConfig(family="graph-wavenet", connectivity="adaptive", hidden=4),
        graph(),
        12,
        12,
        graph().index.sensor_ids,
    )
    a = model.adaptive_adjacency()
    torch.testing.assert_close(a.sum(-1), torch.ones(3))
    assert (a > 0).all()
    encoded = torch.randn(2, 12, 3, 2)
    changed = encoded.clone()
    changed[:, 7:] += 100
    torch.testing.assert_close(model.encode(encoded)[:, :7], model.encode(changed)[:, :7])
    assert model.receptive_field == 4


def test_static_equivariance_and_identity_control() -> None:
    initialize(TrainingConfig(threads=1))
    config = GraphModelConfig(hidden=4, blocks=1)
    original = GraphForecaster(config, graph(), 4, 3, graph().index.sensor_ids).eval()
    permutation = [2, 0, 1]
    ids = tuple(graph().index.sensor_ids[i] for i in permutation)
    reordered = replace(
        graph(),
        index=SensorNodeIndex(ids, ids),
        adjacency=graph().adjacency[np.ix_(permutation, permutation)],
    )
    other = GraphForecaster(config, reordered, 4, 3, ids).eval()
    state = original.state_dict()
    state["forward_walk"] = state["forward_walk"][permutation][:, permutation]
    state["reverse_walk"] = state["reverse_walk"][permutation][:, permutation]
    other.load_state_dict(state)
    x = torch.randn(2, 4, 3, 2)
    torch.testing.assert_close(other(x[:, :, permutation]), original(x)[:, :, permutation])
    isolated = GraphForecaster(
        replace(config, perturbation="identity"), graph(), 4, 3, graph().index.sensor_ids
    ).eval()
    y = isolated(x)
    x[:, :, 1] += 100
    torch.testing.assert_close(y[:, :, 0], isolated(x)[:, :, 0])


@pytest.mark.parametrize("family", ["stgcn", "dcrnn", "graph-wavenet", "urbangraphformer"])
def test_shared_training_and_graph_checkpoint(tmp_path: Path, family: str) -> None:
    frame = pd.DataFrame(
        np.arange(900).reshape(300, 3) % 17,
        columns=["a", "b", "c"],
        index=pd.date_range("2024-01-01", periods=300, freq="5min"),
    )
    dataset = EvaluationDataset(
        tmp_path, frame, {"dataset_id": "synthetic", "canonical_id": "fixture"}
    )
    split = plan_splits(dataset.timestamps, PlanConfig())
    scaler = fit_preprocessing(dataset, split, 0, "global")
    train = NeuralWindows(dataset, split, 0, "train", scaler, maximum=8)
    validation = NeuralWindows(dataset, split, 0, "validation", scaler, maximum=4)
    config = TrainingConfig(epochs=1, threads=1, batch_size=4)
    model_config = GraphModelConfig(family=family, hidden=4, blocks=1)

    def factory() -> GraphForecaster:
        return GraphForecaster(
            model_config, graph(), train.history, train.horizon, train.sensor_ids
        )

    lineage: dict[str, object] = {"graph_id": graph().graph_id, "model": model_config.to_dict()}
    result = fit_model(
        train, validation, config, tmp_path / "epochs", lineage, model_factory=factory
    )
    state, metadata = load_checkpoint(result.checkpoint)
    assert metadata["lineage"] == lineage and state["best_epoch"] == 1
    resumed = fit_model(
        train,
        validation,
        config,
        tmp_path / "epochs",
        lineage,
        model_factory=factory,
        resume=result.checkpoint,
    )
    x = train[0].x.unsqueeze(0)
    torch.testing.assert_close(result.model(x), resumed.model(x), atol=0, rtol=0)
    with pytest.raises(DataError, match="mismatch"):
        fit_model(
            train,
            validation,
            config,
            tmp_path / "epochs",
            {"graph_id": "wrong"},
            model_factory=factory,
            resume=result.checkpoint,
        )
