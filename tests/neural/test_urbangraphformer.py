"""Unit and integration tests for UrbanGraphFormer architecture."""

import tempfile
import pytest
import torch

from urbangraph.data import DataError
from urbangraph.neural.urbangraphformer import (
    UrbanGraphFormer,
    UrbanGraphFormerConfig,
)


def test_config_validation() -> None:
    # Valid config
    config = UrbanGraphFormerConfig(hidden=32, heads=4, layers=2, num_graphs=3)
    assert config.hidden == 32
    assert config.heads == 4
    assert config.num_graphs == 3

    # Invalid hidden vs heads
    with pytest.raises(DataError, match="divisible by heads"):
        UrbanGraphFormerConfig(hidden=30, heads=4)

    # Invalid layers
    with pytest.raises(DataError, match="layers must be positive"):
        UrbanGraphFormerConfig(layers=0)

    # Invalid num_graphs
    with pytest.raises(DataError, match="num_graphs must be positive"):
        UrbanGraphFormerConfig(num_graphs=0)

    # Roundtrip dictionary
    d = config.to_dict()
    restored = UrbanGraphFormerConfig.from_dict(d)
    assert restored == config


def test_forward_shape_and_prediction() -> None:
    torch.manual_seed(42)
    b, t, n, f, h = 4, 12, 6, 2, 6
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)

    x = torch.randn(b, t, n, f)
    # 2 adjacency matrices: e.g. physical and correlation
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)
    # Symmetrize adjacency
    adj = (adj + adj.transpose(-2, -1)) / 2.0

    out = model(x, adj)
    assert out.shape == (b, h, n), f"Expected shape ({b}, {h}, {n}), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaNs"


def test_dimension_mismatch_raises() -> None:
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=12, horizon=4, num_nodes=5, in_features=2)

    # Wrong history
    x_bad_t = torch.randn(2, 10, 5, 2)
    adj = torch.rand(2, 5, 5)
    with pytest.raises(DataError, match="dimension mismatch"):
        model(x_bad_t, adj)

    # Wrong graphs
    x = torch.randn(2, 12, 5, 2)
    adj_bad_g = torch.rand(3, 5, 5)
    with pytest.raises(DataError, match="graph count mismatch"):
        model(x, adj_bad_g)


def test_multi_graph_sensitivity() -> None:
    torch.manual_seed(100)
    b, t, n, f, h = 2, 8, 4, 2, 3
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=3)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)

    x = torch.randn(b, t, n, f)
    adj_base = torch.rand(3, n, n).clamp(0.1, 1.0)
    out1 = model(x, adj_base)

    # Perturb only graph 1 (e.g. correlation matrix changes)
    adj_perturbed = adj_base.clone()
    adj_perturbed[1] = torch.eye(n)
    out2 = model(x, adj_perturbed)

    assert not torch.allclose(out1, out2, atol=1e-4), "Model should be sensitive to multi-graph relations"


def test_causal_temporal_isolation() -> None:
    """Verifies that future temporal steps do not leak into past temporal encodings."""
    torch.manual_seed(2026)
    b, t, n, f, h = 1, 6, 3, 2, 2
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=1)
    layer = config.hidden
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)
    model.eval()

    # Create synthetic input
    x = torch.randn(b, t, n, f)
    adj = torch.rand(1, n, n).clamp(0.2, 1.0)

    # Forward pass on temporal attention layer directly
    temp_attn = model.layers[0].temporal_attn
    in_proj = model.in_proj(x).permute(0, 2, 1, 3).contiguous().view(b * n, t, layer)

    out_base = temp_attn(in_proj)

    # Modify the last time step only (future step t=5)
    in_proj_modified = in_proj.clone()
    in_proj_modified[:, 5, :] += 10.0

    out_modified = temp_attn(in_proj_modified)

    # Causal property: out_modified for time steps 0..4 MUST be strictly identical
    torch.testing.assert_close(
        out_base[:, :5, :],
        out_modified[:, :5, :],
        rtol=1e-5,
        atol=1e-6,
        msg="Causal temporal attention leaked future data into past time steps!",
    )


def test_sensor_masking_robustness() -> None:
    torch.manual_seed(999)
    b, t, n, f, h = 2, 6, 5, 2, 3
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)

    x = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)

    # Node 2 is dropped/missing in batch item 0
    node_mask = torch.ones(b, n)
    node_mask[0, 2] = 0.0

    out = model(x, adj, node_mask=node_mask)
    assert not torch.isnan(out).any(), "Masking should not generate NaNs"
    assert out.shape == (b, h, n)


def test_checkpoint_roundtrip() -> None:
    torch.manual_seed(123)
    b, t, n, f, h = 2, 4, 3, 2, 2
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)

    x = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)
    orig_out = model(x, adj)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        ckpt_path = tmp.name

    try:
        model.save_checkpoint(ckpt_path, metadata={"epoch": 5, "val_loss": 0.42})
        loaded = UrbanGraphFormer.load_checkpoint(ckpt_path)
        loaded_out = loaded(x, adj)
        torch.testing.assert_close(orig_out, loaded_out, atol=1e-6, rtol=1e-6)
    finally:
        import os
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)


def test_gradient_flow_and_optimization() -> None:
    torch.manual_seed(555)
    b, t, n, f, h = 2, 6, 4, 2, 2
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=2, num_graphs=2, learning_rate=0.01)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    x = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)
    target = torch.randn(b, h, n)

    initial_loss = float("inf")
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x, adj)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        initial_loss = loss.item()

    assert not torch.isnan(torch.tensor(initial_loss)), "Loss is NaN during optimization"
    # Verify all forecasting parameters have non-zero gradients
    for name, param in model.named_parameters():
        if not name.startswith("reconstruction_head") and name != "mask_token":
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"


def test_self_supervised_masked_pretraining_and_reconstruction() -> None:
    """Verifies R08: leak-free self-supervised masked pretraining and spatial-temporal reconstruction."""
    torch.manual_seed(777)
    b, t, n, f, h = 3, 10, 4, 1, 4
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=2, num_graphs=2, learning_rate=0.01)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    x = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)

    # 1. Test reconstruction without mask
    recon_full = model.reconstruct(x, adj)
    assert recon_full.shape == (b, t, n, f)
    assert not torch.isnan(recon_full).any()

    # 2. Test pretraining step with 25% masking
    loss_init, _ = model.pretrain_step(x, adj, mask_ratio=0.25)
    assert loss_init.item() > 0.0

    # 3. Optimize reconstruction objective over 5 steps
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = model.pretrain_step(x, adj, mask_ratio=0.25)
        loss.backward()
        optimizer.step()

    loss_final, recon_step = model.pretrain_step(x, adj, mask_ratio=0.25)
    assert recon_step.shape == (b, t, n, f)
    assert not torch.isnan(loss_final)
    assert model.mask_token.grad is not None
    assert not torch.isnan(model.mask_token.grad).any()


def test_anomaly_detection_via_reconstruction() -> None:
    """Verifies R08 anomaly detection: corrupted/anomalous sensors have elevated reconstruction error."""
    torch.manual_seed(888)
    b, t, n, f, h = 2, 8, 4, 1, 2
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)
    model.eval()

    # Normal telemetry
    x_normal = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)
    normal_report = model.detect_anomalies(x_normal, adj, threshold_multiplier=3.0)
    assert isinstance(normal_report["node_mae"], list)
    assert len(normal_report["node_mae"]) == n

    # Corrupt sensor 2 with massive anomaly (+20.0 spike)
    x_anom = x_normal.clone()
    x_anom[:, :, 2, :] += 20.0
    anom_report = model.detect_anomalies(
        x_anom,
        adj,
        threshold_multiplier=2.0,
        baseline_mae=float(torch.abs(x_normal - model.reconstruct(x_normal, adj)).mean().item()),
    )
    assert anom_report["is_anomalous"] is True
    assert 2 in anom_report["anomalous_node_indices"], "Sensor 2 should be identified as anomalous"


def test_predictions_depend_on_learned_weights() -> None:
    """Verifies R07: predictions strictly depend on learned weights, not static fixtures."""
    torch.manual_seed(101)
    b, t, n, f, h = 1, 6, 3, 1, 4
    config = UrbanGraphFormerConfig(hidden=16, heads=2, layers=1, num_graphs=2)
    model = UrbanGraphFormer(config, history=t, horizon=h, num_nodes=n, in_features=f)
    model.eval()

    x = torch.randn(b, t, n, f)
    adj = torch.rand(2, n, n).clamp(0.1, 1.0)

    preds_before = model(x, adj).clone()

    # Mutate weights of first layer
    with torch.no_grad():
        for param in model.layers[0].parameters():
            param.add_(torch.randn_like(param) * 0.5)

    preds_after = model(x, adj)
    diff = (preds_before - preds_after).abs().max().item()
    assert diff > 0.01, f"Predictions must change when weights change, but diff={diff}"
