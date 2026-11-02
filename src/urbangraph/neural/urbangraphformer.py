"""UrbanGraphFormer: Dynamic Multi-Graph Spatio-Temporal Cross-Attention Transformer.

Implements the original architecture specification:
- Causal temporal multi-head attention with positional encoding.
- Multi-relational spatial cross-attention over multiple graph topologies (geographic, correlation, topological).
- Dynamic relation fusion gating with learnable importance across graph channels.
- Missing sensor masking and causal temporal boundary enforcement.
- Deterministic initialization, checkpoint serialization, and prediction head.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from urbangraph.data import DataError


@dataclass(frozen=True)
class UrbanGraphFormerConfig:
    version: str = "urbangraphformer-1.0"
    hidden: int = 32
    heads: int = 4
    layers: int = 2
    num_graphs: int = 3
    feedforward: int = 64
    dropout: float = 0.0
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.version != "urbangraphformer-1.0":
            raise DataError(f"Unsupported UrbanGraphFormer version: {self.version}")
        if self.hidden <= 0 or self.hidden % self.heads != 0:
            raise DataError("hidden must be positive and divisible by heads")
        if self.layers <= 0:
            raise DataError("layers must be positive")
        if self.num_graphs <= 0:
            raise DataError("num_graphs must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise DataError("dropout must be in [0.0, 1.0)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UrbanGraphFormerConfig:
        return cls(
            version=str(data.get("version", "urbangraphformer-1.0")),
            hidden=int(data.get("hidden", 32)),
            heads=int(data.get("heads", 4)),
            layers=int(data.get("layers", 2)),
            num_graphs=int(data.get("num_graphs", 3)),
            feedforward=int(data.get("feedforward", 64)),
            dropout=float(data.get("dropout", 0.0)),
            learning_rate=float(data.get("learning_rate", 0.001)),
            weight_decay=float(data.get("weight_decay", 0.0001)),
            device=str(data.get("device", "cpu")),
        )


class CausalTemporalAttention(nn.Module):
    """Causal multi-head self-attention across time history."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B * N, T, D)
        t_len = x.size(1)
        # Causal mask: upper triangular elements are -inf (cannot attend to future)
        causal_mask = torch.triu(
            torch.full((t_len, t_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        attn_out, _ = self.mha(x, x, x, attn_mask=causal_mask, need_weights=False)
        return cast(Tensor, self.norm(x + self.dropout(attn_out)))


class DynamicMultiGraphSpatialCrossAttention(nn.Module):
    """Spatial cross-attention modulating attention scores with multiple graph topologies."""

    def __init__(self, hidden: int, heads: int, num_graphs: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.num_graphs = num_graphs

        # Separate projections for each graph relation
        self.q_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_graphs)])
        self.k_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_graphs)])
        self.v_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(num_graphs)])

        # Learnable relation bias scaling parameters
        self.relation_scales = nn.Parameter(torch.ones(num_graphs))

        # Dynamic channel-wise relation fusion gate: computes importance of each graph per node
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_graphs),
        )

        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        adjacencies: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """
        x: (B, N, D)
        adjacencies: (G, N, N) where G == num_graphs
        node_mask: (B, N) bool/float mask (1 = valid, 0 = missing)
        """
        b, n, d = x.shape
        g = adjacencies.shape[0]
        if g != self.num_graphs:
            raise DataError(f"Adjacency graph count mismatch: expected {self.num_graphs}, got {g}")

        # Compute dynamic relation weights per node: (B, N, G)
        gate_logits = self.fusion_gate(x)  # (B, N, G)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, N, G)

        relation_outputs: list[Tensor] = []

        for rel_idx in range(self.num_graphs):
            # Projections: (B, N, D) -> (B, H, N, D_k)
            q = (
                self.q_proj[rel_idx](x)
                .view(b, n, self.heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                self.k_proj[rel_idx](x)
                .view(b, n, self.heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj[rel_idx](x)
                .view(b, n, self.heads, self.head_dim)
                .transpose(1, 2)
            )

            # Dot-product attention scores: (B, H, N, N)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Modulate with topological adjacency relation
            adj = adjacencies[rel_idx].clamp(min=1e-6, max=1.0)
            adj_bias = torch.log(adj) * self.relation_scales[rel_idx]  # (N, N)
            scores = scores + adj_bias.unsqueeze(0).unsqueeze(0)

            # Apply node availability mask if provided
            if node_mask is not None:
                # node_mask: (B, N) -> mask j where node j is missing: (B, 1, 1, N)
                mask_val = (node_mask == 0).unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask_val, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            # Handle potential NaNs if all keys are masked
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            # Aggregate values: (B, H, N, D_k) -> (B, N, D)
            rel_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
            relation_outputs.append(rel_out)

        # Stack relation outputs: (B, N, G, D)
        stacked = torch.stack(relation_outputs, dim=2)

        # Dynamic fusion: weight each relation by gate_weights (B, N, G, 1)
        fused = torch.sum(stacked * gate_weights.unsqueeze(-1), dim=2)  # (B, N, D)

        out = self.out_proj(fused)
        return cast(Tensor, self.norm(x + self.dropout(out)))


class UrbanGraphFormerLayer(nn.Module):
    """Full Spatio-Temporal Cross-Attention Layer."""

    def __init__(self, config: UrbanGraphFormerConfig) -> None:
        super().__init__()
        self.temporal_attn = CausalTemporalAttention(
            hidden=config.hidden,
            heads=config.heads,
            dropout=config.dropout,
        )
        self.spatial_cross_attn = DynamicMultiGraphSpatialCrossAttention(
            hidden=config.hidden,
            heads=config.heads,
            num_graphs=config.num_graphs,
            dropout=config.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, config.hidden),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden)

    def forward(
        self,
        x: Tensor,
        adjacencies: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """
        x: (B, T, N, D)
        adjacencies: (G, N, N)
        node_mask: (B, N)
        """
        b, t, n, d = x.shape

        # 1. Causal Temporal Self-Attention per node
        # Reshape to (B * N, T, D)
        x_temp = x.permute(0, 2, 1, 3).contiguous().view(b * n, t, d)
        x_temp = self.temporal_attn(x_temp)
        x = x_temp.view(b, n, t, d).permute(0, 2, 1, 3).contiguous()  # (B, T, N, D)

        # 2. Dynamic Multi-Graph Spatial Cross-Attention per time step
        # (B * T, N, D)
        x_spatial = x.view(b * t, n, d)
        spatial_mask = (
            node_mask.unsqueeze(1).expand(-1, t, -1).contiguous().view(b * t, n)
            if node_mask is not None
            else None
        )
        x_spatial = self.spatial_cross_attn(x_spatial, adjacencies, spatial_mask)
        x = x_spatial.view(b, t, n, d)

        # 3. Feedforward Network
        x = self.ffn_norm(x + self.ffn(x))
        return x


class UrbanGraphFormer(nn.Module):
    """Authoritative UrbanGraphFormer model implementation."""

    def __init__(
        self,
        config: UrbanGraphFormerConfig,
        history: int,
        horizon: int,
        num_nodes: int,
        in_features: int,
    ) -> None:
        super().__init__()
        if history < 1 or horizon < 1 or num_nodes < 1 or in_features < 1:
            raise DataError("Invalid model temporal/sensor dimensions")

        self.config = config
        self.history = history
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.in_features = in_features

        # Input feature projection
        self.in_proj = nn.Linear(in_features, config.hidden)

        # Temporal positional encodings
        positions = torch.arange(history, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, config.hidden, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.hidden)
        )
        encoding = torch.zeros(history, config.hidden)
        encoding[:, 0::2] = torch.sin(positions * scales)
        encoding[:, 1::2] = torch.cos(positions * scales)
        self.register_buffer("pos_encoding", encoding.unsqueeze(0).unsqueeze(2))  # (1, T, 1, D)

        # Stack of Spatio-Temporal Layers
        self.layers = nn.ModuleList([UrbanGraphFormerLayer(config) for _ in range(config.layers)])

        # Multi-horizon forecast projection head: projects history representation to horizon
        self.head = nn.Sequential(
            nn.Linear(history * config.hidden, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, horizon),
        )

        # Self-supervised spatial-temporal reconstruction head (R08)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, config.hidden))
        nn.init.normal_(self.mask_token, std=0.02)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward, in_features),
        )

    def encode(
        self,
        x: Tensor,
        adjacencies: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode spatio-temporal history into node representations.

        x: (B, T, N, F)
        adjacencies: (G, N, N)
        node_mask: (B, N) or None

        Returns: (B, T, N, D)
        """
        b, t, n, f = x.shape
        if t != self.history or n != self.num_nodes or f != self.in_features:
            raise DataError(
                f"Input tensor dimension mismatch: expected (*, {self.history}, {self.num_nodes}, {self.in_features}), "
                f"got ({b}, {t}, {n}, {f})"
            )

        h = self.in_proj(x) + self.pos_encoding
        for layer in self.layers:
            h = layer(h, adjacencies, node_mask)
        return h

    def forward(
        self,
        x: Tensor,
        adjacencies: Tensor,
        node_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass for multi-horizon forecasting.

        Returns: (B, H, N)
        """
        b, t, n, _ = x.shape
        h = self.encode(x, adjacencies, node_mask)
        h_flat = h.permute(0, 2, 1, 3).contiguous().view(b, n, t * self.config.hidden)
        out = self.head(h_flat).transpose(1, 2)
        return cast(Tensor, out)

    def reconstruct(
        self,
        x: Tensor,
        adjacencies: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Self-supervised masked reconstruction across spatio-temporal grid.

        x: (B, T, N, F)
        adjacencies: (G, N, N)
        mask: (B, T, N) boolean or binary float where 1 = masked out, 0 = observed.

        Returns: (B, T, N, F) reconstructed signals.
        """
        b, t, n, f = x.shape
        if t != self.history or n != self.num_nodes or f != self.in_features:
            raise DataError(
                f"Input tensor dimension mismatch: expected (*, {self.history}, {self.num_nodes}, {self.in_features}), "
                f"got ({b}, {t}, {n}, {f})"
            )

        h_feat = self.in_proj(x)
        if mask is not None:
            m_exp = mask.unsqueeze(-1).to(dtype=h_feat.dtype, device=h_feat.device)
            h_feat = h_feat * (1.0 - m_exp) + self.mask_token * m_exp

        h = h_feat + self.pos_encoding
        for layer in self.layers:
            h = layer(h, adjacencies)

        reconstructed = self.reconstruction_head(h)
        return cast(Tensor, reconstructed)

    def pretrain_step(
        self,
        x: Tensor,
        adjacencies: Tensor,
        mask_ratio: float = 0.25,
    ) -> tuple[Tensor, Tensor]:
        """Perform one leak-free self-supervised masked pretraining step (R08).

        Returns: (reconstruction_loss, reconstructed_x)
        """
        b, t, n, _ = x.shape
        mask = (torch.rand(b, t, n, device=x.device) < mask_ratio).float()
        if mask.sum() == 0:
            mask[:, -1, :] = 1.0

        recon = self.reconstruct(x, adjacencies, mask=mask)
        m_exp = mask.unsqueeze(-1)
        masked_diff = torch.abs(recon - x) * m_exp
        loss = masked_diff.sum() / (m_exp.sum() * x.size(-1) + 1e-6)
        return loss, recon

    def detect_anomalies(
        self,
        x: Tensor,
        adjacencies: Tensor,
        threshold_multiplier: float = 2.0,
        baseline_mae: float | None = None,
    ) -> dict[str, Any]:
        """Detect telemetry anomalies and dropouts via reconstruction error."""
        with torch.no_grad():
            recon = self.reconstruct(x, adjacencies, mask=None)
            residuals = torch.abs(x - recon).mean(dim=-1)
            node_mae = residuals.mean(dim=(0, 1))

            expected_mae = baseline_mae if baseline_mae is not None else float(residuals.mean().item())
            cutoff = max(1e-4, expected_mae * threshold_multiplier)
            anomalous_nodes = [
                idx for idx, val in enumerate(node_mae) if val.item() > cutoff
            ]

            return {
                "residuals": residuals.cpu().numpy().tolist(),
                "node_mae": node_mae.cpu().numpy().tolist(),
                "baseline_mae": expected_mae,
                "cutoff_threshold": cutoff,
                "anomalous_node_indices": anomalous_nodes,
                "is_anomalous": len(anomalous_nodes) > 0,
            }

    def save_checkpoint(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """Save deterministic checkpoint."""
        state = {
            "config": self.config.to_dict(),
            "history": self.history,
            "horizon": self.horizon,
            "num_nodes": self.num_nodes,
            "in_features": self.in_features,
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(state, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> UrbanGraphFormer:
        """Load checkpoint deterministically."""
        state = torch.load(path, map_location=device, weights_only=False)
        config = UrbanGraphFormerConfig.from_dict(state["config"])
        model = cls(
            config=config,
            history=state["history"],
            horizon=state["horizon"],
            num_nodes=state["num_nodes"],
            in_features=state["in_features"],
        )
        model.load_state_dict(state["state_dict"], strict=False)
        model.to(device)
        return model
