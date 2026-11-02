"""Recognizable compute-bounded graph architectures; no future-label inputs."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from urbangraph.data import DataError
from urbangraph.graphs.contracts import GraphArtifact
from urbangraph.graphs.normalization import normalize, perturb
from urbangraph.stgnn.contracts import GraphModelConfig
from urbangraph.stgnn.graph_batching import pyg_adjacency


def diffuse(x: Tensor, support: Tensor) -> Tensor:
    """Row-oriented diffusion: node i aggregates support[i,j] * x[j]."""
    return torch.matmul(support, x)


class Diffusion(nn.Module):
    def __init__(self, inputs: int, outputs: int, depth: int, supports: int) -> None:
        super().__init__()
        self.depth = depth
        self.linear = nn.Linear(inputs * (1 + depth * supports), outputs)

    def forward(self, x: Tensor, supports: list[Tensor]) -> Tensor:
        terms = [x]
        for support in supports:
            current = x
            for _ in range(self.depth):
                current = diffuse(current, support)
                terms.append(current)
        return cast(Tensor, self.linear(torch.cat(terms, dim=-1)))


class CausalGate(nn.Module):
    def __init__(self, width: int, dilation: int, kernel: int = 2) -> None:
        super().__init__()
        self.left = dilation * (kernel - 1)
        self.conv = nn.Conv2d(width, 2 * width, (kernel, 1), dilation=(dilation, 1))

    def forward(self, x: Tensor) -> Tensor:
        # [B,T,N,C] -> [B,C,T,N]; left temporal padding only.
        z = self.conv(F.pad(x.permute(0, 3, 1, 2), (0, 0, self.left, 0)))
        value, gate = z.chunk(2, dim=1)
        return (torch.tanh(value) * torch.sigmoid(gate)).permute(0, 2, 3, 1)


class DiffusionCell(nn.Module):
    def __init__(self, inputs: int, hidden: int, depth: int) -> None:
        super().__init__()
        self.gates = Diffusion(inputs + hidden, 2 * hidden, depth, 2)
        self.candidate = Diffusion(inputs + hidden, hidden, depth, 2)

    def forward(self, x: Tensor, state: Tensor, supports: list[Tensor]) -> Tensor:
        reset, update = torch.sigmoid(self.gates(torch.cat([x, state], dim=-1), supports)).chunk(
            2, -1
        )
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * state], dim=-1), supports))
        return cast(Tensor, update * state + (1 - update) * candidate)


class GraphForecaster(nn.Module):
    def __init__(
        self,
        config: GraphModelConfig,
        graph: GraphArtifact,
        history: int,
        horizon: int,
        sensor_ids: tuple[str, ...],
        features: int = 2,
    ) -> None:
        super().__init__()
        if (
            graph.index.sensor_ids != sensor_ids
            or history < 1
            or horizon < 1
            or features not in {2, 4}
        ):
            raise DataError("Graph/model temporal or sensor alignment mismatch")
        self.config, self.graph_id = config, graph.graph_id
        self.history, self.horizon, self.sensor_ids, self.features = (
            history,
            horizon,
            sensor_ids,
            features,
        )
        adjacency = np.array(graph.adjacency, copy=True)
        if config.perturbation != "none":
            adjacency = perturb(adjacency, config.perturbation, seed=config.perturbation_seed)
        # Validate PyG orientation before constructing dense operators. Dense diffusion
        # avoids nondeterministic CUDA scatter reductions for these bounded networks.
        pyg_adjacency(adjacency)
        self.register_buffer("forward_walk", pyg_adjacency(normalize(adjacency, "random-walk")))
        self.register_buffer("reverse_walk", pyg_adjacency(normalize(adjacency, "reverse-walk")))
        width = config.hidden
        self.node_source: nn.Parameter | None = None
        self.node_destination: nn.Parameter | None = None
        if config.family == "urbangraphformer":
            from urbangraph.neural.urbangraphformer import UrbanGraphFormer, UrbanGraphFormerConfig

            supports_count = (
                2
                if config.connectivity == "static"
                else 1
                if config.connectivity == "adaptive"
                else 3
            )
            n_heads = 4 if config.hidden % 4 == 0 else (2 if config.hidden % 2 == 0 else 1)
            self.transformer = UrbanGraphFormer(
                UrbanGraphFormerConfig(
                    hidden=config.hidden,
                    heads=n_heads,
                    layers=config.blocks,
                    num_graphs=supports_count,
                    dropout=config.dropout,
                ),
                history=history,
                horizon=horizon,
                num_nodes=len(sensor_ids),
                in_features=features,
            )
            self.dilations: tuple[int, ...] = ()
            self.receptive_field = history
            if config.connectivity != "static":
                self.node_source = nn.Parameter(
                    torch.randn(len(sensor_ids), config.embedding_dim) * 0.1
                )
                self.node_destination = nn.Parameter(
                    torch.randn(config.embedding_dim, len(sensor_ids)) * 0.1
                )
        else:
            self.projection = nn.Linear(features, width)
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(width, horizon)
            self.norm = nn.LayerNorm(width)
            self.cells = nn.ModuleList()
            self.temporal = nn.ModuleList()
            self.spatial = nn.ModuleList()
            self.after = nn.ModuleList()
            self.skips = nn.ModuleList()
            if config.family == "dcrnn":
                for _ in range(config.blocks):
                    self.cells.append(DiffusionCell(width, width, config.diffusion_depth))
                self.dilations: tuple[int, ...] = ()
                self.receptive_field = history
            else:
                self.dilations = tuple(
                    2**i if config.family == "graph-wavenet" else 1 for i in range(config.blocks)
                )
                self.receptive_field = 1 + sum(self.dilations) * (
                    1 if config.family == "graph-wavenet" else 4
                )
                supports = (
                    2
                    if config.connectivity == "static"
                    else 1
                    if config.connectivity == "adaptive"
                    else 3
                )
                for dilation in self.dilations:
                    self.temporal.append(
                        CausalGate(width, dilation, 3 if config.family == "stgcn" else 2)
                    )
                    self.spatial.append(Diffusion(width, width, config.diffusion_depth, supports))
                    self.after.append(
                        CausalGate(width, 1, 3) if config.family == "stgcn" else nn.Identity()
                    )
                    self.skips.append(
                        nn.Linear(width, width) if config.family == "graph-wavenet" else nn.Identity()
                    )
                if config.connectivity != "static":
                    self.node_source = nn.Parameter(
                        torch.randn(len(sensor_ids), config.embedding_dim) * 0.1
                    )
                    self.node_destination = nn.Parameter(
                        torch.randn(config.embedding_dim, len(sensor_ids)) * 0.1
                    )

    def adaptive_adjacency(self) -> Tensor:
        if self.node_source is None or self.node_destination is None:
            raise DataError("Adaptive connectivity disabled")
        return torch.softmax(torch.relu(self.node_source @ self.node_destination), dim=-1)

    def supports(self) -> list[Tensor]:
        fixed = [cast(Tensor, self.forward_walk), cast(Tensor, self.reverse_walk)]
        if self.config.connectivity == "static":
            return fixed
        adaptive = self.adaptive_adjacency()
        return [adaptive] if self.config.connectivity == "adaptive" else [*fixed, adaptive]

    def encode(self, x: Tensor) -> Tensor:
        z = self.projection(x)
        supports = self.supports()
        if self.config.family == "dcrnn":
            for cell in self.cells:
                state = z.new_zeros(z.shape[0], z.shape[2], self.config.hidden)
                states = []
                for t in range(z.shape[1]):
                    state = cell(z[:, t], state, supports)
                    states.append(state)
                z = torch.stack(states, dim=1)
            return cast(Tensor, z)
        skip = torch.zeros_like(z)
        for temporal, spatial, after, project in zip(
            self.temporal, self.spatial, self.after, self.skips, strict=True
        ):
            filtered = temporal(z)
            mixed = spatial(filtered, supports)
            z = z + self.dropout(after(torch.relu(mixed)))
            skip = skip + project(z)
        return cast(Tensor, skip if self.config.family == "graph-wavenet" else z)

    def forward(self, x: Tensor, *, sensor_ids: tuple[str, ...] | None = None) -> Tensor:
        if (
            x.ndim != 4
            or x.shape[1:] != (self.history, len(self.sensor_ids), self.features)
            or (sensor_ids is not None and sensor_ids != self.sensor_ids)
        ):
            raise DataError("Expected aligned [B,T,N,F] graph inputs")
        if not bool(torch.isfinite(x).all()):
            raise DataError("Graph model requires sanitized inputs and validity indicators")
        if self.config.family == "urbangraphformer":
            adj = torch.stack(self.supports(), dim=0)
            out = self.transformer(x, adj)
            if self.config.residual:
                out = out + x[:, -1, :, 0][:, None, :]
            return cast(Tensor, out.unsqueeze(-1))
        forecast = self.head(self.norm(self.encode(x)[:, -1])).permute(0, 2, 1).unsqueeze(-1)
        if self.config.residual:
            forecast = forecast + x[:, -1, :, 0][:, None, :, None]
        return cast(Tensor, forecast)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
