"""Causal, parameter-matched regional temporal and weighted graph models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from energraph.regions import country_registry
from urbangraph.data import DataError
from urbangraph.graphs.normalization import normalize, validate_adjacency
from urbangraph.stgnn.graph_batching import WeightedMessages
from urbangraph.stgnn.models import CausalGate


@dataclass(frozen=True)
class EnergyModelConfig:
    condition: str = "identity"
    hidden: int = 8
    blocks: int = 4
    version: str = "energy-stgcn-1.0"

    def __post_init__(self) -> None:
        if self.condition not in {
            "temporal",
            "identity",
            "geographic",
            "correlation",
            "random",
            "correlation-unweighted",
        }:
            raise DataError("Unknown energy graph condition")
        if (
            self.version != "energy-stgcn-1.0"
            or type(self.hidden) is not int
            or not 1 <= self.hidden <= 64
            or type(self.blocks) is not int
            or not 1 <= self.blocks <= 6
        ):
            raise DataError("Invalid energy architecture")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EnergyForecaster(nn.Module):
    """Dilated temporal/weighted spatial/temporal blocks; direct horizon head.

    Incoming convention: source-row adjacency A[j,i] sends j to i.
    CPU uses PyG aggregation; CUDA uses exactly the equivalent dense operator.
    Temporal control retains the same learnable dimensions and local mixing only.
    """

    def __init__(
        self,
        config: EnergyModelConfig,
        adjacency: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        regions: tuple[str, ...],
        *,
        history: int = 168,
        horizon: int = 24,
    ) -> None:
        super().__init__()
        country_registry().axis(regions)
        a = validate_adjacency(adjacency)
        if a.shape != (len(regions), len(regions)) or history < 1 or horizon < 1:
            raise DataError("Model graph/axis/temporal mismatch")
        if config.condition in {"identity", "temporal"} and not np.array_equal(
            a, np.eye(len(regions))
        ):
            raise DataError("Independent controls require identity adjacency")
        if config.condition == "correlation-unweighted":
            a = (a > 0).astype(np.float64)
        support = normalize(a, "reverse-walk")
        destination, source = np.nonzero(support)
        self.register_buffer("support", torch.tensor(support, dtype=torch.float32))
        self.register_buffer(
            "edges", torch.tensor(np.array([source, destination]), dtype=torch.long)
        )
        self.register_buffer(
            "weights", torch.tensor(support[destination, source], dtype=torch.float32)
        )
        self.config, self.regions, self.history, self.horizon = config, regions, history, horizon
        self.project = nn.Linear(2, config.hidden)
        self.before = nn.ModuleList(
            CausalGate(config.hidden, 2**i, 3) for i in range(config.blocks)
        )
        self.after = nn.ModuleList(CausalGate(config.hidden, 2**i, 3) for i in range(config.blocks))
        self.spatial = nn.ModuleList(
            nn.Linear(2 * config.hidden, config.hidden) for _ in range(config.blocks)
        )
        self.norm = nn.LayerNorm(config.hidden)
        self.head = nn.Linear(config.hidden, horizon)
        self.messages = WeightedMessages()

    def aggregate(self, x: Tensor, *, dense: bool = False) -> Tensor:
        if self.config.condition == "temporal":
            return x
        if dense or x.device.type == "cuda":
            return torch.matmul(cast(Tensor, self.support), x)
        return cast(Tensor, self.messages(x, self.edges, self.weights))

    def encode(self, x: Tensor) -> Tensor:
        z = self.project(x)
        for before, spatial, after in zip(self.before, self.spatial, self.after, strict=True):
            t = before(z)
            z = z + after(torch.relu(spatial(torch.cat([t, self.aggregate(t)], dim=-1))))
        return cast(Tensor, z)

    def forward(self, x: Tensor, *, regions: tuple[str, ...] | None = None) -> Tensor:
        if (
            x.ndim != 4
            or x.shape[1:] != (self.history, len(self.regions), 2)
            or (regions is not None and regions != self.regions)
        ):
            raise DataError("Expected aligned [B,history,regions,2]")
        if not bool(torch.isfinite(x).all()) or not bool(
            ((x[..., 1] == 0) | (x[..., 1] == 1)).all()
        ):
            raise DataError("Finite sanitized values and binary validity required")
        prediction = self.head(self.norm(self.encode(x)[:, -1])).permute(0, 2, 1).unsqueeze(-1)
        return cast(Tensor, prediction + x[:, -1, :, 0][:, None, :, None])
