"""Validated PyG topology conversion and weighted batched message passing."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing  # type: ignore[import-untyped]
from torch_geometric.utils import to_dense_adj  # type: ignore[import-untyped]

from urbangraph.data import DataError
from urbangraph.graphs.contracts import FloatMatrix
from urbangraph.graphs.normalization import sparse_edges


def pyg_adjacency(adjacency: FloatMatrix) -> Tensor:
    edge_index, weights = sparse_edges(adjacency)
    result = cast(
        Tensor,
        to_dense_adj(
            torch.from_numpy(edge_index),
            edge_attr=torch.from_numpy(weights),
            max_num_nodes=len(adjacency),
        ),
    )[0]
    if not np.array_equal(result.numpy(), adjacency):
        raise DataError("PyG source/destination conversion mismatch")
    return result.to(torch.float32)


class WeightedMessages(MessagePassing):  # type: ignore[misc]
    """Topology is shared across [...,N,F], with no cross-sample node flattening."""

    def __init__(self) -> None:
        super().__init__(aggr="add", node_dim=-2)

    def forward(self, x: Tensor, edge_index: Tensor, weights: Tensor) -> Tensor:
        return cast(Tensor, self.propagate(edge_index, x=x, weights=weights))

    def message(self, x_j: Tensor, weights: Tensor) -> Tensor:
        return x_j * weights[:, None]
