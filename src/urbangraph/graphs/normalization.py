"""Explicit source-row/destination-column operators, without hidden reverse edges."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.graphs.contracts import FloatMatrix


def validate_adjacency(adjacency: FloatMatrix) -> FloatMatrix:
    value = np.asarray(adjacency, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not len(value):
        raise DataError("Adjacency must be a nonempty square matrix")
    if len(value) > 10000 or not np.isfinite(value).all() or (value < 0).any():
        raise DataError("Invalid adjacency size or weights")
    return value


def normalize(adjacency: FloatMatrix, method: str) -> FloatMatrix:
    a = validate_adjacency(adjacency).copy()
    if method == "none":
        return a
    if method == "reverse-walk":
        a = a.T.copy()
    if method in {"random-walk", "reverse-walk"}:
        degree = a.sum(axis=1, keepdims=True)
        return cast(FloatMatrix, np.divide(a, degree, out=np.zeros_like(a), where=degree != 0))
    if method in {"symmetric", "laplacian", "scaled-laplacian"}:
        if not np.array_equal(a, a.T):
            raise DataError("Symmetric operator requires explicitly symmetric adjacency")
        degree = a.sum(axis=1)
        inv = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        normalized = inv[:, None] * a * inv[None, :]
        if method == "symmetric":
            return cast(FloatMatrix, normalized)
        laplacian = np.diag((degree > 0).astype(float)) - normalized
        if method == "laplacian":
            return cast(FloatMatrix, laplacian)
        # Bound lambda_max <= 2; isolated zero Laplacian maps to -I.
        return cast(FloatMatrix, laplacian - np.eye(len(a)))
    raise DataError("Unsupported graph normalization")


def sparse_edges(adjacency: FloatMatrix) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    a = validate_adjacency(adjacency)
    source, destination = np.nonzero(a)
    return np.asarray([source, destination], dtype=np.int64), a[source, destination].copy()


def perturb(adjacency: FloatMatrix, method: str, *, seed: int = 2026) -> FloatMatrix:
    a = validate_adjacency(adjacency).copy()
    rng = np.random.default_rng(seed)
    if method == "identity":
        return np.eye(len(a))
    if method == "remove-edges":
        return np.diag(np.diag(a))
    if method == "unweighted":
        return (a > 0).astype(np.float64)
    if method == "symmetric-max":
        return np.maximum(a, a.T)
    if method == "permute-weights":
        locations = a != 0
        a[locations] = rng.permutation(a[locations])
        return a
    if method == "randomize-destinations":
        # Each row independently permutes columns: preserves outgoing degree/weight multiset.
        return np.stack([rng.permutation(row) for row in a])
    raise DataError("Unsupported graph perturbation")
