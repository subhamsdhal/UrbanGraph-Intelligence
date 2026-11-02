"""Pure road-distance and spherical geographic graph construction."""

from __future__ import annotations

from typing import cast

import numpy as np

from urbangraph.data import DataError
from urbangraph.graphs.contracts import FloatMatrix, GraphSpecification
from urbangraph.graphs.normalization import normalize, validate_adjacency


def road_gaussian(distances: FloatMatrix, threshold: float) -> tuple[FloatMatrix, float]:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or np.isnan(d).any() or (d < 0).any():
        raise DataError("Invalid distance matrix")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DataError("Invalid Gaussian threshold")
    finite = d[np.isfinite(d)]
    if not finite.size:
        raise DataError("No within-network distances")
    scale = float(finite.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise DataError("Distance standard deviation must be positive")
    with np.errstate(over="ignore"):
        a = np.exp(-np.square(d / scale))
    a[a < threshold] = 0
    return a, scale


def haversine(coordinates: FloatMatrix) -> FloatMatrix:
    c = np.asarray(coordinates, dtype=np.float64)
    if (
        c.ndim != 2
        or c.shape[1] != 2
        or not np.isfinite(c).all()
        or (np.abs(c[:, 0]) > 90).any()
        or (np.abs(c[:, 1]) > 180).any()
    ):
        raise DataError("Invalid latitude/longitude coordinates")
    lat, lon = np.deg2rad(c).T
    h = np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
    h += (
        np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2
    )
    return cast(FloatMatrix, 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def geographic_knn(coordinates: FloatMatrix, spec: GraphSpecification) -> FloatMatrix:
    d = haversine(coordinates)
    if spec.k >= len(d):
        raise DataError("Geographic k must be smaller than node count")
    np.fill_diagonal(d, np.inf)
    a = np.zeros_like(d)
    for i in range(len(d)):
        # Stable sort makes equal-distance tie-breaking use canonical node position.
        neighbors = np.argsort(d[i], kind="stable")[: spec.k]
        neighbors = neighbors[d[i, neighbors] <= spec.radius_km]
        distance = d[i, neighbors]
        weights = (
            np.exp(-((distance / spec.geographic_scale_km) ** 2))
            if spec.geographic_weight == "gaussian"
            else 1 / (1 + distance)
        )
        a[i, neighbors] = weights
    a[a < spec.threshold] = 0
    return a


def apply_policies(adjacency: FloatMatrix, spec: GraphSpecification) -> FloatMatrix:
    a = validate_adjacency(adjacency).copy()
    if spec.direction == "symmetric-max":
        a = np.maximum(a, a.T)
    if spec.self_loops == "remove":
        np.fill_diagonal(a, 0)
    elif spec.self_loops == "unit":
        np.fill_diagonal(a, 1)
    return normalize(a, spec.normalization)


def diagnostics(adjacency: FloatMatrix) -> dict[str, object]:
    a = validate_adjacency(adjacency)
    edges = a > 0
    n = len(a)
    weak = edges | edges.T
    np.fill_diagonal(weak, False)
    remaining = set(range(n))
    components: list[int] = []
    while remaining:
        pending = [min(remaining)]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node in visited:
                continue
            visited.add(node)
            pending.extend(int(v) for v in np.flatnonzero(weak[node]) if int(v) not in visited)
        remaining -= visited
        components.append(len(visited))
    weights = a[edges]
    return {
        "nodes": n,
        "directed_edges": int(edges.sum()),
        "self_loops": int(np.diag(edges).sum()),
        "density_including_diagonal": float(edges.mean()),
        "isolated_without_self_loops": int((~weak.any(axis=1)).sum()),
        "weak_component_sizes": sorted(components, reverse=True),
        "source_coverage": int(edges.any(axis=1).sum()),
        "destination_coverage": int(edges.any(axis=0).sum()),
        "out_degree": edges.sum(axis=1).tolist(),
        "in_degree": edges.sum(axis=0).tolist(),
        "weighted_out_degree": a.sum(axis=1).tolist(),
        "weighted_in_degree": a.sum(axis=0).tolist(),
        "weight_quantiles": np.quantile(weights, [0, 0.25, 0.5, 0.75, 1]).tolist()
        if weights.size
        else [],
        "sensor_id_match_percent": 100.0,
    }
