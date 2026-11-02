"""Governed three-input road graph lifecycle using existing graph algorithms.

The canonical observations, locations and distances are distinct obligations.
This local path assumes an OS-protected policy and exclusively controlled trees.
"""

from __future__ import annotations

from pathlib import Path

from urbangraph.data import DataError, verify_snapshot
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.data.rights import AuthorizationRequest, require_authorizations
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.dataset import open_governed_dataset
from urbangraph.graphs.artifacts import _load_graph_unchecked, _publish_graph_unchecked
from urbangraph.graphs.construction import apply_policies, geographic_knn, road_gaussian
from urbangraph.graphs.contracts import GraphArtifact, GraphSource, GraphSpecification
from urbangraph.graphs.sources import read_distances, read_locations


def _verify_raw(snapshot: Path, digest: str) -> None:
    result = verify_snapshot(snapshot)
    if not result.valid or result.manifest is None or result.manifest.snapshot_id != digest:
        raise DataError("Graph upstream raw snapshot differs from current grant")


def _verify_csv(path: Path, approved: str) -> None:
    if stream_file(safe_absolute(path))[1] != approved:
        raise DataError("Graph upstream CSV differs from current grant")


def _authorize_set(
    snapshot: Path,
    locations: Path,
    distances: Path,
    location_reference: str,
    distance_reference: str,
    operation: str,
    destination: Path,
) -> tuple[str, str, str]:
    """Resolve every policy obligation before inspecting any protected source."""
    raw_operation = (
        "canonical_graph_create" if operation == "graph_create" else "canonical_graph_read"
    )
    digests = require_authorizations(
        (
            AuthorizationRequest(
                snapshot.parent.name, snapshot, "graph_analysis", destination, raw_operation
            ),
            AuthorizationRequest(
                location_reference, locations, "graph_analysis", destination, operation
            ),
            AuthorizationRequest(
                distance_reference, distances, "graph_analysis", destination, operation
            ),
        )
    )
    return digests[0], digests[1], digests[2]


def create_governed_road_graph(
    *,
    raw_snapshot_dir: Path,
    canonical_dir: Path,
    locations: Path,
    distances: Path,
    location_reference: str,
    distance_reference: str,
    output_root: Path,
    specification: GraphSpecification,
) -> GraphArtifact:
    """Authorize all three inputs, then run the real road-Gaussian graph builder."""
    if specification.method != "road-gaussian" or location_reference == distance_reference:
        raise DataError("Distinct complete road graph source identities are required")
    paths = (safe_absolute(canonical_dir), safe_absolute(locations), safe_absolute(distances))
    output = safe_absolute(output_root)
    if len(set(paths)) != 3 or any(
        output.is_relative_to(path) or path.is_relative_to(output) for path in paths
    ):
        raise DataError("Graph inputs and output must be distinct and disjoint")
    raw_digest, location_digest, distance_digest = _authorize_set(
        raw_snapshot_dir,
        locations,
        distances,
        location_reference,
        distance_reference,
        "graph_create",
        output,
    )
    _verify_raw(raw_snapshot_dir, raw_digest)
    _verify_csv(locations, location_digest)
    _verify_csv(distances, distance_digest)
    canonical = open_governed_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    if canonical.lineage["dataset_id"] != specification.dataset_id:
        raise DataError("Graph/canonical dataset mismatch")
    index, _, location_source = read_locations(
        locations, canonical.sensor_ids, reference=location_reference, layout="locations-header"
    )
    matrix, distance_source, construction = read_distances(
        distances,
        index,
        reference=distance_reference,
        layout="distance-header",
        duplicate_policy=specification.duplicate_edges,
        outside_policy=specification.outside_nodes,
    )
    if location_source.sha256 != location_digest or distance_source.sha256 != distance_digest:
        raise DataError("Graph sources changed during construction")
    adjacency, scale = road_gaussian(matrix, specification.threshold)
    construction |= {
        "distance_population_std": scale,
        "numeric_precision": "float64",
        "upstream_difference": "upstream stores distances in float32",
    }
    descriptor: dict[str, object] = {
        "schema_version": "graph-1.0",
        "specification": specification.to_dict(),
        "canonical": canonical.lineage,
        "index": index.to_dict(),
        "sources": [location_source.to_dict(), distance_source.to_dict()],
        "construction": construction,
        "orientation": "source-row,destination-column",
    }
    return _publish_graph_unchecked(apply_policies(adjacency, specification), descriptor, output)


def read_governed_road_graph(
    graph_dir: Path,
    *,
    raw_snapshot_dir: Path,
    canonical_dir: Path,
    locations: Path,
    distances: Path,
    location_reference: str,
    distance_reference: str,
) -> GraphArtifact:
    """Recheck every upstream grant before loading the saved numeric graph."""
    if location_reference == distance_reference or locations == distances:
        raise DataError("Distinct complete road graph source identities are required")
    raw_digest, location_digest, distance_digest = _authorize_set(
        raw_snapshot_dir,
        locations,
        distances,
        location_reference,
        distance_reference,
        "graph_reuse",
        graph_dir,
    )
    _verify_raw(raw_snapshot_dir, raw_digest)
    _verify_csv(locations, location_digest)
    _verify_csv(distances, distance_digest)
    canonical = open_governed_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    location_size, current_location = stream_file(safe_absolute(locations))
    distance_size, current_distance = stream_file(safe_absolute(distances))
    if current_location != location_digest or current_distance != distance_digest:
        raise DataError("Graph upstream CSV changed during reuse")
    expected = [
        GraphSource(
            locations.name, location_digest, location_size, location_reference, "locations-header"
        ).to_dict(),
        GraphSource(
            distances.name, distance_digest, distance_size, distance_reference, "distance-header"
        ).to_dict(),
    ]
    # Check the complete claimed source set before opening the numeric archive.
    claimed = read_json(safe_absolute(graph_dir) / "graph.json")
    if claimed.get("canonical") != canonical.lineage or claimed.get("sources") != expected:
        raise DataError("Graph upstream set or canonical lineage mismatch")
    graph = _load_graph_unchecked(graph_dir)
    if graph.descriptor != claimed:
        raise DataError("Graph descriptor changed during governed read")
    return graph


def _authorize_geographic(
    snapshot: Path, locations: Path, reference: str, operation: str, destination: Path
) -> tuple[str, str]:
    raw_operation = (
        "canonical_graph_create" if operation == "graph_create" else "canonical_graph_read"
    )
    digests = require_authorizations(
        (
            AuthorizationRequest(
                snapshot.parent.name, snapshot, "graph_analysis", destination, raw_operation
            ),
            AuthorizationRequest(reference, locations, "graph_analysis", destination, operation),
        )
    )
    return digests[0], digests[1]


def create_governed_geographic_graph(
    *,
    raw_snapshot_dir: Path,
    canonical_dir: Path,
    locations: Path,
    location_reference: str,
    coordinate_reference: str,
    output_root: Path,
    specification: GraphSpecification,
) -> GraphArtifact:
    """Build an actual geographic KNN graph from two independently granted sources."""
    if specification.method != "geographic-knn" or not coordinate_reference.strip():
        raise DataError("Geographic method and coordinate reference required")
    paths = (safe_absolute(canonical_dir), safe_absolute(locations))
    output = safe_absolute(output_root)
    if len(set(paths)) != 2 or any(
        output.is_relative_to(path) or path.is_relative_to(output) for path in paths
    ):
        raise DataError("Geographic graph inputs and output must be disjoint")
    raw_digest, location_digest = _authorize_geographic(
        raw_snapshot_dir, locations, location_reference, "graph_create", output
    )
    _verify_raw(raw_snapshot_dir, raw_digest)
    _verify_csv(locations, location_digest)
    canonical = open_governed_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    if canonical.lineage["dataset_id"] != specification.dataset_id:
        raise DataError("Geographic graph/canonical dataset mismatch")
    index, coordinates, source = read_locations(
        locations,
        canonical.sensor_ids,
        reference=location_reference,
        layout="locations-header",
        coordinate_reference=coordinate_reference,
    )
    if coordinates is None or source.sha256 != location_digest:
        raise DataError("Geographic source changed or coordinates absent")
    adjacency = geographic_knn(coordinates, specification)
    descriptor: dict[str, object] = {
        "schema_version": "graph-1.0",
        "specification": specification.to_dict(),
        "canonical": canonical.lineage,
        "index": index.to_dict(),
        "sources": [source.to_dict()],
        "construction": {
            "coordinate_reference": coordinate_reference,
            "distance_units": "km",
            "earth_radius_km": 6371.0088,
            "inverse_weight": "1/(1+distance_km)",
            "ties": "canonical index",
        },
        "orientation": "source-row,destination-column",
    }
    return _publish_graph_unchecked(apply_policies(adjacency, specification), descriptor, output)


def read_governed_geographic_graph(
    graph_dir: Path,
    *,
    raw_snapshot_dir: Path,
    canonical_dir: Path,
    locations: Path,
    location_reference: str,
) -> GraphArtifact:
    """Reauthorize both geographic upstreams before opening numeric graph bytes."""
    raw_digest, location_digest = _authorize_geographic(
        raw_snapshot_dir, locations, location_reference, "graph_reuse", graph_dir
    )
    _verify_raw(raw_snapshot_dir, raw_digest)
    _verify_csv(locations, location_digest)
    canonical = open_governed_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    location_size, current = stream_file(safe_absolute(locations))
    if current != location_digest:
        raise DataError("Geographic source changed during reuse")
    expected = [
        GraphSource(
            locations.name, location_digest, location_size, location_reference, "locations-header"
        ).to_dict()
    ]
    claimed = read_json(safe_absolute(graph_dir) / "graph.json")
    specification = claimed.get("specification")
    if (
        not isinstance(specification, dict)
        or specification.get("method") != "geographic-knn"
        or claimed.get("canonical") != canonical.lineage
        or claimed.get("sources") != expected
    ):
        raise DataError("Geographic graph upstream set or lineage mismatch")
    graph = _load_graph_unchecked(graph_dir)
    if graph.descriptor != claimed:
        raise DataError("Geographic graph descriptor changed during read")
    return graph
