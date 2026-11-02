"""Append-only numeric graph bundles; source hashes and exact axes determine identity."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

import numpy as np

from urbangraph.data import DataError
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity, open_dataset
from urbangraph.graphs.construction import (
    apply_policies,
    diagnostics,
    geographic_knn,
    road_gaussian,
)
from urbangraph.graphs.contracts import (
    FloatMatrix,
    GraphArtifact,
    GraphSpecification,
    GraphVerificationResult,
    SensorNodeIndex,
)
from urbangraph.graphs.normalization import validate_adjacency
from urbangraph.graphs.sources import read_distances, read_locations


def specification(value: object) -> GraphSpecification:
    obj = object_dict(value)
    if set(obj) != {f.name for f in fields(GraphSpecification)}:
        raise DataError("Graph specification fields mismatch")
    # Dataclass runtime validation checks every field; no coercion of untrusted JSON.
    return GraphSpecification(**obj)  # type: ignore[arg-type]


def adjacency_digest(value: FloatMatrix) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8", order="C").tobytes()).hexdigest()


def load_graph(
    directory: str | Path,
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
    locations: str | Path | None = None,
    distances: str | Path | None = None,
    location_reference: str | None = None,
    distance_reference: str | None = None,
    staged: bool = False,
) -> GraphArtifact:
    """Protected graph use; a caller cannot omit its complete source context."""
    if (
        staged
        or raw_snapshot_dir is None
        or canonical_dir is None
        or locations is None
        or location_reference is None
    ):
        raise DataError("Complete governed graph source context required")
    from urbangraph.graphs.governed import (
        read_governed_geographic_graph,
        read_governed_road_graph,
    )

    if distances is None and distance_reference is None:
        return read_governed_geographic_graph(
            Path(directory),
            raw_snapshot_dir=Path(raw_snapshot_dir),
            canonical_dir=Path(canonical_dir),
            locations=Path(locations),
            location_reference=str(location_reference),
        )
    if distances is None or distance_reference is None:
        raise DataError("Complete governed road source context required")

    return read_governed_road_graph(
        Path(directory),
        raw_snapshot_dir=Path(raw_snapshot_dir),
        canonical_dir=Path(canonical_dir),
        locations=Path(locations),
        distances=Path(distances),
        location_reference=str(location_reference),
        distance_reference=str(distance_reference),
    )


def _load_graph_unchecked(directory: str | Path, *, staged: bool = False) -> GraphArtifact:
    """Integrity-only loader for use after explicit authorization or during publication."""
    root = safe_absolute(directory)
    tree = discover(root)
    if tree.directories or set(tree.files) != {"manifest.json", "graph.json", "adjacency.npz"}:
        raise DataError("Graph inventory mismatch")
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "graph_id", "files"}
        or manifest["schema_version"] != "graph-1.0"
    ):
        raise DataError("Unsupported graph manifest")
    files = object_dict(manifest["files"])
    if set(files) != {"graph.json", "adjacency.npz"}:
        raise DataError("Graph file inventory mismatch")
    for name, digest in files.items():
        if stream_file(root / name)[1] != digest:
            raise DataError("Graph checksum mismatch")
    descriptor = read_json(root / "graph.json")
    required = {
        "schema_version",
        "specification",
        "canonical",
        "index",
        "sources",
        "construction",
        "adjacency_sha256",
        "orientation",
        "diagnostics",
    }
    if set(descriptor) != required or descriptor["schema_version"] != "graph-1.0":
        raise DataError("Graph descriptor fields mismatch")
    spec = specification(descriptor["specification"])
    canonical = object_dict(descriptor["canonical"])
    if canonical.get("dataset_id") != spec.dataset_id or not canonical.get("canonical_id"):
        raise DataError("Graph canonical lineage mismatch")
    if descriptor["orientation"] != "source-row,destination-column":
        raise DataError("Graph orientation mismatch")
    sources = descriptor["sources"]
    if not isinstance(sources, list) or not sources:
        raise DataError("Missing graph sources")
    for item in sources:
        source = object_dict(item)
        if (
            set(source) != {"name", "sha256", "size_bytes", "reference", "layout"}
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
            or not isinstance(source["reference"], str)
            or not source["reference"]
        ):
            raise DataError("Invalid graph source provenance")
    idx = object_dict(descriptor["index"])
    axes = [idx.get("sensor_ids"), idx.get("source_order")]
    if any(not isinstance(v, list) or any(not isinstance(s, str) for s in v) for v in axes):
        raise DataError("Invalid graph sensor index")
    index = SensorNodeIndex(tuple(cast(list[str], axes[0])), tuple(cast(list[str], axes[1])))
    if idx != index.to_dict():
        raise DataError("Graph sensor permutation/fingerprint mismatch")
    graph_id = identity(descriptor)
    if manifest["graph_id"] != graph_id or (not staged and root.name != graph_id):
        raise DataError("Graph content identity mismatch")
    try:
        with ZipFile(root / "adjacency.npz") as archive:
            info = archive.infolist()
            if (
                len(info) != 1
                or info[0].filename != "adjacency.npy"
                or info[0].file_size > 128 * 1024**2
            ):
                raise DataError("Unsafe or oversized graph numeric archive")
        with np.load(root / "adjacency.npz", allow_pickle=False) as archive:
            adjacency = archive["adjacency"]
            if adjacency.dtype != np.dtype("float64"):
                raise DataError("Graph weights must be float64")
            a = validate_adjacency(adjacency).copy()
    except (ValueError, OSError, BadZipFile) as exc:
        raise DataError("Invalid numeric graph archive") from exc
    if (
        a.shape != (len(index.sensor_ids), len(index.sensor_ids))
        or adjacency_digest(a) != descriptor["adjacency_sha256"]
    ):
        raise DataError("Graph numeric identity or sensor shape mismatch")
    if (
        spec.direction == "symmetric-max"
        and spec.normalization in {"none", "symmetric"}
        and not np.array_equal(a, a.T)
    ):
        raise DataError("Graph direction policy mismatch")
    if diagnostics(a) != descriptor["diagnostics"]:
        raise DataError("Graph diagnostics mismatch")
    a.flags.writeable = False
    return GraphArtifact(root, graph_id, index, a, descriptor)


def verify_graph(
    directory: str | Path,
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
    locations: str | Path | None = None,
    distances: str | Path | None = None,
    location_reference: str | None = None,
    distance_reference: str | None = None,
) -> GraphVerificationResult:
    """Verify graph integrity only after current complete upstream authorization."""
    try:
        graph = load_graph(
            directory,
            raw_snapshot_dir=raw_snapshot_dir,
            canonical_dir=canonical_dir,
            locations=locations,
            distances=distances,
            location_reference=location_reference,
            distance_reference=distance_reference,
        )
        return GraphVerificationResult(True, graph.graph_id)
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        return GraphVerificationResult(False, None, (str(exc),))


def publish_graph(
    adjacency: FloatMatrix, descriptor: dict[str, object], output_root: str | Path
) -> GraphArtifact:
    """Legacy direct writer is closed until complete upstream authority is supplied."""
    raise DataError("Use governed graph creation with complete upstream sources")


def _publish_graph_unchecked(
    adjacency: FloatMatrix, descriptor: dict[str, object], output_root: str | Path
) -> GraphArtifact:
    a = validate_adjacency(adjacency)
    complete = descriptor | {"adjacency_sha256": adjacency_digest(a), "diagnostics": diagnostics(a)}
    graph_id = identity(complete)
    root = safe_absolute(output_root)
    for parent in (root, *root.parents):
        if (parent / "manifest.json").exists():
            raise DataError("Cannot nest immutable graph artifacts")
    root.mkdir(parents=True, exist_ok=True)
    with _writer_lock(root):
        destination = root / graph_id
        if os.path.lexists(destination):
            return _load_graph_unchecked(destination)
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            write_json(stage / "graph.json", complete)
            with (stage / "adjacency.npz").open("xb") as handle:
                np.savez(handle, adjacency=np.asarray(a, dtype="<f8"))
                handle.flush()
                os.fsync(handle.fileno())
            write_json(
                stage / "manifest.json",
                {
                    "schema_version": "graph-1.0",
                    "graph_id": graph_id,
                    "files": {
                        name: stream_file(stage / name)[1]
                        for name in ("graph.json", "adjacency.npz")
                    },
                },
            )
            _load_graph_unchecked(stage, staged=True)
            _publish(stage, destination)
        finally:
            _cleanup_stage(stage, root)
    return _load_graph_unchecked(destination)


def build_graph(config: dict[str, object]) -> GraphArtifact:
    """Require a complete source context before the supported graph build."""
    if "specification" not in config:
        raise DataError("Complete governed graph configuration required")
    common = {
        "specification",
        "raw_snapshot_dir",
        "canonical_dir",
        "locations",
        "location_reference",
        "output_root",
    }
    spec = specification(config.get("specification"))
    required = (
        common | {"distances", "distance_reference"}
        if spec.method == "road-gaussian"
        else common | {"coordinate_reference"}
    )
    if set(config) != required or any(
        not isinstance(config[key], str) or not config[key] for key in required - {"specification"}
    ):
        raise DataError("Complete governed graph configuration required")
    from urbangraph.graphs.governed import (
        create_governed_geographic_graph,
        create_governed_road_graph,
    )

    if spec.method == "geographic-knn":
        return create_governed_geographic_graph(
            raw_snapshot_dir=Path(str(config["raw_snapshot_dir"])),
            canonical_dir=Path(str(config["canonical_dir"])),
            locations=Path(str(config["locations"])),
            location_reference=str(config["location_reference"]),
            coordinate_reference=str(config["coordinate_reference"]),
            output_root=Path(str(config["output_root"])),
            specification=spec,
        )

    return create_governed_road_graph(
        raw_snapshot_dir=Path(str(config["raw_snapshot_dir"])),
        canonical_dir=Path(str(config["canonical_dir"])),
        locations=Path(str(config["locations"])),
        distances=Path(str(config["distances"])),
        location_reference=str(config["location_reference"]),
        distance_reference=str(config["distance_reference"]),
        output_root=Path(str(config["output_root"])),
        specification=spec,
    )


def _legacy_build_graph_unchecked(config: dict[str, object]) -> GraphArtifact:
    required = {
        "specification",
        "canonical_dir",
        "locations",
        "location_layout",
        "reference",
        "distances",
        "distance_layout",
        "coordinate_reference",
        "output_root",
    }
    if set(config) != required:
        raise DataError("Graph build configuration fields mismatch")
    for key in required - {"specification", "coordinate_reference"}:
        if not isinstance(config[key], str):
            raise DataError(f"Expected string {key}")
    coordinate_reference = config["coordinate_reference"]
    if coordinate_reference is not None and not isinstance(coordinate_reference, str):
        raise DataError("Invalid coordinate reference")
    spec = specification(config["specification"])
    canonical = open_dataset(str(config["canonical_dir"]))
    if canonical.lineage["dataset_id"] != spec.dataset_id:
        raise DataError("Graph/canonical dataset mismatch")
    output = safe_absolute(str(config["output_root"]))
    for key in ("canonical_dir", "locations", "distances"):
        source = safe_absolute(str(config[key]))
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise DataError("Graph output overlaps an input")
    index, coordinates, locations = read_locations(
        str(config["locations"]),
        canonical.sensor_ids,
        reference=str(config["reference"]),
        layout=str(config["location_layout"]),
        coordinate_reference=coordinate_reference,
    )
    sources = [locations.to_dict()]
    if spec.method == "road-gaussian":
        distances, distance_source, construction = read_distances(
            str(config["distances"]),
            index,
            reference=str(config["reference"]),
            layout=str(config["distance_layout"]),
            duplicate_policy=spec.duplicate_edges,
            outside_policy=spec.outside_nodes,
        )
        adjacency, scale = road_gaussian(distances, spec.threshold)
        construction |= {
            "distance_population_std": scale,
            "numeric_precision": "float64",
            "upstream_difference": "upstream stores distances in float32",
        }
        sources.append(distance_source.to_dict())
    else:
        if coordinates is None:
            raise DataError("Headerless geographic coordinates require explicit source evidence")
        adjacency = geographic_knn(coordinates, spec)
        construction = {
            "coordinate_reference": coordinate_reference,
            "distance_units": "km",
            "earth_radius_km": 6371.0088,
            "inverse_weight": "1/(1+distance_km)",
            "ties": "canonical index",
        }
    descriptor: dict[str, object] = {
        "schema_version": "graph-1.0",
        "specification": spec.to_dict(),
        "canonical": canonical.lineage,
        "index": index.to_dict(),
        "sources": sources,
        "construction": construction,
        "orientation": "source-row,destination-column",
    }
    return publish_graph(apply_policies(adjacency, spec), descriptor, output)
