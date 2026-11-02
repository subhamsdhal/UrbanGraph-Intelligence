"""Evidence-separated energy graph controls with exact node alignment and safe numeric storage."""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    governed_payload,
    require_committed_snapshot,
    resolve_broker_reference,
)
from energraph.canonical import logical_hash
from energraph.graph_authority import authorize_graph_inputs, snapshot_path
from energraph.lineage import FileParent, authorize_sources, resolve_parents
from energraph.regional_data import load_regional_plan
from energraph.regions import country_registry
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.data.filesystem import stream_file
from urbangraph.data.rights import require_raw_authorization
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity
from urbangraph.graphs.normalization import normalize, sparse_edges, validate_adjacency

Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class EnergyGraphSpec:
    method: str
    k: int = 2
    minimum_support: int = 1000
    correlation: str = "pearson"
    weight_policy: str = "positive"
    normalization: str = "random-walk"
    seed: int = 2026
    version: str = "energy-graph-1.0"

    def __post_init__(self) -> None:
        if (
            self.version != "energy-graph-1.0"
            or self.method
            not in {"identity", "empty", "geographic-proxy", "statistical", "random-control"}
            or type(self.k) is not int
            or self.k < 1
            or type(self.minimum_support) is not int
            or self.minimum_support < 2
            or self.correlation not in {"pearson", "spearman"}
            or self.weight_policy not in {"positive", "absolute"}
            or self.normalization not in {"none", "random-walk", "symmetric"}
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise DataError(
                "Unsupported energy graph specification; physical edges require verified evidence"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def geographic_edges(
    source: Path | None,
    order: tuple[str, ...],
    *,
    source_reference: str | None = None,
    geometry_reference: str | None = None,
    destination: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    artifact_reference: BrokerArtifactRef | None = None,
    relative_path: str | None = None,
    geometry_artifact_reference: BrokerArtifactRef | None = None,
    geometry_relative_path: str | None = None,
    broker_provenance: dict[str, object] | None = None,
) -> tuple[Matrix, list[dict[str, object]]]:
    brokered = artifact_reference is not None or broker_context is not None
    relation_content: bytes | None = None
    relation_digest: str | None = None
    geometry_content: bytes | None = None
    geometry_digest: str | None = None
    if brokered:
        if (
            broker_context is None
            or artifact_reference is None
            or artifact_reference.artifact_type != "raw"
            or not relative_path
            or source is not None
            or source_reference is not None
            or destination is None
        ):
            raise DataError("Committed broker geographic identity and scoped reader required")
        relation_content, relation_digest, relation_size, source_id = _read_broker_geographic_file(
            broker_context, artifact_reference, relative_path, operation="energy_graph_create"
        )
        if broker_provenance is not None:
            broker_provenance.update(
                {
                    "artifact_id": artifact_reference.artifact_id,
                    "artifact_sha256": artifact_reference.content_sha256,
                    "source_id": source_id,
                    "relative_path": relative_path,
                    "file_sha256": relation_digest,
                    "file_bytes": relation_size,
                }
            )
        if geometry_artifact_reference is not None:
            if geometry_artifact_reference.artifact_type != "raw" or not geometry_relative_path:
                raise DataError("Geometry requires its own committed raw identity and file path")
            geometry_content, geometry_digest, geometry_size, geometry_source_id = (
                _read_broker_geographic_file(
                    broker_context,
                    geometry_artifact_reference,
                    geometry_relative_path,
                    operation="energy_graph_create",
                    geometry_file=True,
                )
            )
            if broker_provenance is not None:
                broker_provenance["geometry"] = {
                    "artifact_id": geometry_artifact_reference.artifact_id,
                    "artifact_sha256": geometry_artifact_reference.content_sha256,
                    "source_id": geometry_source_id,
                    "relative_path": geometry_relative_path,
                    "file_sha256": geometry_digest,
                    "file_bytes": geometry_size,
                }
        elif geometry_relative_path is not None:
            raise DataError("Geometry file path lacks an independent broker identity")
    else:
        if source is None or not source_reference or destination is None:
            raise DataError("Current geographic source identity and destination required")
        relation_digest = require_raw_authorization(
            source_reference,
            source,
            "graph_analysis",
            destination,
            operation="geography_read",
        )
        if stream_file(source)[1] != relation_digest:
            raise DataError("Geographic source content does not match rights grant")
    country_registry().axis(order)
    if relation_content is not None:
        relation_input: io.BytesIO | Path = io.BytesIO(relation_content)
    elif source is not None:
        relation_input = source
    else:
        raise DataError("Geographic relations were not authorized for reading")
    table = pd.read_csv(relation_input, dtype=str, keep_default_na=False)
    if set(table.columns) != {"CNTR_BN_CODE", "CNTR_CODE"}:
        raise DataError("Invalid or duplicate GISCO boundary relations")
    rejected_relations = 0
    geometry_path: Path | None = (
        None if source is None else source.with_name("CNTR_BN_20M_2020_4326.geojson")
    )
    has_local_geometry = geometry_path is not None and geometry_path.exists()
    if geometry_content is not None or has_local_geometry:
        if brokered:
            if geometry_content is None:
                raise DataError("Geometry file requires its own broker artifact identity")
            geometry_bytes = geometry_content
        else:
            if geometry_path is None or not geometry_reference:
                raise DataError("Current geometry source identity required")
            geometry_digest_expected = require_raw_authorization(
                geometry_reference,
                geometry_path,
                "graph_analysis",
                destination,
                operation="geography_read",
            )
            if stream_file(geometry_path)[1] != geometry_digest_expected:
                raise DataError("Geographic geometry does not match rights grant")
            geometry_bytes = geometry_path.read_bytes()
            geometry_digest = geometry_digest_expected
        geometry = json.loads(geometry_bytes.decode("utf-8"))
        properties = {
            str(f["properties"]["CNTR_BN_CODE"]): f["properties"]
            for f in geometry["features"]
            if f["properties"]["COAS_FLAG"] == "F"
        }
        accepted = []
        for boundary, group in table.groupby("CNTR_BN_CODE", sort=True):
            codes = set(group.CNTR_CODE)
            feature = properties.get(str(boundary))
            if (
                feature is not None
                and len(group) == 2
                and len(codes) == 2
                and feature["CNTR_CODE"] in codes
            ):
                accepted.append(group)
            else:
                rejected_relations += len(group)
        if not accepted:
            raise DataError("No unambiguous noncoastal boundary relations")
        table = pd.concat(accepted, ignore_index=True)
        if not brokered and geometry_path is not None:
            geometry_digest = stream_file(geometry_path)[1]
    selected_codes = {r.split(":")[1] for r in order}
    boundary_ids = set(table.loc[table.CNTR_CODE.isin(selected_codes), "CNTR_BN_CODE"])
    table = table.loc[table.CNTR_BN_CODE.isin(boundary_ids)].copy()
    if table.duplicated().any():
        raise DataError("Duplicate GISCO relations touching selected countries")
    matrix = np.zeros((len(order), len(order)), dtype=np.float64)
    records: dict[tuple[str, str], list[str]] = {}
    for boundary, group in table.groupby("CNTR_BN_CODE", sort=True):
        all_codes = sorted(set(str(v) for v in group.CNTR_CODE))
        if len(all_codes) > 2:
            raise DataError("Ambiguous boundary incidence")
        nodes = sorted("country:" + c for c in all_codes if "country:" + c in order)
        if len(nodes) == 2:
            records.setdefault((nodes[0], nodes[1]), []).append(str(boundary))
    evidence: list[dict[str, object]] = []
    for (a, b), boundaries in sorted(records.items()):
        i, j = order.index(a), order.index(b)
        matrix[i, j] = matrix[j, i] = 1
        evidence.append(
            {
                "source_region": a,
                "destination_region": b,
                "boundary_ids": boundaries,
                "geometry_sha256": geometry_digest,
                "excluded_global_relation_rows": rejected_relations,
                "semantics": "geographic proxy; NOT verified electrical connectivity",
                "direction": "undirected",
                "weight": 1.0,
            }
        )
    return matrix, evidence


def _read_broker_geographic_file(
    context: BrokerPublicationContext,
    reference: BrokerArtifactRef,
    relative_path: str,
    *,
    operation: str,
    geometry_file: bool = False,
) -> tuple[bytes, str, int, str]:
    """Authorize a committed raw directory before releasing one inventoried file."""
    current = resolve_broker_reference(
        context.broker,
        reference.artifact_id,
        purpose="graph_analysis",
        operation=operation,
    )
    if current != reference or reference.artifact_type != "raw":
        raise DataError("Geographic broker identity is forged or no longer current")
    record = context.broker.authorize_directory(
        reference.artifact_id, purpose="graph_analysis", operation=operation
    )
    if (
        record.kind != "raw"
        or record.content_sha256 != reference.content_sha256
        or record.revision != reference.revision
        or len(record.sources) != 1
        or record.snapshot is None
    ):
        raise DataError("Geographic broker source record is not an admitted raw directory")
    files = record.snapshot.get("files")
    if not isinstance(files, list):
        raise DataError("Geographic broker file inventory is unavailable")
    geometry_name = "CNTR_BN_20M_2020_4326.geojson"
    geometry_entries = [
        item
        for item in files
        if isinstance(item, list)
        and item
        and isinstance(item[0], str)
        and item[0].replace("\\", "/").rsplit("/", 1)[-1] == geometry_name
    ]
    if geometry_entries and not geometry_file:
        raise DataError("Geometry source requires a separately admitted broker identity")
    if geometry_file and (
        relative_path.replace("\\", "/").rsplit("/", 1)[-1] != geometry_name
        or len(geometry_entries) != 1
        or geometry_entries[0][0] != relative_path
    ):
        raise DataError("Geometry broker identity does not identify the expected geometry file")
    entries = [item for item in files if isinstance(item, list) and item[0] == relative_path]
    if len(entries) != 1 or len(entries[0]) != 3:
        raise DataError("Geographic file is outside the committed broker inventory")
    size, digest = entries[0][1], entries[0][2]
    if type(size) is not int or not isinstance(digest, str):
        raise DataError("Geographic broker file inventory is malformed")
    content = context.broker.read_directory_file(
        record, relative_path, purpose="graph_analysis", operation=operation
    )
    if len(content) != size or stream_file_from_bytes(content) != digest:
        raise DataError("Broker-delivered geographic bytes differ from committed identity")
    return content, digest, size, record.sources[0][0]


def stream_file_from_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def statistical_edges(
    training: pd.DataFrame, spec: EnergyGraphSpec
) -> tuple[Matrix, dict[str, object]]:
    order = tuple(str(c) for c in training)
    country_registry().axis(order)
    n = len(order)
    weights = np.zeros((n, n), dtype=np.float64)
    counts = np.zeros((n, n), dtype=np.int64)
    for i, j in combinations(range(n), 2):
        a = training.iloc[:, i].to_numpy(dtype=np.float64)
        b = training.iloc[:, j].to_numpy(dtype=np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        count = int(mask.sum())
        counts[i, j] = counts[j, i] = count
        if count < spec.minimum_support:
            continue
        a, b = a[mask], b[mask]
        if spec.correlation == "spearman":
            a = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
            b = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        coefficient = float(np.corrcoef(a, b)[0, 1])
        weight = abs(coefficient) if spec.weight_policy == "absolute" else max(0.0, coefficient)
        weights[i, j] = weights[j, i] = weight
    sparse: Matrix = np.zeros_like(weights)
    for i in range(n):
        chosen = sorted(
            (j for j in range(n) if j != i and weights[i, j] > 0),
            key=lambda j: (-weights[i, j], order[j]),
        )[: spec.k]
        sparse[i, chosen] = weights[i, chosen]
    sparse = np.maximum(sparse, sparse.T)
    return sparse, {
        "pairwise_valid_counts": counts.tolist(),
        "training_logical_sha256": logical_hash(training),
        "fitting_rows": len(training),
        "weight_policy": spec.weight_policy,
        "sparsification": "row top-k; lexical node-ID tie break; undirected union",
        "semantics": "historical association; neither causal nor physical direction",
    }


def realign(matrix: Matrix, source_order: tuple[str, ...], target_order: tuple[str, ...]) -> Matrix:
    country_registry().axis(source_order)
    country_registry().axis(target_order)
    validate_adjacency(matrix)
    if matrix.shape != (len(source_order), len(source_order)) or set(source_order) != set(
        target_order
    ):
        raise DataError("Graph realignment requires identical node set")
    indices = [source_order.index(r) for r in target_order]
    return matrix[np.ix_(indices, indices)].copy()


def diagnostics(matrix: Matrix) -> dict[str, object]:
    a = validate_adjacency(matrix)
    n = len(a)
    neighbors = a.copy()
    np.fill_diagonal(neighbors, 0)
    remaining = set(range(n))
    components = []
    while remaining:
        stack = [min(remaining)]
        component = set()
        while stack:
            i = stack.pop()
            if i in component:
                continue
            component.add(i)
            stack.extend(
                int(j)
                for j in np.flatnonzero((neighbors[i] > 0) | (neighbors[:, i] > 0))
                if int(j) not in component
            )
        remaining -= component
        components.append(sorted(component))
    nonzero = a[a > 0]
    return {
        "nodes": n,
        "stored_directed_edges": int(np.count_nonzero(a)),
        "neighbor_edges": int(np.count_nonzero(neighbors)),
        "undirected": bool(np.array_equal(a, a.T)),
        "weak_components": components,
        "isolated_nodes": np.flatnonzero((neighbors.sum(0) + neighbors.sum(1)) == 0).tolist(),
        "density_excluding_self": float(np.count_nonzero(neighbors) / (n * (n - 1)))
        if n > 1
        else 0.0,
        "out_degree": np.count_nonzero(neighbors, axis=1).tolist(),
        "self_loops": int(np.count_nonzero(np.diag(a))),
        "weight_min": float(nonzero.min()) if len(nonzero) else None,
        "weight_max": float(nonzero.max()) if len(nonzero) else None,
    }


def build_energy_graph(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    root: Path,
    spec: EnergyGraphSpec,
    *,
    fold: int = 0,
    geography: Path | None = None,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
    geography_snapshot: Path | DatasetSnapshot | None = None,
    geography_artifact_reference: BrokerArtifactRef | None = None,
    geography_relative_path: str | None = None,
    geometry_snapshot: Path | DatasetSnapshot | None = None,
    geometry_artifact_reference: BrokerArtifactRef | None = None,
    geometry_relative_path: str | None = None,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> DatasetSnapshot | PendingPublication:
    uses_geography = spec.method in {"geographic-proxy", "random-control"}
    if raw_snapshot_dir is None:
        raise DataError("Complete current source context required for energy graph")
    if broker_context is not None and uses_geography:
        if (
            geography is not None
            or geography_reference is not None
            or geography_snapshot is None
            or geography_artifact_reference is None
            or not geography_relative_path
            or (geometry_snapshot is None) != (geometry_artifact_reference is None)
            or (geometry_snapshot is None) != (geometry_relative_path is None)
        ):
            raise DataError(
                "Geographic graph requires independently admitted broker file identities"
            )
    elif uses_geography and (geography is None or not geography_reference):
        raise DataError("Complete current source context required for energy graph")
    geographic_parents: tuple[tuple[Path | DatasetSnapshot, BrokerArtifactRef], ...] = ()
    if uses_geography and broker_context is not None:
        assert geography_snapshot is not None and geography_artifact_reference is not None
        geographic_parents = ((geography_snapshot, geography_artifact_reference),)
        if geometry_snapshot is not None and geometry_artifact_reference is not None:
            geographic_parents += ((geometry_snapshot, geometry_artifact_reference),)
    canonical_path, plan_path = snapshot_path(canonical), snapshot_path(plan)
    trusted = authorize_graph_inputs(
        canonical,
        plan,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        purpose="graph_analysis",
        operation="energy_graph_create",
        geographic_parents=geographic_parents,
    )
    parents: tuple[Path | FileParent, ...] = (canonical_path, plan_path)
    if uses_geography:
        if broker_context is not None:
            parents += tuple(snapshot_path(item[0]) for item in geographic_parents)
        else:
            if geography is None or geography_reference is None:
                raise DataError("Geographic source context required")
            parents += (FileParent(geography, geography_reference),)
            geometry_path = geography.with_name("CNTR_BN_20M_2020_4326.geojson")
            if geometry_path.exists():
                if not geometry_reference:
                    raise DataError("Geometry source context required")
                parents += (FileParent(geometry_path, geometry_reference),)
    sources = (
        resolve_parents(
            (raw_snapshot_dir,) + tuple(snapshot_path(item[0]) for item in geographic_parents)
        ).sources
        if trusted is not None
        else resolve_parents(parents).sources
    )
    authorize_sources(
        sources,
        purpose="graph_analysis",
        operation="energy_graph_create",
        destination=root,
    )
    data, split, _ = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw_snapshot_dir, broker_context=broker_context
    )
    if type(fold) is not int or not 0 <= fold < len(split.folds):
        raise DataError("Unknown graph fold")
    order = data.sensor_ids
    n = len(order)
    source: dict[str, object] = {}
    evidence: object = []
    train = split.folds[fold].partition("train")
    if spec.method == "identity":
        a = np.eye(n)
    elif spec.method == "empty":
        a = np.zeros((n, n))
    elif spec.method == "statistical":
        a, source = statistical_edges(data.frame.iloc[train.start : train.stop], spec)
        source |= {
            "training_rows": [train.start, train.stop],
            "training_first": str(data.timestamps[train.start]),
            "training_last": str(data.timestamps[train.stop - 1]),
            "fold": split.folds[fold].fold_id,
        }
    else:
        if broker_context is not None:
            if geography_artifact_reference is None or geography_relative_path is None:
                raise DataError("Geographic artifact identity unavailable")
            broker_source: dict[str, object] = {}
            a, evidence = geographic_edges(
                None,
                order,
                destination=root,
                broker_context=broker_context,
                artifact_reference=geography_artifact_reference,
                relative_path=geography_relative_path,
                geometry_artifact_reference=geometry_artifact_reference,
                geometry_relative_path=geometry_relative_path,
                broker_provenance=broker_source,
            )
            source = {
                "brokered_synthetic_sources": broker_source,
                "source_vintage": "invented synthetic fixture",
                "rights": "Synthetic offline fixture; no real-data licensing assertion",
                "historical_validity": "Synthetic only; no real geographic validity assertion",
            }
        else:
            if geography is None:
                raise DataError("Geographic evidence table required")
            a, evidence = geographic_edges(
                geography,
                order,
                source_reference=geography_reference,
                geometry_reference=geometry_reference,
                destination=root,
            )
            source = {
                "sha256": stream_file(geography)[1],
                "bytes": stream_file(geography)[0],
                "reference": "https://gisco-services.ec.europa.eu/distribution/v2/countries/csv/CNTR_RG_BN_20M_2020.csv",
                "source_vintage": "2020",
                "rights": (
                    "Noncommercial research; © EuroGeographics for the administrative boundaries"
                ),
                "historical_validity": (
                    "2020 source used as retrospective static geographic proxy; "
                    "historical topology not certified"
                ),
            }
        if spec.method == "random-control":
            # Permutes node labels; preserves symmetry, degrees as a multiset and weights.
            p = np.random.default_rng(spec.seed).permutation(n)
            a = a[np.ix_(p, p)]
            source["permutation"] = p.tolist()
            evidence = []
    edge_index, edge_weight = sparse_edges(a)
    desc = {
        "schema_version": "energy-graph-artifact-1.0",
        "specification": spec.to_dict(),
        "canonical": data.lineage,
        "plan_id": plan_path.name,
        "split_id": identity(split.to_dict()),
        "node_order": list(order),
        "registry_id": country_registry().registry_id,
        "fold_id": split.folds[fold].fold_id,
        "source": source,
        "edge_evidence": evidence,
        "direction": "source row to destination column; explicitly symmetric",
        "self_loop_policy": "unit" if spec.method == "identity" else "none",
        "evaluation_first": str(data.timestamps[0]),
        "evaluation_last": str(data.timestamps[-1]),
        "physical_topology_verified": False,
        "diagnostics": diagnostics(a),
    }
    graph_id = identity(desc | {"adjacency": a.tolist()})
    with tempfile.TemporaryDirectory(prefix="energy-graph-") as tmp:
        stage = Path(tmp)
        np.savez(
            stage / "graph.npz",
            adjacency=a,
            edge_index=edge_index,
            edge_weight=edge_weight,
            normalized=normalize(a, spec.normalization),
        )
        write_json(stage / "descriptor.json", desc | {"graph_id": graph_id})
        publication = _publish_governed(
            stage,
            root,
            "energy-graph",
            parents=parents,
            broker_context=broker_context if trusted is not None else None,
            broker_parents=trusted,
        )
    if isinstance(publication, PendingPublication):
        return publication
    result = require_committed_snapshot(publication)
    verify_energy_graph(
        result,
        canonical=canonical,
        plan=plan,
        broker_context=broker_context,
        raw_snapshot_dir=raw_snapshot_dir,
        geography=geography if uses_geography else None,
        geography_reference=geography_reference if uses_geography else None,
        geometry_reference=geometry_reference if uses_geography else None,
        geography_snapshot=geography_snapshot if uses_geography else None,
        geography_artifact_reference=(geography_artifact_reference if uses_geography else None),
        geography_relative_path=geography_relative_path if uses_geography else None,
        geometry_snapshot=geometry_snapshot if uses_geography else None,
        geometry_artifact_reference=geometry_artifact_reference if uses_geography else None,
        geometry_relative_path=geometry_relative_path if uses_geography else None,
        expected_order=order,
        expected_plan=plan_path.name,
    )
    return result


def verify_energy_graph(
    directory: Path | DatasetSnapshot,
    *,
    canonical: Path | DatasetSnapshot | None = None,
    plan: Path | DatasetSnapshot | None = None,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    geography: Path | None = None,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
    geography_snapshot: Path | DatasetSnapshot | None = None,
    geography_artifact_reference: BrokerArtifactRef | None = None,
    geography_relative_path: str | None = None,
    geometry_snapshot: Path | DatasetSnapshot | None = None,
    geometry_artifact_reference: BrokerArtifactRef | None = None,
    geometry_relative_path: str | None = None,
    expected_order: tuple[str, ...] | None = None,
    expected_plan: str | None = None,
) -> tuple[Matrix, dict[str, object]]:
    if (
        canonical is None
        or plan is None
        or raw_snapshot_dir is None
        or (geography is not None and not geography_reference)
    ):
        raise DataError("Complete current source context required for energy graph read")
    if broker_context is not None and geography is not None:
        raise DataError(
            "Broker-backed graph verification does not accept path-only geographic input"
        )
    if (
        broker_context is not None
        and geography_snapshot is not None
        and (geography_artifact_reference is None or not geography_relative_path)
    ):
        raise DataError("Broker-backed graph verification requires geographic artifact identity")
    if (geometry_snapshot is None) != (geometry_artifact_reference is None) or (
        geometry_snapshot is None
    ) != (geometry_relative_path is None):
        raise DataError("Geometry snapshot, identity and path must be supplied together")
    geographic_parents: tuple[tuple[Path | DatasetSnapshot, BrokerArtifactRef], ...] = ()
    if geography_snapshot is not None and geography_artifact_reference is not None:
        geographic_parents = ((geography_snapshot, geography_artifact_reference),)
    if geometry_snapshot is not None and geometry_artifact_reference is not None:
        geographic_parents += ((geometry_snapshot, geometry_artifact_reference),)
    trusted = authorize_graph_inputs(
        canonical,
        plan,
        graph=directory,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context,
        purpose="graph_analysis",
        operation="energy_graph_read",
        geographic_parents=geographic_parents,
    )
    parents: tuple[Path | FileParent, ...] = (snapshot_path(canonical), snapshot_path(plan))
    if broker_context is not None:
        parents += tuple(snapshot_path(item[0]) for item in geographic_parents)
    elif geography is not None:
        if geography_reference is None:
            raise DataError("Geographic source context required")
        parents += (FileParent(geography, geography_reference),)
        geometry_path = geography.with_name("CNTR_BN_20M_2020_4326.geojson")
        if geometry_path.exists():
            if geometry_reference is None:
                raise DataError("Geometry source context required")
            parents += (FileParent(geometry_path, geometry_reference),)
    if trusted is not None:
        assert broker_context is not None
        graph_reference = trusted[-1]
        base = governed_payload(
            graph_reference,
            {"graph.npz", "descriptor.json"},
            kind="energy-graph",
            broker=broker_context.broker,
            broker_parents=trusted[:-1],
            materialization_root=broker_context.materialization_root,
        )
    else:
        base = governed_payload(
            snapshot_path(directory),
            {"graph.npz", "descriptor.json"},
            parents=parents,
            kind="energy-graph",
        )
    desc = read_json(base / "descriptor.json")
    nodes = desc.get("node_order")
    if not isinstance(nodes, list) or any(not isinstance(v, str) for v in nodes):
        raise DataError("Invalid node axis")
    order = tuple(str(v) for v in nodes)
    country_registry().axis(order)
    if (
        desc.get("schema_version") != "energy-graph-artifact-1.0"
        or desc.get("registry_id") != country_registry().registry_id
        or (expected_order is not None and order != expected_order)
        or (expected_plan is not None and expected_plan != desc.get("plan_id"))
    ):
        raise DataError("Graph identity/axis/plan mismatch; explicit realignment required")
    with np.load(base / "graph.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"adjacency", "edge_index", "edge_weight", "normalized"}:
            raise DataError("Graph numeric inventory mismatch")
        a = np.asarray(saved["adjacency"], dtype=np.float64)
        validate_adjacency(a)
        if a.shape != (len(order), len(order)):
            raise DataError("Graph shape mismatch")
        edges, weights = sparse_edges(a)
        if not np.array_equal(edges, saved["edge_index"]) or not np.array_equal(
            weights, saved["edge_weight"]
        ):
            raise DataError("Sparse graph mismatch")
        specification = object_dict(desc["specification"])
        if not np.array_equal(
            normalize(a, str(specification["normalization"])), saved["normalized"]
        ):
            raise DataError("Graph normalization mismatch")
    graph_id = desc.pop("graph_id")
    if identity(desc | {"adjacency": a.tolist()}) != graph_id or desc["diagnostics"] != diagnostics(
        a
    ):
        raise DataError("Graph content identity mismatch")
    desc["graph_id"] = graph_id
    return a, desc
