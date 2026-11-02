import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.data.filesystem import stream_file
from urbangraph.graphs.artifacts import (
    _load_graph_unchecked,
    _publish_graph_unchecked,
    verify_graph,
)
from urbangraph.graphs.cli import main
from urbangraph.graphs.construction import (
    apply_policies,
    diagnostics,
    geographic_knn,
    haversine,
    road_gaussian,
)
from urbangraph.graphs.contracts import GraphSpecification, SensorNodeIndex
from urbangraph.graphs.normalization import normalize, perturb, sparse_edges
from urbangraph.graphs.sources import read_distances, read_locations


def authorize_invented_csv(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind this test's one CSV version to its exact read operation and directory."""
    now = datetime.now(UTC)
    record = {
        "source_id": "fixture",
        "source_sha256": stream_file(path)[1],
        "source_path": str(path),
        "purpose": "graph_analysis",
        "operation": "graph_csv_read",
        "destination_root": str(path.parent),
        "evidence_reference": "invented-graph-unit-test-only",
        "status": "approved",
        "effective_utc": (now - timedelta(minutes=1)).isoformat(),
        "expires_utc": (now + timedelta(minutes=10)).isoformat(),
        "retention_days": 1,
    }
    policy = path.parent / "invented-graph-rights.json"
    policy.write_text(json.dumps({"version": 1, "grants": [record]}), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))


def test_sensor_permutation_and_rejection() -> None:
    index = SensorNodeIndex(("b", "a"), ("a", "b"))
    assert index.permutation == (1, 0)
    assert index.fingerprint != SensorNodeIndex(("a", "b"), ("a", "b")).fingerprint
    for values in (("a", "a"), ("a", "missing"), (" a", "b")):
        with pytest.raises(DataError):
            SensorNodeIndex(("a", "b"), values)


def test_location_layouts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "locations.csv"
    p.write_text("index,sensor_id,latitude,longitude\n0,a,34,-118\n1,b,35,-119\n")
    authorize_invented_csv(p, monkeypatch)
    index, coords, source = read_locations(p, ("b", "a"), reference="fixture")
    assert index.permutation == (1, 0)
    np.testing.assert_array_equal(coords, [[35, -119], [34, -118]])
    assert source.sha256
    p.write_text("a,34,-118\nb,35,-119\n")
    authorize_invented_csv(p, monkeypatch)
    _, coords, _ = read_locations(p, ("b", "a"), reference="fixture", layout="bay-headerless")
    assert coords is None
    _, coords, _ = read_locations(
        p,
        ("b", "a"),
        reference="fixture",
        layout="bay-headerless",
        coordinate_reference="synthetic declared lat/lon",
    )
    np.testing.assert_array_equal(coords, [[35, -119], [34, -118]])
    p.write_text("a,91,0\nb,35,0\n")
    authorize_invented_csv(p, monkeypatch)
    with pytest.raises(DataError, match="range"):
        read_locations(
            p,
            ("a", "b"),
            reference="fixture",
            layout="bay-headerless",
            coordinate_reference="fixture",
        )
    unsafe = tmp_path / "untrusted.pkl"
    unsafe.write_bytes(b"invented invalid CSV suffix")
    authorize_invented_csv(unsafe, monkeypatch)
    with pytest.raises(DataError, match="pickle"):
        read_locations(unsafe, ("a",), reference="fixture")


def test_distance_policies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "distances.csv"
    p.write_text("from,to,cost\na,a,0\na,b,2\nb,a,5\nx,y,9\n")
    authorize_invented_csv(p, monkeypatch)
    idx = SensorNodeIndex(("b", "a"), ("a", "b"))
    d, _, details = read_distances(
        p,
        idx,
        reference="fixture",
        layout="distance-header",
        duplicate_policy="reject",
        outside_policy="filter",
    )
    assert d[0, 1] == 5 and d[1, 0] == 2 and details["filtered_outside_rows"] == 1
    with pytest.raises(DataError, match="Unknown"):
        read_distances(
            p,
            idx,
            reference="fixture",
            layout="distance-header",
            duplicate_policy="reject",
            outside_policy="reject",
        )
    p.write_text("a,b,2\na,b,3\n")
    authorize_invented_csv(p, monkeypatch)
    with pytest.raises(DataError, match="Duplicate"):
        read_distances(
            p,
            idx,
            reference="fixture",
            layout="distance-headerless",
            duplicate_policy="reject",
            outside_policy="reject",
        )
    d, _, details = read_distances(
        p,
        idx,
        reference="fixture",
        layout="distance-headerless",
        duplicate_policy="last",
        outside_policy="reject",
    )
    assert d[1, 0] == 3 and details["duplicate_rows"] == 1
    for distance in ("nan", "inf", "-1", "bad"):
        p.write_text(f"a,b,{distance}\n")
        authorize_invented_csv(p, monkeypatch)
        with pytest.raises(DataError):
            read_distances(
                p,
                idx,
                reference="fixture",
                layout="distance-headerless",
                duplicate_policy="reject",
                outside_policy="reject",
            )


def test_gaussian_direction_and_threshold() -> None:
    d = np.array([[0.0, 2.0], [5.0, np.inf]])
    a, std = road_gaussian(d, 0)
    assert std == np.std([0.0, 2.0, 5.0])
    np.testing.assert_allclose(a, [[1, np.exp(-((2 / std) ** 2))], [np.exp(-((5 / std) ** 2)), 0]])
    threshold = float(a[0, 1])
    filtered, _ = road_gaussian(d, threshold)
    assert filtered[0, 1] == threshold and filtered[1, 0] == 0
    with pytest.raises(DataError):
        road_gaussian(np.zeros((2, 2)), 0.1)
    spec = GraphSpecification("metr-la")
    np.testing.assert_array_equal(apply_policies(a, spec), a)
    assert np.diag(apply_policies(a, replace(spec, self_loops="remove"))).sum() == 0
    assert np.diag(apply_policies(a, replace(spec, self_loops="unit"))).sum() == 2
    symmetric = apply_policies(a, replace(spec, direction="symmetric-max"))
    np.testing.assert_array_equal(symmetric, symmetric.T)


def test_operators_and_isolates() -> None:
    a = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        normalize(a, "random-walk"), np.array([[0.0, 1.0, 0.0], [0.25, 0.0, 0.75], [0.0, 0.0, 0.0]])
    )
    np.testing.assert_allclose(normalize(a, "reverse-walk"), [[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(DataError, match="symmetric"):
        normalize(a, "symmetric")
    symmetric = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        normalize(symmetric, "laplacian"), [[1, -1, 0], [-1, 1, 0], [0, 0, 0]], atol=1e-15
    )
    assert np.isfinite(normalize(symmetric, "scaled-laplacian")).all()
    edge_index, weight = sparse_edges(a)
    np.testing.assert_array_equal(edge_index, [[0, 1, 1], [1, 0, 2]])
    np.testing.assert_array_equal(weight, [2, 1, 3])
    diag = diagnostics(symmetric)
    assert diag["isolated_without_self_loops"] == 1 and diag["weak_component_sizes"] == [2, 1]


def test_geographic_and_perturbations() -> None:
    coords = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 3.0]])
    assert haversine(coords)[0, 1] == pytest.approx(111.19508, rel=1e-6)
    spec = GraphSpecification(
        "metr-la",
        method="geographic-knn",
        k=1,
        radius_km=150,
        geographic_weight="inverse-distance",
        threshold=0,
    )
    a = geographic_knn(coords, spec)
    assert a[0, 1] > 0 and a[1, 0] > 0 and not a[2].any()
    np.testing.assert_array_equal(perturb(a, "identity"), np.eye(3))
    assert not perturb(a, "remove-edges").any()
    np.testing.assert_array_equal(perturb(a, "unweighted"), (a > 0))
    np.testing.assert_array_equal(
        np.sort(perturb(a, "randomize-destinations"), axis=1), np.sort(a, axis=1)
    )
    np.testing.assert_array_equal(
        perturb(a, "permute-weights", seed=7), perturb(a, "permute-weights", seed=7)
    )
    perm = [2, 0, 1]
    np.testing.assert_allclose(
        normalize(a[np.ix_(perm, perm)], "random-walk"),
        normalize(a, "random-walk")[np.ix_(perm, perm)],
    )


def descriptor() -> dict[str, object]:
    return {
        "schema_version": "graph-1.0",
        "specification": GraphSpecification("metr-la").to_dict(),
        "canonical": {"dataset_id": "metr-la", "canonical_id": "synthetic"},
        "index": SensorNodeIndex(("a", "b"), ("b", "a")).to_dict(),
        "sources": [
            {
                "name": "synthetic.csv",
                "sha256": "0" * 64,
                "size_bytes": 1,
                "reference": "synthetic fixture",
                "layout": "distance-header",
            }
        ],
        "construction": {},
        "orientation": "source-row,destination-column",
    }


def test_artifact_identity_reuse_corruption_cli(tmp_path: Path) -> None:
    a = np.array([[1.0, 0.3], [0.7, 1.0]])
    first = _publish_graph_unchecked(a, descriptor(), tmp_path / "graphs")
    before = {p.name: p.read_bytes() for p in first.directory.iterdir()}
    second = _publish_graph_unchecked(a, descriptor(), tmp_path / "graphs")
    assert first.graph_id == second.graph_id
    assert before == {p.name: p.read_bytes() for p in first.directory.iterdir()}
    # Context-free verification is now a fail-closed denial, not an integrity bypass.
    assert not verify_graph(first.directory).valid
    with pytest.raises(SystemExit) as missing_context:
        main(["verify", "--graph-dir", str(first.directory)])
    assert missing_context.value.code == 2
    with pytest.raises(SystemExit) as denied:
        main(["inspect", "--graph-dir", str(first.directory)])
    assert denied.value.code == 2
    # Verification now requires independently supplied source authority. The
    # legacy context-free route remains a parser-level denial.
    with pytest.raises(SystemExit) as missing_context:
        main(["verify", "--graph-dir", str(first.directory)])
    assert missing_context.value.code == 2
    a[0, 1] = 0.4
    assert _publish_graph_unchecked(a, descriptor(), tmp_path / "graphs").graph_id != first.graph_id
    (first.directory / "adjacency.npz").write_bytes(b"corrupt")
    assert not verify_graph(first.directory).valid
    with pytest.raises(DataError, match="checksum"):
        _load_graph_unchecked(first.directory)
    with pytest.raises(SystemExit) as missing_context:
        main(["verify", "--graph-dir", str(first.directory)])
    assert missing_context.value.code == 2
    a[0, 1] = 0.3
    with pytest.raises(DataError):
        _publish_graph_unchecked(a, descriptor(), tmp_path / "graphs")


def test_publish_failure_cleans_owned_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise OSError("simulated finalization failure")

    monkeypatch.setattr("urbangraph.graphs.artifacts._publish", fail)
    with pytest.raises(OSError, match="simulated"):
        _publish_graph_unchecked(np.eye(2), descriptor(), tmp_path / "graphs")
    assert list((tmp_path / "graphs").iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        {"direction": "implicit"},
        {"k": True},
        {"threshold": float("nan")},
        {"self_loops": "guess"},
        {"dataset_id": "../bad"},
    ],
)
def test_bad_specification(change: dict[str, object]) -> None:
    with pytest.raises(DataError):
        GraphSpecification(**(GraphSpecification("metr-la").to_dict() | change))  # type: ignore[arg-type]
