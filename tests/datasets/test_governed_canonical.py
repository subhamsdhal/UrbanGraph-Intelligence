"""Actual invented HDF -> canonical Parquet -> later governed read lifecycle."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd
import pytest

from urbangraph.data import DataError, RawFileRecord, content_identity, ingest_dataset
from urbangraph.data.filesystem import discover, stream_file
from urbangraph.datasets.canonical import assert_equivalent, read_observations
from urbangraph.datasets.governed_canonical import create_canonical, read_canonical
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.materialization import (
    load_materialization,
    materialize_snapshot,
    verify_materialization,
)
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.dataset import open_dataset, open_governed_dataset
from urbangraph.graphs.artifacts import build_graph, load_graph, publish_graph, verify_graph
from urbangraph.graphs.contracts import GraphSpecification
from urbangraph.graphs.governed import (
    create_governed_geographic_graph,
    create_governed_road_graph,
    read_governed_geographic_graph,
    read_governed_road_graph,
)


class GraphArguments(TypedDict):
    raw_snapshot_dir: Path
    canonical_dir: Path
    locations: Path
    distances: Path
    location_reference: str
    distance_reference: str


def _grant(
    source_id: str,
    digest: str,
    source: Path,
    operation: str,
    destination: Path,
    *,
    purpose: str = "local_research",
) -> dict[str, object]:
    now = datetime.now(UTC)
    return {
        "source_id": source_id,
        "source_sha256": digest,
        "source_path": str(source),
        "purpose": purpose,
        "operation": operation,
        "destination_root": str(destination),
        "evidence_reference": "invented-canonical-lifecycle-test-only",
        "status": "approved",
        "effective_utc": (now - timedelta(minutes=1)).isoformat(),
        "expires_utc": (now + timedelta(minutes=10)).isoformat(),
        "retention_days": 1,
    }


def _setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    source = tmp_path / "invented-source"
    source.mkdir()
    frame.to_hdf(source / "metr-la.h5", key="traffic", mode="w")
    records = tuple(
        RawFileRecord(relative, *stream_file(source / relative))
        for relative in discover(source).files
    )
    raw_root, output_root = tmp_path / "raw", tmp_path / "canonical"
    grants = [_grant("metr-la", content_identity(records), source, "raw_ingest", raw_root)]
    policy = tmp_path / "invented-policy.json"

    def write() -> None:
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    write()
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    snapshot = ingest_dataset(source, "metr-la", raw_root=raw_root).directory
    identity = snapshot.name
    grants.extend(
        (
            _grant("metr-la", identity, snapshot, "canonical_create", output_root),
            _grant("metr-la", identity, snapshot, "snapshot_parse", snapshot.parent),
        )
    )
    write()
    artifact = create_canonical(snapshot, output_root=output_root).directory
    grants.append(_grant("metr-la", identity, snapshot, "canonical_read", artifact))
    write()
    return snapshot, artifact, policy, grants


def test_actual_canonical_lifecycle_rechecks_current_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    loaded = read_canonical(artifact, raw_snapshot_dir=snapshot)
    assert_equivalent(parse_snapshot(snapshot).observations, loaded)
    assert len(loaded) == len(traffic_frame)
    downstream = open_governed_dataset(artifact, raw_snapshot_dir=snapshot)
    assert_equivalent(loaded, downstream.frame)
    assert downstream.lineage["timestamp_count"] == len(loaded)
    grants[-1]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        read_canonical(artifact, raw_snapshot_dir=snapshot)
    with pytest.raises(DataError, match="not approved"):
        open_governed_dataset(artifact, raw_snapshot_dir=snapshot)
    grants[-1]["status"] = "approved"
    grants[-1]["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="outside validity window"):
        read_canonical(artifact, raw_snapshot_dir=snapshot)


def test_canonical_read_denies_wrong_destination_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    with pytest.raises(DataError, match="No matching rights grant"):
        read_canonical(tmp_path / "other", raw_snapshot_dir=snapshot)
    (artifact / "quality.json").write_bytes(b"tampered")
    with pytest.raises(DataError, match="integrity verification"):
        read_canonical(artifact, raw_snapshot_dir=snapshot)
    assert policy.exists() and len(grants) == 4


@pytest.mark.parametrize(
    "denial", ["missing", "revoked", "expired", "wrong-purpose", "wrong-digest"]
)
def test_canonical_verification_denies_before_artifact_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    traffic_frame: pd.DataFrame,
    denial: str,
) -> None:
    snapshot, artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    grant = grants[-1]
    if denial == "missing":
        grants.remove(grant)
    elif denial == "revoked":
        grant["status"] = "revoked"
    elif denial == "expired":
        grant["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    elif denial == "wrong-purpose":
        grant["purpose"] = "model_training"
    else:
        grant["source_sha256"] = "0" * 64
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("canonical artifact read occurred before current authorization")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.datasets.materialization._load_verified", forbidden)
        result = verify_materialization(artifact, raw_snapshot_dir=snapshot)
    assert not result.valid
    assert "canonical artifact read occurred" not in " ".join(result.issues)
    assert "private-observation-marker" not in " ".join(result.issues)


def test_context_free_canonical_verification_denies_before_artifact_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    traffic_frame: pd.DataFrame,
) -> None:
    _snapshot, artifact, _policy, _grants = _setup(tmp_path, monkeypatch, traffic_frame)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("context-free canonical verification read protected artifacts")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.datasets.materialization._load_verified", forbidden)
        result = verify_materialization(artifact)
    assert not result.valid
    assert any("identity required" in issue for issue in result.issues)


def test_canonical_verify_cli_requires_snapshot_context_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    traffic_frame: pd.DataFrame,
) -> None:
    _snapshot, artifact, _policy, _grants = _setup(tmp_path, monkeypatch, traffic_frame)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("context-free CLI verification read protected artifacts")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.datasets.materialization._load_verified", forbidden)
        with pytest.raises(SystemExit) as exc:
            from urbangraph.datasets.cli import main

            main(["verify", "--artifact-dir", str(artifact)])
    assert exc.value.code == 2


def test_canonical_creation_denied_without_derived_grant_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, _artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    grants[1]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    blocked = tmp_path / "blocked-output"
    with pytest.raises(DataError, match="No matching rights grant"):
        create_canonical(snapshot, output_root=blocked)
    assert not blocked.exists()


def test_missing_policy_and_changed_upstream_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, _policy, _grants = _setup(tmp_path, monkeypatch, traffic_frame)
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY")
    with pytest.raises(DataError, match="policy unavailable"):
        read_canonical(artifact, raw_snapshot_dir=snapshot)
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(tmp_path / "invented-policy.json"))
    (snapshot / "payload" / "metr-la.h5").write_bytes(b"corrupted")
    with pytest.raises(
        DataError,
        match="upstream raw snapshot failed integrity|Upstream raw snapshot failed integrity",
    ):
        read_canonical(artifact, raw_snapshot_dir=snapshot)


def test_forged_lineage_and_missing_upstream_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, _policy, _grants = _setup(tmp_path, monkeypatch, traffic_frame)
    with pytest.raises(DataError):
        read_canonical(artifact, raw_snapshot_dir=tmp_path / "missing-snapshot")
    manifest_path = artifact / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["descriptor"]["raw_snapshot_id"] = "0" * 64
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DataError, match="integrity verification"):
        read_canonical(artifact, raw_snapshot_dir=snapshot)


def test_direct_canonical_interfaces_require_current_grants_before_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    assert len(load_materialization(artifact, raw_snapshot_dir=snapshot)) == len(traffic_frame)
    assert len(open_dataset(artifact, raw_snapshot_dir=snapshot).frame) == len(traffic_frame)
    grants[-1]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    def premature_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("canonical bytes consumed before authorization")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.datasets.materialization._load_verified", premature_load)
        with pytest.raises(DataError, match="not approved"):
            load_materialization(artifact, raw_snapshot_dir=snapshot)
        with pytest.raises(DataError, match="not approved"):
            open_dataset(artifact, raw_snapshot_dir=snapshot)
        with pytest.raises(DataError, match="identity required"):
            load_materialization(artifact)
        with pytest.raises(DataError, match="identity required"):
            open_dataset(artifact)
    grants[1]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    blocked = tmp_path / "blocked-canonical"
    with pytest.raises(DataError, match="No matching rights grant"):
        materialize_snapshot(snapshot, output_root=blocked)
    assert not blocked.exists()


def test_direct_observation_reader_requires_current_rights_before_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, artifact, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    path = artifact / "observations.parquet"
    schema = read_json(artifact / "schema.json")
    assert_equivalent(
        read_canonical(artifact, raw_snapshot_dir=snapshot),
        read_observations(path, schema, raw_snapshot_dir=snapshot),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Parquet read before current permission")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.datasets.canonical.pq.read_table", forbidden)
        with pytest.raises(DataError, match="source context required"):
            read_observations(path, schema)
        grants[-1]["status"] = "revoked"
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        with pytest.raises(DataError, match="not approved"):
            read_observations(path, schema, raw_snapshot_dir=snapshot)


def test_direct_graph_interfaces_deny_without_complete_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def premature_load(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("graph bytes consumed before authorization")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.graphs.artifacts._load_graph_unchecked", premature_load)
        with pytest.raises(DataError, match="Complete governed graph source context required"):
            load_graph(tmp_path / "graph")
    output = tmp_path / "graph-output"
    with pytest.raises(DataError, match="governed graph creation"):
        publish_graph(pd.DataFrame().to_numpy(), {}, output)
    with pytest.raises(DataError, match="Complete governed graph configuration"):
        build_graph({})
    assert not output.exists()


def test_three_source_graph_creation_reuse_and_revocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, canonical, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    locations = tmp_path / "locations.csv"
    distances = tmp_path / "distances.csv"
    locations.write_text(
        "sensor_id,latitude,longitude\nsensor-Z,34,-118\n001,35,-119\nA,36,-120\n",
        encoding="utf-8",
    )
    distances.write_text(
        "from,to,cost\nsensor-Z,001,2\n001,A,3\nA,sensor-Z,4\n",
        encoding="utf-8",
    )
    root = tmp_path / "graphs"
    location_hash, distance_hash = stream_file(locations)[1], stream_file(distances)[1]
    for source_id, digest, path in (
        ("invented-locations", location_hash, locations),
        ("invented-distances", distance_hash, distances),
    ):
        grants.append(
            _grant(source_id, digest, path, "graph_create", root, purpose="graph_analysis")
        )
        grants.append(
            _grant(source_id, digest, path, "graph_csv_read", path.parent, purpose="graph_analysis")
        )
    grants.append(
        _grant(
            "metr-la",
            snapshot.name,
            snapshot,
            "canonical_graph_create",
            root,
            purpose="graph_analysis",
        )
    )
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    arguments: GraphArguments = {
        "raw_snapshot_dir": snapshot,
        "canonical_dir": canonical,
        "locations": locations,
        "distances": distances,
        "location_reference": "invented-locations",
        "distance_reference": "invented-distances",
    }
    builder_config: dict[str, object] = {
        **{key: str(value) for key, value in arguments.items()},
        "output_root": str(root),
        "specification": GraphSpecification("metr-la").to_dict(),
    }
    distance_create = next(
        grant
        for grant in grants
        if grant["source_id"] == "invented-distances" and grant["operation"] == "graph_create"
    )
    distance_create["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    def premature_read(_source: Path, _digest: str) -> None:
        raise AssertionError("protected source was inspected before complete authorization")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.graphs.governed._verify_raw", premature_read)
        with pytest.raises(DataError, match="not approved"):
            create_governed_road_graph(
                **arguments, output_root=root, specification=GraphSpecification("metr-la")
            )
        with pytest.raises(DataError, match="not approved"):
            build_graph(builder_config)
    assert not root.exists()
    distance_create["status"] = "approved"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    graph = create_governed_road_graph(
        **arguments, output_root=root, specification=GraphSpecification("metr-la")
    )
    assert graph.adjacency.shape == (3, 3)
    via_builder = build_graph(builder_config)
    assert via_builder.graph_id == graph.graph_id
    for source_id, digest, path in (
        ("invented-locations", location_hash, locations),
        ("invented-distances", distance_hash, distances),
    ):
        grants.append(
            _grant(
                source_id, digest, path, "graph_reuse", graph.directory, purpose="graph_analysis"
            )
        )
    grants.append(
        _grant(
            "metr-la",
            snapshot.name,
            snapshot,
            "canonical_graph_read",
            graph.directory,
            purpose="graph_analysis",
        )
    )
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    restored = read_governed_road_graph(graph.directory, **arguments)
    assert restored.graph_id == graph.graph_id
    assert load_graph(graph.directory, **arguments).graph_id == graph.graph_id
    assert verify_graph(graph.directory, **arguments).valid
    cli_context = tmp_path / "graph-verification-context.json"
    cli_context.write_text(
        json.dumps({key: str(value) for key, value in arguments.items()}), encoding="utf-8"
    )
    from urbangraph.graphs.cli import main as graph_main

    assert (
        graph_main(["verify", "--graph-dir", str(graph.directory), "--config", str(cli_context)])
        == 0
    )
    grants[-2]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        read_governed_road_graph(graph.directory, **arguments)
    with pytest.raises(DataError, match="not approved"):
        load_graph(graph.directory, **arguments)
        with guarded.context() as denied_loader:
            denied_loader.setattr(
                "urbangraph.graphs.artifacts._load_graph_unchecked", premature_read
            )
            denied = verify_graph(graph.directory, **arguments)
            assert not denied.valid
    grants[-2]["status"] = "approved"
    grants[-2]["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="outside validity window"):
        read_governed_road_graph(graph.directory, **arguments)
    assert not verify_graph(graph.directory, **arguments).valid
    grants[-2]["expires_utc"] = (datetime.now(UTC) + timedelta(minutes=10)).isoformat()
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    descriptor = graph.directory / "graph.json"
    forged = json.loads(descriptor.read_text(encoding="utf-8"))
    forged["sources"].pop()
    descriptor.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(DataError, match="upstream set"):
        read_governed_road_graph(graph.directory, **arguments)


def test_geographic_graph_current_rights_and_complete_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, traffic_frame: pd.DataFrame
) -> None:
    snapshot, canonical, policy, grants = _setup(tmp_path, monkeypatch, traffic_frame)
    locations = tmp_path / "invented-geo.csv"
    locations.write_text(
        "sensor_id,latitude,longitude\nsensor-Z,34,-118\n001,35,-119\nA,36,-120\n",
        encoding="utf-8",
    )
    root = tmp_path / "geo-output"
    reference = "invented-geo-only"
    digest = stream_file(locations)[1]
    grants.extend(
        (
            _grant(reference, digest, locations, "graph_create", root, purpose="graph_analysis"),
            _grant(
                reference,
                digest,
                locations,
                "graph_csv_read",
                locations.parent,
                purpose="graph_analysis",
            ),
            _grant(
                "metr-la",
                snapshot.name,
                snapshot,
                "canonical_graph_create",
                root,
                purpose="graph_analysis",
            ),
        )
    )
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    spec = GraphSpecification("metr-la", method="geographic-knn", k=1, radius_km=500.0)
    config: dict[str, object] = {
        "specification": spec.to_dict(),
        "raw_snapshot_dir": str(snapshot),
        "canonical_dir": str(canonical),
        "locations": str(locations),
        "location_reference": reference,
        "coordinate_reference": "invented test coordinates",
        "output_root": str(root),
    }
    geo_grant = grants[-3]
    geo_grant["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        build_graph(config)
    assert not root.exists()
    geo_grant["status"] = "approved"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    graph = create_governed_geographic_graph(
        raw_snapshot_dir=snapshot,
        canonical_dir=canonical,
        locations=locations,
        location_reference=reference,
        coordinate_reference="invented test coordinates",
        output_root=root,
        specification=spec,
    )
    assert build_graph(config).graph_id == graph.graph_id
    grants.extend(
        (
            _grant(
                reference,
                digest,
                locations,
                "graph_reuse",
                graph.directory,
                purpose="graph_analysis",
            ),
            _grant(
                "metr-la",
                snapshot.name,
                snapshot,
                "canonical_graph_read",
                graph.directory,
                purpose="graph_analysis",
            ),
        )
    )
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    assert (
        read_governed_geographic_graph(
            graph.directory,
            raw_snapshot_dir=snapshot,
            canonical_dir=canonical,
            locations=locations,
            location_reference=reference,
        ).graph_id
        == graph.graph_id
    )
    assert (
        load_graph(
            graph.directory,
            raw_snapshot_dir=snapshot,
            canonical_dir=canonical,
            locations=locations,
            location_reference=reference,
        ).graph_id
        == graph.graph_id
    )
    grants[-2]["status"] = "revoked"
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
    with pytest.raises(DataError, match="not approved"):
        load_graph(
            graph.directory,
            raw_snapshot_dir=snapshot,
            canonical_dir=canonical,
            locations=locations,
            location_reference=reference,
        )
