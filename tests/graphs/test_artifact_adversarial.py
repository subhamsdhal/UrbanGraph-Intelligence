import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.dataset import identity
from urbangraph.graphs.artifacts import (
    _load_graph_unchecked,
    _publish_graph_unchecked,
    verify_graph,
)
from urbangraph.graphs.contracts import GraphSpecification, SensorNodeIndex


def make_graph(root: Path) -> Path:
    descriptor: dict[str, object] = {
        "schema_version": "graph-1.0",
        "specification": GraphSpecification("metr-la").to_dict(),
        "canonical": {"dataset_id": "metr-la", "canonical_id": "fixture"},
        "index": SensorNodeIndex(("a", "b"), ("a", "b")).to_dict(),
        "sources": [
            {
                "name": "fixture.csv",
                "sha256": "0" * 64,
                "size_bytes": 0,
                "reference": "synthetic",
                "layout": "distance-header",
            }
        ],
        "construction": {},
        "orientation": "source-row,destination-column",
    }
    # Integrity-only fixture; production publication requires governed upstreams.
    return _publish_graph_unchecked(np.eye(2), descriptor, root).directory


@pytest.mark.parametrize(
    "damage", ["missing", "extra", "invalid-json", "unsupported-schema", "traversal"]
)
def test_graph_inventory_and_manifest_rejection(tmp_path: Path, damage: str) -> None:
    root = make_graph(tmp_path / "graphs")
    if damage == "missing":
        (root / "adjacency.npz").unlink()
    elif damage == "extra":
        (root / "extra.txt").write_text("unexpected")
    elif damage == "invalid-json":
        (root / "manifest.json").write_text("{broken")
    else:
        manifest = read_json(root / "manifest.json")
        if damage == "unsupported-schema":
            manifest["schema_version"] = "unsupported"
        else:
            manifest["files"] = {"../escape": "0" * 64}
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError):
        _load_graph_unchecked(root)


def test_graph_context_free_verification_denies_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_graph(tmp_path / "graphs")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("context-free graph verification accessed the protected graph")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.graphs.artifacts._load_graph_unchecked", forbidden)
        result = verify_graph(root)
    assert not result.valid
    assert any("source context" in issue for issue in result.issues)


def test_graph_cli_context_free_verification_denies_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_graph(tmp_path / "graphs")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("context-free graph CLI accessed the protected graph")

    with monkeypatch.context() as guarded:
        guarded.setattr("urbangraph.graphs.artifacts._load_graph_unchecked", forbidden)
        with pytest.raises(SystemExit) as exc:
            from urbangraph.graphs.cli import main

            main(["verify", "--graph-dir", str(root)])
    assert exc.value.code == 2


def test_object_array_never_unpickled(tmp_path: Path) -> None:
    root = make_graph(tmp_path / "graphs")
    with (root / "adjacency.npz").open("wb") as stream:
        np.savez(stream, adjacency=np.array([["unsafe"]], dtype=object))
    manifest = read_json(root / "manifest.json")
    files = manifest["files"]
    assert isinstance(files, dict)
    files["adjacency.npz"] = hashlib.sha256((root / "adjacency.npz").read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="numeric"):
        _load_graph_unchecked(root)


def test_forged_sensor_permutation_rejected(tmp_path: Path) -> None:
    root = make_graph(tmp_path / "graphs")
    descriptor = read_json(root / "graph.json")
    index = descriptor["index"]
    assert isinstance(index, dict)
    index["canonical_to_source"] = [1, 0]
    (root / "graph.json").write_text(json.dumps(descriptor))
    manifest = read_json(root / "manifest.json")
    files = manifest["files"]
    assert isinstance(files, dict)
    files["graph.json"] = hashlib.sha256((root / "graph.json").read_bytes()).hexdigest()
    manifest["graph_id"] = identity(descriptor)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataError, match="permutation"):
        _load_graph_unchecked(root, staged=True)
