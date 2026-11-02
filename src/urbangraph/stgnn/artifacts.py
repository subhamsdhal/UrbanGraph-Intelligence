"""Graph-bound model bundles with immutable graph copies and safe reload probes."""

from __future__ import annotations

import copy
import os
import tempfile
from pathlib import Path
from pickle import UnpicklingError
from typing import cast

import numpy as np
import torch
from torch import Tensor

from urbangraph.data import DataError
from urbangraph.data.contracts import validate_relative_path
from urbangraph.data.filesystem import copy_file, discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.data.saved_trust import classify_for_publication, verification_status
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import integer
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.graphs.artifacts import load_graph
from urbangraph.graphs.contracts import GraphArtifact
from urbangraph.stgnn.contracts import model_config
from urbangraph.stgnn.models import GraphForecaster


def probe(model: GraphForecaster) -> Tensor:
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(2, model.history, len(model.sensor_ids), model.features, generator=generator)
    x[..., 1] = 1
    x[0, :, 0] = 0
    return x


def restore(root: Path, metadata: object, graph: GraphArtifact) -> GraphForecaster:
    obj = object_dict(metadata)
    if obj.get("graph_id") != graph.graph_id or obj.get("sensor_ids") != list(
        graph.index.sensor_ids
    ):
        raise DataError("Model graph/sensor identity mismatch")
    filename = obj.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
    ):
        raise DataError("Unsafe model filename")
    model = GraphForecaster(
        model_config(obj.get("config")),
        graph,
        integer(obj.get("history"), "history", 1),
        integer(obj.get("horizon"), "horizon", 1),
        graph.index.sensor_ids,
        integer(obj.get("features"), "features", 1),
    )
    if stream_file(root / filename)[0] > 256 * 1024**2:
        raise DataError("Oversized graph model state")
    expected_walks = {
        k: v.clone() for k, v in model.state_dict().items() if k in {"forward_walk", "reverse_walk"}
    }
    try:
        state = torch.load(root / filename, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(not isinstance(v, Tensor) for v in state.values()):
            raise DataError("Expected only tensor state dictionary")
        model.load_state_dict(cast(dict[str, Tensor], state), strict=True)
    except (RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
        raise DataError("Invalid graph model state") from exc
    for key, expected in expected_walks.items():
        if not torch.equal(model.state_dict()[key], expected):
            raise DataError("Checkpoint graph operators differ from declared graph/configuration")
    if obj.get("parameter_count") != model.parameter_count:
        raise DataError("Model parameter-count mismatch")
    return model.eval()


def verify_run(
    directory: str | Path, *, graph_context: dict[str, str] | None = None
) -> dict[str, object]:
    """Verify a saved run only after current rights for its source graph."""
    required = {
        "graph_dir",
        "raw_snapshot_dir",
        "canonical_dir",
        "locations",
        "location_reference",
        "distances",
        "distance_reference",
    }
    if (
        graph_context is None
        or set(graph_context) != required
        or any(not isinstance(value, str) or not value for value in graph_context.values())
    ):
        raise DataError("Complete STGNN source context required")
    from urbangraph.data.rights import require_verified_snapshot_authorization

    require_verified_snapshot_authorization(
        graph_context["raw_snapshot_dir"],
        "scientific_evaluation",
        directory,
        operation="model_evaluate",
    )
    graph = load_graph(
        graph_context["graph_dir"],
        raw_snapshot_dir=graph_context["raw_snapshot_dir"],
        canonical_dir=graph_context["canonical_dir"],
        locations=graph_context["locations"],
        location_reference=graph_context["location_reference"],
        distances=graph_context["distances"],
        distance_reference=graph_context["distance_reference"],
    )
    result = _verify_run_unchecked(directory)
    if object_dict(result["descriptor"])["graph_id"] != graph.graph_id:
        raise DataError("STGNN run source graph differs from current authorized graph")
    return result


def _verify_run_unchecked(directory: str | Path, *, staged: bool = False) -> dict[str, object]:
    root = safe_absolute(directory)
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "run_id", "files"}
        or manifest["schema_version"] != "stgnn-1.0"
    ):
        raise DataError("Invalid graph model manifest")
    files = object_dict(manifest["files"])
    tree = discover(root)
    if set(tree.files) != {*files, "manifest.json"} or tree.directories != ("graph",):
        raise DataError("Graph model inventory mismatch")
    for name, digest in files.items():
        validate_relative_path(name)
        if stream_file(root / name)[1] != digest:
            raise DataError("Graph model checksum mismatch")
    run = read_json(root / "run.json")
    run_id = identity(run)
    if manifest["run_id"] != run_id or (not staged and root.name != run_id):
        raise DataError("Graph model identity mismatch")
    content = {k: v for k, v in files.items() if k not in {"run.json", "runtime.json"}}
    if run.get("content_sha256") != content:
        raise DataError("Graph model content identity mismatch")
    from urbangraph.graphs.artifacts import _load_graph_unchecked

    graph = _load_graph_unchecked(root / "graph", staged=True)
    if (
        run.get("graph_id") != graph.graph_id
        or run.get("sensor_axis_sha256") != graph.index.fingerprint
    ):
        raise DataError("Run graph reference mismatch")
    models, scalers, selections = (
        read_json(root / name) for name in ("models.json", "scalers.json", "selection.json")
    )
    if not models or set(models) != set(scalers) or set(models) != set(selections):
        raise DataError("Graph model folds/scalers/selections mismatch")
    required = {
        "run.json",
        "runtime.json",
        "models.json",
        "scalers.json",
        "selection.json",
        "metrics.json",
        "graph/manifest.json",
        "graph/graph.json",
        "graph/adjacency.npz",
    }
    for fold, value in models.items():
        obj = object_dict(value)
        model = restore(root, obj, graph)
        required.add(str(obj["file"]))
        scaler = FittedStandardizer.from_dict(scalers[fold])
        if scaler.sensor_ids != graph.index.sensor_ids:
            raise DataError("Graph scaler sensor mismatch")
        with torch.no_grad():
            actual = model(probe(model)).numpy()
        expected = np.asarray(obj.get("reload_prediction"), dtype=np.float32)
        if (
            expected.shape != actual.shape
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
        ):
            raise DataError("Graph model reload probe mismatch")
    if set(files) != required:
        raise DataError("Unexpected graph model payload")
    return verification_status(object_dict(run)) | {
        "run_id": run_id,
        "descriptor": run,
        "folds": len(models),
    }


def publish_run(
    models: dict[str, GraphForecaster],
    scalers: dict[str, object],
    graph: GraphArtifact,
    payloads: dict[str, dict[str, object]],
    descriptor: dict[str, object],
    output_root: Path,
    *,
    raw_snapshot_dir: Path,
    canonical_dir: Path,
    locations: Path,
    location_reference: str,
    distances: Path,
    distance_reference: str,
) -> Path:
    from urbangraph.data.rights import require_verified_snapshot_authorization

    descriptor = classify_for_publication(descriptor)
    for purpose, operation in (
        ("model_training", "model_train"),
        ("scientific_evaluation", "model_evaluate"),
        ("model_training", "model_publish"),
    ):
        require_verified_snapshot_authorization(
            raw_snapshot_dir, purpose, output_root, operation=operation
        )
    current = load_graph(
        graph.directory,
        raw_snapshot_dir=raw_snapshot_dir,
        canonical_dir=canonical_dir,
        locations=locations,
        location_reference=location_reference,
        distances=distances,
        distance_reference=distance_reference,
    )
    if current.graph_id != graph.graph_id:
        raise DataError("Graph changed before model publication")
    root = safe_absolute(output_root)
    if set(payloads) != {"selection", "metrics", "runtime"} or set(models) != set(scalers):
        raise DataError("Invalid graph research payload")
    for parent in (root, *root.parents):
        if (parent / "manifest.json").exists():
            raise DataError("Cannot nest graph research artifacts")
    root.mkdir(parents=True, exist_ok=True)
    with _writer_lock(root):
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            current = load_graph(
                graph.directory,
                raw_snapshot_dir=raw_snapshot_dir,
                canonical_dir=canonical_dir,
                locations=locations,
                location_reference=location_reference,
                distances=distances,
                distance_reference=distance_reference,
            )
            if current.graph_id != graph.graph_id:
                raise DataError("Graph changed during model publication")
            for name in ("manifest.json", "graph.json", "adjacency.npz"):
                copy_file(graph.directory / name, stage / "graph" / name)
            metadata: dict[str, object] = {}
            for i, (fold, fitted) in enumerate(sorted(models.items())):
                model = copy.deepcopy(fitted).cpu().eval()
                if model.graph_id != graph.graph_id or model.sensor_ids != graph.index.sensor_ids:
                    raise DataError("Publication graph/model mismatch")
                filename = f"model-{i}.pt"
                with (stage / filename).open("xb") as handle:
                    torch.save(model.state_dict(), handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                with torch.no_grad():
                    prediction = model(probe(model)).numpy().tolist()
                metadata[fold] = {
                    "file": filename,
                    "config": model.config.to_dict(),
                    "graph_id": graph.graph_id,
                    "sensor_ids": list(model.sensor_ids),
                    "history": model.history,
                    "horizon": model.horizon,
                    "features": model.features,
                    "parameter_count": model.parameter_count,
                    "receptive_field": model.receptive_field,
                    "dilations": list(model.dilations),
                    "reload_prediction": prediction,
                }
            write_json(stage / "models.json", metadata)
            write_json(stage / "scalers.json", scalers)
            for name, payload in payloads.items():
                write_json(stage / f"{name}.json", payload)
            files = {name: stream_file(stage / name)[1] for name in discover(stage).files}
            run = descriptor | {
                "content_sha256": {k: v for k, v in files.items() if k != "runtime.json"}
            }
            run_id = identity(run)
            write_json(stage / "run.json", run)
            files["run.json"] = stream_file(stage / "run.json")[1]
            write_json(
                stage / "manifest.json",
                {"schema_version": "stgnn-1.0", "run_id": run_id, "files": files},
            )
            _verify_run_unchecked(stage, staged=True)
            destination = root / run_id
            for purpose, operation in (
                ("model_training", "model_train"),
                ("scientific_evaluation", "model_evaluate"),
                ("model_training", "model_publish"),
            ):
                require_verified_snapshot_authorization(
                    raw_snapshot_dir, purpose, output_root, operation=operation
                )
            current = load_graph(
                graph.directory,
                raw_snapshot_dir=raw_snapshot_dir,
                canonical_dir=canonical_dir,
                locations=locations,
                location_reference=location_reference,
                distances=distances,
                distance_reference=distance_reference,
            )
            if current.graph_id != graph.graph_id:
                raise DataError("Graph changed before final model publication")
            if os.path.lexists(destination):
                _verify_run_unchecked(destination)
            else:
                _publish(stage, destination)
            return destination
        finally:
            _cleanup_stage(stage, root)
