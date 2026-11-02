"""Checksum-verified native model bundles with append-only publication."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from urbangraph.boosting.contracts import FeatureSpec, config_from_dict
from urbangraph.boosting.models import BoostModel, load_model
from urbangraph.data import DataError
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.data.saved_trust import classify_for_publication, verification_status
from urbangraph.data.saved_use import authorized_saved_lineage, authorized_saved_publication
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import integer
from urbangraph.evaluation.dataset import identity


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(v, str) for v in value):
        raise DataError("Nonempty string schema required")
    return tuple(str(v) for v in value)


def probe(model: BoostModel) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]]:
    values = (
        np.random.default_rng(2026).uniform(0, 80, size=(32, len(model.names))).astype(np.float32)
    )
    for i, name in enumerate(model.names):
        if name == "sensor_id":
            values[:, i] = np.arange(32) % len(model.sensor_ids)
        elif name == "horizon":
            values[:, i] = 1 + np.arange(32) % model.horizon
        elif name.endswith("_missing"):
            values[:, i] = 0
        elif name.startswith(("lag_", "seasonal_")):
            values[0, i] = np.nan
    return values


def model_metadata(model: BoostModel, filename: str) -> dict[str, object]:
    return {
        "file": filename,
        "reload_prediction": model.predict(probe(model), model.names, model.sensor_ids).tolist(),
        "config": model.config.to_dict(),
        "features": model.features.to_dict(),
        "names": list(model.names),
        "groups": list(model.groups),
        "sensor_ids": list(model.sensor_ids),
        "horizon": model.horizon,
        "best_iteration": model.best_iteration,
    }


def restore(root: Path, metadata: object) -> BoostModel:
    obj = object_dict(metadata)
    filename = obj.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or "/" in filename:
        raise DataError("Unsafe native model filename")
    config = config_from_dict(obj.get("config"))
    spec = object_dict(obj.get("features"))
    if set(spec) != {
        "version",
        "variant",
        "lags",
        "rolling",
        "seasonal_policy",
        "missing_indicators",
    }:
        raise DataError("Invalid stored feature schema")
    lags, rolling = spec["lags"], spec["rolling"]
    if not isinstance(lags, list) or not isinstance(rolling, list):
        raise DataError("Feature windows must be lists")
    missing = spec["missing_indicators"]
    if type(missing) is not bool:
        raise DataError("Invalid missing indicator policy")
    features = FeatureSpec(
        version=str(spec["version"]),
        variant=str(spec["variant"]),
        lags=tuple(integer(v, "lag", 1) for v in lags),
        rolling=tuple(integer(v, "window", 1) for v in rolling),
        seasonal_policy=str(spec["seasonal_policy"]),
        missing_indicators=missing,
    )
    return load_model(
        root / filename,
        config,
        features,
        strings(obj.get("names")),
        strings(obj.get("groups")),
        strings(obj.get("sensor_ids")),
        integer(obj.get("horizon"), "horizon", 1),
        integer(obj.get("best_iteration"), "iteration", 1),
    )


def verify_run(
    directory: str | Path,
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
) -> dict[str, object]:
    lineage = authorized_saved_lineage(
        directory,
        raw_snapshot_dir=raw_snapshot_dir,
        canonical_dir=canonical_dir,
        purpose="saved_run_verification",
        operation="model_verify",
    )
    run = object_dict(read_json(safe_absolute(directory) / "run.json"))
    if run.get("canonical") != lineage:
        raise DataError("Saved boosting lineage differs from current authorized source")
    return _verify_run_unchecked(directory)


def _verify_run_unchecked(directory: str | Path, *, staged: bool = False) -> dict[str, object]:
    root = safe_absolute(directory)
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "run_id", "files"}
        or manifest["schema_version"] != "boosting-1.0"
    ):
        raise DataError("Invalid boosting manifest")
    files = object_dict(manifest["files"])
    tree = discover(root)
    if tree.directories or set(tree.files) != {*files, "manifest.json"}:
        raise DataError("Boosting artifact inventory mismatch")
    for name, digest in files.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise DataError("Unsafe artifact filename")
        if stream_file(root / name)[1] != digest:
            raise DataError("Boosting artifact checksum mismatch")
    required = {
        "run.json",
        "models.json",
        "metrics.json",
        "selection.json",
        "runtime.json",
        "importance.json",
    }
    if not required.issubset(files):
        raise DataError("Required boosting payload missing")
    run = read_json(root / "run.json")
    run_id = identity(run)
    if manifest["run_id"] != run_id or (not staged and root.name != run_id):
        raise DataError("Boosting run identity mismatch")
    content = {
        name: digest for name, digest in files.items() if name not in {"run.json", "runtime.json"}
    }
    if run.get("content_sha256") != content:
        raise DataError("Boosting content identity mismatch")
    models = read_json(root / "models.json")
    native_files: set[str] = set()
    for metadata in models.values():
        obj = object_dict(metadata)
        model = restore(root, obj)
        native_files.add(str(obj["file"]))
        expected = np.asarray(obj.get("reload_prediction"), dtype=np.float64)
        actual = model.predict(probe(model), model.names, model.sensor_ids)
        if (
            expected.shape != actual.shape
            or not np.isfinite(expected).all()
            or not np.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        ):
            raise DataError("Native reload prediction mismatch")
        if len(model.names) != len(model.groups) or len(set(model.names)) != len(model.names):
            raise DataError("Invalid native feature schema")
    if not models or set(files) != required | native_files:
        raise DataError("Unexpected boosting payload")
    return verification_status(object_dict(run)) | {"run_id": run_id, "descriptor": run}


def publish_run(
    models: dict[str, BoostModel],
    payloads: dict[str, dict[str, object]],
    descriptor: dict[str, object],
    output_root: str | Path,
    *,
    raw_snapshot_dir: str | Path | None = None,
    canonical_dir: str | Path | None = None,
    plan_dir: str | Path | None = None,
) -> Path:
    descriptor = classify_for_publication(descriptor)
    authorized_saved_publication(
        output_root,
        descriptor,
        raw_snapshot_dir=raw_snapshot_dir,
        canonical_dir=canonical_dir,
        plan_dir=plan_dir,
    )
    root = safe_absolute(output_root)
    for parent in (root, *root.parents):
        if (parent / "manifest.json").exists():
            raise DataError("Cannot nest immutable artifacts")
    if set(payloads) != {"metrics", "selection", "runtime", "importance"}:
        raise DataError("Invalid boosting payload set")
    root.mkdir(parents=True, exist_ok=True)
    with _writer_lock(root):
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            metadata: dict[str, object] = {}
            for i, (fold, model) in enumerate(sorted(models.items())):
                filename = f"model-{i}." + ("txt" if model.config.family == "lightgbm" else "json")
                model.save(stage / filename)
                metadata[fold] = model_metadata(model, filename)
            write_json(stage / "models.json", metadata)
            for name, value in payloads.items():
                write_json(stage / f"{name}.json", value)
            files = {p.name: stream_file(p)[1] for p in sorted(stage.iterdir())}
            run = descriptor | {
                "content_sha256": {n: h for n, h in files.items() if n != "runtime.json"}
            }
            run_id = identity(run)
            write_json(stage / "run.json", run)
            files["run.json"] = stream_file(stage / "run.json")[1]
            write_json(
                stage / "manifest.json",
                {"schema_version": "boosting-1.0", "run_id": run_id, "files": files},
            )
            _verify_run_unchecked(stage, staged=True)
            destination = root / run_id
            authorized_saved_publication(
                output_root,
                descriptor,
                raw_snapshot_dir=raw_snapshot_dir,
                canonical_dir=canonical_dir,
                plan_dir=plan_dir,
            )
            if os.path.lexists(destination):
                _verify_run_unchecked(destination)
            else:
                _publish(stage, destination)
            return destination
        finally:
            _cleanup_stage(stage, root)
