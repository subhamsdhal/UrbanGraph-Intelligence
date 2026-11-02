"""Append-only neural bundles; native state dictionaries and verified CPU reload probes."""

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
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.data.saved_trust import classify_for_publication, verification_status
from urbangraph.data.saved_use import authorized_saved_lineage, authorized_saved_publication
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import integer
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.neural.contracts import config_from_dict
from urbangraph.neural.models import TemporalForecaster


def probe(model: TemporalForecaster) -> Tensor:
    generator = torch.Generator().manual_seed(2026)
    x = torch.randn(2, model.history, len(model.sensor_ids), model.features, generator=generator)
    x[..., 1] = 1
    x[0, :, 0, 1] = 0
    return x


def restore(root: Path, metadata: object) -> TemporalForecaster:
    obj = object_dict(metadata)
    filename = obj.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
        raise DataError("Unsafe neural weight filename")
    sensors = obj.get("sensor_ids")
    if not isinstance(sensors, list) or not sensors or any(not isinstance(s, str) for s in sensors):
        raise DataError("Invalid neural sensor schema")
    config = config_from_dict(obj.get("config"))
    model = TemporalForecaster(
        config,
        integer(obj.get("history"), "history", 1),
        integer(obj.get("horizon"), "horizon", 1),
        tuple(str(s) for s in sensors),
        4 if config.calendar else 2,
    )
    if stream_file(root / filename)[0] > 256 * 1024**2:
        raise DataError("Oversized neural state")
    try:
        loaded = torch.load(root / filename, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or any(not isinstance(v, Tensor) for v in loaded.values()):
            raise DataError("Neural artifact must contain only a state dictionary")
        model.load_state_dict(cast(dict[str, Tensor], loaded), strict=True)
    except (RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
        raise DataError("Invalid neural weights") from exc
    model.eval()
    return model


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
        raise DataError("Saved neural lineage differs from current authorized source")
    return _verify_run_unchecked(directory)


def _verify_run_unchecked(directory: str | Path, *, staged: bool = False) -> dict[str, object]:
    root = safe_absolute(directory)
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "run_id", "files"}
        or manifest["schema_version"] != "neural-1.0"
    ):
        raise DataError("Invalid neural manifest")
    files = object_dict(manifest["files"])
    tree = discover(root)
    if tree.directories or set(tree.files) != {*files, "manifest.json"}:
        raise DataError("Neural artifact inventory mismatch")
    for name, digest in files.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise DataError("Unsafe artifact filename")
        if stream_file(root / name)[1] != digest:
            raise DataError("Neural artifact checksum mismatch")
    run = read_json(root / "run.json")
    run_id = identity(run)
    if manifest["run_id"] != run_id or (not staged and root.name != run_id):
        raise DataError("Neural content identity mismatch")
    content = {k: v for k, v in files.items() if k not in {"run.json", "runtime.json"}}
    if run.get("content_sha256") != content:
        raise DataError("Neural content inventory mismatch")
    models = read_json(root / "models.json")
    scalers = read_json(root / "scalers.json")
    if not models or set(models) != set(scalers):
        raise DataError("Neural fold/scaler mismatch")
    required = {
        "run.json",
        "models.json",
        "scalers.json",
        "selection.json",
        "metrics.json",
        "runtime.json",
    }
    for fold, metadata in models.items():
        obj = object_dict(metadata)
        model = restore(root, obj)
        required.add(str(obj["file"]))
        scaler = FittedStandardizer.from_dict(scalers[fold])
        if scaler.sensor_ids != model.sensor_ids:
            raise DataError("Neural scaler/sensor mismatch")
        with torch.no_grad():
            actual = model(probe(model)).numpy()
        expected = np.asarray(obj.get("reload_prediction"), dtype=np.float32)
        if (
            expected.shape != actual.shape
            or not np.isfinite(expected).all()
            or not np.allclose(actual, expected, rtol=1e-6, atol=1e-6)
        ):
            raise DataError("Neural reload prediction mismatch")
    if set(files) != required:
        raise DataError("Unexpected neural payload")
    return verification_status(object_dict(run)) | {"run_id": run_id, "descriptor": run}


def publish_run(
    models: dict[str, TemporalForecaster],
    scalers: dict[str, object],
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
    if set(payloads) != {"selection", "metrics", "runtime"} or set(models) != set(scalers):
        raise DataError("Invalid neural publication payload")
    for parent in (root, *root.parents):
        if (parent / "manifest.json").exists():
            raise DataError("Cannot nest immutable artifacts")
    root.mkdir(parents=True, exist_ok=True)
    with _writer_lock(root):
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            metadata: dict[str, object] = {}
            for i, (fold, fitted) in enumerate(sorted(models.items())):
                model = copy.deepcopy(fitted).cpu().eval()
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
                    "sensor_ids": list(model.sensor_ids),
                    "history": model.history,
                    "horizon": model.horizon,
                    "parameter_count": model.parameter_count,
                    "reload_prediction": prediction,
                }
            write_json(stage / "models.json", metadata)
            write_json(stage / "scalers.json", scalers)
            for name, payload in payloads.items():
                write_json(stage / f"{name}.json", payload)
            files = {p.name: stream_file(p)[1] for p in sorted(stage.iterdir())}
            run = descriptor | {
                "content_sha256": {k: v for k, v in files.items() if k != "runtime.json"}
            }
            run_id = identity(run)
            write_json(stage / "run.json", run)
            files["run.json"] = stream_file(stage / "run.json")[1]
            write_json(
                stage / "manifest.json",
                {"schema_version": "neural-1.0", "run_id": run_id, "files": files},
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
