"""Append-only checked experiment bundles using the existing filesystem primitives."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from urbangraph.baselines.models import FittedState
from urbangraph.data import DataError
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.data.saved_trust import classify_for_publication, verification_status
from urbangraph.data.saved_use import authorized_saved_lineage, authorized_saved_publication
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity


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
        raise DataError("Saved baseline lineage differs from current authorized source")
    return _verify_run_unchecked(directory)


def _verify_run_unchecked(directory: str | Path, *, staged: bool = False) -> dict[str, object]:
    root = safe_absolute(directory)
    manifest = read_json(root / "manifest.json")
    if (
        set(manifest) != {"schema_version", "run_id", "files"}
        or manifest["schema_version"] != "1.0"
    ):
        raise DataError("Invalid experiment manifest")
    files = object_dict(manifest["files"])
    tree = discover(root)
    if tree.directories or set(tree.files) != {*files, "manifest.json"}:
        raise DataError("Experiment inventory mismatch")
    for name, digest in files.items():
        if "/" in name or "\\" in name or not name.endswith(".json"):
            raise DataError("Unsafe experiment file name")
        if stream_file(root / name)[1] != digest:
            raise DataError("Experiment checksum mismatch")
    required = {
        "run.json",
        "model-config.json",
        "fitted-state.json",
        "metrics.json",
        "availability.json",
        "lineage.json",
        "selection.json",
        "runtime.json",
    }
    if set(files) != required:
        raise DataError("Experiment required files missing or unexpected")
    descriptor = read_json(root / "run.json")
    run_id = identity(descriptor)
    if manifest["run_id"] != run_id or (not staged and root.name != run_id):
        raise DataError("Experiment identity mismatch")
    content = object_dict(descriptor.get("content_sha256"))
    if content != {
        name: digest for name, digest in files.items() if name not in {"run.json", "runtime.json"}
    }:
        raise DataError("Experiment content identity mismatch")
    for value in read_json(root / "fitted-state.json").values():
        fold = object_dict(value)
        if fold.get("state") is not None:
            state = FittedState.from_dict(fold["state"])
            if state.state_id != fold.get("state_id"):
                raise DataError("Fitted state identity mismatch")
    return verification_status(object_dict(descriptor)) | {
        "run_id": run_id,
        "descriptor": descriptor,
    }


def publish_run(
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
            raise DataError("Cannot place experiment inside immutable artifact")
    root.mkdir(parents=True, exist_ok=True)
    with _writer_lock(root):
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            for name, value in payloads.items():
                if name not in {
                    "model-config",
                    "fitted-state",
                    "metrics",
                    "availability",
                    "lineage",
                    "selection",
                    "runtime",
                }:
                    raise DataError("Unknown experiment payload")
                write_json(stage / f"{name}.json", value)
            files = {p.name: stream_file(p)[1] for p in sorted(stage.iterdir())}
            run = descriptor | {
                "content_sha256": {n: h for n, h in files.items() if n != "runtime.json"}
            }
            run_id = identity(run)
            write_json(stage / "run.json", run)
            files["run.json"] = stream_file(stage / "run.json")[1]
            write_json(
                stage / "manifest.json", {"schema_version": "1.0", "run_id": run_id, "files": files}
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
