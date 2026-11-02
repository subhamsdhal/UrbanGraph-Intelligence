"""Append-only trusted training checkpoints with weights-only deserialization."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from pickle import UnpicklingError

import torch

from urbangraph.data import DataError
from urbangraph.data.filesystem import discover, safe_absolute, stream_file
from urbangraph.data.ingestion import _cleanup_stage, _publish, _writer_lock
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity


def save_checkpoint(
    root: Path, epoch: int, state: dict[str, object], metadata: dict[str, object]
) -> Path:
    root = safe_absolute(root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"epoch-{epoch:04d}"
    with _writer_lock(root):
        if destination.exists():
            raise DataError("Checkpoint epoch exists; refusing overwrite")
        stage = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
        try:
            with (stage / "state.pt").open("xb") as handle:
                torch.save(state, handle)
                handle.flush()
                os.fsync(handle.fileno())
            descriptor = metadata | {
                "schema_version": "neural-checkpoint-1.0",
                "epoch": epoch,
                "state_sha256": stream_file(stage / "state.pt")[1],
            }
            write_json(
                stage / "checkpoint.json", descriptor | {"checkpoint_id": identity(descriptor)}
            )
            load_checkpoint(stage)
            _publish(stage, destination)
            return destination
        finally:
            _cleanup_stage(stage, root)


def load_checkpoint(directory: Path) -> tuple[dict[str, object], dict[str, object]]:
    root = safe_absolute(directory)
    tree = discover(root)
    if tree.directories or set(tree.files) != {"state.pt", "checkpoint.json"}:
        raise DataError("Checkpoint inventory mismatch")
    metadata = read_json(root / "checkpoint.json")
    descriptor = {k: v for k, v in metadata.items() if k != "checkpoint_id"}
    if metadata.get("schema_version") != "neural-checkpoint-1.0" or identity(
        descriptor
    ) != metadata.get("checkpoint_id"):
        raise DataError("Checkpoint identity mismatch")
    size, digest = stream_file(root / "state.pt")
    if size > 256 * 1024**2 or digest != metadata.get("state_sha256"):
        raise DataError("Checkpoint checksum/size mismatch")
    try:
        state = object_dict(torch.load(root / "state.pt", map_location="cpu", weights_only=True))
    except (RuntimeError, ValueError, EOFError, UnpicklingError) as exc:
        raise DataError("Invalid trusted checkpoint state") from exc
    required = {
        "model",
        "best_model",
        "optimizer",
        "scheduler",
        "amp",
        "torch_rng",
        "cuda_rng",
        "best_mae",
        "best_epoch",
        "bad_epochs",
        "history",
    }
    if set(state) != required:
        raise DataError("Checkpoint state fields mismatch")
    return state, metadata
