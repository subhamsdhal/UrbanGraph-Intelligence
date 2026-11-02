"""Immutable local checkpoints for approved synthetic replay sequences.

The checkpoint is a recoverable event sequence, not a source of trusted data or
an authorization token. It contains only approved synthetic origin identifiers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from urbangraph.data.filesystem import reject_link, safe_absolute
from urbangraph.serving.local_synthetic import LocalSyntheticEngine
from urbangraph.serving.synthetic_replay import MAX_EVENTS, ReplayOutcome, replay_approved_origins


class CheckpointError(ValueError):
    """A checkpoint cannot be safely published or recovered."""


_LOGGER = logging.getLogger(__name__)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError("Duplicate checkpoint field")
        result[key] = value
    return result


def _document(engine: LocalSyntheticEngine, arrivals: tuple[str, ...]) -> dict[str, object]:
    return {
        "version": 1,
        "bundle_sha256": engine.trusted.bundle_sha256,
        "source_manifest_sha256": engine.data.manifest_sha256,
        "arrivals": list(arrivals),
    }


def _safe_destination(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise CheckpointError("Checkpoint path must be absolute without traversal")
    try:
        destination = safe_absolute(path)
        parent = destination.parent
        reject_link(parent)
        if not parent.is_dir():
            raise CheckpointError("Trusted checkpoint parent must exist")
        checkout = Path(__file__).resolve().parents[3]
        if destination.is_relative_to(checkout):
            raise CheckpointError("Checkpoint must be outside source checkout")
    except OSError as exc:
        raise CheckpointError("Checkpoint destination unavailable") from exc
    return destination


def _publish_no_overwrite(stage: Path, destination: Path) -> None:
    """Publish complete staged bytes without replacing an existing checkpoint."""
    if os.name == "nt":
        # A same-directory Windows rename is atomic and refuses an existing target.
        os.rename(stage, destination)
    else:
        # POSIX rename replaces an existing target; a hard link does not.
        os.link(stage, destination)


def save_checkpoint(
    engine: LocalSyntheticEngine,
    arrivals: tuple[str, ...],
    path: Path,
) -> ReplayOutcome:
    """Run actual forecasts, then publish one complete no-overwrite checkpoint."""
    if len(arrivals) > MAX_EVENTS or any(origin not in engine.data.origins for origin in arrivals):
        raise CheckpointError("Only bounded approved synthetic origins may be checkpointed")
    outcome = replay_approved_origins(engine, arrivals)
    destination = _safe_destination(path)
    document = _document(engine, arrivals)
    payload = _canonical(document)
    wrapper = {"document": document, "sha256": hashlib.sha256(payload).hexdigest()}
    encoded = _canonical(wrapper)
    if len(encoded) > 32768:
        raise CheckpointError("Checkpoint exceeds size limit")
    stage: Path | None = None
    stage_identity: tuple[int, int] | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".replay-stage-", suffix=".json", dir=destination.parent, delete=False
        ) as handle:
            stage = Path(handle.name)
            owned = os.fstat(handle.fileno())
            stage_identity = (owned.st_dev, owned.st_ino)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_overwrite(stage, destination)
    except OSError as exc:
        _LOGGER.warning(
            "Checkpoint publication failed: %s errno=%s winerror=%s",
            type(exc).__name__,
            exc.errno,
            getattr(exc, "winerror", None),
        )
        raise CheckpointError("Checkpoint publication failed without overwrite") from exc
    finally:
        if stage is not None and stage.exists():
            current = reject_link(stage)
            if stage_identity is None or (current.st_dev, current.st_ino) != stage_identity:
                raise CheckpointError("Checkpoint staging path changed; refusing cleanup")
            stage.unlink()
    return outcome


def load_checkpoint(
    engine: LocalSyntheticEngine, path: Path
) -> tuple[tuple[str, ...], ReplayOutcome]:
    """Verify the checkpoint bytes and recompute results from trusted fixtures."""
    source = _safe_destination(path)
    try:
        info = reject_link(source)
        if info.st_size > 32768:
            raise CheckpointError("Checkpoint exceeds size limit")
        raw = source.read_bytes()
        wrapper = json.loads(raw, object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("Checkpoint cannot be read") from exc
    if not isinstance(wrapper, dict) or set(wrapper) != {"document", "sha256"}:
        raise CheckpointError("Checkpoint wrapper invalid")
    document = wrapper["document"]
    if not isinstance(document, dict) or set(document) != {
        "version",
        "bundle_sha256",
        "source_manifest_sha256",
        "arrivals",
    }:
        raise CheckpointError("Checkpoint schema invalid")
    if wrapper["sha256"] != hashlib.sha256(_canonical(document)).hexdigest():
        raise CheckpointError("Checkpoint content changed")
    if (
        document["version"] != 1
        or document["bundle_sha256"] != engine.trusted.bundle_sha256
        or document["source_manifest_sha256"] != engine.data.manifest_sha256
    ):
        raise CheckpointError("Checkpoint provenance differs from trusted fixtures")
    arrivals = document["arrivals"]
    if (
        not isinstance(arrivals, list)
        or len(arrivals) > MAX_EVENTS
        or any(
            not isinstance(origin, str) or origin not in engine.data.origins for origin in arrivals
        )
    ):
        raise CheckpointError("Checkpoint event identity invalid")
    sequence = tuple(arrivals)
    return sequence, replay_approved_origins(engine, sequence)


def resume_checkpoint(
    engine: LocalSyntheticEngine,
    previous: Path,
    additional: tuple[str, ...],
    destination: Path,
) -> ReplayOutcome:
    """Verify an immutable prior checkpoint and publish a new version."""
    old, _ = load_checkpoint(engine, previous)
    return save_checkpoint(engine, old + additional, destination)
