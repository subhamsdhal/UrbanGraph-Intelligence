"""Governed local canonical lifecycle over the existing HDF-to-Parquet pipeline.

The caller supplies paths, but never a permission decision. Every call reloads the
deployment policy. This local application boundary still requires an OS-protected
policy file and trusted, exclusively controlled source/output directories.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urbangraph.data import DataError, verify_snapshot
from urbangraph.data.rights import require_snapshot_authorization
from urbangraph.datasets.contracts import DatasetConfig
from urbangraph.datasets.materialization import (
    MaterializationResult,
    _load_verified,
    _verify_materialization_after_authorization,
    materialize_snapshot,
    verify_materialization,
)


def _authorized_snapshot(snapshot_dir: Path, *, operation: str, destination: Path) -> str:
    expected = require_snapshot_authorization(
        snapshot_dir, "local_research", destination, operation=operation
    )
    verified = verify_snapshot(snapshot_dir)
    if not verified.valid or verified.manifest is None:
        raise DataError("Upstream raw snapshot failed integrity verification")
    if verified.manifest.snapshot_id != expected:
        raise DataError("Upstream raw identity differs from current rights grant")
    return expected


def create_canonical(
    snapshot_dir: str | Path,
    *,
    output_root: str | Path,
    dataset_id: str | None = None,
    config: DatasetConfig | None = None,
) -> MaterializationResult:
    """Authorize this transformation before reading a protected HDF source."""
    snapshot, root = Path(snapshot_dir), Path(output_root)
    expected_snapshot = _authorized_snapshot(
        snapshot, operation="canonical_create", destination=root
    )
    # materialize_snapshot independently enforces snapshot_parse and verifies the
    # source before and after the real HDF-to-Parquet transformation.
    result = materialize_snapshot(snapshot, output_root=root, dataset_id=dataset_id, config=config)
    if not _verify_materialization_after_authorization(
        result.directory,
        raw_snapshot_dir=snapshot,
        expected_snapshot=expected_snapshot,
    ).valid:
        raise DataError("Created canonical artifact failed upstream lineage verification")
    return result


def read_canonical(
    artifact_dir: str | Path,
    *,
    raw_snapshot_dir: str | Path,
) -> pd.DataFrame:
    """Reauthorize the upstream source on every later use, then verify/load it."""
    artifact, snapshot = Path(artifact_dir), Path(raw_snapshot_dir)
    _authorized_snapshot(snapshot, operation="canonical_read", destination=artifact)
    checked = verify_materialization(artifact, raw_snapshot_dir=snapshot)
    if not checked.valid:
        raise DataError("Canonical artifact or upstream lineage failed integrity verification")
    return _load_verified(artifact, enforce_location=True)[0]
