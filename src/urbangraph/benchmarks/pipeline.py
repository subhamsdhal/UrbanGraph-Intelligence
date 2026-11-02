"""Compose immutable raw/canonical/evaluation APIs with explicit evidence."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from urbangraph.benchmarks.sources import Acquisition, source_spec
from urbangraph.data import (
    DataError,
    DatasetSnapshot,
    DatasetSource,
    RawFileRecord,
    content_identity,
    verify_snapshot,
)
from urbangraph.data.filesystem import copy_file, safe_absolute, stream_file
from urbangraph.data.ingestion import _ingest_verified_source
from urbangraph.data.rights import require_raw_authorization
from urbangraph.datasets.contracts import DatasetConfig
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.datasets.materialization import materialize_snapshot, verify_materialization
from urbangraph.datasets.registry import get_dataset_spec
from urbangraph.datasets.serialization import read_json
from urbangraph.evaluation.artifacts import publish_plan
from urbangraph.evaluation.contracts import (
    BENCHMARK_PROTOCOLS,
    EvaluationProtocol,
    PartitionBoundary,
    PlanConfig,
)
from urbangraph.evaluation.dataset import open_dataset


def import_local(
    file: str | Path,
    dataset_id: str,
    acquisition: Acquisition,
    *,
    raw_root: str | Path = "data/raw",
) -> DatasetSnapshot:
    """Explicit opt-in byte import. Parsing remains a separate trusted-file operation."""
    expected_digest = require_raw_authorization(
        dataset_id, file, "local_research", raw_root, operation="benchmark_import"
    )
    spec = get_dataset_spec(dataset_id)
    origin, root = safe_absolute(file), safe_absolute(raw_root)
    if origin.name != spec.filename or not origin.is_file():
        raise DataError("A regular file with the registered benchmark filename is required")
    if origin.is_relative_to(root) or root.is_relative_to(origin.parent):
        raise DataError("Input must be outside and non-nested with the managed raw root")
    before = stream_file(origin)
    if before[1] != expected_digest:
        raise DataError("Benchmark file does not match rights grant")
    # Existing ingestor accepts a directory. Private staging selects exactly one file.
    with tempfile.TemporaryDirectory(prefix="urbangraph-import-") as temporary:
        selected = Path(temporary)
        copy_file(origin, selected / origin.name)
        result = _ingest_verified_source(
            selected,
            dataset_id,
            raw_root=root,
            source=DatasetSource(
                reference=acquisition.reference,
                description="Local import; evidence kind: " + acquisition.evidence_kind,
                licensing_notes=acquisition.usage_notes,
            ),
            expected_digest=content_identity((RawFileRecord(origin.name, *before),)),
        )
    record = result.manifest.files[0]
    if stream_file(origin) != before or (record.size_bytes, record.sha256) != before:
        raise DataError("Input changed during import; do not use this run as acquisition evidence")
    if not verify_snapshot(result.directory).valid:
        raise DataError("Imported raw snapshot failed verification")
    return result


def audit_raw(
    snapshot_dir: str | Path,
    dataset_id: str,
    *,
    output_root: str | Path = "data/interim",
    config: DatasetConfig | None = None,
) -> dict[str, object]:
    """Parse trusted HDF, materialize and verify exact raw lineage; never repair."""
    parsed = parse_snapshot(snapshot_dir, dataset_id=dataset_id, config=config)
    result = materialize_snapshot(
        snapshot_dir, output_root=output_root, dataset_id=dataset_id, config=config
    )
    checked = verify_materialization(result.directory, raw_snapshot_dir=snapshot_dir)
    if not checked.valid:
        raise DataError("Canonical artifact failed verification")
    with pd.HDFStore(
        str(parsed.lineage.snapshot_dir / "payload" / parsed.lineage.input_file.relative_path),
        mode="r",
    ) as store:
        keys = store.keys()
    return {
        "dataset_id": dataset_id,
        "raw_snapshot_id": parsed.raw_snapshot_id,
        "canonical_id": result.transformation_id,
        "canonical_dir": str(result.directory),
        "input": asdict(parsed.lineage.input_file),
        "hdf_keys": keys,
        "selected_key": parsed.config.hdf_key,
        "schema": parsed.schema_dict(),
        "quality": read_json(result.directory / "quality.json"),
        "verified": True,
        "reused": result.reused,
    }


def evaluation_configs(
    rows: int, *, benchmark_protocol: str = "dcrnn-sample-first-v1"
) -> dict[str, PlanConfig]:
    """Version 1: size-only policy, never tuned against observations or outcomes.

    Chronological 60/15/10/remainder percent; rolling starts at 50 percent,
    advances by 10 percent for two folds, with three 10 percent holdouts.
    Resolved integers are embedded in every plan. Too-small datasets fail.
    """
    if type(rows) is not int or rows < 240:
        raise DataError("Default research plans require at least 240 rows")
    if benchmark_protocol not in BENCHMARK_PROTOCOLS:
        raise DataError("Unknown benchmark protocol")
    a, b, c = rows * 60 // 100, rows * 75 // 100, rows * 85 // 100
    return {
        "benchmark": PlanConfig(protocol=EvaluationProtocol(benchmark_protocol, "benchmark")),
        "chronological": PlanConfig(
            protocol=EvaluationProtocol("strict-chronological-v1", "isolated"),
            scaling="per_sensor",
            boundaries=tuple(
                PartitionBoundary(name, start, stop)
                for name, start, stop in (
                    ("train", 0, a),
                    ("validation", a, b),
                    ("calibration", b, c),
                    ("test", c, rows),
                )
            ),
        ),
        "rolling": PlanConfig(
            minimum_train=rows // 2,
            validation_length=rows // 10,
            calibration_length=rows // 10,
            test_length=rows // 10,
            stride=rows // 10,
            folds=2,
            scaling="per_sensor",
        ),
    }


def prepare_evaluation(
    canonical_dir: str | Path,
    dataset_id: str,
    *,
    raw_snapshot_dir: str | Path,
    output_root: str | Path = "data/evaluation",
    benchmark_protocol: str = "dcrnn-sample-first-v1",
) -> dict[str, Path]:
    source_spec(dataset_id)
    dataset = open_dataset(canonical_dir, raw_snapshot_dir=raw_snapshot_dir)
    if dataset.lineage["dataset_id"] != dataset_id:
        raise DataError("Requested dataset differs from canonical identity")
    configs = evaluation_configs(len(dataset.frame), benchmark_protocol=benchmark_protocol)
    return {
        name: publish_plan(
            canonical_dir, config, output_root=output_root, raw_snapshot_dir=raw_snapshot_dir
        ).directory
        for name, config in configs.items()
    }
