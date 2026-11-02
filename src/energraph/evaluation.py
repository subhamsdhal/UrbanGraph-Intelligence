"""Energy-specific selection and plans backed by the existing strict temporal engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    broker_parent_references,
    governed_payload,
    resolve_broker_reference,
)
from energraph.canonical import open_energy
from energraph.lineage import authorize_sources, resolve_parents
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.data.filesystem import safe_absolute
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import (
    EvaluationProtocol,
    FeatureAvailabilityPolicy,
    ForecastWindowSpecification,
    PartitionBoundary,
    PlanConfig,
    SplitManifest,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import plan_splits


def contiguous_range(dataset: EvaluationDataset) -> tuple[int, int]:
    """Longest finite hourly run, earliest tie; data-quality rule, never model-score selection."""
    valid = np.isfinite(dataset.frame.to_numpy(dtype=np.float64)).all(axis=1)
    hourly = np.r_[
        False,
        np.diff(np.asarray(dataset.timestamps.as_unit("ns").astype("int64"), dtype=np.int64))
        == 3600 * 10**9,
    ]
    best_start = best_stop = start = 0
    for i, ok in enumerate(valid):
        if not ok:
            start = i + 1
            continue
        if not hourly[i]:
            start = i
        if i + 1 - start > best_stop - best_start:
            best_start, best_stop = start, i + 1
    if best_stop - best_start < 1200:
        raise DataError("At least 1200 contiguous finite hourly observations required")
    return best_start, best_stop


def selected_dataset(dataset: EvaluationDataset, start: int, stop: int) -> EvaluationDataset:
    if not 0 <= start < stop <= len(dataset.frame):
        raise DataError("Invalid selected energy range")
    frame = dataset.frame.iloc[start:stop].copy()
    if not np.isfinite(frame.to_numpy()).all() or np.any(
        np.diff(np.asarray(frame.index.astype("int64"), dtype=np.int64)) != 3600 * 10**9
    ):
        raise DataError("Selected experiment range must be finite and hourly")
    return EvaluationDataset(
        dataset.directory, frame, dataset.lineage | {"source_rows": [start, stop]}
    )


def default_config(rows: int, protocol: str) -> PlanConfig:
    if protocol not in {"chronological", "rolling"}:
        raise DataError("Energy protocol must be chronological or rolling")
    train = rows * 6 // 10
    val = rows // 10
    calibration = rows // 10
    test = rows - train - val - calibration
    if protocol == "chronological":
        cuts = (0, train, train + val, train + val + calibration, rows)
        return PlanConfig(
            protocol=EvaluationProtocol("strict-chronological-v1", "operational"),
            boundaries=tuple(
                PartitionBoundary(n, a, b)
                for n, a, b in zip(
                    ("train", "validation", "calibration", "test"), cuts, cuts[1:], strict=False
                )
            ),
            window=ForecastWindowSpecification(168, 24, 3600),
            features=FeatureAvailabilityPolicy(False, False, 24),
        )
    length = test // 2
    return PlanConfig(
        protocol=EvaluationProtocol("strict-rolling-origin-v1", "operational"),
        minimum_train=train,
        validation_length=val,
        calibration_length=calibration,
        test_length=length,
        stride=length,
        folds=2,
        window=ForecastWindowSpecification(168, 24, 3600),
        features=FeatureAvailabilityPolicy(False, False, 24),
    )


def build_plan(
    canonical: Path | DatasetSnapshot,
    root: Path,
    protocol: str,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> DatasetSnapshot | PendingPublication:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for energy plan creation")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref: BrokerArtifactRef | None = None
    raw_ref: BrokerArtifactRef | None = None
    if isinstance(canonical, DatasetSnapshot) and canonical.broker_reference is not None:
        if broker_context is None:
            raise DataError("Broker execution context required for broker-backed canonical input")
        canonical_ref = resolve_broker_reference(
            broker_context.broker,
            canonical.broker_reference.artifact_id,
            purpose="scientific_evaluation",
            operation="energy_plan_create",
        )
        if canonical_ref != canonical.broker_reference:
            raise DataError("Canonical snapshot carries an unverified broker reference")
        read_parents = broker_parent_references(
            broker_context.broker,
            canonical_ref,
            purpose="local_research",
            operation="energy_canonical_read",
        )
        if len(read_parents) != 1 or read_parents[0].artifact_type != "raw":
            raise DataError("Canonical broker source closure is not a single raw source")
        raw_ref = read_parents[0]
        local_parents: tuple[Path, ...] = (safe_absolute(raw_snapshot_dir),)
    else:
        local_parents = (canonical_path,)
    graph = resolve_parents(local_parents)
    if graph.sources != resolve_parents((raw_snapshot_dir,)).sources:
        raise DataError("Energy plan raw source does not match canonical lineage")
    authorize_sources(
        graph.sources,
        purpose="scientific_evaluation",
        operation="energy_plan_create",
        destination=root,
    )
    original = open_energy(
        canonical_path,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context if canonical_ref is not None else None,
        broker_reference=canonical_ref,
        raw_broker_reference=raw_ref,
    )
    start, stop = contiguous_range(original)
    data = selected_dataset(original, start, stop)
    config = default_config(len(data.frame), protocol)
    split = plan_splits(data.timestamps, config)
    descriptor = {
        "schema_version": "energy-plan-1.0",
        "canonical": original.lineage,
        "source_rows": [start, stop],
        "selection_rule": "longest-finite-hourly-run-earliest-tie",
        "origin_convention": "UTC instant at origin row; targets begin next hour",
        "availability": "24h interval-start latency assumption; revisions not reconstructed",
        "split": split.to_dict(),
    }
    with tempfile.TemporaryDirectory(prefix="energraph-plan-") as tmp:
        write_json(Path(tmp) / "descriptor.json", descriptor)
        return _publish_governed(
            Path(tmp),
            root,
            "plan",
            parents=(canonical_path,) if canonical_ref is not None else local_parents,
            broker_context=broker_context if canonical_ref is not None else None,
            broker_parents=(canonical_ref,) if canonical_ref is not None else None,
        )


def load_plan(
    canonical: Path | DatasetSnapshot,
    directory: Path | DatasetSnapshot,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> tuple[EvaluationDataset, SplitManifest]:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for energy plan read")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref = canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else None
    plan_ref = directory.broker_reference if isinstance(directory, DatasetSnapshot) else None
    if canonical_ref is not None or plan_ref is not None:
        if broker_context is None or canonical_ref is None or plan_ref is None:
            raise DataError("Complete broker plan and canonical identities required")
        canonical_parents = broker_parent_references(
            broker_context.broker,
            resolve_broker_reference(
                broker_context.broker,
                canonical_ref.artifact_id,
                purpose="local_research",
                operation="energy_canonical_read",
            ),
            purpose="local_research",
            operation="energy_canonical_read",
        )
        if len(canonical_parents) != 1 or canonical_parents[0].artifact_type != "raw":
            raise DataError("Canonical broker source closure is invalid")
        data = open_energy(
            canonical_path,
            raw_snapshot_dir=raw_snapshot_dir,
            broker_context=broker_context,
            broker_reference=canonical_ref,
            raw_broker_reference=canonical_parents[0],
        )
        resolved_plan = resolve_broker_reference(
            broker_context.broker,
            plan_ref.artifact_id,
            purpose="model_training",
            operation="energy_plan_read",
        )
        if resolved_plan != plan_ref:
            raise DataError("Plan snapshot carries an unverified broker reference")
        plan_parents = broker_parent_references(
            broker_context.broker,
            resolved_plan,
            purpose="model_training",
            operation="energy_plan_read",
        )
        if plan_parents != (canonical_ref,):
            raise DataError("Plan broker parent differs from canonical identity")
        base = governed_payload(
            plan_ref,
            {"descriptor.json"},
            kind="plan",
            broker=broker_context.broker,
            broker_parents=plan_parents,
            materialization_root=broker_context.materialization_root,
        )
    else:
        if isinstance(directory, DatasetSnapshot):
            raise DataError("Unregistered plan snapshot requires trusted broker identity")
        data = open_energy(canonical_path, raw_snapshot_dir=raw_snapshot_dir)
        base = governed_payload(
            directory, {"descriptor.json"}, parents=(canonical_path,), kind="plan"
        )
    descriptor = read_json(base / "descriptor.json")
    if (
        descriptor.get("schema_version") != "energy-plan-1.0"
        or descriptor.get("canonical") != data.lineage
    ):
        raise DataError("Energy plan lineage mismatch")
    rows = descriptor.get("source_rows")
    if not isinstance(rows, list) or len(rows) != 2 or any(type(r) is not int for r in rows):
        raise DataError("Invalid selected source rows")
    start, stop = int(str(rows[0])), int(str(rows[1]))
    if (start, stop) != contiguous_range(data):
        raise DataError("Selected range does not match declared quality rule")
    data = selected_dataset(data, start, stop)
    saved = object_dict(descriptor["split"])
    config = config_from_dict(saved["config"])
    if (
        config.window != ForecastWindowSpecification(168, 24, 3600)
        or config.features.observed_delay_steps != 24
    ):
        raise DataError("Energy plan window or availability mismatch")
    if config.protocol.name not in {"strict-chronological-v1", "strict-rolling-origin-v1"}:
        raise DataError("Strict hourly protocol required")
    split = plan_splits(data.timestamps, config)
    if identity(split.to_dict()) != identity(saved):
        raise DataError("Energy plan population mismatch")
    test_ranges = [(f.partition("test").start, f.partition("test").stop) for f in split.folds]
    if any(b[0] < a[1] for a, b in zip(test_ranges, test_ranges[1:], strict=False)):
        raise DataError("Overlapping rolling test partitions")
    return data, split
