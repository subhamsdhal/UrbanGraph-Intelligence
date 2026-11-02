"""Versioned target-specific plans; weather overlap never changes legacy load plans."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    _publish_governed,
    broker_parent_references,
    governed_payload,
    resolve_broker_reference,
)
from energraph.evaluation import contiguous_range, default_config
from energraph.lineage import authorize_sources, resolve_parents
from energraph.multivariable import open_multi
from energraph.variables import registry
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.data.filesystem import safe_absolute
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import (
    FeatureAvailabilityPolicy,
    SplitManifest,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import plan_splits


def select_target(
    canonical: Path | DatasetSnapshot,
    target: str,
    weather: Path | None = None,
    *,
    raw_snapshot_dir: Path | None = None,
    weather_raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    broker_reference: BrokerArtifactRef | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if raw_snapshot_dir is None:
        raise DataError("Current raw context required for target selection")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_reference = (
        canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else broker_reference
    )
    if canonical_reference is not None:
        frame, descriptor = open_multi(
            canonical_path,
            raw_snapshot_dir=raw_snapshot_dir,
            broker_context=broker_context,
            broker_reference=canonical_reference,
            raw_broker_reference=raw_broker_reference,
        )
    else:
        frame, descriptor = open_multi(canonical_path, raw_snapshot_dir=raw_snapshot_dir)
    if target not in frame or registry()[target].kind not in {"load", "generation"}:
        raise DataError("Target must be an available energy power variable")
    original_count = len(frame)
    weather_id: str | None = None
    if weather is not None:
        if weather_raw_snapshot_dir is None:
            raise DataError("Weather source context required")
        wf, _ = open_multi(weather, raw_snapshot_dir=weather_raw_snapshot_dir)
        if not all(registry()[str(c)].kind == "weather" for c in wf):
            raise DataError("Weather overlap requires a weather canonical bundle")
        if set(frame.columns) & set(wf.columns):
            raise DataError("Ambiguous overlapping variable names")
        frame = frame.join(wf, how="inner")
        weather_id = weather.name
    adapter = EvaluationDataset(canonical_path, frame[[target]], {})
    start, stop = contiguous_range(adapter)
    frame = frame.iloc[start:stop].copy()
    lineage: dict[str, object] = {
        "canonical_id": (
            canonical_reference.artifact_id
            if canonical_reference is not None
            else canonical_path.name
        ),
        "canonical_descriptor_id": identity(descriptor),
        "target": target,
        "unit": registry()[target].unit,
        "region": "DE",
        "region_type": "country",
        "weather_canonical_id": weather_id,
        "original_rows": original_count,
        "selected_rows": len(frame),
        "excluded_rows": original_count - len(frame),
        "first_timestamp": str(frame.index[0]),
        "last_timestamp": str(frame.index[-1]),
        "selection": (
            "longest finite hourly target run in optional weather intersection; earliest tie"
        ),
        "source_sha256": descriptor["source_sha256"],
        "source_manifest": descriptor["source_manifest"],
    }
    return frame, lineage


def create_target_plan(
    canonical: Path | DatasetSnapshot,
    root: Path,
    target: str,
    protocol: str,
    *,
    latency: int = 24,
    weather: Path | None = None,
    raw_snapshot_dir: Path | None = None,
    weather_raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> DatasetSnapshot | PendingPublication:
    if type(latency) is not int or not 0 <= latency <= 168:
        raise DataError("Invalid target observation latency")
    if raw_snapshot_dir is None:
        raise DataError("Complete raw source context required for target plan")
    if weather is not None and weather_raw_snapshot_dir is None:
        raise DataError("Complete raw source context required for target plan")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref = canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else None
    if canonical_ref is not None:
        if broker_context is None:
            raise DataError("Broker context required for broker-backed target source")
        resolved = resolve_broker_reference(
            broker_context.broker,
            canonical_ref.artifact_id,
            purpose="scientific_evaluation",
            operation="energy_target_plan_create",
        )
        raw_refs = broker_parent_references(
            broker_context.broker,
            resolved,
            purpose="scientific_evaluation",
            operation="energy_target_plan_create",
        )
        if resolved != canonical_ref or len(raw_refs) != 1 or raw_refs[0].artifact_type != "raw":
            raise DataError("Target canonical broker lineage is unavailable")
        raw_ref = raw_refs[0]
        if weather is None:
            local_parents: tuple[Path, ...] = (safe_absolute(raw_snapshot_dir),)
        else:
            if weather_raw_snapshot_dir is None:
                raise DataError("Weather source context required")
            local_parents = (
                safe_absolute(raw_snapshot_dir),
                safe_absolute(weather_raw_snapshot_dir),
            )
        broker_parents = (canonical_ref,)
    else:
        if broker_context is not None:
            raise DataError("Broker context requires a committed canonical reference")
        raw_ref = None
        local_parents = (canonical_path,) if weather is None else (canonical_path, weather)
        broker_parents = None
    raw_sources = (
        (raw_snapshot_dir,) if weather is None else (raw_snapshot_dir, weather_raw_snapshot_dir)
    )
    graph = resolve_parents(local_parents)
    if (
        graph.sources
        != resolve_parents(tuple(source for source in raw_sources if source is not None)).sources
    ):
        raise DataError("Target plan source closure mismatch")
    authorize_sources(
        graph.sources,
        purpose="scientific_evaluation",
        operation="energy_target_plan_create",
        destination=root,
    )
    frame, lineage = select_target(
        canonical_path,
        target,
        weather,
        raw_snapshot_dir=raw_snapshot_dir,
        weather_raw_snapshot_dir=weather_raw_snapshot_dir,
        broker_context=broker_context if canonical_ref is not None else None,
        broker_reference=canonical_ref,
        raw_broker_reference=raw_ref,
    )
    config = replace(
        default_config(len(frame), protocol),
        features=FeatureAvailabilityPolicy(False, False, latency),
    )
    split = plan_splits(pd.DatetimeIndex(frame.index), config)
    desc = {
        "schema_version": "energy-target-plan-1.0",
        "lineage": lineage,
        "split": split.to_dict(),
        "publication_policy": "assumed latency; no vintage replay",
    }
    with tempfile.TemporaryDirectory(prefix="energy-target-plan-") as tmp:
        write_json(Path(tmp) / "descriptor.json", desc)
        return _publish_governed(
            Path(tmp),
            root,
            "target-plan",
            parents=(canonical_path,)
            if canonical_ref is not None and weather is None
            else local_parents,
            broker_context=broker_context if canonical_ref is not None else None,
            broker_parents=broker_parents,
        )


def read_target_plan(
    canonical: Path | DatasetSnapshot,
    directory: Path | DatasetSnapshot,
    weather: Path | None = None,
    *,
    raw_snapshot_dir: Path | None = None,
    weather_raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> tuple[EvaluationDataset, pd.DataFrame, SplitManifest]:
    if raw_snapshot_dir is None or (weather is not None and weather_raw_snapshot_dir is None):
        raise DataError("Complete raw source context required for target plan read")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref = canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else None
    plan_path = directory.directory if isinstance(directory, DatasetSnapshot) else directory
    plan_ref = directory.broker_reference if isinstance(directory, DatasetSnapshot) else None
    parents = (canonical_path,) if weather is None else (canonical_path, weather)
    raw_refs: tuple[BrokerArtifactRef, ...] = ()
    if canonical_ref is not None or plan_ref is not None:
        if (
            broker_context is None
            or canonical_ref is None
            or plan_ref is None
            or weather is not None
        ):
            raise DataError("Complete single-source broker target-plan context required")
        raw_refs = broker_parent_references(
            broker_context.broker,
            resolve_broker_reference(
                broker_context.broker,
                canonical_ref.artifact_id,
                purpose="local_research",
                operation="energy_multivariable_read",
            ),
            purpose="local_research",
            operation="energy_multivariable_read",
        )
        if len(raw_refs) != 1:
            raise DataError("Target canonical broker source closure is invalid")
        plan_resolved = resolve_broker_reference(
            broker_context.broker,
            plan_ref.artifact_id,
            purpose="model_training",
            operation="energy_target_plan_read",
        )
        plan_parents = broker_parent_references(
            broker_context.broker,
            plan_resolved,
            purpose="model_training",
            operation="energy_target_plan_read",
        )
        if plan_resolved != plan_ref or plan_parents != (canonical_ref,):
            raise DataError("Target plan broker parent differs from canonical identity")
        base = governed_payload(
            plan_ref,
            {"descriptor.json"},
            kind="target-plan",
            broker=broker_context.broker,
            broker_parents=plan_parents,
            materialization_root=broker_context.materialization_root,
        )
    else:
        if broker_context is not None:
            raise DataError("Broker context requires committed target-plan references")
        graph = resolve_parents(parents)
        raw_sources = (
            (raw_snapshot_dir,) if weather is None else (raw_snapshot_dir, weather_raw_snapshot_dir)
        )
        if (
            graph.sources
            != resolve_parents(
                tuple(source for source in raw_sources if source is not None)
            ).sources
        ):
            raise DataError("Target plan read source closure mismatch")
        base = governed_payload(plan_path, {"descriptor.json"}, parents=parents, kind="target-plan")
    desc = read_json(base / "descriptor.json")
    if desc.get("schema_version") != "energy-target-plan-1.0":
        raise DataError("Unsupported target plan")
    saved_lineage = object_dict(desc["lineage"])
    target = str(saved_lineage["target"])
    frame, lineage = select_target(
        canonical_path,
        target,
        weather,
        raw_snapshot_dir=raw_snapshot_dir,
        weather_raw_snapshot_dir=weather_raw_snapshot_dir,
        broker_context=broker_context if canonical_ref is not None else None,
        broker_reference=canonical_ref,
        raw_broker_reference=raw_refs[0] if raw_refs else None,
    )
    if lineage != saved_lineage:
        raise DataError("Target plan source or population mismatch")
    saved = object_dict(desc["split"])
    config = config_from_dict(saved["config"])
    if (
        config.window.history != 168
        or config.window.horizon != 24
        or config.window.interval_seconds != 3600
    ):
        raise DataError("Unsupported target forecast window")
    split = plan_splits(pd.DatetimeIndex(frame.index), config)
    if identity(split.to_dict()) != identity(saved):
        raise DataError("Target plan split mismatch")
    return EvaluationDataset(canonical_path, frame[[target]], lineage), frame, split
