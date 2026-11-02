"""Versioned country-load canonical storage, exact alignment and strict regional plans."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
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
from energraph.broker_tree import inspect_tree
from energraph.canonical import _parse_csv_verified, logical_hash, quality
from energraph.contracts import EnergyVariable
from energraph.evaluation import contiguous_range, default_config
from energraph.lineage import authorize_sources, resolve_parents
from energraph.regions import country_registry
from urbangraph.data import DataError, DatasetSnapshot, verify_snapshot
from urbangraph.data.filesystem import safe_absolute, stream_file
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.contracts import (
    FeatureAvailabilityPolicy,
    SplitManifest,
    config_from_dict,
)
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import plan_splits

FILENAME = "time_series_60min_singleindex.csv"


def align_series(
    series: dict[str, pd.DataFrame], order: tuple[str, ...], policy: str = "masked"
) -> pd.DataFrame:
    country_registry().axis(order)
    if set(series) != set(order) or policy not in {"masked", "strict"}:
        raise DataError("Explicit complete axis and alignment policy required")
    parts = []
    for region in order:
        frame = series[region]
        quality(frame)
        if frame.shape[1] != 1:
            raise DataError("One load attribute per region required")
        parts.append(frame.rename(columns={frame.columns[0]: region}))
    result = pd.concat(parts, axis=1, join="outer").sort_index()
    # Materialize an exact UTC hourly grid; gaps remain missing, never filled.
    result = result.reindex(pd.date_range(result.index[0], result.index[-1], freq="h", tz="UTC"))
    result.index.name = "utc_timestamp"
    if policy == "strict":
        a, b = contiguous_range(EvaluationDataset(Path("."), result, {}))
        result = result.iloc[a:b].copy()
    return result


def regional_quality(frame: pd.DataFrame) -> dict[str, object]:
    per_region = {}
    for region in frame:
        subset = frame[[region]]
        q = quality(subset)
        finite = frame[region].dropna()
        try:
            a, b = contiguous_range(EvaluationDataset(Path("."), subset, {}))
            run: dict[str, object] = {
                "rows": b - a,
                "first": str(frame.index[a]),
                "last": str(frame.index[b - 1]),
            }
        except DataError:
            run = {"rows": 0, "status": "no finite run of at least 1200 hours"}
        per_region[str(region)] = q | {
            "finite": len(finite),
            "missing_rate": float(subset.isna().mean().iloc[0]),
            "first_finite": str(finite.index[0]) if len(finite) else None,
            "last_finite": str(finite.index[-1]) if len(finite) else None,
            "longest_eligible_run": run,
        }
    return {
        "axis": quality(frame),
        "per_region": per_region,
        "joint_finite_rows": int(np.isfinite(frame.to_numpy()).all(axis=1).sum()),
    }


def materialize_regions(
    raw: Path,
    root: Path,
    order: tuple[str, ...],
    *,
    policy: str = "masked",
    broker_context: BrokerPublicationContext | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> DatasetSnapshot | PendingPublication:
    registry = country_registry()
    registry.axis(order)
    graph = resolve_parents((raw,))
    authorize_sources(
        graph.sources,
        purpose="canonicalize",
        operation="energy_regional_canonicalize",
        destination=root,
    )
    verified = verify_snapshot(raw)
    if not verified.valid or verified.manifest is None:
        raise DataError("Raw source integrity failed")
    inventory = {f.relative_path for f in verified.manifest.files}
    if not {FILENAME, "datapackage.json"} <= inventory:
        raise DataError("OPSD CSV and source metadata are required")
    if (broker_context is None) != (raw_broker_reference is None):
        raise DataError("Broker publication requires both broker context and raw identity")
    if broker_context is not None and raw_broker_reference is not None:
        record = broker_context.broker.authorize_directory(
            raw_broker_reference.artifact_id,
            purpose="canonicalize",
            operation="energy_regional_canonicalize",
        )
        if (
            raw_broker_reference.artifact_type != "raw"
            or record.kind != "raw"
            or record.content_sha256 != raw_broker_reference.content_sha256
            or record.revision != raw_broker_reference.revision
            or inspect_tree(raw).digest != record.content_sha256
        ):
            raise DataError("Regional raw snapshot differs from broker-authorized source")
    metadata = read_json(raw / "payload/datapackage.json")
    resources = metadata.get("resources")
    if metadata.get("version") != "2020-10-06" or not isinstance(resources, list):
        raise DataError("Unsupported OPSD release metadata")
    matching = [object_dict(r) for r in resources if object_dict(r).get("path") == FILENAME]
    if len(matching) != 1:
        raise DataError("Ambiguous source resource")
    fields = object_dict(matching[0]["schema"])["fields"]
    if not isinstance(fields, list):
        raise DataError("Missing source schema")
    source_fields = {str(object_dict(f)["name"]): object_dict(f) for f in fields}
    series = {}
    mappings = {}
    for region in order:
        definition = registry.get(region)
        code = definition.source_identifier
        column = code + "_load_actual_entsoe_transparency"
        field = source_fields.get(column)
        if field is None or field.get("unit") != "MW":
            raise DataError("Missing verified country actual-load field or incompatible unit")
        properties = object_dict(field["opsdProperties"])
        if properties != {"Region": code, "Variable": "load_actual_entsoe_transparency"}:
            raise DataError("Source area or actual-load semantics mismatch")
        series[region] = _parse_csv_verified(
            raw / "payload" / FILENAME, EnergyVariable(column=column, region=code)
        )
        mappings[region] = field
    masked = align_series(series, order)
    frame = masked if policy == "masked" else align_series(series, order, policy)
    desc = {
        "schema_version": "energy-regional-canonical-1.0",
        "region_order": list(order),
        "registry": registry.to_dict(),
        "registry_id": registry.registry_id,
        "target_variables": ["actual_load"],
        "layout": "float64[T,region,1]; Parquet columns are region IDs",
        "units": "MW",
        "alignment_policy": policy,
        "source_rows": len(masked),
        "discarded_rows": len(masked) - len(frame),
        "source_quality": regional_quality(masked),
        "quality": regional_quality(frame),
        "source_fields": mappings,
        "source_manifest": verified.manifest.to_dict(),
        "source_sha256": stream_file(raw / "payload" / FILENAME)[1],
        "logical_sha256": logical_hash(frame),
        "mask_policy": "finite per region; zeros valid; no fill",
    }
    with tempfile.TemporaryDirectory(prefix="energy-regions-") as tmp:
        stage = Path(tmp)
        frame.to_parquet(stage / "load.parquet")
        pd.testing.assert_frame_equal(
            frame, pd.read_parquet(stage / "load.parquet"), check_exact=True, check_freq=False
        )
        write_json(stage / "descriptor.json", desc)
        result = _publish_governed(
            stage,
            root,
            "regional-canonical",
            raw_snapshot_dir=raw,
            broker_context=broker_context,
            broker_parents=(raw_broker_reference,) if raw_broker_reference is not None else None,
        )
    if isinstance(result, PendingPublication):
        return result
    open_regions(
        result.directory,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
        broker_reference=result.broker_reference,
        raw_broker_reference=raw_broker_reference,
    )
    return result


def open_regions(
    directory: Path,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    broker_reference: BrokerArtifactRef | None = None,
    raw_broker_reference: BrokerArtifactRef | None = None,
) -> EvaluationDataset:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for regional observations")
    brokered = broker_context is not None or broker_reference is not None
    if brokered:
        if broker_context is None or broker_reference is None or raw_broker_reference is None:
            raise DataError("Complete broker regional-canonical identity required")
        resolved = resolve_broker_reference(
            broker_context.broker,
            broker_reference.artifact_id,
            purpose="local_research",
            operation="energy_regional_canonical_read",
        )
        direct = broker_parent_references(
            broker_context.broker,
            resolved,
            purpose="local_research",
            operation="energy_regional_canonical_read",
        )
        if resolved != broker_reference or direct != (raw_broker_reference,):
            raise DataError("Regional canonical broker lineage changed")
        raw_record = broker_context.broker.authorize_directory(
            raw_broker_reference.artifact_id,
            purpose="local_research",
            operation="energy_regional_canonical_read",
        )
        if (
            raw_record.content_sha256 != raw_broker_reference.content_sha256
            or inspect_tree(raw_snapshot_dir).digest != raw_record.content_sha256
        ):
            raise DataError("Regional raw cache differs from broker-authorized source")
        base = governed_payload(
            resolved,
            {"descriptor.json", "load.parquet"},
            kind="regional-canonical",
            broker=broker_context.broker,
            broker_parents=direct,
            materialization_root=broker_context.materialization_root,
        )
    else:
        base = governed_payload(
            directory,
            {"descriptor.json", "load.parquet"},
            raw_snapshot_dir=raw_snapshot_dir,
            kind="regional-canonical",
        )
    desc = read_json(base / "descriptor.json")
    frame = pd.read_parquet(base / "load.parquet")
    order = tuple(str(c) for c in frame.columns)
    registry = country_registry()
    registry.axis(order)
    if (
        desc.get("schema_version") != "energy-regional-canonical-1.0"
        or desc.get("region_order") != list(order)
        or desc.get("registry_id") != registry.registry_id
        or identity(object_dict(desc.get("registry"))) != identity(registry.to_dict())
        or desc.get("units") != "MW"
        or desc.get("target_variables") != ["actual_load"]
        or any(d != np.dtype("float64") for d in frame.dtypes)
        or desc.get("logical_sha256") != logical_hash(frame)
        or desc.get("quality") != regional_quality(frame)
    ):
        raise DataError("Regional schema, axis or logical integrity mismatch")
    return EvaluationDataset(
        directory,
        frame,
        {
            "canonical_id": (
                broker_reference.artifact_id if brokered and broker_reference else directory.name
            ),
            "descriptor_id": identity(desc),
            "registry_id": registry.registry_id,
            "region_order": list(order),
            "units": "MW",
            "source_sha256": desc["source_sha256"],
        },
    )


def regional_plan(
    canonical: Path | DatasetSnapshot,
    root: Path,
    protocol: str,
    latencies: tuple[int, ...],
    *,
    scaling: str = "per_sensor",
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> DatasetSnapshot | PendingPublication:
    if raw_snapshot_dir is None:
        raise DataError("Current raw context required for regional plan")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref = canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else None
    if canonical_ref is not None:
        if broker_context is None:
            raise DataError("Broker context required for broker-backed regional canonical")
        resolved = resolve_broker_reference(
            broker_context.broker,
            canonical_ref.artifact_id,
            purpose="scientific_evaluation",
            operation="energy_regional_plan_create",
        )
        raw_refs = broker_parent_references(
            broker_context.broker,
            resolved,
            purpose="scientific_evaluation",
            operation="energy_regional_plan_create",
        )
        if resolved != canonical_ref or len(raw_refs) != 1 or raw_refs[0].artifact_type != "raw":
            raise DataError("Regional canonical broker lineage is unavailable")
        raw_ref = raw_refs[0]
        local_parents: tuple[Path, ...] = (safe_absolute(raw_snapshot_dir),)
    else:
        if broker_context is not None:
            raise DataError("Broker context requires a broker-issued canonical reference")
        raw_ref = None
        local_parents = (canonical_path,)
    graph = resolve_parents(local_parents)
    if graph.sources != resolve_parents((raw_snapshot_dir,)).sources:
        raise DataError("Regional plan source closure mismatch")
    authorize_sources(
        graph.sources,
        purpose="scientific_evaluation",
        operation="energy_regional_plan_create",
        destination=root,
    )
    original = open_regions(
        canonical_path,
        raw_snapshot_dir=raw_snapshot_dir,
        broker_context=broker_context if canonical_ref is not None else None,
        broker_reference=canonical_ref,
        raw_broker_reference=raw_ref,
    )
    if len(latencies) != len(original.sensor_ids) or any(
        type(v) is not int or not 0 <= v <= 168 for v in latencies
    ):
        raise DataError("Explicit bounded latency per region required")
    a, b = contiguous_range(original)
    frame = original.frame.iloc[a:b].copy()
    config = replace(
        default_config(len(frame), protocol),
        scaling=scaling,
        features=FeatureAvailabilityPolicy(False, False, max(latencies)),
    )
    split = plan_splits(pd.DatetimeIndex(frame.index), config)
    desc = {
        "schema_version": "energy-regional-plan-1.0",
        "canonical": original.lineage,
        "rows": [a, b],
        "region_latencies": dict(zip(original.sensor_ids, latencies, strict=True)),
        "availability": "per-region assumed interval-start delays; publication vintages unverified",
        "selection": "longest joint finite hourly run, earliest tie",
        "split": split.to_dict(),
    }
    with tempfile.TemporaryDirectory(prefix="energy-region-plan-") as tmp:
        write_json(Path(tmp) / "descriptor.json", desc)
        return _publish_governed(
            Path(tmp),
            root,
            "regional-plan",
            parents=(canonical_path,) if canonical_ref is not None else local_parents,
            broker_context=broker_context if canonical_ref is not None else None,
            broker_parents=(canonical_ref,) if canonical_ref is not None else None,
        )


def load_regional_plan(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    *,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> tuple[EvaluationDataset, SplitManifest, dict[str, int]]:
    if raw_snapshot_dir is None:
        raise DataError("Current raw source context required for regional plan read")
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    canonical_ref = canonical.broker_reference if isinstance(canonical, DatasetSnapshot) else None
    plan_path = plan.directory if isinstance(plan, DatasetSnapshot) else plan
    plan_ref = plan.broker_reference if isinstance(plan, DatasetSnapshot) else None
    if canonical_ref is not None or plan_ref is not None:
        if broker_context is None or canonical_ref is None or plan_ref is None:
            raise DataError("Complete broker regional-plan identities required")
        canonical_resolved = resolve_broker_reference(
            broker_context.broker,
            canonical_ref.artifact_id,
            purpose="local_research",
            operation="energy_regional_canonical_read",
        )
        raw_refs = broker_parent_references(
            broker_context.broker,
            canonical_resolved,
            purpose="local_research",
            operation="energy_regional_canonical_read",
        )
        if canonical_resolved != canonical_ref or len(raw_refs) != 1:
            raise DataError("Regional canonical broker identity changed")
        original = open_regions(
            canonical_path,
            raw_snapshot_dir=raw_snapshot_dir,
            broker_context=broker_context,
            broker_reference=canonical_ref,
            raw_broker_reference=raw_refs[0],
        )
        plan_resolved = resolve_broker_reference(
            broker_context.broker,
            plan_ref.artifact_id,
            purpose="model_training",
            operation="energy_regional_plan_read",
        )
        plan_parents = broker_parent_references(
            broker_context.broker,
            plan_resolved,
            purpose="model_training",
            operation="energy_regional_plan_read",
        )
        if plan_resolved != plan_ref or plan_parents != (canonical_ref,):
            raise DataError("Regional plan broker parent differs from canonical identity")
        base = governed_payload(
            plan_resolved,
            {"descriptor.json"},
            kind="regional-plan",
            broker=broker_context.broker,
            broker_parents=plan_parents,
            materialization_root=broker_context.materialization_root,
        )
    else:
        if broker_context is not None:
            raise DataError("Broker context requires committed regional plan references")
        original = open_regions(canonical_path, raw_snapshot_dir=raw_snapshot_dir)
        base = governed_payload(
            plan_path, {"descriptor.json"}, parents=(canonical_path,), kind="regional-plan"
        )
    desc = read_json(base / "descriptor.json")
    a, b = contiguous_range(original)
    if (
        desc.get("schema_version") != "energy-regional-plan-1.0"
        or desc.get("canonical") != original.lineage
        or desc.get("rows") != [a, b]
    ):
        raise DataError("Regional plan lineage or quality population mismatch")
    latency_obj = object_dict(desc["region_latencies"])
    if set(latency_obj) != set(original.sensor_ids) or any(
        type(v) is not int or not 0 <= v <= 168 for v in latency_obj.values()
    ):
        raise DataError("Invalid regional latency mapping")
    latencies = {k: int(str(v)) for k, v in latency_obj.items()}
    frame = original.frame.iloc[a:b].copy()
    saved = object_dict(desc["split"])
    config = config_from_dict(saved["config"])
    if (
        config.window.history != 168
        or config.window.horizon != 24
        or config.window.interval_seconds != 3600
        or config.features.observed_delay_steps != max(latencies.values())
        or config.protocol.name not in {"strict-chronological-v1", "strict-rolling-origin-v1"}
    ):
        raise DataError("Unsupported regional evaluation contract")
    split = plan_splits(pd.DatetimeIndex(frame.index), config)
    if identity(split.to_dict()) != identity(saved):
        raise DataError("Regional split population mismatch")
    return (
        EvaluationDataset(canonical_path, frame, original.lineage | {"selected_rows": [a, b]}),
        split,
        latencies,
    )
