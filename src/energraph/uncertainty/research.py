"""Immutable calibrated intervals and independent numerical verification."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import cast

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
from energraph.broker_adapter import BrokerAdapter
from energraph.broker_tree import inspect_tree
from energraph.experiments import code_fingerprint
from energraph.lineage import authorize_sources, resolve_parents
from energraph.uncertainty.conformal import CalibrationPolicy, fit_calibrator
from energraph.uncertainty.metrics import dependence, interval_metrics
from energraph.uncertainty.validation import validate_partitions
from urbangraph.data import DataError, DatasetSnapshot, verify_snapshot
from urbangraph.data.verification import read_manifest
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity


def _snapshot_path(value: Path | DatasetSnapshot) -> Path:
    return value.directory if isinstance(value, DatasetSnapshot) else value


def _snapshot_reference(value: DatasetSnapshot | BrokerArtifactRef) -> BrokerArtifactRef:
    reference = value.broker_reference if isinstance(value, DatasetSnapshot) else value
    if reference is None or reference.artifact_type not in {
        "raw",
        "uncertainty-population",
        "interval",
    }:
        raise DataError("Committed broker identity required for uncertainty artifact")
    return reference


def _authorize_broker_closure(
    broker: BrokerAdapter,
    reference: BrokerArtifactRef,
    *,
    purpose: str,
    operation: str,
) -> tuple[BrokerArtifactRef, ...]:
    """Resolve and authorize every broker parent before protected local reads."""
    seen: set[str] = set()
    ordered: list[BrokerArtifactRef] = []

    def visit(current: BrokerArtifactRef) -> None:
        if current.artifact_id in seen:
            raise DataError("Broker lineage contains a cycle or duplicate identity")
        seen.add(current.artifact_id)
        resolved = resolve_broker_reference(
            broker, current.artifact_id, purpose=purpose, operation=operation
        )
        if resolved != current:
            raise DataError("Broker lineage identity changed")
        ordered.append(current)
        direct = broker_parent_references(broker, current, purpose=purpose, operation=operation)
        for parent in direct:
            visit(parent)

    visit(reference)
    return tuple(ordered)


def evaluate_population(
    population: Path | DatasetSnapshot,
    root: Path,
    model: str,
    policy: CalibrationPolicy,
    *,
    population_parents: tuple[Path, ...] | None = None,
    broker_context: BrokerPublicationContext | None = None,
    population_parent_references: tuple[BrokerArtifactRef, ...] | None = None,
) -> Path | DatasetSnapshot | PendingPublication:
    if not population_parents:
        raise DataError("Current population parent context required")
    if model not in {"lightgbm", "persistence"}:
        raise DataError("Unknown point reference")
    population_path = _snapshot_path(population)
    population_reference: BrokerArtifactRef | None = None
    if broker_context is None:
        if population_parent_references is not None:
            raise DataError("Broker parent identities require an authenticated broker context")
        population_base = governed_payload(
            population_path,
            {"predictions.npz", "descriptor.json", "runtime.json"},
            parents=population_parents,
            kind="uncertainty-population",
        )
        population_name = population_path.name
        descriptor = read_json(population_base / "descriptor.json")
        with np.load(population_base / "predictions.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        if population_parent_references is None or not population_parent_references:
            raise DataError("Committed population parent identities are required")
        if not isinstance(population, DatasetSnapshot):
            raise DataError("A local committed population snapshot is required for publication")
        population_reference = _snapshot_reference(population)
        if population_reference.artifact_type != "uncertainty-population":
            raise DataError("Committed uncertainty population is required")
        broker = broker_context.broker
        resolved = resolve_broker_reference(
            broker,
            population_reference.artifact_id,
            purpose="scientific_evaluation",
            operation="energy_interval_create",
        )
        if resolved != population_reference:
            raise DataError("Population broker reference changed")
        actual_parents = broker_parent_references(
            broker,
            population_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_create",
        )
        if tuple(sorted(actual_parents, key=lambda item: item.artifact_id)) != tuple(
            sorted(population_parent_references, key=lambda item: item.artifact_id)
        ):
            raise DataError("Population broker parents differ from supplied identities")
        _authorize_broker_closure(
            broker,
            population_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_create",
        )
        _authorize_broker_closure(
            broker,
            population_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_verify",
        )
        if inspect_tree(population_path / "payload").digest != population_reference.content_sha256:
            raise DataError("Population presentation differs from broker identity")
        with tempfile.TemporaryDirectory(dir=broker_context.materialization_root) as material_root:
            population_base = governed_payload(
                population_reference,
                {"predictions.npz", "descriptor.json", "runtime.json"},
                broker=broker,
                broker_parents=actual_parents,
                materialization_root=Path(material_root),
                kind="uncertainty-population",
            )
            descriptor = read_json(population_base / "descriptor.json")
            with np.load(population_base / "predictions.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
        population_name = population_reference.artifact_id
    source = object_dict(descriptor["specification"])
    spec = {
        "schema_version": "energy-interval-1.0",
        "population_snapshot": population_name,
        "source": source,
        "model": model,
        "policy": policy.to_dict(),
        "code_sha256": code_fingerprint(),
        "scale_policy": "validation mean absolute error per region/horizon; floor 1 MW"
        if policy.normalized
        else "unit scale",
        "clip_bounds": False,
    }
    experiment_id = identity(spec)
    if broker_context is None:
        authorize_sources(
            resolve_parents(population_parents).sources,
            purpose="scientific_evaluation",
            operation="energy_interval_verify",
            destination=root,
        )
        for path in (root / "energraph-interval").glob("*/payload/descriptor.json"):
            if read_json(path).get("experiment_id") == experiment_id:
                verify_intervals(
                    path.parent.parent, population_path, population_parents=population_parents
                )
                return path.parent.parent
    else:
        assert population_reference is not None
        for manifest_path in (root / "energraph-interval").glob("*/manifest.json"):
            manifest = read_manifest(manifest_path)
            source_reference = manifest.source.reference or ""
            prefix = "Committed broker artifact "
            if not source_reference.startswith(prefix):
                continue
            candidate_id = source_reference[len(prefix) :]
            candidate_reference = resolve_broker_reference(
                broker_context.broker,
                candidate_id,
                purpose="scientific_evaluation",
                operation="energy_interval_verify",
            )
            if candidate_reference.artifact_type != "interval":
                continue
            direct = broker_parent_references(
                broker_context.broker,
                candidate_reference,
                purpose="scientific_evaluation",
                operation="energy_interval_verify",
            )
            if direct != (population_reference,):
                continue
            checked = verify_snapshot(manifest_path.parent)
            if not checked.valid or checked.manifest is None:
                raise DataError("Saved committed interval presentation failed integrity")
            if (
                inspect_tree(manifest_path.parent / "payload").digest
                != candidate_reference.content_sha256
            ):
                raise DataError("Saved interval presentation differs from broker identity")
            candidate = DatasetSnapshot(
                manifest_path.parent, checked.manifest, broker_reference=candidate_reference
            )
            verify_intervals(
                candidate,
                population,
                population_parents=population_parents,
                broker_context=broker_context,
                population_parent_references=population_parent_references,
            )
            with tempfile.TemporaryDirectory(
                dir=broker_context.materialization_root
            ) as material_root:
                candidate_descriptor = governed_payload(
                    candidate_reference,
                    {"intervals.npz", "descriptor.json", "runtime.json"},
                    broker=broker_context.broker,
                    broker_parents=(population_reference,),
                    materialization_root=Path(material_root),
                    kind="interval",
                )
                candidate_experiment_id = read_json(candidate_descriptor / "descriptor.json").get(
                    "experiment_id"
                )
            if candidate_experiment_id == experiment_id:
                return candidate
    began = time.perf_counter()
    regions = tuple(str(r) for r in cast(list[object], source["region_ids"]))
    validate_partitions(
        arrays, regions, max(int(str(v)) for v in object_dict(source["latencies"]).values())
    )
    truth = arrays["calibration_truth"]
    pred = arrays["calibration_" + model]
    scales = (
        np.maximum(
            np.mean(np.abs(arrays["validation_truth"] - arrays["validation_" + model]), axis=0), 1.0
        )
        if policy.normalized
        else None
    )
    part = object_dict(object_dict(descriptor["partitions"])["calibration"])
    cal = fit_calibrator(
        truth,
        pred,
        np.isfinite(truth),
        arrays["calibration_origins"],
        policy=policy,
        region_ids=regions,
        population_id=str(part["population_id"]),
        model_id=population_name + ":" + model,
        observation_delay=max(int(str(v)) for v in object_dict(source["latencies"]).values()),
        scales=scales,
    )
    prediction = arrays["test_" + model]
    actual = arrays["test_truth"]
    mask = np.isfinite(actual)
    lower, upper = cal.intervals(
        prediction,
        arrays["test_origins"],
        region_ids=regions,
        model_id=population_name + ":" + model,
    )
    metrics = interval_metrics(actual, prediction, lower, upper, mask, policy.coverage)
    rows = []
    for j, region in enumerate(regions):
        for h in range(24):
            ix = (slice(None), slice(h, h + 1), slice(j, j + 1))
            rows.append(
                {
                    "region": region,
                    "horizon": h + 1,
                    "calibration_count": int(cal.counts[h, j]),
                    "rank": int(cal.ranks[h, j]),
                    "fallback": bool(cal.fallback[h, j]),
                    **interval_metrics(
                        actual[ix], prediction[ix], lower[ix], upper[ix], mask[ix], policy.coverage
                    ),
                }
            )
    blocks = []
    for label, selection in [
        ("earlier", slice(0, len(actual) // 2)),
        ("later", slice(len(actual) // 2, None)),
    ]:
        blocks.append(
            {
                "period": label,
                **interval_metrics(
                    actual[selection],
                    prediction[selection],
                    lower[selection],
                    upper[selection],
                    mask[selection],
                    policy.coverage,
                ),
            }
        )
    diagnostic = dependence(np.abs(truth - pred), arrays["calibration_origins"])
    with tempfile.TemporaryDirectory(prefix="energy-interval-") as tmp:
        stage = Path(tmp)
        np.savez_compressed(
            stage / "intervals.npz",
            lower=lower,
            upper=upper,
            point=prediction,
            truth=actual,
            mask=mask,
            origins=arrays["test_origins"],
            targets_ns=arrays["test_targets_ns"],
        )
        write_json(
            stage / "descriptor.json",
            {
                "specification": spec,
                "experiment_id": experiment_id,
                "calibrator": cal.to_dict(),
                "metrics": metrics,
                "region_horizon": rows,
                "time_blocks": blocks,
                "calibration_dependence": diagnostic,
            },
        )
        write_json(
            stage / "runtime.json",
            {"calibration_and_evaluation_seconds": time.perf_counter() - began},
        )
        if broker_context is None:
            result = _publish_governed(stage, root, "interval", parents=(population_path,))
        else:
            assert population_reference is not None
            result = _publish_governed(
                stage,
                root,
                "interval",
                parents=(population_path,),
                broker_context=broker_context,
                broker_parents=(population_reference,),
            )
    if isinstance(result, PendingPublication):
        return result
    if broker_context is not None:
        return result
    verify_intervals(result.directory, population_path, population_parents=population_parents)
    return result.directory


def verify_intervals(
    directory: Path | DatasetSnapshot | BrokerArtifactRef,
    population: Path | DatasetSnapshot | BrokerArtifactRef,
    *,
    population_parents: tuple[Path, ...] | None = None,
    broker_context: BrokerPublicationContext | None = None,
    population_parent_references: tuple[BrokerArtifactRef, ...] | None = None,
) -> dict[str, object]:
    if broker_context is None:
        if not population_parents or population_parent_references is not None:
            raise DataError("Current population source context required")
        if isinstance(directory, BrokerArtifactRef) or isinstance(population, BrokerArtifactRef):
            raise DataError("Broker artifact identities require an authenticated context")
        directory_path = _snapshot_path(directory)
        population_path = _snapshot_path(population)
        interval_base = governed_payload(
            directory_path,
            {"intervals.npz", "descriptor.json", "runtime.json"},
            parents=(population_path,),
            kind="interval",
        )
        population_base = governed_payload(
            population_path,
            {"predictions.npz", "descriptor.json", "runtime.json"},
            parents=population_parents,
            kind="uncertainty-population",
        )
        population_name = population_path.name
        snapshot_id = directory_path.name
    else:
        if population_parent_references is None or not population_parent_references:
            raise DataError("Committed population ancestry references are required")
        if not isinstance(directory, DatasetSnapshot | BrokerArtifactRef) or not isinstance(
            population, DatasetSnapshot | BrokerArtifactRef
        ):
            raise DataError("Broker verification requires broker-issued artifact identities")
        interval_reference = _snapshot_reference(directory)
        population_reference = _snapshot_reference(population)
        if interval_reference.artifact_type != "interval":
            raise DataError("Committed interval artifact identity is required")
        if population_reference.artifact_type != "uncertainty-population":
            raise DataError("Committed population artifact identity is required")
        broker = broker_context.broker
        interval_direct = broker_parent_references(
            broker,
            interval_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_verify",
        )
        if interval_direct != (population_reference,):
            raise DataError("Interval broker parent differs from committed population")
        population_direct = broker_parent_references(
            broker,
            population_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_verify",
        )
        if tuple(sorted(population_direct, key=lambda item: item.artifact_id)) != tuple(
            sorted(population_parent_references, key=lambda item: item.artifact_id)
        ):
            raise DataError("Population broker ancestry differs from supplied identities")
        _authorize_broker_closure(
            broker,
            interval_reference,
            purpose="scientific_evaluation",
            operation="energy_interval_verify",
        )
        if (
            isinstance(population, DatasetSnapshot)
            and inspect_tree(population.directory / "payload").digest
            != population_reference.content_sha256
        ):
            raise DataError("Population presentation differs from broker identity")
        if (
            isinstance(directory, DatasetSnapshot)
            and inspect_tree(directory.directory / "payload").digest
            != interval_reference.content_sha256
        ):
            raise DataError("Interval presentation differs from broker identity")
        with tempfile.TemporaryDirectory(dir=broker_context.materialization_root) as material_root:
            root = Path(material_root)
            population_base = governed_payload(
                population_reference,
                {"predictions.npz", "descriptor.json", "runtime.json"},
                broker=broker,
                broker_parents=population_direct,
                materialization_root=root,
                kind="uncertainty-population",
            )
            interval_base = governed_payload(
                interval_reference,
                {"intervals.npz", "descriptor.json", "runtime.json"},
                broker=broker,
                broker_parents=(population_reference,),
                materialization_root=root,
                kind="interval",
            )
            return _verify_interval_payloads(
                interval_base,
                population_base,
                population_name=population_reference.artifact_id,
                snapshot_id=interval_reference.artifact_id,
            )
    return _verify_interval_payloads(
        interval_base,
        population_base,
        population_name=population_name,
        snapshot_id=snapshot_id,
    )


def _verify_interval_payloads(
    directory: Path,
    population: Path,
    *,
    population_name: str,
    snapshot_id: str,
) -> dict[str, object]:
    d = read_json(directory / "descriptor.json")
    spec = object_dict(d["specification"])
    if (
        spec.get("schema_version") != "energy-interval-1.0"
        or identity(spec) != d["experiment_id"]
        or spec["population_snapshot"] != population_name
    ):
        raise DataError("Interval identity or population mismatch")
    source_desc = read_json(population / "descriptor.json")
    source = object_dict(source_desc["specification"])
    if source != spec["source"]:
        raise DataError("Interval source provenance mismatch")
    p = object_dict(spec["policy"])
    policy = CalibrationPolicy(
        float(str(p["coverage"])),
        str(p["grouping"]),
        int(str(p["minimum_count"])),
        int(str(p["origin_spacing"])),
        cast(bool, p["normalized"]),
        str(p["version"]),
    )
    model = str(spec["model"])
    if model not in {"lightgbm", "persistence"}:
        raise DataError("Unknown point reference")
    with np.load(population / "predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    regions = tuple(str(r) for r in cast(list[object], source["region_ids"]))
    validate_partitions(
        arrays, regions, max(int(str(v)) for v in object_dict(source["latencies"]).values())
    )
    truth = arrays["calibration_truth"]
    pred = arrays["calibration_" + model]
    scales = (
        np.maximum(
            np.mean(np.abs(arrays["validation_truth"] - arrays["validation_" + model]), axis=0), 1.0
        )
        if policy.normalized
        else None
    )
    part = object_dict(object_dict(source_desc["partitions"])["calibration"])
    cal = fit_calibrator(
        truth,
        pred,
        np.isfinite(truth),
        arrays["calibration_origins"],
        policy=policy,
        region_ids=regions,
        population_id=str(part["population_id"]),
        model_id=population_name + ":" + model,
        observation_delay=max(int(str(v)) for v in object_dict(source["latencies"]).values()),
        scales=scales,
    )
    if cal.to_dict() != d["calibrator"]:
        raise DataError("Saved calibrator differs from independent historical fit")
    lo, hi = cal.intervals(
        arrays["test_" + model],
        arrays["test_origins"],
        region_ids=regions,
        model_id=population_name + ":" + model,
    )
    with np.load(directory / "intervals.npz", allow_pickle=False) as z:
        expected = {
            "lower": lo,
            "upper": hi,
            "point": arrays["test_" + model],
            "truth": arrays["test_truth"],
            "mask": np.isfinite(arrays["test_truth"]),
            "origins": arrays["test_origins"],
            "targets_ns": arrays["test_targets_ns"],
        }
        if set(z.files) != set(expected):
            raise DataError("Unexpected interval arrays")
        for name, value in expected.items():
            if not np.array_equal(z[name], value, equal_nan=True):
                raise DataError("Interval array mismatch: " + name)
        measured = interval_metrics(
            z["truth"], z["point"], z["lower"], z["upper"], z["mask"], policy.coverage
        )
    if measured != d["metrics"]:
        raise DataError("Interval metrics mismatch")
    return {
        "valid": True,
        "snapshot_id": snapshot_id,
        "experiment_id": d["experiment_id"],
        "metrics": measured,
    }
