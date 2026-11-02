"""Matched non-GNN regional LightGBM pilot using existing causal features and native trainer."""

from __future__ import annotations

import tempfile
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
from energraph.energy_boosting import sampled_rows, table_metrics
from energraph.energy_features import EnergyFeatures, build_features
from energraph.energy_graphs import verify_energy_graph
from energraph.experiments import code_fingerprint, origins
from energraph.graph_authority import authorize_graph_inputs, snapshot_path
from energraph.lineage import authorize_sources, resolve_parents
from energraph.regional_data import load_regional_plan
from energraph.regions import country_registry
from urbangraph.boosting.contracts import BoostConfig, config_from_dict
from urbangraph.boosting.models import load_model, train_model
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.data.filesystem import discover
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer, fit_standardizer
from urbangraph.reproducibility import collect_environment_info


@dataclass(frozen=True)
class RegionalFeatures:
    target: str
    inputs: tuple[str, ...]
    latencies: tuple[int, ...]
    mode: str
    version: str = "regional-features-1.0"

    def __post_init__(self) -> None:
        country_registry().axis(self.inputs)
        if (
            self.version != "regional-features-1.0"
            or self.target not in self.inputs
            or self.mode not in {"M0", "M1", "M2"}
            or len(self.inputs) != len(self.latencies)
            or any(type(v) is not int or not 0 <= v <= 168 for v in self.latencies)
            or (self.mode == "M0" and self.inputs != (self.target,))
        ):
            raise DataError("Invalid regional feature/availability contract")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def regional_features(
    frame: pd.DataFrame,
    origin: NDArray[np.int64],
    horizon: NDArray[np.int64],
    spec: RegionalFeatures,
    *,
    lower: int = 0,
) -> tuple[NDArray[np.float32], tuple[str, ...], tuple[str, ...]]:
    chunks = []
    names: list[str] = []
    groups: list[str] = []
    for region, latency in zip(spec.inputs, spec.latencies, strict=True):
        if region not in frame:
            raise DataError("Missing input region")
        local = frame[[region]].rename(columns={region: "load"})
        features = EnergyFeatures(("load",), (latency,), calendar=region == spec.target)
        x, n, _ = build_features(local, origin, horizon, features, lower=lower)
        keep = [i for i, name in enumerate(n) if region == spec.target or name != "horizon"]
        chunks.append(x[:, keep])
        names.extend(region.replace(":", "_") + "_" + n[i] for i in keep)
        groups.extend(region for _ in keep)
    return np.concatenate(chunks, axis=1), tuple(names), tuple(groups)


def run_regional(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    geographic_graph: Path | DatasetSnapshot,
    root: Path,
    config: BoostConfig,
    *,
    raw_snapshot_dir: Path | None = None,
    geography: Path | None = None,
    geography_reference: str | None = None,
    geometry_reference: str | None = None,
    broker_context: BrokerPublicationContext | None = None,
    geography_snapshot: Path | DatasetSnapshot | None = None,
    geography_artifact_reference: BrokerArtifactRef | None = None,
    geography_relative_path: str | None = None,
    geometry_snapshot: Path | DatasetSnapshot | None = None,
    geometry_artifact_reference: BrokerArtifactRef | None = None,
    geometry_relative_path: str | None = None,
) -> Path | PendingPublication | DatasetSnapshot:
    if raw_snapshot_dir is None:
        raise DataError("Complete current regional source context required for training")
    geography_parents: tuple[tuple[Path | DatasetSnapshot, BrokerArtifactRef], ...] = ()
    if broker_context is not None:
        if (
            geography_snapshot is None
            or geography_artifact_reference is None
            or not geography_relative_path
            or (geometry_snapshot is None) != (geometry_artifact_reference is None)
            or (geometry_snapshot is None) != (geometry_relative_path is None)
        ):
            raise DataError(
                "Broker-backed regional training requires admitted geographic identities"
            )
        geography_parents = ((geography_snapshot, geography_artifact_reference),)
        if geometry_snapshot is not None and geometry_artifact_reference is not None:
            geography_parents += ((geometry_snapshot, geometry_artifact_reference),)
        trusted_graph = authorize_graph_inputs(
            canonical,
            plan,
            raw_snapshot_dir=raw_snapshot_dir,
            broker_context=broker_context,
            purpose="model_training",
            operation="energy_regional_train",
            graph=geographic_graph,
            geographic_parents=geography_parents,
        )
        if trusted_graph is None:
            raise DataError("Regional training broker authority unavailable")
        canonical_path, plan_path, graph_path = (
            snapshot_path(canonical),
            snapshot_path(plan),
            snapshot_path(geographic_graph),
        )
        parents: tuple[Path, ...] = (canonical_path, plan_path, graph_path)
        broker_parents = (trusted_graph[0], trusted_graph[1], trusted_graph[-1])
        lineage = resolve_parents(
            (raw_snapshot_dir, *(snapshot_path(item[0]) for item in geography_parents))
        )
    else:
        if geography is None or not geography_reference:
            raise DataError("Complete current regional source context required for training")
        canonical_path, plan_path, graph_path = (
            snapshot_path(canonical),
            snapshot_path(plan),
            snapshot_path(geographic_graph),
        )
        parents = (canonical_path, plan_path, graph_path)
        lineage = resolve_parents(parents)
        if any(
            source not in lineage.sources for source in resolve_parents((raw_snapshot_dir,)).sources
        ):
            raise DataError("Regional training raw source closure mismatch")
        broker_parents = None
    authorize_sources(
        lineage.sources,
        purpose="model_training",
        operation="energy_regional_train",
        destination=root,
    )
    data, split, latencies = load_regional_plan(
        canonical, plan, raw_snapshot_dir=raw_snapshot_dir, broker_context=broker_context
    )
    adjacency, graph = verify_energy_graph(
        geographic_graph,
        canonical=canonical,
        plan=plan,
        raw_snapshot_dir=raw_snapshot_dir,
        geography=geography,
        geography_reference=geography_reference,
        geometry_reference=geometry_reference,
        broker_context=broker_context,
        geography_snapshot=geography_snapshot,
        geography_artifact_reference=geography_artifact_reference,
        geography_relative_path=geography_relative_path,
        geometry_snapshot=geometry_snapshot,
        geometry_artifact_reference=geometry_artifact_reference,
        geometry_relative_path=geometry_relative_path,
        expected_order=data.sensor_ids,
        expected_plan=plan_path.name,
    )
    if object_dict(graph["specification"])["method"] != "geographic-proxy":
        raise DataError("M2 requires geographic proxy")
    if config.family != "lightgbm":
        raise DataError("Regional pilot uses existing native LightGBM")
    spec = {
        "schema_version": "energy-regional-experiment-1.0",
        "canonical": data.lineage,
        "plan_id": plan_path.name,
        "split_id": identity(split.to_dict()),
        "graph_id": graph["graph_id"],
        "region_latencies": latencies,
        "config": config.to_dict(),
        "modes": ["M0", "M1", "M2"],
        "code_sha256": code_fingerprint(),
        "library_version": version("lightgbm"),
        "selection": "minimum sampled validation MAE; first tie; calibration unused",
        "normalization": "training-only per-region standard deviation; physical-unit metrics",
        "graph_claim": "tabular feature-selection pilot, NOT graph neural forecasting",
    }
    experiment_id = identity(spec)
    authorize_sources(
        lineage.sources,
        purpose="saved_run_verification",
        operation="energy_regional_verify",
        destination=root,
    )
    existing = (
        []
        if broker_context is not None
        else [
            p.parent
            for p in (root / "energraph-regional-run").glob("*/manifest.json")
            if read_json(p.parent / "payload/descriptor.json").get("experiment_id") == experiment_id
        ]
    )
    if existing:
        if len(existing) != 1:
            raise DataError("Conflicting regional experiment identity")
        verify_regional_run(
            existing[0],
            canonical=canonical,
            plan=plan,
            graph=geographic_graph,
            raw_snapshot_dir=raw_snapshot_dir,
        )
        return existing[0]
    records = []
    began = time.perf_counter()
    deadline = began + config.run_seconds
    with tempfile.TemporaryDirectory(prefix="regional-pilot-") as tmp:
        stage = Path(tmp)
        for fold_id, fold in enumerate(split.folds):
            scaler = fit_standardizer(data, split, fold=fold_id)
            if scaler.mode != "per_sensor":
                raise DataError("Regional pilot requires per-region scaling")
            write_json(stage / f"{fold.fold_id}-scaler.json", scaler.to_dict())
            scaled = pd.DataFrame(
                scaler.transform(data.frame.to_numpy(dtype=np.float64), sensor_ids=data.sensor_ids),
                index=data.frame.index,
                columns=data.frame.columns,
            )
            train_o, train_h = sampled_rows(
                origins(split, fold_id, "train"), config.max_train_rows, config.seed
            )
            val_o, val_h = sampled_rows(
                origins(split, fold_id, "validation"), config.max_validation_rows, config.seed
            )
            test_o = origins(split, fold_id, "test")
            for target_idx, target in enumerate(data.sensor_ids):
                for mode in ("M0", "M1", "M2"):
                    inputs = (
                        (target,)
                        if mode == "M0"
                        else tuple(
                            r
                            for j, r in enumerate(data.sensor_ids)
                            if mode == "M1" or r == target or adjacency[target_idx, j] > 0
                        )
                    )
                    feature = RegionalFeatures(
                        target, inputs, tuple(latencies[r] for r in inputs), mode
                    )
                    x, names, groups = regional_features(
                        scaled, train_o, train_h, feature, lower=fold.partition("train").start
                    )
                    vx, vn, _ = regional_features(scaled, val_o, val_h, feature)
                    y = scaled[target].to_numpy(dtype=np.float64)
                    trials = []
                    models = []
                    prefix = f"{fold.fold_id}-{target.split(':')[1]}-{mode}"
                    for i, depth in enumerate((config.max_depth, config.alternative_depth)):
                        candidate = replace(
                            config, max_depth=depth, num_leaves=min(2**depth, config.num_leaves)
                        )
                        model, metadata = train_model(
                            candidate,
                            feature,
                            names,
                            groups,
                            (target,),
                            24,
                            x,
                            y[train_o + train_h],
                            vx,
                            y[val_o + val_h],
                            deadline,
                        )
                        score = float(
                            np.abs(model.predict(vx, vn, (target,)) - y[val_o + val_h]).mean()
                        )
                        filename = f"{prefix}-trial-{i}.txt"
                        model.save(stage / filename)
                        trials.append(
                            {
                                "config": candidate.to_dict(),
                                "metadata": metadata,
                                "validation_mae_normalized": score,
                                "model_file": filename,
                            }
                        )
                        models.append(model)
                    choice = int(
                        np.argmin([float(str(t["validation_mae_normalized"])) for t in trials])
                    )
                    selected = models[choice]
                    restored = load_model(
                        stage / str(trials[choice]["model_file"]),
                        selected.config,
                        feature,
                        names,
                        groups,
                        (target,),
                        24,
                        selected.best_iteration,
                    )
                    np.testing.assert_array_equal(
                        selected.predict(vx, names, (target,)),
                        restored.predict(vx, names, (target,)),
                    )
                    probe = pd.DataFrame(vx[:48], columns=list(names))
                    probe["prediction"] = restored.predict(vx[:48], names, (target,))
                    probe.to_parquet(stage / f"{prefix}-probe.parquet", index=False)
                    predictions = []
                    missing = 0
                    cells = 0
                    for offset in range(0, len(test_o), config.evaluation_batch):
                        o = test_o[offset : offset + config.evaluation_batch]
                        tx, tn, _ = regional_features(
                            scaled,
                            np.repeat(o, 24),
                            np.tile(np.arange(1, 25, dtype=np.int64), len(o)),
                            feature,
                        )
                        missing += int(np.isnan(tx).sum())
                        cells += tx.size
                        predictions.append(restored.predict(tx, tn, (target,)))
                    prediction = (
                        np.concatenate(predictions) * scaler.scales[target_idx]
                        + scaler.means[target_idx]
                    )
                    targets = test_o[:, None] + np.arange(1, 25)
                    table = pd.DataFrame(
                        {
                            "origin_row": np.repeat(test_o, 24),
                            "target_row": targets.ravel(),
                            "target_utc": data.timestamps[targets.ravel()],
                            "actual_MW": data.frame[target].to_numpy(dtype=np.float64)[
                                targets.ravel()
                            ],
                            "lightgbm": prediction,
                        }
                    )
                    table.to_parquet(stage / f"{prefix}-predictions.parquet", index=False)
                    metrics = table_metrics(table, "lightgbm")
                    scale = scaler.scales[target_idx]
                    overall = object_dict(metrics["overall"])
                    records.append(
                        {
                            "prefix": prefix,
                            "fold": fold.fold_id,
                            "target": target,
                            "mode": mode,
                            "feature_spec": feature.to_dict(),
                            "names": list(names),
                            "groups": list(groups),
                            "train_rows": len(x),
                            "validation_rows": len(vx),
                            "forecast_origins": len(test_o),
                            "population_id": identity(
                                {
                                    "plan": plan_path.name,
                                    "fold": fold.fold_id,
                                    "target": target,
                                    "origins": test_o.tolist(),
                                }
                            ),
                            "training_scale_MW": scale,
                            "training_mean_MW": scaler.means[target_idx],
                            "normalized_mae": float(str(overall["mae_MW"])) / scale,
                            "normalized_rmse": float(str(overall["rmse_MW"])) / scale,
                            "metrics": metrics,
                            "trials": trials,
                            "selected_trial": choice,
                            "missing_feature_cells": missing,
                            "feature_cells": cells,
                            "prediction_availability": 1.0,
                            "importance": restored.importance(),
                        }
                    )
                    if time.perf_counter() > deadline:
                        raise DataError("Regional run budget exceeded")
        write_json(
            stage / "descriptor.json",
            {
                "specification": spec,
                "experiment_id": experiment_id,
                "split": split.to_dict(),
                "records": records,
            },
        )
        write_json(
            stage / "runtime.json",
            {
                "seconds": time.perf_counter() - began,
                "environment": collect_environment_info().to_dict(),
            },
        )
        result = _publish_governed(
            stage,
            root,
            "regional-run",
            parents=parents,
            broker_context=broker_context,
            broker_parents=broker_parents,
        )
    if isinstance(result, PendingPublication):
        return result
    if isinstance(result, DatasetSnapshot) and broker_context is not None:
        return result
    verify_regional_run(
        result.directory,
        canonical=canonical,
        plan=plan,
        graph=geographic_graph,
        raw_snapshot_dir=raw_snapshot_dir,
    )
    return result.directory


def verify_regional_run(
    directory: Path | DatasetSnapshot,
    *,
    canonical: Path | DatasetSnapshot | None = None,
    plan: Path | DatasetSnapshot | None = None,
    graph: Path | DatasetSnapshot | None = None,
    raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
    geography_snapshot: Path | DatasetSnapshot | None = None,
    geography_artifact_reference: BrokerArtifactRef | None = None,
    geography_relative_path: str | None = None,
    geometry_snapshot: Path | DatasetSnapshot | None = None,
    geometry_artifact_reference: BrokerArtifactRef | None = None,
    geometry_relative_path: str | None = None,
) -> dict[str, object]:
    if canonical is None or plan is None or graph is None or raw_snapshot_dir is None:
        raise DataError("Complete current regional source context required for saved run")
    directory_path = snapshot_path(directory)
    parents = (snapshot_path(canonical), snapshot_path(plan), snapshot_path(graph))
    if broker_context is not None:
        if (
            not isinstance(directory, DatasetSnapshot)
            or directory.broker_reference is None
            or (geometry_snapshot is None) != (geometry_artifact_reference is None)
            or (geometry_snapshot is None) != (geometry_relative_path is None)
        ):
            raise DataError("Committed regional run and source references required")
        if (
            geography_snapshot is None
            or geography_artifact_reference is None
            or not geography_relative_path
        ):
            raise DataError(
                "Committed geographic source context required for regional verification"
            )
        geographic_parents: tuple[tuple[Path | DatasetSnapshot, BrokerArtifactRef], ...] = (
            (geography_snapshot, geography_artifact_reference),
        )
        if geometry_snapshot is not None and geometry_artifact_reference is not None:
            geographic_parents += ((geometry_snapshot, geometry_artifact_reference),)
        trusted = authorize_graph_inputs(
            canonical,
            plan,
            raw_snapshot_dir=raw_snapshot_dir,
            broker_context=broker_context,
            purpose="saved_run_verification",
            operation="energy_regional_verify",
            graph=graph,
            geographic_parents=geographic_parents,
        )
        if trusted is None:
            raise DataError("Regional verification broker authority unavailable")
        run_ref = resolve_broker_reference(
            broker_context.broker,
            directory.broker_reference.artifact_id,
            purpose="saved_run_verification",
            operation="energy_regional_verify",
        )
        if run_ref != directory.broker_reference or run_ref.artifact_type != "regional-run":
            raise DataError("Regional run broker identity changed")
        expected_run_parents = (trusted[0], trusted[1], trusted[-1])
        if set(
            broker_parent_references(
                broker_context.broker,
                run_ref,
                purpose="saved_run_verification",
                operation="energy_regional_verify",
            )
        ) != set(expected_run_parents):
            raise DataError("Regional run broker parents differ from verified inputs")
        if inspect_tree(directory_path / "payload").digest != run_ref.content_sha256:
            raise DataError("Regional run cache differs from committed broker content")
        lineage = resolve_parents(
            (raw_snapshot_dir, *(snapshot_path(item[0]) for item in geographic_parents))
        )
    else:
        lineage = resolve_parents(parents)
        if any(
            source not in lineage.sources for source in resolve_parents((raw_snapshot_dir,)).sources
        ):
            raise DataError("Regional saved run raw source closure mismatch")
    authorize_sources(
        lineage.sources,
        purpose="saved_run_verification",
        operation="energy_regional_verify",
        destination=directory_path,
    )
    desc = read_json(directory_path / "payload/descriptor.json")
    spec = object_dict(desc["specification"])
    if spec.get("schema_version") != "energy-regional-experiment-1.0" or identity(spec) != desc.get(
        "experiment_id"
    ):
        raise DataError("Regional experiment identity mismatch")
    records = desc.get("records")
    if not isinstance(records, list) or not records:
        raise DataError("Missing regional records")
    expected = {"descriptor.json", "runtime.json"}
    for item in records:
        r = object_dict(item)
        prefix = str(r["prefix"])
        if not all(c.isalnum() or c == "-" for c in prefix):
            raise DataError("Unsafe model filename")
        expected.update(
            {
                f"{prefix}-trial-0.txt",
                f"{prefix}-trial-1.txt",
                f"{prefix}-probe.parquet",
                f"{prefix}-predictions.parquet",
                f"{r['fold']}-scaler.json",
            }
        )
    if broker_context is None:
        base = governed_payload(directory_path, expected, parents=parents, kind="regional-run")
    else:
        base = directory_path / "payload"
        actual_files = set(discover(base).files)
        if actual_files != expected | {"governance.json"}:
            raise DataError("Committed regional-run file inventory differs")
    populations: dict[tuple[str, str], str] = {}
    for item in records:
        r = object_dict(item)
        prefix = str(r["prefix"])
        target = str(r["target"])
        feature = object_dict(r["feature_spec"])
        inputs = feature["inputs"]
        latencies = feature["latencies"]
        if not isinstance(inputs, list) or not isinstance(latencies, list):
            raise DataError("Invalid regional feature record")
        f = RegionalFeatures(
            target,
            tuple(str(v) for v in inputs),
            tuple(int(str(v)) for v in latencies),
            str(feature["mode"]),
        )
        names = tuple(str(v) for v in r["names"]) if isinstance(r["names"], list) else ()
        groups = tuple(str(v) for v in r["groups"]) if isinstance(r["groups"], list) else ()
        trials = r["trials"]
        choice = r["selected_trial"]
        if not isinstance(trials, list) or len(trials) != 2 or type(choice) is not int:
            raise DataError("Invalid model selection record")
        if choice != int(
            np.argmin([float(str(object_dict(t)["validation_mae_normalized"])) for t in trials])
        ):
            raise DataError("Validation selection mismatch")
        probe = pd.read_parquet(base / f"{prefix}-probe.parquet")
        for i, item_trial in enumerate(trials):
            trial = object_dict(item_trial)
            if trial["model_file"] != f"{prefix}-trial-{i}.txt":
                raise DataError("Model path mismatch")
            model = load_model(
                base / str(trial["model_file"]),
                config_from_dict(trial["config"]),
                f,
                names,
                groups,
                (target,),
                24,
                int(str(object_dict(trial["metadata"])["best_iteration"])),
            )
            if i == choice:
                np.testing.assert_array_equal(
                    model.predict(probe[list(names)].to_numpy(dtype=np.float32), names, (target,)),
                    probe.prediction.to_numpy(),
                )
        scaler = FittedStandardizer.from_dict(read_json(base / f"{r['fold']}-scaler.json"))
        if scaler.scales[scaler.sensor_ids.index(target)] != r["training_scale_MW"]:
            raise DataError("Training scale mismatch")
        table = pd.read_parquet(base / f"{prefix}-predictions.parquet")
        o = table.origin_row.to_numpy(dtype=np.int64).reshape(-1, 24)
        t = table.target_row.to_numpy(dtype=np.int64).reshape(-1, 24)
        if not np.all(o == o[:, :1]) or not np.array_equal(t, o[:, :1] + np.arange(1, 25)):
            raise DataError("Regional forecast offsets mismatch")
        if identity(table_metrics(table, "lightgbm")) != identity(r["metrics"]):
            raise DataError("Regional scores mismatch")
        population = identity(
            {
                "plan": spec["plan_id"],
                "fold": r["fold"],
                "target": target,
                "origins": o[:, 0].tolist(),
            }
        )
        if population != r["population_id"]:
            raise DataError("Regional target population mismatch")
        key = (str(r["fold"]), target)
        if key in populations and populations[key] != population:
            raise DataError("Unmatched ablation populations")
        populations[key] = population
    return {
        "valid": True,
        "snapshot_id": directory_path.name,
        "experiment_id": desc["experiment_id"],
        "selected_models": len(records),
    }
