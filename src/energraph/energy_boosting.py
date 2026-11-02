"""Bounded energy experiments using the existing statistical and native boosting engines."""

from __future__ import annotations

import tempfile
import time
from dataclasses import replace
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
from energraph.energy_features import EnergyFeatures, build_features
from energraph.experiments import MODELS, code_fingerprint, origins, predict, scores, truth
from energraph.lineage import authorize_sources, resolve_parents
from energraph.target_plans import read_target_plan
from urbangraph.baselines.contracts import ModelConfig
from urbangraph.baselines.history import training_data
from urbangraph.baselines.models import Baseline, FittedState, fit_model
from urbangraph.boosting.contracts import BoostConfig, config_from_dict
from urbangraph.boosting.models import load_model, train_model
from urbangraph.data import DataError, DatasetSnapshot
from urbangraph.data.filesystem import discover
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.dataset import identity
from urbangraph.reproducibility import collect_environment_info


def feature_from_dict(value: object) -> EnergyFeatures:
    d = object_dict(value)
    expected = set(EnergyFeatures().to_dict())
    if set(d) != expected:
        raise DataError("Feature configuration fields mismatch")

    def integers(key: str) -> tuple[int, ...]:
        values = d[key]
        if not isinstance(values, list | tuple) or any(type(v) is not int for v in values):
            raise DataError("Integer feature sequence required")
        return tuple(int(str(v)) for v in values)

    names = d["inputs"]
    if not isinstance(names, list | tuple) or any(not isinstance(v, str) for v in names):
        raise DataError("Variable sequence required")
    if type(d["calendar"]) is not bool:
        raise DataError("Boolean calendar policy required")
    return EnergyFeatures(
        tuple(str(v) for v in names),
        integers("latencies"),
        integers("lags"),
        integers("rolling"),
        d["calendar"],
        str(d["weather_mode"]),
        str(d["version"]),
    )


def sampled_rows(
    population: NDArray[np.int64], cap: int, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Uniform deterministic origin sampling retains all 24 horizons per selected origin.
    count = min(len(population), cap // 24)
    if count < 1:
        raise DataError("Budget must support at least one complete 24-horizon origin")
    idx = np.sort(np.random.default_rng(seed).choice(len(population), count, replace=False))
    return np.repeat(population[idx], 24), np.tile(np.arange(1, 25, dtype=np.int64), count)


def error_summary(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> dict[str, object]:
    from urbangraph.evaluation.metrics import metric

    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise DataError("No silent population reduction for missing predictions")
    denominator = float(np.abs(actual).sum())
    return {
        "count": actual.size,
        "mae_MW": metric("mae", actual, predicted).value,
        "rmse_MW": metric("rmse", actual, predicted).value,
        "mape_percent": metric("mape", actual, predicted).value if np.any(actual != 0) else None,
        "mape_excluded_zeros": int((actual == 0).sum()),
        "wape_percent": float(np.abs(actual - predicted).sum() / denominator * 100)
        if denominator
        else None,
    }


def table_metrics(table: pd.DataFrame, model: str) -> dict[str, object]:
    a = table.actual_MW.to_numpy(dtype=np.float64)
    p = table[model].to_numpy(dtype=np.float64)
    horizons = table.target_row.to_numpy() - table.origin_row.to_numpy()
    result: dict[str, object] = {
        "overall": error_summary(a, p),
        "by_horizon": {
            str(h): error_summary(a[horizons == h], p[horizons == h]) for h in range(1, 25)
        },
    }
    months = pd.DatetimeIndex(table.target_utc).month.to_numpy()
    result["seasonal"] = {
        name: error_summary(a[mask], p[mask])
        for name, mask in (
            ("DJF", np.isin(months, [12, 1, 2])),
            ("MAM", np.isin(months, [3, 4, 5])),
            ("JJA", np.isin(months, [6, 7, 8])),
            ("SON", np.isin(months, [9, 10, 11])),
        )
        if mask.any()
    }
    active = a > 0
    result["retrospective_positive_target_subset"] = (
        error_summary(a[active], p[active]) if active.any() else None
    )
    return result


def _boost_parent_context(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    raw_snapshot_dir: Path,
    broker_context: BrokerPublicationContext | None,
    *,
    purpose: str,
    operation: str,
) -> tuple[
    Path, Path, tuple[BrokerArtifactRef, BrokerArtifactRef] | None, tuple[dict[str, object], ...]
]:
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    plan_path = plan.directory if isinstance(plan, DatasetSnapshot) else plan
    if broker_context is None:
        graph = resolve_parents((canonical_path, plan_path))
        if graph.sources != resolve_parents((raw_snapshot_dir,)).sources:
            raise DataError("Energy training source closure mismatch")
        return canonical_path, plan_path, None, graph.sources
    if (
        not isinstance(canonical, DatasetSnapshot)
        or not isinstance(plan, DatasetSnapshot)
        or canonical.broker_reference is None
        or plan.broker_reference is None
        or canonical.broker_reference.artifact_type != "multivariable"
        or plan.broker_reference.artifact_type != "target-plan"
    ):
        raise DataError("Committed multivariable and target-plan identities are required")
    refs = (canonical.broker_reference, plan.broker_reference)
    broker = broker_context.broker
    resolved = tuple(
        resolve_broker_reference(broker, ref.artifact_id, purpose=purpose, operation=operation)
        for ref in refs
    )
    if resolved != refs:
        raise DataError("Energy broker parent identity changed")
    canonical_parents = broker_parent_references(
        broker, refs[0], purpose=purpose, operation=operation
    )
    plan_parents = broker_parent_references(broker, refs[1], purpose=purpose, operation=operation)
    if (
        len(canonical_parents) != 1
        or canonical_parents[0].artifact_type != "raw"
        or plan_parents != (refs[0],)
    ):
        raise DataError("Energy broker parent closure is invalid")
    if inspect_tree(raw_snapshot_dir).digest != canonical_parents[0].content_sha256:
        raise DataError("Energy raw cache differs from broker source")
    if inspect_tree(canonical_path / "payload").digest != refs[0].content_sha256:
        raise DataError("Energy canonical cache differs from broker artifact")
    if inspect_tree(plan_path / "payload").digest != refs[1].content_sha256:
        raise DataError("Energy target-plan cache differs from broker artifact")
    return canonical_path, plan_path, refs, resolve_parents((raw_snapshot_dir,)).sources


def run_energy(
    canonical: Path | DatasetSnapshot,
    plan: Path | DatasetSnapshot,
    root: Path,
    features: EnergyFeatures,
    config: BoostConfig,
    *,
    weather: Path | None = None,
    raw_snapshot_dir: Path | None = None,
    weather_raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> Path | DatasetSnapshot | PendingPublication:
    if config.family != "lightgbm":
        raise DataError("EG-001B requires native LightGBM")
    if raw_snapshot_dir is None or (weather is not None and weather_raw_snapshot_dir is None):
        raise DataError("Complete current source context required for energy training")
    if broker_context is not None and weather is not None:
        raise DataError("Broker-backed weather run is not supported by this contract")
    canonical_path, plan_path, broker_parents, sources = _boost_parent_context(
        canonical,
        plan,
        raw_snapshot_dir,
        broker_context,
        purpose="model_training",
        operation="energy_boost_train",
    )
    authorize_sources(
        sources,
        purpose="model_training",
        operation="energy_boost_train",
        destination=root,
    )
    data, frame, split = read_target_plan(
        canonical,
        plan,
        weather,
        raw_snapshot_dir=raw_snapshot_dir,
        weather_raw_snapshot_dir=weather_raw_snapshot_dir,
        broker_context=broker_context,
    )
    target = str(data.lineage["target"])
    if target not in features.inputs:
        raise DataError("Target history must be explicitly included")
    target_latency = features.latencies[features.inputs.index(target)]
    if target_latency != split.config.features.observed_delay_steps:
        raise DataError("Target feature latency must match the evaluation plan")
    specification = {
        "schema_version": "energy-boost-experiment-1.0",
        "lineage": data.lineage,
        "plan_id": plan_path.name,
        "split_id": identity(split.to_dict()),
        "features": features.to_dict(),
        "boost_config": config.to_dict(),
        "code_sha256": code_fingerprint(),
        "dependencies": {n: version(n) for n in ("numpy", "pandas", "pyarrow", "lightgbm")},
        "candidate_depths": [config.max_depth, config.alternative_depth],
        "ridge_candidates": [24, 168],
        "ridge_penalty": 10000.0,
        "selection": "validation MAE only, first minimum wins; calibration reserved",
        "sampling": "seeded uniform training/validation origins, all horizons retained",
        "weather_limitation": "W1 historical reanalysis hindsight"
        if features.weather_mode == "W1"
        else "W0 no weather",
        "preprocessing": "none; native missing branches with explicit missing indicators",
    }
    experiment_id = identity(specification)
    _, _, _, verification_sources = _boost_parent_context(
        canonical,
        plan,
        raw_snapshot_dir,
        broker_context,
        purpose="saved_run_verification",
        operation="energy_boost_verify",
    )
    authorize_sources(
        verification_sources,
        purpose="saved_run_verification",
        operation="energy_boost_verify",
        destination=root,
    )
    existing = (
        []
        if broker_context is not None
        else [
            p.parent
            for p in (root / "energraph-energy-boost").glob("*/manifest.json")
            if read_json(p.parent / "payload/descriptor.json").get("experiment_id") == experiment_id
        ]
    )
    if existing:
        if len(existing) != 1:
            raise DataError("Conflicting experiment identities")
        verify_energy_run(
            existing[0],
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw_snapshot_dir,
            weather=weather,
            weather_raw_snapshot_dir=weather_raw_snapshot_dir,
            broker_context=broker_context,
        )
        return existing[0]
    started_run = time.perf_counter()
    deadline = started_run + config.run_seconds
    folds: list[dict[str, object]] = []
    runtime: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="energy-boost-") as tmp:
        stage = Path(tmp)
        for f, fold in enumerate(split.folds):
            train_origins = origins(split, f, "train")
            val_origins = origins(split, f, "validation")
            test_origins = origins(split, f, "test")
            train_o, train_h = sampled_rows(train_origins, config.max_train_rows, config.seed)
            val_o, val_h = sampled_rows(val_origins, config.max_validation_rows, config.seed)
            x, names, groups = build_features(
                frame, train_o, train_h, features, lower=fold.partition("train").start
            )
            vx, vn, vg = build_features(frame, val_o, val_h, features)
            if vn != names or vg != groups:
                raise DataError("Feature schema drift")
            values = frame[target].to_numpy(dtype=np.float64)
            y, vy = values[train_o + train_h], values[val_o + val_h]
            candidates = []
            trials: list[dict[str, object]] = []
            for i, depth in enumerate((config.max_depth, config.alternative_depth)):
                trial_config = replace(
                    config, max_depth=depth, num_leaves=min(config.num_leaves, 2**depth)
                )
                model, metadata = train_model(
                    trial_config, features, names, groups, (target,), 24, x, y, vx, vy, deadline
                )
                score = float(np.abs(model.predict(vx, names, (target,)) - vy).mean())
                filename = f"{fold.fold_id}-trial-{i}.txt"
                model.save(stage / filename)
                trials.append(
                    {
                        "config": trial_config.to_dict(),
                        "validation_mae": score,
                        "model_file": filename,
                        "metadata": metadata,
                    }
                )
                candidates.append(model)
            chosen = int(np.argmin([float(str(t["validation_mae"])) for t in trials]))
            selected = candidates[chosen]
            restored = load_model(
                stage / str(trials[chosen]["model_file"]),
                selected.config,
                features,
                names,
                groups,
                (target,),
                24,
                selected.best_iteration,
            )
            np.testing.assert_array_equal(
                selected.predict(vx, names, (target,)), restored.predict(vx, names, (target,))
            )
            # Store deterministic replay probe for verification without loading large sources.
            probe = pd.DataFrame(vx[: min(64, len(vx))], columns=list(names))
            probe["prediction"] = restored.predict(vx[: len(probe)], names, (target,))
            probe.to_parquet(stage / f"{fold.fold_id}-probe.parquet", index=False)
            trained = training_data(data, split, f)
            ridge_models = []
            ridge_trials = []
            validation_actual = truth(data, split, f, "validation")
            for lag in (24, 168):
                began = time.perf_counter()
                ar = fit_model(
                    ModelConfig("autoregressive", history_fallback=168, lags=lag, ridge=10000.0),
                    trained,
                    24,
                    interval_seconds=3600,
                )
                score = float(
                    np.abs(predict(ar, data, split, f, "validation") - validation_actual).mean()
                )
                ridge_trials.append(
                    {
                        "lags": lag,
                        "validation_mae": score,
                        "fit_seconds": time.perf_counter() - began,
                    }
                )
                ridge_models.append(ar)
            ar_choice = int(np.argmin([t["validation_mae"] for t in ridge_trials]))
            ar = ridge_models[ar_choice]
            if ar.state is None:
                raise DataError("Missing ridge state")
            write_json(stage / f"{fold.fold_id}-ridge.json", ar.state.to_dict())
            actual = truth(data, split, f, "test")
            pos = test_origins[:, None] + np.arange(1, 25)
            table = pd.DataFrame(
                {
                    "origin_row": np.repeat(test_origins, 24),
                    "target_row": pos.ravel(),
                    "target_utc": data.timestamps[pos.ravel()],
                    "actual_MW": actual.ravel(),
                }
            )
            for name in MODELS:
                baseline = (
                    ar
                    if name == "autoregressive"
                    else Baseline(
                        ModelConfig(name, history_fallback=168, fallback="unavailable"),
                        interval_seconds=3600,
                    )
                )
                pred = predict(baseline, data, split, f, "test")
                scores(actual, pred, data, split)  # Existing metric/availability validation.
                table[name] = pred.ravel()
            predictions = []
            missing_cells = total_cells = 0
            for begin in range(0, len(test_origins), config.evaluation_batch):
                batch = test_origins[begin : begin + config.evaluation_batch]
                tx, tn, _ = build_features(
                    frame,
                    np.repeat(batch, 24),
                    np.tile(np.arange(1, 25, dtype=np.int64), len(batch)),
                    features,
                )
                missing_cells += int(np.isnan(tx).sum())
                total_cells += tx.size
                predictions.append(restored.predict(tx, tn, (target,)))
            table["lightgbm"] = np.concatenate(predictions)
            table.to_parquet(stage / f"{fold.fold_id}-predictions.parquet", index=False)
            population_id = identity(
                {
                    "plan": plan_path.name,
                    "fold": fold.fold_id,
                    "origins": test_origins.tolist(),
                    "target": target,
                }
            )
            folds.append(
                {
                    "fold": fold.fold_id,
                    "population_id": population_id,
                    "forecast_origins": len(test_origins),
                    "valid_target_cells": len(table),
                    "train_rows": len(x),
                    "validation_rows": len(vx),
                    "train_origin_id": identity(train_o.tolist()),
                    "validation_origin_id": identity(val_o.tolist()),
                    "training_lineage": trained.lineage,
                    "feature_names": list(names),
                    "feature_groups": list(groups),
                    "trials": trials,
                    "selected_trial": chosen,
                    "ridge_trials": ridge_trials,
                    "selected_ridge": ar_choice,
                    "importance": restored.importance(),
                    "input_missing_cells": missing_cells,
                    "input_feature_cells": total_cells,
                    "missing_feature_policy": "native missing branches, no imputation",
                    "prediction_availability": 1.0,
                    "historical_fallback_count": 0,
                    "metrics": {n: table_metrics(table, n) for n in (*MODELS, "lightgbm")},
                }
            )
            runtime.append(
                {"fold": fold.fold_id, "elapsed_run_seconds": time.perf_counter() - started_run}
            )
            if time.perf_counter() > deadline:
                raise DataError(
                    "Energy experiment run budget exceeded; no incomplete bundle published"
                )
        write_json(
            stage / "descriptor.json",
            {
                "specification": specification,
                "experiment_id": experiment_id,
                "split": split.to_dict(),
                "folds": folds,
            },
        )
        write_json(
            stage / "runtime.json",
            {
                "fold_timings": runtime,
                "run_seconds": time.perf_counter() - started_run,
                "environment": collect_environment_info().to_dict(),
            },
        )
        result = _publish_governed(
            stage,
            root,
            "energy-boost",
            parents=(canonical_path, plan_path)
            if weather is None
            else (canonical_path, plan_path, weather),
            broker_context=broker_context if broker_parents is not None else None,
            broker_parents=broker_parents,
        )
    if isinstance(result, PendingPublication):
        return result
    if broker_context is not None:
        return result
    verify_energy_run(
        result.directory,
        canonical=canonical,
        plan=plan,
        raw_snapshot_dir=raw_snapshot_dir,
        weather=weather,
        weather_raw_snapshot_dir=weather_raw_snapshot_dir,
    )
    return result.directory


def verify_energy_run(
    directory: Path | DatasetSnapshot,
    *,
    canonical: Path | DatasetSnapshot | None = None,
    plan: Path | DatasetSnapshot | None = None,
    raw_snapshot_dir: Path | None = None,
    weather: Path | None = None,
    weather_raw_snapshot_dir: Path | None = None,
    broker_context: BrokerPublicationContext | None = None,
) -> dict[str, object]:
    if (
        canonical is None
        or plan is None
        or raw_snapshot_dir is None
        or (weather is not None and weather_raw_snapshot_dir is None)
    ):
        raise DataError("Complete current source context required for energy run verification")
    directory_path = directory.directory if isinstance(directory, DatasetSnapshot) else directory
    canonical_path = canonical.directory if isinstance(canonical, DatasetSnapshot) else canonical
    plan_path = plan.directory if isinstance(plan, DatasetSnapshot) else plan
    parents = (
        (canonical_path, plan_path) if weather is None else (canonical_path, plan_path, weather)
    )
    if broker_context is not None:
        if (
            weather is not None
            or not isinstance(directory, DatasetSnapshot)
            or directory.broker_reference is None
        ):
            raise DataError("Committed broker run and single-source parents are required")
        _, _, parent_refs, sources = _boost_parent_context(
            canonical,
            plan,
            raw_snapshot_dir,
            broker_context,
            purpose="saved_run_verification",
            operation="energy_boost_verify",
        )
        if parent_refs is None:
            raise DataError("Energy broker parent identities unavailable")
        run_ref = resolve_broker_reference(
            broker_context.broker,
            directory.broker_reference.artifact_id,
            purpose="saved_run_verification",
            operation="energy_boost_verify",
        )
        if run_ref != directory.broker_reference or run_ref.artifact_type != "energy-boost":
            raise DataError("Energy run broker identity changed")
        if set(
            broker_parent_references(
                broker_context.broker,
                run_ref,
                purpose="saved_run_verification",
                operation="energy_boost_verify",
            )
        ) != set(parent_refs):
            raise DataError("Energy run broker parents differ from supplied committed inputs")
        if inspect_tree(directory_path / "payload").digest != run_ref.content_sha256:
            raise DataError("Energy run cache differs from committed broker content")
    else:
        graph = resolve_parents(parents)
        raw_set = (
            (raw_snapshot_dir,) if weather is None else (raw_snapshot_dir, weather_raw_snapshot_dir)
        )
        if (
            graph.sources
            != resolve_parents(tuple(source for source in raw_set if source is not None)).sources
        ):
            raise DataError("Energy run verification source closure mismatch")
        sources = graph.sources
    authorize_sources(
        sources,
        purpose="saved_run_verification",
        operation="energy_boost_verify",
        destination=directory_path,
    )
    desc = read_json(directory_path / "payload/descriptor.json")
    spec = object_dict(desc["specification"])
    if (
        spec.get("schema_version") != "energy-boost-experiment-1.0"
        or identity(spec) != desc["experiment_id"]
    ):
        raise DataError("Energy boosting experiment identity mismatch")
    features = feature_from_dict(spec["features"])
    config_from_dict(spec["boost_config"])
    target = str(object_dict(spec["lineage"])["target"])
    folds = desc["folds"]
    if not isinstance(folds, list) or not folds:
        raise DataError("Missing experiment folds")
    expected = {"descriptor.json", "runtime.json"}
    for item in folds:
        f = object_dict(item)
        name = str(f["fold"])
        if not name.startswith("fold-") or not name[5:].isdigit():
            raise DataError("Invalid fold path")
        expected.update(
            {
                f"{name}-predictions.parquet",
                f"{name}-probe.parquet",
                f"{name}-ridge.json",
                f"{name}-trial-0.txt",
                f"{name}-trial-1.txt",
            }
        )
    if broker_context is None:
        base = governed_payload(directory_path, expected, parents=parents, kind="energy-boost")
    else:
        base = directory_path / "payload"
        if set(discover(base).files) != expected | {"governance.json"}:
            raise DataError("Committed energy-run file inventory differs")
    for item in folds:
        f = object_dict(item)
        name = str(f["fold"])
        trials = f["trials"]
        if not isinstance(trials, list) or len(trials) != 2:
            raise DataError("Invalid trials")
        chosen = f["selected_trial"]
        if type(chosen) is not int or chosen != int(
            np.argmin([float(str(object_dict(t)["validation_mae"])) for t in trials])
        ):
            raise DataError("Selected model does not match validation minimum")
        names = (
            tuple(str(v) for v in f["feature_names"])
            if isinstance(f["feature_names"], list)
            else ()
        )
        groups = (
            tuple(str(v) for v in f["feature_groups"])
            if isinstance(f["feature_groups"], list)
            else ()
        )
        selected = object_dict(trials[chosen])
        if selected["model_file"] != f"{name}-trial-{chosen}.txt":
            raise DataError("Invalid selected model path")
        model = load_model(
            base / str(selected["model_file"]),
            config_from_dict(selected["config"]),
            features,
            names,
            groups,
            (target,),
            24,
            int(str(object_dict(selected["metadata"])["best_iteration"])),
        )
        probe = pd.read_parquet(base / f"{name}-probe.parquet")
        np.testing.assert_array_equal(
            model.predict(probe[list(names)].to_numpy(dtype=np.float32), names, (target,)),
            probe.prediction.to_numpy(dtype=np.float64),
        )
        FittedState.from_dict(read_json(base / f"{name}-ridge.json"))
        table = pd.read_parquet(base / f"{name}-predictions.parquet")
        o = table.origin_row.to_numpy(dtype=np.int64).reshape(-1, 24)
        t = table.target_row.to_numpy(dtype=np.int64).reshape(-1, 24)
        if not np.all(o == o[:, :1]) or not np.array_equal(t, o[:, :1] + np.arange(1, 25)):
            raise DataError("Forecast target offsets mismatch")
        if f["population_id"] != identity(
            {"plan": spec["plan_id"], "fold": name, "origins": o[:, 0].tolist(), "target": target}
        ):
            raise DataError("Population identity mismatch")
        metrics = {n: table_metrics(table, n) for n in (*MODELS, "lightgbm")}
        if identity(metrics) != identity(f["metrics"]) or len(table) != f["valid_target_cells"]:
            raise DataError("Measured metrics or target population mismatch")
    return {
        "valid": True,
        "snapshot_id": directory_path.name,
        "experiment_id": desc["experiment_id"],
        "folds": len(folds),
        "target": target,
        "weather_mode": features.weather_mode,
    }
