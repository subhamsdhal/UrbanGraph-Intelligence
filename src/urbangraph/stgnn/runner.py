"""Independent graph fold fits with validation-only selection and full test populations."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
import torch_geometric  # type: ignore[import-untyped]

from urbangraph.baselines.scoring import GroupedMetrics
from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.ingestion import _writer_lock
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.datasets.serialization import object_dict, read_json, write_json
from urbangraph.evaluation.artifacts import code_provenance, load_plan
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.graphs.artifacts import load_graph
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.evaluation import evaluate_fold
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.trainer import TrainingResult, fit_model
from urbangraph.reproducibility import collect_environment_info
from urbangraph.stgnn.artifacts import publish_run, verify_run
from urbangraph.stgnn.contracts import model_config, training_config
from urbangraph.stgnn.models import GraphForecaster


def run_experiment(configuration: dict[str, object], *, dry_run: bool = False) -> dict[str, object]:
    required = {
        "canonical_dir",
        "raw_snapshot_dir",
        "plan_dir",
        "graph_dir",
        "locations",
        "location_reference",
        "distances",
        "distance_reference",
        "output_root",
        "model",
        "training",
        "search_residual",
    }
    if set(configuration) != required or type(configuration["search_residual"]) is not bool:
        raise DataError("Graph research configuration fields mismatch")
    for key in (
        "canonical_dir",
        "raw_snapshot_dir",
        "plan_dir",
        "graph_dir",
        "locations",
        "location_reference",
        "distances",
        "distance_reference",
        "output_root",
    ):
        if not isinstance(configuration[key], str):
            raise DataError(f"Expected path string {key}")
    architecture = model_config(configuration["model"])
    config = training_config(configuration["training"])
    raw_source = str(configuration["raw_snapshot_dir"])
    output_root = str(configuration["output_root"])
    require_verified_snapshot_authorization(
        raw_source, "model_training", output_root, operation="model_train"
    )
    if not dry_run:
        require_verified_snapshot_authorization(
            raw_source, "scientific_evaluation", output_root, operation="model_evaluate"
        )
    graph_context = {
        key: str(configuration[key])
        for key in (
            "graph_dir",
            "raw_snapshot_dir",
            "canonical_dir",
            "locations",
            "location_reference",
            "distances",
            "distance_reference",
        )
    }
    started = time.perf_counter()
    deadline = started + config.run_seconds
    dataset, split, plan = load_plan(
        str(configuration["plan_dir"]),
        canonical_dir=str(configuration["canonical_dir"]),
        raw_snapshot_dir=str(configuration["raw_snapshot_dir"]),
    )
    graph = load_graph(
        str(configuration["graph_dir"]),
        raw_snapshot_dir=str(configuration["raw_snapshot_dir"]),
        canonical_dir=str(configuration["canonical_dir"]),
        locations=str(configuration["locations"]),
        location_reference=str(configuration["location_reference"]),
        distances=str(configuration["distances"]),
        distance_reference=str(configuration["distance_reference"]),
    )
    if (
        graph.index.sensor_ids != dataset.sensor_ids
        or graph.descriptor["canonical"] != dataset.lineage
    ):
        raise DataError("Graph canonical data/sensor lineage mismatch")
    root = safe_absolute(str(configuration["output_root"]))
    protected = [
        safe_absolute(str(configuration[k])) for k in ("canonical_dir", "plan_dir", "graph_dir")
    ]
    protected += [
        protected[0].parents[2],
        protected[1].parents[2],
        protected[0].parents[2].parent / "raw",
    ]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise DataError("Graph research output overlaps immutable inputs")
    lineage: dict[str, object] = {
        "canonical": dataset.lineage,
        "plan_id": identity(plan),
        "code": code_provenance(),
        "torch": str(torch.__version__),
        "pyg": str(torch_geometric.__version__),
        "graph_id": graph.graph_id,
        "sensor_axis_sha256": graph.index.fingerprint,
    }
    experiment_id = identity(
        lineage
        | {
            "model": architecture.to_dict(),
            "training": config.to_dict(),
            "search_residual": configuration["search_residual"],
        }
    )
    preview = {
        "experiment_id": experiment_id,
        "graph_id": graph.graph_id,
        "model": architecture.to_dict(),
        "training": config.to_dict(),
        "folds": len(split.folds),
        "protocol": split.config.protocol.name,
    }
    if dry_run:
        return preview | {"dry_run": True}
    runs = root / "runs"
    if runs.exists():
        for directory in sorted(runs.iterdir()):
            if (
                directory.is_dir()
                and (directory / "run.json").exists()
                and read_json(directory / "run.json").get("experiment_id") == experiment_id
            ):
                verify_run(directory, graph_context=graph_context)
                raise DataError(
                    "Matching saved STGNN run is integrity-checked but has no "
                    "independently attested execution authorship; completed-run reuse denied"
                )
    checkpoints = root / "checkpoints" / experiment_id
    checkpoints.mkdir(parents=True, exist_ok=True)
    with _writer_lock(checkpoints):
        models: dict[str, GraphForecaster] = {}
        scalers: dict[str, object] = {}
        fitted_scalers: dict[str, FittedStandardizer] = {}
        selections: dict[str, object] = {}
        runtimes: dict[str, object] = {}
        for i, fold in enumerate(split.folds):
            scaler = fit_preprocessing(dataset, split, i, config.scaling)
            train = NeuralWindows(
                dataset,
                split,
                i,
                "train",
                scaler,
                calendar=config.calendar,
                maximum=config.max_train_windows,
                seed=config.seed,
            )
            validation = NeuralWindows(
                dataset,
                split,
                i,
                "validation",
                scaler,
                calendar=config.calendar,
                maximum=config.max_validation_windows,
                seed=config.seed + 1,
            )
            candidates = (
                (False, True) if configuration["search_residual"] else (architecture.residual,)
            )
            best: TrainingResult[GraphForecaster] | None = None
            best_mae = float("inf")
            trials: list[dict[str, object]] = []
            timing: list[dict[str, object]] = []
            for residual in candidates:
                candidate = replace(architecture, residual=residual)
                trial_root = checkpoints / fold.fold_id / ("residual" if residual else "direct")
                trial_root.mkdir(parents=True, exist_ok=True)
                epochs = sorted((trial_root / "epochs").glob("epoch-*"))
                completed = trial_root / "completed.json"

                factory = partial(
                    GraphForecaster,
                    candidate,
                    graph,
                    train.history,
                    train.horizon,
                    train.sensor_ids,
                    4 if config.calendar else 2,
                )

                try:
                    if config.device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    result = fit_model(
                        train,
                        validation,
                        config,
                        trial_root / "epochs",
                        lineage | {"fold": fold.fold_id, "model": candidate.to_dict()},
                        model_factory=factory,
                        resume=epochs[-1] if epochs else None,
                        deadline=deadline,
                    )
                    recorded = dict(result.report)
                    recorded["peak_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated() if config.device == "cuda" else None
                    )
                    if completed.exists():
                        saved = read_json(completed)
                        if saved.get("checkpoint_id") != recorded["checkpoint_id"]:
                            raise DataError("Completed graph trial checkpoint changed")
                        recorded = saved
                    else:
                        write_json(completed, recorded)
                    score = recorded["best_validation_mae"]
                    if not isinstance(score, float):
                        raise DataError("Missing validation score")
                    history = recorded.pop("history")
                    if not isinstance(history, list) or not history:
                        raise DataError("Missing completed validation history")
                    rows = [dict(object_dict(row)) for row in history]
                    seconds = [row.pop("seconds") for row in rows]
                    timing.append(
                        {
                            "model": candidate.to_dict(),
                            "training_seconds": recorded.pop("training_seconds"),
                            "epoch_seconds": seconds,
                            "checkpoint_id": recorded.pop("checkpoint_id"),
                            "peak_allocated_bytes": recorded.pop("peak_allocated_bytes"),
                            "resumed": recorded.pop("resumed"),
                        }
                    )
                    trials.append({"model": candidate.to_dict(), "history": rows, **recorded})
                    if score < best_mae:
                        best, best_mae = result, score
                except (DataError, RuntimeError, OSError) as exc:
                    write_json(
                        trial_root / f"failure-{time.time_ns()}.json",
                        {"status": "INCOMPLETE", "error": str(exc), "model": candidate.to_dict()},
                    )
                    raise
            if best is None:
                raise DataError("No completed graph candidate")
            models[fold.fold_id] = best.model.cpu()
            scalers[fold.fold_id] = scaler.to_dict()
            fitted_scalers[fold.fold_id] = scaler
            selections[fold.fold_id] = {
                "trials": trials,
                "selected_config": best.model.config.to_dict(),
                "policy": "minimum validation MAE; candidate order breaks ties",
                "training_population": train.report,
                "validation_population": validation.report,
                "scaler": scaler.to_dict(),
            }
            runtimes[fold.fold_id] = timing
        metrics = GroupedMetrics()
        population = hashlib.sha256()
        scoring: dict[str, object] = {}
        for i, fold in enumerate(split.folds):
            scaler = fitted_scalers[fold.fold_id]
            windows = NeuralWindows(dataset, split, i, "test", scaler, calendar=config.calendar)
            before = time.perf_counter()
            model = models[fold.fold_id].to(config.device)
            digest = evaluate_fold(
                model,
                windows,
                config,
                metrics,
                deadline=deadline,
                population_update=population.update,
            )
            scoring[fold.fold_id] = {
                "seconds": time.perf_counter() - before,
                "windows": len(windows),
                "population_sha256": digest,
            }
            model.cpu()
        descriptor: dict[str, object] = {
            "schema_version": "stgnn-1.0",
            "experiment_id": experiment_id,
            "family": architecture.family,
            "role": "selected",
            "model": architecture.to_dict(),
            "training": config.to_dict(),
            "search_residual": configuration["search_residual"],
            **lineage,
            "protocol": split.config.protocol.name,
            "population_sha256": population.hexdigest(),
            "batch_size": 256,
            "metric_policy": {
                "mask_zero": split.config.validity.mask_zero,
                "mape": "percent; zero denominators excluded",
                "units": dataset.lineage["units"],
                "aggregation": "micro forecast cells",
            },
            "temporal": temporal_diagnostics(dataset.timestamps, split),
            "prediction_availability": "all eligible cells finite or experiment aborts",
        }
        directory = publish_run(
            models,
            scalers,
            graph,
            {
                "selection": selections,
                "metrics": {"groups": metrics.result()},
                "runtime": {
                    "training": runtimes,
                    "scoring": scoring,
                    "environment": collect_environment_info().to_dict(),
                    "total_seconds_before_publication": time.perf_counter() - started,
                },
            },
            descriptor,
            runs,
            raw_snapshot_dir=Path(str(configuration["raw_snapshot_dir"])),
            canonical_dir=Path(str(configuration["canonical_dir"])),
            locations=Path(str(configuration["locations"])),
            location_reference=str(configuration["location_reference"]),
            distances=Path(str(configuration["distances"])),
            distance_reference=str(configuration["distance_reference"]),
        )
        return verify_run(directory, graph_context=graph_context) | {
            "directory": str(directory),
            "reused": False,
        }
