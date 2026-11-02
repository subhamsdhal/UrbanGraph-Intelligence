"""Independent fold training, frozen selection, then full-population held-out scoring."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from urbangraph.baselines.scoring import GroupedMetrics
from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.ingestion import _writer_lock
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import code_provenance, load_plan
from urbangraph.evaluation.dataset import identity
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.neural.artifacts import publish_run, verify_run
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.evaluation import evaluate_fold
from urbangraph.neural.tuning import select_model
from urbangraph.reproducibility import collect_environment_info


def run_experiment(
    canonical_dir: str | Path,
    plan_dir: str | Path,
    config: NeuralConfig,
    *,
    raw_snapshot_dir: str | Path,
    output_root: str | Path = "data/neural",
) -> Path:
    for purpose, operation in (
        ("model_training", "model_train"),
        ("scientific_evaluation", "model_evaluate"),
    ):
        require_verified_snapshot_authorization(
            raw_snapshot_dir, purpose, output_root, operation=operation
        )
    started = time.perf_counter()
    deadline = started + config.run_seconds
    dataset, split, plan = load_plan(
        plan_dir, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    root = safe_absolute(output_root)
    for protected in (
        safe_absolute(canonical_dir).parents[2],
        safe_absolute(plan_dir).parents[2],
        safe_absolute(canonical_dir).parents[2].parent / "raw",
    ):
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise DataError("Neural output overlaps managed data")
    lineage: dict[str, object] = {
        "canonical": dataset.lineage,
        "plan_id": identity(plan),
        "code": code_provenance(),
        "torch": str(torch.__version__),
    }
    experiment_id = identity(lineage | {"config": config.to_dict()})
    runs = root / "runs"
    # Saved bundles can be inspected, but none has independent runner custody.
    # A matching caller-controlled experiment ID cannot satisfy this execution API.
    if runs.exists():
        for path in sorted(runs.iterdir()):
            if path.is_dir() and (path / "run.json").exists():
                saved = read_json(path / "run.json")
                if saved.get("experiment_id") == experiment_id:
                    verify_run(path, raw_snapshot_dir=raw_snapshot_dir, canonical_dir=canonical_dir)
                    raise DataError(
                        "Matching saved neural run is integrity-checked but has no "
                        "independently attested execution authorship; completed-run reuse denied"
                    )
    checkpoints = root / "checkpoints" / experiment_id
    checkpoints.mkdir(parents=True, exist_ok=True)
    with _writer_lock(checkpoints):
        selected = {
            fold.fold_id: select_model(
                dataset,
                split,
                i,
                config,
                checkpoints / fold.fold_id,
                lineage | {"fold": fold.fold_id},
                deadline,
            )
            for i, fold in enumerate(split.folds)
        }
        metrics = GroupedMetrics()
        population = hashlib.sha256()
        scoring: dict[str, object] = {}
        for i, fold in enumerate(split.folds):
            selection = selected[fold.fold_id]
            windows = NeuralWindows(
                dataset, split, i, "test", selection.training.scaler, calendar=config.calendar
            )
            before = time.perf_counter()
            fold_hash = evaluate_fold(
                selection.fitted.model,
                windows,
                config,
                metrics,
                deadline=deadline,
                population_update=population.update,
            )
            scoring[fold.fold_id] = {
                "seconds": time.perf_counter() - before,
                "population_sha256": fold_hash,
                "windows": len(windows),
            }
        selections: dict[str, object] = {}
        timings: dict[str, object] = {}
        for fold_id, selection in selected.items():
            report = dict(selection.report)
            trial_records = report["trials"]
            assert isinstance(trial_records, list)
            clean = []
            times = []
            for record in trial_records:
                trial = dict(object_dict(record))
                history = trial.pop("history")
                assert isinstance(history, list)
                rows = [dict(object_dict(row)) for row in history]
                epoch_seconds = [row.pop("seconds") for row in rows]
                times.append(
                    {
                        "config": trial["config"],
                        "epoch_seconds": epoch_seconds,
                        "training_seconds": trial.pop("training_seconds"),
                    }
                )
                # Checkpoint IDs include execution timing and remain in runtime provenance.
                times[-1]["checkpoint_id"] = trial.pop("checkpoint_id")
                trial.pop("resumed")
                trial["history"] = rows
                clean.append(trial)
            report["trials"] = clean
            selections[fold_id] = report
            timings[fold_id] = times
        descriptor = {
            "schema_version": "neural-1.0",
            "experiment_id": experiment_id,
            "family": config.family,
            "role": "selected",
            "config": config.to_dict(),
            **lineage,
            "protocol": split.config.protocol.name,
            "population_sha256": population.hexdigest(),
            "batch_size": 256,
            "prediction_availability": "all eligible cells finite or experiment aborts",
            "metric_policy": {
                "mask_zero": split.config.validity.mask_zero,
                "mape": "percent; zero denominators excluded",
                "units": dataset.lineage["units"],
                "aggregation": "micro forecast cells",
            },
            "temporal": temporal_diagnostics(dataset.timestamps, split),
        }
        return publish_run(
            {f: s.fitted.model for f, s in selected.items()},
            {f: s.training.scaler.to_dict() for f, s in selected.items()},
            {
                "metrics": {"groups": metrics.result()},
                "selection": selections,
                "runtime": {
                    "training": timings,
                    "scoring": scoring,
                    "total_seconds_before_publication": time.perf_counter() - started,
                    "environment": collect_environment_info().to_dict(),
                },
            },
            descriptor,
            runs,
            raw_snapshot_dir=raw_snapshot_dir,
            canonical_dir=canonical_dir,
            plan_dir=plan_dir,
        )
