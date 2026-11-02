"""Predeclared direct/residual comparison using only fold training and validation."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

from urbangraph.data import DataError
from urbangraph.datasets.serialization import read_json, write_json
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.preprocessing import fit_preprocessing
from urbangraph.neural.trainer import FitResult, fit


@dataclass
class Selection:
    fitted: FitResult
    training: NeuralWindows
    report: dict[str, object]


def select_model(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    config: NeuralConfig,
    root: Path,
    lineage: dict[str, object],
    deadline: float,
) -> Selection:
    scaler = fit_preprocessing(dataset, split, fold, config.scaling)
    train = NeuralWindows(
        dataset,
        split,
        fold,
        "train",
        scaler,
        calendar=config.calendar,
        maximum=config.max_train_windows,
        seed=config.seed,
    )
    validation = NeuralWindows(
        dataset,
        split,
        fold,
        "validation",
        scaler,
        calendar=config.calendar,
        maximum=config.max_validation_windows,
        seed=config.seed + 1,
    )
    candidates = (False, True) if config.search_residual else (config.residual,)
    trials: list[dict[str, object]] = []
    best: FitResult | None = None
    best_mae = float("inf")
    for residual in candidates:
        candidate = replace(config, residual=residual)
        trial_root = root / ("residual" if residual else "direct")
        trial_root.mkdir(parents=True, exist_ok=True)
        checkpoints = trial_root / "epochs"
        epochs = sorted(checkpoints.glob("epoch-*")) if checkpoints.exists() else []
        completed = trial_root / "completed.json"
        started = time.perf_counter()
        try:
            result = fit(
                train,
                validation,
                candidate,
                checkpoints,
                lineage,
                resume=epochs[-1] if epochs else None,
                deadline=deadline,
            )
            recorded = result.report
            if completed.exists():
                saved = read_json(completed)
                if saved.get("checkpoint_id") != recorded["checkpoint_id"]:
                    raise DataError("Completed trial checkpoint changed")
                recorded = saved
            else:
                write_json(completed, recorded)
            measured = recorded["best_validation_mae"]
            if not isinstance(measured, float):
                raise DataError("Missing validation score")
            trials.append({"config": candidate.to_dict(), "status": "completed", **recorded})
            if measured < best_mae:
                best, best_mae = result, measured
        except (DataError, RuntimeError) as exc:
            # Preserve failed attempts separately; they are not published as scientific results.
            attempt = {
                "config": candidate.to_dict(),
                "status": "failed",
                "error": str(exc),
                "seconds": time.perf_counter() - started,
            }
            destination = trial_root / f"failure-{time.time_ns()}.json"
            write_json(destination, attempt)
            raise
    if best is None:
        raise DataError("No completed neural candidate")
    return Selection(
        best,
        train,
        {
            "trials": trials,
            "selected_config": best.model.config.to_dict(),
            "policy": "minimum validation MAE; candidate order breaks ties",
            "training_population": train.report,
            "validation_population": validation.report,
            "scaler": scaler.to_dict(),
        },
    )
