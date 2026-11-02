"""Verified plans, train-only fit, validation selection and bounded test evaluation."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from urbangraph.baselines.artifacts import publish_run
from urbangraph.baselines.contracts import (
    MODEL_VERSION,
    BoolArray,
    IntArray,
    ModelConfig,
    Prediction,
)
from urbangraph.baselines.history import make_batch, training_data
from urbangraph.baselines.models import Baseline, fit_model
from urbangraph.baselines.scoring import GroupedMetrics, MetricAccumulator
from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.artifacts import code_provenance, load_plan
from urbangraph.evaluation.contracts import SplitManifest, TemporalPartition
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.evaluation.windows import numerical_values, validity_mask
from urbangraph.reproducibility import collect_environment_info

LOGGER = logging.getLogger("urbangraph.baselines")


def origin_batches(partition: TemporalPartition, batch_size: int) -> list[IntArray]:
    if type(batch_size) is not int or batch_size < 1:
        raise DataError("Batch size must be positive")
    origins = np.concatenate(
        [np.arange(r.start, r.stop, dtype=np.int64) for r in partition.samples]
    )
    return [origins[i : i + batch_size] for i in range(0, len(origins), batch_size)]


def common_population(target_valid: BoolArray, predictions: list[Prediction]) -> BoolArray:
    common = target_valid.copy()
    for prediction in predictions:
        if prediction.available.shape != common.shape:
            raise DataError("Comparison predictions must have the same target shape")
        common &= prediction.available
    return common


def select_ar(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    config: ModelConfig,
    batch_size: int,
) -> tuple[Baseline, dict[str, object]]:
    training = training_data(dataset, split, fold)
    # Frozen, small grid; no test or calibration access. No train-plus-validation refit.
    candidates = [
        replace(config, lags=lags, ridge=alpha)
        for lags in sorted({min(3, config.lags), config.lags})
        for alpha in sorted({config.ridge, config.ridge * 100})
    ]
    models = [fit_model(c, training, split.config.window.horizon) for c in candidates]
    scores = [MetricAccumulator() for _ in models]
    values = numerical_values(dataset.frame)
    for origins in origin_batches(split.folds[fold].partition("validation"), batch_size):
        batch = make_batch(dataset, split, fold, "validation", origins, values)
        predicted = [model.predict(batch) for model in models]
        truth = values[batch.target_positions][..., None]
        common = common_population(validity_mask(truth, split.config.validity), predicted)
        for score, prediction in zip(scores, predicted, strict=True):
            score.add(truth, prediction.values, common)
    means = [
        score.weighted.get("mae", 0) / score.counts["mae"]
        if score.counts.get("mae", 0)
        else float("inf")
        for score in scores
    ]
    chosen = int(np.argmin(means))
    if not np.isfinite(means[chosen]):
        raise DataError("No shared validation population for AR selection")
    return models[chosen], {
        "stage": "validation-only",
        "criterion": "common-population MAE",
        "candidates": [
            {"config": c.to_dict(), "metrics": s.result()}
            for c, s in zip(candidates, scores, strict=True)
        ],
        "selected_index": chosen,
        "refit": False,
    }


def run_suite(
    canonical_dir: str | Path,
    plan_dir: str | Path,
    configs: tuple[ModelConfig, ...],
    *,
    raw_snapshot_dir: str | Path,
    output_root: str | Path = "data/baselines",
    batch_size: int = 256,
) -> list[Path]:
    started = time.perf_counter()
    if not configs or len({c.model for c in configs}) != len(configs):
        raise DataError("Suite requires distinct model identifiers")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise DataError("Batch size must be between 1 and 4096")
    for purpose, operation in (
        ("model_training", "model_train"),
        ("scientific_evaluation", "model_evaluate"),
    ):
        require_verified_snapshot_authorization(
            raw_snapshot_dir, purpose, output_root, operation=operation
        )
    dataset, split, plan = load_plan(
        plan_dir, canonical_dir=canonical_dir, raw_snapshot_dir=raw_snapshot_dir
    )
    output = safe_absolute(output_root)
    managed = (
        safe_absolute(canonical_dir).parents[2],
        safe_absolute(plan_dir).parents[2],
        safe_absolute(canonical_dir).parents[2].parent / "raw",
    )
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in managed):
        raise DataError("Experiment output must not overlap managed data or plans")
    if split.config.window.interval_seconds != 300:
        raise DataError("Baseline v1 requires explicit five-minute nominal sampling")
    values = numerical_values(dataset.frame)
    reports = [GroupedMetrics() for _ in configs]
    states: list[dict[str, object]] = [{} for _ in configs]
    selections: list[dict[str, object]] = [{} for _ in configs]
    timings: list[dict[str, object]] = [{} for _ in configs]
    trajectories: list[list[dict[str, object]]] = [[] for _ in configs]
    availability = [
        dict(
            eligible_targets=0,
            valid_targets=0,
            valid_predictions=0,
            scoreable_targets=0,
            fallback_cells=0,
            common_targets=0,
        )
        for _ in configs
    ]
    population = hashlib.sha256()
    for fold_number, fold in enumerate(split.folds):
        LOGGER.info("Fitting baseline suite for %s", fold.fold_id)
        models: list[Baseline] = []
        for i, config in enumerate(configs):
            fit_start = time.perf_counter()
            if config.model == "autoregressive":
                model, selection = select_ar(dataset, split, fold_number, config, batch_size)
                selections[i][fold.fold_id] = selection
            elif config.model == "historical-average":
                model = fit_model(
                    config, training_data(dataset, split, fold_number), split.config.window.horizon
                )
            else:
                model = Baseline(config)
            timings[i][fold.fold_id] = {
                "fit_and_selection_seconds": time.perf_counter() - fit_start
            }
            models.append(model)
            states[i][fold.fold_id] = {
                "config": model.config.to_dict(),
                "state_id": model.state.state_id if model.state else None,
                "state": model.state.to_dict() if model.state else None,
            }
        for origins in origin_batches(fold.partition("test"), batch_size):
            batch = make_batch(dataset, split, fold_number, "test", origins, values)
            predictions = [model.predict(batch) for model in models]
            truth = values[batch.target_positions][..., None]
            valid = validity_mask(truth, split.config.validity)
            common = common_population(valid, predictions)
            population.update(origins.astype("<i8").tobytes())
            population.update(common.tobytes())
            weekday = np.asarray(batch.target_labels.dayofweek).reshape(truth.shape[:2])
            hour = np.asarray(batch.target_labels.hour).reshape(truth.shape[:2])
            for i, prediction in enumerate(predictions):
                p = prediction.values
                av = availability[i]
                av["eligible_targets"] += truth.size
                av["valid_targets"] += int(valid.sum())
                av["valid_predictions"] += int(prediction.available.sum())
                av["scoreable_targets"] += int((valid & prediction.available).sum())
                av["fallback_cells"] += int(prediction.fallback.sum())
                av["common_targets"] += int(common.sum())
                report = reports[i]
                report.add("common/overall", truth, p, common)
                report.add(f"common/fold/{fold.fold_id}", truth, p, common)
                report.add("model-specific/overall", truth, p, valid & prediction.available)
                report.add("zero-masked/overall", truth, p, common & (truth != 0))
                report.add(
                    "finite-targets/overall", truth, p, np.isfinite(truth) & prediction.available
                )
                for h in range(truth.shape[1]):
                    report.add(f"common/horizon/{h + 1:02d}", truth[:, h], p[:, h], common[:, h])
                for sensor, sensor_id in enumerate(dataset.sensor_ids):
                    report.add(
                        f"common/sensor/{sensor_id}",
                        truth[:, :, sensor],
                        p[:, :, sensor],
                        common[:, :, sensor],
                    )
                for label, selected in (("weekday", weekday < 5), ("weekend", weekday >= 5)):
                    report.add(f"diagnostic/{label}", truth, p, common & selected[:, :, None, None])
                for hour_value in np.unique(hour):
                    report.add(
                        f"diagnostic/hour/{hour_value:02d}",
                        truth,
                        p,
                        common & (hour == hour_value)[:, :, None, None],
                    )
                # Fixed value threshold, not fitted on test data; unknown physical units disclosed.
                report.add("retrospective/target-below-40", truth, p, common & (truth < 40))
                report.add("retrospective/target-at-least-40", truth, p, common & (truth >= 40))
                report.add("diagnostic/fallback", truth, p, common & prediction.fallback)
                missing_input = np.asarray(~np.isfinite(batch.x[..., 0]).all(axis=1), dtype=bool)
                report.add(
                    "diagnostic/missing-input", truth, p, common & missing_input[:, None, :, None]
                )
                if len(trajectories[i]) < 48:
                    for row in range(min(len(origins), 48 - len(trajectories[i]))):
                        actual, forecast = truth[row, 0, 0, 0], p[row, 0, 0, 0]
                        trajectories[i].append(
                            {
                                "fold": fold.fold_id,
                                "origin": int(origins[row]),
                                "target_position": int(batch.target_positions[row, 0]),
                                "sensor_id": dataset.sensor_ids[0],
                                "horizon": 1,
                                "actual": float(actual) if np.isfinite(actual) else None,
                                "predicted": float(forecast) if np.isfinite(forecast) else None,
                            }
                        )
        LOGGER.info("Completed test scoring for %s", fold.fold_id)
    duration = time.perf_counter() - started
    environment = collect_environment_info().to_dict()
    descriptor: dict[str, object] = {
        "schema_version": "1.0",
        "code": code_provenance(),
        "model_version": MODEL_VERSION,
        "canonical": dataset.lineage,
        "plan_id": identity(plan),
        "protocol": split.config.protocol.name,
        "batch_size": batch_size,
        "comparison_models": [c.to_dict() for c in configs],
        "population_sha256": population.hexdigest(),
        "metric_policy": {
            "mask_zero": split.config.validity.mask_zero,
            "mape": "percent; zero denominators excluded",
            "aggregation": "micro forecast cells; repeated targets count per origin",
            "primary": "common population MAE",
            "units": dataset.lineage["units"],
            "unmasked": "valid only when no target or prediction excluded",
        },
        "temporal": temporal_diagnostics(dataset.timestamps, split),
    }
    paths: list[Path] = []
    for i, config in enumerate(configs):
        av = availability[i]
        metrics = reports[i].result()
        fold_values = [
            float(
                reports[i].groups[f"common/fold/{f.fold_id}"].weighted["mae"]
                / reports[i].groups[f"common/fold/{f.fold_id}"].counts["mae"]
            )
            for f in split.folds
            if reports[i].groups[f"common/fold/{f.fold_id}"].counts.get("mae", 0)
        ]
        summary: dict[str, object] = {
            "groups": metrics,
            "trajectory": trajectories[i],
            "fold_mae_mean": float(np.mean(fold_values)) if fold_values else None,
            "fold_mae_std_ddof0": float(np.std(fold_values)) if fold_values else None,
        }
        availability_report: dict[str, object] = dict(av)
        fold_statistics: dict[str, object] = {}
        for name in ("mae", "rmse", "mape"):
            measured = [
                object_dict(object_dict(metrics[f"common/fold/{f.fold_id}"])[name])["value"]
                for f in split.folds
            ]
            finite = [float(v) for v in measured if isinstance(v, int | float)]
            fold_statistics[name] = {
                "mean": float(np.mean(finite)) if finite else None,
                "std_ddof0": float(np.std(finite)) if finite else None,
                "fold_count": len(finite),
            }
        summary["fold_statistics"] = fold_statistics
        availability_report.update(
            {
                "prediction_availability_percent": 100
                * av["valid_predictions"]
                / av["eligible_targets"],
                "fallback_percent": 100 * av["fallback_cells"] / av["eligible_targets"],
                "excluded_targets": av["eligible_targets"] - av["scoreable_targets"],
                "unmasked_status": "complete"
                if av["scoreable_targets"] == av["eligible_targets"]
                else "unavailable; use explicitly masked populations",
            }
        )
        paths.append(
            publish_run(
                {
                    "model-config": config.to_dict(),
                    "fitted-state": states[i],
                    "metrics": summary,
                    "availability": availability_report,
                    "lineage": {
                        "canonical": dataset.lineage,
                        "plan_id": identity(plan),
                        "raw_manifest": read_json(Path(canonical_dir) / "raw-manifest.json"),
                    },
                    "selection": selections[i],
                    "runtime": {
                        "suite_fit_selection_scoring_seconds": duration,
                        "model_folds": timings[i],
                        "environment": environment,
                        "memory": "not measured",
                    },
                },
                descriptor | {"model": config.model},
                output_root,
                raw_snapshot_dir=raw_snapshot_dir,
                canonical_dir=canonical_dir,
                plan_dir=plan_dir,
            )
        )
    return paths
