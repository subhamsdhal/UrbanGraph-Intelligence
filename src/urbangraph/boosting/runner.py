"""Bounded full-population scoring after frozen validation-only selection."""

from __future__ import annotations

import hashlib
import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from urbangraph.baselines.contracts import FloatArray, IntArray
from urbangraph.baselines.history import make_batch
from urbangraph.baselines.runner import origin_batches
from urbangraph.baselines.scoring import GroupedMetrics
from urbangraph.boosting.artifacts import publish_run
from urbangraph.boosting.contracts import BoostConfig, FeatureSpec
from urbangraph.boosting.models import BoostModel, BudgetExceeded
from urbangraph.boosting.sampling import feature_chunk
from urbangraph.boosting.tuning import select_models
from urbangraph.data import DataError
from urbangraph.data.filesystem import safe_absolute
from urbangraph.data.rights import require_verified_snapshot_authorization
from urbangraph.evaluation.artifacts import code_provenance, load_plan
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.temporal import temporal_diagnostics
from urbangraph.evaluation.windows import numerical_values, validity_mask
from urbangraph.reproducibility import collect_environment_info

LOGGER = logging.getLogger("urbangraph.boosting")


def predict_batch(
    model: BoostModel,
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    partition: str,
    values: FloatArray,
    origins: IntArray,
    batch_size: int,
) -> tuple[FloatArray, np.ndarray[tuple[int, ...], np.dtype[np.bool_]]]:
    if not isinstance(model.features, FeatureSpec):
        raise DataError("Traffic inference requires a traffic feature specification")
    horizon, sensors = model.horizon, len(model.sensor_ids)
    o = np.repeat(origins, horizon * sensors)
    h = np.tile(np.repeat(np.arange(1, horizon + 1, dtype=np.int64), sensors), len(origins))
    n = np.tile(np.arange(sensors, dtype=np.int64), len(origins) * horizon)
    result = np.empty(len(o), dtype=np.float64)
    unavailable = np.empty(len(o), dtype=bool)
    for start in range(0, len(o), batch_size):
        sl = slice(start, start + batch_size)
        features = feature_chunk(
            dataset, split, fold, partition, values, o[sl], n[sl], h[sl], model.features
        )
        result[sl] = model.predict(features.values, features.names, dataset.sensor_ids)
        unavailable[sl] = features.unavailable_seasonal
    shape = (len(origins), horizon, sensors, 1)
    return result.reshape(shape), unavailable.reshape(shape)


def run_experiment(
    canonical_dir: str | Path,
    plan_dir: str | Path,
    config: BoostConfig,
    *,
    raw_snapshot_dir: str | Path,
    output_root: str | Path = "data/boosting",
) -> list[Path]:
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
    if split.config.window.interval_seconds != 300:
        raise DataError("Temporal v1 features require five-minute nominal sampling")
    output = safe_absolute(output_root)
    for root in (
        safe_absolute(canonical_dir).parents[2],
        safe_absolute(plan_dir).parents[2],
        safe_absolute(canonical_dir).parents[2].parent / "raw",
    ):
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise DataError("Boosting output overlaps managed data")
    values = numerical_values(dataset.frame)
    reports = {name: GroupedMetrics() for name in ("F0", "selected")}
    fitted: dict[str, dict[str, BoostModel]] = {name: {} for name in reports}
    selections: dict[str, object] = {}
    runtimes: dict[str, object] = {}
    population = hashlib.sha256()
    prediction_seconds = {role: 0.0 for role in reports}
    for fold_number, fold in enumerate(split.folds):
        LOGGER.info("Selecting %s for %s", config.family, fold.fold_id)
        selection = select_models(dataset, split, fold_number, config, deadline)
        # Training times are execution metadata, excluded from scientific content identity.
        training_times = selection.report.pop("training_seconds")
        selections[fold.fold_id] = selection.report
        for role, model in selection.models.items():
            fitted[role][fold.fold_id] = model
        scoring_started = time.perf_counter()
        for origins in origin_batches(fold.partition("test"), config.evaluation_batch):
            if time.perf_counter() >= deadline:
                raise BudgetExceeded("Experiment deadline reached during full test scoring")
            batch = make_batch(dataset, split, fold_number, "test", origins, values)
            truth = values[batch.target_positions][..., None]
            valid = validity_mask(truth, split.config.validity)
            population.update(origins.astype("<i8").tobytes())
            population.update(valid.tobytes())
            weekday = np.asarray(batch.target_labels.dayofweek).reshape(truth.shape[:2])
            hour = np.asarray(batch.target_labels.hour).reshape(truth.shape[:2])
            history = batch.x[..., 0]
            finite = np.isfinite(history)
            count = finite.sum(axis=1)
            mean = np.divide(
                np.where(finite, history, 0).sum(axis=1),
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )
            variance = np.divide(
                np.where(finite, (history - mean[:, None, :]) ** 2, 0).sum(axis=1),
                count,
                out=np.full(count.shape, np.nan),
                where=count > 0,
            )
            for role, model in selection.models.items():
                prediction_started = time.perf_counter()
                prediction, seasonal = predict_batch(
                    model,
                    dataset,
                    split,
                    fold_number,
                    "test",
                    values,
                    origins,
                    config.feature_batch,
                )
                prediction_seconds[role] += time.perf_counter() - prediction_started
                report = reports[role]
                report.add("common/overall", truth, prediction, valid)
                report.add(f"common/fold/{fold.fold_id}", truth, prediction, valid)
                report.add("zero-masked/overall", truth, prediction, valid & (truth != 0))
                for h in range(model.horizon):
                    report.add(
                        f"common/horizon/{h + 1:02d}", truth[:, h], prediction[:, h], valid[:, h]
                    )
                for n, sensor in enumerate(dataset.sensor_ids):
                    report.add(
                        f"common/sensor/{sensor}",
                        truth[:, :, n],
                        prediction[:, :, n],
                        valid[:, :, n],
                    )
                for label, mask in (("weekday", weekday < 5), ("weekend", weekday >= 5)):
                    report.add(
                        f"diagnostic/{label}", truth, prediction, valid & mask[:, :, None, None]
                    )
                for hour_value in np.unique(hour):
                    report.add(
                        f"diagnostic/hour/{hour_value:02d}",
                        truth,
                        prediction,
                        valid & (hour == hour_value)[:, :, None, None],
                    )
                for label, mask in (
                    ("target-below-40", truth < 40),
                    ("target-at-least-40", truth >= 40),
                ):
                    report.add(f"retrospective/{label}", truth, prediction, valid & mask)
                report.add("diagnostic/seasonal-unavailable", truth, prediction, valid & seasonal)
                report.add(
                    "diagnostic/missing-input",
                    truth,
                    prediction,
                    valid & np.asarray(~finite.all(axis=1), dtype=bool)[:, None, :, None],
                )
                report.add(
                    "diagnostic/history-std-above-5",
                    truth,
                    prediction,
                    valid & (variance > 25)[:, None, :, None],
                )
                report.add(
                    "diagnostic/history-std-at-most-5",
                    truth,
                    prediction,
                    valid & (variance <= 25)[:, None, :, None],
                )
        runtimes[fold.fold_id] = {
            "training_seconds": training_times,
            "scoring_seconds_both_roles": time.perf_counter() - scoring_started,
        }
        LOGGER.info("Completed %s test scoring", fold.fold_id)
    descriptor: dict[str, object] = {
        "schema_version": "boosting-1.0",
        "prediction_availability": "all eligible cells finite or experiment aborts",
        "family": config.family,
        "config": config.to_dict(),
        "canonical": dataset.lineage,
        "plan_id": identity(plan),
        "protocol": split.config.protocol.name,
        "population_sha256": population.hexdigest(),
        "batch_size": config.evaluation_batch,
        "code": code_provenance(),
        "dependencies": {n: version(n) for n in ("lightgbm", "xgboost", "scipy")},
        "metric_policy": {
            "mask_zero": split.config.validity.mask_zero,
            "mape": "percent; zero denominators excluded",
            "units": dataset.lineage["units"],
            "aggregation": "micro forecast cells",
        },
        "temporal": temporal_diagnostics(dataset.timestamps, split),
        "feature_availability": (
            "historical observed traffic; known-future naive calendar; "
            "static canonical sensor identity"
        ),
    }
    runtime = {
        "folds": runtimes,
        "prediction_seconds_including_feature_construction": prediction_seconds,
        "total_seconds_before_publication": time.perf_counter() - started,
        "environment": collect_environment_info().to_dict(),
    }
    paths = []
    for role, models in fitted.items():
        paths.append(
            publish_run(
                models,
                {
                    "metrics": {"groups": reports[role].result()},
                    "selection": selections,
                    "runtime": runtime,
                    "importance": {fold: model.importance() for fold, model in models.items()},
                },
                descriptor | {"role": role},
                output,
                raw_snapshot_dir=raw_snapshot_dir,
                canonical_dir=canonical_dir,
                plan_dir=plan_dir,
            )
        )
    return paths
