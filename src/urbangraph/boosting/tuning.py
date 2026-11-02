"""Frozen sequential trials selected exclusively by sampled validation MAE."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from urbangraph.baselines.scoring import MetricAccumulator
from urbangraph.boosting.contracts import BoostConfig, FeatureSpec
from urbangraph.boosting.models import BoostModel, BudgetExceeded, train_model
from urbangraph.boosting.sampling import sampled_matrix, select_rows
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.windows import numerical_values


@dataclass
class Selection:
    models: dict[str, BoostModel]
    report: dict[str, object]


def select_models(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    config: BoostConfig,
    deadline: float,
) -> Selection:
    values = numerical_values(dataset.frame)
    training = select_rows(
        dataset, split, fold, "train", config.max_train_rows, config.seed, values
    )
    validation = select_rows(
        dataset, split, fold, "validation", config.max_validation_rows, config.seed + 1, values
    )
    candidates = [(v, config.max_depth) for v in config.variants]
    if "F4" in config.variants and config.alternative_depth != config.max_depth:
        candidates.append(("F4", config.alternative_depth))
    reports: list[dict[str, object]] = []
    selected: dict[str, BoostModel] = {}
    best = float("inf")
    best_index = -1
    training_times: list[object] = []
    for variant, depth in candidates:
        if time.perf_counter() >= deadline:
            raise BudgetExceeded("Experiment deadline reached before next trial")
        spec = FeatureSpec(variant=variant, seasonal_policy=config.seasonal_policy)
        train_x, train_y, schema = sampled_matrix(
            dataset, split, fold, "train", values, training, spec, config.feature_batch
        )
        val_x, val_y, _ = sampled_matrix(
            dataset, split, fold, "validation", values, validation, spec, config.feature_batch
        )
        trial = replace(config, max_depth=depth)
        model, metadata = train_model(
            trial,
            spec,
            schema.names,
            schema.groups,
            dataset.sensor_ids,
            split.config.window.horizon,
            train_x,
            train_y,
            val_x,
            val_y,
            deadline,
        )
        training_times.append(metadata.pop("training_seconds"))
        predicted = model.predict(val_x, schema.names, dataset.sensor_ids)
        score = MetricAccumulator()
        score.add(val_y, predicted, np.ones(val_y.shape, dtype=bool))
        mae = score.weighted["mae"] / score.counts["mae"]
        reports.append(
            {
                "variant": variant,
                "config": trial.to_dict(),
                "metrics": score.result(),
                "training": metadata,
            }
        )
        if variant == "F0":
            selected["F0"] = model
        if variant != "F0" and mae < best:
            best, best_index = mae, len(reports) - 1
            selected["selected"] = model
    if "selected" not in selected:
        selected["selected"] = selected["F0"]
        best_index = 0
    return Selection(
        selected,
        {
            "criterion": "sampled validation MAE; first trial wins ties",
            "training_sample": training.report,
            "validation_sample": validation.report,
            "trials": reports,
            "selected_index": best_index,
            "refit": False,
            "calibration_access": False,
            "training_seconds": training_times,
        },
    )
