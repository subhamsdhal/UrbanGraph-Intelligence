"""Explicit derived scaler policy using the existing training-only implementation."""

from dataclasses import replace

import numpy as np

from urbangraph.baselines.contracts import FloatArray
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.preprocessing import FittedStandardizer, fit_standardizer


def fit_preprocessing(
    dataset: EvaluationDataset, split: SplitManifest, fold: int, mode: str
) -> FittedStandardizer:
    # Scaling changes no boundaries/windows. Preserve original immutable evaluation plan.
    derived = plan_splits(dataset.timestamps, replace(split.config, scaling=mode))
    if derived.folds != split.folds:
        raise DataError("Preprocessing policy changed evaluation populations")
    return fit_standardizer(dataset, derived, fold=fold)


def sanitize_history(
    values: FloatArray,
    valid: np.ndarray[tuple[int, ...], np.dtype[np.bool_]],
    scaler: FittedStandardizer,
    sensors: tuple[str, ...],
) -> FloatArray:
    normalized = scaler.transform(values, sensor_ids=sensors)
    normalized = np.where(valid, normalized, np.nan)
    last: FloatArray = np.zeros(len(sensors), dtype=np.float64)  # train mean in normalized space
    filled = np.empty_like(normalized)
    for step in range(len(values)):
        last = np.where(valid[step], normalized[step], last)
        filled[step] = last
    return filled
