"""Deterministic time-stratified sensor/horizon sampling with explicit fit scope."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from urbangraph.baselines.contracts import FloatArray, IntArray
from urbangraph.baselines.history import HistoricalAccess, training_data
from urbangraph.boosting.contracts import FeatureSpec
from urbangraph.boosting.features import FeatureRows, build_features
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.windows import validity_mask


@dataclass(frozen=True)
class SelectedRows:
    origins: IntArray
    sensors: IntArray
    horizons: IntArray
    report: dict[str, object]


def select_rows(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    partition: str,
    maximum: int,
    seed: int,
    values: FloatArray,
) -> SelectedRows:
    if partition not in {"train", "validation"}:
        raise DataError("Sampling is restricted to training or validation")
    part = split.folds[fold].partition(partition)
    origins = np.concatenate([np.arange(r.start, r.stop, dtype=np.int64) for r in part.samples])
    if partition == "train":
        scope = training_data(dataset, split, fold)
        limits = scope.lineage["fit_rows"]
        assert isinstance(limits, list)
        # Every target in a sampled training window stays in conservative UG-003A fit scope.
        origins = origins[origins + split.config.window.horizon < int(limits[1])]
    sensors, horizon = len(dataset.sensor_ids), split.config.window.horizon
    strata = sensors * horizon
    if maximum < strata or not origins.size:
        raise DataError("Sampling budget must cover every sensor/horizon stratum")
    each = min(len(origins), maximum // strata)
    rng = np.random.default_rng(seed)
    # One jittered origin per temporal bin for every sensor/horizon; bins do not overlap.
    cuts = np.linspace(0, len(origins), each + 1, dtype=np.int64)
    choice = rng.integers(cuts[:-1], cuts[1:], size=(strata, each))
    selected_origins: IntArray = origins[choice].ravel()
    selected_sensors: IntArray = np.repeat(
        np.repeat(np.arange(sensors, dtype=np.int64), horizon), each
    )
    selected_horizons: IntArray = np.repeat(
        np.tile(np.arange(1, horizon + 1, dtype=np.int64), sensors), each
    )
    targets = values[selected_origins + selected_horizons, selected_sensors]
    keep = validity_mask(targets, split.config.validity)
    original_count = len(targets)
    selected_origins = selected_origins[keep]
    selected_sensors = selected_sensors[keep]
    selected_horizons = selected_horizons[keep]
    if not selected_origins.size:
        raise DataError("No finite eligible sampled targets")
    order = np.lexsort((selected_sensors, selected_horizons, selected_origins))
    selected_origins, selected_sensors, selected_horizons = (
        selected_origins[order],
        selected_sensors[order],
        selected_horizons[order],
    )
    report: dict[str, object] = {
        "partition": partition,
        "candidate_cells_before_target_mask": len(origins) * strata,
        "selected_before_target_mask": original_count,
        "selected_rows": len(selected_origins),
        "invalid_selected_targets": original_count - len(selected_origins),
        "seed": seed,
        "policy": "equal sensor/horizon quota; one seeded jittered origin per temporal bin",
        "subsampled": each < len(origins),
        "time_bins": each,
        "sensor_counts": np.bincount(selected_sensors, minlength=sensors).tolist(),
        "horizon_counts": np.bincount(selected_horizons, minlength=horizon + 1)[1:].tolist(),
        "weekday_counts": np.bincount(
            np.asarray(dataset.timestamps[selected_origins].dayofweek), minlength=7
        ).tolist(),
        "hour_counts": np.bincount(
            np.asarray(dataset.timestamps[selected_origins].hour), minlength=24
        ).tolist(),
        "selected_identity": identity(
            [selected_origins.tolist(), selected_sensors.tolist(), selected_horizons.tolist()]
        ),
        "first_origin": int(selected_origins.min()),
        "last_origin": int(selected_origins.max()),
    }
    return SelectedRows(selected_origins, selected_sensors, selected_horizons, report)


def feature_chunk(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    partition: str,
    values: FloatArray,
    origins: IntArray,
    sensors: IntArray,
    horizons: IntArray,
    spec: FeatureSpec,
) -> FeatureRows:
    part = split.folds[fold].partition(partition)
    allowed = np.zeros(origins.shape, dtype=bool)
    for run in part.samples:
        allowed |= (origins >= run.start) & (origins < run.stop)
    if not allowed.all() or np.any(horizons > split.config.window.horizon):
        raise DataError("Feature origins/horizons outside verified plan")
    lower = part.start if split.config.protocol.context == "isolated" else 0
    delay = split.config.features.observed_delay_steps
    access = HistoricalAccess(
        values,
        dataset.timestamps,
        origins,
        delay=delay,
        lower=lower,
        validity=split.config.validity,
    )
    return build_features(
        spec,
        access,
        sensors,
        horizons,
        dataset.timestamps[origins + horizons],
        dataset.sensor_ids,
        delay,
    )


def sampled_matrix(
    dataset: EvaluationDataset,
    split: SplitManifest,
    fold: int,
    partition: str,
    values: FloatArray,
    selected: SelectedRows,
    spec: FeatureSpec,
    batch_size: int,
) -> tuple[NDArray[np.float32], FloatArray, FeatureRows]:
    if type(batch_size) is not int or batch_size < 1:
        raise DataError("Positive feature batch size required")
    matrix: NDArray[np.float32] | None = None
    schema: FeatureRows | None = None
    for start in range(0, len(selected.origins), batch_size):
        sl = slice(start, start + batch_size)
        chunk = feature_chunk(
            dataset,
            split,
            fold,
            partition,
            values,
            selected.origins[sl],
            selected.sensors[sl],
            selected.horizons[sl],
            spec,
        )
        if matrix is None:
            matrix = np.empty((len(selected.origins), len(chunk.names)), dtype=np.float32)
            schema = chunk
        matrix[sl] = chunk.values
    if matrix is None or schema is None:
        raise DataError("Empty selected feature matrix")
    target = np.asarray(
        values[selected.origins + selected.horizons, selected.sensors], dtype=np.float64
    )
    return matrix, target, schema
