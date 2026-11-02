"""Full-population scoring with original labels and shared baseline metric policies."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from urbangraph.baselines.scoring import GroupedMetrics
from urbangraph.data import DataError
from urbangraph.evaluation.windows import validity_mask
from urbangraph.neural.datasets import NeuralWindows, collate
from urbangraph.neural.trainer import BudgetExceeded
from urbangraph.neural.training_policy import TrainingPolicy


def evaluate_fold(
    model: nn.Module,
    windows: NeuralWindows,
    config: TrainingPolicy,
    metrics: GroupedMetrics,
    *,
    deadline: float | None = None,
    population_update: Callable[[bytes], object] | None = None,
) -> str:
    """Hash origins and masks in the existing baseline's 256-origin batch order."""
    if windows.report["partition"] != "test":
        raise DataError("Held-out scoring requires an explicit test population")
    population = hashlib.sha256()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(windows), 256):
            stop = min(start + 256, len(windows))
            origins = windows.origins[start:stop]
            truth = windows.raw_targets(origins)[..., None]
            valid = validity_mask(truth, windows.validity)
            prediction = np.empty(truth.shape, dtype=np.float64)
            for offset in range(start, stop, config.batch_size):
                if deadline is not None and time.perf_counter() >= deadline:
                    raise BudgetExceeded("Deadline during held-out evaluation")
                end = min(offset + config.batch_size, stop)
                x, _, _, _ = collate([windows[i] for i in range(offset, end)])
                normalized = model(x.to(config.device)).cpu().numpy().astype(np.float64)
                predicted = windows.scaler.inverse_transform(
                    normalized[..., 0], sensor_ids=windows.sensor_ids
                )
                prediction[offset - start : end - start, ..., 0] = predicted
            if not np.isfinite(prediction).all():
                raise DataError("Nonfinite forecast; no incomplete population may be scored")
            population.update(origins.astype("<i8").tobytes())
            population.update(valid.tobytes())
            if population_update is not None:
                population_update(origins.astype("<i8").tobytes())
                population_update(valid.tobytes())
            metrics.add("common/overall", truth, prediction, valid)
            metrics.add(f"common/fold/{windows.report['fold']}", truth, prediction, valid)
            metrics.add("zero-masked/overall", truth, prediction, valid & (truth != 0))
            for horizon in range(windows.horizon):
                metrics.add(
                    f"common/horizon/{horizon + 1:02d}",
                    truth[:, horizon],
                    prediction[:, horizon],
                    valid[:, horizon],
                )
            for sensor, name in enumerate(windows.sensor_ids):
                metrics.add(
                    f"common/sensor/{name}",
                    truth[:, :, sensor],
                    prediction[:, :, sensor],
                    valid[:, :, sensor],
                )
    return population.hexdigest()
