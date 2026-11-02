"""Regional availability-aware windows and fixed fair validation objective."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from urbangraph.data import DataError
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.evaluation.windows import validity_mask
from urbangraph.neural.dataloaders import make_loader
from urbangraph.neural.datasets import NeuralSample, NeuralWindows
from urbangraph.neural.preprocessing import sanitize_history
from urbangraph.neural.training_policy import TrainingPolicy

OBJECTIVE = "macro-region-train-sd-normalized-mae-v1"


class RegionalWindows(NeuralWindows):
    def __init__(
        self,
        data: EvaluationDataset,
        split: SplitManifest,
        fold: int,
        partition: str,
        scaler: FittedStandardizer,
        latencies: dict[str, int],
        *,
        maximum: int | None = None,
        seed: int = 2026,
    ) -> None:
        if (
            scaler.mode != "per_sensor"
            or set(latencies) != set(data.sensor_ids)
            or any(
                type(v) is not int or not 0 <= v <= split.config.features.observed_delay_steps
                for v in latencies.values()
            )
        ):
            raise DataError("Aligned regional scales and bounded observation delays required")
        super().__init__(data, split, fold, partition, scaler, maximum=maximum, seed=seed)
        self.latencies = latencies
        self.report["region_latencies"] = latencies
        self.report["input_alignment"] = (
            "common UTC history ending at origin; unavailable recent cells masked"
        )

    def __getitem__(self, index: int) -> NeuralSample:
        sample = super().__getitem__(index)
        # Keep exactly the original delay-shifted history for the equal-delay cohort.
        # Unequal delays use the common history ending at origin-min(delay), masking
        # cells unavailable to each region. Never align distinct UTC labels as equal.
        end = sample.origin - min(self.latencies.values())
        rows = np.arange(end - self.history + 1, end + 1)
        values = self._values[rows].copy()
        valid = validity_mask(values, self.validity)
        for i, region in enumerate(self.sensor_ids):
            valid[:, i] &= rows <= sample.origin - self.latencies[region]
        clean = sanitize_history(values, valid, self.scaler, self.sensor_ids)
        x = np.stack([clean, valid.astype(np.float64)], axis=-1).astype(np.float32)
        if not np.isfinite(x).all():
            raise DataError("Regional input overflow")
        return NeuralSample(torch.from_numpy(x), sample.y, sample.mask, sample.origin)


def normalized_validation(
    model: nn.Module, dataset: NeuralWindows, config: TrainingPolicy
) -> dict[str, object]:
    errors = np.zeros(len(dataset.sensor_ids), dtype=np.float64)
    counts = np.zeros(len(dataset.sensor_ids), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y, mask, _ in make_loader(dataset, config):
            prediction = model(x.to(config.device)).cpu().numpy()[..., 0].astype(np.float64)
            actual = y.numpy()[..., 0].astype(np.float64)
            valid = mask.numpy()[..., 0]
            if not np.isfinite(prediction).all():
                raise DataError("Nonfinite validation predictions")
            errors += np.where(valid, np.abs(prediction - actual), 0).sum(axis=(0, 1))
            counts += valid.sum(axis=(0, 1))
    if (counts == 0).any():
        raise DataError("Every region needs valid validation targets")
    regional = errors / counts
    return {
        "mae": {"value": float(regional.mean())},
        "objective": OBJECTIVE,
        "per_region": regional.tolist(),
        "counts": counts.tolist(),
    }
