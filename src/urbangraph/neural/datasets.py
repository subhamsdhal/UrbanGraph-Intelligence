"""Lazy tensor adapter over verified WindowCollection origin and context contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from urbangraph.baselines.contracts import FloatArray, IntArray
from urbangraph.data import DataError
from urbangraph.evaluation.contracts import SplitManifest
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.preprocessing import FittedStandardizer
from urbangraph.evaluation.windows import WindowCollection, numerical_values, validity_mask
from urbangraph.neural.preprocessing import sanitize_history


@dataclass
class NeuralSample:
    x: Tensor
    y: Tensor
    mask: Tensor
    origin: int


class NeuralWindows(Dataset[NeuralSample]):
    def __init__(
        self,
        dataset: EvaluationDataset,
        split: SplitManifest,
        fold: int,
        partition: str,
        scaler: FittedStandardizer,
        *,
        calendar: bool = False,
        maximum: int | None = None,
        seed: int = 2026,
    ) -> None:
        super().__init__()
        collection = WindowCollection(dataset, split, fold=fold, partition=partition)
        if scaler.sensor_ids != dataset.sensor_ids:
            raise DataError("Neural preprocessing sensor schema mismatch")
        if maximum is not None and (partition not in {"train", "validation"} or maximum < 1):
            raise DataError("Only train/validation may be explicitly subsampled")
        self.sensor_ids, self.scaler, self.calendar = dataset.sensor_ids, scaler, calendar
        self.history, self.horizon = split.config.window.history, split.config.window.horizon
        self.delay, self.validity = (
            split.config.features.observed_delay_steps,
            split.config.validity,
        )
        if calendar:
            split.config.features.validate_request("time_of_day", "known_future", future=False)
        # One canonical numerical view, never materialized overlapping windows.
        self._values = numerical_values(dataset.frame)
        self._timestamps = dataset.timestamps
        origins = np.concatenate(
            [np.arange(r.start, r.stop, dtype=np.int64) for r in collection.partition.samples]
        )
        total = len(origins)
        if maximum is not None and maximum < total:
            cuts = np.linspace(0, total, maximum + 1, dtype=np.int64)
            chosen = np.random.default_rng(seed).integers(cuts[:-1], cuts[1:])
            origins = origins[chosen]
        self.origins = origins
        self.report: dict[str, object] = {
            "partition": partition,
            "fold": collection.fold.fold_id,
            "eligible_windows": total,
            "selected_windows": len(origins),
            "seed": seed,
            "policy": "one seeded jittered window per temporal bin; all sensors and horizons",
            "origin_sha256": identity(origins.tolist()),
            "sensor_ids": list(self.sensor_ids),
            "history": self.history,
            "horizon": self.horizon,
            "calendar": calendar,
            "scaler_id": identity(scaler.to_dict()),
        }

    def __len__(self) -> int:
        return len(self.origins)

    def raw_targets(self, origins: IntArray) -> FloatArray:
        """Original float64 labels, avoiding float32 scaler roundtrip in metrics."""
        if origins.ndim != 1 or not np.isin(origins, self.origins).all():
            raise DataError("Targets requested outside the adapter population")
        positions = origins[:, None] + np.arange(1, self.horizon + 1)[None, :]
        return self._values[positions]

    def __getitem__(self, index: int) -> NeuralSample:
        if not 0 <= index < len(self):
            raise IndexError("Neural window index out of bounds")
        origin = int(self.origins[index])
        latest = origin - self.delay
        values = self._values[latest - self.history + 1 : latest + 1]
        valid = validity_mask(values, self.validity)
        clean = sanitize_history(values, valid, self.scaler, self.sensor_ids)
        features = [clean[..., None], valid.astype(np.float64)[..., None]]
        if self.calendar:
            labels = self._timestamps[latest - self.history + 1 : latest + 1]
            phase = (
                np.asarray(labels.hour * 3600 + labels.minute * 60 + labels.second)
                / 86400
                * 2
                * np.pi
            )
            for component in (np.sin(phase), np.cos(phase)):
                features.append(np.broadcast_to(component[:, None, None], (*clean.shape, 1)))
        target = self._values[origin + 1 : origin + self.horizon + 1]
        mask = validity_mask(target, self.validity)
        normalized = self.scaler.transform(target, sensor_ids=self.sensor_ids)
        normalized = np.where(mask, normalized, 0)
        x = np.concatenate(features, axis=-1).astype(np.float32)
        y = normalized[..., None].astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise DataError("Neural float32 input/target overflow")
        return NeuralSample(
            torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask[..., None]), origin
        )


def collate(samples: list[NeuralSample]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.stack([s.x for s in samples]),
        torch.stack([s.y for s in samples]),
        torch.stack([s.mask for s in samples]),
        torch.tensor([s.origin for s in samples], dtype=torch.int64),
    )
