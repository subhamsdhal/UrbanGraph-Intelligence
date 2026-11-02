"""Explicit train-only standardization; never fit an implicit full dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.contracts import BENCHMARK_PROTOCOLS, SplitManifest, TargetValidityPolicy
from urbangraph.evaluation.dataset import EvaluationDataset, identity
from urbangraph.evaluation.planning import plan_splits
from urbangraph.evaluation.windows import numerical_values, validity_mask


@dataclass(frozen=True)
class FittedStandardizer:
    mode: str
    sensor_ids: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    valid_counts: tuple[int, ...]
    lineage: dict[str, object]
    validity: TargetValidityPolicy

    def __post_init__(self) -> None:
        count = 1 if self.mode == "global" else len(self.sensor_ids)
        if (
            self.mode not in {"global", "per_sensor"}
            or not self.sensor_ids
            or len(set(self.sensor_ids)) != len(self.sensor_ids)
            or any(not isinstance(s, str) or not s for s in self.sensor_ids)
            or any(len(v) != count for v in (self.means, self.scales, self.valid_counts))
            or any(not np.isfinite(v) for v in (*self.means, *self.scales))
            or any(v <= 0 for v in self.scales)
            or any(type(v) is not int or v < 1 for v in self.valid_counts)
        ):
            raise DataError("Invalid fitted standardizer state")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: object) -> FittedStandardizer:
        obj = object_dict(value)
        if set(obj) != {
            "mode",
            "sensor_ids",
            "means",
            "scales",
            "valid_counts",
            "lineage",
            "validity",
        }:
            raise DataError("Invalid fitted state fields")
        mode, sensors, counts = obj["mode"], obj["sensor_ids"], obj["valid_counts"]
        if (
            not isinstance(mode, str)
            or not isinstance(sensors, list)
            or any(not isinstance(s, str) for s in sensors)
            or not isinstance(counts, list)
            or any(type(c) is not int for c in counts)
        ):
            raise DataError("Invalid fitted state labels or counts")
        numbers: list[tuple[float, ...]] = []
        for name in ("means", "scales"):
            values = obj[name]
            if not isinstance(values, list) or any(type(v) not in (int, float) for v in values):
                raise DataError("Invalid fitted numerical parameters")
            numbers.append(tuple(float(v) for v in values))
        validity = object_dict(obj["validity"])
        if set(validity) != {"mask_zero"} or type(validity["mask_zero"]) is not bool:
            raise DataError("Invalid fitted validity policy")
        return cls(
            mode,
            tuple(str(s) for s in sensors),
            numbers[0],
            numbers[1],
            tuple(int(c) for c in counts),
            object_dict(obj["lineage"]),
            TargetValidityPolicy(validity["mask_zero"]),
        )

    def _apply(
        self, values: NDArray[np.float64], sensor_ids: tuple[str, ...], inverse: bool
    ) -> NDArray[np.float64]:
        if sensor_ids != self.sensor_ids or values.ndim < 2 or values.shape[-1] != len(sensor_ids):
            raise DataError("Scaler requires aligned [..., sensors] values and exact sensor IDs")
        array = np.asarray(values, dtype=np.float64)
        means, scales = np.asarray(self.means), np.asarray(self.scales)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            output = array * scales + means if inverse else (array - means) / scales
        if np.any(np.isfinite(array) & ~np.isfinite(output)):
            raise DataError("Standardization overflow; no silent clipping")
        return np.asarray(output, dtype=np.float64)

    def transform(
        self, values: NDArray[np.float64], *, sensor_ids: tuple[str, ...]
    ) -> NDArray[np.float64]:
        return self._apply(values, sensor_ids, False)

    def inverse_transform(
        self, values: NDArray[np.float64], *, sensor_ids: tuple[str, ...]
    ) -> NDArray[np.float64]:
        return self._apply(values, sensor_ids, True)


def fit_standardizer(
    dataset: EvaluationDataset, split: SplitManifest, *, fold: int
) -> FittedStandardizer:
    """Strict: unique rows of TRAIN only. Benchmark: union of training input rows only."""
    if split.config.scaling == "none":
        raise DataError("Standardization requires explicitly configured scaling")
    if plan_splits(dataset.timestamps, split.config) != split:
        raise DataError("Split does not match dataset")
    if type(fold) is not int or not 0 <= fold < len(split.folds):
        raise DataError("Invalid scaler fold")
    train = split.folds[fold].partition("train")
    start, stop = train.start, train.stop
    basis = "unique training partition rows"
    if split.config.protocol.name in BENCHMARK_PROTOCOLS:
        start = train.samples[0].start - split.config.window.history + 1
        stop = train.samples[-1].stop
        basis = "unique training input rows; not upstream overlapping-window-weighted scaler"
    data = numerical_values(dataset.frame.iloc[start:stop])
    mask = validity_mask(data, split.config.validity)
    if split.config.scaling == "global":
        data, mask = data.reshape(-1, 1), mask.reshape(-1, 1)
    means: list[float] = []
    scales: list[float] = []
    counts: list[int] = []
    for i in range(data.shape[1]):
        valid = data[:, i][mask[:, i]]
        if not len(valid):
            raise DataError("All-invalid training feature; cannot fit standardizer")
        # Rescale before reductions to avoid squaring huge physical values.
        bound = float(np.max(np.abs(valid)))
        normalized = valid / bound if bound else valid
        mean = float(np.mean(normalized) * bound)
        scale = float(np.std(normalized, ddof=0) * bound)
        if not np.isfinite(mean) or not np.isfinite(scale):
            raise DataError("Nonfinite fitted parameters")
        means.append(mean)
        scales.append(scale if scale else 1.0)
        counts.append(len(valid))
    return FittedStandardizer(
        split.config.scaling,
        dataset.sensor_ids,
        tuple(means),
        tuple(scales),
        tuple(counts),
        {
            "canonical_lineage": dataset.lineage,
            "split_sha256": identity(split.to_dict()),
            "fold_id": split.folds[fold].fold_id,
            "fit_rows": [start, stop],
            "basis": basis,
            "ddof": 0,
            "zero_variance_scale": 1.0,
            "all_invalid": "error",
        },
        split.config.validity,
    )
