"""Streaming aggregation of existing metrics, never a second metric implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from urbangraph.baselines.contracts import BoolArray, FloatArray
from urbangraph.evaluation.contracts import TargetValidityPolicy
from urbangraph.evaluation.metrics import metric


@dataclass
class MetricAccumulator:
    weighted: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    cells: int = 0
    zero_denominators: int = 0

    def add(self, truth: FloatArray, predicted: FloatArray, mask: BoolArray) -> None:
        self.cells += truth.size
        self.zero_denominators += int(np.count_nonzero(mask & (truth == 0)))
        for name in ("mae", "rmse", "mape"):
            selected = mask & (truth != 0) if name == "mape" else mask
            if not selected.any():
                continue
            measured = metric(name, truth, predicted, validity=TargetValidityPolicy(), mask=mask)
            power = 2 if name == "rmse" else 1
            self.weighted[name] = self.weighted.get(name, 0.0) + (
                measured.value**power * measured.valid_count
            )
            self.counts[name] = self.counts.get(name, 0) + measured.valid_count

    def result(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for name in ("mae", "rmse", "mape"):
            count = self.counts.get(name, 0)
            value = self.weighted.get(name, 0.0) / count if count else None
            if value is not None and name == "rmse":
                value = float(np.sqrt(value))
            result[name] = {
                "value": value,
                "valid_count": count,
                "excluded_count": self.cells - count,
                "zero_denominator_count": self.zero_denominators if name == "mape" else 0,
                "status": "ok" if count else "empty_population",
            }
        return result


@dataclass
class GroupedMetrics:
    groups: dict[str, MetricAccumulator] = field(default_factory=dict)

    def add(self, name: str, truth: FloatArray, prediction: FloatArray, mask: BoolArray) -> None:
        self.groups.setdefault(name, MetricAccumulator()).add(truth, prediction, mask)

    def result(self) -> dict[str, object]:
        return {name: accumulator.result() for name, accumulator in sorted(self.groups.items())}
