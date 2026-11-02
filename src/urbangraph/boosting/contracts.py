"""Explicit, bounded feature and experiment policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.evaluation.contracts import integer


@dataclass(frozen=True)
class FeatureSpec:
    version: str = "temporal-v1"
    variant: str = "F4"
    lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    rolling: tuple[int, ...] = (3, 6, 12)
    seasonal_policy: str = "positional"
    missing_indicators: bool = True

    def __post_init__(self) -> None:
        if self.version != "temporal-v1" or self.variant not in {"F0", "F1", "F2", "F3", "F4"}:
            raise DataError("Unsupported feature specification")
        for numbers in (self.lags, self.rolling):
            if (
                not isinstance(numbers, tuple)
                or not numbers
                or len(set(numbers)) != len(numbers)
                or any(type(v) is not int or not 1 <= v <= 2016 for v in numbers)
            ):
                raise DataError("Feature windows must be unique positive bounded integers")
        if self.seasonal_policy not in {"positional", "timestamp-label"}:
            raise DataError("Unknown seasonal feature policy")
        if type(self.missing_indicators) is not bool:
            raise DataError("Missing indicators must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoostConfig:
    family: str = "lightgbm"
    version: str = "1.0"
    trees: int = 100
    learning_rate: float = 0.08
    max_depth: int = 5
    num_leaves: int = 31
    min_child_samples: int = 40
    min_child_weight: float = 10.0
    subsample: float = 1.0
    colsample: float = 1.0
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    early_stopping: int = 12
    seed: int = 2026
    threads: int = 2
    max_train_rows: int = 100000
    max_validation_rows: int = 20000
    feature_batch: int = 8192
    evaluation_batch: int = 256
    trial_seconds: int = 120
    run_seconds: int = 1800
    variants: tuple[str, ...] = ("F0", "F1", "F2", "F3", "F4")
    alternative_depth: int = 7
    seasonal_policy: str = "positional"

    def __post_init__(self) -> None:
        if self.family not in {"lightgbm", "xgboost"} or self.version != "1.0":
            raise DataError("Unknown boosting family/schema")
        for name in (
            "trees",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "early_stopping",
            "threads",
            "max_train_rows",
            "max_validation_rows",
            "feature_batch",
            "evaluation_batch",
            "trial_seconds",
            "run_seconds",
            "alternative_depth",
        ):
            integer(getattr(self, name), name, 1)
        integer(self.seed, "seed")
        if (
            self.threads > 8
            or self.trees > 2000
            or self.max_depth > 16
            or self.alternative_depth > 16
            or self.max_train_rows > 2000000
            or self.max_validation_rows > 500000
            or self.feature_batch > 65536
            or self.evaluation_batch > 1024
        ):
            raise DataError("Configured local resource ceiling exceeded")
        for name in (
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample",
            "reg_lambda",
            "reg_alpha",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise DataError("Invalid numerical boosting parameter")
        if (
            not 0 < self.learning_rate <= 1
            or not 0 < self.subsample <= 1
            or not 0 < self.colsample <= 1
        ):
            raise DataError("Learning rate and sampling fractions must be in (0,1]")
        if (
            not isinstance(self.variants, tuple)
            or "F0" not in self.variants
            or len(self.variants) != len(set(self.variants))
        ):
            raise DataError("Unique feature variants including F0 required")
        for variant in self.variants:
            FeatureSpec(variant=variant, seasonal_policy=self.seasonal_policy)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def config_from_dict(value: object) -> BoostConfig:
    obj = object_dict(value)
    if set(obj) - {f.name for f in fields(BoostConfig)}:
        raise DataError("Unknown boosting configuration fields")
    d = BoostConfig().to_dict() | obj
    variants = d.pop("variants")
    if not isinstance(variants, list | tuple) or any(not isinstance(v, str) for v in variants):
        raise DataError("Feature variant list required")
    # Explicit reconstruction preserves strict typing and rejects stringified numbers.
    ints = {
        n: integer(d[n], n, 0 if n == "seed" else 1)
        for n in (
            "trees",
            "max_depth",
            "num_leaves",
            "min_child_samples",
            "early_stopping",
            "seed",
            "threads",
            "max_train_rows",
            "max_validation_rows",
            "feature_batch",
            "evaluation_batch",
            "trial_seconds",
            "run_seconds",
            "alternative_depth",
        )
    }
    nums: dict[str, float] = {}
    for name in (
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample",
        "reg_lambda",
        "reg_alpha",
    ):
        v = d[name]
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise DataError("Numerical parameter expected")
        nums[name] = float(v)
    for name in ("family", "version", "seasonal_policy"):
        if not isinstance(d[name], str):
            raise DataError("Text parameter expected")
    return BoostConfig(
        family=str(d["family"]),
        version=str(d["version"]),
        seasonal_policy=str(d["seasonal_policy"]),
        variants=tuple(str(v) for v in variants),
        **ints,
        learning_rate=nums["learning_rate"],
        min_child_weight=nums["min_child_weight"],
        subsample=nums["subsample"],
        colsample=nums["colsample"],
        reg_lambda=nums["reg_lambda"],
        reg_alpha=nums["reg_alpha"],
    )
