"""Strict versioned architecture, training and resource configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict


@dataclass(frozen=True)
class NeuralConfig:
    version: str = "neural-1.0"
    family: str = "gru"
    hidden: int = 32
    layers: int = 1
    heads: int = 4
    feedforward: int = 64
    dropout: float = 0.0
    layer_norm: bool = True
    residual: bool = True
    calendar: bool = False
    epochs: int = 12
    patience: int = 3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    loss: str = "mae"
    clip_norm: float = 1.0
    accumulation: int = 1
    batch_size: int = 8
    max_train_windows: int = 2048
    max_validation_windows: int = 512
    threads: int = 2
    workers: int = 0
    seed: int = 2026
    device: str = "cpu"
    precision: str = "float32"
    deterministic: bool = True
    trial_seconds: int = 600
    run_seconds: int = 3600
    memory_mb: int = 2048
    scaling: str = "global"
    scheduler: bool = True
    search_residual: bool = True

    def __post_init__(self) -> None:
        if self.version != "neural-1.0" or self.family not in {"gru", "lstm", "transformer"}:
            raise DataError("Unsupported neural architecture/schema")
        for name in (
            "hidden",
            "layers",
            "heads",
            "feedforward",
            "epochs",
            "patience",
            "batch_size",
            "max_train_windows",
            "max_validation_windows",
            "threads",
            "accumulation",
            "trial_seconds",
            "run_seconds",
            "memory_mb",
        ):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise DataError(f"{name} requires a positive integer")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.workers) is not int
            or self.workers != 0
        ):
            raise DataError("Nonnegative seed and Windows-safe zero workers required in v1")
        for name in (
            "layer_norm",
            "residual",
            "calendar",
            "deterministic",
            "scheduler",
            "search_residual",
        ):
            if type(getattr(self, name)) is not bool:
                raise DataError(f"{name} requires a boolean")
        for name in ("dropout", "learning_rate", "weight_decay", "clip_norm"):
            v = getattr(self, name)
            if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
                raise DataError(f"Invalid {name}")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.clip_norm <= 0:
            raise DataError("Invalid optimization range")
        if (
            self.hidden > 256
            or self.feedforward > 1024
            or self.max_validation_windows > 50000
            or self.layers > 4
            or self.epochs > 100
            or self.threads > 8
            or self.batch_size > 64
            or self.max_train_windows > 50000
            or self.memory_mb > 8192
        ):
            raise DataError("Local neural resource ceiling exceeded")
        if self.family == "transformer" and (self.hidden % self.heads or self.hidden % 2):
            raise DataError("Transformer hidden width must be even and divisible by heads")
        if self.device not in {"cpu", "cuda"} or self.precision not in {"float32", "float16"}:
            raise DataError("Unsupported device or precision")
        if self.precision == "float16" and self.device != "cuda":
            raise DataError("Mixed precision requires CUDA")
        if self.scaling not in {"global", "per_sensor"} or self.loss not in {"mae", "mse", "huber"}:
            raise DataError("Unsupported preprocessing/loss")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def config_from_dict(value: object) -> NeuralConfig:
    obj = object_dict(value)
    defaults = NeuralConfig().to_dict()
    if set(obj) - set(defaults):
        raise DataError("Unknown neural configuration field")
    merged = defaults | obj
    # Dataclass construction is a narrow, runtime-validated JSON boundary.
    config = NeuralConfig(**merged)  # type: ignore[arg-type]
    if len(fields(config)) != len(merged):
        raise DataError("Incomplete neural configuration")
    return config
