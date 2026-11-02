"""Separate graph architecture and reusable optimization settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict
from urbangraph.neural.contracts import NeuralConfig


@dataclass(frozen=True)
class GraphModelConfig:
    version: str = "stgnn-1.0"
    family: str = "stgcn"
    hidden: int = 16
    blocks: int = 2
    diffusion_depth: int = 2
    embedding_dim: int = 8
    connectivity: str = "static"
    residual: bool = True
    dropout: float = 0.0
    perturbation: str = "none"
    perturbation_seed: int = 2026

    def __post_init__(self) -> None:
        if self.version != "stgnn-1.0" or self.family not in {
            "stgcn",
            "dcrnn",
            "graph-wavenet",
            "urbangraphformer",
        }:
            raise DataError("Unsupported graph architecture")
        for name, high in (
            ("hidden", 64),
            ("blocks", 4),
            ("diffusion_depth", 3),
            ("embedding_dim", 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= high:
                raise DataError(f"Invalid {name}")
        if self.connectivity not in {"static", "adaptive", "static-adaptive"}:
            raise DataError("Invalid connectivity mode")
        if self.family != "graph-wavenet" and self.connectivity != "static":
            raise DataError("Adaptive connectivity requires Graph WaveNet")
        if (
            type(self.residual) is not bool
            or type(self.dropout) not in (int, float)
            or not 0 <= self.dropout < 1
        ):
            raise DataError("Invalid dropout/residual setting")
        if self.perturbation not in {
            "none",
            "identity",
            "remove-edges",
            "unweighted",
            "symmetric-max",
            "permute-weights",
            "randomize-destinations",
        }:
            raise DataError("Unknown graph control")
        if type(self.perturbation_seed) is not int or not 0 <= self.perturbation_seed < 2**32:
            raise DataError("Invalid graph perturbation seed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    family: str = "graph-training"
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
    trial_seconds: int = 1200
    run_seconds: int = 7200
    memory_mb: int = 2048
    scaling: str = "global"
    scheduler: bool = True

    def __post_init__(self) -> None:
        if self.family != "graph-training":
            raise DataError("Invalid training policy schema")
        # Reuse established validation for shared optimization/resource fields only.
        values = self.to_dict()
        values.pop("family")
        NeuralConfig(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def model_config(value: object) -> GraphModelConfig:
    obj = object_dict(value)
    if set(obj) - set(GraphModelConfig().to_dict()):
        raise DataError("Unknown graph model field")
    return GraphModelConfig(**obj)  # type: ignore[arg-type]


def training_config(value: object) -> TrainingConfig:
    obj = object_dict(value)
    if set(obj) - set(TrainingConfig().to_dict()):
        raise DataError("Unknown graph training field")
    return TrainingConfig(**obj)  # type: ignore[arg-type]
