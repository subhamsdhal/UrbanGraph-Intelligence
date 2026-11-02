"""Reusable masked AdamW training with validation-only selection and epoch resume."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler

from urbangraph.baselines.scoring import MetricAccumulator
from urbangraph.data import DataError
from urbangraph.evaluation.dataset import identity
from urbangraph.neural.checkpointing import load_checkpoint, save_checkpoint
from urbangraph.neural.contracts import NeuralConfig
from urbangraph.neural.dataloaders import initialize, make_loader
from urbangraph.neural.datasets import NeuralWindows
from urbangraph.neural.losses import masked_loss
from urbangraph.neural.models import TemporalForecaster
from urbangraph.neural.training_policy import TrainingPolicy

LOGGER = logging.getLogger("urbangraph.neural")


class BudgetExceeded(DataError):
    """Partial training is not a completed research run."""


@dataclass
class FitResult:
    model: TemporalForecaster
    checkpoint: Path
    report: dict[str, object]


def validate(model: nn.Module, dataset: NeuralWindows, config: TrainingPolicy) -> dict[str, object]:
    score = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for x, _, mask, origins in make_loader(dataset, config):
            prediction = model(x.to(config.device)).cpu().numpy().astype(np.float64)[..., 0]
            truth = dataset.raw_targets(origins.numpy())
            prediction = dataset.scaler.inverse_transform(prediction, sensor_ids=dataset.sensor_ids)
            score.add(truth, prediction, mask.numpy()[..., 0])
    return score.result()


ModelT = TypeVar("ModelT", bound=nn.Module)


@dataclass
class TrainingResult(Generic[ModelT]):
    model: ModelT
    checkpoint: Path
    report: dict[str, object]


def fit(
    train: NeuralWindows,
    validation: NeuralWindows,
    config: NeuralConfig,
    checkpoint_root: Path,
    lineage: dict[str, object],
    *,
    resume: Path | None = None,
    deadline: float | None = None,
) -> FitResult:
    """Preserved UG-003C API and exact default model/checkpoint metadata."""
    result = fit_model(
        train,
        validation,
        config,
        checkpoint_root,
        lineage,
        model_factory=lambda: TemporalForecaster(
            config, train.history, train.horizon, train.sensor_ids, 4 if config.calendar else 2
        ),
        resume=resume,
        deadline=deadline,
    )
    return FitResult(result.model, result.checkpoint, result.report)


def fit_model(
    train: NeuralWindows,
    validation: NeuralWindows,
    config: TrainingPolicy,
    checkpoint_root: Path,
    lineage: dict[str, object],
    *,
    model_factory: Callable[[], ModelT],
    validation_function: Callable[[nn.Module, NeuralWindows, TrainingPolicy], dict[str, object]]
    | None = None,
    validation_identity: str | None = None,
    resume: Path | None = None,
    deadline: float | None = None,
) -> TrainingResult[ModelT]:
    if (validation_function is None) != (validation_identity is None):
        raise DataError("Custom validation requires a recorded objective identity")
    started = time.perf_counter()
    limit = min(deadline if deadline is not None else float("inf"), started + config.trial_seconds)
    device = initialize(config)
    model = model_factory().to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        if config.scheduler
        else None
    )
    amp = GradScaler("cuda", enabled=config.precision == "float16")
    best_mae, best_epoch, bad_epochs = float("inf"), 0, 0
    best_model = copy.deepcopy(model.state_dict())
    history: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "config": config.to_dict(),
        "lineage": lineage,
        "training_population": train.report,
        "validation_population": validation.report,
        "scaler": train.scaler.to_dict(),
        "sensor_ids": list(train.sensor_ids),
        "history_length": train.history,
        "horizon": train.horizon,
        "device": device,
    }
    if validation_identity is not None:
        metadata["validation_identity"] = validation_identity
    first_epoch = 1
    last_checkpoint: Path | None = None
    if resume is not None:
        state, recorded = load_checkpoint(resume)
        expected = {
            k: v
            for k, v in recorded.items()
            if k not in {"schema_version", "epoch", "state_sha256", "checkpoint_id"}
        }
        if identity(metadata) != identity(expected):
            raise DataError("Resume configuration/data/scaler/code mismatch")
        model.load_state_dict(cast(dict[str, Tensor], state["model"]))
        optimizer.load_state_dict(cast(dict[str, object], state["optimizer"]))
        if scheduler is not None:
            scheduler.load_state_dict(cast(dict[str, object], state["scheduler"]))
        amp.load_state_dict(cast(dict[str, object], state["amp"]))
        torch.set_rng_state(cast(Tensor, state["torch_rng"]))
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(cast(list[Tensor], state["cuda_rng"]))
        best_model = cast(dict[str, Tensor], state["best_model"])
        best_mae = float(cast(float, state["best_mae"]))
        best_epoch, bad_epochs = (
            int(cast(int, state["best_epoch"])),
            int(cast(int, state["bad_epochs"])),
        )
        history = cast(list[dict[str, object]], state["history"])
        first_epoch = int(cast(int, recorded["epoch"])) + 1
        last_checkpoint = resume
    for epoch in range(first_epoch, config.epochs + 1):
        if bad_epochs >= config.patience:
            break
        if time.perf_counter() >= limit:
            raise BudgetExceeded("Neural trial deadline reached; completed epochs remain resumable")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader = make_loader(train, config, epoch=epoch, training=True)
        total_loss, total_count, group_count = 0.0, 0, 0
        epoch_started = time.perf_counter()
        for batch, (x, y, mask, _) in enumerate(loader):
            if time.perf_counter() >= limit:
                raise BudgetExceeded(
                    "Neural trial interrupted by time budget; no partial result published"
                )
            x, y, mask = x.to(config.device), y.to(config.device), mask.to(config.device)
            count = int(mask.sum().item())
            if count == 0:
                raise DataError("All-invalid training batch; stopping without optimizer update")
            with torch.autocast(
                device_type=config.device,
                dtype=torch.float16,
                enabled=config.precision == "float16",
            ):
                prediction = model(x)
                loss = masked_loss(prediction, y, mask, config.loss)
            torch.autograd.backward(amp.scale(loss * count))
            total_loss += float(loss.detach().item()) * count
            total_count += count
            group_count += count
            if (batch + 1) % config.accumulation == 0 or batch + 1 == len(loader):
                amp.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_count)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_norm, error_if_nonfinite=True
                )
                amp.step(optimizer)
                amp.update()
                optimizer.zero_grad(set_to_none=True)
                group_count = 0
        measured = (validation_function or validate)(model, validation, config)
        item = measured["mae"]
        assert isinstance(item, dict) and isinstance(item["value"], float)
        mae = item["value"]
        if mae < best_mae:
            best_mae, best_epoch, bad_epochs = mae, epoch, 0
            best_model = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        if scheduler is not None:
            scheduler.step(mae)
        history.append(
            {
                "epoch": epoch,
                "training_loss_normalized": total_loss / total_count,
                "validation": measured,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        state_out: dict[str, object] = {
            "model": model.state_dict(),
            "best_model": best_model,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else {},
            "amp": amp.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
            "best_mae": best_mae,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
        }
        last_checkpoint = save_checkpoint(checkpoint_root, epoch, state_out, metadata)
        LOGGER.info("%s epoch %d validation MAE %.6f", config.family, epoch, mae)
    if last_checkpoint is None or not np.isfinite(best_mae):
        raise DataError("No completed neural epoch")
    if time.perf_counter() > limit:
        raise BudgetExceeded(
            "Neural trial exceeded time budget after checkpoint; result incomplete"
        )
    model.load_state_dict(best_model)
    model.eval()
    return TrainingResult(
        model,
        last_checkpoint,
        {
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_mae": best_mae,
            "training_seconds": time.perf_counter() - started,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": device,
            "resumed": resume is not None,
            "checkpoint_id": load_checkpoint(last_checkpoint)[1]["checkpoint_id"],
        },
    )
