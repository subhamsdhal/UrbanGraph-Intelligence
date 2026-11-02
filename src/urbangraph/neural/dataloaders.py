"""Controlled seeds, device selection and Windows-safe loader construction."""

from __future__ import annotations

import os
import random
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from urbangraph.data import DataError
from urbangraph.neural.datasets import NeuralSample, NeuralWindows, collate
from urbangraph.neural.training_policy import TrainingPolicy


def initialize(config: TrainingPolicy) -> dict[str, object]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise DataError("CUDA requested but unavailable; explicitly select CPU configuration")
        total = torch.cuda.get_device_properties(0).total_memory
        if config.memory_mb * 1024**2 > total * 0.8:
            raise DataError("GPU memory budget exceeds safe 80 percent ceiling")
        torch.cuda.set_per_process_memory_fraction(config.memory_mb * 1024**2 / total)
    return {
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": config.device,
        "name": torch.cuda.get_device_name(0) if config.device == "cuda" else "CPU",
        "seed": config.seed,
        "threads": config.threads,
        "workers": config.workers,
        "deterministic": config.deterministic,
        "precision": config.precision,
        "cudnn_benchmark": False,
        "tf32": False,
        "training_order": "epoch-seeded shuffled selected windows",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def make_loader(
    dataset: NeuralWindows, config: TrainingPolicy, *, epoch: int = 0, training: bool = False
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(config.seed + epoch)
    loader: DataLoader[NeuralSample] = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.workers,
        pin_memory=config.device == "cuda",
        generator=generator,
        collate_fn=collate,
        drop_last=False,
    )

    # PyTorch parameterizes loaders by sample type, ignoring collate_fn output.
    return cast("DataLoader[tuple[Tensor, Tensor, Tensor, Tensor]]", loader)
