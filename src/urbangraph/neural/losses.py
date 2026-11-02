"""Losses sanitize invalid labels before arithmetic, preserving zero-valid semantics."""

import torch
from torch import Tensor

from urbangraph.data import DataError


def masked_loss(prediction: Tensor, target: Tensor, mask: Tensor, kind: str = "mae") -> Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape or mask.dtype != torch.bool:
        raise DataError("Loss shape/mask mismatch")
    valid = mask & torch.isfinite(target)
    if not bool(valid.any()):
        raise DataError("All-invalid target batch; optimizer step forbidden")
    if not bool(torch.isfinite(prediction).all()):
        raise DataError("Nonfinite predictions")
    selected_prediction, selected_target = prediction[valid], target[valid]
    difference = selected_prediction - selected_target
    if kind == "mae":
        return difference.abs().mean()
    if kind == "mse":
        return difference.square().mean()
    if kind == "huber":
        absolute = difference.abs()
        return torch.where(absolute <= 1, difference.square() / 2, absolute - 0.5).mean()
    raise DataError("Unsupported loss")
