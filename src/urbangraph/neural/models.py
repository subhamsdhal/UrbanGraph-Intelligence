"""Shared per-sensor temporal encoders; no sensor attention or graph information."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor, nn

from urbangraph.data import DataError
from urbangraph.neural.contracts import NeuralConfig


class TemporalForecaster(nn.Module):
    def __init__(
        self,
        config: NeuralConfig,
        history: int,
        horizon: int,
        sensor_ids: tuple[str, ...],
        features: int,
    ) -> None:
        super().__init__()
        if history < 1 or horizon < 1 or not sensor_ids or len(set(sensor_ids)) != len(sensor_ids):
            raise DataError("Invalid model temporal/sensor schema")
        if features != (4 if config.calendar else 2):
            raise DataError("Input feature schema mismatch")
        self.config = config
        self.history, self.horizon = history, horizon
        self.sensor_ids, self.features = sensor_ids, features
        self.projection = nn.Linear(features, config.hidden)
        self.encoder: nn.GRU | nn.LSTM | nn.TransformerEncoder
        if config.family == "transformer":
            layer = nn.TransformerEncoderLayer(
                config.hidden,
                config.heads,
                config.feedforward,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
            positions = torch.arange(history, dtype=torch.float32)[:, None]
            scales = torch.exp(
                torch.arange(0, config.hidden, 2) * (-math.log(10000) / config.hidden)
            )
            encoding = torch.zeros(history, config.hidden)
            encoding[:, 0::2] = torch.sin(positions * scales)
            encoding[:, 1::2] = torch.cos(positions * scales)
            self.register_buffer("position", encoding)
        else:
            recurrent = nn.GRU if config.family == "gru" else nn.LSTM
            self.encoder = recurrent(
                config.hidden,
                config.hidden,
                config.layers,
                batch_first=True,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
        self.normalization = nn.LayerNorm(config.hidden) if config.layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden, horizon)

    def forward(self, x: Tensor, *, sensor_ids: tuple[str, ...] | None = None) -> Tensor:
        if (
            x.ndim != 4
            or x.shape[1:] != (self.history, len(self.sensor_ids), self.features)
            or (sensor_ids is not None and sensor_ids != self.sensor_ids)
        ):
            raise DataError("Expected aligned [batch,history,sensors,features] input")
        if not bool(torch.isfinite(x).all()):
            raise DataError("Model input must be sanitized with explicit validity indicators")
        b, t, n, f = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(b * n, t, f)
        encoded = self.projection(sequence)
        if isinstance(self.encoder, nn.TransformerEncoder):
            padding = sequence[:, :, 1] == 0
            # All-missing sensors retain one train-mean token, preventing all-masked attention NaNs.
            padding = padding.clone()
            padding[padding.all(dim=1), -1] = False
            encoded = self.encoder(encoded + self.position, src_key_padding_mask=padding)
        else:
            encoded, _ = self.encoder(
                encoded
            )  # recurrent state initialized independently each call
        forecast = self.head(self.dropout(self.normalization(encoded[:, -1])))
        if self.config.residual:
            # Preprocessing forward-fills only within observed history; no future reference.
            forecast = forecast + sequence[:, -1, 0, None]
        return cast(Tensor, forecast.reshape(b, n, self.horizon).permute(0, 2, 1).unsqueeze(-1))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
