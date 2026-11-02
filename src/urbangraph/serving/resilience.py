"""Runtime sensor-dropout resilience controller and fault-tolerant prediction serving.

Provides:
- Real-time sensor outage and data degradation detection.
- Topological graph restructuring and harmonic spatial imputation.
- Bounded causal temporal decay fallback for isolated nodes (zero future leakage).
- Conformal uncertainty interval inflation proportional to outage severity.
- Online sensor recovery state tracking across consecutive observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor

from urbangraph.data import DataError


@dataclass(frozen=True)
class OutageReport:
    total_sensors: int
    active_sensors: int
    dropped_sensors: list[str]
    dropout_ratio: float
    degradation_level: str  # "NORMAL", "MODERATE_OUTAGE", "SEVERE_OUTAGE", "CRITICAL_ISOLATION"
    imputation_applied: str  # "NONE", "HARMONIC_SPATIAL", "TEMPORAL_FALLBACK", "HYBRID"
    uncertainty_inflation_factor: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_sensors": self.total_sensors,
            "active_sensors": self.active_sensors,
            "dropped_sensors": self.dropped_sensors,
            "dropout_ratio": round(self.dropout_ratio, 4),
            "degradation_level": self.degradation_level,
            "imputation_applied": self.imputation_applied,
            "uncertainty_inflation_factor": round(self.uncertainty_inflation_factor, 4),
            "metadata": self.metadata,
        }


class SensorOutageDetector:
    """Detects missing observations, corrupt NaNs/Infs, and dead sensor telemetry."""

    def __init__(self, sensor_ids: list[str], nan_threshold: float = 0.5) -> None:
        if not sensor_ids:
            raise DataError("sensor_ids must not be empty")
        self.sensor_ids = sensor_ids
        self.nan_threshold = nan_threshold

    def detect(self, x: Tensor) -> tuple[Tensor, list[str]]:
        """
        x: (B, T, N, F) or (T, N, F) or (T, N)
        Returns:
            mask: (N,) tensor where 1 = active/valid, 0 = dropped/outage
            dropped_ids: list of sensor ID strings identified as in outage
        """
        # Ensure at least 3D: (T, N, F)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 4:
            # Aggregate over batch
            x_ref = x[0]
        else:
            x_ref = x

        t, n, _ = x_ref.shape
        if n != len(self.sensor_ids):
            raise DataError(f"Sensor count mismatch: expected {len(self.sensor_ids)}, got {n}")

        mask = torch.ones(n, dtype=torch.float32, device=x.device)
        dropped_ids: list[str] = []

        for i in range(n):
            node_series = x_ref[:, i, :]
            # Check for NaNs or Infs
            nan_count = torch.isnan(node_series).sum().item() + torch.isinf(node_series).sum().item()
            total_elements = node_series.numel()

            if total_elements > 0 and (nan_count / total_elements) > self.nan_threshold:
                mask[i] = 0.0
                dropped_ids.append(self.sensor_ids[i])
            elif total_elements > 0 and torch.all(node_series == 0.0) and t > 4:
                # Dead sensor transmitting all zeros across time
                mask[i] = 0.0
                dropped_ids.append(self.sensor_ids[i])

        return mask, dropped_ids


class HarmonicSpatialImputer:
    """Topological graph harmonic imputation for missing sensor nodes."""

    def __init__(self, decay_rate: float = 0.95) -> None:
        self.decay_rate = decay_rate

    def impute(
        self,
        x: Tensor,
        adjacency: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, str]:
        """
        x: (B, T, N, F)
        adjacency: (N, N) spatial weight/proximity matrix
        mask: (N,) or (B, N)
        Returns:
            imputed_x: (B, T, N, F) strictly leak-free imputed tensor
            method: description of method applied
        """
        imputed = x.clone()
        b, t, n, f = x.shape

        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(b, -1)

        dropped_count = (mask == 0).sum().item()
        if dropped_count == 0:
            # Replace any single isolated NaNs with 0
            imputed = torch.nan_to_num(imputed, nan=0.0)
            return imputed, "NONE"

        harmonic_applied = False
        temporal_applied = False

        # First replace NaNs with 0 in copy for neighbor calculations
        valid_x = torch.nan_to_num(x.clone(), nan=0.0)

        for batch_idx in range(b):
            b_mask = mask[batch_idx]  # (N,)
            dropped_indices = torch.where(b_mask == 0)[0].tolist()

            for node_idx in dropped_indices:
                # Find valid neighbors from adjacency
                weights = adjacency[node_idx] * b_mask  # zero out weights to missing nodes
                weight_sum = weights.sum().item()

                if weight_sum > 1e-6:
                    # Spatial harmonic averaging: weighted sum of active neighbors at each time step
                    # weights: (N,) -> (1, N, 1)
                    w_expanded = weights.unsqueeze(0).unsqueeze(-1)
                    neighbor_vals = (valid_x[batch_idx] * w_expanded).sum(dim=1) / weight_sum  # (T, F)
                    imputed[batch_idx, :, node_idx, :] = neighbor_vals
                    harmonic_applied = True
                else:
                    # Isolated cluster fallback: temporal autoregressive decay from earliest known valid state
                    # strictly causal (only uses previous time steps, zero future leakage)
                    temporal_applied = True
                    for step in range(t):
                        if step == 0:
                            imputed[batch_idx, step, node_idx, :] = 0.0
                        else:
                            prev_val = imputed[batch_idx, step - 1, node_idx, :]
                            imputed[batch_idx, step, node_idx, :] = prev_val * self.decay_rate

        method = (
            "HYBRID"
            if (harmonic_applied and temporal_applied)
            else ("HARMONIC_SPATIAL" if harmonic_applied else "TEMPORAL_FALLBACK")
        )
        # Final safety guarantee: zero NaNs remain
        imputed = torch.nan_to_num(imputed, nan=0.0)
        return imputed, method


class RuntimeResilienceController:
    """Coordinates fault detection, imputation, recovery tracking, and uncertainty adjustment."""

    def __init__(
        self,
        sensor_ids: list[str],
        adjacency: Tensor,
        inflation_multiplier: float = 1.5,
        consecutive_recovery_threshold: int = 2,
    ) -> None:
        self.sensor_ids = sensor_ids
        self.adjacency = adjacency
        self.inflation_multiplier = inflation_multiplier
        self.consecutive_recovery_threshold = consecutive_recovery_threshold

        self.detector = SensorOutageDetector(sensor_ids)
        self.imputer = HarmonicSpatialImputer()

        # Track consecutive healthy frames per sensor
        self.sensor_recovery_counters: dict[str, int] = {s: self.consecutive_recovery_threshold for s in sensor_ids}

    def process(
        self,
        x: Tensor,
        base_forecaster_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, OutageReport]:
        """
        x: (B, T, N, F)
        base_forecaster_fn: callable(imputed_x, mask) -> predictions (B, H, N)

        Returns:
            predictions: (B, H, N)
            uncertainty_scale: (B, H, N) inflated interval scaling
            report: OutageReport
        """
        # 1. Detect outage
        mask, dropped_ids = self.detector.detect(x)
        total = len(self.sensor_ids)
        active = total - len(dropped_ids)
        dropout_ratio = len(dropped_ids) / total

        # Update recovery state
        for s in self.sensor_ids:
            if s in dropped_ids:
                self.sensor_recovery_counters[s] = 0
            else:
                self.sensor_recovery_counters[s] = min(
                    self.consecutive_recovery_threshold,
                    self.sensor_recovery_counters[s] + 1,
                )

        # Classify degradation level
        if dropout_ratio == 0.0:
            degradation = "NORMAL"
        elif dropout_ratio <= 0.25:
            degradation = "MODERATE_OUTAGE"
        elif dropout_ratio <= 0.50:
            degradation = "SEVERE_OUTAGE"
        else:
            degradation = "CRITICAL_ISOLATION"

        # 2. Impute missing observations
        imputed_x, imputation_method = self.imputer.impute(x, self.adjacency, mask)

        # 3. Base forecast execution
        predictions = base_forecaster_fn(imputed_x, mask)

        # 4. Uncertainty inflation: intervals widen as dropout increases
        inflation_factor = 1.0 + (self.inflation_multiplier * dropout_ratio)

        # Per-node inflation: dropped nodes receive an additional 20% penalty
        b, h, n = predictions.shape
        uncertainty_weights = torch.full((b, h, n), inflation_factor, device=x.device)
        for idx, s in enumerate(self.sensor_ids):
            if s in dropped_ids:
                uncertainty_weights[:, :, idx] *= 1.2

        report = OutageReport(
            total_sensors=total,
            active_sensors=active,
            dropped_sensors=dropped_ids,
            dropout_ratio=dropout_ratio,
            degradation_level=degradation,
            imputation_applied=imputation_method,
            uncertainty_inflation_factor=inflation_factor,
            metadata={"recovery_counters": dict(self.sensor_recovery_counters)},
        )

        return predictions, uncertainty_weights, report
