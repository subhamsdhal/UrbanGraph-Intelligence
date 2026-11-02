"""Feature and Prediction Distribution Drift Detection for Urban Graph Pipelines.

Satisfies Requirement R21:
- Non-parametric Population Stability Index (PSI).
- Two-sample Kolmogorov-Smirnov (KS) goodness-of-fit test.
- Configurable thresholding with false-alert sample size suppression.
- Structured drift reports with metric breakdown and alert dispatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DriftAlert:
    """Structured alert record when drift surpasses warning or critical thresholds."""

    feature_name: str
    metric: Literal["PSI", "KS_STATISTIC", "WASSERSTEIN"]
    observed_value: float
    threshold: float
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    timestamp_utc: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftReport:
    """Complete drift assessment report for a batch of inference observations."""

    timestamp_utc: str
    sample_count: int
    is_drifted: bool
    overall_status: Literal["STABLE", "MODERATE_DRIFT", "SEVERE_DRIFT"]
    metrics_by_feature: dict[str, dict[str, float]]
    alerts: list[DriftAlert]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "sample_count": self.sample_count,
            "is_drifted": self.is_drifted,
            "overall_status": self.overall_status,
            "metrics_by_feature": self.metrics_by_feature,
            "alerts": [a.to_dict() for a in self.alerts],
        }


def compute_psi(reference: FloatArray, current: FloatArray, num_bins: int = 10, epsilon: float = 1e-4) -> float:
    """Calculate Population Stability Index (PSI) with quantile binning and epsilon smoothing.

    PSI < 0.1: No significant change
    0.1 <= PSI < 0.25: Moderate change / monitoring required
    PSI >= 0.25: Significant drift / model recalibration triggered
    """
    if len(reference) < 10 or len(current) < 10:
        raise DataError("PSI requires at least 10 samples in both reference and current distributions")

    # Quantile bins derived strictly from reference baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure strictly monotonic bin edges
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    for i in range(1, len(bin_edges) - 1):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1e-6

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Fractions with smoothing
    ref_fractions = (ref_counts + epsilon) / (len(reference) + num_bins * epsilon)
    curr_fractions = (curr_counts + epsilon) / (len(current) + num_bins * epsilon)

    psi_val = np.sum((curr_fractions - ref_fractions) * np.log(curr_fractions / ref_fractions))
    return float(max(0.0, psi_val))


def compute_ks(reference: FloatArray, current: FloatArray) -> float:
    """Compute empirical 2-sample Kolmogorov-Smirnov distance."""
    if len(reference) == 0 or len(current) == 0:
        raise DataError("KS test requires non-empty arrays")
    ref_sorted = np.sort(reference)
    curr_sorted = np.sort(current)

    all_vals = np.concatenate([ref_sorted, curr_sorted])
    cdf_ref = np.searchsorted(ref_sorted, all_vals, side="right") / len(ref_sorted)
    cdf_curr = np.searchsorted(curr_sorted, all_vals, side="right") / len(curr_sorted)
    return float(np.max(np.abs(cdf_ref - cdf_curr)))


class FeatureDriftDetector:
    """Monitors feature vectors and model predictions against baseline distributions."""

    def __init__(
        self,
        reference_data: dict[str, FloatArray],
        psi_warning_threshold: float = 0.15,
        psi_critical_threshold: float = 0.25,
        ks_warning_threshold: float = 0.15,
        min_batch_size: int = 20,
    ) -> None:
        if not reference_data:
            raise DataError("Reference data must contain at least one feature distribution")
        self.reference_data = {k: np.asarray(v, dtype=np.float64) for k, v in reference_data.items()}
        for k, v in self.reference_data.items():
            if len(v) < min_batch_size:
                raise DataError(f"Reference feature {k} has fewer than minimum samples ({min_batch_size})")

        self.psi_warning = psi_warning_threshold
        self.psi_critical = psi_critical_threshold
        self.ks_warning = ks_warning_threshold
        self.min_batch_size = min_batch_size

    def check_drift(self, current_data: dict[str, FloatArray]) -> DriftReport:
        """Compare current inference batch against reference baseline."""
        now_str = datetime.now(UTC).isoformat()
        sample_counts = [len(v) for v in current_data.values()]
        min_curr_len = min(sample_counts) if sample_counts else 0

        # Suppression of false alerts on small batches
        if min_curr_len < self.min_batch_size:
            return DriftReport(
                timestamp_utc=now_str,
                sample_count=min_curr_len,
                is_drifted=False,
                overall_status="STABLE",
                metrics_by_feature={},
                alerts=[
                    DriftAlert(
                        feature_name="ALL",
                        metric="PSI",
                        observed_value=0.0,
                        threshold=self.psi_warning,
                        severity="INFO",
                        timestamp_utc=now_str,
                        message=f"Batch size ({min_curr_len}) below threshold ({self.min_batch_size}); alert suppressed.",
                    )
                ],
            )

        metrics_map: dict[str, dict[str, float]] = {}
        alerts: list[DriftAlert] = []

        max_severity: Literal["STABLE", "MODERATE_DRIFT", "SEVERE_DRIFT"] = "STABLE"

        for feature_name, curr_arr in current_data.items():
            if feature_name not in self.reference_data:
                continue

            ref_arr = self.reference_data[feature_name]
            curr_np = np.asarray(curr_arr, dtype=np.float64)

            psi = compute_psi(ref_arr, curr_np)
            ks = compute_ks(ref_arr, curr_np)

            metrics_map[feature_name] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks, 4),
            }

            if psi >= self.psi_critical:
                alerts.append(
                    DriftAlert(
                        feature_name=feature_name,
                        metric="PSI",
                        observed_value=psi,
                        threshold=self.psi_critical,
                        severity="CRITICAL",
                        timestamp_utc=now_str,
                        message=f"Critical distribution drift on {feature_name}: PSI={psi:.3f} >= {self.psi_critical}",
                    )
                )
                max_severity = "SEVERE_DRIFT"
            elif psi >= self.psi_warning or ks >= self.ks_warning:
                alerts.append(
                    DriftAlert(
                        feature_name=feature_name,
                        metric="PSI" if psi >= self.psi_warning else "KS_STATISTIC",
                        observed_value=psi if psi >= self.psi_warning else ks,
                        threshold=self.psi_warning if psi >= self.psi_warning else self.ks_warning,
                        severity="WARNING",
                        timestamp_utc=now_str,
                        message=f"Moderate drift on {feature_name}: PSI={psi:.3f}, KS={ks:.3f}",
                    )
                )
                if max_severity != "SEVERE_DRIFT":
                    max_severity = "MODERATE_DRIFT"

        return DriftReport(
            timestamp_utc=now_str,
            sample_count=min_curr_len,
            is_drifted=(len(alerts) > 0),
            overall_status=max_severity,
            metrics_by_feature=metrics_map,
            alerts=alerts,
        )
