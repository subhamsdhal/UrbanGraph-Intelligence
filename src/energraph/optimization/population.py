"""Exact midnight cohorts and historical residual banks, no forecast relabeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from energraph.artifacts import governed_payload
from energraph.regional_data import load_regional_plan
from energraph.uncertainty.validation import validate_partitions
from urbangraph.data import DataError
from urbangraph.datasets.serialization import object_dict, read_json
from urbangraph.evaluation.dataset import identity


@dataclass(frozen=True)
class DecisionInputs:
    point: NDArray[np.float64]
    residuals: NDArray[np.float64]
    issue_ns: int
    targets_ns: NDArray[np.int64]
    historical_available_ns: NDArray[np.int64]

    def scenarios(self) -> tuple[NDArray[np.float64], int]:
        if (
            self.point.shape != (24,)
            or self.targets_ns.shape != (24,)
            or self.residuals.ndim != 2
            or self.residuals.shape[1] != 24
            or len(self.residuals) < 16
            or self.historical_available_ns.shape != (len(self.residuals),)
            or not np.isfinite(self.point).all()
            or not np.isfinite(self.residuals).all()
            or np.any(self.historical_available_ns >= self.issue_ns)
            or self.targets_ns[0] != self.issue_ns
            or np.any(np.diff(self.targets_ns) != 3600 * 10**9)
            or self.issue_ns % (24 * 3600 * 10**9) != 0
        ):
            raise DataError("Invalid or future-contaminated daily scenario context")
        values = self.point[None, :] + self.residuals
        return np.maximum(values, 0), int((values < 0).sum())


def midnight_indices(targets: NDArray[np.int64], truth: NDArray[np.float64]) -> NDArray[np.int64]:
    if (
        targets.ndim != 2
        or targets.shape[1] != 24
        or truth.ndim != 3
        or truth.shape[:2] != targets.shape
    ):
        raise DataError("Daily target alignment mismatch")
    if np.any(np.diff(targets, axis=1) != 3600 * 10**9):
        raise DataError("Non-hourly daily targets")
    selected = np.flatnonzero(
        (targets[:, 0] % (24 * 3600 * 10**9) == 0) & np.isfinite(truth).all(axis=(1, 2))
    )
    starts = targets[selected, 0]
    if np.any(np.diff(starts) < 24 * 3600 * 10**9):
        raise DataError("Duplicate or overlapping dispatch days")
    return np.asarray(selected, dtype=np.int64)


def load_decision_population(
    population: Path,
    *,
    canonical: Path,
    plan: Path,
    point_run: Path,
    raw_snapshot_dir: Path,
) -> tuple[dict[str, NDArray[Any]], dict[str, object]]:
    base = governed_payload(
        population,
        {"predictions.npz", "descriptor.json", "runtime.json"},
        parents=(canonical, plan, point_run),
        kind="uncertainty-population",
    )
    d = read_json(base / "descriptor.json")
    s = object_dict(d["specification"])
    if (
        str(object_dict(s["canonical"])["canonical_id"]) != canonical.name
        or str(s["plan_id"]) != plan.name
        or str(s["point_run"]) != point_run.name
    ):
        raise DataError("Decision population parent identity mismatch")
    data, split, latencies = load_regional_plan(canonical, plan, raw_snapshot_dir=raw_snapshot_dir)
    with np.load(base / "predictions.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    validate_partitions(arrays, data.sensor_ids, max(latencies.values()))
    fold = int(str(s["fold"]))
    train = split.folds[fold].partition("train")
    raw = data.frame.iloc[train.start : train.stop].to_numpy(dtype=np.float64)
    scales = np.nanmedian(raw, axis=0)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise DataError("Invalid training-only median battery scale")
    days = midnight_indices(arrays["test_targets_ns"], arrays["test_truth"])
    cal = midnight_indices(arrays["calibration_targets_ns"], arrays["calibration_truth"])
    if len(cal) < 16:
        raise DataError("Fewer than 16 complete historical trajectories: CVaR blocked")
    bank = np.sort(np.random.default_rng(2029).choice(cal, size=min(32, len(cal)), replace=False))
    available = arrays["calibration_targets_ns"][bank, -1] + max(latencies.values()) * 3600 * 10**9
    if not len(days) or np.any(available >= arrays["test_targets_ns"][days[0], 0]):
        raise DataError("Scenario bank unavailable at first decision")
    arrays["selected_days"] = days[:30]
    arrays["bank_indices"] = bank
    arrays["bank_available_ns"] = available
    arrays["training_medians"] = scales
    records = {
        "source": s,
        "population_snapshot": population.name,
        "regions": list(data.sensor_ids),
        "training_rows": [train.start, train.stop],
        "training_first_utc": str(data.timestamps[train.start]),
        "training_last_utc": str(data.timestamps[train.stop - 1]),
        "training_median_MW": scales.tolist(),
        "training_scale_id": identity(
            {"values": raw.tolist(), "rows": [train.start, train.stop], "plan": plan.name}
        ),
        "eligible_days": len(days),
        "included_days": len(days[:30]),
        "excluded_after_cap": max(0, len(days) - 30),
        "excluded_non_midnight_or_incomplete": len(arrays["test_truth"]) - len(days),
        "scenario_candidates": len(cal),
        "scenario_selected": len(bank),
        "seed": 2029,
        "scenario_indices": bank.tolist(),
        "scenario_target_ns": arrays["calibration_targets_ns"][bank].tolist(),
        "scenario_available_ns": available.tolist(),
        "selected_target_ns": arrays["test_targets_ns"][days[:30]].tolist(),
        "issue_convention": (
            "saved forecast origin is previous 23:00 UTC; dispatch commitment 00:00 UTC"
        ),
        "scenario_bank_id": identity(
            {"population": population.name, "indices": bank.tolist(), "seed": 2029}
        ),
    }
    return arrays, records
