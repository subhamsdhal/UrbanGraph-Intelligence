"""Binary HiGHS MILP for shared schedules and incremental scenario-loss CVaR."""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib.metadata import version

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]

from energraph.optimization.physical import (
    BatterySpecification,
    DispatchSchedule,
    SyntheticTariff,
    check_schedule,
)
from urbangraph.data import DataError


@dataclass(frozen=True)
class RiskPolicy:
    alpha: float = 0.9
    weight: float = 0.5

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.alpha, self.weight]).all()
            or not 0 < self.alpha < 1
            or self.weight < 0
        ):
            raise DataError("Invalid CVaR policy")


def empirical_cvar(loss: NDArray[np.float64], weights: NDArray[np.float64], alpha: float) -> float:
    if (
        loss.ndim != 1
        or not len(loss)
        or weights.shape != loss.shape
        or not np.isfinite(loss).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
        or not 0 < alpha < 1
    ):
        raise DataError("Invalid weighted loss distribution")
    return min(float(z + weights @ np.maximum(loss - z, 0) / (1 - alpha)) for z in loss)


def optimize(
    scenarios: NDArray[np.float64],
    battery: BatterySpecification,
    tariff: SyntheticTariff,
    *,
    risk: RiskPolicy | None = None,
    time_limit: float = 10.0,
) -> tuple[DispatchSchedule, dict[str, object]]:
    if (
        scenarios.ndim != 2
        or scenarios.shape[1] != 24
        or not len(scenarios)
        or not np.isfinite(scenarios).all()
        or np.any(scenarios < 0)
        or not np.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise DataError("Invalid optimization scenarios or time budget")
    risk = RiskPolicy(weight=0) if risk is None else risk
    n = len(scenarios)
    weights = np.full(n, 1 / n)
    # charge 0:24, discharge 24:48, SoC 48:73, binary 73:97,
    # scenario peak 97:97+n, z 97+n, excess 98+n:98+2n.
    zidx = 97 + n
    size = 98 + 2 * n
    objective = np.zeros(size)
    unit_c = tariff.prices() + tariff.degradation
    unit_d = -tariff.prices() + tariff.degradation
    objective[:24] = unit_c
    objective[24:48] = unit_d
    objective[97 : 97 + n] = tariff.peak * weights
    objective[zidx] = risk.weight
    objective[zidx + 1 :] = risk.weight * weights / (1 - risk.alpha)
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[:48] = battery.power
    lower[48:73] = 0.1 * battery.capacity
    upper[48:73] = battery.capacity
    lower[48] = upper[48] = lower[72] = upper[72] = 0.5 * battery.capacity
    upper[73:97] = 1
    lower[zidx] = -np.inf
    integrality = np.zeros(size, dtype=np.int32)
    integrality[73:97] = 1
    rows = []
    lo = []
    hi = []

    def add(entries: dict[int, float], a: float = -np.inf, b: float = np.inf) -> None:
        row = np.zeros(size)
        for i, value in entries.items():
            row[i] = value
        rows.append(row)
        lo.append(a)
        hi.append(b)

    for t in range(24):
        add({49 + t: 1, 48 + t: -1, t: -battery.eta, 24 + t: 1 / battery.eta}, 0, 0)
        add({t: 1, 73 + t: -battery.power}, b=0)
        add({24 + t: 1, 73 + t: battery.power}, b=battery.power)
        add({t: 1, 24 + t: -1}, a=-float(scenarios[:, t].min()))
        for s in range(n):
            add({t: 1, 24 + t: -1, 97 + s: -1}, b=-float(scenarios[s, t]))
    baseline_peak = tariff.peak * scenarios.max(axis=1)
    for s in range(n):
        entries = {i: float(v) for i, v in enumerate(np.r_[unit_c, unit_d])}
        entries.update({97 + s: tariff.peak, zidx: -1, zidx + 1 + s: -1})
        add(entries, b=float(baseline_peak[s]))
    began = time.perf_counter()
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi)),
        options={"time_limit": time_limit, "mip_rel_gap": 1e-7},
    )
    elapsed = time.perf_counter() - began
    metadata: dict[str, object] = {
        "solver": "SciPy milp / HiGHS",
        "scipy_version": version("scipy"),
        "status": int(result.status),
        "message": str(result.message),
        "seconds": elapsed,
        "variables": size,
        "binary_variables": 24,
        "constraints": len(rows),
        "time_limit_seconds": time_limit,
        "mip_rel_gap_target": 1e-7,
        "mip_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else None,
        "fallback": False,
        "scenario_count": n,
        "alpha": risk.alpha,
        "lambda": risk.weight,
    }
    if not result.success or result.x is None:
        metadata["fallback"] = True
        metadata["fallback_reason"] = "solver did not establish success; zero dispatch"
        return DispatchSchedule.zero(), metadata
    x = np.asarray(result.x, dtype=np.float64)
    schedule = DispatchSchedule(x[:24].copy(), x[24:48].copy())
    checks = [check_schedule(schedule, battery, s) for s in scenarios]
    if not all(c["feasible"] for c in checks):
        metadata["fallback"] = True
        metadata["fallback_reason"] = "independent physical validation rejected solver output"
        return DispatchSchedule.zero(), metadata
    # Explicitly recompute actual peak rather than trusting nonbinding epigraph variables.
    increments = (
        unit_c @ schedule.charge
        + unit_d @ schedule.discharge
        + tariff.peak * (scenarios + schedule.charge - schedule.discharge).max(axis=1)
        - baseline_peak
    )
    metadata["expected_incremental_CU"] = float(weights @ increments)
    metadata["cvar_incremental_CU"] = empirical_cvar(increments, weights, risk.alpha)
    metadata["objective_incremental_CU"] = float(weights @ increments) + risk.weight * float(
        str(metadata["cvar_incremental_CU"])
    )
    metadata["solver_objective_incremental_CU"] = float(result.fun - weights @ baseline_peak)
    metadata["max_constraint_violation"] = float(
        max(
            0,
            np.max(np.asarray(lo) - np.asarray(rows) @ x),
            np.max(np.asarray(rows) @ x - np.asarray(hi)),
        )
    )
    return schedule, metadata
