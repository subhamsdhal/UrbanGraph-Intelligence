"""Synthetic MW/MWh/CU contracts and independent schedule evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from urbangraph.data import DataError

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class BatterySpecification:
    training_median_MW: float
    capacity_ratio: float = 0.10
    roundtrip_efficiency: float = 0.90

    def __post_init__(self) -> None:
        if (
            not np.isfinite(
                [self.training_median_MW, self.capacity_ratio, self.roundtrip_efficiency]
            ).all()
            or self.training_median_MW <= 0
            or self.capacity_ratio < 0
            or not 0 < self.roundtrip_efficiency <= 1
        ):
            raise DataError("Invalid synthetic battery")

    @property
    def capacity(self) -> float:
        return self.training_median_MW * self.capacity_ratio

    @property
    def power(self) -> float:
        return self.capacity / 2

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.roundtrip_efficiency))

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "capacity_MWh": self.capacity,
            "power_MW": self.power,
            "eta_charge": self.eta,
            "eta_discharge": self.eta,
            "minimum_MWh": 0.1 * self.capacity,
            "initial_terminal_MWh": 0.5 * self.capacity,
            "dt_hours": 1,
            "synthetic": True,
        }


@dataclass(frozen=True)
class SyntheticTariff:
    flat: bool = False
    peak: float = 15.0
    degradation: float = 5.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.peak, self.degradation]).all()
            or min(self.peak, self.degradation) < 0
        ):
            raise DataError("Invalid synthetic tariff charges")

    def prices(self) -> Vector:
        return (
            np.full(24, 65.0, dtype=np.float64)
            if self.flat
            else np.array([35.0] * 6 + [65.0] * 11 + [110.0] * 5 + [65.0] * 2)
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "prices_CU_per_MWh": self.prices().tolist(),
            "currency": "synthetic CU",
            "peak_units": "CU/(MW day)",
            "degradation_units": "CU/MWh throughput",
            "timezone": "UTC",
        }


@dataclass(frozen=True)
class DispatchSchedule:
    charge: Vector
    discharge: Vector

    @classmethod
    def zero(cls) -> DispatchSchedule:
        return cls(np.zeros(24), np.zeros(24))


def check_schedule(
    schedule: DispatchSchedule, battery: BatterySpecification, load: Vector
) -> dict[str, object]:
    c, d = schedule.charge, schedule.discharge
    if any(v.shape != (24,) or not np.isfinite(v).all() for v in (c, d, load)) or np.any(load < 0):
        raise DataError("Expected 24 finite MW values and nonnegative load")
    soc = np.r_[
        0.5 * battery.capacity,
        0.5 * battery.capacity + np.cumsum(battery.eta * c - d / battery.eta),
    ]
    tolerance = 1e-6 * max(1, battery.capacity)
    violations = []
    if min(c.min(), d.min()) < -tolerance or max(c.max(), d.max()) > battery.power + tolerance:
        violations.append("power")
    if np.max(np.minimum(c, d)) > tolerance:
        violations.append("simultaneous")
    if soc.min() < 0.1 * battery.capacity - tolerance or soc.max() > battery.capacity + tolerance:
        violations.append("soc")
    if abs(soc[-1] - 0.5 * battery.capacity) > tolerance:
        violations.append("terminal")
    imports = load + c - d
    if imports.min() < -tolerance:
        violations.append("export")
    return {
        "feasible": not violations,
        "violations": violations,
        "soc_MWh": soc.tolist(),
        "min_soc_MWh": float(soc.min()),
        "max_soc_MWh": float(soc.max()),
        "initial_soc_MWh": float(soc[0]),
        "terminal_soc_MWh": float(soc[-1]),
        "min_import_MW": float(imports.min()),
        "peak_import_MW": float(imports.max()),
        "charge_MWh": float(c.sum()),
        "discharge_MWh": float(d.sum()),
        "throughput_MWh": float((c + d).sum()),
        "efficiency_loss_MWh": float(((1 - battery.eta) * c + (1 / battery.eta - 1) * d).sum()),
        "equivalent_full_cycles": float((c + d).sum() / (2 * battery.capacity))
        if battery.capacity
        else 0.0,
    }


def realized_cost(
    schedule: DispatchSchedule, battery: BatterySpecification, tariff: SyntheticTariff, load: Vector
) -> dict[str, object]:
    physical = check_schedule(schedule, battery, load)
    imports = load + schedule.charge - schedule.discharge
    energy = float(tariff.prices() @ imports)
    peak = float(tariff.peak * imports.max())
    degradation = float(tariff.degradation * (schedule.charge + schedule.discharge).sum())
    baseline = float(tariff.prices() @ load + tariff.peak * load.max())
    total = energy + peak + degradation
    return physical | {
        "energy_CU": energy,
        "peak_CU": peak,
        "degradation_CU": degradation,
        "total_CU": total,
        "baseline_CU": baseline,
        "incremental_CU": total - baseline,
        "savings_CU": baseline - total,
        "saving_fraction": (baseline - total) / baseline if baseline > 0 else None,
        "before_peak_MW": float(load.max()),
        "peak_reduction_MW": float(load.max() - imports.max()),
    }


def price_rule(battery: BatterySpecification, tariff: SyntheticTariff) -> DispatchSchedule:
    if battery.capacity == 0 or np.ptp(tariff.prices()) == 0:
        return DispatchSchedule.zero()
    c = np.zeros(24)
    d = np.zeros(24)
    soc = 0.5 * battery.capacity
    # Stable earliest-hour ties. Charge from 50 to 100%, then discharge back to 50%.
    for t in np.flatnonzero(tariff.prices() == tariff.prices().min()):
        c[t] = min(battery.power, (battery.capacity - soc) / battery.eta)
        soc += battery.eta * c[t]
    for t in np.flatnonzero(tariff.prices() == tariff.prices().max()):
        d[t] = min(battery.power, (soc - 0.5 * battery.capacity) * battery.eta)
        soc -= d[t] / battery.eta
    return DispatchSchedule(c, d)


def safety_curtail(
    schedule: DispatchSchedule, battery: BatterySpecification, load: Vector
) -> DispatchSchedule:
    """Current-load-only no-export guard; terminal errors remain explicitly reportable."""
    c = np.zeros(24)
    d = np.zeros(24)
    soc = 0.5 * battery.capacity
    for t in range(24):
        c[t] = max(
            0, min(schedule.charge[t], battery.power, (battery.capacity - soc) / battery.eta)
        )
        d[t] = max(
            0,
            min(
                schedule.discharge[t],
                battery.power,
                (soc - 0.1 * battery.capacity) * battery.eta,
                load[t] + c[t],
            ),
        )
        soc += battery.eta * c[t] - d[t] / battery.eta
    return DispatchSchedule(c, d)


def evaluate_day(
    schedule: DispatchSchedule,
    battery: BatterySpecification,
    tariff: SyntheticTariff,
    actual: Vector,
    label: str,
) -> dict[str, object]:
    """Keep planned and current-load-curtailed executed decisions separately."""
    planned = realized_cost(schedule, battery, tariff, actual)
    executed = schedule if planned["feasible"] else safety_curtail(schedule, battery, actual)
    checked = realized_cost(executed, battery, tariff, actual)
    if not checked["feasible"]:
        executed = DispatchSchedule.zero()
        checked = realized_cost(executed, battery, tariff, actual)
        label += ":safety-zero"
    if not checked["feasible"]:
        raise DataError("Zero dispatch could not satisfy independent physical checker")
    return {
        "label": label,
        "planned": {
            "charge_MW": schedule.charge.tolist(),
            "discharge_MW": schedule.discharge.tolist(),
            "physical": planned,
        },
        "executed": {
            "charge_MW": executed.charge.tolist(),
            "discharge_MW": executed.discharge.tolist(),
            "physical": checked,
        },
        "safety_intervention": not planned["feasible"],
    }
