from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.optimize import OptimizeResult  # type: ignore[import-untyped]

from energraph.optimization.physical import (
    BatterySpecification,
    DispatchSchedule,
    SyntheticTariff,
    check_schedule,
    evaluate_day,
    price_rule,
    realized_cost,
    safety_curtail,
)
from energraph.optimization.population import DecisionInputs, midnight_indices
from energraph.optimization.solver import RiskPolicy, empirical_cvar, optimize
from urbangraph.data import DataError


def test_tariff_cost_units() -> None:
    t = SyntheticTariff()
    b = BatterySpecification(1000)
    np.testing.assert_array_equal(t.prices(), [35] * 6 + [65] * 11 + [110] * 5 + [65] * 2)
    r = realized_cost(DispatchSchedule.zero(), b, t, np.full(24, 1000.0))
    assert r["energy_CU"] == 1605000 and r["peak_CU"] == 15000 and r["degradation_CU"] == 0


@pytest.mark.parametrize("ratio", [0, 0.05, 0.1, 0.2])
def test_flat_and_zero_negative_control(ratio: float) -> None:
    b = BatterySpecification(1000, ratio)
    t = SyntheticTariff(flat=True, peak=0)
    schedule, info = optimize(np.full((1, 24), 1000.0), b, t)
    assert info["fallback"] is False
    assert check_schedule(schedule, b, np.full(24, 1000.0))["feasible"]
    assert np.abs(schedule.charge).sum() + np.abs(schedule.discharge).sum() < 1e-6


def test_ideal_arbitrage_and_peak() -> None:
    b = BatterySpecification(1000, roundtrip_efficiency=1)
    t = SyntheticTariff(peak=0, degradation=0)
    s, info = optimize(np.full((1, 24), 1000.0), b, t)
    assert info["fallback"] is False
    r = realized_cost(s, b, t, np.full(24, 1000.0))
    assert r["incremental_CU"] == pytest.approx(-5550)
    assert r["terminal_soc_MWh"] == pytest.approx(50)
    peaked = np.full(24, 1000.0)
    peaked[18] = 1100
    s, _ = optimize(peaked[None, :], b, SyntheticTariff(flat=True, peak=100, degradation=0))
    assert float(str(check_schedule(s, b, peaked)["peak_import_MW"])) < 1100


def test_rule_and_efficiencies() -> None:
    b = BatterySpecification(1000)
    t = SyntheticTariff()
    s = price_rule(b, t)
    r = check_schedule(s, b, np.full(24, 1000.0))
    assert r["feasible"]
    assert r["terminal_soc_MWh"] == pytest.approx(50)
    assert s.discharge.sum() / s.charge.sum() == pytest.approx(0.9)
    assert realized_cost(s, b, t, np.full(24, 1000.0))["degradation_CU"] == pytest.approx(
        5 * (s.charge + s.discharge).sum()
    )


def test_physical_rejection_and_curtailment() -> None:
    b = BatterySpecification(1000)
    s = DispatchSchedule(np.ones(24), np.ones(24))
    assert "simultaneous" in cast(
        list[str], check_schedule(s, b, np.full(24, 1000.0))["violations"]
    )
    s = price_rule(b, SyntheticTariff())
    low = np.zeros(24)
    assert "export" in cast(list[str], check_schedule(s, b, low)["violations"])
    guarded = safety_curtail(s, b, low)
    report = check_schedule(guarded, b, low)
    assert "export" not in cast(list[str], report["violations"]) and "terminal" in cast(
        list[str], report["violations"]
    )


def test_cvar_weighted_tail_and_solver() -> None:
    assert empirical_cvar(np.array([0.0, 10.0]), np.array([0.8, 0.2]), 0.9) == pytest.approx(10)
    assert empirical_cvar(
        np.array([-10.0, 0.0, 20.0]), np.array([0.5, 0.25, 0.25]), 0.5
    ) == pytest.approx(10)
    scenarios = np.full((3, 24), 1000.0)
    scenarios[0, 18] = 1200
    scenarios[1, 19] = 1250
    scenarios[2, 20] = 1300
    s, info = optimize(scenarios, BatterySpecification(1000), SyntheticTariff(), risk=RiskPolicy())
    assert info["fallback"] is False
    assert info["objective_incremental_CU"] == pytest.approx(
        info["solver_objective_incremental_CU"], abs=1e-5
    )
    assert all(
        check_schedule(s, BatterySpecification(1000), load)["feasible"] for load in scenarios
    )


@pytest.mark.parametrize("weight,alpha", [(-1, 0.9), (1, 1), (1, 0), (float("nan"), 0.9)])
def test_invalid_risk(weight: float, alpha: float) -> None:
    with pytest.raises(DataError):
        RiskPolicy(alpha, weight)


def test_decision_time_and_no_overlaps() -> None:
    hour = 3600 * 10**9
    issue = 100 * 24 * hour
    targets = issue + np.arange(24, dtype=np.int64) * hour
    context = DecisionInputs(
        np.ones(24), np.zeros((32, 24)), issue, targets, np.full(32, issue - hour, dtype=np.int64)
    )
    assert context.scenarios()[0].shape == (32, 24)
    with pytest.raises(DataError):
        DecisionInputs(
            context.point, context.residuals, issue, targets, np.full(32, issue, dtype=np.int64)
        ).scenarios()
    with pytest.raises(DataError):
        midnight_indices(np.stack([targets, targets]), np.ones((2, 24, 1)))


def test_price_only_forecast_irrelevance() -> None:
    battery = BatterySpecification(1000)
    tariff = SyntheticTariff(peak=0)
    flat = np.full((1, 24), 1000.0)
    spikes = flat.copy()
    spikes[0, 18] = 2000
    first, _ = optimize(flat, battery, tariff)
    second, _ = optimize(spikes, battery, tariff)
    np.testing.assert_allclose(first.charge, second.charge, atol=1e-6)
    np.testing.assert_allclose(first.discharge, second.discharge, atol=1e-6)


def test_planned_violation_preserved_and_executed_fallback() -> None:
    battery = BatterySpecification(1000)
    schedule = price_rule(battery, SyntheticTariff())
    result = evaluate_day(schedule, battery, SyntheticTariff(), np.zeros(24), "B1")
    assert result["safety_intervention"]
    planned = cast(dict[str, object], result["planned"])
    executed = cast(dict[str, object], result["executed"])
    assert not cast(dict[str, object], planned["physical"])["feasible"]
    assert cast(dict[str, object], executed["physical"])["feasible"]
    assert cast(str, result["label"]).endswith("safety-zero")


def test_scenario_bank_retains_hourly_path_dependence() -> None:
    hour = 3600 * 10**9
    start = 200 * 24 * hour
    residuals = np.tile(np.arange(24, dtype=np.float64), (16, 1))
    residuals[1] *= -1
    context = DecisionInputs(
        np.full(24, 100.0),
        residuals,
        start,
        start + np.arange(24, dtype=np.int64) * hour,
        np.full(16, start - hour, dtype=np.int64),
    )
    scenarios, clipped = context.scenarios()
    assert clipped == 0
    np.testing.assert_array_equal(scenarios[1] - 100, residuals[1])


def test_reported_solver_failure_uses_zero_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "energraph.optimization.solver.milp",
        lambda *args, **kwargs: OptimizeResult(status=1, success=False, x=None, message="timeout"),
    )
    schedule, metadata = optimize(
        np.full((1, 24), 1000.0), BatterySpecification(1000), SyntheticTariff()
    )
    assert metadata["fallback"] is True and metadata["status"] == 1
    assert not schedule.charge.any() and not schedule.discharge.any()
