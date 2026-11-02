"""Versioned, read-only API response contracts for synthetic fixtures."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Identity(Contract):
    schema_version: Literal["eg006-demo-v1"] = "eg006-demo-v1"
    dataset_kind: Literal["synthetic"] = "synthetic"
    manifest_sha256: str


class Meta(Identity):
    regions: int
    origins_per_region: int
    first_utc: str
    last_utc: str
    disclaimer: str


class Region(Contract):
    id: str
    label: str
    origins_utc: list[str]


class Regions(Identity):
    regions: list[Region]


class SeriesPoint(Contract):
    timestamp_utc: str
    load_MW: float


class Series(Identity):
    region_id: str
    unit: Literal["MW"] = "MW"
    points: list[SeriesPoint]
    count: int


class ForecastPoint(Contract):
    horizon_hours: int
    target_utc: str
    point_MW: float
    truth_MW: float
    lower_MW: float
    upper_MW: float


class Forecasts(Identity):
    region_id: str
    origin_utc: str
    unit: Literal["MW"] = "MW"
    interval_kind: Literal["illustrative_uncalibrated"] = "illustrative_uncalibrated"
    points: list[ForecastPoint]


class GraphNode(Contract):
    id: str
    label: str
    x: float
    y: float


class GraphEdge(Contract):
    source: str
    target: str
    weight: float


class Graph(Identity):
    coordinate_system: Literal["synthetic_unit_square"] = "synthetic_unit_square"
    meaning: Literal["invented display links, not electrical topology"] = (
        "invented display links, not electrical topology"
    )
    regions: list[GraphNode]
    edges: list[GraphEdge]


class DispatchPoint(Contract):
    target_utc: str
    load_MW: float
    import_MW: float
    charge_MW: float
    discharge_MW: float
    soc_MWh_end: float
    price_synthetic_CU_per_MWh: float


class Battery(Contract):
    capacity_MWh: float
    power_limit_MW: float
    initial_terminal_soc_MWh: float
    roundtrip_efficiency: float


class Dispatch(Identity):
    region_id: str
    origin_utc: str
    battery: Battery
    points: list[DispatchPoint]
    feasibility: Literal["fixture_checked"] = "fixture_checked"
    disclaimer: str = "Illustrative schedule and synthetic tariff; no operational or savings claim."


class ErrorDetail(Contract):
    code: str
    message: str


class ErrorResponse(Contract):
    schema_version: Literal["eg006-demo-v1"] = "eg006-demo-v1"
    error: ErrorDetail


class ResilientPredictRequest(Contract):
    region_id: str
    origin_utc: str
    injected_outages: list[str] = []
    custom_series: list[dict[str, Any]] | None = None
    confidence_level: float = 0.90


class ResilientForecastPoint(Contract):
    horizon_hours: int
    target_utc: str
    point_MW: float
    truth_MW: float
    lower_MW: float
    upper_MW: float
    raw_point_MW: float
    uncertainty_inflation_factor: float
    is_imputed: bool


class ResilientPredictResponse(Contract):
    schema_version: Literal["eg008-resilient-inference-v1"] = "eg008-resilient-inference-v1"
    region_id: str
    origin_utc: str
    outage_report: dict[str, Any]
    points: list[ResilientForecastPoint]
    drift_detected: bool
    drift_status: str
    model_id: str = "urbangraph-resilience-v1"


class DriftCheckRequest(Contract):
    features: dict[str, list[float]]


class DriftMonitoringStatus(Contract):
    schema_version: Literal["eg021-observability-v1"] = "eg021-observability-v1"
    overall_status: str
    drift_alerts_total: int
    sensor_outages_total: int
    reference_features: list[str]
    sample_count: int
    latest_report: dict[str, Any]


class SimulationRequest(Contract):
    scenario_type: Literal["dc_power_flow", "battery_optimization"] = "dc_power_flow"
    region_id: str = "SYN_A"
    load_scale: float = 1.0
    line_outages: list[str] = []
    battery_capacity_ratio: float = 0.10
    battery_efficiency: float = 0.90
    tariff_peak: float = 15.0
    tariff_degradation: float = 5.0


class SimulationResponse(Contract):
    schema_version: Literal["eg014-grid-simulation-v1"] = "eg014-grid-simulation-v1"
    scenario_type: str
    region_id: str
    feasible: bool
    status: str
    branch_flows: list[dict[str, Any]]
    bus_angles: dict[str, float]
    constraint_violations: list[str]
    battery_schedule: list[dict[str, Any]] | None = None
    realized_cost_CU: dict[str, float] | None = None
    advisory_notice: str = "Local simulation only. No physical SCADA actuation."

