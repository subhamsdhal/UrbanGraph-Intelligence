"""Typed energy identity and operational feature-availability contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from urbangraph.data import DataError


@dataclass(frozen=True)
class EnergyVariable:
    column: str = "DE_load_actual_entsoe_transparency"
    region: str = "DE"
    area_type: str = "country"
    variable: str = "actual_electricity_load"
    unit: str = "MW"
    source_version: str = "2020-10-06"
    schema_version: str = "energy-1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "energy-1.0" or self.area_type not in {
            "country",
            "bidding-zone",
            "control-area",
        }:
            raise DataError("Unsupported energy schema or geographic identity")
        if any(not isinstance(v, str) or not v.strip() for v in asdict(self).values()):
            raise DataError("Energy identity fields must be nonempty strings")
        if self.unit not in {"MW", "MWh", "degC", "W/m2", "m/s", "EUR/MWh"}:
            raise DataError("Explicit supported physical unit required")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, obj: dict[str, object]) -> EnergyVariable:
        if set(obj) != set(asdict(cls())) or any(not isinstance(v, str) for v in obj.values()):
            raise DataError("Invalid energy variable schema")
        return cls(**{k: str(v) for k, v in obj.items()})


@dataclass(frozen=True)
class Availability:
    category: str
    latency_hours: int = 24

    def __post_init__(self) -> None:
        if self.category not in {
            "historical-observed",
            "known-at-origin",
            "future-calendar",
            "issued-forecast",
            "future-measurement",
            "retrospective-only",
        }:
            raise DataError("Unknown feature availability category")
        if type(self.latency_hours) is not int or self.latency_hours < 1:
            raise DataError("Interval-start measurements require positive availability latency")

    def require(
        self,
        timestamp: pd.Timestamp,
        origin: pd.Timestamp,
        *,
        issued_at: pd.Timestamp | None = None,
    ) -> None:
        if timestamp.tzinfo is None or origin.tzinfo is None:
            raise DataError("Explicit timezone required for availability")
        if self.category == "future-calendar":
            return
        if self.category == "issued-forecast":
            if issued_at is not None and issued_at.tzinfo is not None and issued_at <= origin:
                return
        elif self.category == "historical-observed":
            if timestamp + pd.Timedelta(hours=self.latency_hours) <= origin:
                return
        elif self.category == "known-at-origin" and timestamp <= origin:
            return
        raise DataError("Unavailable feature: future actuals and reanalysis are forbidden")
