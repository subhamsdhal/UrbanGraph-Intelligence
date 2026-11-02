"""Versioned single-region variable identities; no implicit generation aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from urbangraph.data import DataError


@dataclass(frozen=True)
class Variable:
    identifier: str
    upstream: str
    kind: str
    unit: str
    source_release: str
    aggregation: str
    latency_hours: int = 24
    region: str = "DE"
    region_type: str = "country"
    resolution_seconds: int = 3600
    observation_type: str = "actual"
    latency_status: str = "assumed"
    missingness: str = "empty-source-field-only; preserve NaN"
    validity: str = "finite; zeros valid; negatives audited, never clipped"
    schema_version: str = "energy-variable-2.0"

    def __post_init__(self) -> None:
        if (
            self.schema_version != "energy-variable-2.0"
            or not self.identifier
            or not self.upstream
            or self.region != "DE"
            or self.region_type != "country"
            or self.resolution_seconds != 3600
            or type(self.latency_hours) is not int
            or not 0 <= self.latency_hours <= 168
            or self.unit not in {"MW", "degC", "W/m2", "1"}
            or self.observation_type not in {"actual", "reanalysis", "known-calendar"}
            or self.latency_status not in {"assumed", "verified", "not-applicable"}
        ):
            raise DataError("Invalid or unsupported energy variable contract")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def registry() -> dict[str, Variable]:
    entries = [
        Variable(
            "load",
            "DE_load_actual_entsoe_transparency",
            "load",
            "MW",
            "2020-10-06",
            "national hourly load",
        )
    ]
    for kind in ("solar", "wind", "wind_onshore", "wind_offshore"):
        entries.append(
            Variable(
                kind,
                f"DE_{kind}_generation_actual",
                "generation",
                "MW",
                "2020-10-06",
                "national generation; wind total is not additive to components",
            )
        )
        entries.append(
            Variable(
                kind + "_capacity",
                f"DE_{kind}_capacity",
                "capacity",
                "MW",
                "2020-10-06",
                "installed national capacity",
            )
        )
    for key, field, unit in (
        ("temperature", "DE_temperature", "degC"),
        ("direct_radiation", "DE_radiation_direct_horizontal", "W/m2"),
        ("diffuse_radiation", "DE_radiation_diffuse_horizontal", "W/m2"),
    ):
        entries.append(
            Variable(
                key,
                field,
                "weather",
                unit,
                "2020-09-16",
                "country population-weighted MERRA-2 reanalysis",
                observation_type="reanalysis",
            )
        )
    entries.append(
        Variable(
            "calendar",
            "UTC/local Europe/Berlin timestamp",
            "calendar",
            "1",
            "derived-calendar-v1",
            "hour and weekday",
            latency_hours=0,
            observation_type="known-calendar",
            latency_status="not-applicable",
        )
    )
    return {v.identifier: v for v in entries}


def variables(names: tuple[str, ...]) -> tuple[Variable, ...]:
    known = registry()
    if not names or len(set(names)) != len(names) or any(n not in known for n in names):
        raise DataError("Unknown, duplicate or empty variable selection")
    return tuple(known[n] for n in names)
