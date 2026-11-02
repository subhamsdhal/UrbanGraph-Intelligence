"""Explicit region types and evidence-bound, half-open historical area mappings."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum

import pandas as pd

from urbangraph.data import DataError
from urbangraph.evaluation.dataset import identity


class RegionType(str, Enum):
    COUNTRY = "country"
    BIDDING_ZONE = "bidding-zone"
    CONTROL_AREA = "control-area"
    OTHER = "other"


@dataclass(frozen=True)
class TemporalValidity:
    start: str | None = None
    end: str | None = None

    def __post_init__(self) -> None:
        for value in (self.start, self.end):
            if value is not None and pd.Timestamp(value).tzinfo is None:
                raise DataError("Region validity requires timezone-aware instants")
        if (
            self.start is not None
            and self.end is not None
            and pd.Timestamp(self.start) >= pd.Timestamp(self.end)
        ):
            raise DataError("Empty or reversed temporal validity")

    def contains(self, timestamp: pd.Timestamp) -> bool:
        if timestamp.tzinfo is None:
            raise DataError("Area mapping requires an aware instant")
        return (self.start is None or timestamp >= pd.Timestamp(self.start)) and (
            self.end is None or timestamp < pd.Timestamp(self.end)
        )


@dataclass(frozen=True)
class RegionDefinition:
    identifier: str
    name: str
    source_identifier: str
    region_type: RegionType
    source_reference: str
    geographic_interpretation: str
    evidence_status: str
    rights_status: str = "dataset redistribution unresolved"
    validity: TemporalValidity = TemporalValidity()
    eic: str | None = None
    eic_evidence: str | None = None
    version: str = "energy-region-1.0"

    def __post_init__(self) -> None:
        if (
            self.version != "energy-region-1.0"
            or not isinstance(self.region_type, RegionType)
            or not re.fullmatch(r"[a-z-]+:[A-Z0-9_-]+", self.identifier)
            or self.identifier.split(":")[0] != self.region_type.value
            or any(
                not v.strip()
                for v in (
                    self.name,
                    self.source_identifier,
                    self.source_reference,
                    self.geographic_interpretation,
                    self.evidence_status,
                )
            )
        ):
            raise DataError("Invalid explicit region identity")
        # No EIC lookup is implicitly trusted. This milestone has no verified code archive.
        if self.eic is not None or self.eic_evidence is not None:
            raise DataError(
                "EIC assignment requires a separately verified authoritative registry adapter"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceAreaMapping:
    source_identifier: str
    region_identifier: str
    validity: TemporalValidity
    source_reference: str
    relation: str = "source-area-identity"


@dataclass(frozen=True)
class RegionMembership:
    member: str
    parent: str
    validity: TemporalValidity
    evidence_reference: str
    relation: str = "geographic-membership-not-measurement-equivalence"


@dataclass(frozen=True)
class RegionRegistry:
    regions: tuple[RegionDefinition, ...]
    mappings: tuple[SourceAreaMapping, ...] = ()
    memberships: tuple[RegionMembership, ...] = ()

    def __post_init__(self) -> None:
        ids = [r.identifier for r in self.regions]
        if not ids or len(ids) != len(set(ids)):
            raise DataError("Duplicate or empty region registry")
        for mapping in self.mappings:
            if mapping.region_identifier not in ids or not mapping.source_reference:
                raise DataError("Unknown mapping destination or missing evidence")
        for membership in self.memberships:
            if (
                membership.member not in ids
                or membership.parent not in ids
                or not membership.evidence_reference
            ):
                raise DataError("Unknown membership endpoint or evidence")

    def resolve(self, source_identifier: str, at: pd.Timestamp) -> RegionDefinition:
        matches = [
            m
            for m in self.mappings
            if m.source_identifier == source_identifier and m.validity.contains(at)
        ]
        if len(matches) != 1:
            raise DataError("Missing or ambiguous time-bounded source mapping")
        result = self.get(matches[0].region_identifier)
        if not result.validity.contains(at):
            raise DataError("Region definition not valid at requested time")
        return result

    def get(self, identifier: str) -> RegionDefinition:
        for region in self.regions:
            if region.identifier == identifier:
                return region
        raise DataError("Unknown region identifier")

    def axis(self, identifiers: tuple[str, ...], kind: RegionType = RegionType.COUNTRY) -> None:
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise DataError("Duplicate or empty regional axis")
        if any(self.get(i).region_type != kind for i in identifiers):
            raise DataError("Country, bidding-zone and control-area axes cannot be mixed")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def registry_id(self) -> str:
        return identity(self.to_dict())


OPSD = "https://data.open-power-system-data.org/time_series/2020-10-06"
BZR = "https://eepublicdownloads.entsoe.eu/clean-documents/news/bz-review/2018-03_First_Edition_of_the_Bidding_Zone_Review.pdf"


def country_registry() -> RegionRegistry:
    names = {
        "DE": "Germany",
        "FR": "France",
        "NL": "Netherlands",
        "BE": "Belgium",
        "AT": "Austria",
        "CZ": "Czechia",
    }
    regions = tuple(
        RegionDefinition(
            "country:" + code,
            name,
            code,
            RegionType.COUNTRY,
            OPSD,
            "OPSD national actual-load reporting area; not bidding-zone topology",
            "release metadata confirmed; no operational publication-vintage evidence",
        )
        for code, name in names.items()
    )
    return RegionRegistry(
        regions,
        tuple(
            SourceAreaMapping(r.source_identifier, r.identifier, TemporalValidity(), OPSD)
            for r in regions
        ),
    )
