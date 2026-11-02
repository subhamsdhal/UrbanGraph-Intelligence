"""Synthetic-only scientific evaluation with a default-deny provenance boundary.

This module never opens a caller-supplied dataset path. The command-line entry point
loads only the repository's pinned EG-006 fixture package.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from energraph.demo.fixtures import (
    BUILTIN_MANIFEST_SHA256,
    DemoData,
    load_demo,
)
from urbangraph.reproducibility import collect_environment_info

if TYPE_CHECKING:
    from collections.abc import Sequence


class ValidationError(ValueError):
    """Governance or evaluation contract failure."""


class DatasetClass(StrEnum):
    SYNTHETIC = "synthetic"
    RIGHTS_CLEARED = "rights_cleared"
    RESTRICTED_OR_UNKNOWN = "restricted_or_unknown"


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_id: str
    source_id: str
    classification: DatasetClass
    permission_evidence: str | None
    manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.dataset_id.strip()
            or not self.source_id.strip()
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValidationError("Dataset identity, source and SHA-256 are required")
        if self.classification is not DatasetClass.RESTRICTED_OR_UNKNOWN and not (
            self.permission_evidence and self.permission_evidence.strip()
        ):
            raise ValidationError("Usage-permission evidence is required")


@dataclass(frozen=True)
class ForecastObservation:
    region_id: str
    origin_utc: datetime
    target_utc: datetime
    horizon_hours: int
    latest_feature_utc: datetime
    prediction: float
    target: float
    lower: float
    upper: float
    unit: str = "MW"


def _utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValidationError("Evaluation timestamps must be UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValidationError("Missing UTC timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError("Invalid UTC timestamp") from exc
    _utc(result)
    return result.astimezone(UTC)


def _finite(value: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


def _fixture_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError("Missing or nonnumeric fixture value")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError("Nonfinite fixture value")
    return result


def _score_observations(
    observations: Sequence[ForecastObservation],
    *,
    window_start_utc: datetime,
    window_end_utc: datetime,
) -> dict[str, Any]:
    """Calculate unchecked numerical metrics; this helper grants no data authorization."""
    start, end = _utc(window_start_utc), _utc(window_end_utc)
    if start > end or not observations:
        raise ValidationError("Empty or inverted evaluation window")
    seen: set[tuple[str, str, int]] = set()
    groups: dict[tuple[str, int], list[tuple[float, float]]] = {}
    origins: set[str] = set()
    for row in observations:
        origin, target = (
            _utc(row.origin_utc),
            _utc(row.target_utc),
        )
        _utc(row.latest_feature_utc)
        if not row.region_id or row.unit != "MW" or type(row.horizon_hours) is not int:
            raise ValidationError("Invalid region, horizon or unit")
        if not 1 <= row.horizon_hours <= 24:
            raise ValidationError("Unsupported forecast horizon")
        if (
            row.target_utc - row.origin_utc != timedelta(hours=row.horizon_hours)
            or row.latest_feature_utc > row.origin_utc
            or not start <= target <= end
        ):
            raise ValidationError("Misaligned target, future feature or out-of-window row")
        key = (row.region_id, origin, row.horizon_hours)
        if key in seen:
            raise ValidationError("Duplicate forecast population key")
        seen.add(key)
        values = (row.prediction, row.target, row.lower, row.upper)
        if not all(_finite(value) for value in values) or not (
            row.lower <= row.prediction <= row.upper
        ):
            raise ValidationError("Missing, nonfinite or unordered forecast value")
        origins.add(origin)
        groups.setdefault((row.region_id, row.horizon_hours), []).append(
            (row.prediction, row.target)
        )
    metrics = []
    for (region, horizon), pairs in sorted(groups.items()):
        errors = [estimate - actual for estimate, actual in pairs]
        metrics.append(
            {
                "region_id": region,
                "horizon_hours": horizon,
                "count": len(errors),
                "mae_MW": math.fsum(abs(error) for error in errors) / len(errors),
                "rmse_MW": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
            }
        )
    return {
        "unit": "MW",
        "window_start_utc": start,
        "window_end_utc": end,
        "origins_utc": sorted(origins),
        "population_count": len(observations),
        "metrics_by_region_horizon": metrics,
    }


def _bundled_fixture_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eg006"


def _evaluate_authenticated_demo(data: DemoData) -> dict[str, Any]:
    """Adapt only data returned by the internally selected, verified fixture loader."""
    provenance = DatasetProvenance(
        dataset_id="eg006-synthetic-energy",
        source_id="scripts/generate_eg006_synthetic.py",
        classification=DatasetClass.SYNTHETIC,
        permission_evidence="original-project-generated-fixture",
        manifest_sha256=data.manifest_sha256,
    )
    if data.manifest_sha256 != BUILTIN_MANIFEST_SHA256:
        raise ValidationError("Bundled fixture identity differs from the pinned digest")
    series = {
        region: {_parse_utc(row["timestamp_utc"]): _fixture_number(row["load_MW"]) for row in rows}
        for region, rows in data.series.items()
    }
    offsets = {"SYN_A": -2, "SYN_B": 0, "SYN_C": 2}
    observations: list[ForecastObservation] = []
    for (region, origin_text), rows in sorted(data.forecasts.items()):
        if region not in offsets or region not in series:
            raise ValidationError("Unknown synthetic region")
        origin = _parse_utc(origin_text)
        for row in rows:
            target = _parse_utc(row["target_utc"])
            previous_day = target - timedelta(hours=24)
            prior_value = series[region].get(previous_day)
            if prior_value is None or previous_day > origin:
                raise ValidationError("Forecast feature is missing or from the future")
            prediction = _fixture_number(row["point_MW"])
            truth = series[region].get(target)
            if (
                truth is None
                or _fixture_number(row["truth_MW"]) != truth
                or prediction != prior_value + offsets[region]
            ):
                raise ValidationError("Forecast rule or target differs from the synthetic series")
            horizon = row["horizon_hours"]
            if type(horizon) is not int:
                raise ValidationError("Invalid synthetic horizon")
            observations.append(
                ForecastObservation(
                    region_id=region,
                    origin_utc=origin,
                    target_utc=target,
                    horizon_hours=horizon,
                    latest_feature_utc=previous_day,
                    prediction=prediction,
                    target=truth,
                    lower=_fixture_number(row["lower_MW"]),
                    upper=_fixture_number(row["upper_MW"]),
                )
            )
    if len(observations) != 3 * 7 * 24:
        raise ValidationError("Synthetic forecast population is incomplete")
    targets = [row.target_utc for row in observations]
    evaluation = _score_observations(
        observations,
        window_start_utc=min(targets),
        window_end_utc=max(targets),
    )
    return {
        "schema_version": "eg007a-synthetic-evaluation-1.0",
        "provenance": asdict(provenance),
        "evaluation": evaluation,
        "intervals": {"kind": "illustrative_uncalibrated", "evaluated_for_coverage": False},
        "limitations": [
            "Invented synthetic observations and forecasts; no real-world accuracy inference.",
            "Previous-day fixture rule only; no trained model or operational forecast.",
            "Illustrative intervals are not calibrated risk bounds.",
        ],
        "method": "previous-day same-hour synthetic rule with fixed regional offset",
    }


def evaluate_demo() -> dict[str, Any]:
    """Verify the fixed bundled fixture's bytes, then evaluate its observations."""
    data = load_demo(_bundled_fixture_root())
    return _evaluate_authenticated_demo(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate only the pinned bundled synthetic fixture"
    )
    parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = evaluate_demo()
    report["environment"] = collect_environment_info(root).to_dict()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
