"""Scientific checks use only the pinned, invented EG-006 fixture."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from energraph import scientific_validation as validation
from energraph.demo.fixtures import FixtureError, load_demo
from energraph.scientific_validation import (
    DatasetClass,
    DatasetProvenance,
    ForecastObservation,
    ValidationError,
    evaluate_demo,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "eg006"
MANIFEST = "b874f63bd9215a3defdf706691fdc703cf8c5147eaf7dce4415a237352ecf3bd"
ORIGIN = datetime(2030, 1, 1, tzinfo=UTC)


def provenance() -> DatasetProvenance:
    return DatasetProvenance(
        "eg006-synthetic-energy",
        "scripts/generate_eg006_synthetic.py",
        DatasetClass.SYNTHETIC,
        "original-project-generated-fixture",
        MANIFEST,
    )


def observation(**changes: Any) -> ForecastObservation:
    row = ForecastObservation(
        "SYN_A", ORIGIN, ORIGIN + timedelta(hours=1), 1, ORIGIN, 12, 10, 8, 16
    )
    return replace(row, **changes)


def score(*rows: ForecastObservation) -> dict[str, object]:
    return validation._score_observations(
        rows, window_start_utc=ORIGIN, window_end_utc=ORIGIN + timedelta(days=2)
    )


def test_metric_math_and_stable_report() -> None:
    rows = (
        observation(),
        observation(
            origin_utc=ORIGIN + timedelta(hours=1),
            target_utc=ORIGIN + timedelta(hours=2),
            latest_feature_utc=ORIGIN + timedelta(hours=1),
            prediction=16,
            target=10,
            lower=12,
            upper=20,
        ),
    )
    result = score(*rows)
    metrics = result["metrics_by_region_horizon"][0]  # type: ignore[index]
    assert metrics == {
        "region_id": "SYN_A",
        "horizon_hours": 1,
        "count": 2,
        "mae_MW": 4.0,
        "rmse_MW": (20.0) ** 0.5,
    }
    assert json.dumps(result, sort_keys=True) == json.dumps(score(*rows), sort_keys=True)


@pytest.mark.parametrize("kind", [DatasetClass.RIGHTS_CLEARED, DatasetClass.RESTRICTED_OR_UNKNOWN])
def test_non_synthetic_data_default_denied(kind: DatasetClass) -> None:
    declared = replace(provenance(), classification=kind)
    with pytest.raises(TypeError):
        evaluate_demo(declared)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "field", ["dataset_id", "source_id", "permission_evidence", "manifest_sha256"]
)
def test_missing_or_wrong_provenance_denied(field: str) -> None:
    changes: dict[str, Any] = {field: "0" * 64 if field == "manifest_sha256" else "unverified"}
    declared = replace(provenance(), **changes)
    with pytest.raises(TypeError):
        evaluate_demo(declared)  # type: ignore[call-arg]


def test_provenance_requires_identity_digest_and_permission() -> None:
    for changes in (
        {"dataset_id": " "},
        {"source_id": ""},
        {"manifest_sha256": "bad"},
        {"permission_evidence": None},
    ):
        with pytest.raises(ValidationError):
            replace(provenance(), **changes)


@pytest.mark.parametrize(
    "row",
    [
        observation(latest_feature_utc=ORIGIN + timedelta(minutes=1)),
        observation(horizon_hours=2),
        observation(target_utc=ORIGIN - timedelta(hours=1)),
        observation(unit="kW"),
        observation(prediction=float("nan")),
        observation(target=float("inf")),
        observation(lower=13),
        observation(region_id=""),
        observation(origin_utc=datetime(2030, 1, 1)),
    ],
)
def test_malformed_or_leaky_rows_rejected(row: ForecastObservation) -> None:
    with pytest.raises(ValidationError):
        score(row)


def test_duplicate_population_rejected() -> None:
    with pytest.raises(ValidationError, match="Duplicate"):
        score(observation(), observation())


def test_inverted_window_and_empty_population_rejected() -> None:
    with pytest.raises(ValidationError):
        score()
    with pytest.raises(ValidationError):
        validation._score_observations(
            [observation()],
            window_start_utc=ORIGIN + timedelta(days=1),
            window_end_utc=ORIGIN,
        )


def test_bundled_synthetic_integration_and_provenance() -> None:
    first = evaluate_demo()
    second = evaluate_demo()
    assert first == second
    assert first["provenance"]["manifest_sha256"] == MANIFEST
    assert first["evaluation"]["population_count"] == 504
    assert len(first["evaluation"]["metrics_by_region_horizon"]) == 72
    assert first["intervals"]["evaluated_for_coverage"] is False


@pytest.mark.parametrize(
    "changes",
    [
        {"point_MW": 999.0},
        {"truth_MW": 999.0},
        {"target_utc": "2030-01-01T00:00:00Z"},
        {"region_id": "INVENTED_ONLY"},
        {"horizon_hours": 2},
        {"origin_utc": "2030-01-01T00:00:00Z"},
    ],
)
def test_caller_modified_demo_cannot_enter_public_evaluator(changes: dict[str, object]) -> None:
    data = load_demo(FIXTURE)
    forecasts = {key: list(rows) for key, rows in data.forecasts.items()}
    key = next(iter(forecasts))
    forecasts[key][0] = forecasts[key][0] | changes
    changed = replace(data, forecasts={key: tuple(rows) for key, rows in forecasts.items()})
    with pytest.raises(TypeError):
        evaluate_demo(changed)  # type: ignore[call-arg]


@pytest.mark.parametrize("mutation", ["remove", "add", "duplicate"])
def test_caller_population_changes_cannot_enter_public_evaluator(mutation: str) -> None:
    data = load_demo(FIXTURE)
    forecasts = {key: list(rows) for key, rows in data.forecasts.items()}
    key = next(iter(forecasts))
    if mutation == "remove":
        forecasts[key].pop()
    else:
        forecasts[key].append(dict(forecasts[key][0]))
    changed = replace(data, forecasts={key: tuple(rows) for key, rows in forecasts.items()})
    with pytest.raises(TypeError):
        evaluate_demo(changed)  # type: ignore[call-arg]


def test_original_invented_row_bypass_is_closed() -> None:
    invented = observation(region_id="INVENTED_ONLY", prediction=509, target=10, lower=0, upper=600)
    # Even copied approved metadata cannot authenticate a caller-created row.
    with pytest.raises(TypeError):
        evaluate_demo([invented], provenance())  # type: ignore[call-arg]
    assert not hasattr(validation, "evaluate_observations")
    assert not hasattr(validation, "authorize_demo")
    assert "provenance" not in score(invented)


@pytest.mark.parametrize("mutation", ["missing", "tampered", "forged_manifest"])
def test_missing_or_tampered_fixed_fixture_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    copied = tmp_path / "fixture"
    shutil.copytree(FIXTURE, copied)
    forecast_path = copied / "fixtures" / "forecasts_and_intervals.json"
    if mutation == "missing":
        forecast_path.unlink()
    elif mutation == "tampered":
        forecast_path.write_bytes(forecast_path.read_bytes() + b" ")
    else:
        manifest_path = copied / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_id"] = "forged-with-copied-digest"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(validation, "_bundled_fixture_root", lambda: copied)
    with pytest.raises(FixtureError):
        evaluate_demo()
