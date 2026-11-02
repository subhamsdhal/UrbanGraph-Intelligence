"""Synthetic renewable contracts, availability and immutable model evidence."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from energraph.artifacts import PendingPublication, finalize_governed_publication
from energraph.energy_boosting import error_summary, run_energy, verify_energy_run
from energraph.energy_features import EnergyFeatures, build_features
from energraph.multivariable import audit, materialize_multi, open_multi, parse_variables
from energraph.target_plans import create_target_plan, read_target_plan
from energraph.variables import registry, variables
from urbangraph.boosting.contracts import BoostConfig
from urbangraph.data import DataError, DatasetSnapshot, ingest_dataset, verify_snapshot


def _grant(
    rights: Any, raw: Path | DatasetSnapshot, root: Path, purpose: str, operation: str
) -> None:
    directory = raw.directory if isinstance(raw, DatasetSnapshot) else raw
    checked = verify_snapshot(directory)
    assert checked.valid and checked.manifest is not None
    rights._grant(
        directory.parent.name,
        directory,
        checked.manifest.snapshot_id,
        operation,
        root,
        purpose,
    )


def fixture(tmp_path: Path, rights: Any) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    rows = 2400
    hour = np.arange(rows)
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "DE_load_actual_entsoe_transparency": 100 + np.sin(hour / 24) * 20,
            "DE_solar_generation_actual": np.maximum(0, np.sin(hour / 24 * 2 * np.pi)) * 20,
            "DE_wind_generation_actual": 30 + np.sin(hour / 100) * 10,
        }
    )
    frame.to_csv(source / "energy.csv", index=False)
    rights.source(source, "synthetic-energy", tmp_path / "raw")
    raw = ingest_dataset(source, "synthetic-energy", raw_root=tmp_path / "raw")
    for purpose, operation in (
        ("canonicalize", "energy_multivariable"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_multivariable_read"),
    ):
        _grant(rights, raw, tmp_path / "canonical", purpose, operation)
    result = materialize_multi(
        raw.directory, tmp_path / "canonical", "energy.csv", ("load", "solar", "wind")
    )
    assert not isinstance(result, PendingPublication)
    return result.directory, raw.directory


def _plan_grants(rights: Any, raw_path: Path, tmp_path: Path) -> None:
    for purpose, operation in (
        ("scientific_evaluation", "energy_target_plan_create"),
        ("local_research", "derived_commit"),
        ("model_training", "energy_target_plan_read"),
    ):
        _grant(rights, raw_path, tmp_path / "plans", purpose, operation)


@pytest.mark.parametrize("name", list(registry()))
def test_registry(name: str) -> None:
    item = registry()[name]
    assert item.region == "DE" and item.region_type == "country"
    assert item.resolution_seconds == 3600
    assert variables((name,))[0] == item


@pytest.mark.parametrize("names", [("load", "load"), (), ("fake",)])
def test_bad_variables(names: tuple[str, ...]) -> None:
    with pytest.raises(DataError):
        variables(names)


@pytest.mark.parametrize("mode", ["W0", "W2", "W3"])
def test_weather_separation(mode: str) -> None:
    with pytest.raises(DataError):
        EnergyFeatures(("temperature",), (24,), weather_mode=mode)


def test_causal_cross_variable_features() -> None:
    frame = pd.DataFrame(
        {"load": np.arange(500, dtype=float), "solar": np.arange(500, dtype=float) * 2},
        index=pd.date_range("2024-03-20", periods=500, freq="h", tz="UTC"),
    )
    spec = EnergyFeatures(("load", "solar"), (24, 48))
    o, h = np.array([250], dtype=np.int64), np.array([24], dtype=np.int64)
    x, names, _ = build_features(frame, o, h, spec)
    assert x[0, names.index("load_lag_0")] == 226
    assert x[0, names.index("solar_lag_0")] == 404
    changed = frame.copy()
    changed.loc[changed.index[227:], "load"] = -999
    changed.loc[changed.index[203:], "solar"] = -999
    np.testing.assert_array_equal(x, build_features(changed, o, h, spec)[0])
    frame.loc[frame.index[:250], "solar"] = np.nan
    missing = build_features(frame, o, h, spec)[0]
    assert missing[0, names.index("solar_lag_0_missing")] == 1
    assert np.isnan(missing[0, names.index("solar_mean_168")])


def test_zero_metrics_and_quality() -> None:
    zero = np.zeros(24)
    result = error_summary(zero, np.ones(24))
    assert result["mae_MW"] == 1 and result["mape_percent"] is None
    frame = pd.DataFrame(
        {
            "solar": [0.0, -1.0, 12.0],
            "solar_capacity": [10.0, 10.0, 10.0],
            "wind": [3.0, 4.0, 5.0],
            "wind_onshore": [1.0, 2.0, 3.0],
            "wind_offshore": [2.0, 2.0, 1.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
    )
    report = audit(frame)
    assert report["solar_capacity_check"] == {
        "matched": 3,
        "above_capacity": 1,
        "policy": "flag only; no clipping or capacity extrapolation",
    }
    assert frame.solar.iloc[0] == 0 and frame.solar.iloc[1] == -1


@pytest.mark.parametrize("protocol", ["chronological", "rolling"])
def test_target_plan(
    tmp_path: Path,
    protocol: str,
    invented_research_rights: Any,
    broker_context_factory: Any,
) -> None:
    canonical, raw = fixture(tmp_path, invented_research_rights)
    _plan_grants(invented_research_rights, raw, tmp_path)
    context, raw_ref, policy_approver = broker_context_factory(
        raw,
        invented_research_rights,
        (
            ("canonicalize", "energy_multivariable"),
            ("local_research", "energy_multivariable_read"),
            ("scientific_evaluation", "energy_target_plan_create"),
            ("model_training", "energy_target_plan_read"),
        ),
    )
    canonical_pending = materialize_multi(
        raw,
        canonical.parent,
        "energy.csv",
        ("load", "solar", "wind"),
        broker_context=context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical_snapshot = finalize_governed_publication(context, canonical_pending)
    plan_pending = create_target_plan(
        canonical_snapshot,
        tmp_path / "plans",
        "solar",
        protocol,
        raw_snapshot_dir=raw,
        broker_context=context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(context, plan_pending)
    data, frame, split = read_target_plan(
        canonical_snapshot, plan, raw_snapshot_dir=raw, broker_context=context
    )
    assert data.sensor_ids == ("solar",) and len(frame.columns) == 3
    assert split.config.window.history == 168 and split.config.window.horizon == 24
    assert split.config.features.observed_delay_steps == 24
    assert len(split.folds) == (2 if protocol == "rolling" else 1)


def test_native_workflow(
    tmp_path: Path,
    invented_research_rights: Any,
    monkeypatch: pytest.MonkeyPatch,
    broker_context_factory: Any,
) -> None:
    _, raw = fixture(tmp_path, invented_research_rights)
    operations = (
        ("canonicalize", "energy_multivariable"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_multivariable_read"),
        ("scientific_evaluation", "energy_target_plan_create"),
        ("model_training", "energy_target_plan_read"),
        ("model_training", "energy_boost_train"),
        ("saved_run_verification", "energy_boost_verify"),
    )
    broker_context, raw_reference, policy_approver = broker_context_factory(
        raw, invented_research_rights, operations
    )
    canonical_pending = materialize_multi(
        raw,
        tmp_path / "canonical",
        "energy.csv",
        ("load", "solar", "wind"),
        broker_context=broker_context,
        raw_broker_reference=raw_reference,
    )
    assert isinstance(canonical_pending, PendingPublication)
    policy_approver(canonical_pending)
    canonical = finalize_governed_publication(broker_context, canonical_pending)
    _plan_grants(invented_research_rights, raw, tmp_path)
    for purpose, operation in (
        ("model_training", "energy_boost_train"),
        ("local_research", "derived_commit"),
        ("saved_run_verification", "energy_boost_verify"),
    ):
        _grant(invented_research_rights, raw, tmp_path / "runs", purpose, operation)
    plan_pending = create_target_plan(
        canonical,
        tmp_path / "plans",
        "solar",
        "chronological",
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(plan_pending, PendingPublication)
    policy_approver(plan_pending)
    plan = finalize_governed_publication(broker_context, plan_pending)
    config = BoostConfig(
        trees=8,
        max_depth=2,
        alternative_depth=3,
        num_leaves=7,
        min_child_samples=5,
        max_train_rows=2400,
        max_validation_rows=1200,
        trial_seconds=60,
        run_seconds=180,
        threads=1,
    )
    features = EnergyFeatures(("solar", "wind"), (24, 24), lags=(0, 23), rolling=(24,))
    result = run_energy(
        canonical,
        plan,
        tmp_path / "runs",
        features,
        config,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(result, PendingPublication)
    policy_approver(result)
    result = finalize_governed_publication(broker_context, result)
    assert (
        verify_energy_run(
            result,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )["valid"]
        is True
    )
    repeated_pending = run_energy(
        canonical,
        plan,
        tmp_path / "runs",
        features,
        config,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(repeated_pending, PendingPublication)
    policy_approver(repeated_pending)
    repeated = finalize_governed_publication(broker_context, repeated_pending)
    assert (
        verify_energy_run(
            repeated,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )["experiment_id"]
        == verify_energy_run(
            result,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )["experiment_id"]
    )
    verify_grant = next(
        grant
        for grant in invented_research_rights.grants
        if grant["operation"] == "energy_boost_verify"
    )
    verify_grant["status"] = "revoked"
    invented_research_rights.path.write_text(
        json.dumps({"version": 1, "grants": invented_research_rights.grants})
    )

    def forbidden_descriptor(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved boost descriptor read before verify permission")

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.energy_boosting.read_json", forbidden_descriptor)
        with pytest.raises(DataError):
            run_energy(
                canonical,
                plan,
                tmp_path / "runs",
                features,
                config,
                raw_snapshot_dir=raw,
                broker_context=broker_context,
            )
    verify_grant["status"] = "approved"
    invented_research_rights.path.write_text(
        json.dumps({"version": 1, "grants": invented_research_rights.grants})
    )
    model = next((result.directory / "payload").glob("*-trial-0.txt"))
    model.write_bytes(model.read_bytes() + b"corrupt")
    with pytest.raises(DataError):
        verify_energy_run(
            result,
            canonical=canonical,
            plan=plan,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )


def test_roundtrip_corruption(tmp_path: Path, invented_research_rights: Any) -> None:
    canonical, raw = fixture(tmp_path, invented_research_rights)
    frame, _ = open_multi(canonical, raw_snapshot_dir=raw)
    assert (frame.solar == 0).any()
    path = canonical / "payload/observations.parquet"
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(DataError):
        open_multi(canonical, raw_snapshot_dir=raw)


def test_duplicate_timestamp(tmp_path: Path, invented_research_rights: Any) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "duplicate.csv"
    path.write_text(
        "utc_timestamp,DE_solar_generation_actual\n2024-01-01T00:00:00Z,0\n2024-01-01T00:00:00Z,1\n"
    )
    invented_research_rights.source(source, "invented-duplicate", tmp_path / "raw")
    raw = ingest_dataset(source, "invented-duplicate", raw_root=tmp_path / "raw")
    _grant(invented_research_rights, raw, tmp_path, "local_research", "energy_source_read")
    with pytest.raises(DataError):
        parse_variables(
            raw.directory / "payload/duplicate.csv",
            ("solar",),
            raw_snapshot_dir=raw.directory,
            destination=tmp_path,
        )


@pytest.mark.parametrize("latency", [0, 1, 24, 48, 168])
def test_latency_boundaries(latency: int) -> None:
    frame = pd.DataFrame(
        {"load": np.arange(600, dtype=float)},
        index=pd.date_range("2024-01-01", periods=600, freq="h", tz="UTC"),
    )
    spec = EnergyFeatures(("load",), (latency,))
    x, names, _ = build_features(
        frame, np.array([400], dtype=np.int64), np.array([1], dtype=np.int64), spec
    )
    assert x[0, names.index("load_lag_0")] == 400 - latency
    assert x[0, names.index("load_seasonal_24")] <= 400 - latency


@pytest.mark.parametrize("horizon", [0, 25, -1])
def test_invalid_horizon(horizon: int) -> None:
    frame = pd.DataFrame(
        {"load": np.ones(500)}, index=pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    )
    with pytest.raises(DataError):
        build_features(
            frame,
            np.array([300], dtype=np.int64),
            np.array([horizon], dtype=np.int64),
            EnergyFeatures(),
        )


@pytest.mark.parametrize("defect", ["naive", "gap", "duplicate"])
def test_feature_axis_rejected(defect: str) -> None:
    index = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    if defect == "naive":
        index = index.tz_localize(None)
    elif defect == "gap":
        index = index[np.arange(len(index)) != 4]
    else:
        index = pd.DatetimeIndex([*index[:4], index[4], *index[4:]])
    frame = pd.DataFrame({"load": np.ones(len(index))}, index=index)
    with pytest.raises(DataError):
        build_features(
            frame, np.array([300], dtype=np.int64), np.array([1], dtype=np.int64), EnergyFeatures()
        )


def test_w3_issue_time() -> None:
    from energraph.energy_features import require_issued_weather

    origin = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    future = pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime()
    require_issued_weather(future, origin, origin)
    for issue in (None, future, origin.replace(tzinfo=None)):
        with pytest.raises(DataError):
            require_issued_weather(future, origin, issue)


def test_reanalysis_hindsight_only() -> None:
    spec = EnergyFeatures(("temperature",), (24,), weather_mode="W1")
    assert spec.to_dict()["weather_mode"] == "W1"


def test_cli_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], invented_research_rights: Any
) -> None:
    from energraph.cli import main

    canonical, raw = fixture(tmp_path, invented_research_rights)
    assert main(["inspect-multi", "--directory", str(canonical), "--snapshot", str(raw)]) == 0
    assert "energy-multivariable-1.0" in capsys.readouterr().out
    assert (
        main(["inspect-multi", "--directory", str(tmp_path / "absent"), "--snapshot", str(raw)])
        == 1
    )


def test_gap_quality_preserved(tmp_path: Path, invented_research_rights: Any) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "gap.csv"
    path.write_text(
        "utc_timestamp,DE_solar_generation_actual\n2024-01-01T00:00:00Z,0\n2024-01-01T02:00:00Z,\n"
    )
    invented_research_rights.source(source, "invented-gap", tmp_path / "raw")
    raw = ingest_dataset(source, "invented-gap", raw_root=tmp_path / "raw")
    _grant(invented_research_rights, raw, tmp_path, "local_research", "energy_source_read")
    frame = parse_variables(
        raw.directory / "payload/gap.csv",
        ("solar",),
        raw_snapshot_dir=raw.directory,
        destination=tmp_path,
    )
    assert frame.solar.iloc[0] == 0 and np.isnan(frame.solar.iloc[1])
    from energraph.canonical import quality

    assert quality(frame)["missing_hours"] == 1
