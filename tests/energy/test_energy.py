"""Energy-specific scientific invariants and legacy compatibility, without network."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from email.message import Message
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energraph.artifacts import (
    BrokerArtifactRef,
    BrokerPublicationContext,
    PendingPublication,
    finalize_governed_publication,
    require_committed_snapshot,
)
from energraph.broker_adapter import BrokerAdapter
from energraph.broker_contract import BrokerRecord
from energraph.broker_registry import BrokerRegistry
from energraph.broker_service import create_app
from energraph.broker_tree import inspect_tree
from energraph.canonical import materialize, open_energy, parse_csv, quality
from energraph.contracts import Availability, EnergyVariable
from energraph.evaluation import build_plan, load_plan
from energraph.experiments import run_experiment, verify_run
from urbangraph.baselines.contracts import ModelConfig
from urbangraph.baselines.history import make_batch, training_data
from urbangraph.baselines.models import Baseline, fit_model
from urbangraph.data import DataError, DatasetSnapshot, ingest_dataset, verify_snapshot
from urbangraph.evaluation.preprocessing import fit_standardizer
from urbangraph.evaluation.windows import WindowCollection

COLUMN = "DE_load_actual_entsoe_transparency"


class InventedRights(Protocol):
    path: Path
    grants: list[dict[str, object]]

    def source(self, incoming: Path, dataset_id: str, raw_root: Path) -> None: ...

    def _grant(
        self,
        source_id: str,
        source: Path,
        digest: str,
        operation: str,
        destination: Path,
        purpose: str = "local_research",
    ) -> None: ...

    def broker_tree(
        self,
        source_id: str,
        source: Path,
        digest: str,
        operations: tuple[tuple[str, str], ...],
        destination: Path,
    ) -> None: ...


def fixture_csv(tmp_path: Path, rows: int = 2400) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-03-01", periods=rows, freq="h", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            COLUMN: 100 + np.sin(np.arange(rows) / 24) * 20,
        }
    )
    path = source / "time_series_60min_singleindex.csv"
    frame.to_csv(path, index=False)
    return path


def _raw(tmp_path: Path, rights: InventedRights, path: Path) -> DatasetSnapshot:
    rights.source(path.parent, "energy-fixture", tmp_path / "raw")
    return ingest_dataset(path.parent, "energy-fixture", raw_root=tmp_path / "raw")


def _grant(
    rights: InventedRights,
    raw: DatasetSnapshot | Path,
    root: Path,
    purpose: str,
    operation: str,
) -> None:
    directory = raw if isinstance(raw, Path) else raw.directory
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


def canonical(tmp_path: Path, rights: InventedRights) -> tuple[Path, Path]:
    path = fixture_csv(tmp_path)
    raw = _raw(tmp_path, rights, path)
    root = tmp_path / "canonical"
    for purpose, operation in (
        ("canonicalize", "energy_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_canonical_read"),
    ):
        _grant(rights, raw, root, purpose, operation)
    canonical_snapshot = require_committed_snapshot(materialize(raw.directory, root))
    return canonical_snapshot.directory, raw.directory


def _broker_context(
    tmp_path: Path,
    rights: InventedRights,
    raw: DatasetSnapshot,
    monkeypatch: pytest.MonkeyPatch,
    *,
    additional_operations: tuple[tuple[str, str], ...] = (),
) -> tuple[BrokerPublicationContext, BrokerArtifactRef, Any]:
    """Coordinate an invented-only broker with a separate admin test actor."""
    registry = BrokerRegistry(
        tmp_path / "broker/registry.sqlite",
        tmp_path / "broker/staging",
        tmp_path / "broker/artifacts",
    )
    stage = registry.staging / "invented-energy-raw"
    shutil.copytree(raw.directory, stage)
    digest = inspect_tree(stage).digest
    source_id = raw.manifest.dataset_id
    rights.broker_tree(
        source_id,
        stage,
        digest,
        (
            ("local_research", "broker_admit"),
            ("canonicalize", "energy_canonicalize"),
            ("local_research", "energy_canonical_read"),
            ("scientific_evaluation", "energy_plan_create"),
            ("model_training", "energy_plan_read"),
            *additional_operations,
        ),
        registry.artifacts,
    )
    secret_paths: dict[str, Path] = {}
    for role, character in (("admin", "a"), ("worker", "b"), ("reader", "c")):
        secret = tmp_path / f"broker/{role}.secret"
        secret.write_text(character * 48, encoding="ascii")
        secret_paths[role] = secret
    client = TestClient(
        create_app(
            registry,
            admin_secret_file=secret_paths["admin"],
            worker_secret_file=secret_paths["worker"],
            reader_secret_file=secret_paths["reader"],
        )
    )

    class LocalResponse(io.BytesIO):
        status = 200

    def local_transport(request: Any, timeout: float) -> LocalResponse:
        assert timeout == 10
        response = client.post(
            urlsplit(request.full_url).path,
            content=request.data,
            headers=dict(request.headers),
        )
        if response.status_code != 200:
            raise HTTPError(request.full_url, response.status_code, "denied", Message(), None)
        return LocalResponse(response.content)

    monkeypatch.setattr("energraph.broker_adapter.urlopen", local_transport)
    admission = client.post(
        "/v1/admit-source",
        json={
            "stage_id": stage.name,
            "source_id": source_id,
            "expected_sha256": digest,
            "directory": True,
        },
        headers={"Authorization": "Bearer " + secret_paths["admin"].read_text(encoding="ascii")},
    )
    assert admission.status_code == 200
    adapter = BrokerAdapter(
        "http://broker:8080",
        worker_token_file=secret_paths["worker"],
        reader_token_file=secret_paths["reader"],
        staging_root=registry.staging,
    )
    raw_ref = BrokerArtifactRef(admission.json()["artifact_id"], digest, 1, "raw")

    def approve(pending: Any) -> None:
        response = client.post(
            "/v1/approve",
            json={
                "artifact_id": pending.publication_id,
                "expected_sha256": pending.content_sha256,
            },
            headers={
                "Authorization": "Bearer " + secret_paths["admin"].read_text(encoding="ascii")
            },
        )
        if response.status_code != 200:
            raise DataError("Independent test policy actor rejected publication")

    materialized = tmp_path / "broker/materialized"
    materialized.mkdir(mode=0o700)
    return BrokerPublicationContext(adapter, materialized), raw_ref, approve


def test_canonical_roundtrip_idempotence_and_corruption(
    tmp_path: Path, invented_research_rights: InventedRights
) -> None:
    path = fixture_csv(tmp_path)
    before = path.read_bytes()
    rights = invented_research_rights
    raw = _raw(tmp_path, rights, path)
    for purpose, operation in (
        ("canonicalize", "energy_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_canonical_read"),
    ):
        _grant(rights, raw, tmp_path / "canonical", purpose, operation)
    a = require_committed_snapshot(materialize(raw.directory, tmp_path / "canonical"))
    b = require_committed_snapshot(materialize(raw.directory, tmp_path / "canonical"))
    assert a.directory == b.directory and b.reused
    assert path.read_bytes() == before
    data = open_energy(a.directory, raw_snapshot_dir=raw.directory)
    assert len(data.frame) == 2400 and str(data.timestamps.tz) == "UTC"
    assert data.lineage["units"] == "MW"
    parquet = a.directory / "payload" / "observations.parquet"
    parquet.write_bytes(parquet.read_bytes()[:-10])
    with pytest.raises(DataError):
        open_energy(a.directory, raw_snapshot_dir=raw.directory)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "unordered",
        "naive",
        "offset",
        "half-hour",
        "infinity",
        "text",
        "duplicate-header",
        "missing-column",
    ],
)
def test_invalid_csv(tmp_path: Path, case: str, invented_research_rights: InventedRights) -> None:
    path = fixture_csv(tmp_path, 10)
    frame = pd.read_csv(path)
    if case == "duplicate":
        frame.loc[1, "utc_timestamp"] = frame.loc[0, "utc_timestamp"]
    elif case == "unordered":
        frame = frame.iloc[::-1]
    elif case == "naive":
        frame["utc_timestamp"] = frame["utc_timestamp"].str[:-1]
    elif case == "offset":
        frame["utc_timestamp"] = frame["utc_timestamp"].str.replace("Z", "+00:00")
    elif case == "half-hour":
        frame.loc[1, "utc_timestamp"] = "2024-03-01T01:30:00Z"
    elif case == "infinity":
        frame[COLUMN] = np.inf
    elif case == "text":
        frame[COLUMN] = "invalid"
    elif case == "missing-column":
        frame = frame.drop(columns=[COLUMN])
    frame.to_csv(path, index=False)
    if case == "duplicate-header":
        path.write_text(path.read_text().replace("utc_timestamp,", "utc_timestamp,utc_timestamp,"))
    raw = _raw(tmp_path, invented_research_rights, path)
    _grant(invented_research_rights, raw, tmp_path, "local_research", "energy_source_read")
    with pytest.raises(DataError):
        parse_csv(
            raw.directory / "payload" / path.name,
            EnergyVariable(),
            raw_snapshot_dir=raw.directory,
            destination=tmp_path,
            chunk_rows=3,
        )


def test_missing_zero_negative_and_gap_audit(
    tmp_path: Path, invented_research_rights: InventedRights
) -> None:
    path = fixture_csv(tmp_path, 10)
    frame = pd.read_csv(path)
    frame.loc[0, COLUMN] = np.nan
    frame.loc[1, COLUMN] = 0
    frame.loc[2, COLUMN] = -1
    frame = frame.drop(index=5)
    frame.to_csv(path, index=False)
    raw = _raw(tmp_path, invented_research_rights, path)
    _grant(invented_research_rights, raw, tmp_path, "local_research", "energy_source_read")
    data = parse_csv(
        raw.directory / "payload" / path.name,
        EnergyVariable(),
        raw_snapshot_dir=raw.directory,
        destination=tmp_path,
        chunk_rows=2,
    )
    audit = quality(data)
    assert audit["missing_hours"] == 1 and audit["missing_values"] == 1
    assert audit["zero_values"] == 1 and audit["negative_values"] == 1
    assert len(data) == 9


@pytest.mark.parametrize("date", ["2024-03-31", "2024-10-27"])
def test_dst_does_not_change_utc_cadence(
    tmp_path: Path, date: str, invented_research_rights: InventedRights
) -> None:
    path = fixture_csv(tmp_path, 48)
    frame = pd.read_csv(path)
    frame["utc_timestamp"] = pd.date_range(date, periods=48, freq="h", tz="UTC").strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    frame.to_csv(path, index=False)
    raw = _raw(tmp_path, invented_research_rights, path)
    _grant(invented_research_rights, raw, tmp_path, "local_research", "energy_source_read")
    assert (
        quality(
            parse_csv(
                raw.directory / "payload" / path.name,
                EnergyVariable(),
                raw_snapshot_dir=raw.directory,
                destination=tmp_path,
            )
        )["missing_hours"]
        == 0
    )


@pytest.mark.parametrize("category", ["future-measurement", "retrospective-only"])
def test_future_weather_load_rejected(category: str) -> None:
    policy = Availability(category)
    with pytest.raises(DataError):
        policy.require(pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-01", tz="UTC"))


def test_availability_issue_and_latency() -> None:
    now = pd.Timestamp("2024-01-02", tz="UTC")
    Availability("historical-observed").require(now - pd.Timedelta(hours=24), now)
    with pytest.raises(DataError):
        Availability("historical-observed").require(now - pd.Timedelta(hours=23), now)
    Availability("future-calendar").require(now + pd.Timedelta(hours=30), now)
    Availability("issued-forecast").require(now + pd.Timedelta(hours=30), now, issued_at=now)
    with pytest.raises(DataError):
        Availability("issued-forecast").require(now, now)


def test_area_and_units() -> None:
    assert EnergyVariable().to_dict() != EnergyVariable(area_type="bidding-zone").to_dict()
    with pytest.raises(DataError):
        EnergyVariable(unit="unknown")
    with pytest.raises(DataError):
        EnergyVariable(area_type="sensor")


@pytest.mark.parametrize("protocol", ["chronological", "rolling"])
def test_plans_windows_masks_and_training_only(
    tmp_path: Path,
    protocol: str,
    invented_research_rights: InventedRights,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw(tmp_path, invented_research_rights, fixture_csv(tmp_path))
    canonical_root = tmp_path / "canonical"
    for purpose, operation in (
        ("canonicalize", "energy_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_canonical_read"),
        ("scientific_evaluation", "energy_plan_create"),
    ):
        _grant(invented_research_rights, raw, canonical_root, purpose, operation)
    broker_context, raw_ref, policy_approver = _broker_context(
        tmp_path, invented_research_rights, raw, monkeypatch
    )
    canonical_pending = materialize(
        raw.directory,
        canonical_root,
        broker_context=broker_context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(canonical_pending, PendingPublication)
    assert not (canonical_root / "energraph-canonical").exists()
    policy_approver(canonical_pending)
    c = finalize_governed_publication(broker_context, canonical_pending)
    for purpose, operation in (
        ("scientific_evaluation", "energy_plan_create"),
        ("local_research", "derived_commit"),
    ):
        _grant(invented_research_rights, raw, tmp_path / "plans", purpose, operation)
    plan_pending = build_plan(
        c,
        tmp_path / "plans",
        protocol,
        raw_snapshot_dir=raw.directory,
        broker_context=broker_context,
    )
    assert isinstance(plan_pending, PendingPublication)
    assert not (tmp_path / "plans" / "energraph-plan").exists()
    policy_approver(plan_pending)
    p = finalize_governed_publication(broker_context, plan_pending)
    d, s = load_plan(
        c,
        p,
        raw_snapshot_dir=raw.directory,
        broker_context=broker_context,
    )
    windows = WindowCollection(d, s, fold=0, partition="test")
    first = windows[0]
    assert first.x.shape == (168, 1, 1) and first.y.shape == (24, 1, 1)
    assert first.input_positions[-1] == first.origin.position - 24
    previous = -1
    for fold in s.folds:
        for a, b in zip(fold.partitions, fold.partitions[1:], strict=False):
            assert a.stop <= b.start
        assert fold.partition("test").start >= previous
        previous = fold.partition("test").stop
    from dataclasses import replace

    from urbangraph.evaluation.planning import plan_splits

    scaled = plan_splits(d.timestamps, replace(s.config, scaling="global"))
    scaler = fit_standardizer(d, scaled, fold=0)
    train = s.folds[0].partition("train")
    assert scaler.means[0] == pytest.approx(
        d.frame.iloc[train.start : train.stop].to_numpy().mean()
    )
    pop = np.array([first.origin.position], dtype=np.int64)
    batch = make_batch(d, s, 0, "test", pop, d.frame.to_numpy())
    for name, period in [("daily-seasonal", 24), ("weekly-seasonal", 168)]:
        prediction = Baseline(
            ModelConfig(name, fallback="unavailable"), interval_seconds=3600
        ).predict(batch)
        targets = batch.target_positions
        reference = targets - period
        reference -= (
            (np.maximum(reference - batch.history.latest[:, None], 0) + period - 1) // period
        ) * period
        np.testing.assert_array_equal(prediction.values[..., 0], d.frame.to_numpy()[reference])
        assert np.all(reference <= pop[:, None] - 24)
    persistence = Baseline(ModelConfig(), interval_seconds=3600).predict(batch)
    np.testing.assert_array_equal(
        persistence.values, np.full((1, 24, 1, 1), d.frame.iloc[pop[0] - 24, 0])
    )
    # Old fit API still refuses hourly input unless callers explicitly opt in.
    with pytest.raises(DataError):
        fit_model(ModelConfig("autoregressive"), training_data(d, s, 0), 24)


def test_legacy_imports_and_defaults() -> None:
    import urbangraph
    from urbangraph.baselines.contracts import MODEL_IDS
    from urbangraph.datasets.contracts import DatasetConfig
    from urbangraph.evaluation.contracts import ForecastWindowSpecification

    assert urbangraph.__version__ in ("0.1.0", "0.2.0rc2", "0.2.0rc3")
    assert DatasetConfig().nominal_interval_seconds == 300
    assert ForecastWindowSpecification().history == 12
    assert ForecastWindowSpecification().interval_seconds == 300
    assert Baseline(ModelConfig()).interval_seconds == 300
    assert len(MODEL_IDS) == 6


def test_end_to_end_and_cli(
    tmp_path: Path,
    invented_research_rights: InventedRights,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from energraph.cli import main

    rights = invented_research_rights
    raw_snapshot = _raw(tmp_path, rights, fixture_csv(tmp_path))
    raw = raw_snapshot.directory
    for purpose, operation in (
        ("canonicalize", "energy_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_canonical_read"),
        ("scientific_evaluation", "energy_plan_create"),
    ):
        _grant(rights, raw, tmp_path / "canonical", purpose, operation)
    broker_context, raw_ref, policy_approver = _broker_context(
        tmp_path,
        rights,
        raw_snapshot,
        monkeypatch,
        additional_operations=(
            ("model_training", "energy_run_create"),
            ("saved_run_verification", "energy_run_verify"),
        ),
    )
    pending_canonical = materialize(
        raw,
        tmp_path / "canonical",
        broker_context=broker_context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(pending_canonical, PendingPublication)
    policy_approver(pending_canonical)
    c = finalize_governed_publication(broker_context, pending_canonical)
    assert c.broker_reference is not None
    for root, grants in (
        (
            tmp_path / "plans",
            (
                ("scientific_evaluation", "energy_plan_create"),
                ("local_research", "derived_commit"),
                ("model_training", "energy_plan_read"),
            ),
        ),
        (
            tmp_path / "runs",
            (
                ("model_training", "energy_run_create"),
                ("local_research", "derived_commit"),
                ("saved_run_verification", "energy_run_verify"),
            ),
        ),
    ):
        for purpose, operation in grants:
            _grant(invented_research_rights, raw, root, purpose, operation)
    pending_plan = build_plan(
        c,
        tmp_path / "plans",
        "chronological",
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(pending_plan, PendingPublication)
    policy_approver(pending_plan)
    p = finalize_governed_publication(broker_context, pending_plan)
    assert p.broker_reference is not None
    pending_run = run_experiment(
        c,
        p,
        tmp_path / "runs",
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(pending_run, PendingPublication)
    policy_approver(pending_run)
    run = finalize_governed_publication(broker_context, pending_run)
    assert run.broker_reference is not None
    # A trusted local path alone cannot stand in for the committed broker ID.
    with pytest.raises(DataError, match="Committed canonical and plan broker identities"):
        run_experiment(
            c.directory,
            p.directory,
            tmp_path / "another-run",
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )
    with pytest.raises(DataError, match="Committed broker run identity"):
        verify_run(
            run.directory,
            canonical=c,
            plan=p,
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )
    checked = verify_run(
        run,
        canonical=c,
        plan=p,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert checked["valid"] is True
    assert checked["assurance"] == "integrity_and_internal_metric_consistency_only"
    assert checked["integrity"] == "pass"
    assert checked["internal_metric_consistency"] == "pass"
    assert checked["source_truth_agreement"] == "not_established"
    assert checked["authorship_assurance"] == "unverified"
    assert checked["scientific_validation"] == "not_established"
    assert checked["production_approval"] == "not_approved"
    before = _tree_sha256(run.directory)
    with pytest.raises(DataError, match="completed-run reuse is unavailable"):
        run_experiment(
            c,
            p,
            tmp_path / "runs",
            raw_snapshot_dir=raw,
            broker_context=broker_context,
        )
    assert _tree_sha256(run.directory) == before

    broker_args = [
        "--broker-url",
        broker_context.broker.base_url,
        "--broker-worker-token-file",
        str(broker_context.broker.worker_token_file),
        "--broker-reader-token-file",
        str(broker_context.broker.reader_token_file),
        "--broker-materialization-root",
        str(broker_context.materialization_root),
        "--broker-staging-root",
        str(broker_context.broker.staging_root),
        "--canonical-artifact-id",
        c.broker_reference.artifact_id,
        "--plan-artifact-id",
        p.broker_reference.artifact_id,
    ]
    assert (
        main(
            [
                "run",
                "--canonical",
                str(c.directory),
                "--plan",
                str(p.directory),
                "--snapshot",
                str(raw),
                "--root",
                str(tmp_path / "runs"),
                *broker_args,
            ]
        )
        == 1
    )
    reuse_error = json.loads(capsys.readouterr().out)
    assert reuse_error["status"] == "ERROR"
    assert str(run.directory) not in json.dumps(reuse_error)
    assert _tree_sha256(run.directory) == before

    verify_args = [
        "verify-run",
        "--directory",
        str(run.directory),
        "--canonical",
        str(c.directory),
        "--plan",
        str(p.directory),
        "--snapshot",
        str(raw),
        *broker_args,
        "--run-artifact-id",
        run.broker_reference.artifact_id,
    ]
    assert main(verify_args) == 0
    cli_output = json.loads(capsys.readouterr().out)
    assert cli_output["valid"] is True
    assert cli_output["assurance"] == "integrity_and_internal_metric_consistency_only"
    assert cli_output["authorship_assurance"] == "unverified"
    assert cli_output["source_truth_agreement"] == "not_established"
    assert cli_output["scientific_validation"] == "not_established"
    assert cli_output["production_approval"] == "not_approved"
    forged = verify_args.copy()
    forged[forged.index("--run-artifact-id") + 1] = "forged-uncommitted-run"
    assert main(forged) == 1
    denied = json.loads(capsys.readouterr().out)
    assert denied["status"] == "ERROR"
    assert str(run.directory) not in json.dumps(denied)


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest.hexdigest()


def test_matching_run_search_authorizes_before_saved_descriptor_read(
    tmp_path: Path,
    invented_research_rights: InventedRights,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import energraph.experiments as experiments
    from energraph.cli import main

    rights = invented_research_rights
    raw_snapshot = _raw(tmp_path, rights, fixture_csv(tmp_path))
    raw = raw_snapshot.directory
    for purpose, operation in (
        ("canonicalize", "energy_canonicalize"),
        ("local_research", "derived_commit"),
        ("local_research", "energy_canonical_read"),
        ("scientific_evaluation", "energy_plan_create"),
    ):
        _grant(rights, raw, tmp_path / "canonical", purpose, operation)
    broker_context, raw_ref, policy_approver = _broker_context(
        tmp_path,
        rights,
        raw_snapshot,
        monkeypatch,
        additional_operations=(
            ("model_training", "energy_run_create"),
            ("saved_run_verification", "energy_run_verify"),
        ),
    )
    pending_canonical = materialize(
        raw,
        tmp_path / "canonical",
        broker_context=broker_context,
        raw_broker_reference=raw_ref,
    )
    assert isinstance(pending_canonical, PendingPublication)
    policy_approver(pending_canonical)
    canonical_snapshot = finalize_governed_publication(broker_context, pending_canonical)
    assert canonical_snapshot.broker_reference is not None
    plans = tmp_path / "plans"
    runs = tmp_path / "runs"
    for purpose, operation in (
        ("scientific_evaluation", "energy_plan_create"),
        ("local_research", "derived_commit"),
        ("model_training", "energy_plan_read"),
    ):
        _grant(rights, raw, plans, purpose, operation)
    pending_plan = build_plan(
        canonical_snapshot,
        plans,
        "chronological",
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(pending_plan, PendingPublication)
    policy_approver(pending_plan)
    plan = finalize_governed_publication(broker_context, pending_plan)
    assert plan.broker_reference is not None
    for purpose, operation in (
        ("model_training", "energy_run_create"),
        ("local_research", "derived_commit"),
        ("saved_run_verification", "energy_run_verify"),
    ):
        _grant(rights, raw, runs, purpose, operation)
    pending_run = run_experiment(
        canonical_snapshot,
        plan,
        runs,
        raw_snapshot_dir=raw,
        broker_context=broker_context,
    )
    assert isinstance(pending_run, PendingPublication)
    policy_approver(pending_run)
    saved = finalize_governed_publication(broker_context, pending_run)
    assert saved.broker_reference is not None
    original_tree = _tree_sha256(saved.directory)

    descriptor_reads: list[Path] = []
    from urbangraph.datasets.serialization import read_json as original_read_json

    saved_reference = saved.broker_reference
    assert saved_reference is not None

    def observe_read(path: Path) -> object:
        candidate = Path(path)
        if candidate.name == "descriptor.json" and saved.directory in candidate.parents:
            descriptor_reads.append(candidate)
        return original_read_json(path)

    monkeypatch.setattr(experiments, "read_json", observe_read)
    broker_descriptor_reads: list[str] = []
    original_broker_read = broker_context.broker.read_directory_file

    def observe_broker_read(
        record: BrokerRecord, relative_path: str, *, purpose: str, operation: str
    ) -> bytes:
        if relative_path == "descriptor.json" and record.artifact_id == saved_reference.artifact_id:
            broker_descriptor_reads.append(relative_path)
        return original_broker_read(record, relative_path, purpose=purpose, operation=operation)

    monkeypatch.setattr(broker_context.broker, "read_directory_file", observe_broker_read)
    broker_args = [
        "--broker-url",
        broker_context.broker.base_url,
        "--broker-worker-token-file",
        str(broker_context.broker.worker_token_file),
        "--broker-reader-token-file",
        str(broker_context.broker.reader_token_file),
        "--broker-materialization-root",
        str(broker_context.materialization_root),
        "--broker-staging-root",
        str(broker_context.broker.staging_root),
        "--canonical-artifact-id",
        canonical_snapshot.broker_reference.artifact_id,
        "--plan-artifact-id",
        plan.broker_reference.artifact_id,
    ]
    original_grants = json.loads(rights.path.read_text(encoding="utf-8"))["grants"]
    cases = ("missing", "expired", "revoked", "wrong-purpose")
    for case in cases:
        grants = json.loads(json.dumps(original_grants))
        grant_index = next(
            i for i, grant in enumerate(grants) if grant["operation"] == "energy_run_verify"
        )
        if case == "missing":
            grants.pop(grant_index)
        elif case == "expired":
            from datetime import UTC, datetime, timedelta

            grants[grant_index]["expires_utc"] = (
                datetime.now(UTC) - timedelta(minutes=1)
            ).isoformat()
        elif case == "revoked":
            grants[grant_index]["status"] = "revoked"
        else:
            grants[grant_index]["purpose"] = "model_training"
        rights.path.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")
        descriptor_reads.clear()
        broker_descriptor_reads.clear()
        with pytest.raises(DataError):
            run_experiment(
                canonical_snapshot,
                plan,
                runs,
                raw_snapshot_dir=raw,
                broker_context=broker_context,
            )
        assert descriptor_reads == []
        assert broker_descriptor_reads == []
        assert _tree_sha256(saved.directory) == original_tree

        with pytest.raises(DataError):
            verify_run(
                saved,
                canonical=canonical_snapshot,
                plan=plan,
                raw_snapshot_dir=raw,
                broker_context=broker_context,
            )
        assert descriptor_reads == []
        assert broker_descriptor_reads == []

        capsys.readouterr()
        assert (
            main(
                [
                    "run",
                    "--canonical",
                    str(canonical_snapshot.directory),
                    "--plan",
                    str(plan.directory),
                    "--snapshot",
                    str(raw),
                    "--root",
                    str(runs),
                    *broker_args,
                ]
            )
            == 1
        )
        run_error = json.loads(capsys.readouterr().out)
        assert set(run_error) == {"status", "message"}
        assert str(saved.directory) not in json.dumps(run_error)
        assert descriptor_reads == []
        assert broker_descriptor_reads == []
        assert _tree_sha256(saved.directory) == original_tree

        capsys.readouterr()
        assert (
            main(
                [
                    "verify-run",
                    "--directory",
                    str(saved.directory),
                    "--canonical",
                    str(canonical_snapshot.directory),
                    "--plan",
                    str(plan.directory),
                    "--snapshot",
                    str(raw),
                    *broker_args,
                    "--run-artifact-id",
                    saved_reference.artifact_id,
                ]
            )
            == 1
        )
        error_output = json.loads(capsys.readouterr().out)
        assert set(error_output) == {"status", "message"}
        assert str(saved.directory) not in json.dumps(error_output)
        assert descriptor_reads == []
        assert broker_descriptor_reads == []
        assert _tree_sha256(saved.directory) == original_tree
    assert main(["verify", "--directory", str(tmp_path / "missing")]) == 1
