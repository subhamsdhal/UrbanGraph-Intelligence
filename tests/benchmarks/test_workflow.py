"""Synthetic integration evidence; none of these fixtures establish benchmark authenticity."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from urbangraph.benchmarks.audit import leakage_audit, readiness
from urbangraph.benchmarks.cli import main
from urbangraph.benchmarks.pipeline import audit_raw, import_local, prepare_evaluation
from urbangraph.benchmarks.reports import publish_report
from urbangraph.benchmarks.sources import Acquisition, sensor_expectation, source_spec
from urbangraph.benchmarks.spatial import inspect_sensor_metadata
from urbangraph.data import DataError
from urbangraph.data.filesystem import stream_file
from urbangraph.datasets.serialization import object_dict


def evidence(kind: str = "synthetic") -> Acquisition:
    return Acquisition("local controlled fixture", "generated locally", kind, "Test only", True)


def fixture_file(root: Path, dataset: str = "metr-la", gap: bool = False) -> Path:
    source = root / "input with spaces"
    source.mkdir(exist_ok=True)
    index = pd.date_range("2024-01-01", periods=301 if gap else 300, freq="5min")
    if gap:
        index = index[index != index[100]]
    frame = pd.DataFrame(
        {"sensor-Z": np.arange(300, dtype=float), "001": np.arange(300, dtype=float) * 2},
        index=index,
    )
    frame.iloc[20, 0] = np.nan
    path = source / (dataset + ".h5")
    frame.to_hdf(path, key="traffic")
    return path


def invented_rights(
    file: Path, dataset: str, root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, object]], Path]:
    """Exact, disposable local-research grants for this invented input only."""
    policy = root / "invented-benchmark-rights.json"
    grants: list[dict[str, object]] = []
    now = datetime.now(UTC)

    def add(source: Path, digest: str, operation: str, destination: Path) -> None:
        grants.append(
            {
                "source_id": dataset,
                "source_sha256": digest,
                "source_path": str(source),
                "purpose": "local_research",
                "operation": operation,
                "destination_root": str(destination),
                "evidence_reference": "invented-benchmark-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
        policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")

    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    add(file, stream_file(file)[1], "benchmark_import", root)
    return grants, policy


def authorize_snapshot(
    raw: object, grants: list[dict[str, object]], policy: Path, root: Path
) -> None:
    from urbangraph.data import DatasetSnapshot

    assert isinstance(raw, DatasetSnapshot)
    now = datetime.now(UTC)
    for operation, destination in (
        ("snapshot_parse", raw.directory.parent),
        ("canonical_create", root),
        ("canonical_read", root),
    ):
        grants.append(
            {
                "source_id": raw.directory.parent.name,
                "source_sha256": raw.manifest.snapshot_id,
                "source_path": str(raw.directory),
                "purpose": "local_research",
                "operation": operation,
                "destination_root": str(destination),
                "evidence_reference": "invented-benchmark-fixture-only",
                "status": "approved",
                "effective_utc": (now - timedelta(minutes=1)).isoformat(),
                "expires_utc": (now + timedelta(minutes=10)).isoformat(),
                "retention_days": 1,
            }
        )
    policy.write_text(json.dumps({"version": 1, "grants": grants}), encoding="utf-8")


@pytest.mark.parametrize("dataset", ["metr-la", "pems-bay"])
def test_pipeline(tmp_path: Path, dataset: str, monkeypatch: pytest.MonkeyPatch) -> None:
    file = fixture_file(tmp_path, dataset)
    grants, policy = invented_rights(file, dataset, tmp_path, monkeypatch)
    before = file.read_bytes()
    raw = import_local(file, dataset, evidence(), raw_root=tmp_path / "raw")
    authorize_snapshot(raw, grants, policy, tmp_path)
    assert import_local(file, dataset, evidence(), raw_root=tmp_path / "raw").reused
    assert file.read_bytes() == before
    assert list(file.parent.iterdir()) == [file]
    audit = audit_raw(raw.directory, dataset, output_root=tmp_path / "interim")
    canonical = Path(str(audit["canonical_dir"]))
    metrics = object_dict(object_dict(audit["quality"])["metrics"])
    assert metrics["missing_count"] == 1 and metrics["zero_count"] == 2
    plans = prepare_evaluation(
        canonical, dataset, raw_snapshot_dir=raw.directory, output_root=tmp_path / "eval"
    )
    assert leakage_audit(plans["rolling"], canonical, raw_snapshot_dir=raw.directory)["folds"] == 2
    result = readiness(
        dataset, acquisition=evidence(), raw_dir=raw.directory, canonical_dir=canonical, plans=plans
    )
    assert result["status"] == "BLOCKED"
    assert object_dict(result["gates"])["leakage"] == "PASS"
    assert not result["research_ready"]
    exported = publish_report(result, tmp_path / "reports")
    assert publish_report(result, tmp_path / "reports") == exported
    assert json.loads((exported / "report.json").read_text()) == result
    (exported / "report.md").write_text("corrupt")
    with pytest.raises(DataError, match="corrupted"):
        publish_report(result, tmp_path / "reports")
    # Controlled attestation exercises gate logic only, not a real-data claim.
    controlled = readiness(
        dataset,
        acquisition=evidence("real"),
        raw_dir=raw.directory,
        canonical_dir=canonical,
        plans=plans,
    )
    assert controlled["status"] == "FAIL"
    (plans["benchmark"] / "plan.json").write_text("{}")
    result = readiness(dataset, raw_dir=raw.directory, canonical_dir=canonical, plans=plans)
    assert result["status"] == "FAIL"
    (canonical / "observations.parquet").write_bytes(b"corrupt")
    assert readiness(dataset, raw_dir=raw.directory, canonical_dir=canonical)["status"] == "FAIL"
    (raw.directory / "payload" / file.name).write_bytes(b"corrupt")
    with pytest.raises(DataError):
        audit_raw(raw.directory, dataset, output_root=tmp_path / "other")


def test_gaps_and_cross_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = fixture_file(tmp_path, gap=True)
    grants, policy = invented_rights(file, "metr-la", tmp_path, monkeypatch)
    raw = import_local(file, "metr-la", evidence(), raw_root=tmp_path / "raw")
    authorize_snapshot(raw, grants, policy, tmp_path)
    with pytest.raises(DataError):
        audit_raw(raw.directory, "pems-bay", output_root=tmp_path / "interim")
    audit = audit_raw(raw.directory, "metr-la", output_root=tmp_path / "interim")
    metrics = object_dict(object_dict(audit["quality"])["metrics"])
    assert object_dict(metrics["temporal"])["gap_count"] == 1
    with pytest.raises(DataError):
        prepare_evaluation(
            Path(str(audit["canonical_dir"])),
            "metr-la",
            raw_snapshot_dir=raw.directory,
            output_root=tmp_path / "eval",
        )


def test_invalid_import(tmp_path: Path) -> None:
    for dataset in ("invalid", "../metr-la", "metr-la"):
        with pytest.raises(DataError):
            import_local(tmp_path / "absent.h5", dataset, evidence())
    file = fixture_file(tmp_path)
    with pytest.raises(DataError):
        import_local(file, "metr-la", evidence(), raw_root=file.parent / "raw")


def test_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Artificially complete source attestations exercise PASS, not scientific authenticity.
    source = tmp_path / "in"
    source.mkdir()
    file = source / "metr-la.h5"
    pd.DataFrame(
        np.ones((240, 207)), index=pd.date_range("2024-01-01", periods=240, freq="5min")
    ).to_hdf(file, key="traffic")
    grants, policy = invented_rights(file, "metr-la", tmp_path, monkeypatch)
    raw = import_local(file, "metr-la", evidence("real"), raw_root=tmp_path / "r")
    authorize_snapshot(raw, grants, policy, tmp_path)
    audited = audit_raw(raw.directory, "metr-la", output_root=tmp_path / "c")
    canonical = Path(str(audited["canonical_dir"]))
    plans = prepare_evaluation(
        canonical, "metr-la", raw_snapshot_dir=raw.directory, output_root=tmp_path / "e"
    )
    report = readiness(
        "metr-la",
        acquisition=evidence("real"),
        raw_dir=raw.directory,
        canonical_dir=canonical,
        plans=plans,
    )
    assert report["status"] == "PASS"


@pytest.mark.parametrize(
    "kind,date", [("unknown", None), ("real", "tomorrow"), ("synthetic", "2024-02-30")]
)
def test_invalid_acquisition(kind: str, date: str | None) -> None:
    with pytest.raises(DataError):
        Acquisition("reference", "method", kind, "notes", acquired_on=date)


def test_spatial_and_pickle(tmp_path: Path) -> None:
    assert inspect_sensor_metadata(None, ("001",))["status"] == "UNKNOWN"
    pkl = tmp_path / "evil.pkl"
    pkl.write_bytes(b"not opened")
    with pytest.raises(DataError, match="pickle"):
        inspect_sensor_metadata(pkl, ("001",))
    metadata = tmp_path / "locations.csv"
    metadata.write_text("sensor_id,latitude,longitude\n001,34,-118\nsensor-Z,35,-117\n")
    report = inspect_sensor_metadata(metadata, ("sensor-Z", "001"))
    assert report["matched_count"] == 2 and not report["ordering_matches"]
    report = inspect_sensor_metadata(metadata, ("missing",))
    assert report["missing_ids"] == ["missing"] and report["status"] == "WARNING"


def test_sources_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert source_spec("metr-la")["expected_sensor_count"] == 207
    assert source_spec("pems-bay")["expected_sensor_count"] == 325
    assert sensor_expectation("metr-la", None)["status"] == "UNKNOWN"
    assert sensor_expectation("metr-la", 2)["status"] == "WARNING"
    assert main(["inspect-sources", "--json"]) == 0
    assert "metr-la" in json.loads(capsys.readouterr().out)
    assert main(["readiness", "--dataset", "metr-la", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED"
