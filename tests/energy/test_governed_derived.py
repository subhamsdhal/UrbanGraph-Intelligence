"""Invented-only end-to-end checks for the narrow governed energy publisher."""

import json
from pathlib import Path

import pandas as pd
import pytest

from energraph.artifacts import publish
from energraph.canonical import materialize, open_energy, parse_csv
from energraph.cli import main
from energraph.contracts import EnergyVariable
from energraph.multivariable import materialize_multi, open_multi, parse_variables
from urbangraph.data import DataError, ingest_dataset


def _raw(tmp_path: Path, rights: object) -> object:
    source = tmp_path / "invented-input"
    source.mkdir()
    path = source / "time_series_60min_singleindex.csv"
    pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "DE_load_actual_entsoe_transparency": list(range(48)),
        }
    ).to_csv(path, index=False)
    rights.source(source, "invented-energy-only", tmp_path / "raw")
    return ingest_dataset(source, "invented-energy-only", raw_root=tmp_path / "raw")


def _grant(rights: object, raw: object, root: Path, kind: str) -> None:
    operation = "energy_canonicalize" if kind == "canonical" else "energy_multivariable"
    read = "energy_canonical_read" if kind == "canonical" else "energy_multivariable_read"
    for purpose, op in (
        ("canonicalize", operation),
        ("local_research", "derived_commit"),
        ("local_research", read),
    ):
        rights._grant(
            raw.directory.parent.name, raw.directory, raw.manifest.snapshot_id, op, root, purpose
        )


def test_authorized_canonical_create_publish_read_and_revocation(
    tmp_path: Path, invented_research_rights: object
) -> None:
    rights = invented_research_rights
    raw = _raw(tmp_path, rights)
    root = tmp_path / "canonical"
    _grant(rights, raw, root, "canonical")
    created = materialize(raw.directory, root)
    assert len(open_energy(created.directory, raw_snapshot_dir=raw.directory).frame) == 48
    assert (
        main(["inspect", "--directory", str(created.directory), "--snapshot", str(raw.directory)])
        == 0
    )
    assert materialize(raw.directory, root).reused
    with pytest.raises(DataError, match="Current upstream"):
        open_energy(created.directory)
    with pytest.raises(DataError, match="authorization"):
        parse_csv(raw.directory / "payload/time_series_60min_singleindex.csv", EnergyVariable())
    rights.grants[-1]["status"] = "revoked"
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    with pytest.raises(DataError, match="not approved"):
        open_energy(created.directory, raw_snapshot_dir=raw.directory)
    assert (
        main(["inspect", "--directory", str(created.directory), "--snapshot", str(raw.directory)])
        == 1
    )


def test_authorized_multivariable_and_legacy_direct_denial(
    tmp_path: Path, invented_research_rights: object
) -> None:
    rights = invented_research_rights
    raw = _raw(tmp_path, rights)
    root = tmp_path / "multi"
    _grant(rights, raw, root, "multivariable")
    created = materialize_multi(
        raw.directory,
        root,
        "time_series_60min_singleindex.csv",
        ("load",),
    )
    frame, _ = open_multi(created.directory, raw_snapshot_dir=raw.directory)
    assert len(frame) == 48
    assert (
        main(
            [
                "inspect-multi",
                "--directory",
                str(created.directory),
                "--snapshot",
                str(raw.directory),
            ]
        )
        == 0
    )
    with pytest.raises(DataError, match="Current upstream"):
        open_multi(created.directory)
    with pytest.raises(DataError, match="authorization"):
        parse_variables(raw.directory / "payload/time_series_60min_singleindex.csv", ("load",))


def test_derived_commit_requires_separate_final_destination_grant(
    tmp_path: Path, invented_research_rights: object
) -> None:
    rights = invented_research_rights
    raw = _raw(tmp_path, rights)
    root = tmp_path / "approved"
    rights._grant(
        raw.directory.parent.name,
        raw.directory,
        raw.manifest.snapshot_id,
        "energy_canonicalize",
        root,
        "canonicalize",
    )
    with pytest.raises(DataError, match="No matching rights grant"):
        materialize(raw.directory, root)
    assert not root.exists()
    with pytest.raises(DataError, match="No matching rights grant"):
        materialize(raw.directory, tmp_path / "other")
    assert not (tmp_path / "other").exists()


def test_derived_read_rejects_wrong_digest_and_modified_payload(
    tmp_path: Path, invented_research_rights: object
) -> None:
    rights = invented_research_rights
    raw = _raw(tmp_path, rights)
    root = tmp_path / "canonical"
    _grant(rights, raw, root, "canonical")
    created = materialize(raw.directory, root)
    rights.grants[-1]["source_sha256"] = "0" * 64
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    with pytest.raises(DataError, match="differs from current"):
        open_energy(created.directory, raw_snapshot_dir=raw.directory)
    rights.grants[-1]["source_sha256"] = raw.manifest.snapshot_id
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    data = created.directory / "payload" / "observations.parquet"
    data.write_bytes(data.read_bytes()[:-8])
    with pytest.raises(DataError, match="integrity"):
        open_energy(created.directory, raw_snapshot_dir=raw.directory)
    with pytest.raises(DataError, match="corrupted or conflicting"):
        materialize(raw.directory, root)


def test_legacy_stage_path_cannot_be_granted_as_derived_source(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "descriptor.json").write_text("{}")
    with pytest.raises(DataError, match="Governed derived publication"):
        publish(stage, tmp_path / "out", "canonical")
    assert not (tmp_path / "out").exists()


def test_failed_governed_commit_cleans_only_owned_staging(
    tmp_path: Path, invented_research_rights: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urbangraph.data.ingestion as ingestion

    rights = invented_research_rights
    raw = _raw(tmp_path, rights)
    root = tmp_path / "canonical"
    _grant(rights, raw, root, "canonical")

    def fail(stage: Path, destination: Path) -> None:
        raise OSError("simulated governed commit failure")

    monkeypatch.setattr(ingestion, "_publish", fail)
    with pytest.raises(OSError, match="simulated governed commit failure"):
        materialize(raw.directory, root)
    managed = root / "energraph-canonical"
    assert managed.exists()
    assert list(managed.iterdir()) == []
