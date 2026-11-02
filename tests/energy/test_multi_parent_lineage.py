"""Two invented sources exercise the real same-axis energy transformation."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from energraph import artifacts
from energraph.lineage import resolve_derived
from energraph.multivariable import materialize_joined_multi, open_multi, project_multi
from urbangraph.data import DataError, ingest_dataset


def _source(tmp_path: Path, rights: object, name: str, variable: str) -> object:
    incoming = tmp_path / ("source-" + name)
    incoming.mkdir()
    column = (
        "DE_load_actual_entsoe_transparency" if variable == "load" else "DE_solar_generation_actual"
    )
    pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC").strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            column: list(range(48)),
        }
    ).to_csv(incoming / "energy.csv", index=False)
    raw_root = tmp_path / "raw"
    rights.source(incoming, "invented-" + name, raw_root)
    return ingest_dataset(incoming, "invented-" + name, raw_root=raw_root)


def _grant(rights: object, raw: object, output: Path) -> None:
    for purpose, operation, destination in (
        ("canonicalize", "energy_multivariable", output),
        ("local_research", "derived_commit", output),
        ("local_research", "energy_multivariable_read", output),
    ):
        rights._grant(
            raw.directory.parent.name,
            raw.directory,
            raw.manifest.snapshot_id,
            operation,
            destination,
            purpose,
        )


def test_two_source_publication_read_revocation_and_tampering(
    tmp_path: Path, invented_research_rights: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    rights = invented_research_rights
    first = _source(tmp_path, rights, "load", "load")
    second = _source(tmp_path, rights, "solar", "solar")
    root = tmp_path / "joined"
    for raw in (first, second):
        _grant(rights, raw, root)
    inputs = (
        (first.directory, "energy.csv", ("load",)),
        (second.directory, "energy.csv", ("solar",)),
    )
    result = materialize_joined_multi(inputs, root)
    snapshots = tuple(sorted((first.directory, second.directory), key=str))
    frame, descriptor = open_multi(result.directory, raw_snapshot_dirs=snapshots)
    assert list(frame) == [
        name
        for source in snapshots
        for name in (("load",) if source == first.directory else ("solar",))
    ]
    assert len(frame) == 48 and descriptor["schema_version"] == "energy-multivariable-joined-1.0"
    assert len(resolve_derived(result.directory).sources) == 2
    protected_manifest = (result.directory / "manifest.json").read_bytes()
    protected_governance = (result.directory / "payload" / "governance.json").read_bytes()

    def forbidden_derived_metadata(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("derived metadata was opened before safe denial")

    with monkeypatch.context() as guarded:
        guarded.setattr("energraph.lineage._derived", forbidden_derived_metadata)
        with pytest.raises(DataError, match="closure authority unavailable"):
            project_multi(result.directory, snapshots, tmp_path / "projected", ("solar",))
        with pytest.raises(DataError, match="closure authority unavailable"):
            open_multi(
                result.directory,
                raw_snapshot_dirs=snapshots,
                parent_artifact=result.directory,
            )
        # Caller-supplied derived parents cannot establish transitive ancestry.
        with pytest.raises(DataError, match="closure authority unavailable"):
            artifacts._publish_governed(
                tmp_path / "not-a-stage",
                tmp_path / "projected",
                "multivariable",
                parents=(result.directory,),
            )
    assert (result.directory / "manifest.json").read_bytes() == protected_manifest
    assert (result.directory / "payload" / "governance.json").read_bytes() == protected_governance
    with pytest.raises(DataError):
        open_multi(result.directory, raw_snapshot_dirs=(first.directory,))
    with pytest.raises(DataError):
        materialize_joined_multi(inputs, tmp_path / "wrong-destination")

    # Revocation of either parent denies the same previously published artifact.
    for raw in (first, second):
        affected = []
        for grant in rights.grants:
            if grant["source_path"] == str(raw.directory) and grant["operation"] == (
                "energy_multivariable_read"
            ):
                grant["status"] = "revoked"
                affected.append(grant)
        rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
        with pytest.raises(DataError):
            open_multi(result.directory, raw_snapshot_dirs=snapshots)
        for grant in affected:
            grant["status"] = "approved"
        rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))

    # Saved lineage is checksum-covered. A forged missing parent cannot pass.
    governance = result.directory / "payload" / "governance.json"
    original = governance.read_bytes()
    forged = json.loads(original)
    forged["direct_parents"].pop()
    governance.write_text(json.dumps(forged))
    with pytest.raises(DataError):
        open_multi(result.directory, raw_snapshot_dirs=snapshots)
    governance.write_bytes(original)

    # A trusted policy referring to the wrong bytes cannot authorize source use.
    for grant in rights.grants:
        if (
            grant["source_path"] == str(second.directory)
            and grant["operation"] == "energy_multivariable_read"
        ):
            grant["source_sha256"] = "0" * 64
            break
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    with pytest.raises(DataError):
        open_multi(result.directory, raw_snapshot_dirs=snapshots)


@pytest.mark.parametrize("denial", ["missing", "revoked", "expired", "wrong-purpose"])
def test_governed_read_denies_before_saved_artifact_resolution(
    tmp_path: Path, invented_research_rights: object, monkeypatch: pytest.MonkeyPatch, denial: str
) -> None:
    rights = invented_research_rights
    raw = _source(tmp_path, rights, "load", "load")
    root = tmp_path / "multi"
    _grant(rights, raw, root)
    from energraph.multivariable import materialize_multi

    result = materialize_multi(raw.directory, root, "energy.csv", ("load",))
    read_grants = [g for g in rights.grants if g["operation"] == "energy_multivariable_read"]
    assert len(read_grants) == 1
    grant = read_grants[0]
    if denial == "missing":
        rights.grants.remove(grant)
    elif denial == "revoked":
        grant["status"] = "revoked"
    elif denial == "expired":
        grant["expires_utc"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    else:
        grant["purpose"] = "model_training"
    rights.path.write_text(json.dumps({"version": 1, "grants": rights.grants}))
    before = (result.directory / "manifest.json").read_bytes()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved derived artifact resolved before current read permission")

    with monkeypatch.context() as guarded:
        guarded.setattr(artifacts, "resolve_derived", forbidden)
        with pytest.raises(DataError):
            open_multi(result.directory, raw_snapshot_dirs=(raw.directory,))
    assert (result.directory / "manifest.json").read_bytes() == before
