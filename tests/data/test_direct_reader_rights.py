"""Direct-reader default deny with only invented disposable paths."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from energraph.canonical import materialize
from energraph.energy_graphs import geographic_edges
from urbangraph.data import DataError
from urbangraph.datasets.hdf5_adapter import parse_snapshot
from urbangraph.graphs.sources import read_csv


def test_graph_csv_direct_import_denied_before_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    missing = tmp_path / "invented.csv"
    with pytest.raises(DataError, match="policy unavailable"):
        read_csv(missing, reference="invented-test-source", layout="locations-header")
    assert not missing.exists()


def test_explicit_invented_graph_grant_allows_exact_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "invented.csv"
    content = b"sensor_id,latitude,longitude\nA,1,2\n"
    source.write_bytes(content)
    now = datetime.now(UTC)
    grant = {
        "source_id": "invented-graph-only",
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "source_path": str(source),
        "purpose": "graph_analysis",
        "operation": "graph_csv_read",
        "destination_root": str(tmp_path),
        "evidence_reference": "invented-test-only",
        "status": "approved",
        "effective_utc": (now - timedelta(minutes=1)).isoformat(),
        "expires_utc": (now + timedelta(minutes=10)).isoformat(),
        "retention_days": 1,
    }
    policy = tmp_path / "test-policy.json"
    policy.write_text(json.dumps({"version": 1, "grants": [grant]}), encoding="utf-8")
    monkeypatch.setenv("URBANGRAPH_RIGHTS_POLICY", str(policy))
    rows, identity = read_csv(source, reference="invented-graph-only", layout="locations-header")
    assert rows == [["sensor_id", "latitude", "longitude"], ["A", "1", "2"]]
    assert identity.sha256 == hashlib.sha256(content).hexdigest()
    source.write_bytes(content + b"B,3,4\n")
    with pytest.raises(DataError, match="does not match rights grant"):
        read_csv(source, reference="invented-graph-only", layout="locations-header")


def test_geography_direct_import_denied_before_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    missing = tmp_path / "invented.csv"
    with pytest.raises(
        DataError, match="Current geographic source identity and destination required"
    ):
        geographic_edges(missing, ("country:AA", "country:BB"))
    assert not missing.exists()


def test_snapshot_parser_denied_before_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    missing = tmp_path / "metr-la" / ("a" * 64)
    with pytest.raises(DataError, match="policy unavailable"):
        parse_snapshot(missing)
    assert not missing.exists()


def test_energy_canonicalize_denied_before_snapshot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("URBANGRAPH_RIGHTS_POLICY", raising=False)
    missing = tmp_path / "opsd" / ("a" * 64)
    out = tmp_path / "canonical"
    with pytest.raises(DataError, match="policy unavailable"):
        materialize(missing, out)
    assert not missing.exists() and not out.exists()
