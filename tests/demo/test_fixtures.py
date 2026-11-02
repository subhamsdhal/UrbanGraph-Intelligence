"""Reject corruption and inconsistencies in the approved synthetic contract."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from energraph.demo import fixtures as fixture_module
from energraph.demo.fixtures import FixtureError, load_demo


@pytest.fixture(scope="module")
def approved_package() -> Path:
    root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("ENERGRAPH_DEMO_PACKAGE")
    return Path(configured) if configured else root / "tests" / "fixtures" / "eg006"


def test_approved_package_contract(approved_package: Path) -> None:
    demo = load_demo(approved_package)
    assert len(demo.series) == 3
    assert all(len(rows) == 2400 for rows in demo.series.values())
    assert len(demo.forecasts) == 21
    assert len(demo.origins) == 7
    assert all(len(rows) == 24 for rows in demo.forecasts.values())
    assert len(demo.dispatch) == 1


@pytest.mark.parametrize(
    "relative", ["manifest.json", "fixtures/hourly_energy.csv", "fixtures/graph.json"]
)
def test_corruption_fails(approved_package: Path, tmp_path: Path, relative: str) -> None:
    copy = tmp_path / "package"
    shutil.copytree(approved_package, copy)
    target = copy / relative
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(FixtureError):
        load_demo(copy)


def test_missing_and_extra_files_fail(approved_package: Path, tmp_path: Path) -> None:
    copy = tmp_path / "package"
    shutil.copytree(approved_package, copy)
    (copy / "fixtures" / "graph.json").unlink()
    with pytest.raises(FixtureError):
        load_demo(copy)
    shutil.copytree(approved_package, tmp_path / "package2")
    (tmp_path / "package2" / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(FixtureError):
        load_demo(tmp_path / "package2")
    shutil.copytree(approved_package, tmp_path / "package3")
    (tmp_path / "package3" / "unexpected-directory").mkdir()
    with pytest.raises(FixtureError):
        load_demo(tmp_path / "package3")


@pytest.mark.parametrize(
    ("filename", "field", "value"),
    [
        ("graph.json", "weight", 2.0),
        ("forecasts_and_intervals.json", "truth_MW", -1.0),
        ("battery_scenario.json", "import_MW", -1.0),
    ],
)
def test_semantic_corruption_fails_even_with_updated_hashes(
    approved_package: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    field: str,
    value: float,
) -> None:
    copy = tmp_path / "package"
    shutil.copytree(approved_package, copy)
    target = copy / "fixtures" / filename
    document = json.loads(target.read_text(encoding="utf-8"))
    records = document["edges"] if filename == "graph.json" else document["records"]
    records[0][field] = value
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_file = copy / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if item["relative_path"] == f"fixtures/{filename}":
            item["size_bytes"] = target.stat().st_size
            item["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        fixture_module,
        "BUILTIN_MANIFEST_SHA256",
        hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    )
    with pytest.raises(FixtureError):
        load_demo(copy)
